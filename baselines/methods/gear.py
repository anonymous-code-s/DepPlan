"""GeAR: passage-aligned triple graph, SyncGE, and gist-memory retrieval.

The graph is built over all shared candidate passages before query-conditioned
retrieval. Extraction is logged as preprocessing, not hidden in retrieval cost.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np

from ..prompts import gear as prompts
from .agent_utils import ask_json, count_step, payload


def triples_from(value):
    triples = value.get("triples", [])
    if not isinstance(triples, list):
        raise ValueError("GeAR triples must be a list")
    result = []
    for triple in triples:
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            raise ValueError("Each GeAR triple must have subject, relation, object")
        if not all(isinstance(term, str) and term.strip() for term in triple):
            raise ValueError("GeAR triple components must be nonempty strings")
        item = tuple(term.strip() for term in triple)
        if item not in result:
            result.append(item)
    return result


def triple_text(triple):
    return "(" + "; ".join(triple) + ")"


def entity_key(text):
    return re.sub(r"\s+", " ", text.strip()).casefold()


def rrf(rankings, constant=60, limit=None):
    """Fuse ranked passage lists; ties preserve deterministic first appearance."""
    scores, passages, order = defaultdict(float), {}, {}
    for ranking in rankings:
        seen = set()
        for rank, passage in enumerate(ranking, 1):
            key = str(passage["id"])
            if key in seen:
                continue
            seen.add(key)
            scores[key] += 1.0 / (constant + rank)
            passages.setdefault(key, passage)
            order.setdefault(key, len(order))
    keys = sorted(scores, key=lambda key: (-scores[key], order[key]))
    if limit is not None:
        keys = keys[:limit]
    return [dict(passages[key], score=scores[key]) for key in keys]


class TripleGraph:
    def __init__(self, triples, passage_ids):
        self.triples = triples
        self.passage_ids = passage_ids
        self.texts = [triple_text(triple) for triple in triples]
        self.entities = defaultdict(set)
        for index, (head, _, tail) in enumerate(triples):
            self.entities[entity_key(head)].add(index)
            self.entities[entity_key(tail)].add(index)

    def neighbors(self, index):
        head, _, tail = self.triples[index]
        return sorted((self.entities[entity_key(head)] | self.entities[entity_key(tail)]) - {index})


def diverse_beam_search(ctx, graph, query, starts, width=10, length=2,
                        neighbors_per_beam=100, gamma=20):
    """Paper Algorithm 1: visited triples and sibling discount."""
    if min(width, length, neighbors_per_beam) < 1 or gamma <= 0:
        raise ValueError("GeAR beam width/length/neighbors/gamma must be positive")
    if not starts:
        return []
    query_vector = np.asarray(ctx.embed([query]))[0]

    def score_sequences(sequences):
        texts = [" ".join(graph.texts[index] for index in sequence) for sequence in sequences]
        vectors = np.asarray(ctx.embed(texts))
        return (vectors @ query_vector).tolist()

    starts = list(dict.fromkeys(starts))
    beams = sorted(zip(score_sequences([[index] for index in starts]), [[index] for index in starts]),
                   key=lambda item: (-item[0], item[1]))[:width]
    for depth in range(1, length):
        visited = {index for _, sequence in beams for index in sequence}
        candidates = []
        for old_score, sequence in beams:
            neighbors = [index for index in graph.neighbors(sequence[-1]) if index not in visited]
            if len(neighbors) > neighbors_per_beam:
                ranked = ctx.rank_texts(query, [graph.texts[index] for index in neighbors],
                                        k=neighbors_per_beam, mode="dense")
                neighbors = [neighbors[item["index"]] for item in ranked]
            extended = [sequence + [index] for index in neighbors]
            if not extended:
                continue
            siblings = sorted(zip(score_sequences(extended), extended),
                              key=lambda item: (-item[0], item[1]))
            for rank, (score, extended_sequence) in enumerate(siblings):
                discounted = (old_score + score) * math.exp(-min(rank, gamma) / gamma)
                candidates.append((discounted, extended_sequence))
        if not candidates:
            break  # Retain reached evidence when the finite candidate graph ends.
        beams = sorted(candidates, key=lambda item: (-item[0], item[1]))[:width]
        ctx.event("gear_beam", depth=depth + 1,
                  beams=[{"score": score, "triples": sequence} for score, sequence in beams])
    return beams


def beam_passages(beams, graph, passages):
    """Breadth-first flattening required by section 4.2, then passage deduplication."""
    output, seen = [], set()
    for depth in range(max((len(seq) for _, seq in beams), default=0)):
        for _, sequence in beams:
            if depth >= len(sequence):
                continue
            key = str(graph.passage_ids[sequence[depth]])
            if key not in seen:
                seen.add(key)
                output.append(passages[key])
    return output


async def build_graph(ctx):
    triples, passage_ids = [], []
    for passage in ctx.passages:
        # The question is deliberately absent: this is an offline-style index.
        response = await ask_json(ctx, "gear_index", prompts.gear_extract_prompt,
                                  passage=ctx.context([passage]))
        extracted = triples_from(response)
        triples.extend(extracted)
        passage_ids.extend([passage["id"]] * len(extracted))
        ctx.event("gear_index_passage", stage="preprocessing", passage_id=passage["id"],
                  triples=extracted)
    ctx.event("gear_index_complete", stage="preprocessing", triples=len(triples),
              passages=len(ctx.passages))
    return TripleGraph(triples, passage_ids)


async def gear(ctx):
    cfg = ctx.config
    graph = await build_graph(ctx)
    passages = {str(p["id"]): p for p in ctx.passages}
    constant = int(cfg.get("rrf_k", 60))
    top_k = int(cfg.get("top_k", 10))
    dataset = str(getattr(ctx, "dataset", "")).lower()
    default_rounds = 4 if "musique" in dataset else 2
    rounds = int(cfg.get("max_iterations_by_dataset", {}).get(dataset,
                 cfg.get("max_iterations", default_rounds)))
    query, memory, retrievals, previous_queries = ctx.question, [], [], []
    stop = "max_iterations"
    for iteration in range(1, rounds + 1):
        count_step(ctx, "gear_retrieval_round")
        previous_queries.append(query)
        base = ctx.retrieve(query, k=top_k, mode=cfg.get("retriever", "hybrid"))
        proximal = triples_from(await ask_json(
            ctx, "gear_sync_reader", prompts.gear_reader_prompt,
            question=ctx.question, current_query=query, context=ctx.context(base), gist_memory=[]))
        starts = []
        if graph.triples:
            for triple in proximal:
                links = ctx.rank_texts(triple_text(triple), graph.texts, k=1,
                                      mode=cfg.get("triple_link_mode", "hybrid"))
                starts.extend(item["index"] for item in links)
        beams = diverse_beam_search(
            ctx, graph, query, starts, width=int(cfg.get("beam_width", 10)),
            length=int(cfg.get("beam_length", 2)),
            neighbors_per_beam=int(cfg.get("neighbors_per_beam", 100)),
            gamma=float(cfg.get("diversity_gamma", 20)))
        expanded = beam_passages(beams, graph, passages)
        combined = rrf([expanded, base], constant)
        retrievals.append(combined)
        ctx.event("gear_syncge", iteration=iteration, query=query, initial_triples=starts,
                  base_passage_ids=[p["id"] for p in base],
                  expanded_passage_ids=[p["id"] for p in expanded],
                  fused_passage_ids=[p["id"] for p in combined])
        found = triples_from(await ask_json(
            ctx, "gear_gist_reader", prompts.gear_reader_prompt,
            question=ctx.question, current_query=query,
            context=ctx.context(combined), gist_memory=memory))
        # Concatenation in Eq. 5 with exact duplicate elimination. No arbitrary
        # ten-fact cap: later turns retain useful evidence from every iteration.
        memory.extend(triple for triple in found if triple not in memory)
        decision = await ask_json(ctx, "gear_reasoner", prompts.gear_reasoner_prompt,
                                  question=ctx.question, gist_memory=memory)
        ctx.event("gear_gist", iteration=iteration, memory=memory, decision=decision)
        if decision.get("answerable") is True:
            stop = "gist_answerable"
            break
        if iteration < rounds:
            rewritten = await ask_json(ctx, "gear_rewriter", prompts.gear_rewriter_prompt,
                                       question=ctx.question, gist_memory=memory,
                                       reason=decision.get("reason", ""), previous_queries=previous_queries)
            query = str(rewritten.get("query", "")).strip()
            if not query:
                raise ValueError("GeAR query rewriter returned an empty query")
    # Section 5.4: link every gist triple to BOTH passage and triple indexes,
    # fuse those lists, then fuse with all iteration retrievals.
    linked = []
    for triple in memory:
        text = triple_text(triple)
        direct = ctx.retrieve(text, k=top_k, mode=cfg.get("retriever", "hybrid"))
        hits = ctx.rank_texts(text, graph.texts, k=top_k,
                             mode=cfg.get("triple_link_mode", "hybrid")) if graph.texts else []
        via_graph = [passages[str(graph.passage_ids[hit["index"]])] for hit in hits]
        linked.append(rrf([direct, via_graph], constant))
    final_passages = rrf(linked + retrievals, constant, int(cfg.get("answer_top_k", 5)))
    ctx.event("gear_final_retrieval", gist_triples=len(memory),
              passage_ids=[p["id"] for p in final_passages])
    answer = await ctx.ask("gear_answer", prompts.gear_answer_prompt,
                           payload(question=ctx.question, context=ctx.context(final_passages)),
                           temperature=cfg.get("answer_temperature", 0),
                           max_tokens=cfg.get("answer_max_tokens", 1024))
    ctx.stop = stop
    return answer.strip()
