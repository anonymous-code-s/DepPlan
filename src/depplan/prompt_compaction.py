"""Compact prompt payloads while retaining semantic facts and evidence links."""
from __future__ import annotations

import json
from collections import defaultdict


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def indexed_semantic_units(linearized):
    """Write every central fact once and preserve its one-hop context by ID."""
    details = linearized['unit_details']
    triple_ids = defaultdict(list)
    by_id = {unit['id']: unit for unit in details}
    assert len(by_id) == len(details)
    for unit in details:
        triple_ids[tuple(unit['triple'])].append(unit['id'])
    units = []
    for unit in details:
        neighbors = sorted({index for triple in unit['context_triples']
                            for index in triple_ids[tuple(triple)] if index != unit['id']})
        reconstructed = {by_id[index]['central_text'] for index in neighbors}
        assert reconstructed == set(unit['context_texts']), 'Context facts must be preserved'
        suffix = ' | ctx=' + ','.join(map(str, neighbors)) if neighbors else ''
        units.append(f"U{unit['id']}: {unit['central_text']}{suffix}")
    return units


def previous_candidates(proposal, verified, issues, *, round_number=1):
    """Elide only the duplicate copy when two candidate objects are equal."""
    if proposal == verified:
        candidates = (f'Round-{round_number} proposal and Verifier candidate (identical):\n'
                      + compact_json(proposal))
    else:
        candidates = (f'Round-{round_number} proposal:\n' + compact_json(proposal)
                      + '\n\nVerifier candidate:\n' + compact_json(verified))
    return (candidates + '\n\nVerifier issues:\n' + compact_json(issues)
            + '\nReturn the minimally revised plan.')


def final_answer_payload(question, trace):
    """Build the final-answer context from resolved steps and indexed evidence."""
    evidence, steps = {}, []
    for item in trace:
        refs = []
        for passage in item['retrieved_contexts']:
            key = str(passage['index'])
            if key in evidence and evidence[key] != passage['text']:
                raise ValueError('The same passage index has inconsistent text')
            evidence[key] = passage['text']
            if key not in refs:
                refs.append(key)
        step = {'id': item['step_id'], 'question': item['resolved_sub_question'],
                'depends_on': item['depends_on'], 'answer': item['answer'],
                'evidence_ids': refs}
        raw = item.get('raw_answer_output', '')
        if raw:
            from .plan import extract_json_object
            payload = extract_json_object(raw)
            if payload.get('rationale'):
                step['rationale'] = payload['rationale']
        if item.get('unresolved_dependency'):
            step['unresolved_dependency'] = True
        steps.append(step)
    return {'question': question, 'executed_steps': steps, 'evidence': evidence}
