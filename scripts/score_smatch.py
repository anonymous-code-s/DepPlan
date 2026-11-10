"""Compute corpus-level Smatch for aligned predicted and reference AMR files."""
import argparse
import io
import json
from pathlib import Path
import random
import penman
import smatch

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--predicted',type=Path,required=True)
p.add_argument('--gold',type=Path,required=True)
p.add_argument('--seed',type=int,default=2026)
a=p.parse_args()
texts=[]
counts=[]
for path in [a.predicted,a.gold]:
    graphs=list(penman.iterdecode(path.read_text()))
    if not graphs or any(not g.instances() for g in graphs):
        raise ValueError('Both files must contain nonempty AMR graphs')
    counts.append(len(graphs))
    texts.append('\n\n'.join(penman.encode(penman.Graph(g.triples,top=g.top)) for g in graphs)+'\n\n')
if counts[0]!=counts[1]:raise ValueError('Predicted and gold graph counts differ')
class MatchingRandom(random.Random):
    # Smatch 1.0.4 reseeds from OS entropy before each initialization.
    # Keep a single explicitly seeded stream for reproducible matching.
    def seed(self, value=None, version=2):
        if value is not None:
            super().seed(value, version)

smatch.random=MatchingRandom(a.seed)
smatch.single_score=True
smatch.match_triple_dict.clear()
precision,recall,f1=list(smatch.score_amr_pairs(io.StringIO(texts[0]),io.StringIO(texts[1])))[0]
print(json.dumps({'graphs':counts[0],'matching_seed':a.seed,
                  'precision_percent':100*precision,'recall_percent':100*recall,'smatch_f1_percent':100*f1},indent=2))
