"""Retrain a source-matched GCN on the supplied silver-labeled graphs."""
import json
import hashlib
from pathlib import Path
import random
import time
import numpy as np
import torch
from safetensors.torch import save_file
from .model import LABELS, StrategyGCN, collate
from .runtime import read_jsonl, encode_graphs


def train(args):
    from sentence_transformers import SentenceTransformer
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use an empty output directory to preserve released weights')
    rows = read_jsonl(args.input)
    selected = len(rows)
    rows = [r for r in rows if r.get('status', 'success') == 'success']
    if not rows:
        raise ValueError('No valid training graphs')
    sources = {r['structure_source'] for r in rows}
    if len(sources) != 1:
        raise ValueError('Train one classifier per structure source')
    revision = getattr(args, 'encoder_revision', None)
    encoder = SentenceTransformer(args.encoder, device=args.device,
                                  **({'revision': revision} if revision else {}))
    graphs = encode_graphs(rows, encoder)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.synchronize(args.device)
    started = time.perf_counter()
    model = StrategyGCN(graphs[0]['x'].shape[1]).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    rng = random.Random(args.seed)
    history = []
    for epoch in range(30):
        model.train()
        order = list(range(len(graphs)))
        rng.shuffle(order)
        loss_sum = correct = 0
        for start in range(0, len(order), 16):
            batch = [graphs[i] for i in order[start:start+16]]
            x, adj, membership, y = collate(batch, args.device)
            logits = model(x, adj, membership)
            loss = torch.nn.functional.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch)
            correct += int((logits.argmax(-1) == y).sum())
        history.append({'epoch': epoch+1, 'loss': loss_sum/len(graphs),
                        'accuracy': correct/len(graphs)})
        print(json.dumps(history[-1]), flush=True)
    if args.device.startswith('cuda'):
        torch.cuda.synchronize(args.device)
    elapsed = time.perf_counter() - started
    out.mkdir(parents=True, exist_ok=True)
    structure_source = next(iter(sources))
    weights = out / 'model.safetensors'
    save_file({k: v.detach().cpu().contiguous() for k,v in model.state_dict().items()},
              str(weights), metadata={'model_type': 'strategy_gcn', 'format': 'pt',
                                      'structure_source': structure_source,
                                      'training_seed': str(args.seed)})
    config = {'model_type':'strategy_gcn', 'structure_source':structure_source,
              'input_dim':graphs[0]['x'].shape[1], 'labels':list(LABELS),
              'encoder':args.encoder, 'training_seed':args.seed,
              'selected_training_questions':selected, 'effective_training_graphs':len(rows),
              'epochs':30, 'batch_size':16, 'checkpoint_selection':'last_epoch',
              'architecture': {'projection':128, 'gcn_layers':2, 'gcn_hidden':128,
                               'mlp':[128,64,3], 'dropout':0.2},
              'parameter_count':sum(p.numel() for p in model.parameters()),
              'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
              'optimizer':{'name':'AdamW','lr':0.001,'weight_decay':0.01}}
    source_root = Path(__file__).resolve().parents[1]
    provenance = {'training_seed': args.seed, 'device': args.device,
                  'torch_version': torch.__version__, 'numpy_version': np.__version__,
                  'input_sha256': hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
                  'question_ids': [[r.get('dataset'), r['example_id']] for r in rows],
                  'structure_source': structure_source, 'encoder': str(args.encoder),
                  'encoder_revision_requested': revision,
                  'weights_sha256': config['weights_sha256'],
                  'source_hashes': {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(source_root.rglob('*.py'))},
                  'checkpoint_selection': 'last_epoch',
                  'feature_tensor_sha256': hashlib.sha256(b''.join(
                      g['x'].detach().cpu().contiguous().numpy().tobytes() for g in graphs)).hexdigest(),
                  'scope': 'Recorded by this training invocation; does not verify prior released weights'}
    for name, value in [('config',config),('training_history',history),
                        ('training_provenance',provenance),
                        ('timing',{'training_seconds':elapsed,'excludes':'BGE feature preparation and serialization'})]:
        (out / f'{name}.json').write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
