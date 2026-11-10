"""Load and evaluate strategy classifiers."""
from pathlib import Path
import json
import torch
from safetensors.torch import load_file
from .model import LABELS, StrategyGCN, collate, graph_structure


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def encode_graphs(rows, encoder, batch_size=64):
    structures = [graph_structure(row['amr_triples']) for row in rows]
    if any(not concepts for concepts, _ in structures):
        raise ValueError('A successful graph must contain at least one concept node')
    concepts = sorted({c for names, _ in structures for c in names})
    vectors = encoder.encode(concepts, batch_size=batch_size, convert_to_tensor=True,
                             normalize_embeddings=False, show_progress_bar=False).cpu()
    lookup = dict(zip(concepts, vectors))
    return [{'x': torch.stack([lookup[c] for c in names]), 'adj': adj,
             'y': LABELS.index(row['silver_strategy']) if 'silver_strategy' in row else 0}
            for row, (names, adj) in zip(rows, structures)]


class Classifier:
    def __init__(self, checkpoint, encoder=None, device='cpu'):
        self.directory = Path(checkpoint)
        self.config = json.loads((self.directory / 'config.json').read_text())
        if tuple(self.config['labels']) != LABELS:
            raise ValueError('Checkpoint label order does not match the classifier')
        self.device = device
        self.model = StrategyGCN(self.config['input_dim']).to(device).eval()
        self.model.load_state_dict(load_file(str(self.directory / 'model.safetensors')))
        self.encoder_name = encoder or self.config['encoder']
        self._encoder = None

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(str(self.encoder_name), device=self.device)
        return self._encoder

    def predict(self, rows, batch_size=16):
        sources = {r['structure_source'] for r in rows if 'structure_source' in r}
        if sources and sources != {self.config['structure_source']}:
            raise ValueError('Graph source and classifier checkpoint do not match')
        valid = [i for i, r in enumerate(rows) if r.get('status', 'success') == 'success']
        graphs = encode_graphs([rows[i] for i in valid], self.encoder) if valid else []
        predictions = [None] * len(rows)
        with torch.inference_mode():
            for start in range(0, len(graphs), batch_size):
                x, adj, membership, _ = collate(graphs[start:start+batch_size], self.device)
                labels = self.model(x, adj, membership).argmax(-1).cpu().tolist()
                for i, label in zip(valid[start:start+batch_size], labels):
                    predictions[i] = LABELS[label]
        return [dict(dataset=r.get('dataset'), example_id=r['example_id'],
                     silver_strategy=r.get('silver_strategy'), predicted_strategy=p,
                     status='success' if p is not None else 'graph_failed',
                     **({'cohort': r['cohort']} if 'cohort' in r else {}))
                for r, p in zip(rows, predictions)]


def classification_metrics(rows):
    """Failures are incorrect and false negatives for their silver-label class."""
    if not rows:
        raise ValueError('Cannot score an empty evaluation set')
    matrix = [[0] * 4 for _ in LABELS]
    for row in rows:
        truth = LABELS.index(row['silver_strategy'])
        prediction = row.get('predicted_strategy')
        predicted = LABELS.index(prediction) if prediction is not None else 3
        matrix[truth][predicted] += 1
    correct = sum(matrix[i][i] for i in range(3))
    per_class = []
    for i in range(3):
        tp = matrix[i][i]
        fp = sum(matrix[j][i] for j in range(3)) - tp
        fn = sum(matrix[i]) - tp
        per_class.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {'questions': len(rows), 'correct': correct,
            'accuracy_percent': 100 * correct / len(rows),
            'macro_f1_percent': 100 * sum(per_class) / 3,
            'graph_failures': sum(row[3] for row in matrix),
            'labels': list(LABELS),
            'confusion_columns': [*LABELS, 'graph_failed'], 'confusion_matrix': matrix,
            'reference': 'Qwen silver strategy labels'}
