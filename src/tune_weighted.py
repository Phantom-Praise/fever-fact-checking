import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

MODEL_NAME = 'FacebookAI/roberta-base'
TRAIN_FILE = Path('data/processed/train.jsonl')
VAL_FILE = Path('data/processed/validation.jsonl')
MAX_LENGTH = 256
TRAIN_FRACTION = 0.20
MAX_VAL = 2000
SEED = 42
CONFIGS = {
    'H1': {'lr': 2e-5, 'weight_decay': 0.01},
    'H2': {'lr': 1e-5, 'weight_decay': 0.01},
    'H3': {'lr': 3e-5, 'weight_decay': 0.01},
    'H4': {'lr': 2e-5, 'weight_decay': 0.00},
}


class FeverDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        encoding = self.tokenizer(
            example['claim'], truncation=True, padding='max_length',
            max_length=MAX_LENGTH, return_tensors='pt'
        )
        item = {key: value.squeeze(0) for key, value in encoding.items()}
        item['labels'] = torch.tensor(int(example['label_id']), dtype=torch.long)
        return item


def read_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    gold, predicted = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            labels = batch['labels'].to(device)
            inputs = {key: value.to(device) for key, value in batch.items()
                      if key != 'labels'}
            logits = model(**inputs).logits
            total_loss += torch.nn.functional.cross_entropy(logits, labels).item()
            gold.extend(labels.cpu().tolist())
            predicted.extend(logits.argmax(dim=-1).cpu().tolist())
    return {
        'val_loss': total_loss / max(len(loader), 1),
        'val_accuracy': accuracy_score(gold, predicted),
        'val_macro_f1': f1_score(gold, predicted, labels=[0, 1, 2],
                                 average='macro', zero_division=0),
        'prediction_counts': Counter(predicted),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, choices=CONFIGS)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1:
        parser.error('batch-size and epochs must be positive')

    set_seed(args.seed)
    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available() else 'cpu')
    settings = CONFIGS[args.config]
    run_id = f'{args.config}_seed{args.seed}'
    checkpoint_dir = Path('checkpoints/tuning_weighted') / run_id
    result_dir = Path('results/tuning_weighted')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir / f'{run_id}.csv'
    summary_path = result_dir / f'{run_id}_best.json'

    train_examples = read_jsonl(TRAIN_FILE)
    val_examples = read_jsonl(VAL_FILE)
    # A separate RNG ensures the sampled subset is independent of model RNG state.
    random.Random(42).shuffle(train_examples)
    train_examples = train_examples[:int(len(train_examples) * TRAIN_FRACTION)]
    val_examples = val_examples[:MAX_VAL]
    counts = Counter(int(ex['label_id']) for ex in train_examples)
    if any(counts[label] == 0 for label in range(3)):
        raise ValueError(f'Missing class in training subset: {counts}')
    n = len(train_examples)
    weights = torch.tensor([n / (3 * counts[label]) for label in range(3)],
                           dtype=torch.float32, device=device)

    print(f'Run: {run_id} | Device: {device}', flush=True)
    print(f'Train: {n} | Validation: {len(val_examples)} | Counts: {dict(counts)}', flush=True)
    print(f'Learning rate: {settings["lr"]} | Weight decay: {settings["weight_decay"]}', flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3
    ).to(device)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(FeverDataset(train_examples, tokenizer),
                              batch_size=args.batch_size, shuffle=True,
                              generator=generator)
    val_loader = DataLoader(FeverDataset(val_examples, tokenizer),
                            batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'],
                                   weight_decay=settings['weight_decay'])
    fields = ['config', 'seed', 'learning_rate', 'weight_decay', 'batch_size',
              'epoch', 'train_loss', 'val_loss', 'val_accuracy', 'val_macro_f1',
              'predicted_0', 'predicted_1', 'predicted_2', 'elapsed_minutes']
    best = None
    with log_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            model.train()
            loss_total = 0.0
            for step, batch in enumerate(train_loader, 1):
                labels = batch['labels'].to(device)
                inputs = {key: value.to(device) for key, value in batch.items()
                          if key != 'labels'}
                optimizer.zero_grad(set_to_none=True)
                logits = model(**inputs).logits
                loss = torch.nn.functional.cross_entropy(logits, labels,
                                                          weight=weights)
                loss.backward()
                optimizer.step()
                loss_total += loss.item()
                if step % 250 == 0:
                    print(f'{run_id} epoch {epoch} step {step}/{len(train_loader)} '
                          f'train_loss={loss_total / step:.4f}', flush=True)
            metrics = evaluate(model, val_loader, device)
            counts_pred = metrics.pop('prediction_counts')
            row = {
                'config': args.config, 'seed': args.seed,
                'learning_rate': settings['lr'],
                'weight_decay': settings['weight_decay'],
                'batch_size': args.batch_size, 'epoch': epoch,
                'train_loss': loss_total / len(train_loader),
                **metrics,
                'predicted_0': counts_pred.get(0, 0),
                'predicted_1': counts_pred.get(1, 0),
                'predicted_2': counts_pred.get(2, 0),
                'elapsed_minutes': round((time.time() - start) / 60, 2),
            }
            writer.writerow(row)
            stream.flush()
            print(f'Epoch {epoch}: accuracy={row["val_accuracy"]:.4f} '
                  f'macro_f1={row["val_macro_f1"]:.4f}', flush=True)
            if best is None or (row['val_macro_f1'], row['val_accuracy']) > (
                    best['val_macro_f1'], best['val_accuracy']):
                best = row.copy()
                model.save_pretrained(checkpoint_dir / 'best')
                tokenizer.save_pretrained(checkpoint_dir / 'best')
                summary_path.write_text(json.dumps(best, indent=2))
                print(f'New best checkpoint: {checkpoint_dir / "best"}', flush=True)
    print(f'Finished {run_id}. Best epoch: {best["epoch"]}, '
          f'macro_f1={best["val_macro_f1"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
