"""Display a saved training grid and a sparse-random test false negative.

Run from the project root where prepared datasets and evaluation results live.
Uses the actual saved augmentation images; no model loading or regeneration.
"""
import argparse
import hashlib
import json
from pathlib import Path


def records(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def select_examples(args):
    train = [r for r in records(args.train_data / 'train.jsonl')
             if r['split'] == 'train' and r['is_augmented']
             and r['original_id'].startswith('TRAINING/')]
    if args.train_id:
        train = [r for r in train if r['original_id'] == args.train_id]
    if not train:
        raise ValueError('No matching augmented training example.')
    train_row = sorted(train, key=lambda r: r['original_id'])[0]

    test = {}
    for row in records(args.test_data / f'{args.condition}.jsonl'):
        identity = row['original_id']
        if identity in test:
            raise ValueError(f'Duplicate test ID: {identity}')
        test[identity] = row
    candidates, seen = [], set()
    predictions = args.results_root / args.arm / 'predictions.jsonl'
    for prediction in records(predictions):
        if prediction['condition'] != args.condition:
            continue
        identity = prediction['original_id']
        if identity in seen:
            raise ValueError(f'Duplicate prediction: {identity}')
        seen.add(identity)
        row = test.get(identity)
        if row is None:
            raise ValueError(f'Prediction missing from prepared test data: {identity}')
        if prediction['true_label'] != int(row['target'] == 'misogynous'):
            raise ValueError(f'Prediction and test ground truth differ: {identity}')
        if prediction['pattern'] != row['pattern']:
            raise ValueError(f'Prediction and test pattern differ: {identity}')
        if (row['split'] == 'test' and identity.startswith('TEST/')
                and row['is_augmented'] and row['pattern'] == 'sparse_random'
                and prediction['true_label'] == 1 and prediction['pred_label'] == 0
                and (not args.test_id or identity == args.test_id)):
            candidates.append((row, prediction))
    if not candidates:
        raise ValueError(f'No sparse-random false negative for {args.arm}, '
                         f'{args.condition}, requested ID={args.test_id!r}.')
    test_row, prediction = sorted(candidates, key=lambda pair: pair[0]['original_id'])[0]
    return train_row, test_row, prediction


def checked_image(row):
    from PIL import Image

    path = Path(row['image_path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != row['image_sha256']:
        raise ValueError(f'Saved image differs from prepared data: {path}')
    with Image.open(path) as image:
        return image.convert('RGB')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-data', type=Path,
                        default=Path('data/processed/mami_lora_qlora_seed42'))
    parser.add_argument('--test-data', type=Path,
                        default=Path('data/processed/mami_unseen_patterns_seed42'))
    parser.add_argument('--results-root', type=Path,
                        default=Path('results/mami_lora_qlora/unseen-patterns-v1'))
    parser.add_argument('--arm', choices=('fp16-base', 'nf4-base', 'fp16-lora', 'nf4-qlora'),
                        default='fp16-lora')
    parser.add_argument('--condition',
                        choices=('overlay_image_only', 'overlay_with_transcription'),
                        default='overlay_with_transcription')
    parser.add_argument('--train-id', help='Optional exact ID, e.g. TRAINING/123.jpg')
    parser.add_argument('--test-id', help='Optional exact ID, e.g. TEST/12345.jpg')
    parser.add_argument('--output', type=Path,
                        default=Path('outputs/figures/augmented_memes.png'))
    parser.add_argument('--show', action='store_true', help='Also open the saved figure')
    args = parser.parse_args()
    try:
        train, test, prediction = select_examples(args)
        images = [checked_image(row) for row in (train, test)]
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'Error: {error}\n')

    from PIL import Image, ImageDraw, ImageFont, ImageOps

    figure = Image.new('RGB', (1480, 960), 'white')
    draw = ImageDraw.Draw(figure)
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:  # Pillow before version 10.1
        font = ImageFont.load_default()
    titles = [
        f"Training: 4x4 emoji grid\n{train['original_id']} | label: {train['target']}",
        f"Test: sparse random emojis (false negative)\n{test['original_id']} | "
        'true: misogynous | predicted: clean',
    ]
    draw.text((30, 20), f'Test model: {args.arm} | {args.condition}',
              fill='black', font=font)
    for index, (image, title) in enumerate(zip(images, titles)):
        x = 30 + index * 740
        fitted = ImageOps.contain(image, (680, 760))
        figure.paste(fitted, (x + (680-fitted.width)//2, 80 + (760-fitted.height)//2))
        draw.multiline_text((x, 860), title, fill='black', font=font, spacing=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.save(args.output)
    metadata = dict(
        selection='First matching ID in sorted order; illustrative examples.',
        training=train, test=test, prediction=prediction,
        arm=args.arm, condition=args.condition,
    )
    args.output.with_suffix('.json').write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f"Training grid: {train['original_id']} ({train['image_path']})")
    print(f"Test sparse FN: {test['original_id']} ({test['image_path']})")
    print(f"True label=1, predicted label=0; raw answer={prediction['raw_answer']!r}")
    print(f'Saved: {args.output.resolve()}')
    if args.show:
        figure.show()


if __name__ == '__main__':
    main()
