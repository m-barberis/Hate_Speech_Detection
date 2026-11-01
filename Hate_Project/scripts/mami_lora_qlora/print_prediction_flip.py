"""Show a paired test prediction flip: clean true positive -> augmented false negative.

Reuses saved images and predictions for the same model, meme and prompt condition.
Defaults to sparse random emojis; pass --pattern grid for the 4x4 evaluation.
"""
import argparse
import json
from pathlib import Path

from print_augmented_memes import checked_image, records


def indexed(rows, description):
    result = {}
    for row in rows:
        identity = row['original_id']
        if identity in result:
            raise ValueError(f'Duplicate ID in {description}: {identity}')
        result[identity] = row
    return result


def select_flip(args):
    clean_condition = f'clean_{args.prompt}'
    overlay_condition = f'overlay_{args.prompt}'
    clean = indexed(records(args.clean_data / f'{clean_condition}.jsonl'), 'clean data')
    overlay_root = args.clean_data if args.pattern == 'grid' else args.pattern_data
    overlay = indexed(records(overlay_root / f'{overlay_condition}.jsonl'), 'overlay data')
    clean_file = args.clean_results / args.arm / 'predictions.jsonl'
    overlay_file = (args.clean_results if args.pattern == 'grid' else args.pattern_results) / args.arm / 'predictions.jsonl'
    clean_predictions = indexed(
        [r for r in records(clean_file) if r['condition'] == clean_condition], 'clean predictions')
    overlay_predictions = indexed(
        [r for r in records(overlay_file) if r['condition'] == overlay_condition], 'overlay predictions')
    candidates = []
    for identity, augmented in overlay.items():
        if args.pattern != 'grid' and augmented['pattern'] != args.pattern:
            continue
        if args.test_id and identity != args.test_id:
            continue
        if identity not in clean or identity not in clean_predictions or identity not in overlay_predictions:
            raise ValueError(f'Missing paired data or prediction: {identity}')
        original = clean[identity]
        before, after = clean_predictions[identity], overlay_predictions[identity]
        if (not identity.startswith('TEST/') or original['split'] != 'test'
                or augmented['split'] != 'test' or original['is_augmented']
                or not augmented['is_augmented']):
            raise ValueError(f'Expected paired clean/augmented test images: {identity}')
        if original['target'] != augmented['target']:
            raise ValueError(f'Ground truth differs between images: {identity}')
        for row in (before, after):
            if row['true_label'] != int(original['target'] == 'misogynous'):
                raise ValueError(f'Prediction ground truth differs from test data: {identity}')
        if args.pattern != 'grid' and after['pattern'] != augmented['pattern']:
            raise ValueError(f'Prediction pattern differs from image: {identity}')
        for key in ('system_prompt', 'user_prompt'):
            if original[key] != augmented[key]:
                raise ValueError(f'Clean and augmented prompts differ: {identity}')
        if before['true_label'] == 1 and before['pred_label'] == 1 and after['pred_label'] == 0:
            candidates.append((original, augmented, before, after))
    if not candidates:
        raise ValueError(f'No clean TP -> augmented FN pair for {args.arm}, '
                         f'{args.pattern}, {args.prompt}, ID={args.test_id!r}.')
    return sorted(candidates, key=lambda pair: pair[0]['original_id'])[0], len(candidates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, default in (
        ('clean-data', 'data/processed/mami_lora_qlora_test_seed42'),
        ('pattern-data', 'data/processed/mami_unseen_patterns_seed42'),
        ('clean-results', 'results/mami_lora_qlora/test-1000-v1'),
        ('pattern-results', 'results/mami_lora_qlora/unseen-patterns-v1'),
    ):
        parser.add_argument('--' + flag, type=Path, default=Path(default))
    parser.add_argument('--arm', choices=('fp16-base', 'nf4-base', 'fp16-lora', 'nf4-qlora'),
                        default='fp16-lora')
    parser.add_argument('--pattern', choices=('sparse_random', 'dense_random', 'center', 'grid'),
                        default='sparse_random')
    parser.add_argument('--prompt', choices=('with_transcription', 'image_only'),
                        default='with_transcription')
    parser.add_argument('--test-id', help='Optional exact ID, e.g. TEST/12345.jpg')
    parser.add_argument('--output', type=Path, default=Path('outputs/figures/prediction_flip.png'))
    parser.add_argument('--show', action='store_true', help='Also open the saved figure')
    args = parser.parse_args()
    try:
        (clean, augmented, before, after), count = select_flip(args)
        images = [checked_image(row) for row in (clean, augmented)]
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'Error: {error}\n')

    from PIL import Image, ImageDraw, ImageFont, ImageOps

    figure = Image.new('RGB', (1480, 960), 'white')
    draw = ImageDraw.Draw(figure)
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:
        font = ImageFont.load_default()
    draw.text((30, 20), f"{clean['original_id']} | {args.arm} | {args.prompt}",
              fill='black', font=font)
    titles = ['Clean: true positive\nTrue: misogynous | predicted: misogynous',
              f'Augmented ({args.pattern}): false negative\nTrue: misogynous | predicted: clean']
    for index, (image, title) in enumerate(zip(images, titles)):
        x = 30 + index * 740
        fitted = ImageOps.contain(image, (680, 760))
        figure.paste(fitted, (x + (680-fitted.width)//2, 80 + (760-fitted.height)//2))
        draw.multiline_text((x, 860), title, fill='black', font=font, spacing=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.save(args.output)
    metadata = dict(arm=args.arm, pattern=args.pattern, prompt=args.prompt,
                    selection='First matching ID in sorted order; illustrative example.',
                    matching_pairs=count, clean=clean, augmented=augmented,
                    clean_prediction=before, augmented_prediction=after,
                    clean_predictions_file=str(args.clean_results / args.arm / 'predictions.jsonl'),
                    augmented_predictions_file=str((args.clean_results if args.pattern == 'grid'
                                                    else args.pattern_results) / args.arm / 'predictions.jsonl'))
    args.output.with_suffix('.json').write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f"Selected {clean['original_id']} from {count} matching pairs.")
    print(f"Clean: true=1, predicted=1, raw={before['raw_answer']!r}")
    print(f"Augmented: true=1, predicted=0, raw={after['raw_answer']!r}")
    print(f'Saved: {args.output.resolve()}')
    if args.show:
        figure.show()


if __name__ == '__main__':
    main()
