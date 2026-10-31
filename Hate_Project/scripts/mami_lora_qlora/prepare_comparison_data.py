"""Freeze the approved MAMI 90/10 split and reuse the original augmentation.

CPU-only. Existing output directories are never overwritten. A completion marker
is written only after all images and records have been generated successfully.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random

import pandas as pd
from PIL import Image, __version__ as pillow_version

# Reuse the historical implementation without requiring a particular cwd or
# adding its directory to the global Python import search path.
_original_path = Path(__file__).resolve().parents[1] / 'mami_finetuning' / 'augment_train_set.py'
_original_spec = importlib.util.spec_from_file_location('_original_mami_augmentation', _original_path)
original = importlib.util.module_from_spec(_original_spec)
_original_spec.loader.exec_module(original)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def read_training_data(root, expected_per_class=5000):
    table = root / 'training.csv'
    frame = pd.read_csv(table, sep='\t', encoding='utf-8-sig',
                        dtype=str, keep_default_na=False)
    frame.columns = frame.columns.str.strip()
    required = {'file_name', 'misogynous', 'Text Transcription'}
    if not required <= set(frame.columns):
        raise ValueError(f'Missing columns: {required - set(frame.columns)}')
    rows, seen = [], set()
    for record in frame.to_dict('records'):
        name = record['file_name'].strip()
        if not name or Path(name).name != name or name in seen:
            raise ValueError(f'Empty, non-flat, or duplicate filename: {name!r}')
        label = record['misogynous'].strip()
        if label not in ('0', '1'):
            raise ValueError(f'Invalid binary label for {name}: {label!r}')
        image = root / name
        if not image.is_file():
            image = image.with_suffix('.png')
        if not image.is_file():
            raise FileNotFoundError(name)
        text = record['Text Transcription'].strip()
        # Preserve the original script's handling of a literal missing value.
        if text == 'nan':
            text = ''
        rows.append(dict(file_name=name, label=int(label), text=text,
                         image_path=str(image.resolve())))
        seen.add(name)
    if Counter(r['label'] for r in rows) != {0: expected_per_class, 1: expected_per_class}:
        raise ValueError(f'Expected {expected_per_class} originals per class')
    if len({r['image_path'] for r in rows}) != len(rows):
        raise ValueError('Multiple table rows resolve to the same image path')
    return rows


def stratified_split(rows, seed):
    """Select 10% per class independently of augmentation randomness."""
    rng = random.Random(seed)
    validation = set()
    for label in (0, 1):
        names = sorted(r['file_name'] for r in rows if r['label'] == label)
        if not names or len(names) % 10:
            raise ValueError('Each class count must be a positive multiple of ten')
        rng.shuffle(names)
        validation.update(names[:len(names) // 10])
    return {r['file_name']: ('validation' if r['file_name'] in validation else 'train')
            for r in rows}


def load_assets(root):
    assets = []
    for code, name in original.NEW_TRAIN_EMOJIS.items():
        path = next((p for p in (root / f'emoji_u{code}.png',
                                 root / f'emoji_u{code.upper()}.png') if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f'Required training emoji missing: {code}')
        with Image.open(path) as source:
            source.load()
            if source.size != (512, 512) or source.mode != 'RGBA':
                raise ValueError(f'Expected 512x512 RGBA asset: {path}')
            image = source.copy()
        assets.append((image, dict(code=code, name=name, path=str(path.resolve()),
                                   sha256=digest(path))))
    return assets


def prepare(args):
    root, emoji_root, out = (Path(p).resolve() for p in
                             (args.train_dir, args.emoji_dir, args.output))
    if out.exists():
        raise FileExistsError(f'Choose a new output directory: {out}')
    rows = read_training_data(root)
    split = stratified_split(rows, args.seed)
    assets = load_assets(emoji_root)
    source = Path(original.__file__).resolve()
    manifest = dict(
        seed=args.seed, validation_fraction=0.1,
        train_table=dict(path=str(root / 'training.csv'), sha256=digest(root / 'training.csv')),
        augmentation_source=dict(path=str(source), sha256=digest(source)),
        preparation_source_sha256=digest(__file__),
        versions=dict(pandas=pd.__version__, Pillow=pillow_version),
        settings=dict(pattern='original_apply_4x4_grid', nominal_alpha=0.8,
                      training_transcription_dropout=0.5,
                      validation_prompt='image-only', augmented_encoding='original_Pillow_JPEG_defaults'),
        assets=[meta for _, meta in assets],
        originals=[dict(file_name=r['file_name'], label=r['label'],
                        image_path=r['image_path'], sha256=digest(r['image_path']),
                        split=split[r['file_name']]) for r in rows],
        qualifications=[
            'New controlled follow-up; not an exact historical reproduction.',
            'Five known train/test filename collisions have different images.',
            'Cross-split duplicate audit under different filenames deferred.',
            'Noto release unknown; exact supplied training asset hashes recorded.',
            'Validation monitors training; evaluate final two-epoch checkpoint in both arms.',
        ])
    out.mkdir(parents=True)
    for part in ('train', 'validation'):
        (out / part / 'images_augmented').mkdir(parents=True)
    save_json(out / 'manifest.json', manifest)
    rng = random.Random(args.seed)
    image_only_prompt = original.construct_prompts('', rng=random.Random(args.seed))
    counts = Counter()
    jsonl_paths = {k: out / f'{k}.jsonl' for k in ('train', 'validation_clean', 'validation_overlay')}
    from contextlib import ExitStack
    with ExitStack() as stack:
        streams = {k: stack.enter_context(p.open('x', encoding='utf-8'))
                   for k, p in jsonl_paths.items()}
        for index, row in enumerate(rows, 1):
            name, part = row['file_name'], split[row['file_name']]
            # Same draw order as the original: clean dropout, augmented dropout,
            # then emoji selection. CSV order is preserved for these draws.
            prompts = [original.construct_prompts(row['text'], rng=rng) for _ in range(2)]
            image, asset = rng.choice(assets)
            if part == 'validation':
                prompts = [image_only_prompt] * 2
            destination = out / part / 'images_augmented' / f'aug_{name}'
            with Image.open(row['image_path']) as base:
                augmented = original.apply_4x4_grid(base, image, alpha=0.8)
                augmented.save(destination, format='JPEG')
            for augmented_flag in (False, True):
                system, prompt = prompts[int(augmented_flag)]
                key = 'train' if part == 'train' else (
                    'validation_overlay' if augmented_flag else 'validation_clean')
                record = dict(
                    original_id=f'TRAINING/{name}', split=part,
                    image_path=str(destination) if augmented_flag else row['image_path'],
                    system_prompt=system, user_prompt=prompt,
                    target='misogynous' if row['label'] else 'clean',
                    is_augmented=augmented_flag,
                    emoji_code=asset['code'] if augmented_flag else None,
                    image_sha256=digest(destination) if augmented_flag else manifest['originals'][index-1]['sha256'])
                streams[key].write(json.dumps(record, ensure_ascii=False) + '\n')
                counts[key] += 1
            if index % 500 == 0:
                print(f'Prepared {index}/{len(rows)} original images', flush=True)
    sizes = Counter(split.values())
    expected = dict(train=2 * sizes['train'], validation_clean=sizes['validation'],
                    validation_overlay=sizes['validation'])
    if dict(counts) != expected:
        raise ValueError(f'Unexpected record counts: {dict(counts)}')
    complete = dict(counts=dict(counts), manifest_sha256=digest(out / 'manifest.json'),
                    jsonl_sha256={k: digest(p) for k, p in jsonl_paths.items()},
                    final_checkpoint_policy='final_epoch_2_no_validation_selection')
    save_json(out / 'complete.json', complete)
    print(json.dumps(complete, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', default='data/raw/TRAINING')
    parser.add_argument('--emoji-dir', default=os.environ.get('EMOJI_DIR'))
    parser.add_argument('--output', default='data/processed/mami_lora_qlora_seed42')
    parser.add_argument('--seed', type=int, choices=[42], default=42)
    args = parser.parse_args()
    if not args.emoji_dir:
        parser.error('Set EMOJI_DIR or pass --emoji-dir')
    prepare(args)


if __name__ == '__main__':
    main()
