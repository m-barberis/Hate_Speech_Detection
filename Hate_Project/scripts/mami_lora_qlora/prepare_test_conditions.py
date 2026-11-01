"""Freeze all 1,000 MAMI test memes and one unseen-emoji overlay per meme."""
import argparse
import ast
from collections import Counter
import csv
import json
import os
from pathlib import Path
import random

import pandas as pd
from PIL import Image, __version__ as pillow_version
import prepare_comparison_data as prep
import prepare_validation_conditions as validation

CONDITIONS = validation.CONDITIONS
DEFAULT_OUTPUT = 'data/processed/mami_lora_qlora_test_seed42'


def evaluation_emojis():
    path = Path(__file__).resolve().parents[1] / 'mami_finetuning' / 'eval_finetuned.py'
    tree = ast.parse(path.read_text())
    values = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'TOP_20_EMOJIS' for t in node.targets)]
    if len(values) != 1 or len(values[0]) != 20 or set(values[0]) & set(prep.original.NEW_TRAIN_EMOJIS):
        raise ValueError('Expected 20 original evaluation emojis disjoint from training identities')
    return values[0], path


def discover(explicit, filename, root):
    if explicit:
        path = Path(explicit).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    found = sorted(p.resolve() for p in Path(root).rglob('*') if p.is_file() and p.name.lower() == filename.lower())
    if len(found) != 1:
        raise ValueError(f'Expected one {filename} under {root}; found {found}. Pass its explicit path.')
    return found[0]


def read_test(texts, labels, images, expected=1000):
    frame = pd.read_csv(texts, sep='\t', dtype=str, keep_default_na=False, encoding='utf-8-sig')
    frame.columns = frame.columns.str.strip()
    if not {'file_name', 'Text Transcription'} <= set(frame.columns):
        raise ValueError('Test.csv needs file_name and Text Transcription columns')
    transcriptions = {}
    for row in frame.to_dict('records'):
        name = row['file_name'].strip()
        if not name or Path(name).name != name or name in transcriptions:
            raise ValueError(f'Invalid or duplicate test transcription filename: {name}')
        if not row['Text Transcription'].strip():
            raise ValueError(f'Empty test transcription: {name}')
        transcriptions[name] = row['Text Transcription'].strip()
    truth = {}
    with labels.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.reader(stream, delimiter='\t'):
            if not row or row[0].strip() == 'file_name':
                continue
            name = row[0].strip()
            if len(row) < 2 or row[1].strip() not in ('0', '1') or name in truth or Path(name).name != name:
                raise ValueError(f'Invalid or duplicate test label: {row[:2]}')
            truth[name] = int(row[1])
    if set(truth) != set(transcriptions) or len(truth) != expected:
        raise ValueError('Test label/text IDs must match exactly, with 1,000 unique test images')
    if Counter(truth.values()) != {0: expected//2, 1: expected//2}:
        raise ValueError('Expected balanced MAMI test labels')
    rows = []
    for name in sorted(truth):
        image = images / name
        if not image.is_file():
            image = image.with_suffix('.png')
        if not image.is_file():
            raise FileNotFoundError(image)
        with Image.open(image) as source:
            source.load()
        rows.append(dict(file_name=name, label=truth[name], text=transcriptions[name],
                         image_path=str(image.resolve()), sha256=prep.digest(image)))
    if len({r['image_path'] for r in rows}) != expected:
        raise ValueError('Test filenames resolve to duplicate image paths')
    if len({Path(r['file_name']).stem for r in rows}) != expected:
        raise ValueError('Test filenames would overwrite an overlay destination')
    return rows


def assignments(rows, codes):
    rng = random.Random(42)
    assigned = {}
    # Exactly 25 memes of each class per emoji on the full 1,000-image test.
    for label in (0, 1):
        names = sorted(r['file_name'] for r in rows if r['label'] == label)
        rng.shuffle(names)
        for index, name in enumerate(names):
            assigned[name] = codes[index % len(codes)]
    return assigned


def create(args, expected=1000):
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError(f'Preserve existing preparation; choose a new output path: {out}')
    root = Path(args.training_data).resolve()
    _, training = validation.verified_source(root)
    texts = discover(args.texts, 'Test.csv', args.raw_root)
    labels = discover(args.labels, 'test_labels.txt', args.raw_root)
    images = Path(args.images).resolve() if args.images else texts.parent
    rows = read_test(texts, labels, images, expected)
    training_hashes = {r['sha256'] for r in training['originals']}
    duplicates = [r['file_name'] for r in rows if r['sha256'] in training_hashes]
    if duplicates:
        raise ValueError(f'Exact train/validation-test byte duplicates: {duplicates}')
    codes, emoji_source = evaluation_emojis()
    emoji_root = Path(args.emoji_dir).resolve() if args.emoji_dir else Path(training['assets'][0]['path']).parent
    assets, metadata = {}, []
    for code, name in codes.items():
        path = next((p for p in (emoji_root/f'emoji_u{code}.png', emoji_root/f'emoji_u{code.upper()}.png') if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f'Missing evaluation emoji {code} in {emoji_root}; pass --emoji-dir')
        with Image.open(path) as source:
            source.load()
            if source.size != (512, 512) or source.mode != 'RGBA':
                raise ValueError(f'Expected 512x512 RGBA emoji: {path}')
            assets[code] = source.copy()
        metadata.append(dict(code=code, name=name, path=str(path), sha256=prep.digest(path)))
    chosen = assignments(rows, list(codes))
    out.mkdir(parents=True)
    (out/'images_overlay').mkdir()
    conditions = {name: [] for name in CONDITIONS}
    for index, row in enumerate(rows, 1):
        destination = out/'images_overlay'/f"{Path(row['file_name']).stem}.jpg"
        with Image.open(row['image_path']) as image:
            prep.original.apply_4x4_grid(image, assets[chosen[row['file_name']]], alpha=0.8).save(destination, format='JPEG')
        for condition in CONDITIONS:
            overlay = condition.startswith('overlay_')
            with_text = condition.endswith('_with_transcription')
            system, prompt = prep.original.construct_prompts(row['text'] if with_text else '',
                drop_probability=0.0 if with_text else 1.0, rng=random.Random(42))
            path = str(destination) if overlay else row['image_path']
            conditions[condition].append(dict(original_id=f"TEST/{row['file_name']}", split='test',
                image_path=path, image_sha256=prep.digest(path), system_prompt=system, user_prompt=prompt,
                target='misogynous' if row['label'] else 'clean', is_augmented=overlay,
                emoji_code=chosen[row['file_name']] if overlay else None))
        if index % 250 == 0:
            print(f'Prepared test memes {index}/{len(rows)}', flush=True)
    manifest = dict(seed=42, n=len(rows), original_rows=rows, emoji_assignments=chosen,
        assets=metadata, pattern='original_4x4_grid', nominal_alpha=0.8,
        image_encoding='original Pillow JPEG defaults', Pillow=pillow_version,
        labels=dict(path=str(labels), sha256=prep.digest(labels)),
        texts=dict(path=str(texts), sha256=prep.digest(texts)),
        training_manifest_sha256=prep.digest(root/'manifest.json'),
        original_augmentation_sha256=prep.digest(prep.original.__file__),
        emoji_definition_source_sha256=prep.digest(emoji_source),
        same_name_training_collisions=sorted(set(r['file_name'] for r in rows) & set(r['file_name'] for r in training['originals'])),
        qualifications=['All test memes held out of this adapter training; 100 were previously used in inference benchmarking.',
            'Exact byte duplicates against train and validation rejected; cross-name RGB/semantic duplicate audit not exhaustive.',
            'Emoji identities disjoint from augmentation training; one fixed overlay per meme, not a per-emoji full sweep.',
            'Original transcription retained for overlay conditions; clean JPEG is not re-encoded.'])
    prep.save_json(out/'manifest.json', manifest)
    for condition, records in conditions.items():
        with (out/f'{condition}.jsonl').open('x') as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    complete = dict(schema='mami-held-out-test-v1', n=len(rows),
        manifest_sha256=prep.digest(out/'manifest.json'),
        creator_sha256=prep.digest(__file__), counts={c: len(r) for c, r in conditions.items()},
        jsonl_sha256={c: prep.digest(out/f'{c}.jsonl') for c in conditions})
    prep.save_json(out/'complete.json', complete)
    print(json.dumps(complete, indent=2))


def load(root, expected=1000):
    root = Path(root).resolve()
    done = json.loads((root/'complete.json').read_text())
    if done['schema'] != 'mami-held-out-test-v1' or done['n'] != expected:
        raise ValueError('Expected complete 1,000-image test preparation')
    if prep.digest(root/'manifest.json') != done['manifest_sha256'] or set(done['jsonl_sha256']) != set(CONDITIONS):
        raise ValueError('Test manifest or conditions changed')
    conditions, reference = {}, None
    for condition in CONDITIONS:
        path = root/f'{condition}.jsonl'
        if prep.digest(path) != done['jsonl_sha256'][condition]:
            raise ValueError(f'Test records changed: {condition}')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        identities = [(r['original_id'], r['target']) for r in rows]
        if len(rows) != expected or len(set(i for i, _ in identities)) != expected or (reference is not None and identities != reference):
            raise ValueError('Test condition IDs or labels differ')
        reference = identities
        if Counter(r['target'] for r in rows) != {'clean': expected//2, 'misogynous': expected//2}:
            raise ValueError('Test labels are not balanced binary targets')
        for row in rows:
            if row['split'] != 'test' or not row['original_id'].startswith('TEST/'):
                raise ValueError('Non-test row in test condition')
            if row['is_augmented'] != condition.startswith('overlay_') or row['user_prompt'].startswith('Meme Text:') != condition.endswith('_with_transcription'):
                raise ValueError('Test condition semantics differ')
        conditions[condition] = rows
    checked = {}
    for rows in conditions.values():
        for row in rows:
            path = row['image_path']
            if path not in checked:
                checked[path] = prep.digest(path)
            if checked[path] != row['image_sha256']:
                raise ValueError(f'Test image changed: {path}')
    return conditions, done


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-data', default='data/processed/mami_lora_qlora_seed42')
    p.add_argument('--raw-root', default='data/raw')
    for name in ('texts', 'labels', 'images'):
        p.add_argument('--'+name)
    p.add_argument('--emoji-dir', default=os.environ.get('EMOJI_DIR'))
    p.add_argument('--output', default=DEFAULT_OUTPUT)
    create(p.parse_args())


if __name__ == '__main__':
    main()
