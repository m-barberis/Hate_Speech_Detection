"""Freeze four validation prompt conditions using existing images and split."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

import prepare_comparison_data as prep


CONDITIONS = ('clean_image_only', 'overlay_image_only',
              'clean_with_transcription', 'overlay_with_transcription')
SCHEMA = 'mami-four-validation-conditions-v1'


def verified_source(root):
    complete = json.loads((root / 'complete.json').read_text())
    if prep.digest(root / 'manifest.json') != complete['manifest_sha256']:
        raise ValueError('Prepared manifest changed')
    for name, sha in complete['jsonl_sha256'].items():
        if prep.digest(root / f'{name}.jsonl') != sha:
            raise ValueError(f'Prepared records changed: {name}')
    manifest = json.loads((root / 'manifest.json').read_text())
    table = Path(manifest['train_table']['path'])
    if prep.digest(table) != manifest['train_table']['sha256']:
        raise ValueError('Training table changed since preparation')
    return complete, manifest


def create(args, expected_count=1000):
    root, out = Path(args.data).resolve(), Path(args.output).resolve()
    if out.exists():
        raise FileExistsError(f'Use a new validation output directory: {out}')
    _, manifest = verified_source(root)
    originals = {f"TRAINING/{r['file_name']}": r for r in manifest['originals']}
    rows = prep.read_training_data(Path(manifest['train_table']['path']).parent,
                                  expected_per_class=(len(originals) // 2))
    transcriptions = {f"TRAINING/{r['file_name']}": r['text'] for r in rows}
    conditions = {}
    for source_key, image_condition in (('validation_clean', 'clean'), ('validation_overlay', 'overlay')):
        source = [json.loads(line) for line in (root / f'{source_key}.jsonl').read_text().splitlines()]
        conditions[f'{image_condition}_image_only'] = source
        with_text = []
        for row in source:
            text = transcriptions[row['original_id']]
            if not text:
                raise ValueError('Transcription-present condition requires nonempty text')
            # Reuse the historical prompt template, with dropout disabled.
            system, prompt = prep.original.construct_prompts(text, drop_probability=0.0, rng=random.Random(42))
            with_text.append(dict(row, system_prompt=system, user_prompt=prompt))
        conditions[f'{image_condition}_with_transcription'] = with_text
    validate_conditions(conditions, originals, expected_count)
    # Check bytes once per distinct file. No images are rendered or rewritten.
    checked = {}
    for records in conditions.values():
        for row in records:
            path = row['image_path']
            if path not in checked:
                checked[path] = prep.digest(path)
            if checked[path] != row['image_sha256']:
                raise ValueError(f'Validation image changed: {path}')
    out.mkdir(parents=True)
    for condition, records in conditions.items():
        with (out / f'{condition}.jsonl').open('x', encoding='utf-8') as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    complete = dict(schema=SCHEMA, source_prepared_complete_sha256=prep.digest(root / 'complete.json'),
                    source_training_table_sha256=manifest['train_table']['sha256'],
                    source_manifest_sha256=prep.digest(root / 'manifest.json'),
                    creator_sha256=prep.digest(__file__),
                    counts={name: len(records) for name, records in conditions.items()},
                    jsonl_sha256={name: prep.digest(out / f'{name}.jsonl') for name in conditions},
                    transcription_policy='original unmodified transcription for clean and overlay',
                    image_policy='reuse existing validation clean/overlay files; no regeneration')
    prep.save_json(out / 'complete.json', complete)
    print(json.dumps(complete, indent=2))


def validate_conditions(conditions, originals, expected_count=1000):
    if set(conditions) != set(CONDITIONS):
        raise ValueError('Expected all four validation conditions')
    ids = None
    for condition in CONDITIONS:
        rows = conditions[condition]
        current = [r['original_id'] for r in rows]
        if len(rows) != expected_count or len(set(current)) != expected_count:
            raise ValueError(f'Incorrect validation count or duplicate IDs: {condition}')
        if ids is not None and set(current) != ids:
            raise ValueError('Validation conditions contain different original images')
        ids = set(current)
        labels = Counter()
        for row in rows:
            original = originals[row['original_id']]
            if original['split'] != 'validation' or row['split'] != 'validation':
                raise ValueError('Training original found in validation')
            expected_label = 'misogynous' if original['label'] else 'clean'
            if row['target'] != expected_label:
                raise ValueError('Validation target differs from the source manifest')
            labels[original['label']] += 1
            if row['is_augmented'] != condition.startswith('overlay_'):
                raise ValueError('Validation image condition differs')
            if row['user_prompt'].startswith('Meme Text:') != condition.endswith('_with_transcription'):
                raise ValueError('Validation transcription condition differs')
        if labels != {0: expected_count // 2, 1: expected_count // 2}:
            raise ValueError('Validation labels are not balanced')
    for kind in ('clean', 'overlay'):
        plain = {r['original_id']: r for r in conditions[f'{kind}_image_only']}
        for row in conditions[f'{kind}_with_transcription']:
            old = plain[row['original_id']]
            for key in ('image_path', 'image_sha256', 'target', 'is_augmented', 'system_prompt'):
                if row[key] != old[key]:
                    raise ValueError(f'Transcription condition changed more than its user prompt: {key}')


def load(data, validation, expected_count=1000):
    root, validation = Path(data), Path(validation)
    complete = json.loads((validation / 'complete.json').read_text())
    if complete['schema'] != SCHEMA or complete['source_prepared_complete_sha256'] != prep.digest(root / 'complete.json'):
        raise ValueError('Validation conditions belong to different prepared data')
    if complete['source_manifest_sha256'] != prep.digest(root / 'manifest.json'):
        raise ValueError('Validation source manifest changed')
    conditions = {}
    for name in CONDITIONS:
        path = validation / f'{name}.jsonl'
        if prep.digest(path) != complete['jsonl_sha256'][name]:
            raise ValueError(f'Validation records changed: {name}')
        conditions[name] = [json.loads(line) for line in path.read_text().splitlines()]
        if len(conditions[name]) != complete['counts'][name]:
            raise ValueError('Validation completion count differs')
    manifest = json.loads((root / 'manifest.json').read_text())
    originals = {f"TRAINING/{r['file_name']}": r for r in manifest['originals']}
    validate_conditions(conditions, originals, expected_count)
    return conditions, complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/processed/mami_lora_qlora_seed42')
    parser.add_argument('--output', default='data/processed/mami_lora_qlora_validation_seed42')
    create(parser.parse_args())


if __name__ == '__main__':
    main()
