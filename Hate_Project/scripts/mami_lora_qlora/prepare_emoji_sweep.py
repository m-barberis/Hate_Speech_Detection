"""Freeze 100 class-balanced test memes, their existing layouts, and all 20 emojis."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

from PIL import Image, __version__ as pillow_version
import prepare_unseen_patterns as patterns

prep = patterns.prep
grid_data = patterns.source_data
CONDITIONS = ('overlay_with_transcription',)
DEFAULT_OUTPUT = 'data/processed/mami_patterns_100_all20_seed42'
# Preserve the existing layout assignments; sample independently within class/layout.
QUOTAS = {0: dict(dense_random=17, center=16, sparse_random=17),
          1: dict(dense_random=17, center=17, sparse_random=16)}


def select_ids(rows, seed=42):
    if len({r['original_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source meme IDs')
    selected = []
    for label in (0, 1):
        for pattern in patterns.PATTERNS:
            candidates = sorted(r['original_id'] for r in rows
                if int(r['target'] == 'misogynous') == label and r['pattern'] == pattern)
            rng = random.Random(f'{seed}:{label}:{pattern}')
            selected.extend(rng.sample(candidates, QUOTAS[label][pattern]))
    return sorted(selected)


def sources(root):
    converted, _ = patterns.load(root)
    manifest = json.loads((root/'manifest.json').read_text())
    grid_root = Path(manifest['source_data'])
    original, _ = grid_data.load(grid_root)
    return converted[CONDITIONS[0]], manifest, original, grid_root


def create(args):
    source = Path(args.source_data).resolve()
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError(f'Preserve existing preparation: {out}')
    rows, old_manifest, original, grid_root = sources(source)
    ids = select_ids(rows)
    source_rows = {r['original_id']: r for r in rows}
    clean = {r['original_id']: r for r in original['clean_with_transcription']}
    assets = json.loads((grid_root/'manifest.json').read_text())['assets']
    codes = sorted(grid_data.evaluation_emojis()[0])
    if len(assets) != 20 or {a['code'] for a in assets} != set(codes):
        raise ValueError('All 20 original evaluation emoji assets are required')
    opened = {}
    for asset in assets:
        if prep.digest(asset['path']) != asset['sha256']:
            raise ValueError('Evaluation emoji bytes changed')
        with Image.open(asset['path']) as image:
            opened[asset['code']] = image.convert('RGBA')
    out.mkdir(parents=True)
    (out/'images_overlay').mkdir()
    records, geometries = [], {}
    for index, identity in enumerate(ids):
        row = source_rows[identity]
        geometry = old_manifest['geometries'][identity]
        if geometry['pattern'] != row['pattern'] or geometry['nominal_alpha'] != .8:
            raise ValueError('Expected the existing fixed alpha-0.8 layout')
        geometries[identity] = geometry
        with Image.open(clean[identity]['image_path']) as source_image:
            for code in codes:
                destination = out/'images_overlay'/f'{index:03d}_{code}.jpg'
                patterns.render(source_image, opened[code], geometry['placements'], alpha=.8).save(
                    destination, format='JPEG')
                records.append(dict(row, emoji_code=code, image_path=str(destination),
                    image_sha256=prep.digest(destination)))
    with (out/f'{CONDITIONS[0]}.jsonl').open('x') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False)+'\n')
    prep.save_json(out/'manifest.json', dict(source_data=str(source), grid_data=str(grid_root),
        source_preparation_sha256=prep.digest(source/'complete.json'),
        grid_preparation_sha256=prep.digest(grid_root/'complete.json'),
        training_manifest_sha256=old_manifest['training_manifest_sha256'],
        selected_ids=ids, seed=42, class_pattern_quotas=QUOTAS,
        pattern_counts=dict(Counter(source_rows[i]['pattern'] for i in ids)),
        emoji_codes=codes, assets=assets, geometries=geometries,
        creator_sha256=prep.digest(__file__), renderer_sha256=prep.digest(patterns.__file__),
        Pillow=pillow_version,
        qualification='Exploratory 100-meme subset, 50 per class; one inherited layout and fixed placements per meme; all 20 evaluation emojis; current JPEG and inference protocol, not an exact historical reproduction.'))
    prep.save_json(out/'complete.json', dict(schema='mami-patterns-100-all20-v1',
        n_originals=100, predictions_per_arm=2000,
        manifest_sha256=prep.digest(out/'manifest.json'),
        jsonl_sha256={CONDITIONS[0]: prep.digest(out/f'{CONDITIONS[0]}.jsonl')}))
    print(f'Prepared 100 memes x 20 emojis = 2,000 records: {out}')


def load(root):
    root = Path(root).resolve()
    done = json.loads((root/'complete.json').read_text())
    if (done['schema'] != 'mami-patterns-100-all20-v1' or done['n_originals'] != 100 or
        done['predictions_per_arm'] != 2000 or prep.digest(root/'manifest.json') != done['manifest_sha256']):
        raise ValueError('Sweep preparation changed or incomplete')
    manifest = json.loads((root/'manifest.json').read_text())
    source = Path(manifest['source_data'])
    if prep.digest(source/'complete.json') != manifest['source_preparation_sha256']:
        raise ValueError('Original pattern preparation changed')
    rows, old_manifest, original, grid_root = sources(source)
    if str(grid_root) != manifest['grid_data'] or prep.digest(grid_root/'complete.json') != manifest['grid_preparation_sha256']:
        raise ValueError('Original clean/grid preparation changed')
    ids = select_ids(rows)
    if ids != manifest['selected_ids']:
        raise ValueError('Frozen stratified sample differs')
    codes = sorted(grid_data.evaluation_emojis()[0])
    if manifest['emoji_codes'] != codes:
        raise ValueError('Expected all 20 original evaluation emojis')
    original_assets = json.loads((grid_root/'manifest.json').read_text())['assets']
    if manifest['assets'] != original_assets:
        raise ValueError('Emoji inventory differs from original test')
    for asset in manifest['assets']:
        if prep.digest(asset['path']) != asset['sha256']:
            raise ValueError('Evaluation emoji bytes changed')
    condition = CONDITIONS[0]
    path = root/f'{condition}.jsonl'
    if set(done['jsonl_sha256']) != set(CONDITIONS) or prep.digest(path) != done['jsonl_sha256'][condition]:
        raise ValueError('Sweep records changed')
    converted = [json.loads(line) for line in path.read_text().splitlines()]
    if [(r['original_id'], r['emoji_code']) for r in converted] != [(i, c) for i in ids for c in codes]:
        raise ValueError('Missing, duplicate or reordered meme/emoji records')
    old = {r['original_id']: r for r in rows}
    for identity in ids:
        if manifest['geometries'][identity] != old_manifest['geometries'][identity]:
            raise ValueError('Existing pattern placements changed')
    counts = Counter(old[i]['pattern'] for i in ids)
    if dict(counts) != manifest['pattern_counts']:
        raise ValueError('Pattern counts changed')
    for row in converted:
        for key in ('original_id', 'target', 'split', 'system_prompt', 'user_prompt', 'pattern', 'is_augmented'):
            if row[key] != old[row['original_id']][key]:
                raise ValueError(f'Sweep source differs: {key}')
        if prep.digest(row['image_path']) != row['image_sha256']:
            raise ValueError('Sweep overlay bytes changed')
    return {condition: converted}, done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-data', default=patterns.DEFAULT_OUTPUT)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    create(parser.parse_args())


if __name__ == '__main__':
    main()
