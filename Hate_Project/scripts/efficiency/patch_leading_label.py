"""Apply the approved parser revision to a server copy, preserving its dispatch fix."""
from pathlib import Path
import sys

path = Path(sys.argv[1] if len(sys.argv) > 1 else 'scripts/efficiency/compare.py')
source = path.read_text()
if 'leading-label-v1' in source:
    raise SystemExit('Parser revision already present; no changes made.')

start = source.index('def parse_answer(text):')
end = source.index('def quality(', start)
parser = r'''SETTINGS['answer_parser'] = 'leading-label-v1'


def parse_answer(text):
    answer = text.strip().lower()
    match = re.match(r'^(clean|misogynous)(?=$|[\s.,:;!?])', answer)
    if not match:
        return -1
    labels = set(re.findall(r'\b(?:clean|misogynous)\b', answer.splitlines()[0]))
    if len(labels) != 1:
        return -1
    return {'clean': 0, 'misogynous': 1}[match.group(1)]


def strict_answer(text):
    return text.strip().lower() in ('clean', 'misogynous')


'''
updated = source[:start] + parser + source[end:]
old = 'pred_label=parse_answer(raw), raw_answer=raw,'
new = old + '\n                    strict_format=strict_answer(raw),'
if updated.count(old) != 1:
    raise SystemExit('Unexpected prediction writer; no changes made.')
updated = updated.replace(old, new, 1)
old = '        result[mode].update(dict(\n'
new = old + "            strict_format_compliance=sum(strict_answer(r['raw_answer']) for r in records if r['pass_id'] == 0) / len(ids),\n"
if updated.count(old) != 1:
    raise SystemExit('Unexpected summary writer; no changes made.')
updated = updated.replace(old, new, 1)
compile(updated, str(path), 'exec')

# Check interpretation before touching the file; no model imports or GPU work.
import re
namespace = {'SETTINGS': {}, 're': re}
exec(parser, namespace)
for answer, expected in [('clean', 0), ('clean\nExplanation.', 0),
                         ('misogynous', 1), ('not misogynous', -1),
                         ('clean or misogynous', -1), ('cleanliness', -1)]:
    assert namespace['parse_answer'](answer) == expected
assert not namespace['strict_answer']('clean\nExplanation.')

backup = path.with_suffix('.before_parser_fix.py')
with backup.open('x') as stream:
    stream.write(source)
path.write_text(updated)
print('Parser updated; strict compliance will be reported separately.')
print('Backup:', backup)
