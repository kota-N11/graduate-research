import json
from pathlib import Path

deleted = 0
for f in sorted(Path('data/stories').rglob('topic*.json')):
    d = json.loads(f.read_text())
    if len(d['stories']) < 12:
        f.unlink()
        deleted += 1
        print(f.parent.name + '/' + f.name)
print('deleted', deleted)
