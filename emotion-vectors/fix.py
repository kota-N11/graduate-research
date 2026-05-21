import json
import re
from pathlib import Path

SEP = re.compile(r'\*\*\[\d+\]\*\*|\[story\s*\d+\]|\*{3,}|---+', re.IGNORECASE)

fixed = 0
for f in sorted(Path('data/stories').rglob('topic*.json')):
    d = json.loads(f.read_text())
    if len(d['stories']) < 12:
        raw = '\n'.join(d['stories'])
        parts = [p.strip() for p in SEP.split(raw) if len(p.strip()) >= 150]
        if len(parts) > len(d['stories']):
            d['stories'] = parts[:12]
            f.write_text(json.dumps(d, ensure_ascii=False, indent=2))
            fixed += 1
            print(f.parent.name + '/' + f.name, '->', len(parts), 'stories')
print('fixed', fixed, 'files')
