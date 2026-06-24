import re
import zipfile
from pathlib import Path

root = Path('.').resolve()
pattern = re.compile(r'frame[_-]?(\d+)\.png$', re.IGNORECASE)

files = []
for p in root.rglob('*.png'):
    m = pattern.search(p.name)
    if m:
        files.append((int(m.group(1)), p))

files.sort(key=lambda x: x[0])
files = [p for _, p in files]

if not files:
    print('No frame PNG files found.')
    raise SystemExit(1)

zip_path = root / 'frames_archive.zip'
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for p in files:
        arcname = p.relative_to(root)
        z.write(p, arcname)

print(f'Created {zip_path} with {len(files)} files.')
