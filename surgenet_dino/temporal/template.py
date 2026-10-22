"""Official Basic image allowlist; standard library only for the OpenCV environment."""

import csv
import re
from pathlib import Path


def read_template(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['filename', 'phase-id']:
            raise ValueError('Expected the official filename,phase-id template')
        rows = list(reader)
    names = [r['filename'] for r in rows]
    if not names or len(set(names)) != len(names):
        raise ValueError('Empty template or duplicate filename')
    if any(not re.fullmatch(r'\d{2}_\d{6}', n) or int(n.split('_')[1]) % 30 for n in names):
        raise ValueError('Template names must identify whole 1 fps seconds')
    return names
