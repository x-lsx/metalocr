import csv
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def config(path=None):
    settings = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
    if path:
        settings.update(json.loads(Path(path).read_text(encoding='utf-8')))
    if not settings['rotations'] or any(r not in (0, 90, 180, 270) for r in settings['rotations']):
        raise ValueError('rotations должен содержать повороты 0/90/180/270')
    return settings


def normalize(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text)).strip()


def heat_number(text):
    # No global Latin/Cyrillic substitutions: these may be meaningful in markings.
    match = re.search(r'\b(?:19|20)\d{2}[-–][A-Za-zА-Яа-яЁё0-9]+\b', normalize(text))
    return match.group().replace('–', '-') if match else ''


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def read_image(path):
    import cv2
    import numpy as np
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'Повреждено изображение: {path}')
    return img
