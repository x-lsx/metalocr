import json
import math
import threading
from pathlib import Path

import cv2
import numpy as np

from .common import read_image, write_csv


def validate_annotation(value, width, height):
    if not isinstance(value, dict):
        raise ValueError('Ожидается объект разметки')
    rotation = value.get('rotation', 0)
    if type(rotation) is not int or rotation not in (0, 90, 180, 270):
        raise ValueError('Поворот: 0, 90, 180 или 270')
    regions = value.get('regions', [])
    if not isinstance(regions, list) or len(regions) > 200:
        raise ValueError('Допускается до 200 областей')
    clean = []
    for region in regions:
        if region.get('label') not in ('marking', 'date_stamp'):
            raise ValueError('Неизвестный класс')
        points = region.get('points')
        if not isinstance(points, list) or len(points) != 4:
            raise ValueError('Область должна содержать четыре точки')
        for p in points:
            if not isinstance(p, list) or len(p) != 2 or any(type(n) not in (int, float) or not math.isfinite(n) for n in p):
                raise ValueError('Некорректные координаты')
            if not (0 <= p[0] <= width - 1 and 0 <= p[1] <= height - 1):
                raise ValueError('Точки за пределами изображения')
        polygon = np.asarray(points, dtype=np.float32)
        if not cv2.isContourConvex(polygon) or abs(cv2.contourArea(polygon)) < 4:
            raise ValueError('Нужен выпуклый четырёхугольник без пересечений')
        text = region.get('text', '')
        if not isinstance(text, str) or len(text) > 2000 or any(c in text for c in '\r\n\t'):
            raise ValueError('Одна область — одна строка, без табуляции')
        clean.append(dict(label=region['label'], points=points, text=text))
    unreadable, verified = value.get('unreadable', False), value.get('verified', False)
    if type(unreadable) is not bool or type(verified) is not bool:
        raise ValueError('Флаги должны быть логическими')
    heat = value.get('heat', '')
    if not isinstance(heat, str) or len(heat) > 100:
        raise ValueError('Некорректный номер плавки')
    if verified and not unreadable and (not heat.strip() or not any(r['label'] == 'marking' and r['text'].strip() for r in clean)):
        raise ValueError('Для проверки нужны номер плавки и текст маркировки')
    return dict(rotation=rotation, regions=clean, unreadable=unreadable, verified=verified, heat=heat.strip())


class AnnotationStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    def get(self, row):
        path = self.root / (row['id'] + '.json')
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))
        return dict(revision=0, regions=[], rotation=0, heat=row['heat'], verified=False, unreadable=False)

    def save(self, row, value):
        with self.lock:
            current = self.get(row)
            if value.get('revision') != current['revision']:
                raise FileExistsError('Разметка изменена в другой вкладке. Обновите страницу.')
            result = validate_annotation(value, int(row['width']), int(row['height']))
            result.update(revision=current['revision'] + 1, filename=row['filename'], id=row['id'])
            path = self.root / (row['id'] + '.json')
            tmp = path.with_suffix('.tmp')
            tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            tmp.replace(path)
            return result


def crop_quad(image, points):
    p = np.asarray(points, dtype=np.float32)
    width = max(2, int(max(np.linalg.norm(p[1]-p[0]), np.linalg.norm(p[2]-p[3]))))
    height = max(2, int(max(np.linalg.norm(p[3]-p[0]), np.linalg.norm(p[2]-p[1]))))
    target = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(p, target), (width, height))


def export_dataset(rows, images, store, output):
    corrected_groups = {}
    for row in rows:
        annotation = store.get(row)
        heat = annotation['heat'] if annotation['verified'] else row['heat']
        if corrected_groups.setdefault(heat, row['split']) != row['split']:
            raise ValueError(f'После исправления номера плавки обнаружена утечка {heat}. Исправьте реестр и пересоздайте разбиение до обучения.')
    output = Path(output)
    (output / 'crops').mkdir(parents=True, exist_ok=True)
    counts, annotations = {}, []
    for split in ('train', 'val', 'test'):
        rec, det = [], []
        for row in rows:
            if row['split'] != split:
                continue
            ann = store.get(row)
            if not ann['verified'] or ann['unreadable']:
                continue
            image = read_image(Path(images) / row['filename'])
            regions = []
            for i, region in enumerate(ann['regions']):
                regions.append(dict(transcription=region['text'] if region['label'] == 'marking' and region['text'].strip() else '###',
                                    points=region['points']))
                if region['label'] != 'marking' or not region['text'].strip():
                    continue
                crop = crop_quad(image, region['points'])
                relative = f"crops/{row['id']}_{i}.png"
                ok, encoded = cv2.imencode('.png', crop)
                if not ok:
                    raise ValueError('Не удалось сохранить crop')
                encoded.tofile(output / relative)
                rec.append(relative + '\t' + region['text'])
            # Use portable ASCII names for Paddle training.
            (output / 'images').mkdir(exist_ok=True)
            relative = f"images/{row['id']}.png"
            cv2.imencode('.png', image)[1].tofile(output / relative)
            det.append(relative + '\t' + json.dumps(regions, ensure_ascii=False))
            annotations.append(dict(filename=row['filename'], split=split, heat=ann['heat'], rotation=ann['rotation']))
        for kind, lines in [('rec', rec), ('det', det)]:
            (output / f'{kind}_{split}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
        counts[split] = dict(crops=len(rec), images=len(det))
    write_csv(output / 'verified.csv', annotations, ['filename', 'split', 'heat', 'rotation'])
    (output / 'export_report.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    return counts
