import logging
import hashlib
import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .common import EXTENSIONS, config, normalize, read_image
from .heat import extract_heat
from .annotations import crop_quad
from .selection import fuse_views, reading_rotation


def prepare_view(image, mode='clahe', clip=3.0):
    if mode == 'original':
        return image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if mode == 'clahe':
        gray = cv2.createCLAHE(float(clip), (8, 8)).apply(gray)
    elif mode != 'gray':
        raise ValueError(f'Unknown preprocessing mode: {mode}')
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def recognition_canvas(image, height=48, width=480):
    """Use the same aspect-preserving zero-normalized padding as training."""
    h, w = image.shape[:2]
    resized_width = min(width, max(1, int(np.ceil(height * w / h))))
    result = np.full((height, width, 3), 127.5, dtype=np.float32)
    result[:, :resized_width] = cv2.resize(image, (resized_width, height))
    return result


def expanded_quad(points, width, height, margin=2.0):
    p = np.asarray(points, dtype=np.float32)
    u, v = p[1] - p[0], p[3] - p[0]
    u = u / max(1, np.linalg.norm(u)) * margin
    v = v / max(1, np.linalg.norm(v)) * margin
    p = p + np.asarray([-u-v, u-v, u+v, -u+v])
    p[:, 0] = np.clip(p[:, 0], 0, width-1)
    p[:, 1] = np.clip(p[:, 1], 0, height-1)
    return p

DATE = re.compile(r'\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|20\d{2}[./-]\d{1,2}[./-]\d{1,2}|\d{1,2}:\d{2}(?::\d{2})?)\b')


def restore_points(points, rotation, width, height):
    p = np.asarray(points, dtype=float)
    x, y = p[:, 0], p[:, 1]
    if rotation == 90:  # image was rotated clockwise
        return np.stack([y, height - 1 - x], axis=1).tolist()
    if rotation == 180:
        return np.stack([width - 1 - x, height - 1 - y], axis=1).tolist()
    if rotation == 270:
        return np.stack([width - 1 - y, x], axis=1).tolist()
    return p.tolist()


class PaddleBackend:
    def __init__(self, settings):
        self.settings = settings
        self.recognizer = None
        from paddleocr import PaddleOCR
        kwargs = dict(lang=settings['lang'], ocr_version=settings['ocr_version'],
                      device=settings['device'], use_doc_orientation_classify=False,
                      enable_mkldnn=settings.get('enable_mkldnn', False),
                      cpu_threads=settings.get('cpu_threads', 4),
                      use_doc_unwarping=False, use_textline_orientation=False,
                      text_det_limit_side_len=settings['det_limit_side_len'], text_det_limit_type='max')
        for key in ('text_recognition_model_dir', 'text_recognition_model_name',
                    'text_detection_model_dir', 'text_detection_model_name'):
            if settings.get(key):
                kwargs[key] = settings[key]
        self.model = PaddleOCR(**kwargs)

    def predict(self, image):
        options = {key: self.settings[key] for key in
                   ('text_det_thresh', 'text_det_box_thresh', 'text_det_unclip_ratio')
                   if key in self.settings}
        res = next(iter(self.model.predict(image, **options)))
        return [dict(points=np.asarray(p).tolist(), text=str(t), confidence=float(s))
                for p, t, s in zip(res['rec_polys'], res['rec_texts'], res['rec_scores'])]

    def refine(self, source, items):
        if not items:
            return []
        if self.recognizer is None:
            from paddleocr import TextRecognition
            self.recognizer = TextRecognition(
                model_name=self.settings['text_recognition_model_name'],
                model_dir=self.settings['text_recognition_model_dir'],
                device=self.settings['device'], enable_mkldnn=self.settings.get('enable_mkldnn', False),
                input_shape=[3, 48, 480])
        jobs, crops = [], []
        for i, item in enumerate(items):
            p = expanded_quad(item['points'], source.shape[1], source.shape[0],
                              self.settings.get('crop_margin', 2))
            crop = crop_quad(source, p)
            # PaddleOCR also turns tall boxes counterclockwise before reading.
            if crop.shape[0] >= 1.5 * crop.shape[1]:
                crop = np.ascontiguousarray(np.rot90(crop))
                p = p[[1, 2, 3, 0]]
            for flip in ((0, 180) if self.settings.get('crop_try_reverse', False) else (0,)):
                view = np.ascontiguousarray(np.rot90(crop, 2)) if flip else crop
                points = p[[2, 3, 0, 1]] if flip else p
                jobs.append((i, points))
                crops.append(recognition_canvas(view))
        answers = []
        for first in range(0, len(crops), 8):
            answers.extend(self.recognizer.predict(crops[first:first+8], batch_size=8))
        assert len(answers) == len(jobs)
        best = {}
        for (i, points), answer in zip(jobs, answers):
            item = dict(points=points.tolist(), text=str(answer['rec_text']), confidence=float(answer['rec_score']))
            if i not in best or item['confidence'] > best[i]['confidence']:
                best[i] = item
        return [best[i] for i in range(len(items))]

    def close(self):
        if self.recognizer is not None:
            self.recognizer.close()
        self.model.close()


class Pipeline:
    def __init__(self, settings=None, backend=None):
        self.settings = settings or config()
        self.backend = backend
        self.lock = threading.Lock()

    def process(self, path):
        start = time.perf_counter()
        try:
            if Path(path).suffix.lower() not in EXTENSIONS:
                raise ValueError('Поддерживаются только jpg/jpeg/png')
            image = read_image(path)
            with self.lock:
                if self.backend is None:
                    self.backend = PaddleBackend(self.settings)
                result = self._process(image)
            result['error'] = ''
        except Exception as exc:
            logging.exception('OCR failed: %s', path)
            result = dict(text='', heat='', confidence=0., rotation=0,
                          status='error', items=[], error=str(exc))
        result.update(filename=Path(path).name, elapsed_seconds=time.perf_counter() - start)
        return result

    def _process(self, image):
        views = self.settings.get('views')
        if not views:
            return self._process_single(image)
        backend = ViewBackend(self.backend)
        results = []
        previous = getattr(self.backend, 'settings', None)
        try:
            for index, overrides in enumerate(views):
                settings = dict(self.settings, **overrides, debug_candidates=True)
                settings.pop('views', None)
                backend.settings = settings
                result = Pipeline(settings, backend)._process_single(image)
                results.append((overrides.get('name', f'view_{index}'), result))
        finally:
            if previous is not None:
                self.backend.settings = previous
        result = fuse_views(results, self.settings['ok_score'])
        if self.settings.get('debug_candidates'):
            result['view_results'] = {name: value for name, value in results}
        return result

    def _process_single(self, image):
        h, w = image.shape[:2]
        image = image.copy()
        zone = self.settings.get('mask_zone')
        if zone:
            x0, y0, x1, y1 = [int(v * n) for v, n in zip(zone, (w, h, w, h))]
            image[y0:y1, x0:x1] = np.median(image.reshape(-1, 3), axis=0)
        source = image.copy()
        image = prepare_view(image, self.settings.get('preprocess_mode', 'clahe'), self.settings['clahe_clip'])
        best = None
        rotation_candidates = []
        for rotation in self.settings['rotations']:
            rotated = np.ascontiguousarray(np.rot90(image, -rotation // 90))
            original = np.ascontiguousarray(np.rot90(source, -rotation // 90))
            rh, rw = rotated.shape[:2]
            items = []
            detected = self.backend.predict(rotated)
            if self.settings.get('refine_crops', False):
                detected = self.backend.refine(original, detected)
            for item in detected:
                if item['confidence'] < self.settings['min_score']:
                    continue
                text = normalize(item['text'])
                pts = np.asarray(item['points'])
                cx, cy = pts.mean(axis=0)
                corner = (min(cx / rw, 1-cx / rw) < self.settings['date_corner_fraction'] and
                          min(cy / rh, 1-cy / rh) < self.settings['date_corner_fraction'])
                if DATE.search(text) and corner:
                    continue
                # Any remaining date-like token is removed from output too.
                text = normalize(DATE.sub('', text))
                if not text:
                    continue
                items.append(dict(text=text, confidence=item['confidence'],
                                  points=restore_points(pts, rotation, w, h),
                                  _x=float(pts[:, 0].min()), _y=float(pts[:, 1].mean()),
                                  _read_rotation=reading_rotation(restore_points(pts, rotation, w, h)),
                                  _horizontal=bool(np.ptp(pts[:,0]) >= np.ptp(pts[:,1]))))
            # Group boxes into lines, then sort each line left to right.
            lines = []
            for it in sorted(items, key=lambda it: it['_y']):
                pts = np.asarray(it['points'])
                # rotated coordinates are needed for the line height
                tolerance = max(8, (np.ptp(pts[:, 1]) if rotation in (0, 180) else np.ptp(pts[:, 0])) * .5)
                if not lines or abs(it['_y'] - lines[-1][0]['_y']) > tolerance:
                    lines.append([])
                lines[-1].append(it)
            items = [it for line in lines for it in sorted(line, key=lambda it: it['_x'])]
            # Paddle rotates tall crops internally: text length alone can prefer a
            # sideways page. Score horizontal evidence and a valid marking format.
            text = '\n'.join(it['text'] for it in items)
            parsed = extract_heat(text)
            evidence = sum(it['confidence'] * min(len(it['text']), 40) *
                           (1 if it['_horizontal'] else .15) for it in items)
            horizontal_heat = any(it['_horizontal'] and extract_heat(it['text'])['heat_id'] for it in items)
            score = evidence + (25 if horizontal_heat else 0)
            if self.settings.get('orientation_mode') == 'heat':
                heat_confidence = max((it['confidence'] for it in items
                    if extract_heat(it['text'])['heat_id'] == parsed['heat_id'] and parsed['heat_id']), default=0)
                score = evidence + 60 * heat_confidence
            rotation_candidates.append(dict(rotation=rotation, score=score,
                heat=parsed['heat'], horizontal_heat=horizontal_heat))
            if self.settings.get('debug_candidates'):
                rotation_candidates[-1].update(text=text, items=[dict(it) for it in items], **parsed)
            if best is None or score > best[0]:
                best = (score, rotation, items, parsed)
        _, rotation, items, parsed = best
        if self.settings.get('orientation_mode') == 'heat' and parsed['heat_id']:
            heat_items = [it for it in items if extract_heat(it['text'])['heat_id'] == parsed['heat_id']]
            if heat_items:
                rotation = max(heat_items, key=lambda it: it['confidence'])['_read_rotation']
        for item in items:
            item.pop('_x'); item.pop('_y'); item.pop('_horizontal')
            item.pop('_read_rotation')
        text = '\n'.join(it['text'] for it in items)
        confidence = sum(it['confidence'] * len(it['text']) for it in items) / max(1, sum(len(it['text']) for it in items))
        status = 'not_found' if not items else ('ok' if confidence >= self.settings['ok_score'] else 'low_confidence')
        return dict(text=text, **parsed, confidence=confidence,
                    rotation=rotation, status=status, items=items, rotation_candidates=rotation_candidates)


class ViewBackend:
    """Reuse detections within one image; do not retain photos between requests."""
    def __init__(self, core):
        self.core = core
        self.settings = {}
        self.cache = {}

    def key(self, mode, image, options):
        return (mode, image.shape, image.dtype.str, hashlib.sha256(image.tobytes()).digest(), options)

    def predict(self, image):
        options = tuple((k, self.settings.get(k)) for k in
                        ('text_det_thresh', 'text_det_box_thresh', 'text_det_unclip_ratio'))
        key = self.key('detect', image, options)
        if key not in self.cache:
            self.core.settings = self.settings
            self.cache[key] = self.core.predict(image)
        return self.cache[key]

    def refine(self, image, items):
        options = (self.settings.get('crop_margin', 2), self.settings.get('crop_try_reverse', False),
                   tuple(tuple(tuple(p) for p in item['points']) for item in items))
        key = self.key('refine', image, options)
        if key not in self.cache:
            self.core.settings = self.settings
            self.cache[key] = self.core.refine(image, items)
        return self.cache[key]
