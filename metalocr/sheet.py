import math
import re

from .common import normalize
from .heat import extract_heat

DASHED = re.compile(r'^(\d{5})\s*[-–—]\s*(\d{4})$')
TAIL = re.compile(r'^\s+(\d{3,5})\s+(\d{1,3})\s*$')


def _rows(text, items):
    """Keep original OCR rows and join boxes only along a heat box's baseline."""
    rows = [(normalize(line), None) for line in text.splitlines() if normalize(line)]
    for item in items:
        value = normalize(item['text'])
        rows.append((value, float(item['confidence'])))
        if not extract_heat(value)['heat_id'] or not item.get('points'):
            continue
        p = item['points']
        length = lambda a, b: math.hypot(b[0]-a[0], b[1]-a[1])
        a, b = (p[1], p[2]) if length(p[0], p[3]) >= 1.5*length(p[0], p[1]) else (p[0], p[1])
        size = length(a, b)
        if size < 1:
            continue
        ux, uy = (b[0]-a[0])/size, (b[1]-a[1])/size
        project = lambda q: (q[0]*ux+q[1]*uy, -q[0]*uy+q[1]*ux)
        corners = [project(q) for q in p]
        baseline = sum(q[1] for q in corners)/4
        height = max(q[1] for q in corners)-min(q[1] for q in corners)
        aligned = []
        for other in items:
            if not other.get('points'):
                continue
            box = [project(q) for q in other['points']]
            center = sum(q[1] for q in box)/4
            other_height = max(q[1] for q in box)-min(q[1] for q in box)
            comparable = .4*max(height, 1) <= other_height <= 2.5*max(height, 1)
            if comparable and abs(center-baseline) <= .6*max(height, other_height, 1):
                aligned.append((min(q[0] for q in box), other))
        aligned.sort(key=lambda pair: pair[0])
        if aligned:
            rows.append((' '.join(normalize(o['text']) for _, o in aligned),
                         min(float(o['confidence']) for _, o in aligned)))
    return rows


def extract_sheet(text, heat_id=None, items=None, ok_score=.8):
    parsed = extract_heat(text)
    empty = dict(sheet_id='', sheet_status='not_found', sheet_source='unknown',
                 sheet_confidence=None, sheet_raw='', sheet_candidates=[])
    # A selected identifier must also be present unambiguously in OCR text.
    if parsed['heat_ambiguous'] or not parsed['heat_id']:
        return empty
    heat_id = heat_id or parsed['heat_id']
    if heat_id != parsed['heat_id']:
        return empty
    candidates = []
    for row, confidence in _rows(text, items or []):
        value = source = ''
        if heat_id.startswith(('V', 'Z')):
            heat = extract_heat(row)
            if heat['heat_id'] != heat_id:
                continue
            end = row.find(heat['heat_raw']) + len(heat['heat_raw'])
            tail = TAIL.fullmatch(row[end:])
            if tail:
                value, source = tail[2], 'heat_line_tail'
        elif heat_id.startswith('\u0412'):
            code = DASHED.fullmatch(row)
            if code:
                value, source = code[1]+'-'+code[2], 'standalone_dashed'
        if value:
            candidates.append(dict(sheet_id=value, source=source, raw=row, confidence=confidence))
    # Preserve separate conflicting readings; never silently pick a last number.
    unique = {}
    for candidate in candidates:
        key = (candidate['sheet_id'], candidate['raw'])
        previous = unique.get(key)
        if previous is None or (candidate['confidence'] or 0) > (previous['confidence'] or 0):
            unique[key] = candidate
    candidates = list(unique.values())
    empty['sheet_candidates'] = candidates
    if len({c['sheet_id'] for c in candidates}) > 1:
        empty['sheet_status'] = 'ambiguous'
        return empty
    if not candidates:
        return empty
    selected = max(candidates, key=lambda c: c['confidence'] if c['confidence'] is not None else -1)
    confidence = selected['confidence']
    return dict(sheet_id=selected['sheet_id'],
                sheet_status='low_confidence' if confidence is not None and confidence < ok_score else 'ok',
                sheet_source=selected['source'], sheet_confidence=confidence,
                sheet_raw=selected['raw'], sheet_candidates=candidates)
