"""Select OCR evidence across image views without a vocabulary of known heats."""
import copy
import math

import numpy as np

from .common import normalize
from .heat import extract_heat


def reading_rotation(points):
    """Map the reading edge of an automatically rectified box to page rotation."""
    p = np.asarray(points, dtype=float)
    width = max(np.linalg.norm(p[1]-p[0]), np.linalg.norm(p[2]-p[3]))
    height = max(np.linalg.norm(p[3]-p[0]), np.linalg.norm(p[2]-p[1]))
    edge = p[2]-p[1] if height >= 1.5*width else p[1]-p[0]
    return int(round(-math.degrees(math.atan2(edge[1], edge[0]))/90)*90) % 360


def fuse_views(views, ok_score=.8):
    """One vote per view and identifier; repeated rotations are correlated."""
    candidates = {}
    for name, result in views:
        for page in result.get('rotation_candidates', []):
            for item in page.get('items', []):
                parsed = extract_heat(item['text'])
                identifier = parsed['heat_id']
                if not identifier or item['confidence'] < .65:
                    continue
                record = dict(view=name, page=page, item=item, parsed=parsed, result=result)
                previous = candidates.setdefault(identifier, {}).get(name)
                if previous is None or item['confidence'] > previous['item']['confidence']:
                    candidates[identifier][name] = record
    if not candidates:
        return copy.deepcopy(views[0][1])
    ranked = []
    for identifier, votes in candidates.items():
        records = list(votes.values())
        best = max(records, key=lambda r: r['item']['confidence'])
        score = best['item']['confidence'] + .07*min(len(votes)-1, 2)
        ranked.append((score, identifier, best, records))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    score, identifier, best, records = ranked[0]
    items = [{k: copy.deepcopy(v) for k, v in item.items() if not k.startswith('_')}
             for item in best['page']['items']]
    text = '\n'.join(normalize(item['text']) for item in items)
    parsed = extract_heat(text)
    # Keep the parser's ambiguity guard for multiple actual text regions.
    uncertain = (len(ranked) > 1 and (
        abs(score-ranked[1][0]) < 1e-6 or
        (len(records) > 1 and len(ranked[1][3]) > 1 and score-ranked[1][0] < .015)))
    if uncertain:
        parsed.update(heat='', heat_id='', heat_year=None, heat_year_source='unknown',
                      heat_raw='', heat_suffix='', heat_corrections=[], heat_ambiguous=True)
    confidence = sum(item['confidence']*len(item['text']) for item in items) / max(1, sum(len(item['text']) for item in items))
    return dict(text=text, **parsed, confidence=confidence,
                rotation=reading_rotation(best['item']['points']),
                status='ok' if confidence >= ok_score else 'low_confidence', items=items,
                rotation_candidates=[dict(rotation=page['rotation'], score=page['score'],
                                          heat=page['heat'], horizontal_heat=page['horizontal_heat'])
                                     for page in best['result'].get('rotation_candidates', [])],
                heat_selection=dict(method='view_consensus', candidate=identifier, support_views=[r['view'] for r in records],
                                    conflicting_ids=[r[1] for r in ranked[1:]], uncertain=uncertain))
