"""Extract visible heat identifiers without looking at filenames or registries."""
import re

# These are marking formats, not a vocabulary of known heats.
SUFFIXES = {'100', '200', '250'}
FULL = re.compile(r'(?<!\d)((?:19|20)\d{2})\s*[-–—]\s*([A-Za-zА-Яа-яЁё0-9]+)')
BARE = re.compile(r'(?<![A-Za-zА-Яа-яЁё0-9])([VZvzBВbв]\d+)(?![A-Za-zА-Яа-яЁё0-9])')


def _identifier(raw, printed_year=False):
    prefix, digits = raw[0].upper(), raw[1:]
    corrections = []
    if prefix == 'B':
        prefix = 'В'
        corrections.append('Latin B -> Cyrillic В in heat prefix')
    if prefix == '8' and printed_year:
        prefix = 'В'
        corrections.append('8 -> В in printed-year heat prefix')
    length = 5 if prefix in ('V', 'Z') else 6 if prefix == 'В' else None
    if length is None:
        return None
    suffix = ''
    if len(digits) == length + 3 and digits[length:] in SUFFIXES:
        digits, suffix = digits[:length], digits[length:]
    if len(digits) != length or not digits.isdigit():
        return None
    return dict(heat_id=prefix+digits, suffix=suffix, corrections=corrections)


def extract_heat(text):
    candidates = []
    spans = []
    for match in FULL.finditer(text):
        year, raw = match.groups()
        item = _identifier(raw, printed_year=True)
        if item is None:
            # Retain compatibility with other explicit letter+digit heat formats.
            if not re.fullmatch(r'[A-Za-zА-Яа-яЁё]\d+', raw):
                continue
            if raw[0].upper() in ('V', 'Z', 'B', 'В'):
                continue
            item = dict(heat_id=raw, suffix='', corrections=[])
        item.update(heat_year=int(year), raw=match.group())
        candidates.append(item)
        spans.append(match.span())
    for match in BARE.finditer(text):
        if any(start <= match.start() < end for start, end in spans):
            continue
        item = _identifier(match.group())
        if item:
            item.update(heat_year=None, raw=match.group())
            candidates.append(item)
    identifiers = {c['heat_id'] for c in candidates}
    years = {c['heat_year'] for c in candidates if c['heat_year'] is not None}
    ambiguous = len(identifiers) > 1 or len(years) > 1
    selected = max(candidates, key=lambda c:c['heat_year'] is not None) if candidates and not ambiguous else None
    return dict(heat=(f"{selected['heat_year']}-" if selected and selected['heat_year'] else '') +
                     (selected['heat_id'] if selected else ''),
                heat_id=selected['heat_id'] if selected else '',
                heat_year=selected['heat_year'] if selected else None,
                heat_year_source='ocr' if selected and selected['heat_year'] else 'unknown',
                heat_raw=selected['raw'] if selected else '',
                heat_suffix=selected['suffix'] if selected else '',
                heat_corrections=selected['corrections'] if selected else [],
                heat_ambiguous=ambiguous, heat_candidates=candidates)
