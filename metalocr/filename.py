
import re
from pathlib import PurePosixPath

from .heat import extract_heat

SHEET = re.compile(r'(?:лист|sheet)[\s_]*(?:[№#]\s*)?(\d+(?:[-–—]\d+)*)(?!\d)', re.IGNORECASE)


def filename_hints(file_name):
    name = PurePosixPath(file_name.replace('\\', '/')).name
    parsed = extract_heat(name)
    matches = {m[1].replace('–', '-').replace('—', '-') for m in SHEET.finditer(name)}
    return dict(heat_id=parsed['heat_id'], heat_year=parsed['heat_year'],
                sheet_id=next(iter(matches)) if len(matches) == 1 else '')


def compare_filename(result, file_name):
    hints = filename_hints(file_name)
    fields = {}
    for key in ['heat_id', 'heat_year', 'sheet_id']:
        actual, expected = result.get(key), hints[key]
        if expected in ('', None):
            status = 'unavailable'
        elif actual in ('', None):
            status = 'not_found'
        else:
            clean = lambda value: re.sub(r'[-–—\s]', '', str(value)).upper()
            status = 'match' if clean(actual) == clean(expected) else 'mismatch'
        fields[key] = dict(ocr=actual, filename=expected, status=status)
    return dict(hints=hints, fields=fields)
