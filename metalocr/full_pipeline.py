import time

from .pipeline import Pipeline as BasePipeline
from .sheet import extract_sheet


class Pipeline(BasePipeline):
    def process(self, path):
        started = time.perf_counter()
        result = super().process(path)
        fields = extract_sheet(result.get('text', ''), result.get('heat_id'),
                               result.get('items', []), self.settings['ok_score'])
        if result['status'] == 'error' or not result.get('heat_id'):
            fields.update(sheet_id='', sheet_status='not_found', sheet_source='unknown',
                          sheet_confidence=None, sheet_raw='', sheet_candidates=[])
        result.update(fields)
        result['elapsed_seconds'] = time.perf_counter()-started
        return result
