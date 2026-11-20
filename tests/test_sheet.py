import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from metalocr.common import config
from metalocr.full_pipeline import Pipeline
from metalocr.sheet import extract_sheet


def box(text, x, y, width=70, confidence=.95):
    return dict(text=text, confidence=confidence,
                points=[[x,y],[x+width,y],[x+width,y+15],[x,y+15]])


class SheetTests(unittest.TestCase):
    def test_user_examples(self):
        examples = [
            ('12.00*3134*12040\n19120-2000\n2023-В203536 100\nММК 10Г2ФБИ', '19120-2000'),
            ('Z0226В490072 Z26849 1981 20\nK56-2 С.35P.18 В 2217 12020', '20'),
            ('C.41P.18 12 2209 12020\nV30593 1282 27\n8518 V9Td', '27'),
        ]
        for text, expected in examples:
            with self.subTest(expected=expected):
                self.assertEqual(extract_sheet(text)['sheet_id'], expected)

    def test_dimensions_years_and_incomplete_sequences_are_not_sheets(self):
        for text in ['V30593\n12 2209 12020', 'V30593 2023', 'V30593 27',
                     'V30593\n1282 27', '2023-В203536 100\n12.00*3134*12040',
                     '19120-2000', 'V30593 1282 27 extra', 'V30593 1282 27\nZ26849']:
            with self.subTest(text=text):
                self.assertEqual(extract_sheet(text)['sheet_id'], '')

    def test_conflicts_are_reported(self):
        for text in ['V30593 1282 27\nV30593 1282 28',
                     '2023-В203536\n19120-2000\n19121-2000']:
            result = extract_sheet(text)
            self.assertEqual(result['sheet_id'], '')
            self.assertEqual(result['sheet_status'], 'ambiguous')

    def test_split_boxes_need_geometric_alignment(self):
        items = [box('V30593', 0, 0), box('1282', 80, 0), box('27', 160, 0)]
        text = '\n'.join(i['text'] for i in items)
        result = extract_sheet(text, items=items)
        self.assertEqual(result['sheet_id'], '27')
        self.assertEqual(result['sheet_confidence'], .95)
        items[-1] = box('27', 160, 80)
        self.assertEqual(extract_sheet(text, items=items)['sheet_id'], '')

    def test_rotated_split_boxes(self):
        items = [box('V30593', 0, 0), box('1282', 80, 0), box('27', 160, 0)]
        for item in items:
            item['points'] = [[300-y, x] for x,y in item['points']]
        result = extract_sheet('\n'.join(i['text'] for i in items), items=items)
        self.assertEqual(result['sheet_id'], '27')

    def test_large_unrelated_box_is_not_joined_to_heat(self):
        heat = box('V30593', 0, 0)
        unrelated = dict(text='1282 27', confidence=.99,
                         points=[[80,40],[160,40],[160,440],[80,440]])
        result = extract_sheet('V30593\n1282 27', items=[heat, unrelated])
        self.assertEqual(result['sheet_id'], '')

    def test_low_confidence_and_leading_zero_are_preserved(self):
        item = box('V30593 1282 027', 0, 0, confidence=.6)
        result = extract_sheet(item['text'], items=[item])
        self.assertEqual(result['sheet_id'], '027')
        self.assertEqual(result['sheet_status'], 'low_confidence')

    def test_wrong_selected_heat_is_not_used(self):
        self.assertEqual(extract_sheet('V30593 1282 27', heat_id='V30594')['sheet_id'], '')

    def test_full_pipeline_and_error_path(self):
        class Backend:
            def predict(self, image):
                return [box('V30593 1282 27', 5, 5)]
        with patch('metalocr.pipeline.read_image', return_value=np.zeros((60, 200, 3), np.uint8)):
            path = Path('unrelated-filename-999.jpg')
            pipeline = Pipeline(dict(config(), rotations=[0], mask_zone=None), Backend())
            result = pipeline.process(path)
            self.assertEqual(result['heat_id'], 'V30593')
            self.assertEqual(result['sheet_id'], '27')
            self.assertEqual(result['status'], 'ok')
        with patch('metalocr.pipeline.read_image', side_effect=ValueError('broken image')):
            with self.assertLogs(level='ERROR'):
                missing = pipeline.process(Path('missing.jpg'))
            self.assertEqual(missing['status'], 'error')
            self.assertEqual(missing['sheet_id'], '')


if __name__ == '__main__':
    unittest.main()
