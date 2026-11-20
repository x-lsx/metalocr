import csv
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

import cv2
import numpy as np

from metalocr.common import config
from metalocr.filename import compare_filename, filename_hints
from metalocr.output import CSV_FIELDS, csv_record
from metalocr.web_server import MAX_BYTES, make_handler, process_upload


class FakePipeline:
    def __init__(self):
        self.calls = []

    def process(self, path):
        self.calls.append(path)
        return dict(filename=path.name, text='V30593 1282 27', heat_id='V30593', heat_year=None,
                    sheet_id='27', sheet_status='ok', rotation=0, status='ok', elapsed_seconds=.02,
                    items=[dict(text='V30593 1282 27', confidence=.95,
                                points=[[2,2],[60,2],[60,20],[2,20]])])


class AppTests(unittest.TestCase):
    def setUp(self):
        # A named workspace directory also works in restricted Windows runners.
        root = Path(__file__).resolve().parents[1]
        scratch = root/'results/test_tmp'
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.directory = Path(self.temp.name).resolve()
        assert self.directory.is_relative_to(scratch.resolve())
        self.payload = cv2.imencode('.jpg', np.full((64,80,3), 180, np.uint8))[1].tobytes()

    def tearDown(self):
        self.temp.cleanup()

    def test_filename_hints_and_dashed_sheet_equivalence(self):
        name='Труба №19000 Плавка №2023-В203536 Лист №191202000.jpg'
        hints=filename_hints(name)
        self.assertEqual(hints,dict(heat_id='В203536',heat_year=2023,sheet_id='191202000'))
        comparison=compare_filename(dict(heat_id='В203536',heat_year=2023,sheet_id='19120-2000'),name)
        self.assertTrue(all(v['status']=='match' for v in comparison['fields'].values()))

    def test_missing_name_and_missing_ocr_are_distinct(self):
        data=compare_filename(dict(heat_id='V30593',sheet_id='27'),'photo.jpg')
        self.assertTrue(all(v['status']=='unavailable' for v in data['fields'].values()))
        data=compare_filename(dict(heat_id='',sheet_id=''),'Плавка №2023-V30593 Лист №27.png')
        self.assertEqual(data['fields']['heat_id']['status'],'not_found')
        self.assertEqual(data['fields']['sheet_id']['status'],'not_found')

    def test_csv_has_exact_five_columns_and_preserves_text(self):
        self.assertEqual(CSV_FIELDS,['file_name','text','heat_id','heat_year','sheet_id'])
        row=csv_record(dict(filename='a.jpg', text='"quoted", line\nnext',heat_id='V30593',sheet_id='027',heat_year=None))
        self.assertEqual(list(row),CSV_FIELDS)
        self.assertEqual(row['sheet_id'],'027')
        self.assertEqual(row['heat_year'],'')

    def test_upload_creates_crops_and_does_not_use_filename_for_inference(self):
        pipeline=FakePipeline()
        name='Плавка №2022-V30594 Лист №99.jpg'
        result=process_upload(self.payload,name,pipeline,self.directory)
        self.assertEqual(result['heat_id'],'V30593')
        self.assertEqual(result['sheet_id'],'27')
        self.assertEqual(pipeline.calls[0].name,'input.jpg')
        self.assertEqual(result['comparison']['fields']['heat_id']['status'],'mismatch')
        self.assertEqual(result['comparison']['fields']['sheet_id']['status'],'mismatch')
        job=self.directory/result['job_id']
        self.assertTrue((job/'crops/0.jpg').read_bytes().startswith(b'\xff\xd8'))
        with (job/'results.csv').open(encoding='utf-8-sig',newline='') as stream:
            rows=list(csv.DictReader(stream))
        self.assertEqual(list(rows[0]),CSV_FIELDS)
        self.assertEqual(rows[0]['file_name'],name)

    def test_invalid_uploads_do_not_call_model(self):
        pipeline=FakePipeline()
        for payload,name in [(b'broken','x.jpg'),(self.payload,'x.txt'),(b'\xff\xd8broken','x.jpg')]:
            with self.subTest(name=name,payload=payload[:8]):
                with self.assertRaises(ValueError):
                    process_upload(payload,name,pipeline,self.directory)
        self.assertFalse(pipeline.calls)

    def test_http_upload_assets_download_and_origin(self):
        pipeline=FakePipeline()
        server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(config(),self.directory,pipeline))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base='http://127.0.0.1:'+str(server.server_port)
        try:
            for route in ['/','/app.js','/style.css','/api/info']:
                with urlopen(base+route) as response:
                    self.assertEqual(response.status,200)
                    self.assertTrue(response.read())
            headers={'X-Metal-Upload':'1','X-File-Name':quote('Плавка №2023-V30593 Лист №27.jpg'),
                     'Content-Type':'application/octet-stream','Origin':base}
            with urlopen(Request(base+'/api/recognize',data=self.payload,headers=headers)) as response:
                result=json.load(response)
            for route in [result['image_url'],result['regions'][0]['crop_url'],result['csv_url'],result['json_url']]:
                with urlopen(base+route) as response:
                    self.assertEqual(response.status,200)
                    self.assertTrue(response.read())
            with urlopen(base+result['csv_url']) as response:
                rows=list(csv.reader(io.StringIO(response.read().decode('utf-8-sig'))))
                self.assertEqual(rows[0],CSV_FIELDS)
            for route in ['/../config.json','/api/jobs/'+result['job_id']+'/../config.json']:
                with self.assertRaises(HTTPError) as error: urlopen(base+route)
                self.assertEqual(error.exception.code,404)
            for changed,expected in [({'Origin':'https://example.org'},403),({'X-Metal-Upload':'0'},403),
                                      ({'Content-Length':str(MAX_BYTES+1)},413)]:
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(base+'/api/recognize',data=self.payload,headers=dict(headers,**changed)))
                self.assertEqual(error.exception.code,expected)
            self.assertEqual(len(pipeline.calls),1)
        finally:
            server.shutdown();server.server_close();thread.join()


if __name__=='__main__':
    unittest.main()
