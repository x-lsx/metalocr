import argparse
import json
import logging
import mimetypes
import os
import re
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

import cv2
import numpy as np

from .annotations import crop_quad
from .common import ROOT, config, read_image, write_csv
from .filename import compare_filename
from .full_pipeline import Pipeline
from .output import CSV_FIELDS, csv_record

MAX_BYTES = 40 * 1024 * 1024
MAX_PIXELS = 25_000_000


def encode_jpg(path, image):
    ok, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        raise ValueError('Не удалось создать изображение результата')
    path.write_bytes(data.tobytes())


def process_upload(payload, file_name, pipeline, output):
    file_name = file_name.replace('\\', '/').rsplit('/', 1)[-1]
    if not file_name or len(file_name) > 255 or any(ord(c) < 32 for c in file_name):
        raise ValueError('Некорректное имя файла')
    if Path(file_name).suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
        raise ValueError('Поддерживаются JPG, JPEG и PNG')
    if not payload or len(payload) > MAX_BYTES:
        raise ValueError('Выберите изображение размером до 40 МБ')
    if not (payload.startswith(b'\xff\xd8') or payload.startswith(b'\x89PNG\r\n\x1a\n')):
        raise ValueError('Файл не является изображением JPG или PNG')
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Изображение повреждено или не поддерживается')
    height, width = image.shape[:2]
    if height * width > MAX_PIXELS:
        raise ValueError('Изображение слишком большое: максимум 25 мегапикселей')
    job_id = uuid.uuid4().hex
    job = Path(output)/job_id
    (job/'crops').mkdir(parents=True)
    suffix = '.png' if payload.startswith(b'\x89PNG') else '.jpg'
    source = job/('input'+suffix)
    source.write_bytes(payload)
    result = pipeline.process(source)
    # The name is applied after inference, including any sheet extraction.
    result.update(filename=file_name, file_name=file_name)
    encode_jpg(job/'original.jpg', image)
    prefix = '/api/jobs/'+job_id
    regions = []
    for index, item in enumerate(result.get('items', [])):
        crop = crop_quad(image, item['points'])
        if crop.shape[0] >= 1.5*crop.shape[1]:
            crop = np.ascontiguousarray(np.rot90(crop))
        if max(crop.shape[:2]) > 2048:
            factor = 2048/max(crop.shape[:2])
            crop = cv2.resize(crop, (max(1, round(crop.shape[1]*factor)), max(1, round(crop.shape[0]*factor))))
        encode_jpg(job/'crops'/f'{index}.jpg', crop)
        regions.append(dict(item, index=index+1, crop_url=prefix+f'/crops/{index}.jpg'))
    result.update(job_id=job_id, image_url=prefix+'/image', image_width=width, image_height=height,
                  regions=regions, comparison=compare_filename(result, file_name),
                  csv_url=prefix+'/results.csv', json_url=prefix+'/result.json')
    write_csv(job/'results.csv', [csv_record(result, file_name)], CSV_FIELDS)
    (job/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def make_handler(settings, output, pipeline=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    pipeline = pipeline if pipeline is not None else Pipeline(settings)
    assets = ROOT/'web'

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, data, content_type='application/json; charset=utf-8', download=False):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if download:
                self.send_header('Content-Disposition', 'attachment; filename="results.csv"')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/api/info':
                return self.send(200, dict(device=settings['device'], max_bytes=MAX_BYTES,
                                           model='eslav PP-OCRv5 · 21-я эпоха'))
            static = {'/':'index.html', '/app.js':'app.js', '/style.css':'style.css'}
            if path in static:
                file = assets/static[path]
            else:
                match = re.fullmatch(r'/api/jobs/([a-f0-9]{32})/(image|result\.json|results\.csv|crops/\d+\.jpg)', path)
                if not match:
                    return self.send(404, dict(error='Не найдено'))
                file = output/match[1]/('original.jpg' if match[2] == 'image' else match[2])
            if not file.is_file():
                return self.send(404, dict(error='Результат не найден'))
            mime = mimetypes.guess_type(file.name)[0] or 'application/octet-stream'
            if file.suffix in {'.html','.js','.css','.csv','.json'}:
                mime += '; charset=utf-8'
            self.send(200, file.read_bytes(), mime, download=file.name == 'results.csv')

        def do_POST(self):
            if urlsplit(self.path).path != '/api/recognize':
                return self.send(404, dict(error='Не найдено'))
            host = self.headers.get('Host', '')
            origin = self.headers.get('Origin')
            if host.split(':')[0] not in {'localhost','127.0.0.1'} or (origin and origin != 'http://'+host):
                return self.send(403, dict(error='Недопустимый источник запроса'))
            if self.headers.get('X-Metal-Upload') != '1':
                return self.send(403, dict(error='Используйте форму загрузки'))
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if length < 1 or length > MAX_BYTES:
                    return self.send(413, dict(error='Допустимый размер файла: от 1 байта до 40 МБ'))
                payload = self.rfile.read(length)
                if len(payload) != length:
                    raise ValueError('Файл загрузился не полностью')
                file_name = unquote(self.headers.get('X-File-Name', ''), encoding='utf-8', errors='strict')
                result = process_upload(payload, file_name, pipeline, output)
                return self.send(503 if result['status'] == 'error' else 200, result)
            except (ValueError, UnicodeError) as exc:
                self.send(400, dict(error=str(exc)))
            except Exception:
                logging.exception('Upload processing failed')
                self.send(500, dict(error='Не удалось обработать фотографию. Подробности в терминале.'))

    return Handler


def main():
    parser = argparse.ArgumentParser(description='Загрузка фото и просмотр OCR')
    parser.add_argument('--config', default=str(ROOT/'config.final.gpu.json'))
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--out', type=Path, default=ROOT/'results/web')
    args = parser.parse_args()
    os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
    logging.basicConfig(level=logging.INFO)
    settings = config(args.config)
    # Resolve local model directories even when started from another directory.
    for key in ['text_detection_model_dir', 'text_recognition_model_dir']:
        if settings.get(key) and not Path(settings[key]).is_absolute():
            settings[key] = str(ROOT/settings[key])
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(settings, args.out))
    print(f'METAL OCR: http://127.0.0.1:{args.port} · {settings["device"]}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
