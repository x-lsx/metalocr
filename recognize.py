import argparse
import json
import logging
import os
from pathlib import Path

from metalocr.common import EXTENSIONS, config, write_csv
from metalocr.full_pipeline import Pipeline
from metalocr.output import CSV_FIELDS, csv_record


def main():
    parser = argparse.ArgumentParser(description='OCR одного изображения или каталога')
    parser.add_argument('input', type=Path)
    parser.add_argument('--out', type=Path, default=Path('data/results'))
    parser.add_argument('--config', default='config.final.gpu.json')
    parser.add_argument('--labels', type=Path, help='CSV выборки; обрабатывать только её файлы из input-каталога')
    args = parser.parse_args()
    os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
    logging.basicConfig(level=logging.INFO)
    files = sorted(p for p in args.input.iterdir() if p.suffix.lower() in EXTENSIONS) if args.input.is_dir() else [args.input]
    if args.labels:
        from metalocr.common import read_csv
        if not args.input.is_dir():
            parser.error('--labels требует каталог в input')
        files = [(args.input / r['filename']).resolve() for r in read_csv(args.labels)]
        if any(p.parent != args.input.resolve() for p in files):
            parser.error('CSV содержит путь вне input')
    if not files:
        parser.error('Нет изображений для обработки')
    args.out.mkdir(parents=True, exist_ok=True)
    pipeline, results = Pipeline(config(args.config)), []
    for path in files:
        result = pipeline.process(path)
        results.append(result)
        (args.out / (path.name + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        logging.info('%s: %s %.2fs', path.name, result['status'], result['elapsed_seconds'])
    write_csv(args.out / 'results.csv', [csv_record(result) for result in results], CSV_FIELDS)


if __name__ == '__main__':
    main()
