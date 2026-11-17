# Bundled OCR models

These models are based on PaddleOCR, distributed under Apache License 2.0.
The license text is included in `models/LICENSE`.

- `PP-OCRv5_server_det`: unmodified official PaddleOCR text detector.
  Source: https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det
- `metal_rec_expand_20261117/inference`: eslav PP-OCRv5 mobile recognizer,
  modified by fine-tuning on 405 Train photographs / 1214 text crops.
  Selected checkpoint: epoch 21 of 30. Source architecture and original model:
  https://github.com/PaddlePaddle/PaddleOCR

All three inference files in each model directory are required together.
Training checkpoints, optimizer state and training photographs are not included.
