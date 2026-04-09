# OCR service with separate container endpoint

## Что внутри
- `/recognize` — старый эндпоинт под пломбы
- `/recognize-container` — новый эндпоинт под номера контейнеров
- `check_folder.py` — быстрый локальный прогон папки с фото без запуска API

## Куда положить модели
Папки моделей положи сюда:

- `models/paddleocr/PP-OCRv5_server_det`
- `models/paddleocr/en_PP-OCRv5_mobile_rec`
- `models/paddleocr/PP-LCNet_x1_0_textline_ori`
- `models/YOLO/best.pt` — только если нужен старый эндпоинт под пломбы

Если модели лежат в другом месте, можно указать переменные окружения:
- `PADDLE_DET_MODEL_DIR`
- `PADDLE_REC_MODEL_DIR`
- `PADDLE_TEXTLINE_ORI_MODEL_DIR`

## Самый быстрый запуск на Windows
### Вариант 1. Быстро проверить папку с фото контейнеров
```bat
check_folder.bat C:\path\to\container_photos
```

### Вариант 2. Поднять API
```bat
run_api.bat
```
После запуска открой:
- `http://127.0.0.1:8000/docs`

В Swagger используй эндпоинт `POST /recognize-container`.

## Что возвращает новый контейнерный эндпоинт
- `result` — найденный номер контейнера
- `score` — уверенность внутреннего ранжирования
- `is_valid_iso6346` — прошла ли проверка контрольной цифры ISO 6346

## Важно
В архиве нет весов моделей. Код рабочий, но для фактического распознавания нужно положить свои модели в папку `models`.
