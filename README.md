# ML Segmentation & Skeletonization Project

## 1. Структура проекта и зоны ответственности

```
segmentator/
├── ml/
│   ├
│   ├── biomarcers/                # Обучение SegFormer
│   ├   ├── train_segformer_hdd.py
│   ├   ├── model.py
│   ├   ├── dataset.py
│   ├   ├── loss.py
│   ├   └── config.py
│   ├
│   ├── segmentator/               # Основной ML-модуль (segmentation + skeleton)
│   │   ├── model_segmentation.py
│   │   ├── model_skeleton.py
│   │   ├── training_segmentation.py
│   │   ├── training_skeleton.py
│   │   ├── testing_segmentation.py
│   │   ├── testing_skeleton.py
│   │   ├── inference.py
│   │   ├── inference_core.py
│   │   ├── calc_metrics.py
│   │   ├── CI_metrics_segmentation.py
│   │   ├── CI_metrics_skeleton.py
│   │   └── test/                  # Unit-тесты
│   │
│   │── service/                   # Backend сервис для инференса
│   │   └── backend/
│   │       ├── main.py
│   │       ├── app.py
│   │       └── inference_core.py
│
├── Dockerfile
├── docker-compose.yml
└── requirements.txt (для сервиса)

```

---

# 2. Назначение модулей

##  `biomarcers/`
Модуль обучения SegFormer.

Отвечает за:
- загрузку датасета
- конфигурацию обучения
- определение loss-функций
- обучение модели SegFormer
- сохранение чекпоинтов

Используется для экспериментов и обучения.

---

##  `segmentator/`

###  model_segmentation.py
Реализация архитектуры сегментационной модели.

###  model_skeleton.py
Модель для предсказания скелета (skeletonization-aware).

###  training_segmentation.py
Обучение модели сегментации.

###  training_skeleton.py
Обучение модели скелетизации.

###  testing_*.py
Запуск оценки модели на валидации/тесте.

###  calc_metrics.py
Подсчет метрик:
- Dice
- clDice
- вспомогательные метрики

###  CI_metrics_*.py
Расчет доверительных интервалов (confidence intervals).

###  inference.py / inference_core.py
Логика инференса:
- загрузка весов
- preprocessing
- forward pass
- postprocessing

---

##  `service/backend/`

Backend для инференса.

- `main.py` — точка входа
- `app.py` — создание web-приложения
- `inference_core.py` — загрузка модели и инференс

Предназначен для деплоя модели как сервиса.

---

# 3. Используемые архитектуры

## SegFormer

Используется в `biomarcers/`.

- Transformer-based encoder
- Lightweight decoder
- Подходит для dense prediction
- Хорошо работает на медицинских и структурных изображениях

Преимущества:
- Глобальный контекст
- Лучшая обобщающая способность

---

## CNN-based Segmentation Model

В `segmentator/model_segmentation.py`.

Тип:
- Transformer-based encoder (SegFormer)
- Lightweight convolutional decoder
- Fully convolutional, без классических skip connections

Назначение:
- Бинарная сегментация объектов

---

## Skeleton Model

В `model_skeleton.py`.

Тип:
- Transformer-based encoder (SegFormer)
- Lightweight convolutional decoder
- Дифференцируемая морфология для выделения скелета (soft skeletonization)

Назначение:
- Предсказание центральной линии объектов
- Используется для улучшения топологической корректности

---

# 4. Параметры обучения

Используются:

- Optimizer: Adam / AdamW
- Learning rate: задаётся в config.py
- Batch size: конфигурируемый
- Epochs: задаётся вручную
- Loss functions:
  - Dice Loss
  - BCE / BCEWithLogits - сегментация сосудов
  - CE + Tversky - биомаркеры
  - clDice (для skeleton-aware оптимизации)

Поддерживается:
- k-fold cross validation
- сохранение лучших чекпоинтов

---

# 5. Метрики (сегментация сосудов)

## Dice coefficient

Формула:

Dice = 2TP / (2TP + FP + FN)

Зачем:
- Основная метрика для сегментации
- Устойчива к дисбалансу классов

---

## clDice

Topology-aware метрика.

Зачем:
- Оценивает корректность скелета
- Важна для тонких структур (сосуды, дороги и т.д.)
- Наказывает за разрывы

---

## Confidence Intervals (CI)

Рассчитываются в:
- CI_metrics_segmentation.py
- CI_metrics_skeleton.py

Зачем:
- Оценка статистической устойчивости модели
- Позволяет сравнивать модели корректно
- Полезно для научных публикаций

---

# 6. Тесты

Расположены в:

```

segmentator/test/

```

## Unit-тесты моделей
Проверяют:
- корректность forward pass
- размерности выходов
- работу без ошибок


---


## Тесты dataloader
Проверяют:
- корректную загрузку данных
- соответствие размерностей
- отсутствие падений на edge cases

---

# 7. Как запускать

## Обучение скелета

```

python segmentator/training_skeleton.py

```

## Обучение сегментации

```

python segmentator/training_segmentation.py

```

---
## Обучение биомаркеров
```
python biomarcers/train_segformer_hdd.py
```
---

## Тестирование

```

python segmentator/testing_segmentation.py
python segmentator/testing_skeleton.py

```

---

## Запуск тестов

```

pytest segmentator/test/

```

---

##  Запуск сервиса

```
uvicorn service.backend.main:app
streamlit run service.backend.app

```

или через Docker:

```

docker-compose up --build

```
 
Запускается на порту http://127.0.0.1:8501/

---

# 8. Docker

Проект контейнеризирован:

- `Dockerfile` — сборка окружения
- `docker-compose.yml` — запуск сервиса

Позволяет:
- воспроизводимо запускать модель
---