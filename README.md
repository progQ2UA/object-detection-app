# Система виявлення об'єктів (Object Detection System)

Десктопний застосунок на Python + Tkinter для виявлення об'єктів на зображеннях
(аерофотознімки, UAV, інфрачервоні/тепловізійні кадри) за допомогою моделі YOLOv8.

## Можливості

- Завантаження зображень (JPG, PNG, BMP, TIFF)
- Автоматичне завантаження моделі YOLOv8 (`models/best.pt`)
- Режим **нарізки (tiling)** для великих зображень (>1280 px) з NMS-об'єднанням
- Попередня обробка під тип знімка:
  - **Aerial / UAV** — підвищення різкості (sharpen kernel)
  - **Infrared / Thermal** — CLAHE-контрастування
- Налаштування порогів **Confidence** та **IoU** у реальному часі
- Візуалізація bounding box'ів з підписами та кольорами по класах
- Масштабування (колесо миші) та перетягування (drag) зображення
- Статистика виявлених об'єктів за класами

Класи за замовчуванням: `person`, `car`, `bicycle`, `other_vehicle`, `drone`, `mine`, `gun`.

## Вимоги

- Python **3.9+**
- ОС: Windows / Linux / macOS
- (рекомендовано) GPU з CUDA для швидшої детекції — але працює і на CPU

## Встановлення

### 1. Завантажити проєкт

**Варіант А — через Git:**
```bash
git clone https://github.com/<ваш-нік>/object-detection-app.git
cd object-detection-app
```

**Варіант Б — без Git:** на сторінці репозиторію натисніть **Code → Download ZIP**, розпакуйте архів.

### 2. Створити віртуальне середовище (рекомендовано)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 3. Встановити залежності

```bash
pip install -r requirements.txt
```

> Якщо потрібна підтримка CUDA — встановіть PyTorch з офіційного сайту:
> https://pytorch.org/get-started/locally/

## Модель

Застосунок **не потребує датасету** для роботи — потрібна лише файл моделі `.pt`.

### Варіант 1 — використати стандартну модель YOLOv8 (найшвидший старт)

Якщо у вас немає навченої `best.pt`, можна запустити з попередньо навченою `yolov8n.pt`:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

Після цього у папці з'явиться `yolov8n.pt`. Скопіюйте його в `models/` під іменем `best.pt`:

```bash
# Windows (PowerShell)
mkdir models
Copy-Item yolov8n.pt models\best.pt

# Linux / macOS
mkdir -p models
cp yolov8n.pt models/best.pt
```

Або просто перетягніть файл мишкою у провіднику.

> ⚠️ Стандартна `yolov8n.pt` знає лише 80 класів COCO (person, car, bicycle…), але **не знає** `drone`, `mine`, `gun`, `other_vehicle`. Для цих класів потрібна ваша навчена модель (Варіант 2).

### Варіант 2 — ваша навчена модель

Покладіть файл `best.pt` у папку `models/`:

```
models/
└── best.pt
```

Застосунок шукає модель у такому порядку:
1. `models/best.pt`
2. `best.pt` (у корені проєкту)
3. `best_model_aerial.pt`

Якщо жодного немає — натисніть у програмі **«Завантажити модель»** і виберіть `.pt` вручну.

## Запуск

```bash
python ProgramForCourse.py
```

### Порядок роботи в інтерфейсі

1. Натисніть **«Відкрити зображення»** і виберіть файл.
2. (Опційно) Виберіть **тип зображення** (aerial / uav / infrared / thermal / mixed).
3. Налаштуйте **Confidence** та **IoU** повзунками.
4. Натисніть **«ВИЯВИТИ ОБ'ЄКТИ»**.
5. Результат із рамками з'явиться на полотні; статистика — внизу.
6. Колесо миші — масштаб, затиснута ЛКМ — перетягування.

## Структура проєкту

```
.
├── ProgramForCourse.py   # GUI + логіка детекції
├── requirements.txt
├── README.md
├── LICENSE
└── models/
    └── best.pt           # файл моделі
```

## Датасет (опційно)

Датасет **не потрібен для запуску** програми — він потрібен лише якщо ви хочете
**перенавчити** модель під свої класи.

Якщо датасет є — розпакуйте його в папку `data/` (структура YOLO: `images/`, `labels/`, `data.yaml`).
Інструкція з навчання: https://docs.ultralytics.com/modes/train/
