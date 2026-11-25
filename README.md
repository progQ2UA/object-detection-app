# Система виявлення об'єктів (Object Detection System)

Десктопний застосунок на Python + Tkinter для виявлення об'єктів на зображеннях
(аерофотознімки, UAV, інфрачервоні/тепловізійні кадри) за допомогою моделі YOLOv8.

## Можливості

- Завантаження зображень (JPG, PNG, BMP, TIFF)
- Завантаження моделі YOLOv8 (`models/best.pt`)
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

### 1. Клонувати репозиторій

```bash
git clone https://github.com/<ваш-нік>/object-detection-app.git
cd object-detection-app
```

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

## Дані та модель

Через обмеження GitHub великі файли зберігаються окремо:

- **Архів із даними (dataset):** див. розділ [Releases](../../releases) — завантажте `dataset.zip` та розпакуйте у папку `data/`.
- **Модель:** покладіть файл `best.pt` у папку `models/`.
  - Якщо у вас ще немає навченої моделі, можна швидко перевірити роботу зі стандартною `yolov8n.pt`:
    ```bash
    # завантажити попередньо навчену модель
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
    # скопіювати у models/
    mkdir -p models && cp yolov8n.pt models/best.pt
    ```

Після цього структура має виглядати так:
```
models/
└── best.pt
data/
└── (розпакований датасет)
```

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

## Якщо модель не знайдена

Застосунок шукає файли в такому порядку:
1. `models/best.pt`
2. `best.pt` (у корені проєкту)
3. `best_model_aerial.pt`

Якщо жодного немає — натисніть **«Завантажити модель»** у лівій панелі та виберіть `.pt` вручну.

## Структура проєкту

```
.
├── ProgramForCourse.py
├── requirements.txt
├── README.md
├── models/
│   └── best.pt
└── data/
    └── (датасет)
```