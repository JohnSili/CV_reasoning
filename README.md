# PyBullet Pick-and-Place Dataset Generator

Генератор синтетических сцен для дообучения VLM/VLA с structured reasoning supervision.

## Структура проекта

```
pybullet_dataset/
├── Dockerfile
├── requirements.txt
├── config.yaml
├── scripts/
│   ├── scene_builder.py      # создаёт сцену в PyBullet, рендерит RGB
│   ├── relations.py          # вычисляет пространственные отношения, строит reasoning trace
│   └── generate_dataset.py   # основной скрипт генерации
└── output/
    ├── images/               # PNG-рендеры сцен
    └── dataset.jsonl         # датасет в формате JSONL
```

## Запуск через Docker

```bash
# Сборка образа
docker build -t pybullet-datagen .

# Генерация 1000 сцен (по умолчанию)
docker run --rm -v $(pwd)/output:/workspace/output pybullet-datagen

# Кастомное количество сцен
docker run --rm -v $(pwd)/output:/workspace/output pybullet-datagen \
    bash -c "Xvfb :99 -screen 0 1024x768x24 &>/dev/null & sleep 1 && \
             python scripts/generate_dataset.py --scenes 5000 --seed 123"
```

## Запуск локально

```bash
pip install -r requirements.txt
python scripts/generate_dataset.py --scenes 1000 --out output --seed 42
```

## Формат одного сэмпла (dataset.jsonl)

```json
{
  "scene_id": 0,
  "image": "images/scene_00000.png",
  "camera": {"yaw": 45.0, "pitch": -40.0, "distance": 1.2},
  "instruction": "Pick the red cube to the left of the blue cylinder and place it at the target.",
  "relation_type": "left_of",
  "pick_id": "obj_001",
  "anchor_id": "obj_002",
  "place_pose": [0.12, -0.05, 0.648, 0.0],
  "reasoning": [
    "Step 1 [Candidate Selection]: ...",
    "Step 2 [Spatial Check (left_of)]: ...",
    "Step 3 [Pick Selection]: ...",
    "Step 4 [Place Target]: ..."
  ],
  "objects": [
    {"id": "obj_001", "category": "cube", "color": "red", "position": [-0.1, 0.05, 0.648]},
    ...
  ]
}
```

## Поддерживаемые отношения

| Тип             | Описание                                 |
|-----------------|------------------------------------------|
| `left_of`       | pick объект левее anchor по оси X        |
| `right_of`      | pick объект правее anchor по оси X       |
| `in_front_of`   | pick объект ближе к камере по оси Y      |
| `behind`        | pick объект дальше от камеры по оси Y    |
| `nearest_to`    | pick объект ближайший к anchor           |
| `farthest_from` | pick объект наиболее удалённый от anchor |

## Параметры генерации

Все ключевые параметры (количество объектов, диапазоны позиций, список отношений) вынесены в `config.yaml`.
