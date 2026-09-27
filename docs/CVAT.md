# CVAT: ручная верификация разметки (ЛР2, вариант 16)

В работе авторазметка выполнена через **YOLOE visual prompting** (3 опорных кадра
с ручными bbox), затем метки прошли ручной QA. Ниже — как воспроизвести этап
разметки/проверки в CVAT (Computer Vision Annotation Tool).

## 1. Установка CVAT (Docker)

```bash
git clone https://github.com/cvat-ai/cvat.git
cd cvat
docker compose up -d
docker exec -it cvat_db bash -c 'PGPASSWORD=$POSTGRES_PASSWORD psql -U root -c "ALTER USER root WITH SUPERUSER;"'
# браузер: http://localhost:8080 — регистрируемся, логин admin
```

Требования: Docker + Docker Compose v2, ~6 ГБ диска.

## 2. Импорт готовых предразметок

В репозитории лежит `cvat_hammer_yolo11.zip` — экспорт формата
**«CVAT for images 1.1»** с предразметкой всех 55 кадров датасета
(44 train + 11 val, класс `hammer`).

1. Projects → **Create new project** → имя `hammer_lr02`, метка `hammer` (тип Rectangle).
2. В проекте: **Actions → Upload data** → выбрать `cvat_hammer_yolo11.zip`.
3. Галочка *Upload annotations* (архив уже содержит `annotations.xml` и изображения).
4. Открыть задачу — все bbox уже на местах, остаётся выборочно поправить.

## 3. Ручная правка

- `N` — новая рамка; рамки растягиваются за углы; `Ctrl+Z` — отмена.
- Проверка по списку: боксы не должны срезать головку или рукоять молотка.
- Кадры со сложными сценами (молоток «в работе») размечаются целиком: головка + рукоять одним bbox.

## 4. Экспорт обратно в YOLO

Task → **Actions → Export task dataset** → формат **YOLO 1.1** → получим
`obj_train_data/*.txt` + `train.txt` в координатах
`class cx cy w h` (нормализованные) — это совместимо с нашими метками
`data/dataset/labels/`, файлы можно копировать напрямую.

## 5. Зачем это в отчёте

- демонстрирует цикл «авторазметка → ручная верификация → переобучение»;
- CVAT хранит историю правок (кто и когда менял рамку) — удобно для сдачи работы.
