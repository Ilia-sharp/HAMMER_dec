# DEPLOY.md — публикация репозитория и деплой на Render

## 1. Загрузка на GitHub

### Вариант А. Через git (рекомендуется)

1. Создайте **пустой** репозиторий на https://github.com/new:
   имя `SII-LR02-Ivanov`, без README и .gitignore (они уже в проекте).
2. В папке проекта инициализируйте git и запушьте:

```bash
cd SII-LR02-Ivanov
git init
git add .
git commit -m "ЛР2 вариант 16: hammer detection"
git branch -M main
git remote add origin https://github.com/<ВАШ_ЛОГИН>/SII-LR02-Ivanov.git
git push -u origin main
```

   > Файлы `models/best.pt` (5.2 MB) и `models/best.onnx` (9 MB) не попадают
   > под лимиты обычного `git push` — Git LFS не требуется.
   > Можно вместо git загрузить файлы через веб (вариант Б ниже).

3. При запросе логина используйте **Personal Access Token** вместо пароля
   (GitHub → Settings → Developer settings → Tokens classic, scope `repo`).

### Вариант Б. Через веб-интерфейс

GitHub → ваш репозиторий → **Add file → Upload files** → перетащите
СОДЕРЖИМОЕ папки (не сам zip). Ограничение web-загрузки: ≤ 100 файлов за раз
и ≤ 25 MB на файл — в проекте лимиты соблюдены (best.pt ≈ 5.2 MB).
Затем завершите первый коммит кнопкой *Commit changes*.

> GitVerse: действия те же, только URL репозитория
> `https://gitverse.ru/<user>/SII-LR02-Ivanov.git` (команды в README, п.9).

## 2. Деплой на Render по Blueprint

1. Откройте https://dashboard.render.com/ и войдите (можно через GitHub).
2. **New + → Blueprint**.
3. Подключите GitHub-аккаунт (если ещё не подключён) и выберите
   репозиторий `SII-LR02-Ivanov`.
4. Render сам найдёт `render.yaml` в корне и покажет, что будет создано:
   `sii-lr02-hammer-api` — **Web Service, план Free ($0/мес, 0.1 CPU, 512 MB RAM)**.
5. Нажмите **Apply** (или *Create services*). Сборка займёт ~2 минуты
   (torch/ultralytics НЕ ставятся — только onnxruntime + headless-cv2).
6. Дождитесь статуса **Live**.

Что прописано в `render.yaml`:

| Параметр | Значение | Зачем |
|---|---|---|
| runtime | python 3.12.14 | совпадает с локальной средой |
| buildCommand | `pip install -r requirements.txt` + переустановка headless-cv2 | onnxruntime, headless-cv2 (без GUI-либ) |
| startCommand | `uvicorn src.api:app --host 0.0.0.0 --port $PORT --workers 1` | Render сам подаёт порт |
| healthCheckPath | `/health` | отвечает даже во время инференса (детект — в threadpool) |
| OMP/MKL_NUM_THREADS=1 | — | на 0.1 CPU многопоточность только вредит |
| MALLOC_ARENA_MAX=2 | — | экономия RAM в контейнере |
| YOLO_CONFIG_DIR=/tmp/Ultralytics | — | записываемый каталог (нужен, только если включится резервный бэкенд ultralytics) |
| plan: free | 0.1 CPU / 512 MB | требование лабораторной; фактический RSS ≈ 150 MB |

## 3. Проверка после деплоя

```bash
# статус (должен быть status: ok, model_loaded: true)
curl https://<ИМЯ-СЕРВИСА>.onrender.com/health

# детекция
curl -F "file=@data/test_frames/test_01.jpg" https://<ИМЯ-СЕРВИСА>.onrender.com/detect
```

Или просто откройте `https://<ИМЯ-СЕРВИСА>.onrender.com/` в браузере — там
**веб-интерфейс детекции**: перетащите фото (или кнопка «Пример фото») →
**Найти молоток** → рамки с conf поверх фото и статистика. Справа пусто?
Значит сервис ещё «просыпается» — подождите ~50 с и повторите.
JSON-описание сервиса доступно на `/info`, Swagger — на `/docs`.

**Важно про free-тариф:** после 15 минут без запросов сервис засыпает.
Первый запрос будет «холодным» (~50 сек) — это нормально, повторные отвечают
за 1–3 секунды (CPU-инференс YOLO26n).

## 4. Автодеплой

`autoDeploy: true` в `render.yaml`: каждый `git push` в main автоматически
пересобирает сервис. Обновили веса `models/best.pt` → push → через 5 минут
API отдаёт новую модель.

## 5. Частые проблемы

| Симптом | Причина / решение |
|---|---|
| Build failed на pip install | Проверьте, что requirements.txt не менялся и не добавлен torch/ultralytics (они не нужны на сервере — только для обучения локально) |
| Deploy жив, но `/health` → `degraded` | `models/best.onnx` не загрузился (проверьте, что файл закоммичен и не попал в .gitignore) — Logs в Render покажут `error` из onnxruntime; резервный бэкенд ultralytics включится, только если добавить torch в requirements.txt |
| Первый запрос висит 50+ сек | Cold start бесплатного инстанса — ожидаемо |
| 502 Bad Gateway | Сервис спит или упал: зайдите в Render → Manual Deploy → Clear build cache & deploy |
| Изменения не применяются | Проверьте, что push был в ветку main и autoDeploy включён |
