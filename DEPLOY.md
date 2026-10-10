# DEPLOY — Docker, GitHub и Render

## Локальный запуск

Нужны Docker Desktop и Compose. В корне репозитория выполните:

    docker compose up --build

Откройте http://localhost:8000. Подождите состояния «Модель готова», нажмите «Пример фото» и «Найти молоток».

Проверка API:

    curl http://localhost:8000/health
    curl http://localhost:8000/ready
    curl http://localhost:8000/info
    curl -X POST http://localhost:8000/detect -F "file=@static/img/demo.jpg"

В Windows PowerShell используйте curl.exe вместо curl, если команда разрешается в Invoke-WebRequest.

Проверка тестов:

    pip install -r requirements-dev.txt
    pytest -q

После запуска контейнера можно выполнить smoke-тест: bash scripts/smoke_test.sh. Для внешнего URL задайте BASE_URL.

## Render Blueprint

1. Убедитесь, что в корне репозитория есть Dockerfile и render.yaml.
2. Откройте https://dashboard.render.com/.
3. Выберите New + → Blueprint и подключите репозиторий HAMMER_dec.
4. Проверьте, что единственный создаваемый сервис — nazdratenko-hammer-lr8, Docker Web Service, plan free.
5. Примените Blueprint и дождитесь завершения сборки.
6. Откройте выданный Render URL.
7. Проверьте /health, дождитесь /ready со статусом ready и выполните детекцию на сайте.

Пример удалённой проверки:

    curl https://YOUR-SERVICE.onrender.com/health
    curl https://YOUR-SERVICE.onrender.com/ready
    curl -F "file=@static/img/demo.jpg" https://YOUR-SERVICE.onrender.com/detect

Замените YOUR-SERVICE на реальное имя хоста Render. Демонстрационный файл должен существовать на компьютере, откуда выполняется curl.

## Важные ограничения бесплатного тарифа

Blueprint задаёт plan free и один Web Service, не создаёт платную базу данных или отдельный worker. У бесплатного сервиса возможен spin-down после простоя, поэтому первый запрос после сна дольше обычного. Время прогрева измеряется по логам, а не гарантируется фиксированной цифрой.

Пользовательские изображения не сохраняются приложением. Локальная файловая система Render эфемерна. Не загружайте секреты в репозиторий.

## Логи и откат

Откройте Web Service → Logs в Render Dashboard. Каждому запросу соответствует request_id; ищите его в HTTP-логе и записи инференса. Для диагностики и отката руководствуйтесь RUNBOOK.md.
