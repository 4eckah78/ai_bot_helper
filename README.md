# Тестовое задание EORA — ответный сервис с Gigachat и материалами с сайта

## Что я сделал
- Реализовал FastAPI-сервис `POST /answer` который:
  - скачивает страницы из списка ссылок (конфигурируются),
  - извлекает текст и разбивает на фрагменты,
  - индексирует фрагменты TF-IDF (scikit-learn),
  - на запрос извлекает top-k фрагментов и формирует контекст,
  - отправляет контекст + вопрос внешней модели Gigachat (абстрактный клиент).
- Добавил CLI (typer) для локального удобного тестирования.
- Клиент Gigachat имеет два режима:
  - `mock` — чтобы можно было тестировать без реального ключа/эндпоинта;
  - `real` — используйте реальные `GIGACHAT_BASE` и `GIGACHAT_KEY` в окружении (код гибкий, подстройте под реальный формат ответа Gigachat).
- Код разделён: `retriever.py` (scrape+index), `gigachat_client.py` (обёртка), `main.py` (API).

## Как запустить (локально)
1. Создать виртуальное окружение и установить зависимости:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. Подготовить urls.txt — список ссылок (по одной строке).

Построить индекс:

bash
Копировать
Редактировать
python build_index.py --urls-file urls.txt --out-dir indexdata --chunk-size 200
(Если при обучении эмбеддингов в app.utils кэш не существовал — он создаст vectorizer+svd.)

Установить ключ GigaChat (у тебя есть):

bash

export GIGACHAT_AUTH_KEY="Bearer <твой_ключ>"
# если у тебя только токен без "Bearer ", можно экспортировать просто токен — код добавит Bearer автоматически
Запустить сервис:

bash
export INDEX_DIR="indexdata"
uvicorn app.main:app --reload

Запрос:


# EORA RAG + GigaChat

## Что делает
- `build_index.py` — строит FAISS индекс из `urls.txt` (чанки текста → эмбеддинги → faiss + meta.pkl).
- FastAPI (`app.main`) — при запросе делает:
  1. Векторизация вопроса (тот же pipeline),
  2. Поиск top_k чанкoв в FAISS,
  3. Формирование prompt с пометками [1],[2]...,
  4. Отправка в GigaChat (SDK или REST),
  5. Возврат ответа + список использованных URL.

## Как запустить
1. Установить зависимости:
   ```bash
   pip install -r requirements.txt




pip install certifi
curl -k "https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt" -w "\n" >> $(python -m certifi)