# AI помощник

FastAPI приложение с использованием GigaChat

## Как запустить
1. Создать виртуальное окружение и установить зависимости:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. При жедании дополнить urls.txt — список ссылок.

3. Построить индекс:

```bash
python build_index.py --urls-file urls.txt --out-dir indexdata --chunk-size 200
```

4. Установить ваш ключ GigaChat:

```bash
export GIGACHAT_AUTH_KEY="<твой_ключ>"
```
5. Запустить сервис:

```bash
export INDEX_DIR="indexdata"
uvicorn app.main:app --reload
```

## Как пользоваться

Запрос через CLI:

```bash
python cli.py "Что вы можете сделать для ритейлеров"
```

## Пример ответа

```bash
python cli.py "Что вы можете сделать для ритейлеров"

--- Ответ ---

 Создание и внедрение HR-бота, который облегчает первичную обработку откликов кандидатов, записывает их на собеседования, задает вопросы для первичного опроса и поддерживает постоянный контакт с кандидатами. / 6,5 тыс. уникальных пользователей воспользовались ботом с января по март 2023 года. [0](https://eora.ru/cases/chat-boty/hr-bot-dlya-magnit-kotoriy-priglashaet-na-sobesedovanie)

 Разработка чат-бота для контакт-центров, который помогает разгружать отделы поддержки клиентов, автоматизируя общение и работу с типовыми запросами. [3](https://eora.ru/cases/icl-bot-sufler-dlya-kontakt-centra)
```


Если возникнут проблемы с сертификатами:
```bash
pip install certifi
curl -k "https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt" -w "\n" >> $(python -m certifi)
```