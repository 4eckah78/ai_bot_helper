from gigachat import GigaChat


class GigachatClient:
    def __init__(self, auth_key: str):
        if not auth_key:
            raise RuntimeError("GIGACHAT_AUTH_KEY required")
        self.auth = auth_key

    def generate(self, prompt: str, model="GigaChat", max_tokens=800, temperature=1.7):
        messages = [
            {
                "role": "system",
                "content": "Ты помощник. Помогаешь отвечать клиентам. Отвечай строго по контексту",
            },
            {"role": "user", "content": prompt},
        ]
        giga = GigaChat(credentials=self.auth)
        response = giga.chat(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return response.choices[0].message.content

