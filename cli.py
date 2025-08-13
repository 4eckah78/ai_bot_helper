import argparse
import requests
import os
import sys

DEFAULT_URL = os.getenv("EORA_API_URL", "http://127.0.0.1:8000/answer")

def ask(question: str, top_k: int, api_url: str):
    payload = {"question": question, "top_k": top_k}
    try:
        r = requests.post(api_url, json=payload, timeout=20)
    except Exception as e:
        print(f"Request failed: {e}", file=sys.stderr)
        sys.exit(1)

    if r.status_code != 200:
        print(f"Error {r.status_code}: {r.text}", file=sys.stderr)
        sys.exit(1)

    data = r.json()
    print("\n--- Ответ ---\n")
    print(data["answer"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI для API EORA")
    parser.add_argument("question", help="Ваш вопрос", nargs="+") 
    parser.add_argument("--top_k", type=int, default=3, help="Количество результатов")
    parser.add_argument("--api_url", default=DEFAULT_URL, help="URL API")

    args = parser.parse_args()

    question_str = " ".join(args.question)

    ask(question_str, args.top_k, args.api_url)
