# cli.py
import typer
import requests
import os
from pathlib import Path
import json

app = typer.Typer()

DEFAULT_URL = os.getenv("EORA_API_URL", "http://127.0.0.1:8000/answer")

    
@app.command()
def hey(
    word: str,
    top_k: int = 3,
    api_url: str = DEFAULT_URL,
):
    print(f"Hello {word}")

if __name__ == "__main__":
    app()
