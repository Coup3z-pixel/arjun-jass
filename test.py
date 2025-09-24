import os
import requests

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")  # export this in your shell

BASE_URL = "https://api.together.xyz/v1/completions"
MODEL = "jaspertsh08_a03a/Mixtral-8x7B-v0.1-50e88a5a-d680bff3"

def complete(prompt: str, max_tokens: int = 300, temperature: float = 0.7) -> str:
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(BASE_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()  # raises for 4xx/5xx
    data = resp.json()
    # Together returns choices similar to OpenAI:
    return data["choices"][0]["text"]

if __name__ == "__main__":
    out = complete("What are some fun things to do in New York?")
    print(out)
