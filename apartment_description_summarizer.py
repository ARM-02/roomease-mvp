import requests
import re
import json

LMSTUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"

SUMMARIZE_PROMPT = """
You will receive a long real estate listing description (Idealista style).

Your task:
- Summarize it into **1–2 extremely concise sentences**.
- Keep ONLY factual info explicitly mentioned.
- NO creativity.
- NO invented features.
- NO marketing fluff.
- MAX 50 words.

Return ONLY plain text, no introduction to your reponse like "Here is your concise". No markdown, no JSON. Just plain text

Description:
"""

def summarize_description(text: str) -> str:
    prompt = SUMMARIZE_PROMPT + "\n" + text.strip()

    response = requests.post(
        LMSTUDIO_URL,
        json={
            "model": "llama-3.2-1b-instruct",   # match the model you're running
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 120,
        },
        timeout=90,
    )

    response.raise_for_status()
    data = response.json()

    content = data["choices"][0]["message"]["content"].strip()
    # Remove markdown if the model decides to add it (some small models do this)
    content = content.replace("```", "").replace("json", "").strip()

    return content[25:]


