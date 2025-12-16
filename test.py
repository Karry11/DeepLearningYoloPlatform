"""
Simple DeepSeek connectivity test.

This does NOT run training or generate scripts. It just sends a single prompt
("编写斐波那契函数") to the DeepSeek API and prints the raw reply plus any code
block extracted.
"""

import os
import re
import sys
from typing import Optional

try:
    import requests
except Exception as exc:  # pragma: no cover
        print(f"requests is required for this test: {exc}")
        sys.exit(1)


def extract_code_block(text: str) -> Optional[str]:
    """Extract the last fenced code block from the response, if any."""
    match = re.findall(r"```(?:python)?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match[-1].strip()
    return None


def main():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    endpoint = os.getenv("DEEPSEEK_ENDPOIN", "https://api.deepseek.com/v1/chat/completions")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-coder")

    if not api_key:
        print("DEEPSEEK_API_KEY is not set. Please set it before running this test.")
        sys.exit(1)

    prompt = "编写一个 Python 斐波那契函数，返回前 n 项的列表。"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print(f"Calling DeepSeek: endpoint={endpoint}, model={model}")
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
    print(f"HTTP status: {resp.status_code}")
    resp.raise_for_status()

    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        print("No choices returned from DeepSeek.")
        sys.exit(1)

    content = choices[0]["message"]["content"]
    print("\n--- Raw reply ---")
    print(content)

    code = extract_code_block(content)
    if code:
        print("\n--- Extracted code block ---")
        print(code)
    else:
        print("\nNo code block found in the reply.")


if __name__ == "__main__":
    main()
