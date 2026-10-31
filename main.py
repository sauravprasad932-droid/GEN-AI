
import os
import requests
import json
from dotenv import load_dotenv

# 1. Load API key from .env
load_dotenv()

API_KEY = os.getenv("NVIDIA_API_KEY")

if not API_KEY:
    raise ValueError("NVIDIA API key not found in .env file")

# 2. NVIDIA API URL
invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"

# 3. Headers
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "text/event-stream",
    "Content-Type": "application/json"
}

# 4. Request payload
payload = {
    "model": "nvidia/nemotron-3.5-bolt-e4b",
    "messages": [
        {
            "role": "user",
            "content": "Explain Generative AI in simple words."
        }
    ],
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 1024,
    "stream": True,
    "chat_template_kwargs": {
        "enable_thinking": True
    }
}

# 5. Send request
try:
    response = requests.post(
        invoke_url,
        headers=headers,
        json=payload,
        stream=True,
        timeout=120
    )

    response.raise_for_status()

    print("\nAI Response:\n")

    # 6. Receive streaming response
    for line in response.iter_lines(decode_unicode=True):

        if not line:
            continue

        if line.startswith("data: "):

            data = line[6:]

            if data == "[DONE]":
                break

            result = json.loads(data)

            content = result["choices"][0]["delta"].get("content", "")

            if content:
                print(content, end="", flush=True)

    print("\n")

except requests.exceptions.HTTPError:
    print("HTTP Error:", response.status_code)
    print(response.text)

except requests.exceptions.RequestException as e:
    print("Connection Error:", e)

except Exception as e:
    print("Error:", e)
