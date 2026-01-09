import requests
import json
import time

OLLAMA_URL = "http://localhost:11434/api/generate"

def test_connection():
    payload = {
        "model": "gemini-3-flash-preview:cloud",
        "prompt": "Say who you are and nothing else.",
        "stream": False,
        "options": {
            "num_ctx": 2048  # Minimum memory footprint
        }
    }

    print(f"[{time.strftime('%H:%M:%S')}] Sending 'Hello World' to Gemma 3...")
    
    try:
        # 30s timeout is more than enough for a 2-word response
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json().get("response", "")
            print(f"✔ Success! Ollama replied: {result.strip()}")
        else:
            print(f"✘ Server Error: {response.status_code}")
            print(f"Details: {response.text}")
            
    except requests.exceptions.Timeout:
        print("✘ Error: The request timed out. Ollama is taking too long to load the model.")
    except Exception as e:
        print(f"✘ Error: {e}")

if __name__ == "__main__":
    test_connection()