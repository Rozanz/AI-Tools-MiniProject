import requests
import time

url = "http://localhost:11434/api/generate"

payload = {
    "model": "llama3.2:3b",
    "prompt": "Analyze the following behavioral event: A process spawns PowerShell with an encoded command, disables Volume Shadow Copies via vssadmin, and initiates mass file renames. What specific threat behavior does this indicate, and what should be the immediate containment action?",
    "stream": False,
    "options": {
        "temperature": 0.2
    }
}

print("Sending request to Ollama local API...")
start_time = time.time()
response = requests.post(url, json=payload).json()
total_time = time.time() - start_time

print("\n=== Model Output ===")
print(response.get("response"))

print("\n=== Performance Metrics (Task 5) ===")
print(f"Total Response Time: {total_time:.2f} seconds")
eval_count = response.get("eval_count", 0)
eval_duration_sec = response.get("eval_duration", 1) / 1e9
tokens_per_sec = eval_count / eval_duration_sec if eval_duration_sec > 0 else 0
print(f"Tokens Generated: {eval_count}")
print(f"Generation Speed: {tokens_per_sec:.2f} tokens/second")