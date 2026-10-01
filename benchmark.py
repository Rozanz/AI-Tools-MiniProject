import time
import requests
import json
import csv

OLLAMA_URL = "http://localhost:11434/api/generate"
MODELS = ["llama3.2:3b", "qwen2.5:7b", "hf-qwen:latest"]

# Task 4 Settings: Low Temp, High Temp, and modified top_p
TEST_CONFIGS = [
    {"name": "Low_Temp", "options": {"temperature": 0.1, "top_p": 0.9}},
    {"name": "High_Temp", "options": {"temperature": 1.0, "top_p": 0.9}},
    {"name": "Strict_Sampling", "options": {"temperature": 0.7, "top_p": 0.5}}
]

PROMPTS = [
    {"id": "P1_Logic", "text": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Explain your step-by-step reasoning."},
    {"id": "P2_Code", "text": "Write a concise Python script using hashlib to compute the SHA-256 hash of a file given its path."},
    {"id": "P3_Cybersecurity", "text": "Explain the difference between pre-encryption and post-encryption detection in ransomware defense, and why process termination latency matters."},
    {"id": "P4_Systems", "text": "Summarize the trade-offs between unified memory architectures and discrete GPU PCIe configurations for low-latency LLM inference in three bullet points."},
    {"id": "P5_FailureTest", "text": "Provide the exact binary hex dump of the first 64 bytes of an empty ext4 filesystem superblock, followed by counting how many times the letter 'r' appears in the word 'strawberry' backwards."}
]

records = []

print(f"{'Model':<15} | {'Config':<15} | {'Prompt':<16} | {'Speed (t/s)':<12} | {'Elapsed (s)':<10}")
print("-" * 75)

for model in MODELS:
    for cfg in TEST_CONFIGS:
        for p in PROMPTS:
            payload = {
                "model": model,
                "prompt": p["text"],
                "stream": False,
                "options": cfg["options"]
            }
            start_wall = time.time()
            try:
                res = requests.post(OLLAMA_URL, json=payload, timeout=120).json()
                elapsed_wall = time.time() - start_wall
                
                load_duration_s = res.get("load_duration", 0) / 1e9
                eval_count = res.get("eval_count", 0)
                eval_duration_s = res.get("eval_duration", 1) / 1e9
                tps = eval_count / eval_duration_s if eval_duration_s > 0 else 0
                
                print(f"{model:<15} | {cfg['name']:<15} | {p['id']:<16} | {tps:<12.2f} | {elapsed_wall:<10.2f}")
                
                records.append({
                    "model": model,
                    "config": cfg["name"],
                    "temperature": cfg["options"]["temperature"],
                    "top_p": cfg["options"]["top_p"],
                    "prompt_id": p["id"],
                    "load_time_sec": round(load_duration_s, 3),
                    "eval_count": eval_count,
                    "elapsed_sec": round(elapsed_wall, 2),
                    "tokens_per_sec": round(tps, 2),
                    "response": res.get("response", "")
                })
            except Exception as e:
                print(f"{model:<15} | {cfg['name']:<15} | {p['id']:<16} | ERROR: {e}")

# Save full responses to JSON
with open("benchmark_data.json", "w") as f:
    json.dump(records, f, indent=2)

# Export metrics to CSV for report tables
keys = ["model", "config", "temperature", "top_p", "prompt_id", "load_time_sec", "eval_count", "elapsed_sec", "tokens_per_sec"]
with open("benchmark_data.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)

print("\nFinished! Data exported to benchmark_data.csv and benchmark_data.json.")