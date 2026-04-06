"""Benchmark a GGUF model via llama-cpp-python.

Usage:
    python benchmark_gguf.py --model-path /data/llm-eval/models/model.gguf --n-gpu-layers -1
"""
import argparse
import json
import os
import time

# Fix NVIDIA library paths before importing llama_cpp
nvidia_dir = os.path.join(os.path.dirname(os.__file__), "site-packages/nvidia")
cuda_paths = []
for sub in ["cuda_runtime", "cublas", "cufft", "cusolver", "cusparse"]:
    p = os.path.join(nvidia_dir, sub, "lib")
    if os.path.exists(p):
        cuda_paths.append(p)
if cuda_paths:
    os.environ["LD_LIBRARY_PATH"] = ":".join(cuda_paths) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

# Preload system libstdc++ to fix GLIBCXX version issue
import ctypes
try:
    ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libstdc++.so.6", mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass

from llama_cpp import Llama


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to .gguf file")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="-1 = all layers on GPU")
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--output-dir", default="/data/llm-eval/results/text")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.model_path} (n_gpu_layers={args.n_gpu_layers})...")
    t0 = time.time()
    llm = Llama(
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.ctx_size,
        verbose=False,
    )
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    # Speed benchmark
    prompts = [
        "Explain how A* pathfinding works in 3 sentences.",
        "Write a Python function to compute the shortest path between two points.",
        "What are the key differences between minimum-snap and minimum-jerk trajectories?",
        "Design a reward function for multi-UAV formation flying with obstacle avoidance.",
        "Explain the concept of differential flatness in quadrotor control.",
    ]

    print("\n=== Speed Benchmark ===")
    results = []
    for i, prompt in enumerate(prompts[:args.num_runs]):
        t0 = time.time()
        output = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0,
        )
        elapsed = time.time() - t0
        text = output["choices"][0]["message"]["content"]
        tokens = output["usage"]["completion_tokens"]
        tok_s = tokens / elapsed
        results.append({"tokens": tokens, "time_s": round(elapsed, 2), "tok_s": round(tok_s, 1)})
        print(f"  Run {i+1}: {tokens} tokens in {elapsed:.2f}s = {tok_s:.1f} tok/s")

    avg_speed = sum(r["tok_s"] for r in results) / len(results)

    # Quality check
    print("\n=== Quality Check ===")
    qa = [
        ("What is 15 * 23?", "345"),
        ("What is the capital of France?", "Paris"),
        ("In Python, what does `len([1,2,3])` return?", "3"),
        ("What is the derivative of x^3?", "3x^2"),
        ("Sort [5, 2, 8, 1] in ascending order.", "1, 2, 5, 8"),
    ]
    correct = 0
    for q, expected in qa:
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": q}],
            max_tokens=128, temperature=0)
        resp = out["choices"][0]["message"]["content"]
        if expected.lower() in resp.lower():
            correct += 1
            print(f"  ✅ Q: {q[:50]} → contains '{expected}'")
        else:
            print(f"  ❌ Q: {q[:50]} → {resp[:80]}")

    # Save
    model_name = os.path.basename(args.model_path).replace(".gguf", "")
    out = {
        "model": args.model_path,
        "model_name": model_name,
        "n_gpu_layers": args.n_gpu_layers,
        "ctx_size": args.ctx_size,
        "load_time_s": round(load_time, 1),
        "speed": {"avg_tok_s": round(avg_speed, 1), "runs": results},
        "quality": {"correct": correct, "total": len(qa), "accuracy": round(correct / len(qa) * 100, 1)},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_file = os.path.join(args.output_dir, f"{model_name}.json")
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Model:   {model_name}")
    print(f"Speed:   {avg_speed:.1f} tok/s avg")
    print(f"Quality: {correct}/{len(qa)} ({correct/len(qa)*100:.0f}%)")
    print(f"Saved:   {out_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
