"""Benchmark a single LLM/VLM model: speed + quality + VRAM.

Usage:
    python benchmark_model.py --model google/gemma-4-31B --quant int4
    python benchmark_model.py --model Qwen/Qwen2.5-VL-7B-Instruct --quant int4 --vision
"""
import argparse
import json
import os
import time

import torch


def get_gpu_mem():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9, torch.cuda.max_memory_allocated() / 1e9
    return 0, 0


def load_model(model_id: str, quant: str, cache_dir: str, is_vision: bool = False):
    """Load model with quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, BitsAndBytesConfig

    print(f"Loading {model_id} ({quant})...")
    t0 = time.time()

    kwargs = {
        "device_map": "auto",
        "cache_dir": cache_dir,
        "trust_remote_code": True,
    }

    if quant == "int4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    elif quant == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kwargs["torch_dtype"] = torch.float16

    if is_vision:
        # Try multiple auto classes for VLM compatibility
        model = None
        for auto_cls_name in ["AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"]:
            try:
                auto_cls = getattr(__import__("transformers", fromlist=[auto_cls_name]), auto_cls_name)
                model = auto_cls.from_pretrained(model_id, **kwargs)
                print(f"  Loaded with {auto_cls_name}")
                break
            except (ImportError, AttributeError, ValueError):
                continue
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
        tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
        processor = None

    load_time = time.time() - t0
    cur_mem, peak_mem = get_gpu_mem()
    print(f"  Loaded in {load_time:.1f}s, VRAM: {cur_mem:.1f}GB (peak {peak_mem:.1f}GB)")

    return model, tokenizer, processor, {"load_time_s": load_time, "vram_gb": cur_mem, "peak_vram_gb": peak_mem}


def benchmark_text_speed(model, tokenizer, num_runs=5):
    """Benchmark text generation speed."""
    prompts = [
        "Explain how A* pathfinding works in 3 sentences.",
        "Write a Python function to compute the shortest path between two points.",
        "What are the key differences between minimum-snap and minimum-jerk trajectories?",
        "Design a reward function for multi-UAV formation flying with obstacle avoidance.",
        "Explain the concept of differential flatness in quadrotor control.",
    ]

    results = []
    for i, prompt in enumerate(prompts[:num_runs]):
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = f"<s>[INST] {prompt} [/INST]"

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        gen_tokens = out.shape[1] - input_len
        tok_per_s = gen_tokens / elapsed
        ttft = elapsed / gen_tokens * 1000 if gen_tokens > 0 else 0  # rough estimate

        output_text = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)

        results.append({
            "prompt_idx": i,
            "input_tokens": input_len,
            "output_tokens": gen_tokens,
            "time_s": round(elapsed, 2),
            "tokens_per_s": round(tok_per_s, 1),
            "output_preview": output_text[:200],
        })
        print(f"  Run {i+1}: {gen_tokens} tokens in {elapsed:.2f}s = {tok_per_s:.1f} tok/s")

    avg_speed = sum(r["tokens_per_s"] for r in results) / len(results)
    return results, {"avg_tokens_per_s": round(avg_speed, 1)}


def benchmark_quality(model, tokenizer):
    """Quick quality check with known-answer questions."""
    qa_pairs = [
        ("What is 15 * 23?", "345"),
        ("What is the capital of France?", "Paris"),
        ("In Python, what does `len([1,2,3])` return?", "3"),
        ("What is the derivative of x^3?", "3x^2"),
        ("Sort [5, 2, 8, 1] in ascending order.", "1, 2, 5, 8"),
    ]

    correct = 0
    for q, expected in qa_pairs:
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        except Exception:
            text = f"<s>[INST] {q} [/INST]"

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        if expected.lower() in response.lower():
            correct += 1
            print(f"  ✅ Q: {q[:50]} → contains '{expected}'")
        else:
            print(f"  ❌ Q: {q[:50]} → got: {response[:100]}")

    return {"correct": correct, "total": len(qa_pairs), "accuracy": round(correct / len(qa_pairs) * 100, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--quant", default="int4", choices=["fp16", "int8", "int4"])
    parser.add_argument("--vision", action="store_true", help="Load as vision model")
    parser.add_argument("--cache-dir", default="/data/llm-eval/cache")
    parser.add_argument("--output-dir", default="/data/llm-eval/results")
    parser.add_argument("--num-runs", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    # Determine output subdirectory
    category = "vision" if args.vision else "text"
    out_dir = os.path.join(args.output_dir, category)
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    model, tokenizer, processor, load_info = load_model(
        args.model, args.quant, args.cache_dir, args.vision)

    # Benchmark speed
    print("\n=== Speed Benchmark ===")
    speed_results, speed_summary = benchmark_text_speed(model, tokenizer, args.num_runs)

    # Benchmark quality
    print("\n=== Quality Check ===")
    quality_results = benchmark_quality(model, tokenizer)

    # Final VRAM
    cur_mem, peak_mem = get_gpu_mem()

    # Compile results
    results = {
        "model": args.model,
        "quantization": args.quant,
        "is_vision": args.vision,
        "load": load_info,
        "speed": speed_summary,
        "speed_details": speed_results,
        "quality": quality_results,
        "final_vram_gb": round(cur_mem, 1),
        "peak_vram_gb": round(peak_mem, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save
    safe_name = args.model.replace("/", "_").replace("-", "_")
    out_file = os.path.join(out_dir, f"{safe_name}_{args.quant}.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Model:  {args.model} ({args.quant})")
    print(f"VRAM:   {results['peak_vram_gb']}GB peak")
    print(f"Speed:  {speed_summary['avg_tokens_per_s']} tok/s avg")
    print(f"Quality: {quality_results['correct']}/{quality_results['total']} ({quality_results['accuracy']}%)")
    print(f"Saved:  {out_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
