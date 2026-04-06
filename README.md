# LLM/VLM Evaluation for UAV MEC Research

Open-source model evaluation on **RTX 4090 (24GB VRAM) + 96GB RAM**.

Goal: identify the best LLM/VLM models for UAV autonomous navigation, MEC task scheduling, and communication-aware trajectory optimization.

## Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GeForce RTX 4090 (24GB VRAM) |
| CPU | Intel i9-13900KF (24C/32T) |
| RAM | 96GB DDR5 |
| OS | Ubuntu 24.04 |

## Model Categories

### A. Text LLMs (Reasoning / Code / Planning)

| Model | Params | VRAM (INT4) | Fits 24GB | Status |
|-------|--------|-------------|-----------|--------|
| Gemma 4 31B | 31B | ~17GB | ✅ INT4 | 🔲 TODO |
| Gemma 4 26B (MoE) | 26B (3.8B active) | ~15GB | ✅ Q4_K_M | 🔲 TODO |
| Qwen 3 32B | 32.8B | ~18GB | ✅ INT4 | 🔲 TODO |
| Qwen 3 30B-A3B (MoE) | 30.5B (3.3B active) | ~17GB | ✅ INT4 | 🔲 TODO |
| Qwen 3.5 35B-A3B (MoE) | 36B (3.5B active) | ~19.5GB | ✅ Q4_K_M | 🔲 TODO |
| Mistral Small 24B | 24B | ~14GB | ✅ INT4 | 🔲 TODO |
| Llama 3.2 8B | 8B | ~5.5GB | ✅ INT4 | 🔲 TODO |
| Gemma 2 9B | 9.2B | ~6GB | ✅ INT4 | 🔲 TODO |
| DeepSeek V3 671B (MoE) | 671B (37B active) | N/A | ❌ (need offload) | ⬜ Skip |

### B. Vision-Language Models (VLM)

| Model | Params | VRAM (INT4) | Fits 24GB | Status |
|-------|--------|-------------|-----------|--------|
| **Gemma 4 31B** (native vision) | 31B | ~17GB | ✅ INT4 | 🔲 TODO |
| Gemma 3 12B (vision) | 12B | ~7.5GB | ✅ Q6_K | 🔲 TODO |
| Qwen2.5-VL 7B | 8.3B | ~5.7GB | ✅ INT4 | 🔲 TODO |
| Qwen3-VL 30B-A3B (MoE) | 31.1B | ~17GB | ✅ INT4 | 🔲 TODO |
| Llama 3.2 11B-Vision | 11B | ~7GB | ✅ Q8_0 | 🔲 TODO |
| InternVL2.5 8B | 8B | ~5.5GB | ✅ INT4 | 🔲 TODO |
| InternVL2.5 26B | 26B | ~14.5GB | ✅ INT4 | 🔲 TODO |
| PaliGemma2 10B | 10B | ~6.5GB | ✅ INT4 | 🔲 TODO |
| SmolVLM 2B | 2.2B | ~2.6GB | ✅ FP16 | 🔲 TODO |

### C. Vision-Language-Action Models (VLA)

| Model | Params | VRAM (INT4) | Fits 24GB | Status |
|-------|--------|-------------|-----------|--------|
| OpenVLA 7B | 7.7B | ~5.3GB | ✅ INT4 | 🔲 TODO |

## Evaluation Dimensions

1. **Inference Speed**: tokens/s, first-token latency
2. **Reasoning Quality**: MMLU, HumanEval, GSM8K (text); VQA, OCR (vision)
3. **UAV Domain Tasks**: trajectory planning, obstacle description, communication optimization
4. **Memory Efficiency**: peak VRAM usage, batch size capacity

## Tools

- [llmfit](https://github.com/AlexsJones/llmfit) — Hardware compatibility checking
- [vLLM](https://github.com/vllm-project/vllm) — High-throughput inference
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — Quantized inference
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — Standardized benchmarks

## Project Structure

```
LLM-Evaluation/
├── README.md
├── scripts/
│   ├── run_llmfit.sh          # Hardware compatibility scan
│   ├── benchmark_text.py      # Text LLM benchmarks
│   ├── benchmark_vision.py    # VLM benchmarks
│   └── benchmark_uav.py      # UAV domain-specific tests
├── results/
│   ├── llmfit/                # llmfit output
│   ├── text_llm/              # Text benchmark results
│   └── vision_llm/            # Vision benchmark results
├── prompts/
│   └── uav_tasks.json         # UAV domain test prompts
└── docs/
    └── evaluation_plan.md     # Detailed evaluation plan
```

## Quick Start

```bash
# 1. Run llmfit hardware scan
llmfit --cli > results/llmfit/scan.txt

# 2. Install evaluation tools
pip install vllm transformers accelerate

# 3. Run benchmarks (see scripts/)
python scripts/benchmark_text.py --model google/gemma-4-31B --quant int4
```
