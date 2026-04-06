# Model Evaluation Plan

## Phase 1: Hardware Compatibility (Day 1)

- [x] Run llmfit full scan (530 compatible models found)
- [x] Calculate VRAM for target models (FP16/INT8/INT4)
- [x] Confirm Gemma 4 31B fits on RTX 4090 at INT4 (~17GB)

## Phase 2: Text LLM Benchmarks (Day 2-3)

Priority order (by expected capability × efficiency):

1. **Gemma 4 31B** — #3 Arena ELO (1452), native vision, Apache 2.0
2. **Qwen 3 32B** — strong coding + reasoning
3. **Qwen 3 30B-A3B (MoE)** — only 3.3B active params, fast inference
4. **Mistral Small 24B** — fits comfortably, good general performance
5. **Llama 3.2 8B** — lightweight baseline

Benchmarks:
- MMLU (general knowledge)
- HumanEval (coding)
- GSM8K (math reasoning)
- Throughput (tokens/s via vLLM)
- First-token latency

## Phase 3: Vision LLM Benchmarks (Day 3-4)

Priority order:

1. **Gemma 4 31B** (native multimodal) — same model as text, built-in vision
2. **Qwen2.5-VL 7B** — fast, well-tested VLM
3. **Qwen3-VL 30B-A3B** — MoE vision, latest
4. **InternVL2.5 26B** — strong visual reasoning
5. **Llama 3.2 11B-Vision** — Meta baseline
6. **SmolVLM 2B** — ultra-lightweight, edge deployment reference

Benchmarks:
- VQAv2 (visual question answering)
- TextVQA (OCR + reasoning)
- MMMU (multimodal understanding)
- UAV-specific: aerial image description, obstacle identification

## Phase 4: UAV Domain Evaluation (Day 4-5)

Custom benchmarks for our research:

1. **Trajectory Planning**: Given start/goal + obstacles, output waypoints
2. **Obstacle Description**: Given aerial image, describe obstacles and suggest path
3. **Communication Optimization**: Given channel conditions, suggest UAV positions
4. **Task Scheduling**: Given MEC tasks + UAV positions, allocate resources

## Phase 5: Consolidation (Day 5)

- Compile results into comparison tables
- Select top models for each use case
- Document deployment configurations
- Update Outline with final recommendations

## Key Decisions

### Gemma 4 31B Analysis

**Can it run on RTX 4090?** YES

| Quantization | VRAM | Speed | Quality Loss |
|-------------|------|-------|-------------|
| FP16 | ~62GB | — | ❌ Doesn't fit |
| INT8 | ~31GB | — | ❌ Doesn't fit |
| INT4 | ~17GB | ~30 tok/s | <1% |
| NVFP4 | ~16GB | ~35 tok/s | 0.25% (Google's claim) |
| Q4_K_M | ~18.5GB | ~27 tok/s | ~0.5% |

**Recommendation**: Use Q4_K_M via llama.cpp or INT4-AWQ via vLLM. Fits with ~6GB headroom for KV cache (sufficient for 4K-8K context).

### Model Selection for UAV Research

| Use Case | Recommended Model | Reason |
|----------|------------------|--------|
| General reasoning | Gemma 4 31B (INT4) | Highest Arena ELO in budget |
| Vision understanding | Gemma 4 31B (native) | Single model for text+vision |
| Fast inference | Qwen 3 30B-A3B (MoE) | 3.3B active, ~300 tok/s |
| Edge deployment | SmolVLM 2B | 2.6GB VRAM, real-time |
| VLA baseline | OpenVLA 7B | Only open-source VLA |
| Coding/planning | Qwen 3 32B (INT4) | Strong code generation |
