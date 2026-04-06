#!/bin/bash
# Run llmfit hardware compatibility scan and save results

echo "=== llmfit Hardware Scan ==="
echo "Date: $(date)"
echo ""

# Full scan
llmfit --cli > ../results/llmfit/full_scan.txt 2>&1
echo "Full scan saved to results/llmfit/full_scan.txt"

# Perfect fits only
llmfit fit --perfect -n 30 --cli > ../results/llmfit/perfect_fits.txt 2>&1
echo "Perfect fits saved to results/llmfit/perfect_fits.txt"

# Specific model plans
for model in \
    "google/gemma-4-31B" \
    "google/gemma-3-27b-it" \
    "Qwen/Qwen3-32B" \
    "Qwen/Qwen3-30B-A3B" \
    "meta-llama/Llama-3.2-11B-Vision-Instruct" \
    "mistralai/Mistral-Small-24B-Instruct-2501"; do
    echo ""
    echo "--- Plan for: $model ---"
    llmfit plan "$model" --context 8192 2>/dev/null
done
