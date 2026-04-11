# Aspect-Based Opinion Extraction

## 1. Project deliverable information

- **Authors:** Paul-Emile Barranger, Valentin Dugay, Clément Callaert
- **Course:** NLP Course @ CentraleSupélec (2025–2026)
- **Branch:** The complete code for this implementation is located on the *finetuning lora* branch.

## 2. Project overview

This repository implements an **Aspect-Based Sentiment Analysis (ABSA)** system for French restaurant reviews. The objective is to extract the overall opinion across three predefined aspects: **Price**, **Food**, and **Service**. The valid classes are **Positive**, **Negative**, **Mixed**, and **No Opinion**.

## 3. Methodology: Qwen3-4B LoRA fine-tuning

We selected `Qwen/Qwen3-4B` as our base causal language model due to its optimal parameter-to-performance ratio for instruction-following and JSON formatting tasks.

### 3.1. Architectural choices & PEFT

To adapt the model to the ABSA classification task efficiently, we utilized **Parameter-Efficient Fine-Tuning (PEFT)** via **LoRA** (Low-Rank Adaptation).

- **Target modules:** LoRA is applied to all linear projections within the attention mechanism and MLP blocks (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- **Hyperparameters:** rank *r* = 16 and *α* = 32 were chosen to maximize the adaptation subspace while maintaining training stability.

### 3.2. Hardware optimization & constraints handling

The assignment enforces a strict `per_device_train_batch_size = 1` for 4B models and prohibits the use of external custom kernels like flash-attn. To strictly minimize wall-clock execution time, we implemented the following optimization stack:

- **Native SDPA:** PyTorch 2.9.1 Scaled Dot-Product Attention (`attn_implementation="sdpa"`) is used for memory-efficient attention.
- **Precision & compute:** The model weights are loaded in native `bfloat16`. TensorFloat-32 (TF32) matrix multiplications are enabled (`torch.backends.cuda.matmul.allow_tf32 = True`) to accelerate tensor core throughput on Ada Lovelace/Blackwell architectures.
- **Overhead reduction:** We use `adamw_torch_fused` to minimize CUDA kernel launches and asynchronous data loading (`dataloader_num_workers=4`, `pin_memory=True`) to reduce host-to-device transfer bottlenecks.

### 3.3. Inference & robust parsing

During evaluation we use greedy decoding (`temperature=0.0`, `do_sample=False`) for deterministic outputs. To avoid pipeline failures from malformed LLM output, we implemented a zero-trust extraction pipeline:

1. Regex-based JSON block isolation.
2. Fallback regex literal matching for malformed JSON strings.

## 4. Results

Macro-average accuracy on dev data: 

## 5. Running the project

The environment requires **torch** 2.9.1, **transformers** 5.5.0, **trl** 1.0.0, **peft** 0.18.1, and **accelerate** 1.13.0 as per the strict assignment guidelines.

To launch the training and evaluation pipeline, navigate to the `src` directory and run:

```bash
accelerate launch runproject.py --ollama_model=qwen3:4b
```

