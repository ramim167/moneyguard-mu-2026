# GuardLM on a consumer NVIDIA GPU

GuardLM was first trained with MLX on Apple Silicon. This path trains the same
LoRA adapter on Windows or Linux with PyTorch + PEFT on a single consumer GPU,
and evaluates it on the same held-out test set with the same scoring code
(`moneyguard/ml/guardlm_eval.py`, imported rather than rewritten).

## Measured run (evidence: `evidence/ml-cuda/`)

| Setting | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Ti, 8 GB (peak use 5.2 GB) |
| Base model | `Qwen/Qwen3-1.7B` at revision `70d244cc`, bf16 |
| Adapter | LoRA rank 16, scale 20, last 16 layers, ~10M trainable parameters |
| Training | 450 iterations, batch 4 (1,800 examples seen), **7.3 minutes** |
| Validation loss | 0.465 → 0.0085 |
| Test set | full held-out split: 300 planning + 200 extraction examples, counterparty names unseen in training |

| Held-out metric | Base Qwen3-1.7B | GuardLM (CUDA) |
|---|---|---|
| Exact tool plan (every tool and argument) | 21.3% | **92.7%** |
| Intent understood | 13.3% | **94.7%** |
| Correct tools chosen | 37.3% | **95.7%** |
| Record extraction F1 | 0.304 | **0.988** |
| Field accuracy | 67.5% | **99.5%** |
| Hidden-instruction (injection) flag accuracy | 62.0% | **100%** |

Greedy decoding, Qwen3 thinking disabled, batched generation (about 0.25 s per
example). Raw outputs for every example are in `evidence/ml-cuda/*_outputs.jsonl`
and adapter SHA-256 hashes are in `guardlm_cuda_eval.json`.

## Reproduce

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers peft accelerate safetensors
python -m moneyguard.ml.guardlm_data                    # writes data/guardlm/{train,valid,test}.jsonl
python -m moneyguard.ml.guardlm_cuda_train              # adapter -> ~/moneyguard-models/guardlm-cuda-adapter
python -m moneyguard.ml.guardlm_cuda_eval               # base model
python -m moneyguard.ml.guardlm_cuda_eval --adapter ~/moneyguard-models/guardlm-cuda-adapter
```

To use it from the app, serve it on the loopback interface and point MoneyGuard's
local-model setting at it:

```bash
python scripts/serve_guardlm_cuda.py --port 8796
MONEYGUARD_LOCAL_MODEL_URL=http://127.0.0.1:8796/v1 MONEYGUARD_LOCAL_TEXT_MODEL=guardlm-cuda-qwen3-1.7b python run.py
```

The model still has no authority: every plan passes the same validator, replies
are written from tool output, and extracted records stay drafts until a person
confirms them.

## Scope

Synthetic, generated training and test data; these figures measure the task the
generator defines, not real-world accuracy. The base differs from the released
MLX adapter (Qwen3-4B), so the two sets of scores are separate evaluations. The
weights are not committed to the repository.
