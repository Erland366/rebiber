# AGENTS.md — Repository Guidelines

## 1. Purpose (Local Truth)

This repository is for **ablation studies on Softpick Transformers**.

The primary deliverable is a **correct, fast Triton implementation of the Softpick attention algorithm** (a modified softmax), located under:

```
fla/ops/attn/
```

This kernel must:

* Match a pure PyTorch reference implementation
* Be validated via unit tests
* Be exercised in end-to-end training using the `flame/` framework

This repo is **research-oriented**, but held to **production-grade correctness and performance standards**.

---

## 2. Project Structure and Documentation

* `fla/ops/attn/`
  **Core focus**. Triton Softpick kernels + PyTorch reference paths.

* `configs/`
  Training/model configs.
  Primary ablation config:
  `configs/softpick_transformer_1B.json`

* `flame/`
  Training framework. Entry point:
  `python -m flame.train` (via `bash train.sh`)

* `tests/`
  Correctness tests (`pytest`). Keep tests small, deterministic, and targeted.

* `3rdparty/`
  Git submodules (`flash-linear-attention`, `torchtitan`, etc.).
  Treat as upstream snapshots; do not modify casually.

* `resources/`, `exp/`, `downloaded_*`
  Experimental artifacts or outputs. Do **not** commit large generated files.

This document serves as the high-level entry point. Detailed technical documentation for each module has been moved to `docs/<project_name>/` to keep this blueprint manageable.

### 2.1. Documentation Policy for New Features

*   **New Features:** When adding a new feature, you **must** document it in detail.
*   **Where to Document:**
    *   If the feature fits within an existing module, add it to the corresponding `.md` file in `docs/copus/`.
    *   If it is a completely new module, create a new file (e.g., `docs/<project_name>/NEW_MODULE.md`).
*   **Update the Index:** Always update this section (Section 2 of this .md file) to reference the new or updated documentation.

### 2.2. Module Documentation Links
This will be updated during development

*   **[Experiments & Roadmap](docs/<project_name>/EXPERIMENTS_AND_ROADMAP.md)**
    *   *Contents:* Planned experiments, workflows, development roadmap.
*   **[Research Hypothesis](docs/<project_name>/RESEARCH_HYPOTHESIS.md)**
    *   *Contents:* Scientific motivation.

---

---

## 3. Build, Test, and Development

### Setup

```bash
pip install -e .
git submodule update --init --recursive
```

### Tests

```bash
pytest -q
# example
pytest -q tests/test_naive_relusoftpick.py
```

### Training

```bash
bash train.sh --model.config configs/softpick_transformer_1B.json ...
```

---

## 4. Kernel Implementation & Testing Expectations

When working in `fla/ops/attn/`:

**Required invariants**

* Triton implementation + pure PyTorch reference
* Forward and backward correctness
* Causal masking correctness
* Safe version compared with unsafe version correctness 
* Both layouts: `head_first=True/False`

**Test coverage**

* dtypes: `fp16`, `bf16`, `fp32` (where supported)
* shapes: small `t`, odd `d`, non-power-of-two sizes
* Use `torch.testing.assert_close` with explicit `rtol` / `atol`

Correctness comes **before** performance; performance work must never regress correctness.

---

## 5. Training & Parallelism Constraints (AMD NUMA)

This machine is an **8-GPU AMD node with two NUMA groups**:

* **Group A**: GPUs `0,1,2,3`
* **Group B**: GPUs `4,5,6,7`

**Rules**

* Never mix GPUs across NUMA groups
* Each experiment uses **≤ 4 GPUs**
* Each concurrent run must use a **unique `MASTER_PORT`** and `--job.dump_folder`

**Recommended launches**

```bash
# Group A
HIP_VISIBLE_DEVICES=0,1,2,3 NGPU=4 MASTER_PORT=29501 \
bash train.sh --model.config configs/softpick_transformer_1B.json ...

# Group B
HIP_VISIBLE_DEVICES=4,5,6,7 NGPU=4 MASTER_PORT=29502 \
bash train.sh --model.config configs/softpick_transformer_1B.json ...
```

---

## 6. Engineering Ethos (Imported, Scoped)

This repo follows the same **craftsmanship principles** as the larger MoE / systems work:

* **One clear path per use-case** — no duplicate stacks
* **Explicit over magical** — control flow and contracts must be obvious
* **Hot paths first** — kernels and inner loops stay lean and measurable
* **Fail fast, fail loud** — no silent fallbacks
* **Minimal dependencies** — add surface area only if it improves clarity *and* performance
* **Reproducibility matters** — configs and provenance must allow reruns months later

Ask before introducing:

* New config formats
* Alternate execution paths
* Hidden environment-dependent behavior

---

## 7. Coding, Commits, and PRs

* Python 3.x, 4-space indent, `snake_case`
* Prefer extending existing modules over creating new ones

**Commits**

* Single-line, imperative subject (`Add …`, `Fix …`, `Refactor …`)
* No “written by AI” markers
* Include reproduction commands when touching kernels or training

---

## 8. Agent / AI Operating Rules

* Read files fully before modifying them; avoid partial context
* Do not introduce second execution paths or fallback behavior
* If you change how something works, update documentation in the same change
* Tests are mandatory for kernel or numerical changes
* Prefer deleting complexity over adding abstractions

If a change does **not** improve:

* correctness,
* performance,
* or experimental clarity,

…it likely does not belong.

---

**Scope reminder:**
This repo exists to answer *specific architectural and kernel questions* about Softpick attention — do that one thing exceedingly well.
