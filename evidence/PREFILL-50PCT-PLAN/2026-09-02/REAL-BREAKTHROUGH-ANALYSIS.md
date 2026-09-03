# The real breakthrough — what the arithmetic says (2026-09-02)

## The pivotal number nobody checked until now

The blk.0 ffn_gate K2 direct-op is 182.5 G-MAC = 89.13M m16n8k16 HMMAs.
Measured on RTX 5090 (170 SM, ~1.9 GHz):

| kernel | direct-op | cycles/HMMA/SM | relative |
|---|---:|---:|---:|
| Bee shared-B control | 1.591 ms | 5.77 | 1.00× |
| **Official escham_code_gemm** | ~1.31 ms | **4.75** | **1.21× faster** |

**The official kernel — the reference implementation of this exact decode —
is only 21% faster than our control on identical tensor-core work.** Both emit
`HMMA.16816.F16`. This matches the EXP-05 audit ("official 0.78–0.90× Bee" on
gate/up) and explains why the official full-model number is ~3030, not 3500+.

## What this proves about V1–V5 (and the overlap thesis)

Sean's hypothesis — "double or quadruple the stage overlay and we have the
solution" — was tested by V4 (2×/4× B-ring) and V5 (producer/consumer). V5 got
to 1.28× (closest yet). But the arithmetic shows why deeper overlap can't
produce a 2× breakthrough:

**The control kernel is already within ~21% of the reference kernel's MMA
throughput.** There is no hidden 2× in decode restructuring, because the decode
isn't the binding cost at the tensor level — the control's decode-once scheme
already hides it. V1–V5 all ADDED work on top of a near-MMA-bound kernel, which
is why they were all slower.

## The two real paths to your goal

### Path A — match the official kernel exactly (packed artifact → ~3030)
Target is NOT "beat control by 15%." Target is **reach official's 4.75
cyc/HMMA = +21% over control = ~3030 tok/s on the PACKED artifact** — which is
literally "meets escha W2 in SGLang in speed" for the exact-packed model.
V3 approximated the official structure (4-K16, 64 HMMA, 45 KiB) and got 2.17×
slower — so the real official has details V3 missed (its exact warp-local
decode pattern, LDS schedule, issue pattern). This requires the forensic SASS
reconstruction that the Codex cap cut short. Feasible, bounded (+21%), and it
would make the canonical 8.6 GB artifact meet SGLang parity.

### Path B — the hybrid (exceeds 3030 toward native)
The frontier shows the ONLY way past the packed ceiling is standardizing body
families. NEW FINDING from the measured frontier:
- FFN std → 2962 (8.48 GB)
- +gate Q2_K → **3023 (8.599 GB) = SGLang parity AT the ~8.5 GB target**
- +embed → 3228 (8.808 GB)
- **+full-attn std → 3307 (9.036 GB) — the winner**
- +linear-attn std → 3283 (9.345 GB) — **the 48 linear QKV/SSM standardizations
  add 0.31 GB with ZERO prefill benefit**

## Sean's stated goal, checked against the data

"Smaller escha hybrid around 8.5 GB that meets original escha W2 in SGLang in
speed" → **P-ARCH-23: 8.599 GB / 3,023 tok/s already MEETS this** (SGLang
parity at the target size). That is the deliverable TODAY.

"If we can beat that toward native GGUF speeds" → the only evidence-backed step
is full-attention standardization (23I-step: 9.036 GB / 3,307 tok/s), NOT
linear-attention (proven waste).

## Recommendation

1. Bank **P-ARCH-23 (8.599 GB / 3,023 tok/s)** as the goal-fit release — it
   meets your exact stated spec.
2. If the packed artifact itself must reach SGLang parity: fund the **official
   forensic port** (Path A) — bounded +21%, target = official's 4.75 cyc/HMMA,
   NOT a mythical 2×.
3. If 3300-class is the bar: standardize full-attention (Path B, 23I-step) and
   drop the linear-attn standardization as confirmed waste.
