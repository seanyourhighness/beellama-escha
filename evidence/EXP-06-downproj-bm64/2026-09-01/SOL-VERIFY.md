# EXP-06 independent Gate 3 VERIFY

VERIFY=CONFIRM

CHECKS:

- Target calculation: PASS. Re-running the checked-in summarizer over the two blocking logs reproduces `profile-comparison.json` byte-for-byte (SHA-256 `ddf72f1b...`). It discards the first four observations per family, so the target medians use 60 of 64 completed target calls: control 2.25125 ms, candidate 3.30810 ms, `(3.30810/2.25125-1)*100 = +46.94503%` slower. The required target gain was at least 15%; this is an unambiguous fail. The successful non-profile smoke independently gives `(2275.93/2457.93-1)*100 = -7.40460%`. It corroborates rejection but, as disclosed, is one run per arm and is not the preregistered full-wall ABBA gate.
- Route accounting: PASS. Both raw blocking logs contain exactly 400 `ESCHA_PROFILE` records. Control has 64 exact K3/IC17408/OC5120/M2048 `mma-fp32-mixedacc` calls; candidate has 64 exact `mma-down-bm64-exp` calls. All 64 experimental tags match the approved target. The other 336 calls have identical shape/count/tag distributions across arms. Gate 1 and Gate 2 constrain the candidate to BM128->64 with BN128/WN2, FP32, 256 threads, exact M2048, and no other structural variable.
- Profile abort framing: PASS. In both raw blocking logs the 400th profile is line 402 and the CUDA error starts on line 403. The same post-completion abort in frozen control and candidate is correctly characterized as inherited harness behavior. These event timings are diagnostic route/target-family evidence only, not a successful full-wall run.
- Resource/SASS: PASS within stated static scope. All six frozen binary hashes still verify. Focused macro-off candidate BM128 SASS is byte-identical to frozen control (`cmp` success; both SHA-256 `ed103498...`). Candidate BM64 focused SASS is `a4e17ce...`, uses FP32 HMMA, shows 92 registers, STACK=0, LOCAL=0, no local-store instructions, and lowers the A-stage async copy to 64-bit versus control 128-bit. Static shared is 1024 B; planned launch dynamic shared is 9728 B. Static A-stage 1024/1024 and FP32 output-store 8192/8192 coverage proofs have no missing, duplicate, or OOB writes.
- Rollback/source equality: PASS. HEAD is exactly `eb66791590696733facb5b1f573e4b923adfa4c0`. `git diff 4bc1afc1d -- ggml/src/ggml-cuda/escha-moe.cu` is empty. `cf53d803c` adds only the guarded BM64 implementation plus static proof files; `eb6679159` reverses that source hunk exactly while retaining evidence/docs. Current source is therefore the promoted EXP-04 Stage 2 mixed-accumulator control.
- Evidence scope/exclusions: PASS. No binary, weight/GGUF, venv, build directory, result tree, or full cuobjdump is committed in the EXP-06 closure. Only focused SASS/resources, hashes, logs, scripts, report, provenance, docs, and the exact source rollback are committed. The full cuobjdump remains outside git at `/tmp/escha-wheel/exp06-full-cuobjdump` and exists at review time.
- Docs: PASS. `docs/current-state.md` and `docs/escha-prefill-experiment-ledger.md` accurately record rejection, rollback, diagnostic-only profile evidence, the corroborating smoke, omitted downstream gates, and the retained Stage 2 control.

DECISION: REJECT+REVERT valid. Candidate must not be promoted. The preregistered target-family fail-fast gate alone decides rejection; stopping before ABBA/depth/P2/P7/decode/quality was correct.

NEXT TARGET: Execute the separately Sol-planned output-finalize fusion experiment in `evidence/EXP-04-nextvar/2026-09-01/NEXTVAR-PLAN.md`; do not combine it with BM64 geometry.

Exact gaps: no clean profiled full-wall run; no dynamic occupancy/residency counters; no end-to-end numerical, P2/P7, decode, or quality certification; and no qualified nine-pair full-wall result. These are deliberate fail-fast exclusions and cannot support promotion, but none weakens the negative target result. Minor reporting nuance: JSON `*_samples=64` is the raw completed count, while each median uses 60 after the first four per-family observations are discarded.

VERDICT=CONFIRM
