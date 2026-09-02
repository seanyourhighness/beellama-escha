# EXP-11 Attempt-2 Slice-1 native transcoder

This directory is deliberately self-contained.  It does not link BeeLlama,
modify model-loader code, or invoke llama binaries.  The tool accepts only the
funded canonical source GGUF and only builds the layer-0 gate/up/down Q2_K
shard authorized by `REVISION-AND-SLICE1-PLAN.md`.

Build and run the Slice-1 evidence harness:

```bash
cmake -S tools/escha_native_transcode -B build-escha-native-slice1
cmake --build build-escha-native-slice1 -j
python3 tools/escha_native_transcode/slice1_test.py \
  --binary build-escha-native-slice1/escha-native-transcode \
  --evidence evidence/EXP-11-transcode-cache/2026-09-02/attempt-2/slice-1
```

Publication is `layers/blk.000.gguf.tmp.<pid>` followed by file fsync,
read-back structural/hash verification, atomic rename, and an independently
fsynced/renamed `blk.000.receipt.json`.  Resume trusts neither temporary files
nor a receipt alone: both the published shard and receipt hashes are checked.

