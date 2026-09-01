# SASS extraction note

Full `cuobjdump --dump-sass --dump-resource-usage` outputs (~5.7 GB each) were
generated and verified for the Stage 2 report but are NOT committed (too large
for git). The committed per-symbol SASS sections are:
- ctrl-k2-fp32.txt / ctrl-k3-fp32.txt (control lib)
- ma-k2-fp16.txt / ma-k2-fp32.txt / ma-k3-fp16.txt / ma-k3-fp32.txt (mixedacc lib)

The per-symbol resource usage (REG/STACK/LOCAL/SHARED) is committed separately
in `resource-usage.txt`, produced by:
  cuobjdump --dump-sass --dump-resource-usage \
    -fun <mangled-symbol> build-cuda-exp04-stage2-{control,mixedacc}/bin/libggml-cuda.so.0.19.0

Regenerate full dumps anytime with:
  cuobjdump --dump-sass --dump-resource-usage \
    build-cuda-exp04-stage2-{control,mixedacc}/bin/libggml-cuda.so.0.19.0
