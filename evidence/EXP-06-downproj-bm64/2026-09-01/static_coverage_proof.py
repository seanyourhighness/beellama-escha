#!/usr/bin/env python3
"""EXP-06 static coverage proof for the Sol-approved BM64 instantiation.
Mirrors the NVIDIA float tile<16,8,float>::get_i/get_j mapping from mma.cuh
and the exact A-stage cp_m/cp_h formulas in escha-moe.cu. No GPU required.
"""
import collections, json

def a_map(bm):
    cpb = bm * 16 * 2 // 256
    hits = collections.Counter()
    for tid in range(256):
        cp_m = tid // (16*2 // cpb)
        cp_h = (tid % (16*2 // cpb)) * (cpb//2)
        for h in range(cp_h, cp_h + cpb//2): hits[(cp_m,h)] += 1
    expected={(m,h) for m in range(bm) for h in range(16)}
    return {'BM':bm,'CPB':cpb,'elements':len(hits),'expected':len(expected),
            'min_writes':min(hits.values()),'max_writes':max(hits.values()),
            'missing':len(expected-set(hits)),'out_of_range':len(set(hits)-expected)}

def get_i(l,lane): return (l//2)*8 + lane//4
def get_j(l,lane): return (lane%4)*2 + (l%2)
def fragment_map(bm=64,bn=128):
    # WN=2, WM=4, MT=BM/16/WM=1, NTT=BN/8/WN=8, tile_c.ne=4.
    WN=2; WM=8//WN; MT=bm//16//WM; NTT=bn//8//WN; hits=collections.Counter()
    for warp in range(8):
        wm,wn=divmod(warp,WN)
        for lane in range(32):
            for i in range(MT):
                for j in range(NTT):
                    for l in range(4):
                        m=wm*(16*MT)+i*16+get_i(l,lane)
                        n=wn*(8*NTT)+j*8+get_j(l,lane)
                        hits[(m,n)] += 1
    expected={(m,n) for m in range(bm) for n in range(bn)}
    return {'BM':bm,'BN':bn,'WN':WN,'WM':WM,'MT':MT,'NTT':NTT,'tile_c_ne':4,
            'elements':len(hits),'expected':len(expected),'min_writes':min(hits.values()),
            'max_writes':max(hits.values()),'missing':len(expected-set(hits)),'out_of_range':len(set(hits)-expected)}
result={'a_stage_BM128':a_map(128),'a_stage_BM64':a_map(64),'fp32_fragment_store_BM64':fragment_map()}
for r in result.values():
    assert r['missing']==0 and r['out_of_range']==0 and r['min_writes']==1 and r['max_writes']==1, r
print(json.dumps(result,indent=2,sort_keys=True))
