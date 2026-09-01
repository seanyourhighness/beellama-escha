#!/usr/bin/env python3
"""Summarize identical 400-call ESCHA_PROFILE harness runs; profile abort is shared
by control/candidate after all calls, so per-call completed timing samples remain
explicitly labeled diagnostic, not a successful full-wall run."""
import json, re, statistics, sys
pat=re.compile(r'ESCHA_PROFILE k=(\d+) ic=(\d+) oc=(\d+) rows=(\d+) gen=(\d+) route=([^ ]+) total_ms=([\d.]+) rotate_ms=([\d.]+) matmul_ms=([\d.]+) epilogue_ms=([\d.]+)')
def load(path):
 d={}
 for line in open(path):
  m=pat.search(line)
  if not m: continue
  k,ic,oc,rows,gen,route,*vals=m.groups()
  key=(int(k),int(ic),int(oc),int(rows),route)
  d.setdefault(key,[]).append({'total_ms':float(vals[0]),'rotate_ms':float(vals[1]),'matmul_ms':float(vals[2]),'epilogue_ms':float(vals[3])})
 return d
control=load(sys.argv[1]); candidate=load(sys.argv[2])
out={'control_completed_profiles':sum(map(len,control.values())),'candidate_completed_profiles':sum(map(len,candidate.values())),'families':[],'caveat':'Both runs abort only after completing exactly 400 ESCHA_PROFILE calls; these per-call CUDA-event samples are diagnostic route/timing evidence, not a clean successful full-wall benchmark.'}
# map candidate exp down route to control fp32 route only; other matching by shape
for ck,cvals in control.items():
 k,ic,oc,rows,route=ck
 cand_key=next((x for x in candidate if x[:4]==(k,ic,oc,rows)),None)
 if cand_key is None: continue
 x=cvals[4:] if len(cvals)>4 else cvals
 y=candidate[cand_key][4:] if len(candidate[cand_key])>4 else candidate[cand_key]
 cm=statistics.median(z['matmul_ms'] for z in x); dm=statistics.median(z['matmul_ms'] for z in y)
 out['families'].append({'shape':{'K':k,'IC':ic,'OC':oc,'M':rows},'control_route':route,'candidate_route':cand_key[4], 'control_samples':len(cvals),'candidate_samples':len(candidate[cand_key]),'control_matmul_median_ms':cm,'candidate_matmul_median_ms':dm,'candidate_vs_control_pct':100*(dm/cm-1)})
print(json.dumps(out,indent=2,sort_keys=True))
