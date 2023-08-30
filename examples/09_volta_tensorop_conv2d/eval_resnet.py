import torch
import subprocess
from statistics import stdev
import re 
import json

hw = {
    64: {"h": 56, "w": 56},
    128: {"h": 28, "w": 28},
    256: {"h": 14, "w": 14},
    512: {"h": 7, "w": 7}
}
tiles = {
    1:  {64: {"TileSizes": [64, 64, 32, 32, 32, 32], 
              "cusync": {"split_k_slices":2}, "baseline": {"split_k_slices":1}},
        128: {"TileSizes": [64, 64, 32, 32, 32, 32], 
              "baseline": {"split_k_slices":2}},
        256: {"TileSizes": [64, 64, 32, 32, 32, 32], 
        "baseline": {"split_k_slices":4}},
        512: {"TileSizes": [64, 64, 32, 32, 32, 32], "baseline": {"split_k_slices":8}}},
    4:  {64: {"TileSizes": [64, 64, 32, 32, 32, 32], 
            "cusync": {"split_k_slices":1}, "baseline": {"split_k_slices":1}},,
        128: {"TileSizes": [64, 64, 32, 32, 32, 32], 
        "baseline": {"split_k_slices":3}},
        256: {"TileSizes": [64, 64, 32, 32, 32, 32], 
        "baseline": {"split_k_slices":2}},
        512: {"TileSizes": [64, 64, 32, 32, 32, 32], "baseline": {"split_k_slices":8}}},
    # 6:  {64: {"TileSizes": [64, 64, 32, 32, 32, 32]},
    #     128: {"TileSizes": [64, 64, 32, 32, 32, 32]},
    #     256: {"TileSizes": [64, 64, 32, 32, 32, 32]},
    #     512: {"TileSizes": [64, 64, 32, 32, 32, 32]}},
    8:  {64: {"TileSizes": [64, 64, 32, 32, 32, 32], 
            "cusync": {"split_k_slices":1}, "baseline": {"split_k_slices":1}},
        128: {"TileSizes": [64, 64, 32, 32, 32, 32], 
        "baseline": {"split_k_slices":1}},
        256: {"TileSizes": [64, 64, 32, 32, 32, 32], 
        "baseline": {"split_k_slices":3}},
        512: {"TileSizes": [64, 64, 32, 32, 32, 32], "baseline": {"split_k_slices":8}}},
    12: {64: {"TileSizes": [64, 64, 32, 32, 32, 32], 
            "cusync": {"split_k_slices":1}, "baseline": {"split_k_slices":1}},
        128: {"TileSizes": [64, 64, 32, 32, 32, 32],
               "baseline": {"split_k_slices":1}},
        256: {"TileSizes": [64, 64, 32, 32, 32, 32], 
               "baseline": {"split_k_slices":2}},
        512: {"TileSizes": [64, 64, 32, 32, 32, 32], "baseline": {"split_k_slices":8}},
    16: {64: {"TileSizes": [64, 64, 32, 32, 32, 32], 
            "cusync": {"split_k_slices":1}, "baseline": {"split_k_slices":1}},
        128: {"TileSizes": [64, 64, 32, 32, 32, 32], 
            "baseline": {"split_k_slices":1}},
        256: {"TileSizes": [64, 64, 32, 32, 32, 32], 
        "baseline": {"split_k_slices":3}},
        512: {"TileSizes": [64, 64, 32, 32, 32, 32], "baseline": {"split_k_slices":8}},
    20: {64: {"TileSizes": [64, 64, 32, 32, 32, 32],
            "cusync": {"split_k_slices":1}, "baseline": {"split_k_slices":1}},
        128: {"TileSizes": [64, 64, 32, 32, 32, 32], 
        "baseline": {"split_k_slices":1}},
        256: {"TileSizes": [64, 64, 32, 32, 32, 32],
             "baseline": {"split_k_slices":2}},
        512: {"TileSizes": [64, 64, 32, 32, 32, 32], "baseline": {"split_k_slices":2}},
    24: {64: {"TileSizes": [64, 64, 32, 32, 32, 32],
            "cusync": {"split_k_slices":1}, "baseline": {"split_k_slices":1}},
        128: {"TileSizes": [64, 64, 32, 32, 32, 32], 
        "baseline": {"split_k_slices":1}},
        256: {"TileSizes": [64, 64, 32, 32, 32, 32], 
        "baseline": {"split_k_slices":2}},
        512: {"TileSizes": [64, 64, 32, 32, 32, 32],"baseline": {"split_k_slices":3}},
    28: {64: {"TileSizes": [64, 64, 32, 32, 32, 32], 
            "cusync": {"split_k_slices":1}, "baseline": {"split_k_slices":1}},
        128: {"TileSizes": [64, 64, 32, 32, 32, 32], 
        "baseline": {"split_k_slices":1}},
        256: {"TileSizes": [64, 64, 32, 32, 32, 32], "baseline": {"split_k_slices":2}},
        512: {"TileSizes": [64, 64, 32, 32, 32, 32], "baseline": {"split_k_slices":3}},
    32: {64: {"TileSizes": [64, 64, 32, 32, 32, 32], 
            "cusync": {"split_k_slices":1}, "baseline": {"split_k_slices":1}},
        128: {"TileSizes": [64, 64, 32, 32, 32, 32], 
        "baseline": {"split_k_slices":1}},
        256: {"TileSizes": [64, 64, 32, 32, 32, 32],"baseline": {"split_k_slices":2}},
        512: {"TileSizes": [64, 64, 32, 32, 32, 32], "baseline": {"split_k_slices":3}}
}

def getAllTimes(s, START, END):
  alltimes = {}
  assert START in s
  assert END in s
  s = s[s.find(START):s.find(END)]
  s = s[s.find("\n"):]
  alljsons = []
  for l in re.findall(r".+", s):
    j = json.loads(l)
    alljsons += [j]
  
  def sortkey(elem):
    return elem["Total"]
  
  alljsons.sort(key=sortkey)
  p = 0.9
  alljsons = alljsons[:int(len(alljsons)*0.9)]
  for j in alljsons:
    for k in j:
      if k not in alltimes:
        alltimes[k] = [] 
      alltimes[k] += [float(j[k])]

  return alltimes

def avg(l):
  return sum(l)/len(l)

for c in [64, 128, 256, 512]:
  for m in [1, 4, 8,12, 16, 20, 24, 28, 32]:
    (s, o) = subprocess.getstatusoutput(f"python3 torchconv2d.py {m} {c}")
    if s != 0:
        print(o)
    else:
        torchTime = float(o)
    
    for syncType in [0]:
      command = f"./a.out --n={m} --h={hw[c]['h']} --w={hw[c]['w']} --c={c} --k={c} --r=3 --s=3 --split_k_slices={tiles[m][c]['split_k_slices']} --syncType={syncType}"
      (s, o) = subprocess.getstatusoutput(command)
      baselineTimes = getAllTimes(o, "START-BASELINE", "END-BASELINE")
      overlapTimes = getAllTimes(o, "START-OVERLAP", "END-OVERLAP")
      bTimes = baselineTimes["Total"]
      oTimes = overlapTimes["Total"]
      gemm_problem_size = re.findall(r"gemm problem size:(.+)", o)[0]
      threadblocks = re.findall(r"Number of thread blocks for both convs: (.+)", o)[0]
      print(f"{m} & {c} & {'Row-Sync' if syncType == 1 else 'Tile-Sync'} & {gemm_problem_size.strip()} & {threadblocks.strip()} & {'%.2f'%torchTime} & {'%.2f'%avg(bTimes)} & {'%.2f'%stdev(bTimes)} & {'%.2f'%avg(oTimes)} & {'%.2f'%stdev(oTimes)} & {'%.2f'%((1-avg(oTimes)/avg(bTimes))*100)}")
      