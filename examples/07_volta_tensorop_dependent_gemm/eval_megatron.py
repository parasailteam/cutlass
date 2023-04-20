import subprocess
import re

baselineTimes = {}
cublasTimes = {}
overlappedTimes = {}
minimumTimes = {}
speedup = {}
maxspeedup = {}

# Dictionary of tile sizes for each M
tiles = {
  2048: {
    "TileSizes" : [256, 128, 32, 128, 64, 32], "SoftmaxRowTile" : 8, "MaxTBsPerSM": 2, "Best-Policy": "Row-Sync",
    6144: {"split_ks": [2,1]},
    8192: {"split_ks": [1,1]},
    10240: {"split_ks": [1,1]},
    12288: {"split_ks": [1,1]},
    16384: {"split_ks": [1,1]},
    20480: {"split_ks": [1,1]},
    25600: {"split_ks": [1,1]},
  },
  1024: {"TileSizes" : [256, 128, 32, 128, 64, 32], "split_ks": [1,1], "MaxTBsPerSM": 2, "SoftmaxRowTile": 2, "Best-Policy": "Row-Sync",
    6144: {"split_ks": [2,1]},
    8192: {"split_ks": [1,1]},
    10240: {"split_ks": [1,1]},
    12288: {"split_ks": [1,1]},
    16384: {"split_ks": [1,1]},
    20480: {"split_ks": [1,1]},
    25600: {"split_ks": [1,1]},
  },
  512: {"TileSizes" : [256, 128, 32, 128, 64, 32], "SoftmaxRowTile": 1, "MaxTBsPerSM": 2, "Best-Policy": "Row-Sync",
    6144: {"split_ks":   [2,1]},
    8192: {"split_ks":   [2,1]},
    10240: {"split_ks":  [2,1]},
    12288: {"split_ks":  [2,1]},
    16384: {"split_ks":  [2,1]},
    20480: {"split_ks":  [2,1]},
    25600: {"split_ks":  [2,1]},
  },
  256: {"TileSizes" : [256, 128, 32, 128, 64, 32], "SoftmaxRowTile": 1, "MaxTBsPerSM": 2, "Best-Policy": "Tile-Sync",
    6144: {"split_ks":   [4,1]},
    8192: {"split_ks":   [4,1]},
    10240: {"split_ks":  [4,1]},
    12288: {"split_ks":  [4,1]},
    16384: {"split_ks":  [4,1]},
    20480: {"split_ks":  [4,1]},
    25600: {"split_ks":  [4,1]},
  },
  128: {"TileSizes" : [128, 128, 32, 64, 64, 32], "SoftmaxRowTile": 1, "MaxTBsPerSM": 2, "Best-Policy": "Tile-Sync",
    6144: {"split_ks":   [4,1]},
    8192: {"split_ks":   [4,1]},
    10240: {"split_ks":  [4,1]},
    12288: {"split_ks":  [4,1]},
    16384: {"split_ks":  [4,1]},
    20480: {"split_ks":  [4,1]},
    25600: {"split_ks":  [4,1]},},
  64: {"TileSizes" : [64, 128, 32, 32, 64, 32], "SoftmaxRowTile": 1, "MaxTBsPerSM": 3, "Best-Policy": "Tile-Sync",
    6144: {"split_ks":   [4,1]},
    8192: {"split_ks":   [4,1]},
    10240: {"split_ks":  [4,1]},
    12288: {"split_ks":  [4,1]},
    16384: {"split_ks":  [4,1]},
    20480: {"split_ks":  [4,1]},
    25600: {"split_ks":  [4,1]}},
  32: {"TileSizes" : [64, 128, 32, 32, 64, 32], "split_ks": [4,1]},
  16: {"TileSizes" : [64, 128, 32, 32, 64, 32], "split_ks": [4,1]},
  8: {"TileSizes" : [64, 128, 32, 32, 64, 32], "split_ks": [4, 1]},
  4: {"TileSizes" : [64, 128, 32, 32, 64, 32], "split_ks": [4, 1]},
  2: {"TileSizes" : [64, 128, 32, 32, 64, 32], "split_ks": [4,1]},
  1: {"TileSizes" : [64, 128, 32, 32, 64, 32], "split_ks": [4,1]}
}

import json
from statistics import stdev

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

#SelfAttention
for h in [6144, 8192, 10240, 12288, 16384, 20480, 25600]:
  m = 64

  (s, o) = subprocess.getstatusoutput(f"python3 torchAttention.py {m} {int(h/8)} {h} {h}")
  if s == -1:
    print("error " + o)
  else:
    ctime = o
    cublasTimes[m] = ctime
  (s, o) = subprocess.getstatusoutput(f"./attention {m} {int(h/8)} {h} {h} check=false split_k1_slices={tiles[m][h]['split_ks'][0]} split_k2_slices={tiles[m][h]['split_ks'][1]} rowSyncOrTileSync={1 if tiles[m]['Best-Policy'] == 'Row-Sync' else 0}")
  # print(o)
  if s != 0 or "Invalid" in o:
    print("error " + o)
  else:
    baselinetimes = getAllTimes(o, 'START-BASELINE', 'END-BASELINE')
    overlaptimes  = getAllTimes(o, 'START-OVERLAPPED', 'END-OVERLAPPED')
    matmul1TBs = int(re.findall(r"Number of first matmul TBs: (\d+)", o)[0])
    matmul2TBs = int(re.findall(r"Number of second matmul TBs: (\d+)", o)[0])
    softmaxTBs = m//tiles[m]["SoftmaxRowTile"]
    bTimeTotal = baselinetimes["Total"]
    bTimeMatmul1 = baselinetimes["matmul1Time"]
    bTimeSoftmax = baselinetimes["softmaxTime"]
    bTimeMatmul2 = baselinetimes["matmul2Time"]
    maxtbs = tiles[m]["MaxTBsPerSM"]
    otime = overlaptimes["Total"]

    print(f"{m} & {h} & {ctime} & {avg(bTimeTotal)} & {stdev(bTimeTotal)} & {avg(bTimeMatmul1)} & {avg(bTimeSoftmax)} & {avg(bTimeMatmul2)} & {maxtbs} & {matmul1TBs} & {softmaxTBs} & {matmul2TBs} & {avg(otime)} & {stdev(otime)} & {100 - avg(otime)/avg(bTimeTotal)*100}")
    # btime = re.findall(r'START-BASELINE: ([\.\d]+)', o)
    # baselineTimes[m] = btime[0]
    # otime = re.findall(r'START-OVERLAPPED elapsedtime ([\.\d]+)', o)
    # overlappedTimes[m] = otime[0]
    # speedup[m] = float(btime[0])/float(otime[0])

    # print(f"{h} & {btime[0]} & {ctime} & {otime[0]}")

# print(baselineTimes)
# print(cublasTimes)
# print(overlappedTimes)
# print(minimumTimes)
print("M & N & K & L & TBs & Baseline(us) & cuBLAS(us) & Overlapped(us) & Minimum(us) & Speedup & MaxSpeedup")

for m in baselineTimes:
  print(f"{m} & {n} & {k} & {l} & {m//128*n//128} & {baselineTimes[m]} & {cublasTimes[m]} & {overlappedTimes[m]} & {minimumTimes[m]} & {speedup[m]} & {maxspeedup[m]}")