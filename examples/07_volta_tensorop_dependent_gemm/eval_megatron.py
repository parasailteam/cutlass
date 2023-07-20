import subprocess
import re
import sys

attention_or_mlp = sys.argv[1]
model = sys.argv[2]

assert attention_or_mlp in ["attention", "mlp"]

baselineTimes = {}
cublasTimes = {}
overlappedTimes = {}
minimumTimes = {}
speedup = {}
maxspeedup = {}
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

if attention_or_mlp == "attention":
  # Dictionary of tile sizes for each M
  tiles = {
    2048: {
      "TileSizes" : [256, 128, 32, 128, 64, 32], "SoftmaxRowTile" : 4, "MaxTBsPerSM": 2, "Best-Policy": "Row-Sync",
      6144: {"split_ks": [2,1]},
      8192: {"split_ks": [2,1]},
      10240: {"split_ks": [1,1]},
      12288: {"split_ks": [1,1]},
      16384: {"split_ks": [1,1]},
      20480: {"split_ks": [1,1]},
      25600: {"split_ks": [1,1]},
    },
    1024: {"TileSizes" : [256, 128, 32, 128, 64, 32], "split_ks": [1,1], "MaxTBsPerSM": 2, "SoftmaxRowTile": 2, "Best-Policy": "Row-Sync",
      6144: {"split_ks": [2,1]},
      8192: {"split_ks": [2,1]},
      10240: {"split_ks": [3,1]},
      12288: {"split_ks": [1,1]},
      16384: {"split_ks": [2,1]},
      20480: {"split_ks": [2,1]},
      25600: {"split_ks": [2,1]},
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
      6144: {"split_ks":   [12,3]},
      8192: {"split_ks":   [12,3]},
      10240: {"split_ks":  [12,3]},
      12288: {"split_ks":  [12,3]},
      16384: {"split_ks":  [12,3]},
      20480: {"split_ks":  [12,3]},
      25600: {"split_ks":  [12,3]}},
    64: {"TileSizes" : [64, 256, 32, 32, 128, 32], "SoftmaxRowTile": 1, "MaxTBsPerSM": 3, "Best-Policy": "Tile-Sync",
      6144: {"split_ks":   [3,1]},
      8192: {"split_ks":   [4,1]},
      10240: {"split_ks":  [12,3]},
      12288: {"split_ks":  [4,1], "TileBatch":1},
      16384: {"split_ks":  [12,3]},
      20480: {"split_ks":  [12,3]},
      25600: {"split_ks":  [12,3]}},
    32: {"TileSizes" : [64, 256, 32, 32, 128, 32], "split_ks": [4,1], "MaxTBsPerSM": 3,
      6144: {"split_ks":   [3,1]},
      8192: {"split_ks":   [4,1]},
      10240: {"split_ks":  [4,1]},
      12288: {"split_ks":  [4,1], "TileBatch":1},
      16384: {"split_ks":  [4,1]},
      20480: {"split_ks":  [4,1]},
      25600: {"split_ks":  [4,1]}},
    16: {"TileSizes" : [64, 256, 32, 32, 128, 32], "split_ks": [4,1], "MaxTBsPerSM": 3,
      6144: {"split_ks":   [4,1]},
      8192: {"split_ks":   [4,1]},
      10240: {"split_ks":  [4,1]},
      12288: {"split_ks":  [4,1], "TileBatch":1},
      16384: {"split_ks":  [4,1]},
      20480: {"split_ks":  [4,1]},
      25600: {"split_ks":  [4,1]}},
    8: {"TileSizes" : [64, 256, 32, 32, 128, 32], "split_ks": [4, 1], "MaxTBsPerSM": 3,
      6144: {"split_ks":   [4,1]},
      8192: {"split_ks":   [4,1]},
      10240: {"split_ks":  [4,1]},
      12288: {"split_ks":  [4,1], "TileBatch":1},
      16384: {"split_ks":  [4,1]},
      20480: {"split_ks":  [4,1]},
      25600: {"split_ks":  [4,1]}},
    4: {"TileSizes" : [64, 256, 32, 32, 128, 32], "split_ks": [4, 1], "MaxTBsPerSM": 3,
      6144: {"split_ks":   [4,1]},
      8192: {"split_ks":   [4,1]},
      10240: {"split_ks":  [4,1]},
      12288: {"split_ks":  [4,1], "TileBatch":1},
      16384: {"split_ks":  [4,1]},
      20480: {"split_ks":  [4,1]},
      25600: {"split_ks":  [4,1]}},
    2: {"TileSizes" : [64, 256, 32, 32, 128, 32], "split_ks": [4,1], "MaxTBsPerSM": 3,
      6144: {"split_ks":   [4,1]},
      8192: {"split_ks":   [4,1]},
      10240: {"split_ks":  [4,1]},
      12288: {"split_ks":  [4,1], "TileBatch":1},
      16384: {"split_ks":  [4,1]},
      20480: {"split_ks":  [4,1]},
      25600: {"split_ks":  [4,1]}},
    1: {"TileSizes" : [64, 256, 32, 32, 128, 32], "split_ks": [4,1], "MaxTBsPerSM": 3,
      6144: {"split_ks":   [4,1]},
      8192: {"split_ks":   [4,1]},
      10240: {"split_ks":  [4,1]},
      12288: {"split_ks":  [4,1], "TileBatch":1}, 
      16384: {"split_ks":  [4,1]},
      20480: {"split_ks":  [4,1]},
      25600: {"split_ks":  [4,1]}}
  }

elif attention_or_mlp == "mlp":
    # Dictionary of tile sizes for each M
  tiles = {
    2048: {
      "TileSizes" : [256, 128, 32, 128, 64, 32], "MaxTBsPerSM": 2, "Best-Policy": "Row-Sync",
      6144: {"split_ks": [3,1]},
      8192: {"split_ks": [1,1]},
      10240: {"split_ks": [1,1]},
      12288: {"split_ks": [1,1]},
      14336: {"split_ks": [1,1]},
      16384: {"split_ks": [1,1]},
      20480: {"split_ks": [1,1]},
      25600: {"split_ks": [1,1]},
    },
    1024: {"TileSizes" : [256, 128, 32, 128, 64, 32], "MaxTBsPerSM": 2, "Best-Policy": "Row-Sync",
      6144: {"split_ks": [3,1]},
      8192: {"split_ks": [1,1]},
      10240: {"split_ks": [1,1]},
      12288: {"split_ks": [1,1]},
      14336: {"split_ks": [1,1]},
      16384: {"split_ks": [1,1]},
      20480: {"split_ks": [1,1]},
      25600: {"split_ks": [1,1]},
    },
    512: {"TileSizes" : [256, 128, 32, 128, 64, 32], "MaxTBsPerSM": 2, "Best-Policy": "Row-Sync",
      6144: {"split_ks":   [3,1]},
      8192: {"split_ks":   [2,1]},
      10240: {"split_ks":  [2,1]},
      12288: {"split_ks":  [2,1]},
      14336: {"split_ks":  [2,1]},
      16384: {"split_ks":  [2,1]},
      20480: {"split_ks":  [2,1]},
      25600: {"split_ks":  [2,1]},
    },
    256: {"TileSizes" : [256, 128, 32, 128, 64, 32], "MaxTBsPerSM": 2, "Best-Policy": "Tile-Sync",
      6144: {"split_ks":   [4,1]},
      8192: {"split_ks":   [4,1]},
      10240: {"split_ks":  [4,1]},
      12288: {"split_ks":  [4,1]},
      14336: {"split_ks":  [4,1]},
      16384: {"split_ks":  [4,1]},
      20480: {"split_ks":  [4,1]},
      25600: {"split_ks":  [4,1]},
    },
    128: {"TileSizes" : [128, 128, 32, 64, 64, 32], "MaxTBsPerSM": 2, "Best-Policy": "Tile-Sync",
      6144: {"split_ks":   [12,3]},
      8192: {"split_ks":   [12,3]},
      10240: {"split_ks":  [12,3]},
      12288: {"split_ks":  [12,3]},
      14336: {"split_ks":  [12,3]},
      16384: {"split_ks":  [2,1]},
      20480: {"split_ks":  [12,3]},
      25600: {"split_ks":  [12,3]},},
    64: {"TileSizes" : [64, 256, 32, 32, 128, 32], "MaxTBsPerSM": 3, "Best-Policy": "Tile-Sync",
      6144: {"split_ks":   [4,1], "TileBatch": 2},
      8192: {"split_ks":   [4,1], "TileBatch": 4},
      10240: {"split_ks":  [12,3]},
      12288: {"split_ks":  [2,2], "TileBatch": 8},
      14336: {"split_ks":  [3,1]},
      16384: {"split_ks":  [2,1], "TileBatch": 8},
      20480: {"split_ks":  [12,3]},
      25600: {"split_ks":  [12,3]}},
    32: {"TileSizes" : [64, 256, 32, 32, 128, 32], "split_ks": [4,1], "MaxTBsPerSM": 3, "Best-Policy": "Tile-Sync",
      6144: {"split_ks":   [4,1], "TileBatch": 2},
      8192: {"split_ks":   [4,1], "TileBatch": 4},
      10240: {"split_ks":  [12,3]},
      12288: {"split_ks":  [2,2], "TileBatch": 8},
      14336: {"split_ks":  [3,1]},
      16384: {"split_ks":  [2,1], "TileBatch": 8},
      20480: {"split_ks":  [12,3]},
      25600: {"split_ks":  [12,3]}},
    16: {"TileSizes" : [64, 256, 32, 32, 128, 32], "split_ks": [4,1],
    "MaxTBsPerSM": 3, "Best-Policy": "Tile-Sync",
      6144: {"split_ks":   [4,1], "TileBatch": 2},
      8192: {"split_ks":   [4,1], "TileBatch": 4},
      10240: {"split_ks":  [4,1]},
      12288: {"split_ks":  [2,2], "TileBatch": 8},
      14336: {"split_ks":  [3,1]},
      16384: {"split_ks":  [2,1], "TileBatch": 8},
      20480: {"split_ks":  [4,1]},
      25600: {"split_ks":  [4,1]}},
    8: {"TileSizes" : [64, 256, 32, 32, 128, 32], "split_ks": [4, 1], 
    "MaxTBsPerSM": 3, "Best-Policy": "Tile-Sync",
      6144: {"split_ks":   [4,1], "TileBatch": 2},
      8192: {"split_ks":   [4,1], "TileBatch": 4},
      10240: {"split_ks":  [4,1]},
      12288: {"split_ks":  [2,2], "TileBatch": 8},
      14336: {"split_ks":  [3,1]},
      16384: {"split_ks":  [2,1], "TileBatch": 8},
      20480: {"split_ks":  [4,1]},
      25600: {"split_ks":  [4,1]}},
    4: {"TileSizes" : [64, 256, 32, 32, 128, 32], "split_ks": [4, 1],
    "MaxTBsPerSM": 3, "Best-Policy": "Tile-Sync",
      6144: {"split_ks":   [4,1], "TileBatch": 2},
      8192: {"split_ks":   [4,1], "TileBatch": 4},
      10240: {"split_ks":  [4,1]},
      12288: {"split_ks":  [2,2], "TileBatch": 8},
      14336: {"split_ks":  [3,1]},
      16384: {"split_ks":  [2,1], "TileBatch": 8},
      20480: {"split_ks":  [4,1]},
      25600: {"split_ks":  [4,1]}},
    2: {"TileSizes" : [64, 256, 32, 32, 128, 32], "split_ks": [4,1], "MaxTBsPerSM": 3, "Best-Policy": "Tile-Sync",
      6144: {"split_ks":   [4,1], "TileBatch": 2},
      8192: {"split_ks":   [4,1], "TileBatch": 4},
      10240: {"split_ks":  [4,1]},
      12288: {"split_ks":  [2,2], "TileBatch": 8},
      14336: {"split_ks":  [3,1]},
      16384: {"split_ks":  [2,1], "TileBatch": 8},
      20480: {"split_ks":  [4,1]},
      25600: {"split_ks":  [4,1]}},
    1: {"TileSizes" : [64, 256, 32, 32, 128, 32], "split_ks": [4,1], "MaxTBsPerSM": 3, "Best-Policy": "Tile-Sync",
      6144: {"split_ks":   [4,1], "TileBatch": 2},
      8192: {"split_ks":   [4,1], "TileBatch": 4},
      10240: {"split_ks":  [4,1]},
      12288: {"split_ks":  [2,2], "TileBatch": 8}, #split_k: 3,1 64x256x32 32x128x32, tilebatch:8, load B before A
      14336: {"split_ks":  [3,1]}, #TODO: Tile batch
      16384: {"split_ks":  [2,1], "TileBatch": 8},
      20480: {"split_ks":  [4,1]},
      25600: {"split_ks":  [4,1]}}
  }

if model.lower() == "BLOOM".lower():
  H = 14336
elif model.lower() == "GPT-3".lower():
  H = 12288
else:
  print ("No Hidden dim for ", model)
  sys.exit(0)

for h in [H]:#[6144,8192, 12288, 16384]: # , 20480, 25600]: #[10240, 20480, 25600]:
  for m in [1,2,4,8,16,32,64,128]:#[256, 512, 1024, 2048]: # 256, [1,2,4,8,16,32,64,128]:
    if attention_or_mlp == "attention":
      (s, o) = subprocess.getstatusoutput(f"python3 torchAttention.py {m} {int(h/8)} {h} {h}")
    else:
      (s, o) = subprocess.getstatusoutput(f"python3 torchmlp.py {m} {int(4*h/8)} {h} {h}")
    if s == -1:
      print("error " + o)
    else:
      ctime = o
      cublasTimes[m] = ctime

    for syncPolicy in ['rowsync', 'tilesync']:#'Row-Sync',
      if attention_or_mlp == "mlp":
        command = f"./mlp-{syncPolicy} {m} {int(4*h/8)} {h} {h}"
      else:
        command = f"./attention {m} {int(h/8)} {h} {h}"
      (s, o) = subprocess.getstatusoutput(command + f" check=false split_k1_slices={tiles[m][h]['split_ks'][0]} split_k2_slices={tiles[m][h]['split_ks'][1]}")
      # print(o)
      # print(s, o)
      if "Invalid" in o:
        pass
      elif s != 0:
        print("error " + o)
      else:
        # print(o)
        baselinetimes = getAllTimes(o, 'START-BASELINE', 'END-BASELINE')
        overlaptimes  = getAllTimes(o, 'START-OVERLAPPED', 'END-OVERLAPPED')
        matmul1TBs = 0 #int(re.findall(r"Number of first matmul TBs: (\d+)", o)[0])
        matmul2TBs = 0 #int(re.findall(r"Number of second matmul TBs: (\d+)", o)[0])
        bTimeTotal = baselinetimes["Total"]
        bTimeMatmul1 = baselinetimes["matmul1Time"]
        bTimeMatmul2 = baselinetimes["matmul2Time"]
        maxtbs = tiles[m]["MaxTBsPerSM"]
        otime = overlaptimes["Total"]

        print(f'{m} & {h} & {syncPolicy} & {"%.2f"%float(ctime)} & {"%.2f"%avg(bTimeTotal)} & {"%.2f"%stdev(bTimeTotal)} & {"%.2f"%avg(bTimeMatmul1)} & {"%.2f"%avg(bTimeMatmul2)} & {maxtbs} & {matmul1TBs} & {matmul2TBs} & {"%.2f"%avg(otime)} & {"%.2f"%stdev(otime)} & {"%.2f"%(100 - avg(otime)/avg(bTimeTotal)*100)}')
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