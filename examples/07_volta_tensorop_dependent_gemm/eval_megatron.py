import subprocess
import re

baselineTimes = {}
cublasTimes = {}
overlappedTimes = {}
minimumTimes = {}
speedup = {}
maxspeedup = {}

#SelfAttention
for h in [8192, 10240, 12288, 16384, 20480, 25600]:
  m = 2048

  (s, o) = subprocess.getstatusoutput(f"python3 torchAttention.py {m} {int(h/8)} {h} {h}")
  if s == -1:
    print("error " + o)
  else:
    ctime = o
    cublasTimes[m] = ctime
  (s, o) = subprocess.getstatusoutput(f"./attention {m} {int(h/8)} {h} {h} check=false split_k1_slices=1 split_k2_slices=1 rowSyncOrTileSync=1")
  print(o)
  if s != 0 or "Invalid" in o:
    print("error " + o)
  else:
    btime = re.findall(r'baseline elapsedtime ([\.\d]+)', o)
    baselineTimes[m] = btime[0]
    otime = re.findall(r'overlapped elapsedtime ([\.\d]+)', o)
    overlappedTimes[m] = otime[0]
    speedup[m] = float(btime[0])/float(otime[0])

    print(f"{h} & {btime[0]} & {ctime} & {otime[0]}")

# print(baselineTimes)
# print(cublasTimes)
# print(overlappedTimes)
# print(minimumTimes)
print("M & N & K & L & TBs & Baseline(us) & cuBLAS(us) & Overlapped(us) & Minimum(us) & Speedup & MaxSpeedup")

for m in baselineTimes:
  print(f"{m} & {n} & {k} & {l} & {m//128*n//128} & {baselineTimes[m]} & {cublasTimes[m]} & {overlappedTimes[m]} & {minimumTimes[m]} & {speedup[m]} & {maxspeedup[m]}")