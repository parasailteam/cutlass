import subprocess
import re

baselineTimes = {}
overlappedTimes = {}
speedup = {}

for d in range(1, 160, 1):
  m = 128 * d
  n = 128
  k = 128
  l = 128

  (s, o) = subprocess.getstatusoutput("./a.out %d %d %d %d 100"%(m, n, k, l))
  
  if s == -1:
    print("error " + o)
  else:
    btime = re.findall(r'baseline elapsedtime ([\.\d]+) milliseconds', o)
    baselineTimes[m] = btime[0]
    otime = re.findall(r'overlapped elapsedtime ([\.\d]+) milliseconds', o)
    overlappedTimes[m] = otime[0]
    speedup[m] = float(btime[0])/float(otime[0])
  print(f"{m} & 128 & 128 & 128 & {btime[0]} & {otime[0]}")

print(baselineTimes)
print(overlappedTimes)

print("M & N & K & L & Baseline(ms) & Overlapped(ms) & Speedup")

for d in baselineTimes:
  print(f"{d} & 128 & 128 & 128 & {baselineTimes[d]} & {overlappedTimes[d]} & {speedup[d]}")