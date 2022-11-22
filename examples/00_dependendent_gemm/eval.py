import subprocess
import re

baselineTimes = {}
overlappedTimes = {}
speedup = {}

for d in range(1, 160, 4):
  m = 128 * d
  n = 128
  k = 128
  l = 128

  (s, o) = subprocess.getstatusoutput("./a.out %d %d %d %d"%(m, n, k, l))
  if s == -1:
    print("error " + o)
  else:
    btime = re.findall(r'baseline elapsedtime ([\.\d]+) milliseconds', o)
    baselineTimes[m] = btime[0]
    otime = re.findall(r'overlapped elapsedtime ([\.\d]+) milliseconds', o)
    overlappedTimes[m] = otime[0]
    speedup[m] = float(btime[0])/float(otime[0])
  print(f"{m}x128x128x128 & {btime[0]} & {otime[0]}")

print(baselineTimes)
print(overlappedTimes)

print("Size(MxNxKxL) & Baseline(ms) & Overlapped(ms) & Speedup")

for d in baselineTimes:
  print(f"{d}x128x128x128 & {baselineTimes[d]} & {overlappedTimes[d]} & {speedup[d]}")