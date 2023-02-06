import subprocess
import re

baselineTimes = {}
overlappedTimes = {}
minimumTimes = {}
speedup = {}
maxspeedup = {}

if False:
  for d in range(240, 400, 4):
    m = 128 * d
    n = 128
    k = 128
    l = 128

    (s, o) = subprocess.getstatusoutput("./a.out %d %d %d %d"%(m, n, k, l))
    if s == -1:
      print("error " + o)
    else:
      btime = re.findall(r'baseline elapsedtime ([\.\d]+)', o)
      baselineTimes[m] = btime[0]
      mtime = re.findall(r'minimum elapsedtime ([\.\d]+)', o)
      minimumTimes[m] = mtime[0]
      otime = re.findall(r'overlapped elapsedtime ([\.\d]+)', o)
      overlappedTimes[m] = otime[0]
      speedup[m] = float(btime[0])/float(otime[0])
      maxspeedup[m] = float(btime[0])/float(mtime[0])
    print(f"{m} & 128 & 128 & 128 & {d} & {btime[0]} & {otime[0]} & {mtime[0]}")

  print(baselineTimes)
  print(overlappedTimes)
  print(minimumTimes)
  print("M & N & K & L & TBs & Baseline(ms) & Overlapped(ms) & Minimum(ms) & Speedup & MaxSpeedup")

  for d in baselineTimes:
    print(f"{d} & 128 & 128 & 128 & {d//128} & {baselineTimes[d]} & {overlappedTimes[d]} & {speedup[d]} & {maxspeedup[d]}")

else:
  for i in range(int(80*3.5), int(80*10.5), 80):
    m = 128*i
    baselineTimes[m] = {}
    overlappedTimes[m] = {}
    speedup[m] = {}
    minimumTimes[m] = {}
    maxspeedup[m] = {}

    for d in range(1, 32, 2):
      n = d*128
      k = d*128
      l = d*128

      (s, o) = subprocess.getstatusoutput("./a.out %d %d %d %d"%(m, n, k, l))
      if s == -1:
        print("error " + o)
      else:
        # print(o)
        btime = re.findall(r'baseline elapsedtime ([\.\d]+)', o)
        baselineTimes[m][k] = btime[0]
        otime = re.findall(r'overlapped elapsedtime ([\.\d]+)', o)
        overlappedTimes[m][k] = otime[0]
        mtime = re.findall(r'minimum elapsedtime ([\.\d]+)', o)
        minimumTimes[m][k] = mtime[0]
        maxspeedup[m][k] = float(btime[0])/float(mtime[0])
        speedup[m][k] = float(btime[0])/float(otime[0])
      print(f"{m} & {k} & {k} & {k} & {(m//128)*(n//128)} & {btime[0]} & {otime[0]} & {mtime[0]}")

  print(baselineTimes)
  print(overlappedTimes)

  print("M & N & K & L & TBs & Baseline(ms) & Overlapped(ms) & Minimum(ms) & Speedup & MaxSpeedup")

  for m in baselineTimes:
    for k in baselineTimes[m]:
      print(f"{m} & {k} & {k} & {k} & {(m//128)*(n//128)} & {baselineTimes[m][k]} & {overlappedTimes[m][k]} & {minimumTimes[m][k]} & {speedup[m][k]} & {maxspeedup[m][k]}")
