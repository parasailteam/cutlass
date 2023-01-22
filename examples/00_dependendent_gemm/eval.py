import subprocess
import re

baselineTimes = {}
overlappedTimes = {}
speedup = {}

if False:
  for d in range(2, 400, 4):
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
      otime = re.findall(r'overlapped elapsedtime ([\.\d]+)', o)
      overlappedTimes[m] = otime[0]
      speedup[m] = float(btime[0])/float(otime[0])
    print(f"{m} & 128 & 128 & 128 & {d} & {btime[0]} & {otime[0]}")

  print(baselineTimes)
  print(overlappedTimes)

  print("M & N & K & L & TBs & Baseline(ms) & Overlapped(ms) & Speedup")

  for d in baselineTimes:
    print(f"{d} & 128 & 128 & 128 & {d} & {baselineTimes[d]} & {overlappedTimes[d]} & {speedup[d]}")

else:
  for i in range(int(80*3.5), int(80*10.5), 80):
    m = 128*i
    baselineTimes[m] = {}
    overlappedTimes[m] = {}
    speedup[m] = {}

    for d in range(128, 128*9, 128):
      n = 128
      k = d
      l = 128

      (s, o) = subprocess.getstatusoutput("./a.out %d %d %d %d"%(m, n, k, l))
      if s == -1:
        print("error " + o)
      else:
        # print(o)
        btime = re.findall(r'baseline elapsedtime ([\.\d]+)', o)
        baselineTimes[m][k] = btime[0]
        otime = re.findall(r'overlapped elapsedtime ([\.\d]+)', o)
        overlappedTimes[m][k] = otime[0]
        speedup[m][k] = float(btime[0])/float(otime[0])
      print(f"{m} & 128 & {k} & 128 & {m//128} & {btime[0]} & {otime[0]}")

  print(baselineTimes)
  print(overlappedTimes)

  print("M & N & K & L & TBs & Baseline(ms) & Overlapped(ms) & Speedup")

  for m in baselineTimes:
    for k in baselineTimes[m]:
      print(f"{m} & 128 & {k} & 128 & {m//128} & {baselineTimes[m][k]} & {overlappedTimes[m][k]} & {speedup[m][k]}")