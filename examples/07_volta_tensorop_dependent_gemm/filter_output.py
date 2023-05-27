import sys
import re

f = open('o', 'r')
s = f.read()
found_vals = []
smids = [0 for i in range(0,80)]

for l in re.findall(r'.+', s):
    if '400:' in l:
        found_vals += [int(l.split(' ')[3])]
    elif 'linearid' in l:
        smids[int(l.split(' ')[3])]+=1

f.close()

print(sorted(found_vals))
for x,i in enumerate(smids):
    print(x,i)