import torch
import sys
import time

M = int(sys.argv[1])
N = int(sys.argv[2])
K = int(sys.argv[3])
L = int(sys.argv[4])

X = torch.ones((M, K), dtype=torch.half).cuda()
W1 = torch.ones((K, N), dtype=torch.half).cuda()
W2 = torch.ones((N, L), dtype=torch.half).cuda()

for i in range(10):
    XW1 = X@W1
    out = XW1@W2
torch.cuda.synchronize()

epochs = 20
start = time.time_ns()

for i in range(epochs):
    XW1 = X@W1
    out = XW1@W2
torch.cuda.synchronize()
end = time.time_ns()

print((end-start)/epochs/1e3)