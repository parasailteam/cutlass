from ctypes import *
import torch

libc = cdll.LoadLibrary("./libmlp.so")

B = 1
H = 8192
FFN = int((((H/3)+127)//128)*128)

x = torch.ones((B, 8192), dtype=torch.half).cuda()

w1 = torch.full((H, FFN), 0.01, dtype=torch.half).cuda()
v = torch.full((H, FFN), 0.01, dtype=torch.half).cuda()
w2 = torch.ones((FFN, H), dtype=torch.half).cuda()

silu = torch.zeros((B, FFN), dtype=torch.half).cuda()
xv = torch.zeros((B, FFN), dtype=torch.half).cuda()

out = torch.zeros((B, H), dtype=torch.half).cuda()

libc.initMLPParams(c_void_p(w1.data_ptr()), c_void_p(v.data_ptr()), c_void_p(w2.data_ptr()), c_int(B))

libc.runLLAMA(c_void_p(x.data_ptr()), c_void_p(silu.data_ptr()), c_void_p(xv.data_ptr()), c_void_p(out.data_ptr()))

m = torch.nn.SiLU()
ref_xw1 = torch.matmul(x, w1)
ref_silu = m(ref_xw1)
print(torch.eq(ref_silu, silu))
ref_xv = torch.matmul(x, v)
ref_xv = ref_silu * ref_xv

print(torch.eq(ref_xv, xv))

for i in range(1):
  print(ref_xv[0][i], xv[0][i])
