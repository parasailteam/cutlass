from ctypes import *
import torch

libc = cdll.LoadLibrary("./libmlp.so")
libc.initMLPParams.restype = POINTER(c_int)

B = 2
H = 8192
multiple_of = 64
FFN = int((((H/3)+multiple_of-1)//multiple_of)*multiple_of)
print(FFN)

def load_tensor(filepath):
    import json

    return torch.load(filepath)

x = load_tensor("/home/saemal/msccl-demo/llama-chat/x.data")
x = x.reshape(1, H).type(dtype=torch.half).cuda()
x = torch.concat((x,)*B, 0)
# w1 = load_tensor("/home/saemal/msccl-demo/llama-chat/w1.data")
# w1 = w1.reshape((H, FFN)).type(dtype=torch.half).cuda()

w1 = [] #torch.zeros((H, FFN), dtype=torch.half)

for i in range(H):
    o = [(j+i)/1000. for j in range(FFN)] #torch.full((FFN,), i/1000, dtype=torch.half)
    w1 += [o]

w1 = torch.Tensor(w1)
w1 = w1.type(dtype=torch.half).cuda()

v = torch.full((H, FFN), 0.01, dtype=torch.half).cuda()
w2 = torch.full((FFN, H), 0.01, dtype=torch.half).cuda()

silu = torch.ones((B, FFN), dtype=torch.half).cuda()
xv = torch.zeros((B, FFN), dtype=torch.half).cuda()

out = torch.zeros((B, H), dtype=torch.half).cuda()

mlpParams = libc.initMLPParams(c_void_p(w1.data_ptr()), c_void_p(v.data_ptr()), c_void_p(w2.data_ptr()), c_int(B))

libc.runLLAMA(mlpParams, c_void_p(x.data_ptr()), c_void_p(silu.data_ptr()), c_void_p(xv.data_ptr()), c_void_p(out.data_ptr()))

m = torch.nn.SiLU()
ref_xw1 = torch.matmul(x, w1)

def host_matmul(t1, t2):
    for i in range(t1.shape[0]):
        for j in range(t2.shape[1]):
            t3 = 0
            for k in range(t1.shape[1]):
                t3 += t1[i][k].type(dtype=torch.float).item() * t2[k][j].type(dtype=torch.float).item()
            if j < 32:
                print(t3)
            else:
                break

# host_matmul(x, w1)

ref_silu = ref_xw1
# print(ref_xw1[0][0], ref_silu[0][0], silu[0][0])
for i in range(B):
    for j in range(FFN):
        if j > 32:
            break
        print(ref_silu[i][j], silu[i][j])
# print(torch.allclose(ref_silu, silu, atol=1e-5))
ref_xv = torch.matmul(x, v)
ref_xv = ref_silu * ref_xv

print(torch.eq(ref_xv, xv))

w2 = torch.full((FFN, H), 0.01, dtype=torch.half).cuda()
print(ref_xv.shape, w2.shape)
ref_out = torch.matmul(ref_xv, w2)
print(torch.eq(ref_out, out))
