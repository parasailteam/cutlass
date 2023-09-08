from ctypes import *
import torch

libc = cdll.LoadLibrary("./libmlp.so")
libc.initMLPParams.restype = POINTER(c_int)

B = 1
H = 8192
multiple_of = 128
FFN = int((((H/3)+multiple_of-1)//multiple_of)*multiple_of)
print(FFN)

def load_tensor(filepath):
    import json

    return torch.load(filepath)

def random_tensor():
    global H
    values = torch.tensor([0.05, 0.2, 0.01, 3, 0.4])
    x = torch.randint(low=0, high=values.shape[0], size=(H,))
    x = torch.index_select(values, 0, x)
    x = x.reshape(1, H).type(dtype=torch.half).cuda()
    w1 = values[torch.randint(low=0, high=values.shape[0], size=(FFN * H,))]
    w1 = w1.reshape(H, FFN).type(dtype=torch.half).cuda()
    return x, w1

# x = load_tensor("/home/saemal/msccl-demo/llama-chat/x.data")
# x = x.reshape(1, H).type(dtype=torch.half).cuda()
# x = torch.concat((x,)*B, 0)
# w1 = load_tensor("/home/saemal/msccl-demo/llama-chat/w1.data")
# w1 = w1.reshape((H, FFN)).type(dtype=torch.half).cuda()

# w1 = [] #torch.zeros((H, FFN), dtype=torch.half)

# for i in range(H):
#     o = [(j+i)/1000. for j in range(FFN)] #torch.full((FFN,), i/1000, dtype=torch.half)
#     w1 += [o]

# w1 = torch.Tensor(w1)
# w1 = w1.type(dtype=torch.half).cuda()

x, w1 = random_tensor()

print(x.device, x.shape)
print(w1.device, w1.shape)
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
c = torch.isclose(ref_silu, silu, rtol=1e-3)
for i in range(B):
    for j in range(FFN):
        if c[i][j].item() == False:
            print (ref_silu[i][j], silu[i][j])
print(torch.allclose(ref_silu, silu, rtol=1e-3))
ref_xv = torch.matmul(x, v)
ref_xv = ref_silu * ref_xv

print(torch.eq(ref_xv, xv))

w2 = torch.full((FFN, H), 0.01, dtype=torch.half).cuda()
print(ref_xv.shape, w2.shape)
ref_out = torch.matmul(ref_xv, w2)
print(torch.eq(ref_out, out))
