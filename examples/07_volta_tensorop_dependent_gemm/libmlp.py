from ctypes import *
import torch

libc = cdll.LoadLibrary("./libmlp.so")

t = torch.zeros((1024, 1024))
print(hex(t.data_ptr()), t.numel())
libc.initMLPParams(c_void_p(t.data_ptr()), t.numel())