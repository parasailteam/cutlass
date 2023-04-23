import torch
from torch import nn
import sys 
import time

N = int(sys.argv[1])
H = 224 #int(sys.argv[2])
W = 224 #int(sys.argv[3])
C = 3 #int(sys.argv[4])
K = 1#int(sys.argv[5])
R = 1#int(sys.argv[6])
S = 1#int(sys.argv[7])

imgs = torch.ones((N, 3, H, W), dtype=torch.float16)
imgs = imgs.cuda()
conv1 = torch.nn.Conv2d(3, 64, 7, stride=2,padding=3, dtype=torch.float16).cuda()
conv2 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1).cuda()
conv3 = torch.nn.Conv2d(64, 64, 3, stride=1,padding=1, dtype=torch.float16).cuda()
# conv3 = torch.nn.Conv2d(64, 64, 3, stride=1,padding=1)

conv1_o = conv1(imgs)
conv2_o = conv2(conv1_o)

def conv64x64_3(input):
    conv3_o = conv3(input)
    # conv3_o = conv3(conv3_o)
    # conv3_o = conv3(conv3_o)
    
for i in range(10):
    conv64x64_3(conv2_o)

torch.cuda.synchronize()

epochs = 20
start = time.time_ns()

for i in range(epochs):
    conv64x64_3(conv2_o)

torch.cuda.synchronize()
end = time.time_ns()

print((end-start)/epochs/1e3)