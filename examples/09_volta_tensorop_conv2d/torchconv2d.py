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
conv4 = torch.nn.Conv2d(64, 128, 3, stride=2,padding=1, dtype=torch.float16).cuda()
conv5 = torch.nn.Conv2d(128, 128, 3, stride=1,padding=1, dtype=torch.float16).cuda()
conv6 = torch.nn.Conv2d(128, 256, 3, stride=2,padding=1, dtype=torch.float16).cuda()
conv7 = torch.nn.Conv2d(256, 256, 3, stride=1,padding=1, dtype=torch.float16).cuda()
conv8 = torch.nn.Conv2d(256, 512, 3, stride=2,padding=1, dtype=torch.float16).cuda()
conv9 = torch.nn.Conv2d(512, 512, 3, stride=1,padding=1, dtype=torch.float16).cuda()

# conv3 = torch.nn.Conv2d(64, 64, 3, stride=1,padding=1)

conv1_o = conv1(imgs)
conv2_o = conv2(conv1_o)

conv5_in = conv4(conv2_o)
print("Input shape for 128, 3x3", conv5_in.shape)

conv7_in = conv6(conv5_in)
print("Input shape for 256, 3x3", conv7_in.shape)

conv9_in = conv8(conv7_in)
print("Input shape for 512, 3x3", conv9_in.shape)

def conv64x64_3(input):
    conv3_o = conv3(input)
    # conv3_o = conv3(conv3_o)
    # conv3_o = conv3(conv3_o)

def conv128x128_3(input):
    conv5_o = conv5(input)
    # conv3_o = conv3(conv3_o)
    # conv3_o = conv3(conv3_o)

def conv256x256_3(input):
    conv7_o = conv7(input)
    # conv3_o = conv3(conv3_o)
    # conv3_o = conv3(conv3_o)

def conv512x512_3(input):
    conv7_o = conv9(input)
    # conv3_o = conv3(conv3_o)
    # conv3_o = conv3(conv3_o)


for i in range(10):
    conv64x64_3(conv2_o)

torch.cuda.synchronize()

def execute(f, input):
    epochs = 20
    start = time.time_ns()

    for i in range(epochs):
        f(input)

    torch.cuda.synchronize()
    end = time.time_ns()
    return (end-start)/epochs/1e3

print("3x3 64: ", execute(conv64x64_3, conv2_o))
print("3x3 128: ", execute(conv128x128_3, conv5_in))
print("3x3 256: ", execute(conv256x256_3, conv7_in))
print("3x3 512: ", execute(conv512x512_3, conv9_in))