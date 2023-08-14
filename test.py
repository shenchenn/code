import torch

avg = torch.nn.AvgPool3d(kernel_size=(3,1,1),stride=(1,1,1),padding=(1,0,0))
x = torch.rand(1,3,1,1)
print(x)
b,c,w,h = x.size()
x = x.reshape(b,-1,c,w,h)
x = avg(x)
x = x.reshape(b,c,w,h)
print(x)