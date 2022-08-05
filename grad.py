import torch
import torch.nn as nn

batch, feat, hid = 5, 3, 20
module = torch.nn.Sequential(nn.Linear(feat, hid), nn.Linear(hid, 2))

x = torch.randn((batch, feat))
x.requires_grad = True
y = torch.LongTensor([1] * batch)
grads1 = torch.autograd.grad(nn.CrossEntropyLoss()(module(x), y), x)
print(grads1[0].shape)
