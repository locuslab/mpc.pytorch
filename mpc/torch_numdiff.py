#!/usr/bin/env python3

import numpy as np

import torch
from torch.autograd import Function, Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module

import itertools

import sys

def grad(net, inputs, eps=1e-4):
    assert(inputs.ndimension() == 2)
    nBatch, nDim = inputs.size()
    xp, xn = [], []
    e = 0.5*eps*torch.eye(nDim).type_as(inputs.data)
    for b in range(nBatch):
        for i in range(nDim):
            xp.append((inputs.data[b].clone()+e[i]).unsqueeze(0))
            xn.append((inputs.data[b].clone()-e[i]).unsqueeze(0))
    xs = Variable(torch.cat(xp+xn))
    fs = net(xs)
    fDim = fs.size(1) if fs.ndimension() > 1 else 1
    fs_p, fs_n = torch.split(fs, nBatch*nDim)
    g = ((fs_p-fs_n)/eps).view(nBatch, nDim, fDim).squeeze(2)
    return g

def hess(net, inputs, eps=1e-4):
    assert(inputs.ndimension() == 2)
    nBatch, nDim = inputs.size()
    xpp, xpn, xnp, xnn = [], [], [], []
    e = eps*torch.eye(nDim).type_as(inputs.data)
    for b,i,j in itertools.product(range(nBatch), range(nDim), range(nDim)):
        xpp.append((inputs.data[b].clone()+e[i]+e[j]).unsqueeze(0))
        xpn.append((inputs.data[b].clone()+e[i]-e[j]).unsqueeze(0))
        xnp.append((inputs.data[b].clone()-e[i]+e[j]).unsqueeze(0))
        xnn.append((inputs.data[b].clone()-e[i]-e[j]).unsqueeze(0))
    xs = Variable(torch.cat(xpp+xpn+xnp+xnn))
    fs = net(xs)
    fDim = fs.size(1) if fs.ndimension() > 1 else 1
    fpp, fpn, fnp, fnn = torch.split(fs, nBatch*nDim*nDim)
    h = ((fpp-fpn-fnp+fnn)/(4*eps*eps)).view(nBatch, nDim, nDim, fDim).squeeze(3)
    return h

def test():
    torch.manual_seed(0)

    class Net(Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2,10)
            self.fc2 = nn.Linear(10,1)

        def forward(self, x):
            x = F.softplus(self.fc1(x))
            x = self.fc2(x).squeeze()
            return x

    # class Net(Function):
    #     def forward(self, inputs):
    #         Q = torch.eye(3,3)
    #         return 0.5*inputs.mm(Q).unsqueeze(1).bmm(inputs.unsqueeze(2)).squeeze()

    net = Net().double()
    nBatch = 4
    x = Variable(torch.randn(nBatch,2).double())
    x.requires_grad = True
    y = net(x)
    y.backward(torch.ones(nBatch).double())
    print(x.grad)
    x_grad = grad(net, x, eps=1e-4)
    print(x_grad)
    x_hess = hess(net, x, eps=1e-4)
    print(x_hess)

if __name__=='__main__':
    test()
