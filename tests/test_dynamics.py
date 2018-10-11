#!/usr/bin/env python3

import torch
from torch.autograd import Function, Variable, grad
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

import numpy as np
import numpy.random as npr
import numpy.testing as npt

import cvxpy as cp

from block import block

from mpc.dynamics import NNDynamics


# import sys
# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(mode='Verbose',
#      color_scheme='Linux', call_pdb=1)

def test_grad_input():
    torch.manual_seed(0)

    n_batch, n_state, n_ctrl = 2,3,4
    hidden_sizes = [42]*5

    for act in ['relu', 'sigmoid']:
        x = Variable(torch.rand(n_batch, n_state), requires_grad=True)
        u = Variable(torch.rand(n_batch, n_ctrl), requires_grad=True)
        net = NNDynamics(
            n_state, n_ctrl, hidden_sizes=hidden_sizes, activation=act)
        x_ = net(x, u)

        R, S = [], []
        for i in range(n_batch):
            Ri, Si = [], []
            for j in range(n_state):
                grad_xij, grad_uij = grad(
                x_[i,j], [x, u], create_graph=True)
                grad_xij = grad_xij[i]
                grad_uij = grad_uij[i]
                Ri.append(grad_xij)
                Si.append(grad_uij)
            R.append(torch.stack(Ri))
            S.append(torch.stack(Si))
        R = torch.stack(R)
        S = torch.stack(S)

        R_, S_ = net.grad_input(x, u)

        npt.assert_allclose(R.data.numpy(), R_.data.numpy(), rtol=1e-4)
        npt.assert_allclose(S.data.numpy(), S_.data.numpy(), rtol=1e-4)

if __name__=='__main__':
    test_grad_input()
