#!/usr/bin/env python3

import argparse

import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

import numpy as np

import os

from mpc import mpc, util
from mpc.mpc import GradMethods
import mpc.util as eutil
from mpc.env_dx import pendulum, cartpole

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='pendulum')
    args = parser.parse_args()

    n_batch = 1
    if args.env == 'pendulum':
        T = 20
        dx = pendulum.PendulumDx()
        xinit = torch.zeros(n_batch, dx.n_state)
        th = 1.0
        xinit[:,0] = np.cos(th)
        xinit[:,1] = np.sin(th)
        xinit[:,2] = -0.5
    elif args.env == 'cartpole':
        T = 20
        dx = cartpole.CartpoleDx()
        xinit = torch.zeros(n_batch, dx.n_state)
        th = 0.5
        xinit[:,2] = np.cos(th)
        xinit[:,3] = np.sin(th)
    else:
        assert False

    q, p = dx.get_true_obj()

    u = None
    ep_length = 100
    for t in range(ep_length):
        x, u = solve_lqr(
            dx, xinit, q, p, T, dx.linesearch_decay, dx.max_linesearch_iter, u)

        fig, ax = dx.get_frame(x[0])
        fig.savefig('{:03d}.png'.format(t))
        plt.close(fig)

        u = torch.cat((u[1:-1], u[-2:]), 0).contiguous()
        xinit = x[1]

    vid_file = 'ctrl_{}.mp4'.format(args.env)
    if os.path.exists(vid_file):
        os.remove(vid_file)
    cmd = ('/usr/bin/ffmpeg -loglevel quiet '
            '-r 32 -f image2 -i %03d.png -vcodec '
            'libx264 -crf 25 -pix_fmt yuv420p {}').format(
        vid_file
    )
    os.system(cmd)
    for t in range(ep_length):
        os.remove('{:03d}.png'.format(t))


def solve_lqr(dx, xinit, q, p, T,
              linesearch_decay, max_linesearch_iter, u_init=None):
    n_sc = dx.n_state+dx.n_ctrl

    n_batch = 1
    Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(
        T, n_batch, 1, 1
    )
    p = p.unsqueeze(0).repeat(T, n_batch, 1)

    cost = mpc.QuadCost(Q, p)

    lqr_iter = 100 if u_init is None else 10
    x_lqr, u_lqr, objs_lqr = mpc.MPC(
        dx.n_state, dx.n_ctrl, T,
        u_lower=dx.lower, u_upper=dx.upper, u_init=u_init,
        lqr_iter=lqr_iter,
        verbose=1,
        exit_unconverged=False,
        detach_unconverged=False,
        linesearch_decay=linesearch_decay,
        max_linesearch_iter=max_linesearch_iter,
        grad_method=GradMethods.AUTO_DIFF,
        eps=1e-4,
        # slew_rate_penalty=self.slew_rate_penalty,
        # prev_ctrl=prev_ctrl,
    )(xinit, cost, dx)
    return x_lqr, u_lqr


if __name__ == '__main__':
    main()
