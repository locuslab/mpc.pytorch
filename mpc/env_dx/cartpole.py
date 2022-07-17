#!/usr/bin/env python3

import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

import numpy as np

from mpc import util

import os

import shutil
FFMPEG_BIN = shutil.which('ffmpeg')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

# import sys
# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(mode='Verbose',
#      color_scheme='Linux', call_pdb=1)

class CartpoleDx(nn.Module):
    def __init__(self, params=None):
        super().__init__()

        self.n_state = 5
        self.n_ctrl = 1

        # model parameters
        if params is None:
            # gravity, masscart, masspole, length
            self.params = Variable(torch.Tensor((9.8, 1.0, 0.1, 0.5)))
        else:
            self.params = params
        assert len(self.params) == 4
        self.force_mag = 100.

        self.theta_threshold_radians = np.pi#12 * 2 * np.pi / 360
        self.x_threshold = 2.4
        self.max_velocity = 10

        self.dt = 0.05

        self.lower = -self.force_mag
        self.upper = self.force_mag

        # 0  1      2        3   4
        # x dx cos(th) sin(th) dth
        self.goal_state = torch.Tensor(  [ 0.,  0.,  1., 0.,   0.])
        self.goal_weights = torch.Tensor([0.1, 0.1,  1., 1., 0.1])
        self.ctrl_penalty = 0.001

        self.mpc_eps = 1e-4
        self.linesearch_decay = 0.5
        self.max_linesearch_iter = 2

    def forward(self, state, u):
        squeeze = state.ndimension() == 1

        if squeeze:
            state = state.unsqueeze(0)
            u = u.unsqueeze(0)

        if state.is_cuda and not self.params.is_cuda:
            self.params = self.params.cuda()
        gravity, masscart, masspole, length = torch.unbind(self.params)
        total_mass = masspole + masscart
        polemass_length = masspole * length

        u = torch.clamp(u[:,0], -self.force_mag, self.force_mag)

        x, dx, cos_th, sin_th, dth = torch.unbind(state, dim=1)
        th = torch.atan2(sin_th, cos_th)

        cart_in = (u + polemass_length * dth**2 * sin_th) / total_mass
        th_acc = (gravity * sin_th - cos_th * cart_in) / \
                 (length * (4./3. - masspole * cos_th**2 /
                                     total_mass))
        xacc = cart_in - polemass_length * th_acc * cos_th / total_mass

        x = x + self.dt * dx
        dx = dx + self.dt * xacc
        th = th + self.dt * dth
        dth = dth + self.dt * th_acc

        state = torch.stack((
            x, dx, torch.cos(th), torch.sin(th), dth
        ), 1)

        return state

    def get_frame(self, state, ax=None):
        state = util.get_data_maybe(state.view(-1))
        assert len(state) == 5
        x, dx, cos_th, sin_th, dth = torch.unbind(state)
        gravity, masscart, masspole, length = torch.unbind(self.params)
        th = np.arctan2(sin_th, cos_th)
        th_x = sin_th*length
        th_y = cos_th*length

        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))
        else:
            fig = ax.get_figure()
        ax.plot((x,x+th_x), (0, th_y), color='k')
        ax.set_xlim((-length*2, length*2))
        ax.set_ylim((-length*2, length*2))
        return fig, ax

    def get_true_obj(self):
        q = torch.cat((
            self.goal_weights,
            self.ctrl_penalty*torch.ones(self.n_ctrl)
        ))
        assert not hasattr(self, 'mpc_lin')
        px = -torch.sqrt(self.goal_weights)*self.goal_state #+ self.mpc_lin
        p = torch.cat((px, torch.zeros(self.n_ctrl)))
        return Variable(q), Variable(p)

if __name__ == '__main__':
    dx = CartpoleDx()
    n_batch, T = 8, 50
    u = torch.zeros(T, n_batch, dx.n_ctrl)
    xinit = torch.zeros(n_batch, dx.n_state)
    th = 1.
    xinit[:,2] = np.cos(th)
    xinit[:,3] = np.sin(th)
    x = xinit
    for t in range(T):
        x = dx(x, u[t])
        fig, ax = dx.get_frame(x[0])
        fig.savefig('{:03d}.png'.format(t))
        plt.close(fig)

    vid_file = 'cartpole_vid.mp4'
    if os.path.exists(vid_file):
        os.remove(vid_file)
    cmd = ('{} -loglevel quiet '
            '-r 32 -f image2 -i %03d.png -vcodec '
            'libx264 -crf 25 -pix_fmt yuv420p {}').format(
        FFMPEG_BIN,
        vid_file
    )
    os.system(cmd)
    for t in range(T):
        os.remove('{:03d}.png'.format(t))
