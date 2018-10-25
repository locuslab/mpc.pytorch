import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

import numpy as np

from mpc import util

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

class PendulumDx(nn.Module):
    def __init__(self, params=None, simple=True):
        super().__init__()
        self.simple = simple

        self.max_torque = 2.0
        self.dt = 0.05
        self.n_state = 3
        self.n_ctrl = 1

        if params is None:
            if simple:
                # gravity (g), mass (m), length (l)
                self.params = Variable(torch.Tensor((10., 1., 1.)))
            else:
                # gravity (g), mass (m), length (l), damping (d), gravity bias (b)
                self.params = Variable(torch.Tensor((10., 1., 1., 0., 0.)))
        else:
            self.params = params

        assert len(self.params) == 3 if simple else 5

        self.goal_state = torch.Tensor([1., 0., 0.])
        self.goal_weights = torch.Tensor([1., 1., 0.1])
        self.ctrl_penalty = 0.001
        self.lower, self.upper = -2., 2.

        self.mpc_eps = 1e-3
        self.linesearch_decay = 0.2
        self.max_linesearch_iter = 5

    def forward(self, x, u):
        squeeze = x.ndimension() == 1

        if squeeze:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)

        assert x.ndimension() == 2
        assert x.shape[0] == u.shape[0]
        assert x.shape[1] == 3
        assert u.shape[1] == 1
        assert u.ndimension() == 2

        if x.is_cuda and not self.params.is_cuda:
            self.params = self.params.cuda()

        if not hasattr(self, 'simple') or self.simple:
            g, m, l = torch.unbind(self.params)
        else:
            g, m, l, d, b = torch.unbind(self.params)

        u = torch.clamp(u, -self.max_torque, self.max_torque)[:,0]
        cos_th, sin_th, dth = torch.unbind(x, dim=1)
        th = torch.atan2(sin_th, cos_th)
        if not hasattr(self, 'simple') or self.simple:
            newdth = dth + self.dt*(-3.*g/(2.*l) * (-sin_th) + 3. * u / (m*l**2))
        else:
            sin_th_bias = torch.sin(th + b)
            newdth = dth + self.dt*(
                -3.*g/(2.*l) * (-sin_th_bias) + 3. * u / (m*l**2) - d*th)
        newth = th + newdth*self.dt
        state = torch.stack((torch.cos(newth), torch.sin(newth), newdth), dim=1)

        if squeeze:
            state = state.squeeze(0)
        return state

    def get_frame(self, x, ax=None):
        x = util.get_data_maybe(x.view(-1))
        assert len(x) == 3
        l = self.params[2].item()

        cos_th, sin_th, dth = torch.unbind(x)
        th = np.arctan2(sin_th, cos_th)
        x = sin_th*l
        y = cos_th*l

        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))
        else:
            fig = ax.get_figure()

        ax.plot((0,x), (0, y), color='k')
        ax.set_xlim((-l*1.2, l*1.2))
        ax.set_ylim((-l*1.2, l*1.2))
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
    dx = PendulumDx()
    n_batch, T = 8, 50
    u = torch.zeros(T, n_batch, dx.n_ctrl)
    xinit = torch.zeros(n_batch, dx.n_state)
    xinit[:,0] = np.cos(0)
    xinit[:,1] = np.sin(0)
    x = xinit
    for t in range(T):
        x = dx(x, u[t])
        fig, ax = dx.get_frame(x[0])
        fig.savefig('{:03d}.png'.format(t))
        plt.close(fig)

    vid_file = 'pendulum_vid.mp4'
    if os.path.exists(vid_file):
        os.remove(vid_file)
    cmd = ('(/usr/bin/ffmpeg -loglevel quiet '
            '-r 32 -f image2 -i %03d.png -vcodec '
            'libx264 -crf 25 -pix_fmt yuv420p {}/) &').format(
        vid_file
    )
    os.system(cmd)
    for t in range(T):
        os.remove('{:03d}.png'.format(t))
