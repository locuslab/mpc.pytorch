import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

import numpy as np

from empc import util

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

class PendulumDx(nn.Module):
    def __init__(self, params=None):
        super().__init__()
        self.max_torque = 2.0
        self.dt = 0.05
        self.n_state = 3
        self.n_ctrl = 1

        if params is None:
            # g, m, l
            self.params = Variable(torch.Tensor((10., 1., 1.)))
        else:
            self.params = params
        assert len(self.params) == 3

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

        if x.is_cuda and not self.params.is_cuda:
            self.params = self.params.cuda()
        g, m, l = torch.unbind(self.params)

        u = torch.clamp(u, -self.max_torque, self.max_torque)[:,0]
        cos_th, sin_th, dth = torch.unbind(x, dim=1)
        th = torch.atan2(sin_th, cos_th)
        newdth = dth + self.dt*(-3.*g/(2.*l) * (-sin_th) + 3. / (m*l**2)*u)
        newth = th + newdth*self.dt
        state = torch.stack((torch.cos(newth), torch.sin(newth), newdth), dim=1)

        if squeeze:
            state = state.squeeze(0)
        return state

    def get_frame(self, x):
        x = util.get_data_maybe(x.view(-1))
        assert len(x) == 3
        g, m, l = torch.unbind(self.params)
        l = l.data[0]

        cos_th, sin_th, dth = torch.unbind(x)
        th = np.arctan2(sin_th, cos_th)
        x = sin_th*l
        y = cos_th*l
        fig, ax = plt.subplots(figsize=(6,6))
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
