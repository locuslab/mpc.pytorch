import logging
import math
import time

import gym
import numpy as np
import torch
import torch.autograd
from gym import wrappers, logger as gym_log
from mpc import mpc

gym_log.set_level(gym_log.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')

if __name__ == "__main__":
    ENV_NAME = "Pendulum-v0"
    TIMESTEPS = 10  # T
    N_BATCH = 1
    LQR_ITER = 50
    ACTION_LOW = -2.0
    ACTION_HIGH = 2.0
    d = "cpu"
    dtype = torch.double
    # new hyperparameters for approximate dynamics
    H_UNITS = 16
    TRAIN_EPOCH = 150
    BOOT_STRAP_ITER = 100

    nx = 2
    nu = 1

    network = torch.nn.Sequential(
        torch.nn.Linear(nx + nu, H_UNITS),
        torch.nn.Tanh(),
        torch.nn.Linear(H_UNITS, H_UNITS),
        torch.nn.Tanh(),
        torch.nn.Linear(H_UNITS, nx)
    ).double()


    class PendulumDynamics(torch.nn.Module):
        def forward(self, state, perturbed_action):
            u = torch.clamp(perturbed_action, ACTION_LOW, ACTION_HIGH)
            if state.dim() is 1 or u.dim() is 1:
                state = state.view(1, -1)
                u = u.view(1, -1)
            if u.shape[1] > 1:
                u = u[:, 0].view(-1, 1)
            xu = torch.cat((state, u), dim=1)
            state_residual = network(xu)
            next_state = state + state_residual
            next_state[:, 0] = angle_normalize(next_state[:, 0])
            return next_state


    def angle_normalize(x):
        return (((x + math.pi) % (2 * math.pi)) - math.pi)


    def true_dynamics(state, action):
        th = state[:, 0].view(-1, 1)
        thdot = state[:, 1].view(-1, 1)

        g = 10
        m = 1
        l = 1
        dt = 0.05

        u = action
        u = torch.clamp(u, -2, 2)

        newthdot = thdot + (-3 * g / (2 * l) * torch.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = torch.clamp(newthdot, -8, 8)

        state = torch.cat((angle_normalize(newth), newthdot), dim=1)
        return state


    dataset = None
    dynamics = PendulumDynamics()
    # create some true dynamics validation set to compare model against
    Nv = 1000
    statev = torch.cat(((torch.rand(Nv, 1, dtype=torch.double) - 0.5) * 2 * math.pi,
                        (torch.rand(Nv, 1, dtype=torch.double) - 0.5) * 16), dim=1)
    actionv = (torch.rand(Nv, 1, dtype=torch.double) - 0.5) * (ACTION_HIGH - ACTION_LOW)


    def angular_diff_batch(a, b):
        """Angle difference from b to a (a - b) assumes the angles are not different by more than 2pi"""
        d = a - b
        d[d > math.pi] -= 2 * math.pi
        d[d < -math.pi] += 2 * math.pi
        return d


    def train(new_data):
        global dataset
        # not normalized inside the simulator
        new_data[:, 0] = angle_normalize(new_data[:, 0])
        if not torch.is_tensor(new_data):
            new_data = torch.from_numpy(new_data)
        # clamp actions
        new_data[:, -1] = torch.clamp(new_data[:, -1], ACTION_LOW, ACTION_HIGH)
        # append data to whole dataset
        if dataset is None:
            dataset = new_data
        else:
            dataset = torch.cat((dataset, new_data), dim=0)

        # train on the whole dataset (assume small enough we can train on all together)
        XU = dataset.detach()
        dtheta = angular_diff_batch(XU[1:, 0], XU[:-1, 0])
        dtheta_dt = XU[1:, 1] - XU[:-1, 1]
        Y = torch.cat((dtheta.view(-1, 1), dtheta_dt.view(-1, 1)), dim=1)  # x' - x residual
        XU = XU[:-1]  # make same size as Y

        # thaw network
        for param in network.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(network.parameters())
        for epoch in range(TRAIN_EPOCH):
            optimizer.zero_grad()
            # MSE loss
            Yhat = network(XU)
            loss = (Y - Yhat).norm(2, dim=1) ** 2
            loss.mean().backward()
            optimizer.step()
            logger.debug("ds %d epoch %d loss %f", dataset.shape[0], epoch, loss.mean().item())

        # freeze network
        for param in network.parameters():
            param.requires_grad = False

        # evaluate network against true dynamics
        yt = true_dynamics(statev, actionv)
        yp = dynamics(statev, actionv)
        dtheta = angular_diff_batch(yp[:, 0], yt[:, 0])
        dtheta_dt = yp[:, 1] - yt[:, 1]
        E = torch.cat((dtheta.view(-1, 1), dtheta_dt.view(-1, 1)), dim=1).norm(dim=1)
        logger.info("Error with true dynamics theta %f theta_dt %f norm %f", dtheta.abs().mean(),
                    dtheta_dt.abs().mean(), E.mean())
        logger.debug("Start next collection sequence")


    u_init = None
    render = True
    retrain_after_iter = 50
    retrain_dynamics = train
    run_iter = 500

    downward_start = True
    env = gym.make(ENV_NAME).env  # bypass the default TimeLimit wrapper
    env.reset()
    if downward_start:
        env.state = [np.pi, 1]

    # bootstrap network with random actions
    if BOOT_STRAP_ITER:
        logger.info("bootstrapping with random action for %d actions", BOOT_STRAP_ITER)
        new_data = np.zeros((BOOT_STRAP_ITER, nx + nu))
        for i in range(BOOT_STRAP_ITER):
            pre_action_state = env.state
            action = np.random.uniform(low=ACTION_LOW, high=ACTION_HIGH)
            env.step([action])
            # env.render()
            new_data[i, :nx] = pre_action_state
            new_data[i, nx:] = action

        train(new_data)
        logger.info("bootstrapping finished")

    env = wrappers.Monitor(env, '/tmp/box_ddp_pendulum_approx/', force=True)
    env.reset()
    if downward_start:
        env.env.state = [np.pi, 1]

    # swingup goal (observe theta and theta_dt)
    goal_weights = torch.tensor((1., 0.1), dtype=dtype)  # nx
    goal_state = torch.tensor((0., 0.), dtype=dtype)  # nx
    ctrl_penalty = 0.001
    q = torch.cat((
        goal_weights,
        ctrl_penalty * torch.ones(nu, dtype=dtype)
    ))  # nx + nu
    px = -torch.sqrt(goal_weights) * goal_state
    p = torch.cat((px, torch.zeros(nu, dtype=dtype)))
    Q = torch.diag(q).repeat(TIMESTEPS, N_BATCH, 1, 1)  # T x B x nx+nu x nx+nu
    p = p.repeat(TIMESTEPS, N_BATCH, 1)
    cost = mpc.QuadCost(Q, p)  # T x B x nx+nu (linear component of cost)

    # run MPC
    collected_dataset = torch.zeros((retrain_after_iter, nx + nu), dtype=dtype)
    total_reward = 0
    for i in range(run_iter):
        state = env.state.copy()
        state = torch.tensor(state, dtype=dtype).view(1, -1)
        command_start = time.perf_counter()
        # recreate controller using updated u_init (kind of wasteful right?)
        ctrl = mpc.MPC(nx, nu, TIMESTEPS, u_lower=ACTION_LOW, u_upper=ACTION_HIGH, lqr_iter=LQR_ITER,
                       exit_unconverged=False, eps=1e-2,
                       n_batch=N_BATCH, backprop=False, verbose=0, u_init=u_init,
                       grad_method=mpc.GradMethods.AUTO_DIFF)

        # compute action based on current state, dynamics, and cost
        nominal_states, nominal_actions, nominal_objs = ctrl(state, cost, dynamics)
        action = nominal_actions[0]  # take first planned action
        u_init = torch.cat((nominal_actions[1:], torch.zeros(1, N_BATCH, nu, dtype=dtype)), dim=0)

        elapsed = time.perf_counter() - command_start
        s, r, _, _ = env.step(action.detach().numpy())
        total_reward += r
        logger.debug("action taken: %.4f cost received: %.4f time taken: %.5fs", action, -r, elapsed)
        if render:
            env.render()

        di = i % retrain_after_iter
        if di == 0 and i > 0:
            retrain_dynamics(collected_dataset)
        collected_dataset[di, :nx] = torch.tensor(state)
        collected_dataset[di, nx:] = action

    logger.info("Total reward %f", total_reward)
