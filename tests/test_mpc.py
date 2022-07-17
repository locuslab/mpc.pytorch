#!/usr/bin/env python3

import torch
from torch.autograd import Function, Variable, grad
from torch.nn.parameter import Parameter

import numpy as np
import numpy.random as npr
import numpy.testing as npt
from numpy.testing import dec

import cvxpy as cp

import numdifftools as nd

import gc
import os

from mpc import mpc, util, pnqp
from mpc.dynamics import NNDynamics, AffineDynamics
from mpc.lqr_step import LQRStep
from mpc.mpc import GradMethods, QuadCost, LinDx

def lqr_qp_cp(C, c, lower, upper):
    n = c.shape[0]
    x = cp.Variable(n)
    obj = 0.5*cp.quad_form(x, C) + cp.sum(cp.multiply(c, x))
    cons = [lower <= x, x <= upper]
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve()
    assert 'optimal' in prob.status
    return np.array(x.value)


def lqr_cp(C, c, F, f, x_init, T, n_state, n_ctrl, u_lower, u_upper):
    """Solve min_{tau={x,u}} sum_t 0.5 tau_t^T C_t tau_t + c_t^T tau_t
                        s.t. x_{t+1} = A_t x_t + B_t u_t + f_t
                             x_0 = x_init
                             u_lower <= u <= u_upper
    """
    tau = cp.Variable((n_state+n_ctrl, T))
    assert (u_lower is None) == (u_upper is None)

    objs = []
    x0 = tau[:n_state,0]
    u0 = tau[n_state:,0]
    cons = [x0 == x_init]
    for t in range(T):
        xt = tau[:n_state,t]
        ut = tau[n_state:,t]
        objs.append(0.5*cp.quad_form(tau[:,t], C[t]) +
                    cp.sum(cp.multiply(c[t], tau[:,t])))
        if u_lower is not None:
            cons += [u_lower[t] <= ut, ut <= u_upper[t]]
        if t+1 < T:
            xtp1 = tau[:n_state, t+1]
            cons.append(xtp1 == F[t]*tau[:,t]+f[t])
    prob = cp.Problem(cp.Minimize(sum(objs)), cons)
    # prob.solve(solver=cp.SCS, verbose=True)
    prob.solve()
    assert 'optimal' in prob.status
    return np.array(tau.value), np.array([obj_t.value for obj_t in objs])


def test_lqr_qp():
    npr.seed(1)

    n_batch = 2
    n = 100
    C = npr.randn(n_batch, n, n)
    C = np.matmul(C.transpose(0, 2, 1), C)
    c = npr.randn(n_batch, n)
    lower = -npr.random((n_batch, n))
    upper = npr.random((n_batch, n))

    opt_cp0 = lqr_qp_cp(C[0], c[0], lower[0], upper[0])
    opt_cp1 = lqr_qp_cp(C[1], c[1], lower[1], upper[1])

    C, c, lower, upper = [
        torch.Tensor(x).double() if x is not None else None
        for x in [C, c, lower, upper]
    ]

    t = pnqp.pnqp(C, c, lower, upper)
    opt_pnqp = t[0]

    npt.assert_allclose(opt_cp0, opt_pnqp[0].numpy(), rtol=1e-3)
    npt.assert_allclose(opt_cp1, opt_pnqp[1].numpy(), rtol=1e-3)


def test_lqr_linear_unbounded():
    npr.seed(1)

    n_batch = 2
    n_state, n_ctrl = 3, 4
    n_sc = n_state + n_ctrl
    T = 5
    C = npr.randn(T, n_batch, n_sc, n_sc)
    C = np.matmul(C.transpose(0, 1, 3, 2), C)
    c = npr.randn(T, n_batch, n_sc)
    alpha = 0.2
    R = np.tile(np.eye(n_state)+alpha*np.random.randn(n_state, n_state),
                (T, n_batch, 1, 1))
    S = np.tile(np.random.randn(n_state, n_ctrl), (T, n_batch, 1, 1))
    F = np.concatenate((R, S), axis=3)
    f = np.tile(npr.randn(n_state), (T, n_batch, 1))
    x_init = npr.randn(n_batch, n_state)
    # u_lower = -100.*npr.random((T, n_batch, n_ctrl))
    # u_upper = 100.*npr.random((T, n_batch, n_ctrl))
    u_lower = -1e4*np.ones((T, n_batch, n_ctrl))
    u_upper = 1e4*np.ones((T, n_batch, n_ctrl))

    tau_cp, objs_cp = lqr_cp(
        C[:,0], c[:,0], F[:,0], f[:,0], x_init[0], T, n_state, n_ctrl,
        None, None
    )
    tau_cp = tau_cp.T
    x_cp = tau_cp[:,:n_state]
    u_cp = tau_cp[:,n_state:]

    C, c, R, S, F, f, x_init, u_lower, u_upper = [
        Variable(torch.Tensor(x).double()) if x is not None else None
        for x in [C, c, R, S, F, f, x_init, u_lower, u_upper]
    ]

    dynamics = AffineDynamics(R[0,0], S[0,0], f[0,0])

    u_lqr = None
    x_lqr, u_lqr, objs_lqr = mpc.MPC(
        n_state, n_ctrl, T, u_lower, u_upper, u_lqr,
        lqr_iter=10,
        backprop=False,
        verbose=1,
        exit_unconverged=True,
    )(x_init, QuadCost(C, c), dynamics)
    tau_lqr = torch.cat((x_lqr, u_lqr), 2)
    tau_lqr = util.get_data_maybe(tau_lqr)
    npt.assert_allclose(tau_cp, tau_lqr[:,0].numpy(), rtol=1e-3)

    u_lqr = None
    x_lqr, u_lqr, objs_lqr = mpc.MPC(
        n_state, n_ctrl, T, None, None, u_lqr,
        lqr_iter=10,
        backprop=False,
        exit_unconverged=False,
    )(x_init, QuadCost(C, c), dynamics)
    tau_lqr = torch.cat((x_lqr, u_lqr), 2)
    tau_lqr = util.get_data_maybe(tau_lqr)
    npt.assert_allclose(tau_cp, tau_lqr[:,0].numpy(), rtol=1e-3)


def test_lqr_linear_bounded():
    npr.seed(1)

    n_batch = 2
    n_state, n_ctrl, T = 3, 4, 5
    # n_state, n_ctrl, T = 50, 20, 30
    n_sc = n_state + n_ctrl
    C = npr.randn(T, n_batch, n_sc, n_sc)
    C = np.matmul(C.transpose(0, 1, 3, 2), C)
    c = npr.randn(T, n_batch, n_sc)
    alpha = 0.2
    R = np.tile(np.eye(n_state)+alpha*np.random.randn(n_state, n_state),
                (T, n_batch, 1, 1))
    S = np.tile(np.random.randn(n_state, n_ctrl), (T, n_batch, 1, 1))
    F = np.concatenate((R, S), axis=3)
    f = np.tile(npr.randn(n_state), (T, n_batch, 1))
    x_init = npr.randn(n_batch, n_state)
    u_lower = -npr.random((T, n_batch, n_ctrl))
    u_upper = npr.random((T, n_batch, n_ctrl))

    tau_cp, objs_cp = lqr_cp(
        C[:,0], c[:,0], F[:,0], f[:,0], x_init[0], T, n_state, n_ctrl,
        u_lower[:,0], u_upper[:,0],
    )
    tau_cp = tau_cp.T
    x_cp = tau_cp[:,:n_state]
    u_cp = tau_cp[:,n_state:]

    C, c, R, S, F, f, x_init, u_lower, u_upper = [
        Variable(torch.Tensor(x).double()) if x is not None else None
        for x in [C, c, R, S, F, f, x_init, u_lower, u_upper]
    ]
    dynamics = AffineDynamics(R[0,0], S[0,0], f[0,0])

    x_lqr, u_lqr, objs_lqr = mpc.MPC(
        n_state, n_ctrl, T, u_lower, u_upper,
        lqr_iter=20, verbose=1,
        backprop=False,
        exit_unconverged=False,
    )(x_init, QuadCost(C, c), dynamics)
    tau_lqr = util.get_data_maybe(torch.cat((x_lqr, u_lqr), 2))

    npt.assert_allclose(tau_cp, tau_lqr[:,0].numpy(), rtol=1e-3)


def test_lqr_linear_bounded_delta():
    npr.seed(1)

    n_batch = 2
    n_state, n_ctrl, T = 3, 4, 5
    n_sc = n_state + n_ctrl
    C = npr.randn(T, n_batch, n_sc, n_sc)
    C = np.matmul(C.transpose(0, 1, 3, 2), C)
    c = npr.randn(T, n_batch, n_sc)
    alpha = 0.2
    R = np.tile(np.eye(n_state)+alpha*np.random.randn(n_state, n_state),
                (T, n_batch, 1, 1))
    S = 0.01*np.tile(np.random.randn(n_state, n_ctrl), (T, n_batch, 1, 1))
    F = np.concatenate((R, S), axis=3)
    f = np.tile(npr.randn(n_state), (T, n_batch, 1))
    x_init = npr.randn(n_batch, n_state)
    u_lower = -npr.random((T, n_batch, n_ctrl))
    u_upper = npr.random((T, n_batch, n_ctrl))

    tau_cp, objs_cp = lqr_cp(
        C[:,0], c[:,0], F[:,0], f[:,0], x_init[0], T, n_state, n_ctrl,
        u_lower[:,0], u_upper[:,0],
    )
    tau_cp = tau_cp.T
    x_cp = tau_cp[:,:n_state]
    u_cp = tau_cp[:,n_state:]

    C, c, R, S, F, f, x_init, u_lower, u_upper = [
        Variable(torch.Tensor(x).double()) if x is not None else None
        for x in [C, c, R, S, F, f, x_init, u_lower, u_upper]
    ]
    dynamics = AffineDynamics(R[0,0], S[0,0], f[0,0])

    delta_u = 0.1
    x_lqr, u_lqr, objs_lqr = mpc.MPC(
        n_state, n_ctrl, T, u_lower, u_upper,
        lqr_iter=1, verbose=1,
        delta_u=delta_u,
        backprop=False,
        exit_unconverged=False,
    )(x_init, QuadCost(C, c), dynamics)

    u_lqr = util.get_data_maybe(u_lqr)
    assert torch.abs(u_lqr).max() <= delta_u


@dec.skipif(not torch.cuda.is_available())
def test_lqr_cuda_singleton():
    npr.seed(1)

    n_batch = 5
    n_state, n_ctrl = 3, 1
    n_sc = n_state + n_ctrl
    T = 5
    C = npr.randn(T, n_batch, n_sc, n_sc)
    C = np.matmul(C.transpose(0, 1, 3, 2), C)
    c = npr.randn(T, n_batch, n_sc)
    alpha = 0.2
    R = np.tile(np.eye(n_state)+alpha*np.random.randn(n_state, n_state),
                (T, n_batch, 1, 1))
    S = np.tile(np.random.randn(n_state, n_ctrl), (T, n_batch, 1, 1))
    F = np.concatenate((R, S), axis=3)
    f = np.tile(npr.randn(n_state), (T, n_batch, 1))
    x_init = npr.randn(n_batch, n_state)
    # u_lower = -100.*npr.random((T, n_batch, n_ctrl))
    # u_upper = 100.*npr.random((T, n_batch, n_ctrl))
    u_lower = -1e4*np.ones((T, n_batch, n_ctrl))
    u_upper = 1e4*np.ones((T, n_batch, n_ctrl))

    tau_cp, objs_cp = lqr_cp(
        C[:,0], c[:,0], F[:,0], f[:,0], x_init[0], T, n_state, n_ctrl,
        None, None
    )
    tau_cp = tau_cp.T
    x_cp = tau_cp[:,:n_state]
    u_cp = tau_cp[:,n_state:]

    C, c, R, S, F, f, x_init, u_lower, u_upper = [
        Variable(torch.Tensor(x).double().cuda()) if x is not None else None
        for x in [C, c, R, S, F, f, x_init, u_lower, u_upper]
    ]

    dynamics = AffineDynamics(R[0,0], S[0,0], f[0,0])

    u_lqr = None
    x_lqr, u_lqr, objs_lqr = mpc.MPC(
        n_state, n_ctrl, T, u_lower, u_upper, u_lqr,
        lqr_iter=10,
        backprop=False,
    )(x_init, QuadCost(C, c), dynamics)
    tau_lqr = torch.cat((x_lqr, u_lqr), 2)
    tau_lqr = util.get_data_maybe(torch.cat((x_lqr, u_lqr), 2))
    npt.assert_allclose(tau_cp, tau_lqr[:,0].cpu().numpy(), rtol=1e-3)

    u_lqr = None
    x_lqr, u_lqr, objs_lqr = mpc.MPC(
        n_state, n_ctrl, T, None, None, u_lqr,
        lqr_iter=10,
        backprop=False,
    )(x_init, QuadCost(C, c), dynamics)
    tau_lqr = torch.cat((x_lqr, u_lqr), 2)
    tau_lqr = util.get_data_maybe(torch.cat((x_lqr, u_lqr), 2))
    npt.assert_allclose(tau_cp, tau_lqr[:,0].cpu().numpy(), rtol=1e-3)


# TODO: Lots of duplicated code here. Should clean up.
def test_lqr_backward_cost_linear_dynamics_unconstrained():
    npr.seed(0)
    torch.manual_seed(0)
    n_batch, n_state, n_ctrl, T = 1, 2, 2, 3
    hidden_sizes = [10, 10]
    n_sc = n_state + n_ctrl

    C = 10.*npr.randn(T, n_batch, n_sc, n_sc).astype(np.float64)
    C = np.matmul(C.transpose(0, 1, 3, 2), C)
    c = 10.*npr.randn(T, n_batch, n_sc).astype(np.float64)

    x_init = npr.randn(n_batch, n_state).astype(np.float64)
    beta = 100.
    u_lower = -beta*np.ones((T, n_batch, n_ctrl)).astype(np.float64)
    u_upper = beta*np.ones((T, n_batch, n_ctrl)).astype(np.float64)

    F = npr.randn(T-1, n_batch, n_state, n_sc)
    f = npr.randn(T-1, n_batch, n_state)

    def forward_numpy(C, c, x_init, u_lower, u_upper, F, f):
        _C, _c, _x_init, _u_lower, _u_upper, F, f = [
            Variable(torch.Tensor(x).double()) if x is not None else None
            for x in [C, c, x_init, u_lower, u_upper, F, f]
        ]

        u_init = None
        x_lqr, u_lqr, objs_lqr = mpc.MPC(
            n_state, n_ctrl, T, _u_lower, _u_upper, u_init,
            lqr_iter=40,
            verbose=1,
            exit_unconverged=False,
            backprop=False,
            max_linesearch_iter=2,
        )(_x_init, QuadCost(_C, _c), LinDx(F, f))
        return util.get_data_maybe(u_lqr.view(-1)).numpy()

    def f_c(c_flat):
        c_ = c_flat.reshape(T, n_batch, n_sc)
        return forward_numpy(C, c_, x_init, u_lower, u_upper, F, f)

    def f_F(F_flat):
        F_ = F_flat.reshape(T-1, n_batch, n_state, n_sc)
        return forward_numpy(C, c, x_init, u_lower, u_upper, F_ ,f)

    def f_f(f_flat):
        f_ = f_flat.reshape(T-1, n_batch, n_state)
        return forward_numpy(C, c, x_init, u_lower, u_upper, F, f_)

    u = forward_numpy(C, c, x_init, u_lower, u_upper, F, f)

    # Make sure the solution is not on the boundary.
    assert np.all(u != u_lower.reshape(-1)) and np.all(u != u_upper.reshape(-1))

    du_dc_fd = nd.Jacobian(f_c)(c.reshape(-1))
    du_dF_fd = nd.Jacobian(f_F)(F.reshape(-1))
    du_df_fd = nd.Jacobian(f_f)(f.reshape(-1))

    _C, _c, _x_init, _u_lower, _u_upper, F, f = [
        Variable(torch.Tensor(x).double(), requires_grad=True)
        if x is not None else None
        for x in [C, c, x_init, u_lower, u_upper, F, f]
    ]

    u_init = None
    x_lqr, u_lqr, objs_lqr = mpc.MPC(
        n_state, n_ctrl, T, _u_lower, _u_upper, u_init,
        lqr_iter=20,
        verbose=1,
        exit_unconverged=False,
    )(_x_init, QuadCost(_C, _c), LinDx(F, f))
    u_lqr = u_lqr.view(-1)

    du_dC = []
    du_dc = []
    du_dF = []
    du_df = []
    for i in range(len(u_lqr)):
        dCi = grad(u_lqr[i], [_C], retain_graph=True)[0].view(-1)
        dci = grad(u_lqr[i], [_c], retain_graph=True)[0].view(-1)
        dF = grad(u_lqr[i], [F], retain_graph=True)[0].view(-1)
        df = grad(u_lqr[i], [f], retain_graph=True)[0].view(-1)
        du_dC.append(dCi)
        du_dc.append(dci)
        du_dF.append(dF)
        du_df.append(df)
    du_dC = torch.stack(du_dC).data.numpy()
    du_dc = torch.stack(du_dc).data.numpy()
    du_dF = torch.stack(du_dF).data.numpy()
    du_df = torch.stack(du_df).data.numpy()

    npt.assert_allclose(du_dc_fd, du_dc, atol=1e-4)
    npt.assert_allclose(du_dF, du_dF_fd, atol=1e-4)
    npt.assert_allclose(du_df, du_df_fd, atol=1e-4)


def test_lqr_backward_cost_linear_dynamics_constrained():
    npr.seed(0)
    torch.manual_seed(0)
    n_batch, n_state, n_ctrl, T = 1, 2, 2, 3
    hidden_sizes = [10, 10]
    n_sc = n_state + n_ctrl

    C = 10.*npr.randn(T, n_batch, n_sc, n_sc).astype(np.float64)
    C = np.matmul(C.transpose(0, 1, 3, 2), C)
    c = 10.*npr.randn(T, n_batch, n_sc).astype(np.float64)

    x_init = npr.randn(n_batch, n_state).astype(np.float64)
    beta = 0.5
    u_lower = -beta*np.ones((T, n_batch, n_ctrl)).astype(np.float64)
    u_upper = beta*np.ones((T, n_batch, n_ctrl)).astype(np.float64)

    F = npr.randn(T-1, n_batch, n_state, n_sc)
    f = npr.randn(T-1, n_batch, n_state)

    def forward_numpy(C, c, x_init, u_lower, u_upper, F, f):
        _C, _c, _x_init, _u_lower, _u_upper, F, f = [
            Variable(torch.Tensor(x).double()) if x is not None else None
            for x in [C, c, x_init, u_lower, u_upper, F, f]
        ]

        u_init = None
        x_lqr, u_lqr, objs_lqr = mpc.MPC(
            n_state, n_ctrl, T, _u_lower, _u_upper, u_init,
            lqr_iter=40,
            verbose=1,
            exit_unconverged=True,
            backprop=False,
            max_linesearch_iter=2,
        )(_x_init, QuadCost(_C, _c), LinDx(F, f))
        return util.get_data_maybe(u_lqr.view(-1)).numpy()

    def f_c(c_flat):
        c_ = c_flat.reshape(T, n_batch, n_sc)
        return forward_numpy(C, c_, x_init, u_lower, u_upper, F, f)

    def f_F(F_flat):
        F_ = F_flat.reshape(T-1, n_batch, n_state, n_sc)
        return forward_numpy(C, c, x_init, u_lower, u_upper, F_, f)

    def f_f(f_flat):
        f_ = f_flat.reshape(T-1, n_batch, n_state)
        return forward_numpy(C, c, x_init, u_lower, u_upper, F, f_)

    def f_x_init(x_init):
        x_init = x_init.reshape(1, -1)
        return forward_numpy(C, c, x_init, u_lower, u_upper, F, f)

    u = forward_numpy(C, c, x_init, u_lower, u_upper, F, f)

    # Make sure the solution is strictly partially on the boundary.
    assert np.any(u == u_lower.reshape(-1)) or np.any(u == u_upper.reshape(-1))
    assert np.any((u != u_lower.reshape(-1)) & (u != u_upper.reshape(-1)))

    du_dc_fd = nd.Jacobian(f_c)(c.reshape(-1))
    du_dF_fd = nd.Jacobian(f_F)(F.reshape(-1))
    du_df_fd = nd.Jacobian(f_f)(f.reshape(-1))
    du_dxinit_fd = nd.Jacobian(f_x_init)(x_init[0])

    _C, _c, _x_init, _u_lower, _u_upper, F, f = [
        Variable(torch.Tensor(x).double(), requires_grad=True)
        if x is not None else None
        for x in [C, c, x_init, u_lower, u_upper, F, f]
    ]

    u_init = None
    x_lqr, u_lqr, objs_lqr = mpc.MPC(
        n_state, n_ctrl, T, _u_lower, _u_upper, u_init,
        lqr_iter=20,
        verbose=1,
    )(_x_init, QuadCost(_C, _c), LinDx(F, f))
    u_lqr_flat = u_lqr.view(-1)

    du_dC = []
    du_dc = []
    du_dF = []
    du_df = []
    du_dx_init = []
    for i in range(len(u_lqr_flat)):
        dCi = grad(u_lqr_flat[i], [_C], retain_graph=True)[0].view(-1)
        dci = grad(u_lqr_flat[i], [_c], retain_graph=True)[0].view(-1)
        dF = grad(u_lqr_flat[i], [F], retain_graph=True)[0].view(-1)
        df = grad(u_lqr_flat[i], [f], retain_graph=True)[0].view(-1)
        dx_init = grad(u_lqr_flat[i], [_x_init], retain_graph=True)[0].view(-1)
        du_dC.append(dCi)
        du_dc.append(dci)
        du_dF.append(dF)
        du_df.append(df)
        du_dx_init.append(dx_init)
    du_dC = torch.stack(du_dC).data.numpy()
    du_dc = torch.stack(du_dc).data.numpy()
    du_dF = torch.stack(du_dF).data.numpy()
    du_df = torch.stack(du_df).data.numpy()
    du_dx_init = torch.stack(du_dx_init).data.numpy()

    npt.assert_allclose(du_dc_fd, du_dc, atol=1e-4)
    npt.assert_allclose(du_dF, du_dF_fd, atol=1e-4)
    npt.assert_allclose(du_df, du_df_fd, atol=1e-4)
    npt.assert_allclose(du_dx_init, du_dxinit_fd, atol=1e-4)


def test_lqr_backward_cost_affine_dynamics_module_constrained():
    npr.seed(0)
    torch.manual_seed(0)
    n_batch, n_state, n_ctrl, T = 1, 2, 2, 2
    hidden_sizes = [10]
    n_sc = n_state + n_ctrl

    C = 10.*npr.randn(T, n_batch, n_sc, n_sc).astype(np.float64)
    C = np.matmul(C.transpose(0, 1, 3, 2), C)
    c = 10.*npr.randn(T, n_batch, n_sc).astype(np.float64)

    x_init = npr.randn(n_batch, n_state).astype(np.float64)
    # beta = 0.5
    beta = 2.0
    u_lower = -beta*np.ones((T, n_batch, n_ctrl)).astype(np.float64)
    u_upper = beta*np.ones((T, n_batch, n_ctrl)).astype(np.float64)

    _C, _c, _x_init, _u_lower, _u_upper = [
        Variable(torch.Tensor(x).double(), requires_grad=True)
        if x is not None else None
        for x in [C, c, x_init, u_lower, u_upper]
    ]
    F = Variable(
        torch.randn(1, 1, n_state, n_sc).repeat(T-1, 1, 1, 1).double(),
        requires_grad=True)
    dynamics = AffineDynamics(F[0,0,:,:n_state], F[0,0,:,n_state:])

    u_init = None
    x_lqr, u_lqr, objs_lqr = mpc.MPC(
        n_state, n_ctrl, T, _u_lower, _u_upper, u_init,
        lqr_iter=20,
        verbose=1,
    )(_x_init, QuadCost(_C, _c), LinDx(F))
    u_lqr_flat = u_lqr.view(-1)

    du_dF = []
    for i in range(len(u_lqr_flat)):
        dF = grad(u_lqr_flat[i], [F], retain_graph=True)[0].view(-1)
        du_dF.append(dF)
    du_dF = torch.stack(du_dF).data.numpy()

    u_init = None
    x_lqr, u_lqr, objs_lqr = mpc.MPC(
        n_state, n_ctrl, T, _u_lower, _u_upper, u_init,
        lqr_iter=20,
        verbose=1,
    )(_x_init, QuadCost(_C, _c), dynamics)
    u_lqr_flat = u_lqr.view(-1)

    du_dF_ = []
    for i in range(len(u_lqr_flat)):
        dF = grad(u_lqr_flat[i], [F], retain_graph=True)[0].view(-1)
        du_dF_.append(dF)
    du_dF_ = torch.stack(du_dF_).data.numpy()

    npt.assert_allclose(du_dF, du_dF_, atol=1e-4)

def test_lqr_backward_cost_nn_dynamics_module_constrained():
    npr.seed(0)
    torch.manual_seed(0)
    n_batch, n_state, n_ctrl, T = 1, 2, 2, 2
    hidden_sizes = [10, 10]
    n_sc = n_state + n_ctrl

    C = 10.*npr.randn(T, n_batch, n_sc, n_sc).astype(np.float64)
    C = np.matmul(C.transpose(0, 1, 3, 2), C)
    c = 10.*npr.randn(T, n_batch, n_sc).astype(np.float64)

    x_init = npr.randn(n_batch, n_state).astype(np.float64)
    beta = 1.
    u_lower = -beta*np.ones((T, n_batch, n_ctrl)).astype(np.float64)
    u_upper = beta*np.ones((T, n_batch, n_ctrl)).astype(np.float64)

    dynamics = NNDynamics(
        n_state, n_ctrl, hidden_sizes, activation='sigmoid').double()
    fc0b = dynamics.fcs[0].bias.view(-1).data.numpy().copy()

    def forward_numpy(C, c, x_init, u_lower, u_upper, fc0b):
        _C, _c, _x_init, _u_lower, _u_upper, fc0b = [
            Variable(torch.Tensor(x).double()) if x is not None else None
            for x in [C, c, x_init, u_lower, u_upper, fc0b]
        ]

        dynamics.fcs[0].bias.data[:] = fc0b.data
        # dynamics.A.data[:] = fc0b.view(n_state, n_state).data
        u_init = None
        x_lqr, u_lqr, objs_lqr = mpc.MPC(
            n_state, n_ctrl, T, _u_lower, _u_upper, u_init,
            lqr_iter=40,
            verbose=-1,
            exit_unconverged=True,
            backprop=False,
            max_linesearch_iter=1,
        )(_x_init, QuadCost(_C, _c), dynamics)
        return util.get_data_maybe(u_lqr.view(-1)).numpy()

    def f_c(c_flat):
        c_ = c_flat.reshape(T, n_batch, n_sc)
        return forward_numpy(C, c_, x_init, u_lower, u_upper, fc0b)

    def f_fc0b(fc0b):
        return forward_numpy(C, c, x_init, u_lower, u_upper, fc0b)

    u = forward_numpy(C, c, x_init, u_lower, u_upper, fc0b)

    # Make sure the solution is strictly partially on the boundary.
    assert np.any(u == u_lower.reshape(-1)) or np.any(u == u_upper.reshape(-1))
    assert np.any((u != u_lower.reshape(-1)) & (u != u_upper.reshape(-1)))

    du_dc_fd = nd.Jacobian(f_c)(c.reshape(-1))
    du_dfc0b_fd = nd.Jacobian(f_fc0b)(fc0b.reshape(-1))

    dynamics.fcs[0].bias.data = torch.DoubleTensor(fc0b).clone()

    _C, _c, _x_init, _u_lower, _u_upper, fc0b = [
        Variable(torch.Tensor(x).double(), requires_grad=True)
        if x is not None else None
        for x in [C, c, x_init, u_lower, u_upper, fc0b]
    ]

    u_init = None
    x_lqr, u_lqr, objs_lqr = mpc.MPC(
        n_state, n_ctrl, T, _u_lower, _u_upper, u_init,
        lqr_iter=20,
        verbose=-1,
        max_linesearch_iter=1,
        grad_method=GradMethods.ANALYTIC,
    )(_x_init, QuadCost(_C, _c), dynamics)
    u_lqr_flat = u_lqr.view(-1)

    du_dC = []
    du_dc = []
    du_dfc0b = []
    for i in range(len(u_lqr_flat)):
        dCi = grad(u_lqr_flat[i], [_C], retain_graph=True)[0].view(-1)
        dci = grad(u_lqr_flat[i], [_c], retain_graph=True)[0].view(-1)
        dfc0b = grad(u_lqr_flat[i], [dynamics.fcs[0].bias],
                     retain_graph=True)[0].view(-1)
        du_dC.append(dCi)
        du_dc.append(dci)
        du_dfc0b.append(dfc0b)
    du_dC = torch.stack(du_dC).data.numpy()
    du_dc = torch.stack(du_dc).data.numpy()
    du_dfc0b = torch.stack(du_dfc0b).data.numpy()

    npt.assert_allclose(du_dc_fd, du_dc, atol=1e-3)
    npt.assert_allclose(du_dfc0b_fd, du_dfc0b, atol=1e-3)


def test_lqr_backward_cost_nn_dynamics_module_constrained_slew():
    npr.seed(0)
    torch.manual_seed(0)
    n_batch, n_state, n_ctrl, T = 1, 2, 2, 2
    hidden_sizes = [10, 10]
    n_sc = n_state + n_ctrl

    C = 10.*npr.randn(T, n_batch, n_sc, n_sc).astype(np.float64)
    C = np.matmul(C.transpose(0, 1, 3, 2), C)
    c = 10.*npr.randn(T, n_batch, n_sc).astype(np.float64)

    x_init = npr.randn(n_batch, n_state).astype(np.float64)
    beta = 1.
    u_lower = -beta*np.ones((T, n_batch, n_ctrl)).astype(np.float64)
    u_upper = beta*np.ones((T, n_batch, n_ctrl)).astype(np.float64)

    dynamics = NNDynamics(
        n_state, n_ctrl, hidden_sizes, activation='sigmoid').double()
    fc0b = dynamics.fcs[0].bias.view(-1).data.numpy().copy()

    def forward_numpy(C, c, x_init, u_lower, u_upper, fc0b):
        _C, _c, _x_init, _u_lower, _u_upper, fc0b = [
            Variable(torch.Tensor(x).double(), requires_grad=True)
            if x is not None else None
            for x in [C, c, x_init, u_lower, u_upper, fc0b]
        ]

        dynamics.fcs[0].bias.data[:] = fc0b.data
        # dynamics.A.data[:] = fc0b.view(n_state, n_state).data
        u_init = None
        x_lqr, u_lqr, objs_lqr = mpc.MPC(
            n_state, n_ctrl, T, _u_lower, _u_upper, u_init,
            lqr_iter=40,
            verbose=-1,
            exit_unconverged=True,
            backprop=False,
            max_linesearch_iter=1,
            slew_rate_penalty=1.0,
        )(_x_init, QuadCost(_C, _c), dynamics)
        return util.get_data_maybe(u_lqr.view(-1)).numpy()

    def f_c(c_flat):
        c_ = c_flat.reshape(T, n_batch, n_sc)
        return forward_numpy(C, c_, x_init, u_lower, u_upper, fc0b)

    def f_fc0b(fc0b):
        return forward_numpy(C, c, x_init, u_lower, u_upper, fc0b)

    u = forward_numpy(C, c, x_init, u_lower, u_upper, fc0b)

    # Make sure the solution is strictly partially on the boundary.
    assert np.any(u == u_lower.reshape(-1)) or np.any(u == u_upper.reshape(-1))
    assert np.any((u != u_lower.reshape(-1)) & (u != u_upper.reshape(-1)))

    du_dc_fd = nd.Jacobian(f_c)(c.reshape(-1))
    du_dfc0b_fd = nd.Jacobian(f_fc0b)(fc0b.reshape(-1))

    dynamics.fcs[0].bias.data = torch.DoubleTensor(fc0b).clone()

    _C, _c, _x_init, _u_lower, _u_upper, fc0b = [
        Variable(torch.Tensor(x).double(), requires_grad=True)
        if x is not None else None
        for x in [C, c, x_init, u_lower, u_upper, fc0b]
    ]

    u_init = None
    x_lqr, u_lqr, objs_lqr = mpc.MPC(
        n_state, n_ctrl, T, _u_lower, _u_upper, u_init,
        lqr_iter=20,
        verbose=-1,
        max_linesearch_iter=1,
        grad_method=GradMethods.ANALYTIC,
        slew_rate_penalty=1.0,
    )(_x_init, QuadCost(_C, _c), dynamics)
    u_lqr_flat = u_lqr.view(-1)

    du_dC = []
    du_dc = []
    du_dfc0b = []
    for i in range(len(u_lqr_flat)):
        dCi = grad(u_lqr_flat[i], [_C], retain_graph=True)[0].contiguous().view(-1)
        dci = grad(u_lqr_flat[i], [_c], retain_graph=True)[0].contiguous().view(-1)
        dfc0b = grad(u_lqr_flat[i], [dynamics.fcs[0].bias],
                     retain_graph=True)[0].view(-1)
        du_dC.append(dCi)
        du_dc.append(dci)
        du_dfc0b.append(dfc0b)
    du_dC = torch.stack(du_dC).data.numpy()
    du_dc = torch.stack(du_dc).data.numpy()
    du_dfc0b = torch.stack(du_dfc0b).data.numpy()

    npt.assert_allclose(du_dc_fd, du_dc, atol=1e-3)
    npt.assert_allclose(du_dfc0b_fd, du_dfc0b, atol=1e-3)


def test_lqr_linearization():
    npr.seed(0)
    torch.manual_seed(0)
    n_batch, n_state, n_ctrl, T = 2, 3, 4, 5
    hidden_sizes = [10]
    n_sc = n_state + n_ctrl

    C = 10.*npr.randn(T, n_batch, n_sc, n_sc).astype(np.float64)
    C = np.matmul(C.transpose(0, 1, 3, 2), C)
    c = 10.*npr.randn(T, n_batch, n_sc).astype(np.float64)

    x_init = npr.randn(n_batch, n_state).astype(np.float64)
    # beta = 0.5
    beta = 2.0
    u_lower = -beta*np.ones((T, n_batch, n_ctrl)).astype(np.float64)
    u_upper = beta*np.ones((T, n_batch, n_ctrl)).astype(np.float64)

    _C, _c, _x_init, _u_lower, _u_upper = [
        Variable(torch.Tensor(x).double(), requires_grad=True)
        if x is not None else None
        for x in [C, c, x_init, u_lower, u_upper]
    ]
    F = Variable(
        torch.randn(1, 1, n_state, n_sc).repeat(T-1, 1, 1, 1).double(),
        requires_grad=True)
    dynamics = NNDynamics(
        n_state, n_ctrl, hidden_sizes, activation='sigmoid').double()

    u_init = None
    _lqr = mpc.MPC(
        n_state, n_ctrl, T, _u_lower, _u_upper, u_init,
        grad_method=GradMethods.ANALYTIC,
    )

    u = torch.randn(T, n_batch, n_ctrl).type_as(_x_init.data)
    x = util.get_traj(T, u, x_init=_x_init, dynamics=dynamics)
    Fan, fan = _lqr.linearize_dynamics(x, u, dynamics, diff=False)

    _lqr.grad_method=GradMethods.AUTO_DIFF
    Fau, fau = _lqr.linearize_dynamics(x, u, dynamics, diff=False)
    npt.assert_allclose(Fan.data.numpy(), Fau.data.numpy(), atol=1e-4)
    npt.assert_allclose(fan.data.numpy(), fau.data.numpy(), atol=1e-4)

    # Make sure diff version doesn't crash:
    Fau, fau = _lqr.linearize_dynamics(x, u, dynamics, diff=True)

    _lqr.grad_method=GradMethods.FINITE_DIFF
    Ff, ff = _lqr.linearize_dynamics(x, u, dynamics, diff=False)
    npt.assert_allclose(Fan.data.numpy(), Ff.data.numpy(), atol=1e-4)
    npt.assert_allclose(fan.data.numpy(), ff.data.numpy(), atol=1e-4)

    # Make sure diff version doesn't crash:
    Ff, ff = _lqr.linearize_dynamics(x, u, dynamics, diff=True)


def test_lqr_slew_rate():
    n_batch = 2
    n_state, n_ctrl = 3, 4
    n_sc = n_state + n_ctrl
    T = 5
    alpha = 0.2

    torch.manual_seed(1)
    C = torch.randn(T, n_batch, n_sc, n_sc)
    C = C.transpose(2,3).matmul(C)
    c = torch.randn(T, n_batch, n_sc)
    x_init = torch.randn(n_batch, n_state)
    R = torch.eye(n_state) + alpha*torch.randn(n_state, n_state)
    S = torch.randn(n_state, n_ctrl)
    f = torch.randn(n_state)
    C, c, x_init, R, S, f = map(Variable, (C, c, x_init, R, S, f))

    dynamics = AffineDynamics(R, S, f)

    x, u, objs = mpc.MPC(
        n_state, n_ctrl, T,
        u_lower=None, u_upper=None, u_init=None,
        lqr_iter=10,
        backprop=False,
        verbose=1,
        exit_unconverged=False,
        eps=1e-4,
    )(x_init, QuadCost(C, c), dynamics)

    # The solution should be the same when the slew rate approaches 0.
    x_slew_eps, u_slew_eps, objs_slew_eps = mpc.MPC(
        n_state, n_ctrl, T,
        u_lower=None, u_upper=None, u_init=None,
        lqr_iter=10,
        backprop=False,
        verbose=1,
        exit_unconverged=False,
        eps=1e-4,
        slew_rate_penalty=1e-6,
    )(x_init, QuadCost(C, c), dynamics)

    npt.assert_allclose(x.data.numpy(), x_slew_eps.data.numpy(), atol=1e-3)
    npt.assert_allclose(u.data.numpy(), u_slew_eps.data.numpy(), atol=1e-3)

    x_slew, u_slew, objs_slew= mpc.MPC(
        n_state, n_ctrl, T,
        u_lower=None, u_upper=None, u_init=None,
        lqr_iter=10,
        backprop=False,
        verbose=1,
        exit_unconverged=False,
        eps=1e-4,
        slew_rate_penalty=1.,
    )(x_init, QuadCost(C, c), dynamics)

    assert np.alltrue((objs < objs_slew).numpy())

    d = torch.norm(u[:-1] - u[1:]).item()
    d_slew = torch.norm(u_slew[:-1] - u_slew[1:]).item()
    assert d_slew < d


def test_memory():
    import psutil

    torch.manual_seed(0)

    n_batch, n_state, n_ctrl, T = 2, 3, 4, 5
    n_sc = n_state + n_ctrl

    # Randomly initialize a PSD quadratic cost and linear dynamics.
    C = torch.randn(T*n_batch, n_sc, n_sc)
    C = torch.bmm(C, C.transpose(1, 2)).view(T, n_batch, n_sc, n_sc)
    c = torch.randn(T, n_batch, n_sc)

    alpha = 0.2
    R = (torch.eye(n_state)+alpha*torch.randn(n_state, n_state)).repeat(T, n_batch, 1, 1)
    S = torch.randn(T, n_batch, n_state, n_ctrl)
    F = torch.cat((R, S), dim=3)

    # The initial state.
    x_init = torch.randn(n_batch, n_state)

    # The upper and lower control bounds.
    u_lower = -torch.rand(T, n_batch, n_ctrl)
    u_upper = torch.rand(T, n_batch, n_ctrl)

    process = psutil.Process(os.getpid())

    # gc.collect()
    # start_mem = process.memory_info().rss

    # _lqr = LQRStep(
    #     n_state=n_state,
    #     n_ctrl=n_ctrl,
    #     T=T,
    #     u_lower=u_lower,
    #     u_upper=u_upper,
    #     u_zero_I=u_zero_I,
    #     true_cost=cost,
    #     true_dynamics=dynamics,
    #     delta_u=delta_u,
    #     delta_space=True,
    #     # current_x=x,
    #     # current_u=u,
    # )
    # e = Variable(torch.Tensor())
    # x, u = _lqr(x_init, C, c, F, f if f is not None else e)

    # gc.collect()
    # mem_used = process.memory_info().rss - start_mem
    # print(mem_used)
    # assert mem_used == 0

    gc.collect()
    start_mem = process.memory_info().rss

    _mpc = mpc.MPC(
        n_state=n_state,
        n_ctrl=n_ctrl,
        T=T,
        u_lower=u_lower,
        u_upper=u_upper,
        lqr_iter=20,
        verbose=1,
        backprop=False,
        exit_unconverged=False,
    )
    _mpc(x_init, QuadCost(C, c), LinDx(F))
    del _mpc

    gc.collect()
    mem_used = process.memory_info().rss - start_mem
    print(mem_used)
    assert mem_used == 0


if __name__=='__main__':
    import sys
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(mode='Verbose',
         color_scheme='Linux', call_pdb=1)

    test_lqr_qp()
    test_lqr_linear_unbounded()
    test_lqr_linear_bounded()
    test_lqr_linear_bounded_delta()
    test_lqr_backward_cost_linear_dynamics_unconstrained()
    test_lqr_backward_cost_linear_dynamics_constrained()
    test_lqr_backward_cost_affine_dynamics_module_constrained()
    test_lqr_backward_cost_nn_dynamics_module_constrained()
    test_lqr_backward_cost_nn_dynamics_module_constrained_slew()
    test_lqr_linearization()
    test_lqr_slew_rate()

    # test_lqr_cuda_singleton()
    # test_memory()
