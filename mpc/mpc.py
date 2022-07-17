import torch
from torch.autograd import Function, Variable
from torch.nn import Module
from torch.nn.parameter import Parameter

import numpy as np
import numpy.random as npr

from collections import namedtuple

from enum import Enum

import sys

from . import util
from .pnqp import pnqp
from .lqr_step import LQRStep
from .dynamics import CtrlPassthroughDynamics


QuadCost = namedtuple('QuadCost', 'C c')
LinDx = namedtuple('LinDx', 'F f')

# https://stackoverflow.com/questions/11351032
QuadCost.__new__.__defaults__ = (None,) * len(QuadCost._fields)
LinDx.__new__.__defaults__ = (None,) * len(LinDx._fields)


class GradMethods(Enum):
    AUTO_DIFF = 1
    FINITE_DIFF = 2
    ANALYTIC = 3
    ANALYTIC_CHECK = 4


class SlewRateCost(Module):
    """Hacky way of adding the slew rate penalty to costs."""
    # TODO: It would be cleaner to update this to just use the slew
    # rate penalty instead of # slew_C
    def __init__(self, cost, slew_C, n_state, n_ctrl):
        super().__init__()
        self.cost = cost
        self.slew_C = slew_C
        self.n_state = n_state
        self.n_ctrl = n_ctrl

    def forward(self, tau):
        true_tau = tau[:, self.n_ctrl:]
        true_cost = self.cost(true_tau)
        # The slew constraints are time-invariant.
        slew_cost = 0.5 * util.bquad(tau, self.slew_C[0])
        return true_cost + slew_cost

    def grad_input(self, x, u):
        raise NotImplementedError("Implement grad_input")


class MPC(Module):
    """A differentiable box-constrained iLQR solver.

    This provides a differentiable solver for the following box-constrained
    control problem with a quadratic cost (defined by C and c) and
    non-linear dynamics (defined by f):

        min_{tau={x,u}} sum_t 0.5 tau_t^T C_t tau_t + c_t^T tau_t
                        s.t. x_{t+1} = f(x_t, u_t)
                            x_0 = x_init
                            u_lower <= u <= u_upper

    This implements the Control-Limited Differential Dynamic Programming
    paper with a first-order approximation to the non-linear dynamics:
    https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf

    Some of the notation here is from Sergey Levine's notes:
    http://rll.berkeley.edu/deeprlcourse/f17docs/lecture_8_model_based_planning.pdf

    Required Args:
        n_state, n_ctrl, T

    Optional Args:
        u_lower, u_upper: The lower- and upper-bounds on the controls.
            These can either be floats or shaped as [T, n_batch, n_ctrl]
        u_init: The initial control sequence, useful for warm-starting:
            [T, n_batch, n_ctrl]
        lqr_iter: The number of LQR iterations to perform.
        grad_method: The method to compute the Jacobian of the dynamics.
            GradMethods.ANALYTIC: Use a manually-defined Jacobian.
                + Fast and accurate, use this if possible
            GradMethods.AUTO_DIFF: Use PyTorch's autograd.
                + Slow
            GradMethods.FINITE_DIFF: Use naive finite differences
                + Inaccurate
        delta_u (float): The amount each component of the controls
            is allowed to change in each LQR iteration.
        verbose (int):
            -1: No output or warnings
             0: Warnings
            1+: Detailed iteration info
        eps: Termination threshold, on the norm of the full control
             step (without line search)
        back_eps: `eps` value to use in the backwards pass.
        n_batch: May be necessary for now if it can't be inferred.
                 TODO: Infer, potentially remove this.
        linesearch_decay (float): Multiplicative decay factor for the
            line search.
        max_linesearch_iter (int): Can be used to disable the line search
            if 1 is used for some problems the line search can
            be harmful.
        exit_unconverged: Assert False if a fixed point is not reached.
        detach_unconverged: Detach examples from the graph that do
            not hit a fixed point so they are not differentiated through.
        backprop: Allow the solver to be differentiated through.
        slew_rate_penalty (float): Penalty term applied to
            ||u_t - u_{t+1}||_2^2 in the objective.
        prev_ctrl: The previous nominal control sequence to initialize
            the solver with.
        not_improved_lim: The number of iterations to allow that don't
            improve the objective before returning early.
        best_cost_eps: Absolute threshold for the best cost
            to be updated.
    """

    def __init__(
            self, n_state, n_ctrl, T,
            u_lower=None, u_upper=None,
            u_zero_I=None,
            u_init=None,
            lqr_iter=10,
            grad_method=GradMethods.ANALYTIC,
            delta_u=None,
            verbose=0,
            eps=1e-7,
            back_eps=1e-7,
            n_batch=None,
            linesearch_decay=0.2,
            max_linesearch_iter=10,
            exit_unconverged=True,
            detach_unconverged=True,
            backprop=True,
            slew_rate_penalty=None,
            prev_ctrl=None,
            not_improved_lim=5,
            best_cost_eps=1e-4
    ):
        super().__init__()

        assert (u_lower is None) == (u_upper is None)
        assert max_linesearch_iter > 0

        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.T = T
        self.u_lower = u_lower
        self.u_upper = u_upper

        if not isinstance(u_lower, float):
            self.u_lower = util.detach_maybe(self.u_lower)

        if not isinstance(u_upper, float):
            self.u_upper = util.detach_maybe(self.u_upper)

        self.u_zero_I = util.detach_maybe(u_zero_I)
        self.u_init = util.detach_maybe(u_init)
        self.lqr_iter = lqr_iter
        self.grad_method = grad_method
        self.delta_u = delta_u
        self.verbose = verbose
        self.eps = eps
        self.back_eps = back_eps
        self.n_batch = n_batch
        self.linesearch_decay = linesearch_decay
        self.max_linesearch_iter = max_linesearch_iter
        self.exit_unconverged = exit_unconverged
        self.detach_unconverged = detach_unconverged
        self.backprop = backprop
        self.not_improved_lim = not_improved_lim
        self.best_cost_eps = best_cost_eps

        self.slew_rate_penalty = slew_rate_penalty
        self.prev_ctrl = prev_ctrl


    # @profile
    def forward(self, x_init, cost, dx):
        # QuadCost.C: [T, n_batch, n_tau, n_tau]
        # QuadCost.c: [T, n_batch, n_tau]
        assert isinstance(cost, QuadCost) or \
            isinstance(cost, Module) or isinstance(cost, Function)
        assert isinstance(dx, LinDx) or \
            isinstance(dx, Module) or isinstance(dx, Function)

        # TODO: Clean up inferences, expansions, and assumptions made here.
        if self.n_batch is not None:
            n_batch = self.n_batch
        elif isinstance(cost, QuadCost) and cost.C.ndimension() == 4:
            n_batch = cost.C.size(1)
        else:
            print('MPC Error: Could not infer batch size, pass in as n_batch')
            sys.exit(-1)


        # if c.ndimension() == 2:
        #     c = c.unsqueeze(1).expand(self.T, n_batch, -1)

        if isinstance(cost, QuadCost):
            C, c = cost
            if C.ndimension() == 2:
                # Add the time and batch dimensions.
                C = C.unsqueeze(0).unsqueeze(0).expand(
                    self.T, n_batch, self.n_state+self.n_ctrl, -1)
            elif C.ndimension() == 3:
                # Add the batch dimension.
                C = C.unsqueeze(1).expand(
                    self.T, n_batch, self.n_state+self.n_ctrl, -1)

            if c.ndimension() == 1:
                # Add the time and batch dimensions.
                c = c.unsqueeze(0).unsqueeze(0).expand(self.T, n_batch, -1)
            elif c.ndimension() == 2:
                # Add the batch dimension.
                c = c.unsqueeze(1).expand(self.T, n_batch, -1)

            if C.ndimension() != 4 or c.ndimension() != 3:
                print('MPC Error: Unexpected QuadCost shape.')
                sys.exit(-1)
            cost = QuadCost(C, c)

        assert x_init.ndimension() == 2 and x_init.size(0) == n_batch

        if self.u_init is None:
            u = torch.zeros(self.T, n_batch, self.n_ctrl).type_as(x_init.data)
        else:
            u = self.u_init
            if u.ndimension() == 2:
                u = u.unsqueeze(1).expand(self.T, n_batch, -1).clone()
        u = u.type_as(x_init.data)

        if self.verbose > 0:
            print('Initial mean(cost): {:.4e}'.format(
                torch.mean(util.get_cost(
                    self.T, u, cost, dx, x_init=x_init
                )).item()
            ))

        best = None

        n_not_improved = 0
        for i in range(self.lqr_iter):
            u = Variable(util.detach_maybe(u), requires_grad=True)
            # Linearize the dynamics around the current trajectory.
            x = util.get_traj(self.T, u, x_init=x_init, dynamics=dx)
            if isinstance(dx, LinDx):
                F, f = dx.F, dx.f
            else:
                F, f = self.linearize_dynamics(
                    x, util.detach_maybe(u), dx, diff=False)

            if isinstance(cost, QuadCost):
                C, c = cost.C, cost.c
            else:
                C, c, _ = self.approximate_cost(
                    x, util.detach_maybe(u), cost, diff=False)

            x, u, n_total_qp_iter, costs, full_du_norm, mean_alphas = \
              self.solve_lqr_subproblem(x_init, C, c, F, f, cost, dx, x, u)
            n_not_improved += 1

            assert x.ndimension() == 3
            assert u.ndimension() == 3

            if best is None:
                best = {
                    'x': list(torch.split(x, split_size_or_sections=1, dim=1)),
                    'u': list(torch.split(u, split_size_or_sections=1, dim=1)),
                    'costs': costs,
                    'full_du_norm': full_du_norm,
                }
            else:
                for j in range(n_batch):
                    if costs[j] <= best['costs'][j] + self.best_cost_eps:
                        n_not_improved = 0
                        best['x'][j] = x[:,j].unsqueeze(1)
                        best['u'][j] = u[:,j].unsqueeze(1)
                        best['costs'][j] = costs[j]
                        best['full_du_norm'][j] = full_du_norm[j]

            if self.verbose > 0:
                util.table_log('lqr', (
                    ('iter', i),
                    ('mean(cost)', torch.mean(best['costs']).item(), '{:.4e}'),
                    ('||full_du||_max', max(full_du_norm).item(), '{:.2e}'),
                    # ('||alpha_du||_max', max(alpha_du_norm), '{:.2e}'),
                    # TODO: alphas, total_qp_iters here is for the current
                    # iterate, not the best
                    ('mean(alphas)', mean_alphas.item(), '{:.2e}'),
                    ('total_qp_iters', n_total_qp_iter),
                ))

            if max(full_du_norm) < self.eps or \
               n_not_improved > self.not_improved_lim:
                break


        x = torch.cat(best['x'], dim=1)
        u = torch.cat(best['u'], dim=1)
        full_du_norm = best['full_du_norm']

        if isinstance(dx, LinDx):
            F, f = dx.F, dx.f
        else:
            F, f = self.linearize_dynamics(x, u, dx, diff=True)

        if isinstance(cost, QuadCost):
            C, c = cost.C, cost.c
        else:
            C, c, _ = self.approximate_cost(x, u, cost, diff=True)

        x, u = self.solve_lqr_subproblem(
            x_init, C, c, F, f, cost, dx, x, u, no_op_forward=True)

        if self.detach_unconverged:
            if max(best['full_du_norm']) > self.eps:
                if self.exit_unconverged:
                    assert False

                if self.verbose >= 0:
                    print("LQR Warning: All examples did not converge to a fixed point.")
                    print("Detaching and *not* backpropping through the bad examples.")

                I = full_du_norm < self.eps
                Ix = Variable(I.unsqueeze(0).unsqueeze(2).expand_as(x)).type_as(x.data)
                Iu = Variable(I.unsqueeze(0).unsqueeze(2).expand_as(u)).type_as(u.data)
                x = x*Ix + x.clone().detach()*(1.-Ix)
                u = u*Iu + u.clone().detach()*(1.-Iu)

        costs = best['costs']
        return (x, u, costs)

    def solve_lqr_subproblem(self, x_init, C, c, F, f, cost, dynamics, x, u,
                             no_op_forward=False):
        if self.slew_rate_penalty is None or isinstance(cost, Module):
            _lqr = LQRStep(
                n_state=self.n_state,
                n_ctrl=self.n_ctrl,
                T=self.T,
                u_lower=self.u_lower,
                u_upper=self.u_upper,
                u_zero_I=self.u_zero_I,
                true_cost=cost,
                true_dynamics=dynamics,
                delta_u=self.delta_u,
                linesearch_decay=self.linesearch_decay,
                max_linesearch_iter=self.max_linesearch_iter,
                delta_space=True,
                current_x=x,
                current_u=u,
                back_eps=self.back_eps,
                no_op_forward=no_op_forward,
            )
            e = Variable(torch.Tensor())
            return _lqr(x_init, C, c, F, f if f is not None else e)
        else:
            nsc = self.n_state + self.n_ctrl
            _n_state = nsc
            _nsc = _n_state + self.n_ctrl
            n_batch = C.size(1)
            _C = torch.zeros(self.T, n_batch, _nsc, _nsc).type_as(C)
            half_gamI = self.slew_rate_penalty*torch.eye(
                self.n_ctrl).unsqueeze(0).unsqueeze(0).repeat(self.T, n_batch, 1, 1)
            _C[:,:,:self.n_ctrl,:self.n_ctrl] = half_gamI
            _C[:,:,-self.n_ctrl:,:self.n_ctrl] = -half_gamI
            _C[:,:,:self.n_ctrl,-self.n_ctrl:] = -half_gamI
            _C[:,:,-self.n_ctrl:,-self.n_ctrl:] = half_gamI
            slew_C = _C.clone()
            _C = _C + torch.nn.ZeroPad2d((self.n_ctrl, 0, self.n_ctrl, 0))(C)

            _c = torch.cat((
                torch.zeros(self.T, n_batch, self.n_ctrl).type_as(c),c), 2)

            _F0 = torch.cat((
                torch.zeros(self.n_ctrl, self.n_state+self.n_ctrl),
                torch.eye(self.n_ctrl),
            ), 1).type_as(F).unsqueeze(0).unsqueeze(0).repeat(
                self.T-1, n_batch, 1, 1
            )
            _F1 = torch.cat((
                torch.zeros(
                    self.T-1, n_batch, self.n_state, self.n_ctrl
                ).type_as(F),F), 3)
            _F = torch.cat((_F0, _F1), 2)

            if f is not None:
                _f = torch.cat((
                    torch.zeros(self.T-1, n_batch, self.n_ctrl).type_as(f),f), 2)
            else:
                _f = Variable(torch.Tensor())

            u_data = util.detach_maybe(u)
            if self.prev_ctrl is not None:
                prev_u = self.prev_ctrl
                if prev_u.ndimension() == 1:
                    prev_u = prev_u.unsqueeze(0)
                if prev_u.ndimension() == 2:
                    prev_u = prev_u.unsqueeze(0)
                prev_u = prev_u.data
            else:
                prev_u = torch.zeros(1, n_batch, self.n_ctrl).type_as(u)
            utm1s = torch.cat((prev_u, u_data[:-1])).clone()
            _x = torch.cat((utm1s, x), 2)

            _x_init = torch.cat((Variable(prev_u[0]), x_init), 1)

            if not isinstance(dynamics, LinDx):
                _dynamics = CtrlPassthroughDynamics(dynamics)
            else:
                _dynamics = None

            if isinstance(cost, QuadCost):
                _true_cost = QuadCost(_C, _c)
            else:
                _true_cost = SlewRateCost(
                    cost, slew_C, self.n_state, self.n_ctrl
                )

            _lqr = LQRStep(
                n_state=_n_state,
                n_ctrl=self.n_ctrl,
                T=self.T,
                u_lower=self.u_lower,
                u_upper=self.u_upper,
                u_zero_I=self.u_zero_I,
                true_cost=_true_cost,
                true_dynamics=_dynamics,
                delta_u=self.delta_u,
                linesearch_decay=self.linesearch_decay,
                max_linesearch_iter=self.max_linesearch_iter,
                delta_space=True,
                current_x=_x,
                current_u=u,
                back_eps=self.back_eps,
                no_op_forward=no_op_forward,
            )
            x, *rest = _lqr(_x_init, _C, _c, _F, _f)
            x = x[:,:,self.n_ctrl:]
            return [x] + rest

    def approximate_cost(self, x, u, Cf, diff=True):
        with torch.enable_grad():
            tau = torch.cat((x, u), dim=2).data
            tau = Variable(tau, requires_grad=True)
            if self.slew_rate_penalty is not None:
                print("""
MPC Error: Using a non-convex cost with a slew rate penalty is not yet implemented.
The current implementation does not correctly do a line search.
More details: https://github.com/locuslab/mpc.pytorch/issues/12
""")
                sys.exit(-1)
                differences = tau[1:, :, -self.n_ctrl:] - tau[:-1, :, -self.n_ctrl:]
                slew_penalty = (self.slew_rate_penalty * differences.pow(2)).sum(-1)
            costs = list()
            hessians = list()
            grads = list()
            for t in range(self.T):
                tau_t = tau[t]
                if self.slew_rate_penalty is not None:
                    cost = Cf(tau_t) + (slew_penalty[t-1] if t > 0 else 0)
                else:
                    cost = Cf(tau_t)

                grad = torch.autograd.grad(cost.sum(), tau_t,
                                           create_graph=True, retain_graph=True)[0]
                hessian = list()
                for v_i in range(tau.shape[2]):
                    hessian.append(
                        torch.autograd.grad(grad[:, v_i].sum(), tau_t,
                                            retain_graph=True)[0]
                    )
                hessian = torch.stack(hessian, dim=-1)
                costs.append(cost)
                grads.append(grad - util.bmv(hessian, tau_t))
                hessians.append(hessian)
            costs = torch.stack(costs, dim=0)
            grads = torch.stack(grads, dim=0)
            hessians = torch.stack(hessians, dim=0)
            if not diff:
                return hessians.data, grads.data, costs.data
            return hessians, grads, costs

    # @profile
    def linearize_dynamics(self, x, u, dynamics, diff):
        # TODO: Cleanup variable usage.

        n_batch = x[0].size(0)

        if self.grad_method == GradMethods.ANALYTIC:
            _u = Variable(u[:-1].view(-1, self.n_ctrl), requires_grad=True)
            _x = Variable(x[:-1].contiguous().view(-1, self.n_state),
                          requires_grad=True)

            # This inefficiently calls dynamics again, but is worth it because
            # we can efficiently compute grad_input for every time step at once.
            _new_x = dynamics(_x, _u)

            # This check is a little expensive and should only be done if
            # modifying this code.
            # assert torch.abs(_new_x.data - torch.cat(x[1:])).max() <= 1e-6

            if not diff:
                _new_x = _new_x.data
                _x = _x.data
                _u = _u.data

            R, S = dynamics.grad_input(_x, _u)

            f = _new_x - util.bmv(R, _x) - util.bmv(S, _u)
            f = f.view(self.T-1, n_batch, self.n_state)

            R = R.contiguous().view(self.T-1, n_batch, self.n_state, self.n_state)
            S = S.contiguous().view(self.T-1, n_batch, self.n_state, self.n_ctrl)
            F = torch.cat((R, S), 3)

            if not diff:
                F, f = list(map(Variable, [F, f]))
            return F, f
        else:
            # TODO: This is inefficient and confusing.
            x_init = x[0]
            x = [x_init]
            F, f = [], []
            for t in range(self.T):
                if t < self.T-1:
                    xt = Variable(x[t], requires_grad=True)
                    ut = Variable(u[t], requires_grad=True)
                    xut = torch.cat((xt, ut), 1)
                    new_x = dynamics(xt, ut)

                    # Linear dynamics approximation.
                    if self.grad_method in [GradMethods.AUTO_DIFF,
                                             GradMethods.ANALYTIC_CHECK]:
                        Rt, St = [], []
                        for j in range(self.n_state):
                            Rj, Sj = torch.autograd.grad(
                                new_x[:,j].sum(), [xt, ut],
                                retain_graph=True)
                            if not diff:
                                Rj, Sj = Rj.data, Sj.data
                            Rt.append(Rj)
                            St.append(Sj)
                        Rt = torch.stack(Rt, dim=1)
                        St = torch.stack(St, dim=1)

                        if self.grad_method == GradMethods.ANALYTIC_CHECK:
                            assert False # Not updated
                            Rt_autograd, St_autograd = Rt, St
                            Rt, St = dynamics.grad_input(xt, ut)
                            eps = 1e-8
                            if torch.max(torch.abs(Rt-Rt_autograd)).data[0] > eps or \
                            torch.max(torch.abs(St-St_autograd)).data[0] > eps:
                                print('''
        nmpc.ANALYTIC_CHECK error: The analytic derivative of the dynamics function may be off.
                                ''')
                            else:
                                print('''
        nmpc.ANALYTIC_CHECK: The analytic derivative of the dynamics function seems correct.
        Re-run with GradMethods.ANALYTIC to continue.
                                ''')
                            sys.exit(0)
                    elif self.grad_method == GradMethods.FINITE_DIFF:
                        Rt, St = [], []
                        for i in range(n_batch):
                            Ri = util.jacobian(
                                lambda s: dynamics(s, ut[i]), xt[i], 1e-4
                            )
                            Si = util.jacobian(
                                lambda a : dynamics(xt[i], a), ut[i], 1e-4
                            )
                            if not diff:
                                Ri, Si = Ri.data, Si.data
                            Rt.append(Ri)
                            St.append(Si)
                        Rt = torch.stack(Rt)
                        St = torch.stack(St)
                    else:
                        assert False

                    Ft = torch.cat((Rt, St), 2)
                    F.append(Ft)

                    if not diff:
                        xt, ut, new_x = xt.data, ut.data, new_x.data
                    ft = new_x - util.bmv(Rt, xt) - util.bmv(St, ut)
                    f.append(ft)

                if t < self.T-1:
                    x.append(util.detach_maybe(new_x))

            F = torch.stack(F, 0)
            f = torch.stack(f, 0)
            if not diff:
                F, f = list(map(Variable, [F, f]))
            return F, f
