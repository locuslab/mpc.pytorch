import torch
from torch.autograd import Function, Variable
from torch.nn import Module
from torch.nn.parameter import Parameter

import operator

def jacobian(f, x, eps):
    if x.ndimension() == 2:
        assert x.size(0) == 1
        x = x.squeeze()

    e = Variable(torch.eye(len(x)).type_as(get_data_maybe(x)))
    J = []
    for i in range(len(x)):
        J.append((f(x + eps*e[i]) - f(x - eps*e[i]))/(2.*eps))
    J = torch.stack(J).transpose(0,1)
    return J


def expandParam(X, n_batch, nDim):
    if X.ndimension() in (0, nDim):
        return X, False
    elif X.ndimension() == nDim - 1:
        return X.unsqueeze(0).expand(*([n_batch] + list(X.size()))), True
    else:
        raise RuntimeError("Unexpected number of dimensions.")


def bdiag(d):
    nBatch, sz = d.size()
    dtype = d.type() if not isinstance(d, Variable) else d.data.type()
    D = torch.zeros(nBatch, sz, sz).type(dtype)
    I = torch.eye(sz).repeat(nBatch, 1, 1).type(dtype).byte()
    D[I] = d.squeeze()
    return D


def bger(x, y):
    return x.unsqueeze(2).bmm(y.unsqueeze(1))


def bmv(X, y):
    return X.bmm(y.unsqueeze(2)).squeeze(2)


def bquad(x, Q):
    return x.unsqueeze(1).bmm(Q).bmm(x.unsqueeze(2)).squeeze()


def bdot(x, y):
    return torch.bmm(x.unsqueeze(1), y.unsqueeze(2)).squeeze()


def eclamp(x, lower, upper):
    # In-place!!
    if type(lower) == type(x):
        assert x.size() == lower.size()

    if type(upper) == type(x):
        assert x.size() == upper.size()

    I = x < lower
    x[I] = lower[I] if not isinstance(lower, float) else lower

    I = x > upper
    x[I] = upper[I] if not isinstance(upper, float) else upper

    return x


def get_data_maybe(x):
    return x if not isinstance(x, Variable) else x.data


_seen_tables = []
def table_log(tag, d):
    # TODO: There's probably a better way to handle formatting here,
    # or a better way altogether to replace this quick hack.
    global _seen_tables

    def print_row(r):
        print('| ' + ' | '.join(r) + ' |')

    if tag not in _seen_tables:
        print_row(map(operator.itemgetter(0), d))
        _seen_tables.append(tag)

    s = []
    for di in d:
        assert len(di) in [2,3]
        if len(di) == 3:
            e, fmt = di[1:]
            s.append(fmt.format(e))
        else:
            e = di[1]
            s.append(str(e))
    print_row(s)


def get_traj(T, u, x_init, dynamics=None, F=None, f=None):
    assert (dynamics is None) != (F is None)

    F = get_data_maybe(F)
    f = get_data_maybe(f)
    x = [get_data_maybe(x_init)]
    for t in range(T):
        xt = x[t]
        ut = get_data_maybe(u[t])
        if t < T-1:
            # new_x = f(Variable(xt), Variable(ut)).data
            if dynamics is not None:
                new_x = dynamics(Variable(xt), Variable(ut)).data
            else:
                xut = torch.cat((xt, ut), 1)
                new_x = bmv(F[t], xut)
                if f is not None:
                    new_x += f
            x.append(new_x)
    return x


def get_cost(T, C, c, u, x_init=None, x=None, dynamics=None, F=None, f=None):
    assert (dynamics is None) != (F is None)

    C = get_data_maybe(C)
    c = get_data_maybe(c)

    if x is None:
        x = get_traj(T, u, x_init, dynamics, F, f)
    objs = []
    for t in range(T):
        xt = x[t]
        ut = u[t]
        xut = torch.cat((xt, ut), 1)
        obj = 0.5*bquad(xut, C[t]) + bdot(xut, c[t])
        objs.append(obj)
    objs = torch.stack(objs, dim=0)
    total_obj = torch.sum(objs, dim=0)
    return total_obj
