import numpy as np

f = np.float32
H = f(1e-3)


def tanh(v):
    return np.tanh(v.astype(np.float32)).astype(np.float32)


def leaky(v, a=f(0.1)):
    return np.where(v > 0, v, a * v).astype(np.float32)


def leaky_d(v, a=f(0.1)):
    return np.where(v > 0, f(1), a).astype(np.float32)


class Dense:
    def __init__(self, n_in, n_out, act, act_bwd):
        self.n_in, self.n_out = n_in, n_out
        self.p = np.zeros(n_in * n_out + n_out, dtype=np.float32)
        self.grad = np.zeros_like(self.p)
        self.act, self.act_bwd = act, act_bwd

    def W(self, i, j):
        return self.p[i * self.n_in + j]

    def forward(self, x):
        self.x = x.astype(np.float32)
        z = np.zeros(self.n_out, dtype=np.float32)
        for i in range(self.n_out):
            s = self.p[self.n_in * self.n_out + i]
            for j in range(self.n_in):
                s = f(s + f(self.W(i, j) * self.x[j]))
            z[i] = s
        self.z = z
        self.y = self.act(z)
        return self.y

    def backward(self, g):
        g = g.astype(np.float32)
        d = self.act_bwd(self.z, self.y, g)
        gx = np.zeros(self.n_in, dtype=np.float32)
        for j in range(self.n_in):
            s = f(0)
            for i in range(self.n_out):
                s = f(s + f(self.W(i, j) * d[i]))
            gx[j] = s
        for i in range(self.n_out):
            for j in range(self.n_in):
                self.grad[i * self.n_in + j] = f(self.grad[i * self.n_in + j] + f(d[i] * self.x[j]))
            k = self.n_in * self.n_out + i
            self.grad[k] = f(self.grad[k] + d[i])
        return gx


def tanh_bwd(z, y, g):
    return (g * (f(1) - y * y)).astype(np.float32)


def leaky_bwd(z, y, g):
    return (g * leaky_d(z)).astype(np.float32)


def fmt(v):
    return "(" + ", ".join(f"{float(np.float32(x)):.7g}" for x in v) + ")"


def projected(layer, x, g):
    y = layer.forward(x)
    return f(sum(f(g[i] * y[i]) for i in range(len(g))))


def fd_params(layer, x, g):
    base = layer.p.copy()
    out = np.zeros_like(base)
    for k in range(len(base)):
        layer.p = base.copy(); layer.p[k] = f(base[k] + H); pp = projected(layer, x, g)
        layer.p = base.copy(); layer.p[k] = f(base[k] - H); pm = projected(layer, x, g)
        out[k] = f(f(pp - pm) / f(2 * H))
    layer.p = base
    return out


def fd_input(layer, x, g):
    out = np.zeros(len(x), dtype=np.float32)
    for j in range(len(x)):
        xp = x.copy(); xp[j] = f(x[j] + H)
        xm = x.copy(); xm[j] = f(x[j] - H)
        out[j] = f(f(projected(layer, xp, g) - projected(layer, xm, g)) / f(2 * H))
    return out


print("=== Layer fixture: Dense<float,3,2>, tanh, theta = W rows then b")
theta = np.array([0.1, -0.2, 0.3, 0.4, 0.5, -0.6, 0.2, -0.1], dtype=np.float32)
x1 = np.array([0.3, -0.7, 1.1], dtype=np.float32)
g1 = np.array([0.8, -1.3], dtype=np.float32)
x2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
g2 = np.array([-0.5, 0.25], dtype=np.float32)

L = Dense(3, 2, tanh, tanh_bwd)
L.p = theta.copy()
print("grad at construction+Forward:", fmt(L.grad))
y1 = L.forward(x1)
print("z1", fmt(L.z), "y1", fmt(y1))
d1 = tanh_bwd(L.z, y1, g1)
print("delta1", fmt(d1))
gx1 = L.backward(g1)
grad1 = L.grad.copy()
print("input grad 1", fmt(gx1))
print("param grad 1", fmt(grad1))
fd1 = fd_params(L, x1, g1)
print("FD param grad 1", fmt(fd1), "max err", float(np.max(np.abs(fd1 - grad1))))
fdx1 = fd_input(L, x1, g1)
print("FD input grad 1", fmt(fdx1), "max err", float(np.max(np.abs(fdx1 - gx1))))

y2 = L.forward(x2)
print("z2", fmt(L.z), "y2", fmt(y2))
d2 = tanh_bwd(L.z, y2, g2)
print("delta2", fmt(d2))
gx2 = L.backward(g2)
acc = L.grad.copy()
print("input grad 2 (per-sample, not accumulated)", fmt(gx2))
print("accumulated grad 1+2", fmt(acc))

S = Dense(3, 2, tanh, tanh_bwd)
S.p = theta.copy()
S.forward(x2)
S.backward(g2)
grad2 = S.grad.copy()
print("single-sample grad 2", fmt(grad2))
print("sum check max |acc-(g1+g2)|", float(np.max(np.abs(acc - (grad1 + grad2)))))
fd2 = fd_params(S, x2, g2)
print("FD param grad 2", fmt(fd2), "max err", float(np.max(np.abs(fd2 - grad2))))
fdx2 = fd_input(S, x2, g2)
print("FD input grad 2", fmt(fdx2), "max err", float(np.max(np.abs(fdx2 - gx2))))

print()
print("=== Base-reference forward with default Dense params (biases 0), x=(1,2,3), tanh")
B = Dense(3, 2, tanh, tanh_bwd)
B.p = np.array([0.1, -0.2, 0.3, 0.4, 0.5, -0.6, 0, 0], dtype=np.float32)
print("y", fmt(B.forward(x2)), "z", fmt(B.z))

print()
print("=== Model fixture: Dense<2,3>(LeakyReLU 0.1) -> Dense<3,1>(tanh), x=(2,0.5), upstream 1")
Hd = Dense(2, 3, leaky, leaky_bwd)
Od = Dense(3, 1, tanh, tanh_bwd)
Hd.p = np.array([0.5, -1.0, 1.5, 0.25, -0.5, 0.75, 0, 0, 0], dtype=np.float32)
Od.p = np.array([1.0, -0.5, 2.0, 0], dtype=np.float32)
xm = np.array([2.0, 0.5], dtype=np.float32)


def model_out(theta_all):
    Hd.p = theta_all[:9].copy(); Od.p = theta_all[9:].copy()
    return Od.forward(Hd.forward(xm))[0]


theta_m = np.concatenate([Hd.p, Od.p])
yo = model_out(theta_m)
print("hidden z", fmt(Hd.z), "hidden a", fmt(Hd.y), "out z", fmt(Od.z), "y", float(yo))
gh = Od.backward(np.array([1.0], dtype=np.float32))
gxm = Hd.backward(gh)
grads_m = np.concatenate([Hd.grad, Od.grad])
print("delta_out", fmt(tanh_bwd(Od.z, Od.y, np.array([1.0], dtype=np.float32))))
print("upstream to hidden", fmt(gh))
print("model input grad", fmt(gxm))
print("model Gradients()", fmt(grads_m))
fdm = np.zeros_like(theta_m)
for k in range(len(theta_m)):
    tp = theta_m.copy(); tp[k] = f(tp[k] + H)
    tm = theta_m.copy(); tm[k] = f(tm[k] - H)
    fdm[k] = f(f(model_out(tp) - model_out(tm)) / f(2 * H))
print("FD model Gradients()", fmt(fdm), "max err", float(np.max(np.abs(fdm - grads_m))))

print()
print("=== double-precision cross-check of layer grad 1 and model grads")
xd, gd = x1.astype(np.float64), g1.astype(np.float64)
Wd = theta[:6].astype(np.float64).reshape(2, 3); bd = theta[6:].astype(np.float64)
z = Wd @ xd + bd; y = np.tanh(z); dd = gd * (1 - y * y)
print("dW", np.outer(dd, xd).ravel(), "db", dd, "dx", Wd.T @ dd)
W1 = np.array([[0.5, -1.0], [1.5, 0.25], [-0.5, 0.75]]); W2 = np.array([[1.0, -0.5, 2.0]])
xx = np.array([2.0, 0.5]); zh = W1 @ xx; ah = np.where(zh > 0, zh, 0.1 * zh)
zo = W2 @ ah; yo = np.tanh(zo); do = 1 - yo ** 2
dh = (W2.T @ do) * np.where(zh > 0, 1.0, 0.1)
print("model grads f64", np.concatenate([np.outer(dh, xx).ravel(), dh, (do[0] * ah), do]))
