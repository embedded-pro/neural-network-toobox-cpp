import numpy as np

X, H = 2, 3
Wx = [[0.5, -0.3], [0.2, 0.8], [-0.6, 0.1]]
Wh = [[0.1, -0.4, 0.3], [0.5, 0.2, -0.1], [-0.3, 0.6, 0.25]]
b = [0.05, -0.1, 0.2]
theta = [v for r in Wx for v in r] + [v for r in Wh for v in r] + b
P = H * (X + H + 1)
assert len(theta) == P

xs = [(1.0, -0.5), (0.3, 0.9), (-0.7, 0.4)]
g = (0.7, -1.2, 0.5)
gs = [(0.3, 0.2, -0.4), (-0.6, 0.1, 0.9), g]
s0 = (0.5, -0.25, 0.8)
FD_H = 1e-3


class Rnn:
    def __init__(self, dt, K, th):
        self.dt, self.K = dt, K
        self.th = [dt(v) for v in th]
        self.G = [dt(0)] * P
        self.reset()

    def wx(self, i, j):
        return self.th[i * X + j]

    def wh(self, i, k):
        return self.th[H * X + i * H + k]

    def bias(self, i):
        return self.th[H * X + H * H + i]

    def reset(self):
        self.set_state((0.0,) * H)

    def set_state(self, s):
        self.out = [self.dt(v) for v in s]
        self.ring = []

    def forward(self, x):
        dt = self.dt
        x = [dt(v) for v in x]
        hp = list(self.out)
        self.ring.append((x, hp))
        if len(self.ring) > self.K:
            self.ring.pop(0)
        z = []
        for i in range(H):
            s = self.bias(i)
            for j in range(X):
                s = dt(s + dt(self.wx(i, j) * x[j]))
            for k in range(H):
                s = dt(s + dt(self.wh(i, k) * hp[k]))
            z.append(s)
        self.out = [dt(np.tanh(v)) for v in z]
        return list(self.out)

    def backward(self, gv):
        dt = self.dt
        one = dt(1)
        d = [dt(dt(gv[i]) * dt(one - dt(self.out[i] * self.out[i]))) for i in range(H)]
        ig = []
        for j in range(X):
            s = dt(0)
            for i in range(H):
                s = dt(s + dt(self.wx(i, j) * d[i]))
            ig.append(s)
        n = len(self.ring)
        for m in range(n):
            x, hp = self.ring[n - 1 - m]
            for i in range(H):
                for j in range(X):
                    self.G[i * X + j] = dt(self.G[i * X + j] + dt(d[i] * x[j]))
                for k in range(H):
                    idx = H * X + i * H + k
                    self.G[idx] = dt(self.G[idx] + dt(d[i] * hp[k]))
                idx = H * X + H * H + i
                self.G[idx] = dt(self.G[idx] + d[i])
            if m + 1 < n:
                nd = []
                for k in range(H):
                    s = dt(0)
                    for i in range(H):
                        s = dt(s + dt(self.wh(i, k) * d[i]))
                    nd.append(dt(s * dt(one - dt(hp[k] * hp[k]))))
                d = nd
        return ig


def dot(a, bb, dt):
    s = dt(0)
    for u, v in zip(a, bb):
        s = dt(s + dt(dt(u) * dt(v)))
    return s


def fmt(v):
    return "(" + ", ".join(f"{float(e):.7g}" for e in v) + ")"


def run_seq(dt, K, th, seq, state=None):
    r = Rnn(dt, K, th)
    if state is not None:
        r.set_state(state)
    outs = [r.forward(x) for x in seq]
    return r, outs


def fd_theta(dt, loss):
    num = []
    for k in range(P):
        tp = [dt(v) for v in theta]
        tm = [dt(v) for v in theta]
        tp[k] = dt(tp[k] + dt(FD_H))
        tm[k] = dt(tm[k] - dt(FD_H))
        num.append(dt(dt(loss(tp) - loss(tm)) / dt(2 * FD_H)))
    return num


def maxerr(a, bb):
    return max(abs(float(u) - float(v)) for u, v in zip(a, bb))


f32, f64 = np.float32, np.float64

print("== sizes: P =", P, " theta =", fmt(theta))

for dt, name in ((f32, "f32"), (f64, "f64")):
    _, outs = run_seq(dt, 4, theta, xs)
    print(f"A forward {name}: h1={fmt(outs[0])} h2={fmt(outs[1])} h3={fmt(outs[2])}")
    _, o = run_seq(dt, 4, theta, [xs[0]], s0)
    print(f"B setstate {name}: h={fmt(o[0])}")

r0, outs0 = run_seq(f64, 4, theta, xs)
pre1 = [0.05 + 0.5 * 1.0 - 0.3 * -0.5, -0.1 + 0.2 * 1.0 + 0.8 * -0.5, 0.2 - 0.6 * 1.0 + 0.1 * -0.5]
print("A z1 (f64) =", fmt(pre1))

r, outs = run_seq(f32, 4, theta, xs)
ig = r.backward(g)
num = []
for j in range(X):
    def L(delta):
        seq = [xs[0], xs[1], tuple(xs[2][q] + (delta if q == j else 0.0) for q in range(X))]
        seq = [tuple(f32(v) for v in s) for s in seq]
        _, o = run_seq(f32, 4, theta, seq)
        return dot(g, o[-1], f32)
    num.append(f32(f32(L(FD_H) - L(-FD_H)) / f32(2 * FD_H)))
print("C inputgrad analytic", fmt(ig), " fd", fmt(num), " maxerr", f"{maxerr(ig, num):.2g}")
delta3 = [f32(g[i]) * (f32(1) - outs[2][i] ** 2) for i in range(H)]
print("C delta3", fmt(delta3))

r, outs = run_seq(f32, 4, theta, xs)
r.backward(g)
full = r.G
num_full = fd_theta(f32, lambda th: dot(g, run_seq(f32, 4, th, xs)[1][-1], f32))
print("D fullbptt analytic", fmt(full))
print("D fullbptt fd      ", fmt(num_full), " maxerr", f"{maxerr(full, num_full):.2g}")
r64, _ = run_seq(f64, 4, theta, xs)
r64.backward(g)
print("D f64 analytic     ", fmt(r64.G), " f32-f64", f"{maxerr(full, r64.G):.2g}")

r, outs = run_seq(f32, 2, theta, xs)
r.backward(g)
trunc = r.G
h1 = outs[0]
num_trunc = fd_theta(f32, lambda th: dot(g, run_seq(f32, 2, th, xs[1:], h1)[1][-1], f32))
print("E trunc K=2 analytic", fmt(trunc))
print("E trunc fd (SetState(h1), replay x2,x3)", fmt(num_trunc), " maxerr", f"{maxerr(trunc, num_trunc):.2g}")
diffs = [float(a) - float(c) for a, c in zip(trunc, num_full)]
print("E trunc - full  ", fmt(diffs), " max |diff|", f"{max(abs(d) for d in diffs):.3g}",
      " argmax", int(np.argmax([abs(d) for d in diffs])))
print("E h1 (state before window)", fmt(h1))

r = Rnn(f32, 4, theta)
for x, gv in zip(xs, gs):
    r.forward(x)
    r.backward(gv)
acc = r.G


def Lsum(th):
    _, o = run_seq(f32, 4, th, xs)
    s = f32(0)
    for gv, h in zip(gs, o):
        s = f32(s + dot(gv, h, f32))
    return s


num_acc = fd_theta(f32, Lsum)
print("F many-to-many analytic", fmt(acc))
print("F many-to-many fd      ", fmt(num_acc), " maxerr", f"{maxerr(acc, num_acc):.2g}")
r = Rnn(f32, 4, theta)
r.forward(xs[0]); r.backward(gs[0])
print("F after step1 only G", fmt(r.G))

Wd, cd = (0.4, -0.7, 0.2), 0.1
r, outs = run_seq(f32, 4, theta, xs)
ys = []
for h in outs:
    zd = f32(f32(cd) + dot(Wd, h, f32))
    ys.append((float(zd), float(f32(np.tanh(zd)))))
print("H model (z, y) per step", ys)

Xe, He, Ke = 3, 16, 8
Pe = He * (Xe + He + 1)
ram = 2 * Pe + Ke * (Xe + He) + He + Xe
print(f"RAM example X={Xe} H={He} K={Ke}: P={Pe} 2P={2*Pe} ring={Ke*(Xe+He)} total={ram} floats = {4*ram} B")
fwd = He * Xe + He * He
bwd = Ke * (He * Xe + 2 * He * He) + He * Xe
print(f"  Forward MACs={fwd}  Backward MACs (full window, incl. input grad)={bwd}; last step skips one W_h^T delta: {bwd - He*He}")
print(f"  Flash variant RAM: H={He} floats; flash P={Pe} floats = {4*Pe} B")
Pf = H * (X + H + 1)
print(f"fixture ElmanRnn<float,2,3,4>: P={Pf} RAM floats={2*Pf + 4*(X+H) + H + X}")

r = Rnn(f32, 4, theta)
r.forward(xs[0]); r.forward(xs[1])
r.set_state(s0)
o = r.forward(xs[0])
r.backward(g)
num_b = fd_theta(f32, lambda th: dot(g, run_seq(f32, 4, th, [xs[0]], s0)[1][-1], f32))
print("B setstate out", fmt(o))
print("B setstate G analytic", fmt(r.G))
print("B setstate G fd      ", fmt(num_b), " maxerr", f"{maxerr(r.G, num_b):.2g}")
r2 = Rnn(f32, 4, theta)
r2.forward(xs[0]); r2.forward(xs[1]); r2.forward(xs[0]); r2.backward(g)
print("B without SetState (history kept) G", fmt(r2.G))

r = Rnn(f32, 1, theta)
for x in xs: r.forward(x)
r.backward(g)
print("K=1 G", fmt(r.G))
