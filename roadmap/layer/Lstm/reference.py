import numpy as np

X, H = 2, 3
G4 = 4 * H
Wx = [[0.5, -0.3], [0.2, 0.8], [-0.6, 0.1],
      [0.3, 0.4], [-0.2, 0.5], [0.7, -0.1],
      [0.9, -0.4], [-0.5, 0.6], [0.4, 0.3],
      [-0.3, 0.2], [0.6, -0.5], [0.1, 0.7]]
Wh = [[0.1, -0.4, 0.3], [0.5, 0.2, -0.1], [-0.3, 0.6, 0.25],
      [0.2, 0.1, -0.3], [-0.4, 0.3, 0.2], [0.1, -0.2, 0.5],
      [0.6, -0.2, 0.1], [0.3, 0.5, -0.4], [-0.1, 0.4, 0.7],
      [-0.2, 0.3, 0.4], [0.1, -0.5, 0.2], [0.4, 0.2, -0.3]]
b = [0.05, -0.1, 0.2, 1.0, 0.8, 1.2, 0.0, 0.15, -0.05, 0.1, -0.2, 0.3]
theta = [v for r in Wx for v in r] + [v for r in Wh for v in r] + b
P = 4 * (H * (X + H) + H)
assert len(theta) == P == 72
OWH = G4 * X
OB = OWH + G4 * H
ctor_theta = [v for r in Wx for v in r] + [v for r in Wh for v in r] + [0.0] * H + [1.0] * H + [0.0] * (2 * H)

xs = [(1.0, -0.5), (0.3, 0.9), (-0.7, 0.4)]
g = (0.7, -1.2, 0.5)
gs = [(0.3, 0.2, -0.4), (-0.6, 0.1, 0.9), g]
s0 = (0.5, -0.25, 0.8, 0.3, -0.6, 1.2)
FD_H = 1e-3


def sigmoid(dt, v):
    one = dt(1)
    if v >= 0:
        return dt(one / dt(one + dt(np.exp(dt(-v)))))
    e = dt(np.exp(v))
    return dt(e / dt(one + e))


class Lstm:
    def __init__(self, dt, K, th):
        self.dt, self.K = dt, K
        self.th = [dt(v) for v in th]
        self.G = [dt(0)] * P
        self.reset()

    def wx(self, r, j):
        return self.th[r * X + j]

    def wh(self, r, k):
        return self.th[OWH + r * H + k]

    def bb(self, r):
        return self.th[OB + r]

    def reset(self):
        self.set_state((0.0,) * (2 * H))

    def set_state(self, s):
        self.h = [self.dt(v) for v in s[:H]]
        self.c = [self.dt(v) for v in s[H:]]
        self.ring = []

    def forward(self, x):
        dt = self.dt
        x = [dt(v) for v in x]
        hp, cp = list(self.h), list(self.c)
        I, F, Gg, O, TC = [], [], [], [], []
        hn, cn = [], []
        for i in range(H):
            a = [self.bb(q * H + i) for q in range(4)]
            for j in range(X):
                for q in range(4):
                    a[q] = dt(a[q] + dt(self.wx(q * H + i, j) * x[j]))
            for k in range(H):
                for q in range(4):
                    a[q] = dt(a[q] + dt(self.wh(q * H + i, k) * hp[k]))
            ig = sigmoid(dt, a[0]); fg = sigmoid(dt, a[1]); gg = dt(np.tanh(a[2])); og = sigmoid(dt, a[3])
            c = dt(dt(fg * cp[i]) + dt(ig * gg))
            tc = dt(np.tanh(c))
            h = dt(og * tc)
            I.append(ig); F.append(fg); Gg.append(gg); O.append(og); TC.append(tc)
            hn.append(h); cn.append(c)
        self.h, self.c = hn, cn
        self.ring.append((x, hp, cp, I, F, Gg, O, TC))
        if len(self.ring) > self.K:
            self.ring.pop(0)
        return list(self.h)

    def backward(self, gv):
        dt = self.dt
        one = dt(1)
        dh = [dt(v) for v in gv]
        dc = [dt(0)] * H
        n_steps = len(self.ring)
        ig_out = None
        for m in range(n_steps):
            x, hp, cp, I, F, Gg, O, TC = self.ring[n_steps - 1 - m]
            di, df, dg, do = [], [], [], []
            for i in range(H):
                dct = dt(dc[i] + dt(dt(dh[i] * O[i]) * dt(one - dt(TC[i] * TC[i]))))
                do.append(dt(dt(dh[i] * TC[i]) * dt(O[i] * dt(one - O[i]))))
                di.append(dt(dt(dct * Gg[i]) * dt(I[i] * dt(one - I[i]))))
                df.append(dt(dt(dct * cp[i]) * dt(F[i] * dt(one - F[i]))))
                dg.append(dt(dt(dct * I[i]) * dt(one - dt(Gg[i] * Gg[i]))))
                dc[i] = dt(dct * F[i])
            delta = di + df + dg + do
            if m == 0:
                ig_out = []
                for j in range(X):
                    s = dt(0)
                    for i in range(H):
                        for q in range(4):
                            s = dt(s + dt(self.wx(q * H + i, j) * delta[q * H + i]))
                    ig_out.append(s)
            for row in range(G4):
                for j in range(X):
                    idx = row * X + j
                    self.G[idx] = dt(self.G[idx] + dt(delta[row] * x[j]))
                for k in range(H):
                    idx = OWH + row * H + k
                    self.G[idx] = dt(self.G[idx] + dt(delta[row] * hp[k]))
                self.G[OB + row] = dt(self.G[OB + row] + delta[row])
            if m + 1 < n_steps:
                nd = []
                for k in range(H):
                    s = dt(0)
                    for i in range(H):
                        for q in range(4):
                            s = dt(s + dt(self.wh(q * H + i, k) * delta[q * H + i]))
                    nd.append(s)
                dh = nd
        return ig_out


def vec_forward(th, seq, s=None, order=(0, 1, 2, 3), forget_add=0.0, cell_tanh=True):
    th = np.array(th, dtype=np.float64)
    WX = th[:OWH].reshape(G4, X)
    WH = th[OWH:OB].reshape(G4, H)
    B = th[OB:]
    h = np.zeros(H) if s is None else np.array(s[:H], dtype=np.float64)
    c = np.zeros(H) if s is None else np.array(s[H:], dtype=np.float64)
    sg = lambda v: 1 / (1 + np.exp(-v))
    hs, cs = [], []
    for x in seq:
        a = WX @ np.array(x) + WH @ h + B
        blk = [a[q * H:(q + 1) * H] for q in range(4)]
        ai, af, ag, ao = (blk[order[0]], blk[order[1]], blk[order[2]], blk[order[3]])
        i_, f_, g_, o_ = sg(ai), sg(af + forget_add), np.tanh(ag), sg(ao)
        c = f_ * c + i_ * g_
        h = o_ * (np.tanh(c) if cell_tanh else c)
        hs.append(h); cs.append(c)
    return hs, cs


def dot(a, bb, dt):
    s = dt(0)
    for u, v in zip(a, bb):
        s = dt(s + dt(dt(u) * dt(v)))
    return s


def fmt(v):
    return "(" + ", ".join(f"{float(e):.7g}" for e in v) + ")"


def run_seq(dt, K, th, seq, state=None):
    r = Lstm(dt, K, th)
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


def fd64(loss):
    num = []
    for k in range(P):
        tp = list(theta); tm = list(theta)
        tp[k] += 1e-6; tm[k] -= 1e-6
        num.append((loss(tp) - loss(tm)) / 2e-6)
    return num


def maxerr(a, bb):
    return max(abs(float(u) - float(v)) for u, v in zip(a, bb))


def blocks(G):
    G = [float(v) for v in G]
    return "\n   Wx " + fmt(G[:OWH]) + "\n   Wh " + fmt(G[OWH:OB]) + "\n   b  " + fmt(G[OB:])


f32, f64 = np.float32, np.float64
print("== P =", P, " theta =", fmt(theta))
print("ctor theta b block", fmt(ctor_theta[OB:]))

for dt, name in ((f32, "f32"), (f64, "f64")):
    r, outs = run_seq(dt, 4, theta, xs)
    print(f"A forward {name}: h1={fmt(outs[0])} h2={fmt(outs[1])} h3={fmt(outs[2])} c3={fmt(r.c)}")
    for st in range(3):
        x_, hp, cp, I, F, Gg, O, TC = r.ring[st]
        print(f"A step{st+1} {name}: i={fmt(I)} f={fmt(F)} g={fmt(Gg)} o={fmt(O)} tanh(c)={fmt(TC)}")
r, outs = run_seq(f32, 4, theta, xs)
print("A states c1,c2 f32:", fmt(run_seq(f32, 4, theta, xs[:1])[0].c), fmt(run_seq(f32, 4, theta, xs[:2])[0].c))
hv, cv = vec_forward(theta, xs)
print("A vectorised f64 (PyTorch formula):", [fmt(o) for o in hv], " c3", fmt(cv[-1]))
print("A f32 vs f64-vec max", max(maxerr(a, bb) for a, bb in zip(outs, hv)))
print("A wrong gate order i,g,f,o (TF1 LSTMCell) h1:", fmt(vec_forward(theta, xs, order=(0, 2, 1, 3))[0][0]))
print("A wrong runtime forget_bias +1 (TF1) h1, h2:", fmt(vec_forward(theta, xs, forget_add=1.0)[0][0]), fmt(vec_forward(theta, xs, forget_add=1.0)[0][1]))
print("A wrong h = o*c (no tanh) h1:", fmt(vec_forward(theta, xs, cell_tanh=False)[0][0]))
_, o_ct = run_seq(f32, 4, ctor_theta, [xs[0]])
print("A ctor params (b_f = 1) h1:", fmt(o_ct[0]))

for dt, name in ((f32, "f32"), (f64, "f64")):
    rr, o = run_seq(dt, 4, theta, [xs[0]], s0)
    print(f"B setstate {name}: h={fmt(o[0])} c={fmt(rr.c)}")

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
r64, _ = run_seq(f64, 4, theta, xs)
print("C inputgrad f64", fmt(r64.backward(g)))

r, outs = run_seq(f32, 4, theta, xs)
r.backward(g)
full = r.G
num_full = fd_theta(f32, lambda th: dot(g, run_seq(f32, 4, th, xs)[1][-1], f32))
print("D fullbptt", blocks(full))
print("D fullbptt fd maxerr", f"{maxerr(full, num_full):.2g}")
fd_full64 = fd64(lambda th: float(np.dot(g, vec_forward(th, xs)[0][-1])))
print("D vectorised-f64 FD vs f32 analytic maxerr", f"{maxerr(full, fd_full64):.2g}")




def backward_no_cell(self, gv):
    dt = self.dt
    saved = self.ring
    G0 = list(self.G)
    one = dt(1)
    dh = [dt(v) for v in gv]
    n_steps = len(self.ring)
    for m in range(n_steps):
        x, hp, cp, I, F, Gg, O, TC = self.ring[n_steps - 1 - m]
        delta = [None] * G4
        for i in range(H):
            dct = dt(dt(dh[i] * O[i]) * dt(one - dt(TC[i] * TC[i])))
            delta[3 * H + i] = dt(dt(dh[i] * TC[i]) * dt(O[i] * dt(one - O[i])))
            delta[i] = dt(dt(dct * Gg[i]) * dt(I[i] * dt(one - I[i])))
            delta[H + i] = dt(dt(dct * cp[i]) * dt(F[i] * dt(one - F[i])))
            delta[2 * H + i] = dt(dt(dct * I[i]) * dt(one - dt(Gg[i] * Gg[i])))
        for row in range(G4):
            self.G[OB + row] = dt(self.G[OB + row] + delta[row])
        if m + 1 < n_steps:
            dh = [sum_(self, k, delta, dt) for k in range(H)]


def sum_(self, k, delta, dt):
    s = dt(0)
    for i in range(H):
        for q in range(4):
            s = dt(s + dt(self.wh(q * H + i, k) * delta[q * H + i]))
    return s


rn, _ = run_seq(f32, 4, theta, xs)
backward_no_cell(rn, g)
print("D wrong (no dc recursion through f) G[b]", fmt(rn.G[OB:]))

r, outs = run_seq(f32, 2, theta, xs)
r.backward(g)
trunc = r.G
st1 = run_seq(f32, 4, theta, xs[:1])[0]
s1 = tuple(st1.h) + tuple(st1.c)
print("E frozen state after step 1 (h1; c1)", fmt(s1))
num_trunc = fd_theta(f32, lambda th: dot(g, run_seq(f32, 2, th, xs[1:], s1)[1][-1], f32))
print("E trunc K=2", blocks(trunc))
print("E trunc fd maxerr", f"{maxerr(trunc, num_trunc):.2g}")
diffs = [float(a) - float(c) for a, c in zip(trunc, full)]
print("E trunc - full max", f"{max(abs(d) for d in diffs):.3g}", " argmax", int(np.argmax([abs(d) for d in diffs])),
      " full/trunc at argmax", float(full[int(np.argmax([abs(d) for d in diffs]))]), float(trunc[int(np.argmax([abs(d) for d in diffs]))]))
print("E indices where trunc==full:", [k for k, d in enumerate(diffs) if abs(d) < 1e-7])

r = Lstm(f32, 4, theta)
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
print("F many-to-many", blocks(acc))
print("F fd maxerr", f"{maxerr(acc, num_acc):.2g}")

Wd, cd = (0.4, -0.7, 0.2), 0.1
r, outs = run_seq(f32, 4, theta, xs)
ys = []
for h in outs:
    zd = f32(f32(cd) + dot(Wd, h, f32))
    ys.append((float(zd), float(f32(np.tanh(zd)))))
print("H model (z, y) per step", ys)

r = Lstm(f32, 4, theta)
r.forward(xs[0]); r.forward(xs[1])
r.set_state(s0)
o = r.forward(xs[0])
r.backward(g)
num_b = fd_theta(f32, lambda th: dot(g, run_seq(f32, 4, th, [xs[0]], s0)[1][-1], f32))
print("B setstate out", fmt(o), " c", fmt(r.c))
print("B setstate G", blocks(r.G))
print("B setstate fd maxerr", f"{maxerr(r.G, num_b):.2g}")
r2 = Lstm(f32, 4, theta)
r2.forward(xs[0]); r2.forward(xs[1]); r2.forward(xs[0]); r2.backward(g)
print("B without SetState (continued, history kept) G[0..3]", fmt(r2.G[:4]), " G[OWH]", float(r2.G[OWH]))
print("B with SetState G[0..3]", fmt(r.G[:4]), " G[OWH]", float(r.G[OWH]))

r = Lstm(f32, 1, theta)
for x in xs: r.forward(x)
r.backward(g)
print("K=1 G", blocks(r.G))

Xe, He, Ke = 3, 16, 8
Pe = 4 * (He * (Xe + He) + He)
Pg = 3 * (He * (Xe + He) + 2 * He)
ram = 2 * Pe + Ke * (Xe + 7 * He) + 3 * He + Xe
print(f"example X={Xe} H={He} K={Ke}: P={Pe} (GRU {Pg}) ring={Ke*(Xe+7*He)} RAM={ram} floats = {4*ram} B, flash={4*Pe} B")
fwd = 4 * He * Xe + 4 * He * He
bwd = Ke * (4 * He * Xe + 4 * He * He) + (Ke - 1) * 4 * He * He + 4 * He * Xe
print(f"  Forward MACs={fwd}  Backward MACs full window={bwd}  formula={Ke*(4*He*Xe+8*He*He)-4*He*He+4*He*Xe}")
print(f"fixture Lstm<float,2,3,4>: P={P} ring={4*(X+7*H)} RAM floats={2*P + 4*(X+7*H) + 3*H + X}")

r4 = Lstm(f32, 4, theta)
r4.forward(xs[0]); r4.forward(xs[1])
r4.h = [f32(v) for v in s0[:H]]; r4.c = [f32(v) for v in s0[H:]]
r4.forward(xs[0]); r4.backward(g)
print("B wrong: SetState that keeps the ring (count not reset) G[0..3]", fmt(r4.G[:4]), " G[OWH]", float(r4.G[OWH]))
