import numpy as np

X, H = 2, 3
G3 = 3 * H
Wx = [[0.5, -0.3], [0.2, 0.8], [-0.6, 0.1],
      [0.3, 0.4], [-0.2, 0.5], [0.7, -0.1],
      [0.9, -0.4], [-0.5, 0.6], [0.4, 0.3]]
Wh = [[0.1, -0.4, 0.3], [0.5, 0.2, -0.1], [-0.3, 0.6, 0.25],
      [0.2, 0.1, -0.3], [-0.4, 0.3, 0.2], [0.1, -0.2, 0.5],
      [0.6, -0.2, 0.1], [0.3, 0.5, -0.4], [-0.1, 0.4, 0.7]]
bx = [0.05, -0.1, 0.2, 0.1, 0.0, -0.2, 0.0, 0.15, -0.05]
bh = [0.02, 0.03, -0.04, -0.1, 0.2, 0.05, 0.3, -0.2, 0.1]
theta = [v for r in Wx for v in r] + [v for r in Wh for v in r] + bx + bh
P = 3 * (H * (X + H) + 2 * H)
assert len(theta) == P == 63
OWH = G3 * X
OBX = OWH + G3 * H
OBH = OBX + G3

xs = [(1.0, -0.5), (0.3, 0.9), (-0.7, 0.4)]
g = (0.7, -1.2, 0.5)
gs = [(0.3, 0.2, -0.4), (-0.6, 0.1, 0.9), g]
s0 = (0.5, -0.25, 0.8)
FD_H = 1e-3


def sigmoid(dt, v):
    one = dt(1)
    if v >= 0:
        return dt(one / dt(one + dt(np.exp(dt(-v)))))
    e = dt(np.exp(v))
    return dt(e / dt(one + e))


class Gru:
    def __init__(self, dt, K, th):
        self.dt, self.K = dt, K
        self.th = [dt(v) for v in th]
        self.G = [dt(0)] * P
        self.reset()

    def wx(self, r, j):
        return self.th[r * X + j]

    def wh(self, r, k):
        return self.th[OWH + r * H + k]

    def b_x(self, r):
        return self.th[OBX + r]

    def b_h(self, r):
        return self.th[OBH + r]

    def reset(self):
        self.set_state((0.0,) * H)

    def set_state(self, s):
        self.out = [self.dt(v) for v in s]
        self.ring = []

    def forward(self, x):
        dt = self.dt
        x = [dt(v) for v in x]
        hp = list(self.out)
        rr, zz, nn, uu = [], [], [], []
        for i in range(H):
            ar = dt(self.b_x(i) + self.b_h(i))
            az = dt(self.b_x(H + i) + self.b_h(H + i))
            xn = self.b_x(2 * H + i)
            u = self.b_h(2 * H + i)
            for j in range(X):
                ar = dt(ar + dt(self.wx(i, j) * x[j]))
                az = dt(az + dt(self.wx(H + i, j) * x[j]))
                xn = dt(xn + dt(self.wx(2 * H + i, j) * x[j]))
            for k in range(H):
                ar = dt(ar + dt(self.wh(i, k) * hp[k]))
                az = dt(az + dt(self.wh(H + i, k) * hp[k]))
                u = dt(u + dt(self.wh(2 * H + i, k) * hp[k]))
            r = sigmoid(dt, ar)
            z = sigmoid(dt, az)
            n = dt(np.tanh(dt(xn + dt(r * u))))
            rr.append(r); zz.append(z); nn.append(n); uu.append(u)
        self.out = [dt(nn[i] + dt(zz[i] * dt(hp[i] - nn[i]))) for i in range(H)]
        self.ring.append((x, hp, rr, zz, nn, uu))
        if len(self.ring) > self.K:
            self.ring.pop(0)
        return list(self.out)

    def backward(self, gv):
        dt = self.dt
        one = dt(1)
        dh = [dt(v) for v in gv]
        n_steps = len(self.ring)
        ig = None
        for m in range(n_steps):
            x, hp, rr, zz, nn, uu = self.ring[n_steps - 1 - m]
            dar, daz, dan, du = [], [], [], []
            for i in range(H):
                an = dt(dt(dh[i] * dt(one - zz[i])) * dt(one - dt(nn[i] * nn[i])))
                az = dt(dt(dh[i] * dt(hp[i] - nn[i])) * dt(zz[i] * dt(one - zz[i])))
                ar = dt(dt(an * uu[i]) * dt(rr[i] * dt(one - rr[i])))
                dan.append(an); daz.append(az); dar.append(ar); du.append(dt(an * rr[i]))
            if m == 0:
                ig = []
                for j in range(X):
                    s = dt(0)
                    for i in range(H):
                        s = dt(s + dt(self.wx(i, j) * dar[i]))
                        s = dt(s + dt(self.wx(H + i, j) * daz[i]))
                        s = dt(s + dt(self.wx(2 * H + i, j) * dan[i]))
                    ig.append(s)
            ga = dar + daz + dan
            gh = dar + daz + du
            for row in range(G3):
                for j in range(X):
                    idx = row * X + j
                    self.G[idx] = dt(self.G[idx] + dt(ga[row] * x[j]))
                for k in range(H):
                    idx = OWH + row * H + k
                    self.G[idx] = dt(self.G[idx] + dt(gh[row] * hp[k]))
                self.G[OBX + row] = dt(self.G[OBX + row] + ga[row])
                self.G[OBH + row] = dt(self.G[OBH + row] + gh[row])
            if m + 1 < n_steps:
                nd = []
                for k in range(H):
                    s = dt(dh[k] * zz[k])
                    for i in range(H):
                        s = dt(s + dt(self.wh(i, k) * dar[i]))
                        s = dt(s + dt(self.wh(H + i, k) * daz[i]))
                        s = dt(s + dt(self.wh(2 * H + i, k) * du[i]))
                    nd.append(s)
                dh = nd
        return ig


def vec_forward(th, seq, h0=None, reset_after=True, z_on_state=True):
    th = np.array(th, dtype=np.float64)
    WX = th[:OWH].reshape(G3, X)
    WH = th[OWH:OBX].reshape(G3, H)
    BX = th[OBX:OBH]
    BH = th[OBH:]
    h = np.zeros(H) if h0 is None else np.array(h0, dtype=np.float64)
    sg = lambda v: 1 / (1 + np.exp(-v))
    outs = []
    for x in seq:
        x = np.array(x)
        gx = WX @ x + BX
        r = sg(gx[:H] + WH[:H] @ h + BH[:H])
        z = sg(gx[H:2*H] + WH[H:2*H] @ h + BH[H:2*H])
        if reset_after:
            n = np.tanh(gx[2*H:] + r * (WH[2*H:] @ h + BH[2*H:]))
        else:
            n = np.tanh(gx[2*H:] + WH[2*H:] @ (r * h) + BH[2*H:])
        h = (1 - z) * n + z * h if z_on_state else z * n + (1 - z) * h
        outs.append(h)
    return outs


def dot(a, bb, dt):
    s = dt(0)
    for u, v in zip(a, bb):
        s = dt(s + dt(dt(u) * dt(v)))
    return s


def fmt(v):
    return "(" + ", ".join(f"{float(e):.7g}" for e in v) + ")"


def run_seq(dt, K, th, seq, state=None):
    r = Gru(dt, K, th)
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
    return {"Wx": fmt(G[:OWH]), "Wh": fmt(G[OWH:OBX]), "bx": fmt(G[OBX:OBH]), "bh": fmt(G[OBH:])}


f32, f64 = np.float32, np.float64
print("== P =", P, " theta =", fmt(theta))

for dt, name in ((f32, "f32"), (f64, "f64")):
    r, outs = run_seq(dt, 4, theta, xs)
    print(f"A forward {name}: h1={fmt(outs[0])} h2={fmt(outs[1])} h3={fmt(outs[2])}")
    x_, hp, rr, zz, nn, uu = r.ring[0]
    print(f"A step1 {name}: r={fmt(rr)} z={fmt(zz)} n={fmt(nn)} u={fmt(uu)}")
    x_, hp, rr, zz, nn, uu = r.ring[1]
    print(f"A step2 {name}: r={fmt(rr)} z={fmt(zz)} n={fmt(nn)} u={fmt(uu)}")
v = vec_forward(theta, xs)
print("A vectorised f64 (PyTorch formula):", [fmt(o) for o in v])
print("A Cho reset-before h2:", fmt(vec_forward(theta, xs, reset_after=False)[1]))
print("A z-swapped (Chung) h1, h2:", fmt(vec_forward(theta, xs, z_on_state=False)[0]), fmt(vec_forward(theta, xs, z_on_state=False)[1]))
print("A b_hn folded into b_xn h2:", end=" ")
thf = list(theta)
for i in range(H):
    thf[OBX + 2 * H + i] += thf[OBH + 2 * H + i]; thf[OBH + 2 * H + i] = 0.0
print(fmt(vec_forward(thf, xs)[1]), " h1:", fmt(vec_forward(thf, xs)[0]))
for dt, name in ((f32, "f32"), (f64, "f64")):
    _, o = run_seq(dt, 4, theta, [xs[0]], s0)
    print(f"B setstate {name}: h={fmt(o[0])}")

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
print("D fullbptt analytic", fmt(full))
print("D blocks", blocks(full))
print("D fullbptt fd maxerr", f"{maxerr(full, num_full):.2g}")
fd_full64 = fd64(lambda th: float(np.dot(g, vec_forward(th, xs)[-1])))
print("D vectorised-f64 FD vs f32 analytic maxerr", f"{maxerr(full, fd_full64):.2g}")

r, outs = run_seq(f32, 2, theta, xs)
r.backward(g)
trunc = r.G
h1 = outs[0]
num_trunc = fd_theta(f32, lambda th: dot(g, run_seq(f32, 2, th, xs[1:], h1)[1][-1], f32))
print("E trunc K=2 analytic", fmt(trunc))
print("E blocks", blocks(trunc))
print("E trunc fd maxerr", f"{maxerr(trunc, num_trunc):.2g}")
diffs = [float(a) - float(c) for a, c in zip(trunc, full)]
print("E trunc - full", fmt(diffs), " max", f"{max(abs(d) for d in diffs):.3g}", " argmax", int(np.argmax([abs(d) for d in diffs])))
print("E indices where trunc==full:", [k for k, d in enumerate(diffs) if abs(d) < 1e-7])

r = Gru(f32, 4, theta)
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
print("F blocks", blocks(acc))
print("F fd maxerr", f"{maxerr(acc, num_acc):.2g}")

Wd, cd = (0.4, -0.7, 0.2), 0.1
r, outs = run_seq(f32, 4, theta, xs)
ys = []
for h in outs:
    zd = f32(f32(cd) + dot(Wd, h, f32))
    ys.append((float(zd), float(f32(np.tanh(zd)))))
print("H model (z, y) per step", ys)

r = Gru(f32, 4, theta)
r.forward(xs[0]); r.forward(xs[1])
r.set_state(s0)
o = r.forward(xs[0])
r.backward(g)
num_b = fd_theta(f32, lambda th: dot(g, run_seq(f32, 4, th, [xs[0]], s0)[1][-1], f32))
print("B setstate out", fmt(o))
print("B setstate G analytic", fmt(r.G))
print("B blocks", blocks(r.G))
print("B setstate fd maxerr", f"{maxerr(r.G, num_b):.2g}")
r2 = Gru(f32, 4, theta)
r2.forward(xs[0]); r2.forward(xs[1]); r2.forward(xs[0]); r2.backward(g)
print("B without SetState (history kept) G[0..3]", fmt(r2.G[:4]), " G[OWH]", float(r2.G[OWH]))
print("B with SetState G[0..3]", fmt(r.G[:4]), " G[OWH]", float(r.G[OWH]))

r = Gru(f32, 1, theta)
for x in xs: r.forward(x)
r.backward(g)
print("K=1 G", fmt(r.G))

Xe, He, Ke = 3, 16, 8
Pe = 3 * (He * (Xe + He) + 2 * He)
Pl = 4 * (He * (Xe + He) + He)
ram = 2 * Pe + Ke * (Xe + 5 * He) + He + Xe
print(f"example X={Xe} H={He} K={Ke}: P={Pe} (LSTM {Pl}, {100*(1-Pe/Pl):.2f}% fewer) ring={Ke*(Xe+5*He)} RAM={ram} floats = {4*ram} B, flash={4*Pe} B")
fwd = 3 * He * Xe + 3 * He * He
bwd = Ke * (3 * He * Xe + 3 * He * He) + (Ke - 1) * 3 * He * He + 3 * He * Xe
print(f"  Forward MACs={fwd}  Backward MACs full window={bwd}")
print(f"fixture Gru<float,2,3,4>: P={P} RAM floats={2*P + 4*(X+5*H) + H + X}")
print(f"Elman same example P={He*(Xe+He+1)}")
