import numpy as np

f = np.float32
H = f(1e-3)
IN, HID, OUT = 2, 3, 1
P1 = IN * HID + HID
P2 = HID * OUT + OUT
P = P1 + P2
SLICES = [(0, P1), (P1, P)]

W1 = np.array([[0.5, -1.0], [1.5, 0.25], [-0.5, 0.75]], dtype=f)
W2 = np.array([[1.0, -0.5, 2.0]], dtype=f)
THETA0 = np.concatenate([W1.ravel(), np.zeros(HID, f), W2.ravel(), np.zeros(OUT, f)]).astype(f)

X1, Y1 = np.array([2.0, 0.5], f), np.array([-0.5], f)
X2, Y2 = np.array([-1.0, 1.0], f), np.array([0.25], f)
X3, Y3 = np.array([0.5, -1.5], f), np.array([0.75], f)


def unpack(t):
    w1 = t[0:6].reshape(HID, IN)
    b1 = t[6:9]
    w2 = t[9:12].reshape(OUT, HID)
    b2 = t[12:13]
    return w1, b1, w2, b2


def leaky(z):
    return np.where(z > 0, z, f(0.1) * z).astype(f)


def leaky_d(z):
    return np.where(z > 0, f(1), f(0.1)).astype(f)


def forward(t, x):
    w1, b1, w2, b2 = unpack(t)
    z1 = (w1 @ x + b1).astype(f)
    a1 = leaky(z1)
    z2 = (w2 @ a1 + b2).astype(f)
    a2 = np.tanh(z2).astype(f)
    return z1, a1, z2, a2


def backward(t, x, g):
    w1, b1, w2, b2 = unpack(t)
    z1, a1, z2, a2 = forward(t, x)
    d2 = ((f(1) - a2 * a2) * g).astype(f)
    gw2 = np.outer(d2, a1).astype(f)
    ga1 = (w2.T @ d2).astype(f)
    d1 = (leaky_d(z1) * ga1).astype(f)
    gw1 = np.outer(d1, x).astype(f)
    return np.concatenate([gw1.ravel(), d1, gw2.ravel(), d2]).astype(f), z1


def mse(yhat, y):
    e = (yhat - y).astype(f)
    return f(np.sum(e * e) / f(len(y))), (f(2) * e / f(len(y))).astype(f)


def sample(t, x, y):
    _, _, _, yhat = forward(t, x)
    c, g = mse(yhat, y)
    G, z1 = backward(t, x, g)
    return c, g, G, yhat, z1


def step(t, samples, eta, trainable=(True, True), decay=f(0), clip=None):
    G = np.zeros(P, f)
    cost = f(0)
    for x, y in samples:
        c, _, Gs, _, _ = sample(t, x, y)
        G = (G + Gs).astype(f)
        cost = f(cost + c)
    k = f(len(samples))
    g = (G / k).astype(f)
    for (lo, hi), tr in zip(SLICES, trainable):
        if not tr:
            g[lo:hi] = 0
    norm = f(np.sqrt(np.sum(g.astype(f) * g)))
    clipped = False
    if clip is not None and norm > clip:
        g = (g * f(clip / norm)).astype(f)
        clipped = True
    new = t.copy()
    for (lo, hi), tr in zip(SLICES, trainable):
        if tr:
            new[lo:hi] = (t[lo:hi] - f(eta) * (g[lo:hi] + f(decay) * t[lo:hi])).astype(f)
    return new, f(cost / k), norm, g, clipped


def batch_objective(t, samples, decay=f(0)):
    j = f(0)
    for x, y in samples:
        _, _, _, yhat = forward(t, x)
        j = f(j + mse(yhat, y)[0])
    return f(j / f(len(samples)) + f(decay) / f(2) * f(np.sum(t * t)))


def fd(t, samples, decay=f(0)):
    num = np.zeros(P, f)
    for k in range(P):
        tp, tm = t.copy(), t.copy()
        tp[k] += H
        tm[k] -= H
        num[k] = (batch_objective(tp, samples, decay) - batch_objective(tm, samples, decay)) / (f(2) * H)
    return num


def fmt(v):
    return "(" + ", ".join(f"{float(x):.7g}" for x in np.atleast_1d(v)) + ")"


def section(name):
    print(f"\n## {name}")


section("A collaborator: Backward(g = 1) at x1 (StrictMock loss)")
G, z1 = backward(THETA0, X1, np.array([1.0], f))
print("yhat", fmt(forward(THETA0, X1)[3]), "z1", fmt(z1))
print("G", fmt(G))

section("per-sample MSE")
for n, (x, y) in enumerate([(X1, Y1), (X2, Y2), (X3, Y3)], 1):
    c, g, Gs, yhat, z1 = sample(THETA0, x, y)
    print(f"s{n}: yhat {fmt(yhat)} L {c:.7g} dL/dyhat {fmt(g)} z1 {fmt(z1)}")
    print(f"s{n}: G {fmt(Gs)}")

batch = [(X1, Y1), (X2, Y2)]
section("B batch step, eta = 0.1")
new, cost, norm, g, _ = step(THETA0, batch, 0.1)
num = fd(THETA0, batch)
print("meanLoss", f"{cost:.7g}", "norm", f"{norm:.7g}")
print("gbar", fmt(g))
print("FD  ", fmt(num), "maxerr", f"{np.max(np.abs(num - g)):.2g}")
print("theta1", fmt(new))
print("float64 check gbar", fmt((THETA0.astype(np.float64) - new.astype(np.float64)) / 0.1))

section("C flush after one sample (B = 4), eta = 0.1")
newc, costc, normc, gc, _ = step(THETA0, [(X1, Y1)], 0.1)
print("meanLoss", f"{costc:.7g}", "norm", f"{normc:.7g}")
print("theta", fmt(newc))

section("D frozen hidden layer {false, true}")
newd, costd, normd, gd, _ = step(THETA0, batch, 0.1, trainable=(False, True))
print("norm", f"{normd:.7g}", "theta", fmt(newd))

section("E weight decay lambda = 0.1")
newe, coste, norme, ge, _ = step(THETA0, batch, 0.1, decay=f(0.1))
nume = fd(THETA0, batch, f(0.1))
eff = ((THETA0 - newe) / f(0.1)).astype(f)
print("norm (loss part, excl. decay)", f"{norme:.7g}")
print("gbar + lambda theta", fmt(eff))
print("FD(J + R)", fmt(nume), "maxerr", f"{np.max(np.abs(nume - eff)):.2g}")
print("theta", fmt(newe))

section("H SetLearningRate(0.05)")
newh, _, _, _, _ = step(THETA0, batch, 0.05)
print("theta", fmt(newh))

section("I epoch of 3 samples, B = 2, eta = 0.1")
t1, c12, _, _, _ = step(THETA0, batch, 0.1)
c3, _, _, yhat3, z13 = sample(t1, X3, Y3)
t2, c3b, _, _, _ = step(t1, [(X3, Y3)], 0.1)
print("L1+L2 mean", f"{c12:.7g}", "L3 at theta1", f"{c3:.7g}", "z1 at theta1", fmt(z13))
print("epoch meanLoss", f"{(c12 * 2 + c3) / 3:.7g}", "steps 2")
print("theta2", fmt(t2))

section("F' clip c = 0.25 (below norm) and c = 1.0 (above norm)")
for c in (0.25, 1.0):
    newf, costf, normf, gf, clf = step(THETA0, batch, 0.1, clip=f(c))
    print(f"c = {c}: clipped {clf} norm(before) {normf:.7g} scale {min(1.0, c / normf):.7g} step norm {np.linalg.norm((THETA0 - newf) / f(0.1)):.7g}")
    print("theta", fmt(newf))

section("K clip c = 0.25 then weight decay lambda = 0.1 (decay not clipped)")
newk, _, normk, _, clk = step(THETA0, batch, 0.1, decay=f(0.1), clip=f(0.25))
print("clipped", clk, "norm", f"{normk:.7g}", "theta", fmt(newk))

section("L convergence: TrainEpoch x 40 on {(x1,y1),(x2,y2)}, B = 2, eta = 0.1")
t = THETA0.copy()
hist = []
for epoch in range(40):
    t, c, _, _, _ = step(t, batch, 0.1)
    hist.append(c)
print("epoch 1", f"{hist[0]:.7g}", "epoch 20", f"{hist[19]:.3g}", "epoch 22", f"{hist[21]:.3g}", "epoch 40", f"{hist[39]:.3g}")
print("predictions after 40 epochs", fmt(forward(t, X1)[3]), fmt(forward(t, X2)[3]))

section("K' alternative order (clip g + lambda theta together) must differ from K")
_, _, _, gk, _ = step(THETA0, batch, 0.1)
full = (gk + f(0.1) * THETA0).astype(f)
alt = (THETA0 - f(0.1) * (f(0.25) / f(np.sqrt(np.sum(full * full)))) * full).astype(f)
print("alt theta", fmt(alt), "max |alt - K|", f"{np.max(np.abs(alt - newk)):.3g}")
print("E vs B max diff", f"{np.max(np.abs(newe - new)):.3g}")
