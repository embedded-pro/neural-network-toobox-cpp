import numpy as np

f32 = np.float32
H = f32(1e-3)


def conv_forward(x, theta, L, Cin, Cout, K, S, act):
    Lout = (L - K) // S + 1
    W = theta[: K * Cin * Cout].reshape(Cout, K * Cin)
    b = theta[K * Cin * Cout:]
    z = np.zeros(Lout * Cout, dtype=f32)
    for t in range(Lout):
        base = t * S * Cin
        for o in range(Cout):
            s = f32(b[o])
            for j in range(K * Cin):
                s = f32(s + f32(W[o, j] * x[base + j]))
            z[t * Cout + o] = s
    return z, act(z)


def conv_backward(x, theta, L, Cin, Cout, K, S, z, y, g, dact):
    Lout = (L - K) // S + 1
    W = theta[: K * Cin * Cout].reshape(Cout, K * Cin)
    delta = dact(z, y, g)
    gx = np.zeros(L * Cin, dtype=f32)
    gt = np.zeros_like(theta)
    off = K * Cin * Cout
    for t in range(Lout):
        base = t * S * Cin
        for o in range(Cout):
            d = delta[t * Cout + o]
            for j in range(K * Cin):
                gx[base + j] = f32(gx[base + j] + f32(W[o, j] * d))
                gt[o * K * Cin + j] = f32(gt[o * K * Cin + j] + f32(d * x[base + j]))
            gt[off + o] = f32(gt[off + o] + d)
    return gx, gt, delta


def leaky(z):
    return np.where(z > 0, z, f32(0.1) * z).astype(f32)


def tanh(z):
    return np.tanh(z).astype(f32)


def dtanh(z, y, g):
    return (g * (f32(1) - y * y)).astype(f32)


def fmt(v):
    return "(" + ", ".join(f"{float(a):.7g}" for a in v) + ")"


L, Cin, Cout, K, S = 6, 2, 2, 3, 2
x = np.array([0.5, -1.0, 0.25, 0.75, -0.5, 1.5, 1.0, -0.25, 0.0, 2.0, -1.5, 0.5], dtype=f32)
W0 = [0.2, -0.1, 0.4, 0.3, -0.5, 0.1]
W1 = [-0.3, 0.2, 0.1, -0.4, 0.6, 0.05]
b = [0.1, -0.2]
theta = np.array(W0 + W1 + b, dtype=f32)
theta0 = np.array(W0 + W1 + [0.0, 0.0], dtype=f32)
g = np.array([0.8, -1.3, 0.4, 0.6], dtype=f32)
x2 = np.array([-0.2, 0.4, 1.2, -0.6, 0.3, 0.9, -1.1, 0.7, 0.5, -0.3, 0.8, -0.4], dtype=f32)
g2 = np.array([-0.5, 0.25, 1.0, -0.75], dtype=f32)

print("Lout", (L - K) // S + 1, "P", K * Cin * Cout + Cout)
for (LL, KK, SS) in [(6, 3, 2), (7, 3, 2), (8, 3, 2), (5, 5, 1), (5, 1, 1), (6, 2, 3), (128, 5, 1), (128, 5, 2)]:
    print("L", LL, "K", KK, "S", SS, "Lout", (LL - KK) // SS + 1)

z, y = conv_forward(x, theta0, L, Cin, Cout, K, S, leaky)
print("ctor theta0 forward z", fmt(z), "leaky", fmt(y))
z, y = conv_forward(x, theta, L, Cin, Cout, K, S, leaky)
print("forward theta z", fmt(z), "leaky y", fmt(y))
zd = np.array(z, dtype=np.float64)
Wd = np.array(W0 + W1, dtype=np.float64).reshape(2, 6)
xd = np.array(x, dtype=np.float64)
zref = [b[o] + Wd[o] @ xd[t * 4: t * 4 + 6] for t in range(2) for o in range(2)]
print("float64 z", zref)

zt, yt = conv_forward(x, theta, L, Cin, Cout, K, S, tanh)
print("tanh z", fmt(zt), "tanh y", fmt(yt))
gx, gt, delta = conv_backward(x, theta, L, Cin, Cout, K, S, zt, yt, g, dtanh)
print("delta", fmt(delta))
print("input grad", fmt(gx))
print("param grad", fmt(gt))


def proj_x(xx):
    _, yy = conv_forward(xx, theta, L, Cin, Cout, K, S, tanh)
    return f32(np.sum((g * yy).astype(f32), dtype=f32))


def proj_t(tt, xx, gg):
    _, yy = conv_forward(xx, tt, L, Cin, Cout, K, S, tanh)
    return f32(np.sum((gg * yy).astype(f32), dtype=f32))


num = np.zeros_like(x)
for j in range(len(x)):
    p = x.copy(); m = x.copy()
    p[j] = f32(p[j] + H); m[j] = f32(m[j] - H)
    num[j] = f32((proj_x(p) - proj_x(m)) / f32(2 * H))
print("FD input grad", fmt(num), "maxerr", float(np.max(np.abs(num - gx))))

numt = np.zeros_like(theta)
for k in range(len(theta)):
    p = theta.copy(); m = theta.copy()
    p[k] = f32(p[k] + H); m[k] = f32(m[k] - H)
    numt[k] = f32((proj_t(p, x, g) - proj_t(m, x, g)) / f32(2 * H))
print("FD param grad", fmt(numt), "maxerr", float(np.max(np.abs(numt - gt))))

zt2, yt2 = conv_forward(x2, theta, L, Cin, Cout, K, S, tanh)
gx2, gt2, _ = conv_backward(x2, theta, L, Cin, Cout, K, S, zt2, yt2, g2, dtanh)
print("sample2 param grad", fmt(gt2))
acc = (gt + gt2).astype(f32)
print("accumulated G1+G2", fmt(acc))
numt2 = np.zeros_like(theta)
for k in range(len(theta)):
    p = theta.copy(); m = theta.copy()
    p[k] = f32(p[k] + H); m[k] = f32(m[k] - H)
    numt2[k] = f32((proj_t(p, x, g) + proj_t(p, x2, g2) - proj_t(m, x, g) - proj_t(m, x2, g2)) / f32(2 * H))
print("FD accumulated", fmt(numt2), "maxerr", float(np.max(np.abs(numt2 - acc))))


def stream(frames, theta, Cin, Cout, K, S, act):
    hist = np.zeros((K - 1) * Cin, dtype=f32)
    oldest = 0
    phase = 0
    W = theta[: K * Cin * Cout].reshape(Cout, K * Cin)
    bb = theta[K * Cin * Cout:]
    outs = []
    for fr in frames:
        emit = phase == 0
        if emit:
            pre = np.zeros(Cout, dtype=f32)
            for o in range(Cout):
                s = f32(bb[o])
                for r in range(K - 1):
                    slot = oldest + r
                    if slot >= K - 1:
                        slot -= K - 1
                    for c in range(Cin):
                        s = f32(s + f32(W[o, r * Cin + c] * hist[slot * Cin + c]))
                for c in range(Cin):
                    s = f32(s + f32(W[o, (K - 1) * Cin + c] * fr[c]))
                pre[o] = s
            outs.append((True, act(pre)))
        else:
            outs.append((False, None))
        if K > 1:
            hist[oldest * Cin: oldest * Cin + Cin] = fr
            oldest = 0 if oldest + 1 == K - 1 else oldest + 1
        phase = 0 if phase + 1 == S else phase + 1
    return outs


frames = [x[i * Cin:(i + 1) * Cin] for i in range(L)]
outs = stream(frames, theta, Cin, Cout, K, S, leaky)
for i, (e, o) in enumerate(outs):
    print("stream frame", i, e, fmt(o) if e else "")
padded = np.concatenate([np.zeros((K - 1) * Cin, dtype=f32), x])
zp, yp = conv_forward(padded, theta, L + K - 1, Cin, Cout, K, S, leaky)
print("batch on padded (L=8) z", fmt(zp), "y", fmt(yp))

reset_frames = [np.array([3.0, -2.0], dtype=f32), np.array([1.0, 1.0], dtype=f32), np.array([-4.0, 0.5], dtype=f32)]
o1 = stream(reset_frames, theta, Cin, Cout, K, S, leaky)
print("pre-reset stream (frames 0 and 2 emitted)", [fmt(o) for e, o in o1 if e])
print("frame0 alone", fmt(stream([frames[0]], theta, Cin, Cout, K, S, leaky)[0][1]))

outs1 = stream(frames, theta, Cin, Cout, K, 1, leaky)
zp1, yp1 = conv_forward(padded, theta, L + K - 1, Cin, Cout, K, 1, leaky)
print("stride1 stream", [fmt(o) for e, o in outs1], "batch", fmt(yp1))

Wd1 = np.array([0.5, -0.25, 0.75, 1.0], dtype=f32)
_, yc = conv_forward(x, theta, L, Cin, Cout, K, S, leaky)
pre = f32(0)
for j in range(4):
    pre = f32(pre + f32(Wd1[j] * yc[j]))
print("model dense pre", float(pre), "tanh", float(np.tanh(pre).astype(f32)))


def ram(L, Cin, Cout, K, S):
    Lout = (L - K) // S + 1
    P = K * Cin * Cout + Cout
    tr = 2 * P + 2 * L * Cin + 2 * Lout * Cout
    st = (K - 1) * Cin + 2 * Cout
    mac = Lout * Cout * K * Cin
    dense_p = (L * Cin) * (Lout * Cout) + Lout * Cout
    return Lout, P, tr, tr * 4, st, mac, dense_p


for cfg in [(128, 3, 8, 5, 1), (128, 3, 8, 5, 2), (6, 2, 2, 3, 2)]:
    print("RAM", cfg, ram(*cfg))
