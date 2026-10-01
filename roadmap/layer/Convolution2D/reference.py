import numpy as np

f32 = np.float32
H_FD = f32(1e-3)


def fmt(v):
    return "(" + ", ".join(f"{float(a):.7g}" for a in v) + ")"


def conv_forward(x, th, H, W, Ci, Co, KH, KW, SH, SW):
    Ho, Wo = (H - KH) // SH + 1, (W - KW) // SW + 1
    P = KH * KW * Ci
    z = np.zeros(Ho * Wo * Co, dtype=f32)
    for r in range(Ho):
        for s in range(Wo):
            for o in range(Co):
                acc = f32(th[P * Co + o])
                for p in range(KH):
                    row = ((r * SH + p) * W + s * SW) * Ci
                    tap = o * P + p * KW * Ci
                    for j in range(KW * Ci):
                        acc = f32(acc + f32(th[tap + j] * x[row + j]))
                z[(r * Wo + s) * Co + o] = acc
    return z


def conv_backward(x, th, delta, H, W, Ci, Co, KH, KW, SH, SW):
    Ho, Wo = (H - KH) // SH + 1, (W - KW) // SW + 1
    P = KH * KW * Ci
    gx = np.zeros(H * W * Ci, dtype=f32)
    gt = np.zeros(P * Co + Co, dtype=f32)
    for r in range(Ho):
        for s in range(Wo):
            for o in range(Co):
                d = delta[(r * Wo + s) * Co + o]
                for p in range(KH):
                    row = ((r * SH + p) * W + s * SW) * Ci
                    tap = o * P + p * KW * Ci
                    for j in range(KW * Ci):
                        gx[row + j] = f32(gx[row + j] + f32(th[tap + j] * d))
                        gt[tap + j] = f32(gt[tap + j] + f32(d * x[row + j]))
                gt[P * Co + o] = f32(gt[P * Co + o] + d)
    return gx, gt


def conv_ref64(x, th, H, W, Ci, Co, KH, KW, SH, SW):
    X = np.array(x, dtype=np.float64).reshape(H, W, Ci)
    P = KH * KW * Ci
    F = np.array(th[:P * Co], dtype=np.float64).reshape(Co, KH, KW, Ci)
    b = np.array(th[P * Co:], dtype=np.float64)
    Ho, Wo = (H - KH) // SH + 1, (W - KW) // SW + 1
    z = np.zeros((Ho, Wo, Co))
    for r in range(Ho):
        for s in range(Wo):
            for o in range(Co):
                z[r, s, o] = b[o] + np.sum(F[o] * X[r * SH:r * SH + KH, s * SW:s * SW + KW, :])
    return z.reshape(-1)


leaky = lambda z: np.where(z > 0, z, f32(0.1) * z).astype(f32)
dleaky = lambda z: np.where(z > 0, f32(1), f32(0.1)).astype(f32)
tanh = lambda z: np.tanh(z).astype(f32)

G = dict(H=5, W=4, Ci=2, Co=2, KH=2, KW=3, SH=2, SW=1)
x = np.array([0.5, -1.0, 0.25, 0.75, -0.5, 1.5, 1.0, -0.25,
              0.0, 2.0, -1.5, 0.5, 0.75, -0.5, 1.25, 0.25,
              -0.25, 1.0, 2.0, -0.75, 0.5, 0.5, -1.0, 1.5,
              1.5, -0.5, -0.75, 0.25, 0.25, -1.25, -0.25, 1.0,
              -2.0, 0.75, 1.0, -1.5, 0.0, 0.5, 0.75, -1.0], dtype=f32)
F0 = [0.2, -0.1, 0.4, 0.3, -0.5, 0.1, 0.3, 0.2, -0.2, 0.1, 0.1, -0.4]
F1 = [-0.3, 0.2, 0.1, -0.4, 0.6, 0.05, -0.1, 0.3, 0.2, -0.2, 0.4, 0.1]
theta = np.array(F0 + F1 + [0.1, -0.2], dtype=f32)
g = np.array([0.8, -1.3, 0.4, 0.6, -0.5, 0.25, 1.0, -0.75], dtype=f32)
x2 = np.array([f32(0.1) * f32(((7 * k) % 11) - 5) for k in range(40)], dtype=f32)
g2 = np.array([0.3, 0.9, -1.1, 0.2, 0.7, -0.4, -0.6, 1.2], dtype=f32)

print("== sizes")
for (H, W, KH, KW, SH, SW) in [(5, 4, 2, 3, 2, 1), (28, 28, 3, 3, 2, 2), (49, 10, 3, 3, 1, 1), (4, 3, 4, 3, 1, 1), (4, 3, 1, 1, 1, 1)]:
    print(H, W, KH, KW, SH, SW, "->", (H - KH) // SH + 1, (W - KW) // SW + 1)
print("P", 2 * 3 * 2 * 2 + 2)

print("== forward")
z = conv_forward(x, theta, **G)
print("z", fmt(z))
print("z64", fmt(conv_ref64(x, theta, **G)))
print("leaky", fmt(leaky(z)))
th0 = theta.copy(); th0[24:] = 0
print("ctor z (b=0)", fmt(conv_forward(x, th0, **G)))
Ff = np.array(F0 + F1, dtype=f32).reshape(2, 2, 3, 2)[:, ::-1, ::-1, :].reshape(-1)
print("flipped-kernel z", fmt(conv_forward(x, np.concatenate([Ff, theta[24:]]), **G)))
Ft = np.array(F0 + F1, dtype=f32).reshape(2, 2, 3, 2).transpose(0, 3, 1, 2).reshape(-1)
print("channel-first-kernel z", fmt(conv_forward(x, np.concatenate([Ft, theta[24:]]), **G)))
print("swapped-stride z", fmt(conv_forward(x, theta, H=5, W=4, Ci=2, Co=2, KH=2, KW=3, SH=1, SW=2)) if True else "")

print("== height one == Convolution1D fixture")
x1 = np.array([0.5, -1.0, 0.25, 0.75, -0.5, 1.5, 1.0, -0.25, 0.0, 2.0, -1.5, 0.5], dtype=f32)
th1 = np.array([0.2, -0.1, 0.4, 0.3, -0.5, 0.1, -0.3, 0.2, 0.1, -0.4, 0.6, 0.05, 0.1, -0.2], dtype=f32)
z1 = conv_forward(x1, th1, H=1, W=6, Ci=2, Co=2, KH=1, KW=3, SH=1, SW=2)
print("z", fmt(z1), "leaky", fmt(leaky(z1)))


def fd_input(fwd, x, g):
    n = np.zeros_like(x)
    for j in range(len(x)):
        xp = x.copy(); xp[j] = f32(xp[j] + H_FD)
        xm = x.copy(); xm[j] = f32(xm[j] - H_FD)
        n[j] = f32(f32(np.sum(g * fwd(xp), dtype=f32) - np.sum(g * fwd(xm), dtype=f32)) / f32(2 * H_FD))
    return n


print("== backward tanh")
def grads(x, g):
    z = conv_forward(x, theta, **G)
    y = tanh(z)
    d = (g * (f32(1) - y * y)).astype(f32)
    gx, gt = conv_backward(x, theta, d, **G)
    return y, d, gx, gt

y, d, gx, gt = grads(x, g)
print("tanh y", fmt(y))
print("delta", fmt(d))
print("gx", fmt(gx))
nx = fd_input(lambda xx: tanh(conv_forward(xx, theta, **G)), x, g)
print("gx fd", fmt(nx), "maxerr", np.max(np.abs(nx - gx)))
print("gt", fmt(gt))
def fd_theta(x, g):
    n = np.zeros_like(theta)
    for k in range(len(theta)):
        tp = theta.copy(); tp[k] = f32(tp[k] + H_FD)
        tm = theta.copy(); tm[k] = f32(tm[k] - H_FD)
        n[k] = f32(f32(np.sum(g * tanh(conv_forward(x, tp, **G)), dtype=f32) - np.sum(g * tanh(conv_forward(x, tm, **G)), dtype=f32)) / f32(2 * H_FD))
    return n
nt = fd_theta(x, g)
print("gt fd", fmt(nt), "maxerr", np.max(np.abs(nt - gt)))
print("x2", fmt(x2))
_, _, _, gt2 = grads(x2, g2)
print("gt2", fmt(gt2))
acc = (gt + gt2).astype(f32)
print("acc", fmt(acc))
nt2 = fd_theta(x2, g2)
print("acc fd maxerr", np.max(np.abs((nt + nt2) - acc)))
print("z2", fmt(conv_forward(x2, theta, **G)))

print("== pooling")
def pool_max(x, H, W, C, PH, PW, SH, SW):
    Ho, Wo = (H - PH) // SH + 1, (W - PW) // SW + 1
    y = np.zeros(Ho * Wo * C, dtype=f32); a = np.zeros(Ho * Wo * C, dtype=int); margin = 1e9
    for r in range(Ho):
        for s in range(Wo):
            for c in range(C):
                best = None; idx = None; vals = []
                for p in range(PH):
                    for q in range(PW):
                        k = ((r * SH + p) * W + s * SW + q) * C + c
                        vals.append(x[k])
                        if best is None or x[k] > best:
                            best, idx = x[k], k
                vs = sorted(vals, reverse=True); margin = min(margin, vs[0] - vs[1])
                y[(r * Wo + s) * C + c] = best; a[(r * Wo + s) * C + c] = idx
    return y, a, margin

def pool_avg(x, H, W, C, PH, PW, SH, SW):
    Ho, Wo = (H - PH) // SH + 1, (W - PW) // SW + 1
    inv = f32(1) / f32(PH * PW)
    y = np.zeros(Ho * Wo * C, dtype=f32)
    for r in range(Ho):
        for s in range(Wo):
            for c in range(C):
                acc = f32(0)
                for p in range(PH):
                    for q in range(PW):
                        acc = f32(acc + x[((r * SH + p) * W + s * SW + q) * C + c])
                y[(r * Wo + s) * C + c] = f32(acc * inv)
    return y

def pool_avg_back(g, H, W, C, PH, PW, SH, SW):
    Ho, Wo = (H - PH) // SH + 1, (W - PW) // SW + 1
    inv = f32(1) / f32(PH * PW)
    gx = np.zeros(H * W * C, dtype=f32)
    for r in range(Ho):
        for s in range(Wo):
            for p in range(PH):
                for q in range(PW):
                    for c in range(C):
                        k = ((r * SH + p) * W + s * SW + q) * C + c
                        gx[k] = f32(gx[k] + f32(g[(r * Wo + s) * C + c] * inv))
    return gx

pg = np.array([0.8, -1.3, 0.4, 0.25, -0.6, 1.1, 0.5, -0.9], dtype=f32)
y, a, m = pool_max(x, 5, 4, 2, 2, 2, 2, 2)
print("max 2x2 y", fmt(y), "argmax", list(a), "margin", m)
y, a, m = pool_max(x, 5, 4, 2, 2, 3, 2, 1)
print("max 2x3 s(2,1) y", fmt(y), "argmax", list(a), "margin", m)
gx = np.zeros(40, dtype=f32)
for j in range(8): gx[a[j]] += pg[j]
print("max back", fmt(gx))
nx = fd_input(lambda xx: pool_max(xx, 5, 4, 2, 2, 3, 2, 1)[0], x, pg)
print("max back fd", fmt(nx), "maxerr", np.max(np.abs(nx - gx)))
ya = pool_avg(x, 5, 4, 2, 2, 3, 2, 1)
print("avg y", fmt(ya))
ga = pool_avg_back(pg, 5, 4, 2, 2, 3, 2, 1)
print("avg back", fmt(ga))
nx = fd_input(lambda xx: pool_avg(xx, 5, 4, 2, 2, 3, 2, 1), x, pg)
print("avg back fd maxerr", np.max(np.abs(nx - ga)))
yg = pool_avg(x, 5, 4, 2, 5, 4, 1, 1)
print("gap y", fmt(yg), "sums", np.sum(x.reshape(20, 2).astype(np.float64), axis=0))
gg = np.array([0.8, -1.3], dtype=f32)
gb = pool_avg_back(gg, 5, 4, 2, 5, 4, 1, 1)
print("gap back per pos", fmt(gb[:2]))
nx = fd_input(lambda xx: pool_avg(xx, 5, 4, 2, 5, 4, 1, 1), x, gg)
print("gap back fd maxerr", np.max(np.abs(nx - gb)))
xt = np.array([-0.3, 0.7, 0.2, 0.2, 0.7, 0.1, -0.5, 0.2], dtype=f32)
y, a, m = pool_max(xt, 2, 4, 1, 2, 2, 2, 2)
gx = np.zeros(8, dtype=f32)
for j, gv in enumerate([1.0, 2.0]): gx[a[j]] += gv
print("tie y", fmt(y), "argmax", list(a), "back", fmt(gx))

print("== model conv(leaky) -> GAP -> dense(tanh)")
Wd = np.array([0.5, -0.25], dtype=f32); bd = f32(0.1)
def model(xx):
    z = conv_forward(xx, theta, **G); yl = leaky(z)
    pooled = pool_avg(yl, 2, 2, 2, 2, 2, 1, 1)
    zz = f32(f32(bd + f32(Wd[0] * pooled[0])) + f32(Wd[1] * pooled[1]))
    return np.array([np.tanh(zz)], dtype=f32), z, yl, pooled, zz
ym, z, yl, pooled, zz = model(x)
print("pooled", fmt(pooled), "zz", zz, "y", ym)
dz = f32(1) - ym[0] * ym[0]
dpool = (Wd * dz).astype(f32)
dyl = pool_avg_back(dpool, 2, 2, 2, 2, 2, 1, 1)
dconv = (dyl * dleaky(z)).astype(f32)
dx, _ = conv_backward(x, theta, dconv, **G)
print("dpool", fmt(dpool), "dx", fmt(dx))
nx = fd_input(lambda xx: model(xx)[0], x, np.array([1.0], dtype=f32))
print("dx fd maxerr", np.max(np.abs(nx - dx)), "min |z|", np.min(np.abs(z)))

print("== memory examples")
def conv_mem(H, W, Ci, Co, KH, KW, SH, SW):
    Ho, Wo = (H - KH) // SH + 1, (W - KW) // SW + 1
    P = KH * KW * Ci * Co + Co
    ram = 2 * P + 2 * H * W * Ci + 2 * Ho * Wo * Co
    macs = Ho * Wo * Co * KH * KW * Ci
    im2col = Ho * Wo * KH * KW * Ci
    dense = H * W * Ci * Ho * Wo * Co + Ho * Wo * Co
    print(dict(H=H, W=W, Ci=Ci, Co=Co, KH=KH, KW=KW, SH=SH, SW=SW, Ho=Ho, Wo=Wo, P=P, ram_floats=ram, ram_bytes=4 * ram, macs=macs, infer=2 * Ho * Wo * Co, im2col=im2col, dense=dense))
    return Ho, Wo
Ho, Wo = conv_mem(49, 10, 1, 8, 3, 3, 1, 1)
Hp, Wp = (Ho - 2) // 2 + 1, (Wo - 2) // 2 + 1
print("maxpool 2x2 on", Ho, Wo, 8, "->", Hp, Wp, "out", Hp * Wp * 8, "ram max", 2 * Hp * Wp * 8 + Ho * Wo * 8)
conv_mem(28, 28, 1, 8, 3, 3, 2, 2)
print("fixture ram", 2 * 26 + 2 * 40 + 2 * 8)
