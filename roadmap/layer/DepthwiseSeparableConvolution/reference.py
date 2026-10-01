import numpy as np

f32 = np.float32
H_FD = f32(1e-3)


def fmt(v):
    return "(" + ", ".join(f"{float(a):.7g}" for a in np.ravel(v)) + ")"


def leaky(z):
    return np.array([a if a > 0 else f32(f32(0.1) * a) for a in z], dtype=f32)


def dleaky(z, y, g):
    return np.array([gg if a > 0 else f32(f32(0.1) * gg) for a, gg in zip(z, g)], dtype=f32)


def tanh(z):
    return np.tanh(z.astype(f32)).astype(f32)


def dtanh(z, y, g):
    return np.array([f32(gg * f32(f32(1) - f32(yy * yy))) for yy, gg in zip(y, g)], dtype=f32)


def sigm(z):
    return np.array([f32(1) / (f32(1) + np.exp(-a, dtype=f32)) for a in z], dtype=f32)


def dsigm(z, y, g):
    return np.array([f32(gg * f32(yy * f32(f32(1) - yy))) for yy, gg in zip(y, g)], dtype=f32)


def dims(Hh, Ww, KH, KW, SH, SW):
    return (Hh - KH) // SH + 1, (Ww - KW) // SW + 1


def dw_forward(x, th, Hh, Ww, C, KH, KW, SH, SW):
    Ho, Wo = dims(Hh, Ww, KH, KW, SH, SW)
    z = np.zeros(Ho * Wo * C, dtype=f32)
    boff = KH * KW * C
    for r in range(Ho):
        for s in range(Wo):
            out = (r * Wo + s) * C
            for c in range(C):
                z[out + c] = th[boff + c]
            for p in range(KH):
                for q in range(KW):
                    base = ((r * SH + p) * Ww + s * SW + q) * C
                    tap = (p * KW + q) * C
                    for c in range(C):
                        z[out + c] = f32(z[out + c] + f32(th[tap + c] * x[base + c]))
    return z


def dw_backward(x, th, delta, G, Hh, Ww, C, KH, KW, SH, SW, goff=0):
    Ho, Wo = dims(Hh, Ww, KH, KW, SH, SW)
    gx = np.zeros(Hh * Ww * C, dtype=f32)
    boff = KH * KW * C
    for r in range(Ho):
        for s in range(Wo):
            out = (r * Wo + s) * C
            for p in range(KH):
                for q in range(KW):
                    base = ((r * SH + p) * Ww + s * SW + q) * C
                    tap = (p * KW + q) * C
                    for c in range(C):
                        d = delta[out + c]
                        gx[base + c] = f32(gx[base + c] + f32(th[tap + c] * d))
                        G[goff + tap + c] = f32(G[goff + tap + c] + f32(d * x[base + c]))
            for c in range(C):
                G[goff + boff + c] = f32(G[goff + boff + c] + delta[out + c])
    return gx


def dw_layer(x, th, dims_, act):
    z = dw_forward(x, th, *dims_)
    return z, act(z)


def sep_forward(x, th, Hh, Ww, Ci, Co, KH, KW, SH, SW, a1f, a2f):
    Ho, Wo = dims(Hh, Ww, KH, KW, SH, SW)
    nd = KH * KW * Ci + Ci
    z1 = dw_forward(x, th[:nd], Hh, Ww, Ci, KH, KW, SH, SW)
    a1 = a1f(z1)
    pw = th[nd: nd + Co * Ci]
    bp = th[nd + Co * Ci:]
    z2 = np.zeros(Ho * Wo * Co, dtype=f32)
    for m in range(Ho * Wo):
        for o in range(Co):
            acc = f32(bp[o])
            for c in range(Ci):
                acc = f32(acc + f32(pw[o * Ci + c] * a1[m * Ci + c]))
            z2[m * Co + o] = acc
    return z1, a1, z2, a2f(z2)


def sep_backward(x, th, G, g, Hh, Ww, Ci, Co, KH, KW, SH, SW, a1f, a2f, d1f, d2f):
    Ho, Wo = dims(Hh, Ww, KH, KW, SH, SW)
    nd = KH * KW * Ci + Ci
    z1, a1, z2, y = sep_forward(x, th, Hh, Ww, Ci, Co, KH, KW, SH, SW, a1f, a2f)
    d2 = d2f(z2, y, g)
    g1 = np.zeros(Ho * Wo * Ci, dtype=f32)
    for m in range(Ho * Wo):
        for o in range(Co):
            d = d2[m * Co + o]
            for c in range(Ci):
                g1[m * Ci + c] = f32(g1[m * Ci + c] + f32(th[nd + o * Ci + c] * d))
                G[nd + o * Ci + c] = f32(G[nd + o * Ci + c] + f32(d * a1[m * Ci + c]))
            G[nd + Co * Ci + o] = f32(G[nd + Co * Ci + o] + d)
    d1 = d1f(z1, a1, g1)
    gx = dw_backward(x, th, d1, G, Hh, Ww, Ci, KH, KW, SH, SW, 0)
    return gx, d2, g1, d1


def fd(fun, v):
    num = np.zeros_like(v)
    for k in range(len(v)):
        p = v.copy(); m = v.copy()
        p[k] = f32(p[k] + H_FD); m[k] = f32(m[k] - H_FD)
        num[k] = f32((fun(p) - fun(m)) / f32(2 * H_FD))
    return num


def proj(g, y):
    return f32(np.sum((g * y).astype(f32), dtype=f32))


print("== sizes")
for (Hh, Ww, C, KH, KW, SH, SW) in [(3, 5, 2, 2, 2, 1, 2), (1, 6, 2, 1, 3, 1, 2), (25, 5, 64, 3, 3, 1, 1), (49, 10, 1, 3, 3, 1, 1)]:
    print(Hh, Ww, C, KH, KW, SH, SW, "->", dims(Hh, Ww, KH, KW, SH, SW))

DWD = (3, 5, 2, 2, 2, 1, 2)
Kdw = [[0.5, -0.3], [-0.2, 0.4], [0.1, 0.6], [0.3, -0.25]]
bdw = [0.05, -0.1]
th_dw = np.array(sum(Kdw, []) + bdw, dtype=f32)
th_dw0 = np.array(sum(Kdw, []) + [0, 0], dtype=f32)
x = np.array([0.4, -0.6, 1.0, 0.2, -0.8, 0.5, 0.3, -1.2, 0.7, 0.9,
              -0.5, 1.1, 0.25, -0.4, 0.6, 0.8, -1.0, -0.3, 0.2, 1.5,
              0.9, -0.2, -0.7, 0.35, 1.2, -0.9, 0.45, 0.1, -0.15, 0.65], dtype=f32)
g = np.array([0.7, -1.1, 0.3, 0.5, -0.4, 0.9, 1.2, -0.6], dtype=f32)
x2 = np.array([-0.3, 0.8, 0.6, -0.1, 0.2, 0.4, -0.9, 0.5, 1.1, -0.7,
               0.35, -0.25, -0.6, 1.3, 0.15, 0.05, 0.7, -0.45, -1.1, 0.2,
               0.5, 0.9, -0.2, -0.8, 0.4, 0.6, -0.35, 1.0, 0.75, -0.5], dtype=f32)
g2 = np.array([-0.5, 0.4, 1.0, -0.8, 0.6, 0.2, -0.3, 0.9], dtype=f32)

print("P_dw", len(th_dw))
print("== depthwise forward")
z, y = dw_layer(x, th_dw, DWD, leaky)
print("z", fmt(z)); print("leaky", fmt(y))
# float64 check
x64 = x.astype(np.float64); t64 = th_dw.astype(np.float64)
z64 = []
for r in range(2):
    for s in range(2):
        for c in range(2):
            acc = t64[8 + c]
            for p in range(2):
                for q in range(2):
                    acc += t64[(p * 2 + q) * 2 + c] * x64[((r + p) * 5 + 2 * s + q) * 2 + c]
            z64.append(acc)
print("z64", fmt(z64))
z0, _ = dw_layer(x, th_dw0, DWD, leaky)
print("ctor (b=0) z", fmt(z0))
# wrong-layout variants to make sure the fixture discriminates
thf = th_dw.copy(); thf[:8] = np.array(Kdw, dtype=f32)[::-1].ravel()
print("flipped-kernel z", fmt(dw_forward(x, thf, *DWD)))
thcm = th_dw.copy(); thcm[:8] = np.array(Kdw, dtype=f32).T.ravel()
print("channel-major-kernel z", fmt(dw_forward(x, thcm, *DWD)))

print("== depthwise backward tanh")
zt, yt = dw_layer(x, th_dw, DWD, tanh)
print("tanh y", fmt(yt))
delta = dtanh(zt, yt, g)
print("delta", fmt(delta))
G = np.zeros_like(th_dw)
gx = dw_backward(x, th_dw, delta, G, *DWD)
print("gx", fmt(gx))
nx = fd(lambda xx: proj(g, dw_layer(xx, th_dw, DWD, tanh)[1]), x)
print("gx fd", fmt(nx), "maxerr", float(np.max(np.abs(nx - gx))))
print("G", fmt(G))
nt = fd(lambda tt: proj(g, dw_layer(x, tt, DWD, tanh)[1]), th_dw)
print("G fd", fmt(nt), "maxerr", float(np.max(np.abs(nt - G))))
zt2, yt2 = dw_layer(x2, th_dw, DWD, tanh)
G2 = np.zeros_like(th_dw)
dw_backward(x2, th_dw, dtanh(zt2, yt2, g2), G2, *DWD)
print("G2", fmt(G2))
acc = G.copy()
dw_backward(x2, th_dw, dtanh(zt2, yt2, g2), acc, *DWD)
print("G+G2", fmt(acc))
nacc = fd(lambda tt: f32(proj(g, dw_layer(x, tt, DWD, tanh)[1]) + proj(g2, dw_layer(x2, tt, DWD, tanh)[1])), th_dw)
print("acc fd maxerr", float(np.max(np.abs(nacc - acc))))
print("min |z| (tanh sample 1, 2)", float(np.min(np.abs(zt))), float(np.min(np.abs(zt2))))

print("== separable")
SD = (3, 5, 2, 3, 2, 2, 1, 2)
Kpw = [[0.6, -0.4], [-0.3, 0.8], [0.2, 0.5]]
bpw = [0.1, -0.05, 0.2]
th_s = np.array(sum(Kdw, []) + bdw + sum(Kpw, []) + bpw, dtype=f32)
th_s0 = np.array(sum(Kdw, []) + [0, 0] + sum(Kpw, []) + [0, 0, 0], dtype=f32)
print("P_sep", len(th_s), "ctor", fmt(th_s0))
gs = np.array([0.5, -0.8, 0.3, 1.1, -0.2, 0.6, -0.7, 0.4, 0.9, 0.25, -1.0, 0.15], dtype=f32)
z1, a1, z2, ys = sep_forward(x, th_s, *SD, leaky, leaky)
print("z1", fmt(z1)); print("a1", fmt(a1)); print("z2", fmt(z2)); print("y leaky/leaky", fmt(ys))
# float64
a164 = [v if v > 0 else 0.1 * v for v in z64]
z264 = []
for m in range(4):
    for o in range(3):
        z264.append(bpw[o] + sum(Kpw[o][c] * a164[m * 2 + c] for c in range(2)))
print("z2_64", fmt(z264))
# no intermediate nonlinearity (linear dw) to show difference
print("== separable backward tanh(dw) sigmoid(pw)")
z1, a1, z2, ys = sep_forward(x, th_s, *SD, tanh, sigm)
print("a1", fmt(a1)); print("y", fmt(ys))
Gs = np.zeros_like(th_s)
gxs, d2, g1, d1 = sep_backward(x, th_s, Gs, gs, *SD, tanh, sigm, dtanh, dsigm)
print("delta2", fmt(d2)); print("g1", fmt(g1)); print("delta1", fmt(d1))
print("gx", fmt(gxs))
nxs = fd(lambda xx: proj(gs, sep_forward(xx, th_s, *SD, tanh, sigm)[3]), x)
print("gx fd", fmt(nxs), "maxerr", float(np.max(np.abs(nxs - gxs))))
print("G", fmt(Gs))
nts = fd(lambda tt: proj(gs, sep_forward(x, tt, *SD, tanh, sigm)[3]), th_s)
print("G fd", fmt(nts), "maxerr", float(np.max(np.abs(nts - Gs))))
gs2 = np.array([-0.4, 0.7, 0.2, -0.9, 0.5, -0.3, 1.0, -0.6, 0.35, 0.8, -0.15, 0.45], dtype=f32)
Gs2 = np.zeros_like(th_s)
sep_backward(x2, th_s, Gs2, gs2, *SD, tanh, sigm, dtanh, dsigm)
print("G2", fmt(Gs2))
accs = Gs.copy()
sep_backward(x2, th_s, accs, gs2, *SD, tanh, sigm, dtanh, dsigm)
print("G+G2", fmt(accs))
naccs = fd(lambda tt: f32(proj(gs, sep_forward(x, tt, *SD, tanh, sigm)[3]) + proj(gs2, sep_forward(x2, tt, *SD, tanh, sigm)[3])), th_s)
print("acc fd maxerr", float(np.max(np.abs(naccs - accs))))

print("== 1D alias, Convolution1D fixture input")
D1 = (1, 6, 2, 3, 1, 3, 1, 2)
x1 = np.array([0.5, -1.0, 0.25, 0.75, -0.5, 1.5, 1.0, -0.25, 0.0, 2.0, -1.5, 0.5], dtype=f32)
Kdw1 = [[0.2, -0.1], [0.4, 0.3], [-0.5, 0.1]]
th1 = np.array(sum(Kdw1, []) + [0.05, -0.1] + sum(Kpw, []) + bpw, dtype=f32)
z1a, a1a, z2a, y1a = sep_forward(x1, th1, *D1, leaky, leaky)
print("P1", len(th1), "z1", fmt(z1a), "a1", fmt(a1a), "z2", fmt(z2a), "y", fmt(y1a))

print("== model sep(leaky, leaky) -> dense(tanh)")
wd = np.array([0.3, -0.2, 0.5, 0.1, 0.4, -0.6, -0.25, 0.2, 0.35, 0.15, -0.1, 0.45], dtype=f32)
_, _, _, ys = sep_forward(x, th_s, *SD, leaky, leaky)
zz = f32(0)
for i in range(12):
    zz = f32(zz + f32(wd[i] * ys[i]))
print("dense pre", float(zz), "y", float(np.tanh(zz, dtype=f32)))

print("== cost examples")
for (Hh, Ww, Ci, Co, KH, KW, SH, SW) in [(25, 5, 64, 64, 3, 3, 1, 1), (3, 5, 2, 3, 2, 2, 1, 2), (1, 128, 3, 16, 1, 5, 1, 1)]:
    Ho, Wo = dims(Hh, Ww, KH, KW, SH, SW)
    Pdw = KH * KW * Ci + Ci; Ppw = Ci * Co + Co; P = Pdw + Ppw
    Pstd = KH * KW * Ci * Co + Co
    mdw = Ho * Wo * Ci * KH * KW; mpw = Ho * Wo * Ci * Co; mstd = Ho * Wo * Co * KH * KW * Ci
    ram_sep = 2 * P + 2 * Hh * Ww * Ci + 4 * Ho * Wo * Ci + 3 * Ho * Wo * Co
    ram_dw = 2 * Pdw + 2 * Hh * Ww * Ci + 3 * Ho * Wo * Ci
    infer = Ho * Wo * Co + Ci
    infer_unfused = Ho * Wo * Ci + Ho * Wo * Co
    print(dict(H=Hh, W=Ww, Ci=Ci, Co=Co, K=(KH, KW), Ho=Ho, Wo=Wo, Pdw=Pdw, Ppw=Ppw, P=P, Pstd=Pstd,
               macs_dw=mdw, macs_pw=mpw, macs=mdw + mpw, macs_std=mstd, ratio=(mdw + mpw) / mstd,
               ratio_formula=1 / Co + 1 / (KH * KW), ram_sep=ram_sep, ram_sep_B=4 * ram_sep, ram_dw=ram_dw,
               infer_fused=infer, infer_unfused=infer_unfused))

print("== discrimination for the separable forward")
ident = lambda z: z.astype(f32)
print("no depthwise activation y", fmt(sep_forward(x, th_s, *SD, ident, leaky)[3]))
thT = th_s.copy(); thT[10:16] = np.array(Kpw, dtype=f32).T.ravel()[:6]
print("pointwise read as [c][o] y", fmt(sep_forward(x, thT, *SD, leaky, leaky)[3]))
print("z2 min abs", float(np.min(np.abs(sep_forward(x, th_s, *SD, leaky, leaky)[2]))))
print("tanh/sigmoid separable min |z1|, |z2|", float(np.min(np.abs(sep_forward(x, th_s, *SD, tanh, sigm)[0]))), float(np.min(np.abs(sep_forward(x, th_s, *SD, tanh, sigm)[2]))))
