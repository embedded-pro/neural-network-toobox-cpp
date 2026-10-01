import struct, math

def f(x):
    return struct.unpack('f', struct.pack('f', x))[0]

def fv(v):
    return [f(x) for x in v]

def fmul(a, b): return f(f(a) * f(b))
def fadd(a, b): return f(f(a) + f(b))
def fsub(a, b): return f(f(a) - f(b))
def fdiv(a, b): return f(f(a) / f(b))

def forward(a, b, x):
    return [fadd(fmul(ai, xi), bi) for ai, bi, xi in zip(a, b, x)]

def proj(a, b, x, g):
    y = forward(a, b, x)
    s = f(0.0)
    for gi, yi in zip(g, y):
        s = fadd(s, fmul(gi, yi))
    return s

h = f(1e-3)
print("== fixture")
a = fv([2.0, -0.5, 0.25]); b = fv([1.0, 0.5, -3.0])
x = fv([0.3, -0.7, 1.1]); g = fv([0.8, -1.3, 0.4])
print("forward exact", [ai * xi + bi for ai, bi, xi in zip([2.0, -0.5, 0.25], [1.0, 0.5, -3.0], [0.3, -0.7, 1.1])])
print("forward f32  ", forward(a, b, x))
print("identity fwd ", forward(fv([1,1,1]), fv([0,0,0]), x))

p2 = fv([1.5, 0.0, -4.0, -1.0, 2.0, 0.5])
print("setparams exact", [1.5*0.3-1.0, 0.0*-0.7+2.0, -4.0*1.1+0.5])
print("setparams f32  ", forward(p2[:3], p2[3:], x))

an = [fmul(ai, gi) for ai, gi in zip(a, g)]
print("backward analytic", an, "exact", [2.0*0.8, -0.5*-1.3, 0.25*0.4])
num = []
for j in range(3):
    xp = list(x); xm = list(x)
    xp[j] = fadd(xp[j], h); xm[j] = fsub(xm[j], h)
    num.append(fdiv(fsub(proj(a, b, xp, g), proj(a, b, xm, g)), f(2.0) * h))
print("backward FD      ", num, "maxerr", max(abs(u - v) for u, v in zip(an, num)))

print("N8 dL/da = g*x exact", [0.8*0.3, -1.3*-0.7, 0.4*1.1], "dL/db = g", [0.8, -1.3, 0.4])
dan = []
for j in range(3):
    ap = list(a); am = list(a); ap[j] = fadd(ap[j], h); am[j] = fsub(am[j], h)
    dan.append(fdiv(fsub(proj(ap, b, x, g), proj(am, b, x, g)), f(2.0) * h))
dbn = []
for j in range(3):
    bp = list(b); bm = list(b); bp[j] = fadd(bp[j], h); bm[j] = fsub(bm[j], h)
    dbn.append(fdiv(fsub(proj(a, bp, x, g), proj(a, bm, x, g)), f(2.0) * h))
print("N8 FD dL/da", dan, "FD dL/db", dbn)

print("== FromBatchNorm")
gamma = [1.5, 0.8, 2.0]; beta = [0.1, -0.2, 0.0]; mu = [0.5, -1.0, 3.0]; var = [4.0, 0.25, 0.01]; eps = 1e-3
ae = [gm / math.sqrt(v + eps) for gm, v in zip(gamma, var)]
be = [bt - ai * m for bt, ai, m in zip(beta, ae, mu)]
print("a exact", ae); print("b exact", be)
af = [fdiv(f(gm), f(math.sqrt(fadd(v, eps)))) for gm, v in zip(gamma, var)]
bf = [fsub(bt, fmul(ai, m)) for bt, ai, m in zip(beta, af, mu)]
print("a f32", af); print("b f32", bf)
xb = [1.0, -0.5, 3.2]
bn = [gm * (xi - m) / math.sqrt(v + eps) + bt for gm, xi, m, v, bt in zip(gamma, xb, mu, var, beta)]
print("BN direct exact at", xb, bn)
print("folded f32     ", forward(af, bf, fv(xb)))

print("== Standardisation")
mean = [9.81, -1.2, 3.3]; sd = [0.5, 35.0, 0.02]
a_s = [fdiv(f(1.0), f(s)) for s in sd]
b_s = [f(-fmul(m, ai)) for m, ai in zip(mean, a_s)]
print("a", a_s, "exact", [1/s for s in sd]); print("b", b_s, "exact", [-m/s for m, s in zip(mean, sd)])
print("Forward(mean)", forward(a_s, b_s, fv(mean)))
x1 = [fadd(m, s) for m, s in zip(mean, sd)]
print("x = mean+sd f32", x1, "Forward", forward(a_s, b_s, x1))
x2 = [fsub(m, f(2.0) * f(s)) for m, s in zip(mean, sd)]
print("x = mean-2sd f32", x2, "Forward", forward(a_s, b_s, x2))

print("== cancellation example")
mP, sP, xP = 101325.0, 50.0, 101400.0
aP = fdiv(1.0, sP); bP = f(-fmul(mP, aP))
print("a*x+b f32", fadd(fmul(aP, xP), bP), "a*(x-mu) f32", fmul(aP, fsub(xP, mP)), "exact", (xP - mP) / sP)
xP2 = 101401.0
print("x=101401 a*x+b", fadd(fmul(aP, xP2), bP), "exact", (xP2 - mP) / sP)
worst = 0.0
for k in range(0, 2001):
    xx = f(101200.0 + k * 0.125)
    worst = max(worst, abs(fadd(fmul(aP, xx), bP) - (xx - mP) / sP))
print("worst abs err over [101200, 101450] step 0.125:", worst)

print("== Model integration: Affine(3) -> Dense(3,1,tanh)")
W = [0.4, -0.3, 0.2]; c = 0.1
z = sum(wi * (ai * xi + bi) for wi, ai, xi, bi in zip(W, [2.0, -0.5, 0.25], [0.3, -0.7, 1.1], [1.0, 0.5, -3.0])) + c
print("pre", z, "out", math.tanh(z))
Wf = [wi * ai for wi, ai in zip(W, [2.0, -0.5, 0.25])]
cf = c + sum(wi * bi for wi, bi in zip(W, [1.0, 0.5, -3.0]))
print("folded W'", Wf, "c'", cf, "out", math.tanh(sum(wi * xi for wi, xi in zip(Wf, [0.3, -0.7, 1.1])) + cf))
xf = fv([0.3, -0.7, 1.1])
yf = forward(a, b, xf)
zf = f(c)
for wi, yi in zip(fv(W), yf): zf = fadd(zf, fmul(wi, yi))
print("f32 chain pre", zf, "out", f(math.tanh(zf)))
zf2 = f(cf)
for wi, xi in zip(fv(Wf), xf): zf2 = fadd(zf2, fmul(wi, xi))
print("f32 folded pre", zf2, "out", f(math.tanh(zf2)))
print("model params", [2.0, -0.5, 0.25, 1.0, 0.5, -3.0] + W + [c])
