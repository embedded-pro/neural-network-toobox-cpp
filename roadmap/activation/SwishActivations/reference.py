import math
import numpy as np
np.seterr(all='ignore')
F = np.float32
h = F(1e-3); twoH = F(2) * h; tol = 1e-3
one, zero, three, half = F(1), F(0), F(3), F(0.5)
sixth = F(1) / F(6); third = F(1) / F(3)

def sigmoid(x):
    x = F(x)
    if x >= 0: return F(one / F(one + np.exp(-x)))
    e = np.exp(x); return F(e / F(one + e))

def silu_f(x): x = F(x); return F(x * sigmoid(x))
def silu_b(x):
    x = F(x); s = sigmoid(x); return F(s * F(one + F(x * F(one - s))))
def silu_b_outreuse(x):
    x = F(x); s = sigmoid(x); a = F(x * s); return F(a + F(s * F(one - a)))

def hsig_f(x):
    x = F(x)
    if x <= -three: return zero
    if x >= three: return one
    return F(F(x * sixth) + half)
def hsig_b(x):
    x = F(x); return sixth if (-three < x < three) else zero

def hsw_f(x):
    x = F(x)
    if x <= -three: return zero
    if x >= three: return x
    return F(F(x * F(x + three)) * sixth)
def hsw_b(x):
    x = F(x)
    if x <= -three: return zero
    if x >= three: return one
    return F(F(x * third) + half)

def silu64(x): return x / (1 + math.exp(-x)) if x > -700 else 0.0
def silu64_b(x):
    s = 1 / (1 + math.exp(-x)) if x > -700 else 0.0
    return s * (1 + x * (1 - s))

def fd(f, x): x = F(x); return F(F(f(F(x + h)) - f(F(x - h))) / twoH)
def fdvec(f, b, xs, g):
    xs = [F(v) for v in xs]; g = [F(v) for v in g]
    def proj(xx):
        s = F(0)
        for i in range(len(xx)): s = F(s + F(g[i] * f(xx[i])))
        return s
    num = []
    for i in range(len(xs)):
        p = list(xs); m = list(xs); p[i] = F(p[i] + h); m[i] = F(m[i] - h)
        num.append(F(F(proj(p) - proj(m)) / twoH))
    ana = [F(g[i] * b(xs[i])) for i in range(len(xs))]
    return ana, num

def show(name, vals): print(f"  {name}: " + ", ".join(f"{float(v):.7g}" for v in vals))

xstar = -1.0
for _ in range(60):
    s = 1 / (1 + math.exp(-xstar)); g = 1 + xstar * (1 - s)
    dg = (1 - s) - xstar * s * (1 - s)
    xstar -= g / dg
print(f"SiLU minimum x* = {xstar:.9f}, f(x*) = {silu64(xstar):.9f} (= x*+1 = {xstar+1:.9f})")

print("SiLU")
X = [-2.0, 0.0, 1.0, 3.0]
show("Forward f32  " + str(X), [silu_f(x) for x in X]); show("Forward f64", [silu64(x) for x in X])
Xb = [0.0, 2.0, -2.0, xstar]
show("Backward f32 " + str(Xb), [silu_b(x) for x in Xb]); show("Backward f64", [silu64_b(x) for x in Xb])
Xfd = [-3.0, -0.5, 1.0, 4.0]
for x in Xfd:
    print(f"  x={x}: analytic={float(silu_b(x)):.7g} fd={float(fd(silu_f, x)):.7g} err={abs(float(silu_b(x)) - float(fd(silu_f, x))):.2e}")
up = [0.3, -1.2, 0.8, -0.4]
a, n = fdvec(silu_f, silu_b, Xfd, up); show("vec analytic", a); show("vec fd", n); print("  vec max err", max(abs(float(p) - float(q)) for p, q in zip(a, n)))
for x in [50.0, -50.0, 1e30, -1e30]:
    print(f"  x={x:g}: f={float(silu_f(x))!r} f'={float(silu_b(x))!r} f'(output-reuse a+s(1-a))={float(silu_b_outreuse(x))!r}")
for x in [1e6, 1e7, 1e8]:
    print(f"  output-reuse form at x={x:g}: {float(silu_b_outreuse(x))!r} vs chosen {float(silu_b(x))!r}")
grid = np.linspace(-20, 20, 400001)
err = max(abs(float(silu_b(x)) - silu64_b(float(F(x)))) for x in grid[::10])
print(f"  max |f'_f32 - f'_f64| on [-20,20]: {err:.2e}")
fmin = min(silu64_b(float(x)) for x in grid[::10]); xm = grid[::10][int(np.argmin([silu64_b(float(x)) for x in grid[::10]]))]
print(f"  min f' = {fmin:.6f} at x ~ {xm:.3f}; max f' = {max(silu64_b(float(x)) for x in grid[::10]):.6f}")

print("HardSigmoid")
X = [-4.0, -3.0, -1.5, 0.0, 1.5, 3.0, 4.0]
show("Forward " + str(X), [hsig_f(x) for x in X])
Xb = [-4.0, -3.0, 0.0, 2.9, 3.0, 4.0]
show("Backward " + str(Xb), [hsig_b(x) for x in Xb]); print(f"  1/6 f32 = {float(sixth)!r}")
for x in [-2.0, 0.0, 1.0, 2.5]:
    print(f"  x={x}: analytic={float(hsig_b(x)):.7g} fd={float(fd(hsig_f, x)):.7g} err={abs(float(hsig_b(x)) - float(fd(hsig_f, x))):.2e}")
Xv = [-4.0, -1.5, 0.7, 3.5]
a, n = fdvec(hsig_f, hsig_b, Xv, up); show("vec analytic " + str(Xv), a); show("vec fd", n); print("  vec max err", max(abs(float(p) - float(q)) for p, q in zip(a, n)))

print("HardSwish")
X = [-4.0, -3.0, -1.5, 0.0, 1.0, 3.0, 5.0]
show("Forward " + str(X), [hsw_f(x) for x in X])
Xb = [-4.0, -3.0, -1.5, 0.0, 1.0, 3.0, 5.0]
show("Backward " + str(Xb), [hsw_b(x) for x in Xb])
for x in [-2.0, -0.5, 1.0, 2.5]:
    print(f"  x={x}: analytic={float(hsw_b(x)):.7g} fd={float(fd(hsw_f, x)):.7g} err={abs(float(hsw_b(x)) - float(fd(hsw_f, x))):.2e}")
Xv = [-4.0, -1.0, 0.7, 3.5]
a, n = fdvec(hsw_f, hsw_b, Xv, up); show("vec analytic " + str(Xv), a); show("vec fd", n); print("  vec max err", max(abs(float(p) - float(q)) for p, q in zip(a, n)))
for x in [1e30, -1e30]:
    print(f"  x={x:g}: f={float(hsw_f(x))!r} f'={float(hsw_b(x))!r}")
print("  kink one-sided limits: f'(-3-)=0 f'(-3+)=", (2*-3+3)/6, " f'(3-)=", (2*3+3)/6, " f'(3+)=1")
g = np.linspace(-8, 8, 160001)
d = [abs((x * min(max(x + 3, 0), 6) / 6) - silu64(x)) for x in g]
i = int(np.argmax(d)); print(f"  max |HardSwish - SiLU| on [-8,8] = {d[i]:.6f} at x = {g[i]:.4f}")
d2 = [abs(min(max(x + 3, 0), 6) / 6 - 1 / (1 + math.exp(-x))) for x in g]
i = int(np.argmax(d2)); print(f"  max |HardSigmoid - sigmoid| on [-8,8] = {d2[i]:.6f} at x = {g[i]:.4f}")
