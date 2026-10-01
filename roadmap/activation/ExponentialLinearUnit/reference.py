import math
import numpy as np

f32 = np.float32
H = f32(1e-3)

SELU_ALPHA = 1.6732632423543772848170429916717
SELU_SCALE = 1.0507009873554804934193349852946


class Elu:
    def __init__(self, alpha=1.0, scale=1.0):
        self.scale = f32(scale)
        self.scaledAlpha = f32(f32(scale) * f32(alpha))

    def forward(self, x):
        x = f32(x)
        if x > f32(0):
            return f32(self.scale * x)
        return f32(self.scaledAlpha * f32(f32(np.exp(x)) - f32(1)))

    def backward(self, x):
        x = f32(x)
        if x > f32(0):
            return self.scale
        return f32(self.scaledAlpha * f32(np.exp(x)))

    def forward_vector(self, xs):
        return [self.forward(x) for x in xs]

    def backward_vector(self, pre, out, up):
        res = []
        for x, y, g in zip(pre, out, up):
            d = self.scale if x > f32(0) else f32(y + self.scaledAlpha)
            res.append(f32(g * d))
        return res


def exact_forward(x, alpha, scale):
    return scale * x if x > 0 else scale * alpha * math.expm1(x)


def exact_backward(x, alpha, scale):
    return scale if x > 0 else scale * alpha * math.exp(x)


def central(act, x):
    x = f32(x)
    return f32(f32(act.forward(f32(x + H)) - act.forward(f32(x - H))) / f32(f32(2) * H))


def projected(act, xs, up):
    out = act.forward_vector(xs)
    s = f32(0)
    for u, o in zip(up, out):
        s = f32(s + f32(u * o))
    return s


def central_gradient(act, xs, up):
    g = []
    for i in range(len(xs)):
        p = list(xs)
        m = list(xs)
        p[i] = f32(p[i] + H)
        m[i] = f32(m[i] - H)
        g.append(f32(f32(projected(act, p, up) - projected(act, m, up)) / f32(f32(2) * H)))
    return g


def analytic_gradient(act, xs, up):
    out = act.forward_vector(xs)
    return act.backward_vector(xs, out, up)


def phi_cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def fmt(v):
    return "{" + ", ".join(f"{float(a):.7g}" for a in v) + "}"


elu = Elu()
selu = Elu(SELU_ALPHA, SELU_SCALE)

print("SELU constants float32: alpha", f"{float(f32(SELU_ALPHA)):.9g}", "scale", f"{float(f32(SELU_SCALE)):.9g}",
      "scaledAlpha", f"{float(selu.scaledAlpha):.9g}", "exact scale*alpha", f"{SELU_SCALE * SELU_ALPHA:.12g}")

print("\n== ELU Forward")
xs = [-3.0, -1.0, -0.5, 0.0, 2.0]
print("x", xs)
print("float32", fmt([elu.forward(x) for x in xs]))
print("exact  ", fmt([exact_forward(x, 1.0, 1.0) for x in xs]))

print("\n== SELU ForwardVector")
xs = [-2.0, -0.5, 0.0, 0.7, 1.5]
print("x", xs)
print("float32", fmt(selu.forward_vector(xs)))
print("exact  ", fmt([exact_forward(x, SELU_ALPHA, SELU_SCALE) for x in xs]))

print("\n== Saturation / large magnitude")
for act, name, a, s in ((elu, "ELU", 1.0, 1.0), (selu, "SELU", SELU_ALPHA, SELU_SCALE)):
    print(name, "F(-100)", f"{float(act.forward(-100.0)):.9g}", "B(-100)", float(act.backward(-100.0)),
          "F(1e30)", f"{float(act.forward(1e30)):.9g}", "B(1e30)", float(act.backward(1e30)))
    y = act.forward(-100.0)
    print(name, "vector-derivative at -100 (y + scaledAlpha)", float(f32(y + act.scaledAlpha)))

print("\n== ELU Backward")
xs = [-2.0, -0.5, 0.0, 1.5]
print("x", xs)
print("float32", fmt([elu.backward(x) for x in xs]))
print("exact  ", fmt([exact_backward(x, 1.0, 1.0) for x in xs]))
fdx = [-2.0, -0.5, 0.7]
fd = [central(elu, x) for x in fdx]
print("ELU FD at", fdx, fmt(fd), "err", fmt([abs(float(a) - exact_backward(x, 1, 1)) for a, x in zip(fd, fdx)]))
print("ELU FD at 0 (C1 point)", float(central(elu, 0.0)), "err", abs(float(central(elu, 0.0)) - 1.0))

print("\n== SELU Backward")
xs = [-2.0, -0.5, 0.0, 0.5]
print("x", xs)
print("float32", fmt([selu.backward(x) for x in xs]))
print("exact  ", fmt([exact_backward(x, SELU_ALPHA, SELU_SCALE) for x in xs]))
fd = [central(selu, x) for x in fdx]
print("SELU FD at", fdx, fmt(fd), "err", fmt([abs(float(a) - exact_backward(x, SELU_ALPHA, SELU_SCALE)) for a, x in zip(fd, fdx)]))
print("SELU FD at 0 (kink, NOT a test point)", float(central(selu, 0.0)), "one-sided", SELU_SCALE * SELU_ALPHA, SELU_SCALE)

print("\n== BackwardVector vs FD")
inp = [f32(v) for v in (-2.0, -0.5, 0.7, 1.5)]
up = [f32(v) for v in (0.3, -1.2, 0.8, -0.4)]
for act, name, a, s in ((elu, "ELU", 1.0, 1.0), (selu, "SELU", SELU_ALPHA, SELU_SCALE)):
    an = analytic_gradient(act, inp, up)
    nu = central_gradient(act, inp, up)
    ex = [float(u) * exact_backward(float(x), a, s) for x, u in zip(inp, up)]
    print(name, "analytic", fmt(an))
    print(name, "exact   ", fmt(ex))
    print(name, "FD      ", fmt(nu))
    print(name, "max |analytic-FD|", max(abs(float(p) - float(q)) for p, q in zip(an, nu)),
          "max |analytic-exact|", max(abs(float(p) - q) for p, q in zip(an, ex)))

print("\n== BackwardVector reuse vs recompute at very negative x (absolute error)")
for x in (-10.0, -20.0):
    y = selu.forward(x)
    reuse = float(f32(y + selu.scaledAlpha))
    print("SELU x", x, "reuse", reuse, "exact", exact_backward(x, SELU_ALPHA, SELU_SCALE),
          "abs err", abs(reuse - exact_backward(x, SELU_ALPHA, SELU_SCALE)))

print("\n== SELU fixed point, closed form (double)")
mean = SELU_SCALE * (1.0 / math.sqrt(2 * math.pi) + SELU_ALPHA * (math.exp(0.5) * phi_cdf(-1.0) - 0.5))
second = SELU_SCALE ** 2 * (0.5 + SELU_ALPHA ** 2 * (math.exp(2.0) * phi_cdf(-2.0) - 2 * math.exp(0.5) * phi_cdf(-1.0) + 0.5))
print("E[selu(z)] =", mean, " E[selu(z)^2] =", second)

print("\n== SELU fixed point, float32 midpoint rule on [-10, 10], 2000 cells (as the test computes it)")
cells = 2000
lo = f32(-10.0)
step = f32(20.0 / cells)
invSqrt2Pi = f32(1.0 / math.sqrt(2 * math.pi))
m1 = f32(0)
m2 = f32(0)
for k in range(cells):
    z = f32(lo + f32(f32(f32(k) + f32(0.5)) * step))
    w = f32(f32(invSqrt2Pi * f32(np.exp(f32(f32(-0.5) * f32(z * z))))) * step)
    y = selu.forward(z)
    m1 = f32(m1 + f32(w * y))
    m2 = f32(m2 + f32(w * f32(y * y)))
print("mean", float(m1), "second moment", float(m2), "variance", float(m2) - float(m1) ** 2)

print("\n== Same quadrature with ELU (shows the property is specific to SELU)")
m1 = f32(0)
m2 = f32(0)
for k in range(cells):
    z = f32(lo + f32(f32(f32(k) + f32(0.5)) * step))
    w = f32(f32(invSqrt2Pi * f32(np.exp(f32(f32(-0.5) * f32(z * z))))) * step)
    y = elu.forward(z)
    m1 = f32(m1 + f32(w * y))
    m2 = f32(m2 + f32(w * f32(y * y)))
print("ELU mean", float(m1), "second moment", float(m2))
