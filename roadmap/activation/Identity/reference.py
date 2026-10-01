import struct
def f32(x): return struct.unpack('f', struct.pack('f', float(x)))[0]
def add(a, b): return f32(f32(a) + f32(b))
def sub(a, b): return f32(f32(a) - f32(b))
def mul(a, b): return f32(f32(a) * f32(b))
def div(a, b): return f32(f32(a) / f32(b))
h = f32(1e-3)
tol = 1e-3
two_h = mul(2.0, h)
fwd = lambda x: f32(x)
bwd = lambda x: 1.0

print("Forward:")
for x in [-2.5, 0.0, 3.75, 1e30]:
    print(f"  f({x}) = {fwd(x)!r} bit-exact={fwd(x) == f32(x)}")
print("Backward:")
for x in [-1e30, 0.0, 7.5]:
    print(f"  f'({x}) = {bwd(x)}")
print("Scalar central difference (float32, h=1e-3):")
for x in [-2.0, 0.0, 0.7]:
    fd = div(sub(fwd(add(x, h)), fwd(sub(x, h))), two_h)
    print(f"  x={x}: fd={fd!r} |fd-1|={abs(fd-1):.3e} ok={abs(fd-1)<tol}")
x = 1e30
fd = div(sub(fwd(add(x, h)), fwd(sub(x, h))), two_h)
print(f"  x=1e30: fd={fd!r} (FD collapses -> do not FD-check at huge |x|)")

print("ForwardVector:", [fwd(v) for v in [-1.5, 0.0, 0.25, 4.0]])

print("BackwardVector JVP + FD (helper ProjectedOutput, float32 accumulation):")
xs = [f32(v) for v in [-2.0, -0.5, 0.7, 1.5]]
g = [f32(v) for v in [0.3, -1.2, 0.8, -0.4]]
analytic = [mul(gi, 1.0) for gi in g]
def proj(xx):
    s = 0.0
    for i in range(4): s = add(s, mul(g[i], fwd(xx[i])))
    return s
fdv = []
for i in range(4):
    p = list(xs); m = list(xs); p[i] = add(p[i], h); m[i] = sub(m[i], h)
    fdv.append(div(sub(proj(p), proj(m)), two_h))
diffs = [abs(a - b) for a, b in zip(analytic, fdv)]
print("  analytic", analytic)
print("  fd      ", fdv)
print("  max|diff|", max(diffs), "ok", max(diffs) < tol)

print("Dense<float,3,2> + Identity (float64 exact arithmetic):")
W = [[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]]
b = [0.2, -0.1]
xin = [1.0, 2.0, 3.0]
y = [sum(W[i][j]*xin[j] for j in range(3)) + b[i] for i in range(2)]
print("  output = Wx+b", [round(v, 9) for v in y])
y0 = [sum(W[i][j]*xin[j] for j in range(3)) for i in range(2)]
print("  output zero bias", [round(v, 9) for v in y0])
up = [0.8, -1.3]
gin = [sum(W[i][j]*up[i] for i in range(2)) for j in range(3)]
print("  input gradient W^T g", [round(v, 9) for v in gin])
fdin = []
hh = 1e-3
for j in range(3):
    p = list(xin); m = list(xin); p[j] += hh; m[j] -= hh
    yp = [sum(W[i][k]*p[k] for k in range(3)) + b[i] for i in range(2)]
    ym = [sum(W[i][k]*m[k] for k in range(3)) + b[i] for i in range(2)]
    fdin.append((sum(up[i]*yp[i] for i in range(2)) - sum(up[i]*ym[i] for i in range(2))) / (2*hh))
print("  FD input gradient", [round(v, 9) for v in fdin])
print("  dW = g x^T", [[round(up[i]*xin[j], 9) for j in range(3)] for i in range(2)], "db = g", up)
