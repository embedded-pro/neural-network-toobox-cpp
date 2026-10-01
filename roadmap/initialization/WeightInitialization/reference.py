import struct, math

M = 0xFFFFFFFF

def f(x):
    return struct.unpack('f', struct.pack('f', x))[0]

def fmul(a, b): return f(f(a) * f(b))
def fadd(a, b): return f(f(a) + f(b))
def fsub(a, b): return f(f(a) - f(b))
def fdiv(a, b): return f(f(a) / f(b))
def fsqrt(a): return f(math.sqrt(f(a)))
def flog(a): return f(math.log(f(a)))
def fcos(a): return f(math.cos(f(a)))

def fmix32(h):
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & M
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & M
    h ^= h >> 16
    return h

DEFAULT = 2463534242

def xorshift_step(y):
    y ^= (y << 13) & M
    y ^= y >> 17
    y ^= (y << 5) & M
    return y

class Xorshift32:
    def __init__(self, seed):
        s = fmix32(seed)
        self.state = s if s != 0 else DEFAULT
    def next(self):
        self.state = xorshift_step(self.state)
        return self.state
    def uniform(self):
        return unit(self.next())
    def normal(self):
        u1 = positive_unit(self.next())
        u2 = unit(self.next())
        r = fsqrt(fmul(-2.0, flog(u1)))
        return fmul(r, fcos(fmul(TWO_PI, u2)))

INV24 = f(1.0 / 16777216.0)
TWO_PI = f(2.0 * math.pi)

def unit(bits):
    return fmul(f(bits >> 8), INV24)

def positive_unit(bits):
    return fmul(f((bits >> 8) + 1), INV24)

def std(scale, mode, fan_in, fan_out):
    fan = {'in': f(fan_in), 'out': f(fan_out), 'avg': fdiv(f(fan_in + fan_out), 2.0)}[mode]
    return fsqrt(fdiv(f(scale), fan))

def limit(scale, mode, fan_in, fan_out):
    fan = {'in': f(fan_in), 'out': f(fan_out), 'avg': fdiv(f(fan_in + fan_out), 2.0)}[mode]
    return fsqrt(fdiv(fmul(3.0, f(scale)), fan))

def fill(gen, dist, scale, mode, fan_in, fan_out, n):
    out = []
    if dist == 'uniform':
        a = limit(scale, mode, fan_in, fan_out)
        for _ in range(n):
            u = gen.uniform()
            out.append(fmul(a, fsub(fmul(2.0, u), 1.0)))
    else:
        s = std(scale, mode, fan_in, fan_out)
        for _ in range(n):
            out.append(fmul(s, gen.normal()))
    return out

print("== sanity: Marsaglia recurrence from raw state 1 (well-known 270369)")
y = 1
raw = []
for _ in range(3):
    y = xorshift_step(y); raw.append(y)
print(raw)
print("fmix32(1) =", fmix32(1), hex(fmix32(1)), " fmix32(0) =", fmix32(0))
print("linearity of raw xorshift: step(2) == 2*step(1)?", xorshift_step(2), 2 * xorshift_step(1))

print("== T1 Next() stream, seed 1 and seed 2 (seed scrambled by fmix32)")
for seed in (1, 2):
    g = Xorshift32(seed)
    print(seed, "state0", g.state, [g.next() for _ in range(3)])

print("== T2 seed 0 -> Marsaglia default state")
g = Xorshift32(0)
print("state0", g.state, [g.next() for _ in range(3)])
g2 = Xorshift32(0)
print("first unit from seed 0:", g2.uniform())

print("== T3 unit conversions")
for b in (0, 0xFF, 0x100, 0x80000000, 0xFFFFFFFF):
    print(hex(b), "unit", repr(unit(b)), "positive", repr(positive_unit(b)))
print("1 - 2^-24 =", repr(f(1 - 2**-24)), " 2^-24 =", repr(f(2**-24)))
print("max |z| bound sqrt(-2 ln 2^-24) =", repr(fsqrt(fmul(-2.0, flog(f(2**-24))))), math.sqrt(48 * math.log(2)))

print("== T4 uniform moments seed 42, 65536 draws, float accumulation")
N = 65536
g = Xorshift32(42)
s1 = f(0.0); s2 = f(0.0)
mn = 1.0; mx = 0.0
dsum = 0.0; dsum2 = 0.0
for _ in range(N):
    u = g.uniform()
    s1 = fadd(s1, u); s2 = fadd(s2, fmul(u, u))
    dsum += u; dsum2 += u * u
    mn = min(mn, u); mx = max(mx, u)
mean = fdiv(s1, float(N)); m2 = fdiv(s2, float(N)); var = fsub(m2, fmul(mean, mean))
print("float: mean", repr(mean), "var", repr(var), " double: mean", dsum / N, "var", dsum2 / N - (dsum / N) ** 2)
print("min", mn, "max", mx, "expected mean 0.5 var", 1 / 12, " se(mean)", math.sqrt(1 / 12 / N), " se(var)", math.sqrt((1 / 80 - 1 / 144) / N))

print("== T5 normal first draws seed 1")
g = Xorshift32(1)
bits = []
gg = Xorshift32(1)
for _ in range(4):
    bits.append(gg.next())
print("bits", bits)
u1 = positive_unit(bits[0]); u2 = unit(bits[1])
print("u1", repr(u1), "u2", repr(u2), "double z0", math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2))
print("z", [repr(g.normal()) for _ in range(2)])

print("== T6 normal moments seed 42, 65536 draws")
g = Xorshift32(42)
s1 = f(0.0); s2 = f(0.0); s4 = f(0.0); mxa = 0.0
d1 = d2 = d4 = 0.0
for _ in range(N):
    z = g.normal()
    z2 = fmul(z, z)
    s1 = fadd(s1, z); s2 = fadd(s2, z2); s4 = fadd(s4, fmul(z2, z2))
    d1 += z; d2 += z * z; d4 += z ** 4
    mxa = max(mxa, abs(z))
print("float: mean", repr(fdiv(s1, float(N))), "E[z^2]", repr(fdiv(s2, float(N))), "E[z^4]", repr(fdiv(s4, float(N))))
print("double: mean", d1 / N, "E[z^2]", d2 / N, "E[z^4]", d4 / N, "max|z|", mxa)
print("se(mean)", 1 / math.sqrt(N), "se(E z^2)", math.sqrt(2 / N), "se(E z^4)", math.sqrt(96 / N))

print("== T7 scale table fanIn=4 fanOut=2")
rows = [("GlorotUniform", 1.0, 'avg'), ("GlorotNormal", 1.0, 'avg'), ("HeUniform", 2.0, 'in'), ("HeNormal", 2.0, 'in'),
        ("LeCunUniform", 1.0, 'in'), ("LeCunNormal", 1.0, 'in')]
for name, sc, mode in rows:
    print(name, "std", repr(std(sc, mode, 4, 2)), "limit", repr(limit(sc, mode, 4, 2)), " exact std", math.sqrt(sc / {'avg': 3, 'in': 4}[mode]), "exact limit", math.sqrt(3 * sc / {'avg': 3, 'in': 4}[mode]))
a = f(0.1)
he_leaky = fdiv(2.0, fadd(1.0, fmul(a, a)))
print("HeNormal(0.1) scale", repr(he_leaky), "std", repr(std(he_leaky, 'in', 4, 2)), "exact", math.sqrt(2 / 1.01 / 4))
print("GlorotUniform fanIn=4 fanOut=12 (fan_out matters): std", repr(std(1.0, 'avg', 4, 12)), "limit", repr(limit(1.0, 'avg', 4, 12)), "exact", math.sqrt(2 / 16), math.sqrt(6 / 16))
print("LeCun fanOut mode sanity (FanOut=2): std", repr(std(1.0, 'out', 4, 2)))

print("== T8 Weights<2,4>() seed 1: HeNormal and GlorotUniform (row-major)")
w = fill(Xorshift32(1), 'normal', 2.0, 'in', 4, 2, 8)
print("HeNormal", [repr(x) for x in w])
w = fill(Xorshift32(1), 'uniform', 1.0, 'avg', 4, 2, 8)
print("GlorotUniform", [repr(x) for x in w])
g = Xorshift32(1)
print("  underlying 2u-1", [repr(fsub(fmul(2.0, g.uniform()), 1.0)) for _ in range(8)])

print("== T9 stream continuation seed 7 GlorotUniform fanIn 4 fanOut 2")
g = Xorshift32(7)
a8 = fill(g, 'uniform', 1.0, 'avg', 4, 2, 8)
a2 = fill(g, 'uniform', 1.0, 'avg', 4, 2, 2)
b10 = fill(Xorshift32(7), 'uniform', 1.0, 'avg', 4, 2, 10)
print("A first 8", [repr(x) for x in a8])
print("A next 2", [repr(x) for x in a2], " B[8:10]", [repr(x) for x in b10[8:]], "equal", a2 == b10[8:])
print("fresh-seed-7 first 2 (must differ from A next 2)", [repr(x) for x in b10[:2]])

print("== variance-propagation sanity (double, analytic)")
print("He: Var[z_l] = n * Var[w] * E[x^2]; ReLU E[x^2] = Var[z_{l-1}]/2 -> n*Var[w]/2 = 1 -> Var[w] = 2/n")
print("Glorot: forward n_in Var[w] = 1, backward n_out Var[w] = 1 -> Var[w] = 2/(n_in+n_out); uniform U(-a,a) var a^2/3 -> a = sqrt(6/(n_in+n_out))")
print("uniform var check a=1:", 1 / 3, " vs 2/(4+2):", 2 / 6)
