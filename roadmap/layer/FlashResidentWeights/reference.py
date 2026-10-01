import numpy as np
f = np.float32

def dense(theta, x, n_in, n_out, act):
    theta = np.asarray(theta, dtype=f); x = np.asarray(x, dtype=f)
    z = np.zeros(n_out, dtype=f)
    for i in range(n_out):
        s = theta[n_in * n_out + i]
        for j in range(n_in):
            s = f(s + f(theta[i * n_in + j] * x[j]))
        z[i] = s
    return z, act(z)

tanh = lambda z: np.tanh(z).astype(f)
leaky = lambda z: np.where(z > 0, z, f(0.1) * z).astype(f)
def softmax(z):
    e = np.exp((z - z.max()).astype(f)).astype(f)
    return (e / e.sum(dtype=f)).astype(f)

theta = [0.1, -0.2, 0.3, 0.4, 0.5, -0.6, 0.2, -0.1]
x1 = [0.3, -0.7, 1.1]; x2 = [1.0, 2.0, 3.0]
for name, x, act in [("tanh x1", x1, tanh), ("softmax x2", x2, softmax), ("leaky x2", x2, leaky), ("tanh x2", x2, tanh)]:
    z, a = dense(theta, x, 3, 2, act)
    print(f"{name}: z={z.tolist()} a={[float(np.format_float_positional(v, 7, unique=False, trim='-')) for v in a]} sum={a.sum(dtype=f)}")

hidden = [0.5, -1.0, 1.5, 0.25, -0.5, 0.75, 0.0, 0.0, 0.0]
out = [1.0, -0.5, 2.0, 0.0]
inp = [2.0, 0.5]
zh, ah = dense(hidden, inp, 2, 3, leaky)
zo, ao = dense(out, ah, 3, 1, tanh)
print("model hidden z", zh.tolist(), "a", ah.tolist(), "out z", zo.tolist(), "y", repr(ao[0]))
zo2, ao2 = dense([0.5, 0.5, 0.5, 0.1], ah, 3, 1, tanh)
print("mixed after SetParameters(0.5,0.5,0.5,0.1): z", zo2.tolist(), "y", repr(ao2[0]))

P = 64 * 64 + 64
dense_floats = 2 * P + 2 * 64 + 2 * 64
print("P(64,64)", P, "bytes", 4 * P, "Dense floats", dense_floats, "bytes", 4 * dense_floats, "FlashDense RAM floats", 64, "bytes", 256)
print("Cortex-M FlashDense bytes = 256 + 3*4 =", 256 + 12, " ratio host", 34328 / 280)

# float64 cross-check of closed forms
print("float64 tanh x1", np.tanh([0.7, -0.99]), "softmax(0.8,-0.5)", 1 / (1 + np.exp(-1.3)), 1 / (1 + np.exp(1.3)))
print("float64 model y", np.tanh(1.0 * 0.5 - 0.5 * 3.125 + 2.0 * -0.0625), "mixed y", np.tanh(0.5 * (0.5 + 3.125 - 0.0625) + 0.1))
