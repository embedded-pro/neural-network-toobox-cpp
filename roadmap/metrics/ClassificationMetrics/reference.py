import numpy as np
from fractions import Fraction as F
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)

f32 = np.float32
K = 4
logits = np.array([
    [2.0, 0.1, -1.0, 0.0],
    [0.5, 1.5, 0.2, -0.3],
    [-1.0, 0.3, 2.2, 0.1],
    [1.2, 1.1, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.9],
    [3.0, -2.0, 0.5, 1.0],
    [-0.5, 2.5, 2.4, 0.0],
    [0.1, 0.2, 0.3, -5.0],
    [0.7, 0.7, 0.1, 0.2],
    [-2.0, 1.0, 0.0, 0.5],
], dtype=f32)
actual = np.array([0, 1, 2, 1, 2, 0, 2, 2, 1, 1])


def argmax_first(v):
    best = 0
    for i in range(1, len(v)):
        if v[i] > v[best]:
            best = i
    return best


pred = np.array([argmax_first(r) for r in logits])
assert (pred == np.argmax(logits, axis=1)).all()
print("predicted", pred.tolist())

C = np.zeros((K, K), dtype=np.uint32)
for a, p in zip(actual, pred):
    C[a, p] += 1
print("confusion (rows actual, cols predicted)\n", C)
assert (C == confusion_matrix(actual, pred, labels=range(K))).all()

total = int(C.sum())
correct = int(np.trace(C))
acc = F(correct, total)
print("total", total, "correct", correct, "accuracy", acc, float(acc), float(f32(correct) / f32(total)))
assert abs(float(acc) - accuracy_score(actual, pred)) < 1e-12

rows = C.sum(axis=1).astype(int)
cols = C.sum(axis=0).astype(int)
prec, rec, f1 = [], [], []
for k in range(K):
    tp = int(C[k, k])
    prec.append(F(tp, cols[k]) if cols[k] else None)
    rec.append(F(tp, rows[k]) if rows[k] else None)
    f1.append(F(2 * tp, rows[k] + cols[k]) if rows[k] + cols[k] else None)
    print(f"class {k}: support {rows[k]} predicted {cols[k]} P {prec[k]} R {rec[k]} F1 {f1[k]}",
          [None if x is None else round(float(x), 7) for x in (prec[k], rec[k], f1[k])])

sk_p = precision_score(actual, pred, labels=range(K), average=None, zero_division=np.nan)
sk_r = recall_score(actual, pred, labels=range(K), average=None, zero_division=np.nan)
sk_f = f1_score(actual, pred, labels=range(K), average=None, zero_division=np.nan)
for k in range(K):
    for mine, sk in ((prec[k], sk_p[k]), (rec[k], sk_r[k]), (f1[k], sk_f[k])):
        assert (mine is None and np.isnan(sk)) or abs(float(mine) - sk) < 1e-12, (k, mine, sk)
for k in range(K):
    if prec[k] is not None and rec[k] is not None and prec[k] + rec[k] > 0:
        assert f1[k] == 2 * prec[k] * rec[k] / (prec[k] + rec[k])

defined_r = [r for r in rec if r is not None]
bal = sum(defined_r) / len(defined_r)
defined_f = [x for x in f1 if x is not None]
macro = sum(defined_f) / len(defined_f)
print("balanced accuracy", bal, float(bal), "sklearn", balanced_accuracy_score(actual, pred))
print("macro F1", macro, float(macro), "sklearn (labels = union)", f1_score(actual, pred, average="macro"))
assert abs(float(bal) - balanced_accuracy_score(actual, pred)) < 1e-12
assert abs(float(macro) - f1_score(actual, pred, average="macro")) < 1e-12
macroP = sum(p for p in prec if p is not None) / len([p for p in prec if p is not None])
macroR = bal
sokolova = 2 * macroP * macroR / (macroP + macroR)
print("Sokolova F of macro P/R (NOT provided)", sokolova, float(sokolova))


def softmax32(z):
    z = z.astype(f32)
    e = np.exp(z - z.max()).astype(f32)
    return (e / e.sum(dtype=f32)).astype(f32)


for r in logits:
    assert argmax_first(softmax32(r)) == argmax_first(r)
print("fixture: argmax(softmax32(z)) == argmax(z) for all 10 rows")

near = np.array([0.0, 1e-8], dtype=f32)
s = softmax32(near)
print("near tie z = (0, 1e-8):", near, "softmax32", s, "equal", s[0] == s[1],
      "argmax z", argmax_first(near), "argmax softmax", argmax_first(s))
sat = np.array([0.0, 120.0, 121.0], dtype=f32)
print("saturation z = (0,120,121): softmax32", softmax32(sat), "argmax", argmax_first(softmax32(sat)))

print("float32(2**24) + 1 == 2**24:", f32(2 ** 24) + f32(1) == f32(2 ** 24))
print("uint32 max", 2 ** 32 - 1, "years at 100 Hz", (2 ** 32 - 1) / 100 / 3600 / 24 / 365.25)

for v in ([-3, -1, -2, -4], [2, 0.1, -1, 0], [0, 0, 0, 0.9], [-0.5, 2.5, 2.4, 0], [0.7, 0.7, 0.1, 0.2],
          [0.2, 0.9, 0.9, 0.9], [-np.inf, -5, -np.inf, -7]):
    print("argmax", v, argmax_first(np.array(v, dtype=f32)))

for score, thr in ((0.5, 0.5), (np.nextafter(f32(0.5), f32(1)), 0.5), (0.49, 0.5), (0.0, 0.0), (1e-30, 0.0), (-1e-30, 0.0)):
    print("binary", repr(f32(score)), thr, int(f32(score) > f32(thr)))

sig = lambda z: f32(1) / (f32(1) + np.exp(-f32(z)))
z = f32(1e-8)
print("sigmoid32(1e-8) =", repr(sig(z)), "> 0.5 ?", sig(z) > f32(0.5), "; logit 1e-8 > 0 ?", z > 0)

for K_ in (2, 10, 12, 35):
    print(f"K={K_}: ConfusionMatrix RAM {K_ * K_ + 2} words = {4 * (K_ * K_ + 2)} B")
