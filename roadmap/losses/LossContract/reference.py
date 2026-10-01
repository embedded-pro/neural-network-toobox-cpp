import numpy as np
f=np.float32
h=f(1e-3)
np.set_printoptions(precision=8, suppress=False)

def fsum(v):
    s=f(0)
    for x in v: s=f(s+f(x))
    return s

# ---------- BCE with logits (float32 mirror of the C++ loop) ----------
def bcel_cost32(z,y):
    N=len(z); inv=f(1.0/N); s=f(0)
    for zi,yi in zip(z.astype(f),y.astype(f)):
        e=f(np.exp(-abs(zi)))
        l=f(f(max(zi,f(0))-f(zi*yi))+f(np.log(f(f(1)+e))))
        s=f(s+l)
    return f(s*inv)
def sig32(zi):
    e=f(np.exp(-abs(f(zi))))
    return f(f(1)/f(1+e)) if zi>=0 else f(e/f(1+e))
def bcel_grad32(z,y):
    N=len(z); inv=f(1.0/N)
    return np.array([f(f(sig32(zi)-yi)*inv) for zi,yi in zip(z.astype(f),y.astype(f))],f)
# float64 textbook reference: -[y log s + (1-y) log(1-s)]
def bcel_cost64(z,y):
    z=z.astype(np.float64); y=y.astype(np.float64)
    return np.mean(np.logaddexp(0,z) - y*z)   # softplus(z) - y z
def bcel_textbook64(z,y):
    z=z.astype(np.float64); y=y.astype(np.float64); s=1/(1+np.exp(-z))
    return np.mean(-(y*np.log(s)+(1-y)*np.log(1-s)))
def bcel_grad64(z,y):
    z=z.astype(np.float64); return (1/(1+np.exp(-z))-y)/len(z)
def fd(cost,p,t):
    g=np.zeros(len(p),f)
    for i in range(len(p)):
        a=p.astype(f).copy();b=p.astype(f).copy();a[i]=f(a[i]+h);b[i]=f(b[i]-h)
        g[i]=f(f(cost(a,t)-cost(b,t))/f(2*h))
    return g

# probability BCE (existing, eps clamp)
EPS=f(1e-7)
def bce_prob_cost32(p,y):
    N=len(p); s=f(0)
    for pi,yi in zip(p.astype(f),y.astype(f)):
        q=f(min(max(pi,EPS),f(1)-EPS))
        s=f(s-f(f(yi*f(np.log(q)))+f(f(1-yi)*f(np.log(f(1-q))))))
    return f(s*f(1.0/N))
def bce_prob_grad32(p,y):
    N=len(p); out=[]
    for pi,yi in zip(p.astype(f),y.astype(f)):
        q=f(min(max(pi,EPS),f(1)-EPS)); out.append(f(f(f(q-yi)/f(q*f(1-q)))*f(1.0/N)))
    return np.array(out,f)

print("=== BCE with logits, fixture")
z=np.array([2.0,-1.0,0.5,-3.0],f); y=np.array([1.0,0.0,0.25,1.0],f)
print("per-element l (f64):", np.logaddexp(0,z.astype(np.float64))-y*z)
print("cost f32", repr(bcel_cost32(z,y)), "softplus f64", bcel_cost64(z,y), "textbook f64", bcel_textbook64(z,y))
print("sigma(z) f64", 1/(1+np.exp(-z.astype(np.float64))))
print("grad f32", bcel_grad32(z,y), "f64", bcel_grad64(z,y))
g=fd(bcel_cost32,z,y); print("FD", g, "maxerr", np.max(np.abs(g-bcel_grad64(z,y))))
s=np.array([sig32(zi) for zi in z],f)
print("prob-BCE(sigmoid z) cost", bce_prob_cost32(s,y), "grad*sigma'", bce_prob_grad32(s,y)*s*(1-s))
z0=np.zeros(4,f)
print("at z=0: cost", bcel_cost32(z0,y), "ln2", np.log(2), "grad", bcel_grad32(z0,y), "f64", bcel_grad64(z0,y))
g0=fd(bcel_cost32,z0,y); print("FD z=0", g0, "maxerr", np.max(np.abs(g0-bcel_grad64(z0,y))))

print("=== BCE with logits, extreme")
ze=np.array([100.0,-100.0,1e30,-1e30],f); ye=np.array([1.0,0.0,0.0,1.0],f)
c=bcel_cost32(ze,ye); print("cost f32", repr(c), "== 5e29f?", c==f(5e29), "f64", bcel_cost64(ze,ye))
print("grad f32", bcel_grad32(ze,ye), "f64", bcel_grad64(ze,ye))
se=np.array([sig32(zi) for zi in ze],f)
with np.errstate(all='ignore'):
    print("prob-BCE on sigmoid(z) (clamped) cost", bce_prob_cost32(se,ye), "grad", bce_prob_grad32(se,ye))
print("z=20,y=0: logits", bcel_cost32(np.array([20.0],f),np.array([0.0],f)), "prob", bce_prob_cost32(np.array([sig32(20.0)],f),np.array([0.0],f)))
print("1+exp(-17) == 1 in f32:", f(1)+f(np.exp(f(-17)))==f(1), "exp(-17)=",np.exp(-17.0), "2^-24=",2.0**-24, "threshold |z| >", -np.log(2.0**-24))

# ---------- CCE (logits) ----------
def cce_dense32(z,y):
    z=z.astype(f); m=max(z); ex=np.array([f(np.exp(f(zi-m))) for zi in z],f); L=f(np.log(fsum(ex)))
    return fsum([f(yi*f(L-f(zi-m))) for zi,yi in zip(z,y.astype(f))])
def cce_dense_grad32(z,y):
    z=z.astype(f); m=max(z); ex=np.array([f(np.exp(f(zi-m))) for zi in z],f); s=fsum(ex); S=fsum(y)
    inv=f(f(1)/s)
    return np.array([f(f(f(ei*inv)*S)-yi) for ei,yi in zip(ex,y.astype(f))],f)
def cce_idx32(z,k):
    z=z.astype(f); m=max(z); ex=np.array([f(np.exp(f(zi-m))) for zi in z],f); L=f(np.log(fsum(ex)))
    return f(L-f(z[k]-m))
def cce_idx_grad32(z,k):
    z=z.astype(f); m=max(z); ex=np.array([f(np.exp(f(zi-m))) for zi in z],f); inv=f(f(1)/fsum(ex))
    return np.array([f(f(ei*inv)-(f(1) if i==k else f(0))) for i,ei in enumerate(ex)],f)
def cce_idx64(z,k):
    z=z.astype(np.float64); m=z.max(); return m+np.log(np.sum(np.exp(z-m)))-z[k]
def cce_idx_grad64(z,k):
    z=z.astype(np.float64); m=z.max(); e=np.exp(z-m); g=e/e.sum(); g[k]-=1; return g

print("=== CCE index target")
zl=np.array([1.0,2.0,3.0],f)
print("k=1 cost f32", cce_idx32(zl,1), "f64", cce_idx64(zl,1), "dense one-hot", cce_dense32(zl,np.array([0,1,0],f)))
print("k=1 grad f32", cce_idx_grad32(zl,1), "f64", cce_idx_grad64(zl,1), "dense", cce_dense_grad32(zl,np.array([0,1,0],f)))
zp=np.array([0.4,-1.2,2.5],f)
gi=fd(lambda a,k: cce_idx32(a,k), zp, 0); print("FD k=0 at",zp, gi, "analytic", cce_idx_grad64(zp,0), "maxerr", np.max(np.abs(gi-cce_idx_grad64(zp,0))), "cost", cce_idx32(zp,0), cce_idx64(zp,0))
zh=np.array([1e30,0.0,-1e30],f)
for k in (0,2):
    print("huge k=%d cost"%k, repr(cce_idx32(zh,k)), "grad", cce_idx_grad32(zh,k))
print("2e30f", repr(f(2e30)), cce_idx32(zh,2)==f(2e30))
# naive (no shift) overflow reference
with np.errstate(all='ignore'):
    print("naive exp(1e30) f32:", np.exp(f(1e30)))

print("=== MSE contract (runtime target)")
pred=np.array([1.0,-1.5,2.5,1.0],f); tA=np.array([0.5,-1.0,2.0,0.0],f)
e=(pred-tA).astype(np.float64)
print("MSE cost A", np.mean(e**2), "grad A", 2*e/4, "cost B(=pred)", 0.0)

print("=== migrated existing tests (regulariser removed)")
print("MSE cost", np.mean(e**2), "grad", 2*e/4)
print("MAE cost", np.mean(np.abs(e)), "grad", np.sign(e)/4)
yb=np.array([1,0,1,0],f); pb=np.array([0.9,0.2,0.6,0.3],f)
print("BCE cost", bce_prob_cost32(pb,yb), "grad", bce_prob_grad32(pb,yb))
pb64=pb.astype(np.float64); yb64=yb.astype(np.float64)
print("BCE f64 cost", np.mean(-(yb64*np.log(pb64)+(1-yb64)*np.log(1-pb64))), "grad", (pb64-yb64)/(pb64*(1-pb64))/4)
print("CCE cost", cce_dense32(zl,np.array([0,1,0],f)), "grad", cce_dense_grad32(zl,np.array([0,1,0],f)))
