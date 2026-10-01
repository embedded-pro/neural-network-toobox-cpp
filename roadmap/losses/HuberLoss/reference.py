import numpy as np
f=np.float32
N=4
target=np.array([0.5,-1.0,2.0,0.0],f)
pred=np.array([1.0,-1.5,4.0,-3.0],f)
reg_cost=f(0.1); reg_grad=np.array([0.01,0.02,0.03,0.04],f)
h=f(1e-3)
def huber_cost(p,d):
    d=f(d); e=(p-target).astype(f); c=np.clip(e,-d,d).astype(f)
    rho=(c*(e-f(0.5)*c)).astype(f)
    s=f(0)
    for r in rho: s=f(s+r)
    return f(s*f(1.0/N))
def huber_grad(p,d):
    d=f(d); e=(p-target).astype(f); return (np.clip(e,-d,d)*f(1.0/N)).astype(f)
def piecewise(p,d):  # independent textbook form, float64
    e=p.astype(np.float64)-target.astype(np.float64)
    return np.mean(np.where(np.abs(e)<=d,0.5*e*e,d*(np.abs(e)-0.5*d)))
def fd(p,d):
    g=np.zeros(N,f)
    for i in range(N):
        a=p.copy();b=p.copy();a[i]=f(a[i]+h);b[i]=f(b[i]-h)
        g[i]=f((huber_cost(a,d)-huber_cost(b,d))/(f(2)*h))
    return g
d=1.5
print("e",pred-target)
print("cost d=1.5", huber_cost(pred,d), "f64", piecewise(pred,d), "+reg", f(huber_cost(pred,d)+reg_cost))
print("grad d=1.5", huber_grad(pred,d), "+reg", huber_grad(pred,d)+reg_grad)
fdg=fd(pred,d); print("fd d=1.5", fdg, "maxerr", np.max(np.abs(fdg-huber_grad(pred,d))))
e64=pred.astype(np.float64)-target
mse=np.mean(e64**2); mae=np.mean(np.abs(e64))
print("MSE",mse,"half",mse/2,"MSEgrad",2*e64/N,"half",e64/N)
print("huber d=10",huber_cost(pred,10),piecewise(pred,10),"grad",huber_grad(pred,10))
print("MAE",mae,"MAEgrad",np.sign(e64)/N)
print("huber d=0.25",huber_cost(pred,0.25),piecewise(pred,0.25),"d*MAE-d^2/2",0.25*mae-0.25**2/2,"grad",huber_grad(pred,0.25),"d*MAEgrad",0.25*np.sign(e64)/N)
# FD at d=0.25 and 10 as well
print("fd d=10 maxerr",np.max(np.abs(fd(pred,10)-huber_grad(pred,10))),"fd d=.25 maxerr",np.max(np.abs(fd(pred,0.25)-huber_grad(pred,0.25))))
# outlier
out=target.copy(); out[0]=f(target[0]+f(1e20))
print("outlier pred",out, "e",(out-target))
c=huber_cost(out,d); print("outlier cost",repr(c),"exact f64", piecewise(out,d), "grad",huber_grad(out,d))
with np.errstate(over='ignore'):
    e=(out-target).astype(f); print("MSE float32 on outlier", f(np.sum((e*e).astype(f))*f(0.25)))
# boundary FD: e exactly = delta
bp=target.copy(); bp[0]=f(target[0]+f(1.5))
print("boundary e",bp-target,"cost",huber_cost(bp,d),"grad",huber_grad(bp,d),"fd",fd(bp,d))
print("float max e before overflow ~ FLT_MAX/delta", np.finfo(f).max/1.5, "MSE overflow |e| >", np.sqrt(np.finfo(f).max))
