import numpy as np
f=np.float32
H=f(1e-3)

def out_len(L,K,S): return (L-K)//S+1

def maxpool(x,L,C,K,S):
    Lo=out_len(L,K,S); y=np.zeros(Lo*C,f); am=np.zeros(Lo*C,int)
    for o in range(Lo):
        for c in range(C):
            b=o*S*C+c; best=x[b]; bi=b
            for k in range(1,K):
                i=(o*S+k)*C+c
                if x[i]>best: best=x[i]; bi=i
            y[o*C+c]=best; am[o*C+c]=bi
    return y,am
def maxpool_back(g,am,L,C):
    d=np.zeros(L*C,f)
    for j,i in enumerate(am): d[i]=f(d[i]+g[j])
    return d
def avgpool(x,L,C,K,S):
    Lo=out_len(L,K,S); y=np.zeros(Lo*C,f); inv=f(1)/f(K)
    for o in range(Lo):
        for c in range(C):
            s=f(0)
            for k in range(K): s=f(s+x[(o*S+k)*C+c])
            y[o*C+c]=f(s*inv)
    return y
def avgpool_back(g,L,C,K,S):
    Lo=out_len(L,K,S); d=np.zeros(L*C,f); inv=f(1)/f(K)
    for o in range(Lo):
        for c in range(C):
            gi=f(g[o*C+c]*inv)
            for k in range(K): i=(o*S+k)*C+c; d[i]=f(d[i]+gi)
    return d
def gap(x,L,C): return avgpool(x,L,C,L,1)
def gap_back(g,L,C): return avgpool_back(g,L,C,L,1)

def fd(fun,x,g):
    n=np.zeros(len(x),f)
    for j in range(len(x)):
        p=x.copy(); m=x.copy(); p[j]=f(p[j]+H); m[j]=f(m[j]-H)
        n[j]=f(f(np.dot(g,fun(p)).astype(f)-np.dot(g,fun(m)).astype(f))/f(2*H))
    return n

x=np.array([0.5,-0.8,-1.2,0.9,2.0,0.1,0.3,-2.1,-0.4,1.4,1.7,0.6],f)
L,C=6,2
print("OutputSizes", [out_len(6,2,2)*2, out_len(6,3,2)*2, out_len(7,3,2)*1, out_len(5,5,1)*1])
y,_=maxpool(x,L,C,2,2); print("max K2S2 fwd",y)
y,am=maxpool(x,L,C,3,2); g=np.array([0.8,-1.3,0.4,0.25],f)
print("max K3S2 fwd",y,"argmax",am)
a=maxpool_back(g,am,L,C); n=fd(lambda z: maxpool(z,L,C,3,2)[0],x,g)
print("max K3S2 bwd",a,"\n fd",n,"maxerr",np.max(np.abs(a-n)))
xt=np.array([0.7,0.7,-0.3,-0.3],f); yt,amt=maxpool(xt,4,1,2,2)
print("tie fwd",yt,"bwd",maxpool_back(np.array([1.0,2.0],f),amt,4,1))
print("avg K3S2 fwd",avgpool(x,L,C,3,2))
a=avgpool_back(g,L,C,3,2); n=fd(lambda z: avgpool(z,L,C,3,2),x,g)
print("avg K3S2 bwd",a,"\n fd",n,"maxerr",np.max(np.abs(a-n)))
print("gap fwd",gap(x,L,C))
gg=np.array([0.8,-1.3],f); a=gap_back(gg,L,C); n=fd(lambda z: gap(z,L,C),x,gg)
print("gap bwd",a,"\n fd",n,"maxerr",np.max(np.abs(a-n)))

W=np.array([0.5,-0.25],f); b=f(0.1)
def model(z):
    m,_=maxpool(z,6,2,2,2); p=gap(m,3,2); zz=f(np.dot(W,p)+b); return np.array([np.tanh(zz)],f)
m,am2=maxpool(x,6,2,2,2); p=gap(m,3,2); zz=f(np.dot(W,p)+b); yy=f(np.tanh(zz))
print("model max",m,"gap",p,"z",zz,"y",yy)
dz=f(1-yy*yy); dp=(W*dz).astype(f); dm=gap_back(dp,3,2); dx=maxpool_back(dm,am2,6,2)
n=fd(model,x,np.array([1.0],f))
print("model dz",dz,"dp",dp,"dm",dm,"\n dx",dx,"\n fd",n,"maxerr",np.max(np.abs(dx-n)))
