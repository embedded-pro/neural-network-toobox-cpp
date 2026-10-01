import numpy as np
f32=np.float32
np.set_printoptions(precision=9, suppress=False)
def lit(v):
    v=f32(v)
    if not np.isfinite(v): raise ValueError("non-finite")
    if v!=0 and abs(v)<np.finfo(f32).tiny: v=f32(np.copysign(0.0,v))
    s="%.9g"%v
    if not any(c in s for c in ".en"): s+=".0"
    return s+"f"
print("literals:",[lit(v) for v in [1.0,-0.0,0.1,1e-40,-1e-40,3.4028235e38,1e-5]])

# A: Keras Dense kernel (in,out)
W1=np.array([[0.5,-1.0],[1.5,0.25],[-0.5,0.75]],dtype=f32); b1=np.array([0.1,-0.2,0.05],dtype=f32)
kernel=W1.T.copy()
print("A keras kernel",kernel.tolist(),"theta",np.concatenate([kernel.T.reshape(-1),b1]).tolist())

# B: Dense(linear) -> BN -> LeakyReLU(0.1), fold pre-activation (float64 fold, float32 emit)
gamma=np.array([1.2,0.5,2.0]); beta=np.array([0.1,-0.3,0.0]); mu=np.array([0.4,2.0,-0.5]); var=np.array([0.25,4.0,0.09]); eps=1e-3
a=gamma/np.sqrt(var+eps); b=beta-a*mu
W=W1.astype(np.float64); c=b1.astype(np.float64)
Wf=a[:,None]*W; cf=a*c+b
x=np.array([2.0,0.5])
z=W@x+c; bn=gamma*(z-mu)/np.sqrt(var+eps)+beta; zf=Wf@x+cf
lr=lambda t: np.where(t>0,t,0.1*t)
print("B a",a,"b",b)
print("B folded theta (f32)",[lit(v) for v in np.concatenate([Wf.reshape(-1),cf])])
print("B unfolded bn",bn,"folded z",zf,"maxdiff",np.max(abs(bn-zf)))
print("B after LeakyReLU",lr(bn))
# float32 forward with folded f32 params vs float64 reference
Wf32=Wf.astype(f32); cf32=cf.astype(f32)
zf32=np.array([f32(f32(cf32[i]+f32(Wf32[i,0]*f32(2)))+f32(Wf32[i,1]*f32(0.5))) for i in range(3)],dtype=f32)
print("B folded forward float32",zf32,"abs err vs f64",np.max(abs(zf32-bn)))

# C: Dense1(W1,b1,LeakyReLU) -> BN (post-activation) -> Dense2(W2,c2,tanh): fold into Dense2
W2=np.array([[1.0,-0.5,2.0]]); c2=np.array([-0.3])
h=lr(W@x+c); u=a*h+b; y=np.tanh(W2@u+c2)
W2f=W2*a[None,:]; c2f=c2+W2@b; yf=np.tanh(W2f@h+c2f)
print("C W2'",W2f,"c2'",c2f,"y",y,"yf",yf,"diff",abs(y-yf))
print("C folded theta1",[lit(v) for v in np.concatenate([W2f.reshape(-1),c2f])])

# D: Conv1D K=2, Cin=2, Cout=2; keras kernel[k][c][o]
kk=np.arange(1,9,dtype=f32).reshape(2,2,2)
theta_keras=kk.transpose(2,0,1).reshape(-1)
pt=kk.transpose(2,1,0)   # torch weight[o][c][k] = kernel[k][c][o]
theta_torch=pt.transpose(0,2,1).reshape(-1)
print("D keras kernel[k][c][o]",kk.tolist())
print("D torch weight[o][c][k]",pt.tolist())
print("D theta keras",theta_keras.tolist(),"theta torch",theta_torch.tolist())
# check convolution equivalence: x (L=3, Cin=2) channels-last
xc=np.array([[1.0,-1.0],[0.5,2.0],[-0.5,0.25]])
yk=np.array([[sum(kk[k,cc,o]*xc[t+k,cc] for k in range(2) for cc in range(2)) for o in range(2)] for t in range(2)])
th=theta_keras.reshape(2,2,2)  # [o][k][c]
yt=np.array([[sum(th[o,k,cc]*xc[t+k,cc] for k in range(2) for cc in range(2)) for o in range(2)] for t in range(2)])
print("D conv out keras",yk.tolist(),"theta",yt.tolist())

# E: LSTM torch H=1 X=1; GRU keras H=1 X=1
w_ih=np.array([[1],[2],[3],[4]],dtype=f32); w_hh=np.array([[5],[6],[7],[8]],dtype=f32)
b_ih=np.array([0.1,0.2,0.3,0.4],dtype=f32); b_hh=np.array([1,1,1,1],dtype=f32)
print("E lstm theta",np.concatenate([w_ih.reshape(-1),w_hh.reshape(-1),b_ih+b_hh]).tolist())
kz=np.array([[0.1,0.2,0.3]],dtype=f32); rk=np.array([[0.4,0.5,0.6]],dtype=f32); gb=np.array([[1,2,3],[4,5,6]],dtype=f32)
perm=[1,0,2]
H=1
def blocks(m,axis): return np.split(m,3,axis=axis)
Wx=np.concatenate([blocks(kz,1)[p] for p in perm],axis=1).T; Wh=np.concatenate([blocks(rk,1)[p] for p in perm],axis=1).T
bx=np.concatenate([np.split(gb[0],3)[p] for p in perm]); bh=np.concatenate([np.split(gb[1],3)[p] for p in perm])
print("E gru theta",np.concatenate([Wx.reshape(-1),Wh.reshape(-1),bx,bh]).tolist())
z2=W2@u+c2; z2f=W2f@h+c2f
print("C h",h,"u",u,"z2",z2,"z2f",z2f)
W2f32=W2f.astype(f32); c2f32=c2f.astype(f32); h32=h.astype(f32)
s=c2f32[0]
for j in range(3): s=f32(s+f32(W2f32[0,j]*h32[j]))
print("C folded float32 z2",s,"err",abs(float(s)-z2[0]))
