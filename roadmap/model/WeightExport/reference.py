import numpy as np, struct, zlib, math
f32=np.float32

def fnv1a32(words):
    h=0x811C9DC5
    for w in words:
        for b in struct.pack('<I',w):
            h^=b; h=(h*0x01000193)&0xFFFFFFFF
    return h

def descriptor(layers):
    d=[len(layers)]
    for (i,o,p) in layers: d+= [i,o,p]
    return d

L_fix=[(2,3,9),(3,1,4)]
L_alt=[(1,2,4),(2,3,9)]
h_fix=fnv1a32(descriptor(L_fix)); h_alt=fnv1a32(descriptor(L_alt))
print("hash fixture 0x%08X  alt 0x%08X"%(h_fix,h_alt))
print("fnv1a32('') 0x%08X  'a' check:"%fnv1a32([]))
# FNV-1a known vector: "a" -> 0xE40C292C
h=0x811C9DC5
for b in b"a": h^=b; h=(h*0x01000193)&0xFFFFFFFF
print("fnv1a('a')=0x%08X"%h)

theta0=np.array([0.5,-1.0,1.5,0.25,-0.5,0.75,0.1,-0.2,0.05],dtype=f32)
theta1=np.array([1.0,-0.5,2.0,-0.3],dtype=f32)
MAGIC=0x42574E4E
hdr=struct.pack('<IHHIII',MAGIC,1,1,h_fix,2,13)
payload=theta0.astype('<f4').tobytes()+theta1.astype('<f4').tobytes()
body=hdr+payload
crc=zlib.crc32(body)&0xFFFFFFFF
blob=body+struct.pack('<I',crc)
print("magic bytes",hdr[:4], "size",len(blob),"crc 0x%08X"%crc)
print("offsets payload0",len(hdr),"payload1",len(hdr)+36,"crc at",len(body))
for k in range(0,len(blob),12):
    print("  %2d: "%k+", ".join("0x%02X"%b for b in blob[k:k+12]))
# forward in float32: dense sums bias first then j ascending
def dense(theta,In,Out,x,act):
    z=np.zeros(Out,dtype=f32)
    for i in range(Out):
        s=f32(theta[In*Out+i])
        for j in range(In): s=f32(s+f32(theta[i*In+j]*x[j]))
        z[i]=s
    return z,act(z)
lrelu=lambda z: np.where(z>0,z,f32(0.1)*z).astype(f32)
tanh=lambda z: np.tanh(z).astype(f32)
x=np.array([2.0,0.5],dtype=f32)
z1,a1=dense(theta0,2,3,x,lrelu); z2,y=dense(theta1,3,1,a1,tanh)
print("z1",z1,"a1",a1,"z2",z2,"y %.7f"%y[0])
# float64 ref
W1=np.array([[0.5,-1],[1.5,0.25],[-0.5,0.75]]);b1=np.array([0.1,-0.2,0.05])
zz=W1@np.array([2,0.5])+b1; aa=np.where(zz>0,zz,0.1*zz); yy=np.tanh(np.array([1,-0.5,2])@aa-0.3)
print("float64 y %.9f"%yy)
# nan variant
nanblob=bytearray(body); nanblob[20:24]=struct.pack('<I',0x7FC00000)
nancrc=zlib.crc32(bytes(nanblob))&0xFFFFFFFF
print("nan variant crc 0x%08X"%nancrc)
# flipped
fl=bytearray(blob); fl[20]^=0x01
print("flipped crc recomputed 0x%08X (stored 0x%08X)"%(zlib.crc32(bytes(fl[:-4]))&0xFFFFFFFF,crc))
print("crc32 check 123456789 = 0x%08X"%zlib.crc32(b"123456789"))
# float formatting
for v in [0.5,-1.0,1.5,0.25,-0.5,0.75,0.1,-0.2,0.05,1.0,-0.5,2.0,-0.3]:
    s="%.9g"%f32(v); assert f32(float(s))==f32(v); print(v,"->",s+"f")
# subnormal
sub=f32(1e-40); print("1e-40 float32",sub, "FLT_MIN",np.finfo(f32).tiny, "subnormal?",abs(sub)<np.finfo(f32).tiny)
def lit(v):
    v=f32(v); s="%.9g"%v
    if not any(c in s for c in ".en"): s+=".0"
    return s+"f"
print("layer0 {", ", ".join(lit(v) for v in theta0), "}")
print("layer1 {", ", ".join(lit(v) for v in theta1), "}")
alt_theta=np.arange(1,14,dtype=f32)
