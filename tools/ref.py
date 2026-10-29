import numpy as np, onnxparse as o
def im2col_conv(x,w,b,pads,dil):
    # x: (1,C,H,W) float32, w (O,C,k,k)
    O,C,k,_=w.shape; d=dil; p=pads
    xp=np.pad(x[0],((0,0),(p,p),(p,p)))
    H=x.shape[2];W=x.shape[3]
    cols=np.empty((C,k,k,H,W),np.float32)
    for i in range(k):
        for j in range(k):
            cols[:,i,j]=xp[:,i*d:i*d+H,j*d:j*d+W]
    y=w.reshape(O,-1)@cols.reshape(C*k*k,H*W)
    if b is not None: y+=b[:,None]
    return y.reshape(1,O,H,W)
def maxpool(x):
    N,C,H,W=x.shape;H2=-(-H//2);W2=-(-W//2)
    xp=np.full((N,C,H2*2,W2*2),-np.inf,np.float32);xp[:,:,:H,:W]=x
    return xp.reshape(N,C,H2,2,W2,2).max(axis=(3,5))
def resize_lin(x,oh,ow):
    N,C,H,W=x.shape
    def coords(o,i):
        s=i/o; c=(np.arange(o)+0.5)*s-0.5
        if o==1: c=np.zeros(1)
        c=np.clip(c,0,i-1); f=np.floor(c).astype(int); f2=np.minimum(f+1,i-1); t=(c-f).astype(np.float32)
        return f,f2,t
    y0,y1,ty=coords(oh,H);x0,x1,tx=coords(ow,W)
    a=x[:,:,y0]*(1-ty)[:,None]+x[:,:,y1]*ty[:,None]
    return a[:,:,:,x0]*(1-tx)+a[:,:,:,x1]*tx
def run_onnx(path,inp,want):
    nodes,inits,ins,outs=o.load(path)
    env=dict(inits);env[ins[0]]=inp
    for n in nodes:
        op=n['op'];a=n['attrs'];I=[env[t] if t else None for t in n['in']]
        if op=='Constant': r=[a['value']]
        elif op=='Conv':
            assert a.get('group',1)==1 and a['strides']==[1,1]
            r=[im2col_conv(I[0],I[1],I[2] if len(I)>2 else None,a['pads'][0],a['dilations'][0])]
        elif op=='DynamicQuantizeLinear':
            x=I[0];mn=min(0,x.min());mx=max(0,x.max());s=np.float32((mx-mn)/255) or np.float32(1)
            zp=np.clip(np.round(-mn/s),0,255);q=np.clip(np.round(x/s)+zp,0,255)
            r=[q.astype(np.float32),np.float32(s),np.float32(zp)]
        elif op=='ConvInteger':
            x=I[0]-I[2];w=I[1].astype(np.float32)-np.float32(I[3])
            # pad with zp => after subtraction zero; ok
            r=[im2col_conv(x,w,None,a['pads'][0],a['dilations'][0])]
        elif op=='Cast': r=[I[0].astype(np.float32) if a['to']==1 else I[0].astype(np.int64)]
        elif op=='Mul': r=[I[0]*I[1]]
        elif op=='Add': r=[I[0]+I[1]]
        elif op=='Reshape': r=[I[0].reshape(I[1])]
        elif op=='Relu': r=[np.maximum(I[0],0)]
        elif op=='Sigmoid': r=[1/(1+np.exp(-I[0]))]
        elif op=='MaxPool': r=[maxpool(I[0])]
        elif op=='Concat': r=[np.concatenate([np.atleast_1d(t) for t in I],axis=a['axis'])]
        elif op=='Shape': r=[np.array(I[0].shape,np.int64)]
        elif op=='Gather': r=[np.take(I[0],I[1],axis=a.get('axis',0))]
        elif op=='Unsqueeze': r=[np.expand_dims(I[0],a['axes'][0])]
        elif op=='Slice':
            s=int(I[1][0]);e=int(I[2][0]);ax=int(I[3][0]) if len(I)>3 else 0
            e=min(e,len(I[0]));r=[I[0][s:e]]
        elif op=='Resize':
            sz=I[3] if len(I)>3 and I[3] is not None and I[3].size else None
            if sz is None: sc=I[2];sz=[int(I[0].shape[i]*sc[i]) for i in range(4)]
            r=[resize_lin(I[0],int(sz[2]),int(sz[3]))]
        else: raise Exception(op)
        for t,v in zip(n['out'],r): env[t]=v
    return env[want],env
