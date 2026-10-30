import numpy as np, onnxparse as o, json, sys, os
def convert(path,outdir,name,quant):
    nodes,inits,ins,outs=o.load(path)
    prod={t:n for n in nodes for t in n['out']}
    cons={}
    for n in nodes:
        for t in n['in']: cons.setdefault(t,[]).append(n)
    ops=[];weights=[];meta=[]  # meta per conv weights
    blobs=[]; off=[0]
    def addblob(arr):
        b=arr.tobytes(); o0=off[0]; blobs.append(b); off[0]+=len(b); 
        pad=(-off[0])%4; 
        if pad: blobs.append(b'\0'*pad); off[0]+=pad
        return o0
    memo={}
    def emit(t):
        # returns canonical tensor name for float tensor t
        if t in memo: return memo[t]
        if t==ins[0]: memo[t]='x'; return 'x'
        n=prod[t];op=n['op']
        if op=='Relu':
            src=n['in'][0]; c=prod[src]
            r=emit_conv(src) if c['op'] in('Conv','Add') else None
            assert r is not None
            ops[-1]['relu']=1 if ops[-1]['out']==r else (_ for _ in ()).throw(Exception('relu'))
            memo[t]=r; return r
        if op in('Conv','Add') and (op=='Conv' or prod.get(n['in'][0],{}).get('op') in ('Mul',)):
            r=emit_conv(t); memo[t]=r; return r
        if op=='Add':
            a=emit(n['in'][0]);b=emit(n['in'][1]); r='t%d'%len(ops); ops.append({'op':'add','a':a,'b':b,'out':r}); memo[t]=r; return r
        if op=='MaxPool':
            a=emit(n['in'][0]); r='t%d'%len(ops); ops.append({'op':'pool','in':a,'out':r}); memo[t]=r; return r
        if op=='Concat':
            ii=[emit(x) for x in n['in']]; r='t%d'%len(ops); ops.append({'op':'cat','in':ii,'out':r}); memo[t]=r; return r
        if op=='Resize':
            a=emit(n['in'][0])
            # like: a sibling non-Resize input of the consuming Concat/Add
            like=None
            for c in cons[t]:
                for s in c['in']:
                    if s!=t and s in prod and prod[s]['op']!='Resize': like=s
            assert like, t
            l=emit(like); r='t%d'%len(ops); ops.append({'op':'up','in':a,'like':l,'out':r}); memo[t]=r; return r
        if op=='Sigmoid':
            a=emit(n['in'][0]); r='t%d'%len(ops); ops.append({'op':'sig','in':a,'out':r}); memo[t]=r; return r
        raise Exception(op)
    def emit_conv(t):
        n=prod[t]
        if n['op']=='Conv':
            x=emit(n['in'][0]); w=inits[n['in'][1]].astype(np.float32); b=inits[n['in'][2]].astype(np.float32)
            a=n['attrs']; d=a['dilations'][0]; k=a['kernel_shape'][0]
            assert a['strides']==[1,1] and a['pads'][0]==d*(k-1)//2 and a.get('group',1)==1
            ent={'op':'conv','in':x,'k':k,'d':d,'cin':w.shape[1],'cout':w.shape[0]}
            # store HWIO
            whwio=np.ascontiguousarray(w.transpose(2,3,1,0))
            if quant:
                raise Exception('float conv in quant model')
            ent['w']=addblob(whwio.astype(np.float32)); ent['b']=addblob(b)
        else:  # Add(bias) <- Mul(scale) <- Cast <- ConvInteger
            add=n; mul=prod[add['in'][0]]; bias=add['in'][1]
            b=inits[prod[bias]['in'][0]] if bias in prod else inits[bias]
            cast=[prod[s] for s in mul['in'] if s in prod and prod[s]['op']=='Cast'][0]
            ci=prod[cast['in'][0]]; dq=prod[ci['in'][0]]
            x=emit(dq['in'][0])
            W=inits[ci['in'][1]]; zz=np.asarray(inits[ci['in'][3]]).ravel(); assert zz.size==1,zz.shape; wzp=float(zz[0]);
            smul=[prod[s] for s in mul['in'] if s in prod and prod[s]['op']=='Mul'][0]
            ww=np.asarray([inits[s] for s in smul['in'] if s in inits][0]).ravel(); assert ww.size==1; ws=float(ww[0])
            a=ci['attrs']; d=a['dilations'][0]; k=a['kernel_shape'][0]
            assert a['strides']==[1,1] and a['pads'][0]==d*(k-1)//2 and a.get('group',1)==1
            ent={'op':'conv','in':x,'k':k,'d':d,'cin':W.shape[1],'cout':W.shape[0],'q':[ws,wzp]}
            ent['w']=addblob(np.ascontiguousarray(W.transpose(2,3,1,0)).astype(np.uint8)); ent['b']=addblob(b.astype(np.float32))
        r='t%d'%len(ops); ent['out']=r; ops.append(ent); return r
    final=emit(outs[0])
    data=b''.join(blobs)
    CH=14*1024*1024; parts=[]
    for i in range(0,len(data),CH):
        fn='%s.%d.wasm'%(name,i//CH); open(os.path.join(outdir,fn),'wb').write(data[i:i+CH]); parts.append(fn)
    js={'ops':ops,'out':final,'parts':parts,'bytes':len(data)}
    json.dump(js,open(os.path.join(outdir,name+'.json'),'w'),separators=(',',':'))
    return js,data
if __name__=='__main__':
    # 사용법: python3 convert.py <모델.onnx> <출력폴더> <이름>
    #   예) python3 convert.py u2netp.onnx ../models u2netp
    import argparse
    ap=argparse.ArgumentParser(description='rembg U2-Net ONNX 모델을 이미지 백그라운드 제거 앱용 JSON + 가중치 파일로 변환')
    ap.add_argument('onnx'); ap.add_argument('outdir'); ap.add_argument('name')
    a=ap.parse_args()
    nodes,inits,_,_=o.load(a.onnx)
    quant=any(n['op']=='ConvInteger' for n in nodes)
    os.makedirs(a.outdir,exist_ok=True)
    js,_=convert(a.onnx,a.outdir,a.name,quant)
    print('ops:',len(js['ops']),'bytes:',js['bytes'],'parts:',js['parts'])
