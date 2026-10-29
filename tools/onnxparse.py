import struct, numpy as np, sys
def varint(b,i):
    r=0;s=0
    while True:
        c=b[i];i+=1;r|=(c&0x7f)<<s;s+=7
        if c<0x80: return r,i
def fields(b):
    i=0;n=len(b)
    while i<n:
        k,i=varint(b,i);f=k>>3;w=k&7
        if w==0: v,i=varint(b,i)
        elif w==1: v=b[i:i+8];i+=8
        elif w==2:
            l,i=varint(b,i);v=b[i:i+l];i+=l
        elif w==5: v=b[i:i+4];i+=4
        else: raise Exception('wt %d'%w)
        yield f,w,v
def packed_varints(v):
    out=[];i=0
    while i<len(v):
        x,i=varint(v,i)
        if x>=1<<63: x-=1<<64
        out.append(x)
    return out
DT={2:np.uint8,3:np.int8,1:np.float32,7:np.int64,6:np.int32,11:np.float64,10:np.float16}
def tensor(b):
    dims=[];dt=1;name='';raw=None;fd=[];i64=[]
    for f,w,v in fields(b):
        if f==1: dims+= packed_varints(v) if w==2 else [v]
        elif f==2: dt=v
        elif f==8: name=v.decode()
        elif f==9: raw=v
        elif f==4: fd+= list(struct.unpack('<%df'%(len(v)//4),v)) if w==2 else [struct.unpack('<f',v)[0]]
        elif f==7 or f==5: i64+= packed_varints(v) if w==2 else [v]
    if raw is not None: a=np.frombuffer(raw,dtype=DT[dt]).copy()
    elif dt==1: a=np.array(fd,np.float32)
    else: a=np.array(i64,DT.get(dt,np.int64))
    return name,(a.reshape(dims) if dims else (a.reshape(()) if a.size==1 else a))
def attr(b):
    name='';val=None;fl=[];il=[];t=None
    for f,w,v in fields(b):
        if f==1: name=v.decode()
        elif f==2: val=struct.unpack('<f',v)[0]
        elif f==3: val=v if v<1<<63 else v-(1<<64)
        elif f==4: val=v.decode()
        elif f==5: val=tensor(v)[1]
        elif f==7: fl+= list(struct.unpack('<%df'%(len(v)//4),v)) if w==2 else [struct.unpack('<f',v)[0]]
        elif f==8: il+= packed_varints(v) if w==2 else [v]
    if fl: val=fl
    if il: val=il
    return name,val
def node(b):
    d={'in':[],'out':[],'op':'','attrs':{}}
    for f,w,v in fields(b):
        if f==1: d['in'].append(v.decode())
        elif f==2: d['out'].append(v.decode())
        elif f==4: d['op']=v.decode()
        elif f==5:
            k,val=attr(v); d['attrs'][k]=val
    return d
def valueinfo_name(b):
    for f,w,v in fields(b):
        if f==1: return v.decode()
def load(path):
    b=open(path,'rb').read()
    g=None
    for f,w,v in fields(b):
        if f==7: g=v
    nodes=[];inits={};ins=[];outs=[]
    for f,w,v in fields(g):
        if f==1: nodes.append(node(v))
        elif f==5:
            n,a=tensor(v); inits[n]=a
        elif f==11: ins.append(valueinfo_name(v))
        elif f==12: outs.append(valueinfo_name(v))
    ins=[x for x in ins if x not in inits]
    return nodes,inits,ins,outs
if __name__=='__main__':
    nodes,inits,ins,outs=load(sys.argv[1])
    from collections import Counter
    print(len(nodes),len(inits),ins,outs)
    print(Counter(n['op'] for n in nodes))
    seen=set()
    for n in nodes:
        key=(n['op'],str(sorted((k,str(v)) for k,v in n['attrs'].items() if not hasattr(v,'shape'))))
        if key not in seen:
            seen.add(key); print(key)
