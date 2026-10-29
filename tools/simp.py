import numpy as np, json, ref
def load(name,d='../models'):
    js=json.load(open(f'{d}/{name}.json')); data=b''.join(open(f'{d}/{p}','rb').read() for p in js['parts'])
    return js,data
def run(js,data,x):  # x NCHW
    env={'x':x}
    for op in js['ops']:
        t=op['op']
        if t=='conv':
            k,ci,co=op['k'],op['cin'],op['cout']; n=k*k*ci*co
            if 'q' in op:
                w=(np.frombuffer(data,np.uint8,n,op['w']).astype(np.float32)-op['q'][1])*op['q'][0]
            else: w=np.frombuffer(data,np.float32,n,op['w'])
            w=w.reshape(k,k,ci,co).transpose(3,2,0,1)
            b=np.frombuffer(data,np.float32,co,op['b'])
            y=ref.im2col_conv(env[op['in']],w,b,op['d']*(k-1)//2,op['d'])
            if op.get('relu'): y=np.maximum(y,0)
        elif t=='pool': y=ref.maxpool(env[op['in']])
        elif t=='up': l=env[op['like']]; y=ref.resize_lin(env[op['in']],l.shape[2],l.shape[3])
        elif t=='cat': y=np.concatenate([env[i] for i in op['in']],1)
        elif t=='add': y=env[op['a']]+env[op['b']]
        elif t=='sig': y=1/(1+np.exp(-env[op['in']]))
        env[op['out']]=y
    return env[js['out']]
