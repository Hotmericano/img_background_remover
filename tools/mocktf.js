// naive NHWC tf mock for testing graph logic
class T{constructor(d,s){this.d=d;this.shape=s;this.disposed=false}dispose(){this.disposed=true}}
const chk=t=>{if(t.disposed)throw new Error('use after dispose');return t};
const tf={
 tensor4d:(d,s)=>new T(Float32Array.from(d),s), tensor1d:d=>new T(Float32Array.from(d),[d.length]),
 fused:{conv2d({x,filter,dilations:dl,bias,activation}){chk(x);const [,H,W,C]=x.shape;const [k,,ci,co]=filter.shape;if(ci!==C)throw new Error('cin');
  const p=dl*(k-1)/2;const o=new Float32Array(H*W*co);const xd=x.d,wd=filter.d,bd=bias.d;
  for(let y=0;y<H;y++)for(let xx=0;xx<W;xx++){const ob=(y*W+xx)*co;for(let c=0;c<co;c++)o[ob+c]=bd[c];
   for(let i=0;i<k;i++){const yy=y+i*dl-p;if(yy<0||yy>=H)continue;for(let j=0;j<k;j++){const x2=xx+j*dl-p;if(x2<0||x2>=W)continue;
    const xb=(yy*W+x2)*C,wb=(i*k+j)*ci*co;for(let c=0;c<C;c++){const v=xd[xb+c];if(v===0)continue;const wr=wb+c*co;for(let q=0;q<co;q++)o[ob+q]+=v*wd[wr+q];}}}
   if(activation==='relu')for(let c=0;c<co;c++)if(o[ob+c]<0)o[ob+c]=0;}
  return new T(o,[1,H,W,co]);}},
 maxPool(x){chk(x);const [,H,W,C]=x.shape;const H2=Math.ceil(H/2),W2=Math.ceil(W/2);const o=new Float32Array(H2*W2*C).fill(-Infinity);
  for(let y=0;y<H;y++)for(let xx=0;xx<W;xx++)for(let c=0;c<C;c++){const i=((y>>1)*W2+(xx>>1))*C+c;const v=x.d[(y*W+xx)*C+c];if(v>o[i])o[i]=v;}return new T(o,[1,H2,W2,C]);},
 image:{resizeBilinear(x,[oh,ow]){chk(x);const [,H,W,C]=x.shape;const o=new Float32Array(oh*ow*C);
  const cc=(o2,i)=>{const r=[];for(let a=0;a<o2;a++){let c=(a+0.5)*i/o2-0.5;c=Math.min(Math.max(c,0),i-1);const f=Math.floor(c);r.push([f,Math.min(f+1,i-1),c-f]);}return r};
  const Y=cc(oh,H),X=cc(ow,W);for(let y=0;y<oh;y++)for(let xx=0;xx<ow;xx++)for(let c=0;c<C;c++){const [y0,y1,ty]=Y[y],[x0,x1,tx]=X[xx];const g=(a,b)=>x.d[(a*W+b)*C+c];
   o[(y*ow+xx)*C+c]=(g(y0,x0)*(1-tx)+g(y0,x1)*tx)*(1-ty)+(g(y1,x0)*(1-tx)+g(y1,x1)*tx)*ty;}return new T(o,[1,oh,ow,C]);}},
 concat(ts){ts.forEach(chk);const [,H,W]=ts[0].shape;const C=ts.reduce((s,t)=>s+t.shape[3],0);const o=new Float32Array(H*W*C);
  for(let p=0;p<H*W;p++){let off=0;for(const t of ts){const c=t.shape[3];o.set(t.d.subarray(p*c,p*c+c),p*C+off);off+=c;}}return new T(o,[1,H,W,C]);},
 add(a,b){chk(a);chk(b);return new T(a.d.map((v,i)=>v+b.d[i]),a.shape)},
 sigmoid(a){chk(a);return new T(a.d.map(v=>1/(1+Math.exp(-v))),a.shape)}
};
module.exports=tf;
