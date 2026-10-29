const fs=require('fs');const tf=require('./mocktf');const {buildModel}=require('./engine');
const name=process.argv[2];const spec=JSON.parse(fs.readFileSync(`../models/${name}.json`));
const parts=spec.parts.map(p=>fs.readFileSync('../models/'+p));const all=Buffer.concat(parts);const buf=all.buffer.slice(all.byteOffset,all.byteOffset+all.length);
const m=buildModel(tf,spec,buf);
const x=new Float32Array(fs.readFileSync('x64.bin').buffer.slice(0));
const y=m.run(new T0(x));function T0(d){return {d,shape:[1,64,64,3],dispose(){}}}
fs.writeFileSync(`y64_${name}.bin`,Buffer.from(y.d.buffer));console.log('ok',y.shape);
