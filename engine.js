// ===== U²-Net 추론 엔진 (tf.js 위에서 동작) =====
// 모델 = { ops: [...], out, parts:[...] } + 가중치 바이너리
function buildModel(tf, spec, buf) {
  const lastUse = {};
  spec.ops.forEach((op, i) => {
    const ins = op.op === 'cat' ? op.in : op.op === 'add' ? [op.a, op.b] : op.op === 'up' ? [op.in, op.like] : [op.in];
    ins.forEach(t => { lastUse[t] = i; });
  });
  lastUse[spec.out] = Infinity;
  const W = {};
  for (const op of spec.ops) {
    if (op.op !== 'conv') continue;
    const n = op.k * op.k * op.cin * op.cout;
    let w;
    if (op.q) {
      const u8 = new Uint8Array(buf, op.w, n);
      w = new Float32Array(n);
      const s = op.q[0], z = op.q[1];
      for (let i = 0; i < n; i++) w[i] = (u8[i] - z) * s;
    } else {
      w = new Float32Array(buf.slice(op.w, op.w + n * 4));
    }
    const b = new Float32Array(buf.slice(op.b, op.b + op.cout * 4));
    W[op.out] = {
      w: tf.tensor4d(w, [op.k, op.k, op.cin, op.cout]),
      b: tf.tensor1d(b)
    };
  }
  function run(x /* [1,H,W,3] */) {
    const env = { x };
    spec.ops.forEach((op, i) => {
      let y;
      switch (op.op) {
        case 'conv':
          y = tf.fused.conv2d({
            x: env[op.in], filter: W[op.out].w, strides: 1, pad: 'same',
            dilations: op.d, bias: W[op.out].b, activation: op.relu ? 'relu' : 'linear'
          });
          break;
        case 'pool': y = tf.maxPool(env[op.in], 2, 2, 'same'); break;
        case 'up': {
          const s = env[op.like].shape;
          y = tf.image.resizeBilinear(env[op.in], [s[1], s[2]], false, true);
          break;
        }
        case 'cat': y = tf.concat(op.in.map(t => env[t]), 3); break;
        case 'add': y = tf.add(env[op.a], env[op.b]); break;
        case 'sig': y = tf.sigmoid(env[op.in]); break;
        default: throw new Error('unknown op ' + op.op);
      }
      env[op.out] = y;
      for (const t in env) {
        if (t !== 'x' && lastUse[t] <= i) { env[t].dispose(); delete env[t]; }
      }
    });
    return env[spec.out];
  }
  function dispose() { for (const k in W) { W[k].w.dispose(); W[k].b.dispose(); } }
  return { run, dispose };
}
if (typeof module !== 'undefined') module.exports = { buildModel };
