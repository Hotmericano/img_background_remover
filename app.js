// ===== 화면, 사진 처리, 저장 =====
// 추론 엔진(buildModel)은 engine.js, TensorFlow.js(tf)와 JSZip은 index.html에서 먼저 불러와요.
// 전체를 (() => { ... })() 안에 넣어서 변수가 다른 스크립트와 섞이지 않게 해요.
(() => {
  const SIZE = 320;
  const isMobile = matchMedia('(pointer: coarse)').matches || /Android|iPhone|iPad|iPod/i.test(navigator.userAgent) || (navigator.maxTouchPoints > 1 && /Macintosh/.test(navigator.userAgent));
  // 휴대폰 브라우저는 캔버스 크기와 메모리 한도가 작아서 결과 해상도를 낮게 잡아요.
  const MAX_SIDE = isMobile ? 2560 : 4096, MAX_PIXELS = isMobile ? 5e6 : 16e6;
  const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
  // 정밀 모델이 기기 한도를 넘었을 때 나오는 에러 메시지 모양
  //   GPU_LIMIT_ERROR: 이런 에러면 빠름 모델로 자동으로 바꿔 다시 시도해요.
  //   GPU_MEMORY_HINT: 자동 전환 뒤에도 실패하면 "빠름 모델로 바꿔 보세요"를 덧붙여요.
  const GPU_LIMIT_ERROR = /memory|texture|context|WebGL|allocat|lost|failed to compile/i;
  const GPU_MEMORY_HINT = /memory|texture|context|WebGL/i;
  const $ = s => document.querySelector(s);
  const grid = $('#grid'), statusEl = $('#status'), drop = $('#drop'), fileIn = $('#file');
  let modelName = 'silueta', bgMode = 'clear', cropOn = true;
  const models = {};            // name -> Promise<{run,dispose}>
  const items = [];             // {id, name, blob, url, card, ...}
  const queue = []; let working = false; let nextId = 1;
  let downloads = null;

  let savedPick = false;
  try { const m = localStorage.getItem('bg-remover-model') || localStorage.getItem('nukki-model'); if (m === 'u2netp' || m === 'silueta') { modelName = m; savedPick = true; } } catch (e) {}
  if (isMobile && !savedPick) { modelName = 'u2netp'; $('#devnote').hidden = false; }
  syncSeg();

  // ---------- UI helpers ----------
  function syncSeg() {
    document.querySelectorAll('[data-model]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.model === modelName)));
    document.querySelectorAll('[data-bg]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.bg === bgMode)));
    document.querySelectorAll('.stage').forEach(s => s.classList.toggle('white', bgMode === 'white'));
    document.querySelectorAll('[data-crop]').forEach(b => b.setAttribute('aria-pressed', String((b.dataset.crop === 'on') === cropOn)));
  }
  document.querySelectorAll('[data-model]').forEach(b => b.addEventListener('click', () => {
    modelName = b.dataset.model; syncSeg();
    try { localStorage.setItem('bg-remover-model', modelName); } catch (e) {}
  }));
  document.querySelectorAll('[data-bg]').forEach(b => b.addEventListener('click', () => { bgMode = b.dataset.bg; syncSeg(); }));
  document.querySelectorAll('[data-crop]').forEach(b => b.addEventListener('click', () => {
    cropOn = b.dataset.crop === 'on'; syncSeg();
    items.forEach(showResult);
  }));
  function setStatus(text, frac) {
    statusEl.textContent = text;
    if (frac != null) {
      const bar = document.createElement('span'); bar.className = 'bar';
      const i = document.createElement('i'); i.style.width = Math.round(frac * 100) + '%'; bar.appendChild(i);
      statusEl.appendChild(bar);
    }
  }
  let toastT;
  function toast(msg) {
    let t = $('.toast'); if (!t) { t = document.createElement('div'); t.className = 'toast'; t.setAttribute('role', 'status'); document.body.appendChild(t); }
    t.textContent = msg; t.hidden = false; clearTimeout(toastT); toastT = setTimeout(() => t.hidden = true, 2600);
  }
  const fmtKB = n => n > 1048576 ? (n / 1048576).toFixed(1) + 'MB' : Math.max(1, Math.round(n / 1024)) + 'KB';
  const baseName = n => (n || 'image').replace(/\.[^.]+$/, '');

  // ---------- 파일 입력 ----------
  drop.addEventListener('click', () => fileIn.click());
  drop.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileIn.click(); } });
  fileIn.addEventListener('change', () => { addFiles(fileIn.files); fileIn.value = ''; });
  ['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', e => addFiles(e.dataTransfer.files));
  window.addEventListener('paste', e => {
    const fs = [...(e.clipboardData?.files || [])]; if (fs.length) addFiles(fs);
  });

  function addFiles(list) {
    const files = [...list].filter(f => f.type.startsWith('image/'));
    if (!files.length) { toast('이미지 파일만 올릴 수 있어요'); return; }
    removeExample();
    for (const f of files) {
      const it = { id: nextId++, name: f.name || ('붙여넣은 이미지 ' + nextId + '.png'), file: f, model: modelName };
      it.card = makeCard(it, URL.createObjectURL(f));
      grid.prepend(it.card.root);
      items.push(it); queue.push(it);
    }
    pump();
  }

  function makeCard(it, origUrl, example) {
    const root = document.createElement('article'); root.className = 'card';
    const stage = document.createElement('div'); stage.className = 'stage' + (bgMode === 'white' ? ' white' : '');
    const out = document.createElement('img'); out.alt = it.name + ' 배경 제거 결과'; out.hidden = true;
    const orig = document.createElement('img'); orig.className = 'orig'; orig.alt = '원본'; orig.src = origUrl;
    const busy = document.createElement('div'); busy.className = 'busy';
    const busyText = document.createElement('span'); busyText.textContent = '대기 중';
    const busyBar = document.createElement('div'); busyBar.className = 'pbar';
    const busyFill = document.createElement('i'); busyBar.append(busyFill);
    busyBar.hidden = true;   // 차례를 기다리는 동안에는 막대 숨김
    busy.append(busyText, busyBar);
    stage.append(out, orig, busy);
    const info = document.createElement('div'); info.className = 'info';
    const nm = document.createElement('div'); nm.className = 'name';
    const b = document.createElement('b'); b.textContent = it.name; b.title = it.name;
    const meta = document.createElement('span'); meta.textContent = '—';
    nm.append(b, meta);
    const acts = document.createElement('div'); acts.className = 'acts';
    if (example) { const c = document.createElement('span'); c.className = 'chip ex'; c.textContent = '예시'; acts.append(c); }
    const save = document.createElement('button'); save.className = 'btn'; save.textContent = 'PNG 저장'; save.disabled = true;
    save.addEventListener('click', () => saveOne(it));
    const del = document.createElement('button'); del.className = 'btn'; del.textContent = '삭제'; del.setAttribute('aria-label', it.name + ' 삭제');
    del.addEventListener('click', () => removeItem(it));
    if (!example) acts.append(save);
    acts.append(del);
    info.append(nm, acts);
    root.append(stage, info);
    return { root, stage, out, orig, busy, busyText, busyBar, busyFill, meta, save, acts };
  }

  function removeItem(it) {
    const i = items.indexOf(it); if (i >= 0) items.splice(i, 1);
    const q = queue.indexOf(it); if (q >= 0) queue.splice(q, 1);
    it.card.root.remove();
    if (it.resultUrl) URL.revokeObjectURL(it.resultUrl);
    if (it.card.orig.src.startsWith('blob:')) URL.revokeObjectURL(it.card.orig.src);
    updateSaveAll();
  }
  $('#clearAll').addEventListener('click', () => { [...items].forEach(removeItem); removeExample(); });

  function updateSaveAll() {
    const done = items.filter(i => i.result).length;
    $('#saveAll').disabled = !downloads || done === 0;
    $('#saveAll').textContent = done > 1 ? `전체 저장 (ZIP · ${done}장)` : '전체 저장 (ZIP)';
  }

  // ---------- 모델 ----------
  async function fetchWithProgress(urls, onProg) {
    const resps = await Promise.all(urls.map(u => fetch(u)));
    resps.forEach((r, i) => { if (!r.ok) throw new Error('모델 파일을 받지 못했어요 (' + urls[i] + ', ' + r.status + ')'); });
    const totals = resps.map(r => +r.headers.get('content-length') || 0);
    const total = totals.reduce((a, b) => a + b, 0); let got = 0;
    const bufs = await Promise.all(resps.map(async r => {
      if (!r.body || !total) return new Uint8Array(await r.arrayBuffer());
      const reader = r.body.getReader(); const chunks = []; let len = 0;
      for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); len += value.length; got += value.length; onProg(got / total); }
      const out = new Uint8Array(len); let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; } return out;
    }));
    const len = bufs.reduce((a, b) => a + b.length, 0); const all = new Uint8Array(len); let o = 0;
    for (const b of bufs) { all.set(b, o); o += b.length; }
    return all.buffer;
  }
  let backendReady = null;
  function ensureBackend() {
    if (!backendReady) backendReady = (async () => {
      if (typeof tf === 'undefined') throw new Error('TensorFlow.js를 불러오지 못했어요. 새로고침해 보세요.');
      try {
        tf.env().set('WEBGL_DELETE_TEXTURE_THRESHOLD', 0);   // 다 쓴 GPU 메모리를 바로 돌려줌
        if (isMobile) tf.env().set('WEBGL_FLUSH_THRESHOLD', 1);
      } catch (e) {}
      try { await tf.setBackend('webgl'); } catch (e) {}
      if (tf.getBackend() !== 'webgl') { await tf.setBackend('cpu'); }
      await tf.ready();
      if (tf.getBackend() === 'webgl') {
        // 휴대폰 GPU는 텍스처(GPU가 다루는 2차원 데이터) 한 장의 최대 크기가 작아요. 보통 4096×4096이에요.
        // TensorFlow.js는 합성곱을 빠르게 하려고 입력을 큰 행렬로 펼친 뒤(im2col) 행렬 곱으로 계산하는데,
        // 이 모델은 320×320 사진에서 그 행렬이 최대 10862×10862 텍스처만큼 커져서 한도를 넘어요.
        // 그래서 한도가 작은 기기에서는 이 방식을 끄고 합성곱을 직접 계산해요. 조금 느리지만 한도 안에서 돌아가요.
        const maxTex = tf.env().getNumber('WEBGL_MAX_TEXTURE_SIZE');
        if (maxTex < 11000) tf.env().set('WEBGL_CONV_IM2COL', false);
      }
      return tf.getBackend();
    })();
    return backendReady;
  }
  function getModel(name) {
    if (!models[name]) models[name] = (async () => {
      const backend = await ensureBackend();
      const label = name === 'silueta' ? '정밀 모델' : '빠름 모델';
      setStatus(label + ' 내려받는 중', 0);
      const spec = await (await fetch('models/' + name + '.json')).json();
      const buf = await fetchWithProgress(spec.parts.map(p => 'models/' + p), f => setStatus(label + ' 내려받는 중 ' + Math.round(f * 100) + '%', f));
      setStatus(label + ' 준비 중 (GPU 셰이더 컴파일)…');
      await new Promise(r => setTimeout(r, 30));
      const m = buildModel(tf, spec, buf);
      // 워밍업: 첫 실행에 셰이더가 컴파일됨
      const y = tf.tidy(() => m.run(tf.zeros([1, SIZE, SIZE, 3])));
      await y.data(); y.dispose();
      m.backend = backend;
      return m;
    })().catch(e => { delete models[name]; throw e; });
    return models[name];
  }

  // ---------- 처리 ----------
  async function pump() {
    if (working) return; working = true;
    while (queue.length) {
      const it = queue.shift();
      const left = queue.length;
      try {
        it.card.busyBar.hidden = false;
        const step = makeSteps(it.card.busyFill, it.card.busyText, ['모델 준비', '사진 읽기', '오리는 중', 'PNG 만들기']);
        await step(0);
        const m = await getModel(it.model);
        setStatus(`오리는 중… (남은 사진 ${left}장)` + (m.backend === 'cpu' ? ' · GPU를 못 써서 느려요' : ''));
        await process(it, m, step);
        await new Promise(r => setTimeout(r, 0));   // 사진 사이에 화면이 멈추지 않도록 한 번 숨 고르기
      } catch (e) {
        const msg = (e && e.message) || String(e);
        if (it.model === 'silueta' && !it.retried && GPU_LIMIT_ERROR.test(msg)) {
          // 정밀 모델이 기기 메모리를 넘으면 빠름 모델로 자동 전환
          it.retried = true; it.model = 'u2netp';
          try { const sm = await models.silueta; sm && sm.dispose(); } catch (_) {}
          delete models.silueta;
          modelName = 'u2netp'; syncSeg();
          toast('메모리가 부족해서 빠름 모델로 바꿔 다시 오릴게요');
          queue.unshift(it); queue.forEach(q => { if (q.model === 'silueta') q.model = 'u2netp'; });
          continue;
        }
        console.error(e);
        const hint = it.model === 'silueta' && GPU_MEMORY_HINT.test(msg) ? ' · 빠름 모델로 바꿔 보세요' : '';
        it.card.busyBar.hidden = true;
        it.card.busyText.textContent = '실패: ' + msg.slice(0, 80) + hint;
        it.card.meta.innerHTML = '<span class="chip err">오류</span>';
      }
    }
    working = false;
    const done = items.filter(i => i.result).length;
    setStatus(done ? `완료 · ${done}장 오렸어요` : '모델은 첫 사진을 올릴 때 내려받아요.');
    updateSaveAll();
  }

  async function process(it, m, step = async () => {}) {
    const t0 = performance.now();
    await step(1);
    const bmp = await decodeImage(it.file);
    const { W, H } = fitSize(bmp);
    await step(2);
    const mask = await predictMask(m, bmp, 0, 0, bmp.width, bmp.height);
    await step(3);
    await composeResult(it, bmp, mask, W, H);
    bmp.close && bmp.close(); mask.width = mask.height = 0;
    it.ms = Math.round(performance.now() - t0);
    it.auto = { full: it.full, cropped: it.cropped, ms: it.ms };   // "자동 결과로 되돌리기"용
    showResult(it);
    it.card.out.hidden = false; it.card.busy.remove();
    it.card.save.disabled = !downloads;
    addRegionButton(it);
    updateSaveAll();
  }

  // 화면이 한 번 다시 그려질 때까지 기다리기
  // 자바스크립트가 쉬지 않고 일하면 브라우저는 화면을 갱신하지 못해요. 진행 막대를 바꾼 뒤 이걸 기다려야 바뀐 막대가 보여요.
  // (탭이 백그라운드라 requestAnimationFrame이 멈춰도 50ms 뒤에는 넘어가도록 setTimeout을 같이 걸어 둠)
  function nextFrame() {
    return new Promise(resolve => {
      let done = false; const go = () => { if (!done) { done = true; resolve(); } };
      requestAnimationFrame(() => setTimeout(go, 0));
      setTimeout(go, 50);
    });
  }

  // 단계 기준 진행 막대 만들기
  // labels: 단계 이름 배열. 돌려받은 step(i)를 부르면 "i번째 단계 시작"으로 막대와 글자를 바꿔요.
  function makeSteps(fill, text, labels) {
    return async i => {
      fill.style.width = Math.round(i / labels.length * 100) + '%';
      text.textContent = `${i + 1}/${labels.length} · ${labels[i]}`;
      await nextFrame();
    };
  }

  // 결과 이미지 크기 정하기 (휴대폰은 메모리 한도 때문에 더 작게)
  function fitSize(bmp) {
    const sc = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height), Math.sqrt(MAX_PIXELS / (bmp.width * bmp.height)));
    return { W: Math.round(bmp.width * sc), H: Math.round(bmp.height * sc) };
  }

  // 이미지의 (sx, sy, sw, sh) 부분을 모델에 넣어 320×320 마스크 캔버스를 얻기
  // 사진 전체를 넣으면 자동 결과, 사각형 부분만 넣으면 영역 지정 결과가 됨
  async function predictMask(m, src, sx, sy, sw, sh) {
    // 1) 320×320 입력 만들기
    const c320 = document.createElement('canvas'); c320.width = c320.height = SIZE;
    const g = c320.getContext('2d', { willReadFrequently: true });
    g.imageSmoothingQuality = 'high'; g.drawImage(src, sx, sy, sw, sh, 0, 0, SIZE, SIZE);
    const px = g.getImageData(0, 0, SIZE, SIZE).data;
    c320.width = c320.height = 0;
    let mx = 0; for (let i = 0; i < px.length; i += 4) { if (px[i] > mx) mx = px[i]; if (px[i + 1] > mx) mx = px[i + 1]; if (px[i + 2] > mx) mx = px[i + 2]; }
    mx = mx || 1;
    const inp = new Float32Array(SIZE * SIZE * 3);
    for (let p = 0, q = 0; p < px.length; p += 4, q += 3) {
      for (let c = 0; c < 3; c++) inp[q + c] = (px[p + c] / mx - MEAN[c]) / STD[c];
    }
    // 2) 추론
    const x = tf.tensor4d(inp, [1, SIZE, SIZE, 3]);
    const y = m.run(x); x.dispose();
    const pred = await y.data(); y.dispose();
    // 3) 0~1로 늘려서 알파 마스크 캔버스로
    let lo = Infinity, hi = -Infinity; for (const v of pred) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (!isFinite(lo) || !isFinite(hi)) throw new Error('GPU 계산 결과가 올바르지 않아요 (WebGL 정밀도 문제). 다른 브라우저에서 시도해 보세요');
    const span = (hi - lo) || 1;
    const mc = document.createElement('canvas'); mc.width = mc.height = SIZE;
    const mg = mc.getContext('2d'); const mi = mg.createImageData(SIZE, SIZE);
    for (let i = 0; i < pred.length; i++) mi.data[i * 4 + 3] = Math.round((pred[i] - lo) / span * 255);
    mg.putImageData(mi, 0, 0);
    return mc;
  }

  // 사각형마다 모델을 따로 돌려서, W×H 크기의 마스크 하나로 합치기
  // boxes: 원본 이미지 좌표 {x, y, w, h}. 사각형 밖은 투명(배경), 겹치면 큰 값
  async function regionMask(m, bmp, boxes, W, H, onBox = async () => {}) {
    const out = document.createElement('canvas'); out.width = W; out.height = H;
    const og = out.getContext('2d');
    const all = og.createImageData(W, H);   // 처음엔 전부 0 = 전부 배경
    const kx = W / bmp.width, ky = H / bmp.height;   // 원본 좌표 → 결과 좌표 배율
    for (const [n, b] of boxes.entries()) {
      await onBox(n);   // 진행 막대: "사각형 n+1/전체" 단계 시작
      const mask = await predictMask(m, bmp, b.x, b.y, b.w, b.h);
      const dx = Math.floor(b.x * kx), dy = Math.floor(b.y * ky);
      const dw = Math.max(1, Math.min(W - dx, Math.round(b.w * kx)));
      const dh = Math.max(1, Math.min(H - dy, Math.round(b.h * ky)));
      // 320×320 마스크를 사각형 크기로 늘리기
      const t = document.createElement('canvas'); t.width = dw; t.height = dh;
      const tg = t.getContext('2d'); tg.imageSmoothingQuality = 'high';
      tg.drawImage(mask, 0, 0, dw, dh);
      const part = tg.getImageData(0, 0, dw, dh).data;
      for (let y = 0; y < dh; y++) {
        for (let x = 0; x < dw; x++) {
          const a = part[(y * dw + x) * 4 + 3];
          const i = ((dy + y) * W + (dx + x)) * 4 + 3;
          if (a > all.data[i]) all.data[i] = a;   // 픽셀마다 큰 값 고르기
        }
      }
      mask.width = mask.height = 0; t.width = t.height = 0;
    }
    og.putImageData(all, 0, 0);
    return out;
  }

  // 마스크를 알파로 합성하고, 원본 크기 결과와 여백을 자른 결과를 둘 다 만들어 두기
  async function composeResult(it, bmp, mask, W, H) {
    const oc = document.createElement('canvas'); oc.width = W; oc.height = H;
    const og = oc.getContext('2d');
    og.imageSmoothingQuality = 'high';
    og.drawImage(bmp, 0, 0, W, H);
    og.globalCompositeOperation = 'destination-in';
    og.drawImage(mask, 0, 0, W, H);
    const toPng = c => new Promise((res, rej) => c.toBlob(b => b ? res(b) : rej(new Error('PNG로 저장하지 못했어요')), 'image/png'));
    const full = { blob: await toPng(oc), w: W, h: H };
    let cropped = full;
    const box = findBounds(oc);
    if (box && (box.w < W || box.h < H)) {
      const cc = document.createElement('canvas'); cc.width = box.w; cc.height = box.h;
      cc.getContext('2d').drawImage(oc, box.x, box.y, box.w, box.h, 0, 0, box.w, box.h);
      cropped = { blob: await toPng(cc), w: box.w, h: box.h };
      cc.width = cc.height = 0;
    }
    oc.width = oc.height = 0;
    it.full = full; it.cropped = cropped;
  }

  // 알파 값이 기준값보다 큰 픽셀을 모두 감싸는 가장 작은 사각형(바운딩 박스) 찾기
  function findBounds(canvas, threshold = 10) {
    const W = canvas.width, H = canvas.height;
    const d = canvas.getContext('2d').getImageData(0, 0, W, H).data;
    let top = H, bottom = -1, left = W, right = -1;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        if (d[(y * W + x) * 4 + 3] > threshold) {   // RGBA 중 4번째(인덱스 3)가 알파
          if (y < top) top = y;
          if (y > bottom) bottom = y;
          if (x < left) left = x;
          if (x > right) right = x;
        }
      }
    }
    if (bottom < 0) return null;   // 물체를 하나도 못 찾음 → 자르지 않음
    return { x: left, y: top, w: right - left + 1, h: bottom - top + 1 };
  }

  // 여백 설정에 맞는 결과를 미리보기와 저장에 쓰도록 바꾸기
  function showResult(it) {
    if (!it.full) return;   // 아직 처리 중이거나 실패한 사진
    const r = cropOn ? it.cropped : it.full;
    if (it.resultUrl) URL.revokeObjectURL(it.resultUrl);
    it.result = r.blob; it.resultUrl = URL.createObjectURL(r.blob); it.w = r.w; it.h = r.h;
    it.card.out.src = it.resultUrl;
    it.card.meta.textContent = `${r.w}×${r.h} · ${fmtKB(r.blob.size)} · ${it.ms}ms` + (it.boxes ? ' · 영역 지정' : '');
  }

  // ---------- 영역 지정 ----------
  const rg = $('#rg'), rgWrap = $('#rgWrap'), rgImg = $('#rgImg');
  let editing = null;   // 지금 편집 중인 사진
  let boxes = [];       // 그린 사각형들. 사진 크기에 대한 0~1 비율 좌표 {x, y, w, h}
  let drawing = null;   // 지금 그리고 있는 사각형

  function addRegionButton(it) {
    if (it.card.region) return;
    const b = document.createElement('button'); b.className = 'btn rg-open'; b.type = 'button'; b.textContent = '영역 지정';
    b.addEventListener('click', () => openEditor(it));
    it.card.acts.insertBefore(b, it.card.save);
    it.card.region = b;
  }

  function openEditor(it) {
    editing = it;
    boxes = (it.boxes || []).map(b => ({ ...b }));   // 전에 그린 사각형이 있으면 다시 보여 주기
    rgImg.src = it.card.orig.src;
    $('#rgReset').disabled = !it.boxes;
    rg.hidden = false; document.body.style.overflow = 'hidden';
    renderBoxes();
  }

  function closeEditor() {
    rg.hidden = true; document.body.style.overflow = '';
    editing = null; drawing = null; boxes = [];
    rgImg.removeAttribute('src');
  }

  function renderBoxes() {
    rgWrap.querySelectorAll('.rg-box').forEach(el => el.remove());
    const list = drawing ? [...boxes, drawing] : boxes;
    list.forEach((b, i) => {
      const el = document.createElement('div'); el.className = 'rg-box';
      el.style.left = b.x * 100 + '%'; el.style.top = b.y * 100 + '%';
      el.style.width = b.w * 100 + '%'; el.style.height = b.h * 100 + '%';
      if (b !== drawing) {
        const del = document.createElement('button'); del.className = 'rg-del'; del.type = 'button';
        del.textContent = '×'; del.setAttribute('aria-label', `사각형 ${i + 1} 지우기`);
        del.addEventListener('pointerdown', e => e.stopPropagation());   // 지우기 버튼을 누를 때 새 사각형이 그려지지 않게
        del.addEventListener('click', () => { boxes.splice(i, 1); renderBoxes(); });
        el.append(del);
      }
      rgWrap.append(el);
    });
    $('#rgCount').textContent = `사각형 ${boxes.length}개`;
    $('#rgApply').disabled = boxes.length === 0;
  }

  // 화면 좌표(clientX, clientY) → 사진 안의 0~1 비율 좌표
  function toImage(e) {
    const r = rgImg.getBoundingClientRect();
    const clamp = v => Math.min(1, Math.max(0, v));   // 사진 밖으로 나가면 경계에 맞춤
    return { x: clamp((e.clientX - r.left) / r.width), y: clamp((e.clientY - r.top) / r.height) };
  }

  // 마우스와 터치를 Pointer Events 하나로 처리
  rgWrap.addEventListener('pointerdown', e => {
    if (e.pointerType === 'mouse' && e.button !== 0) return;   // 왼쪽 버튼만
    e.preventDefault();
    rgWrap.setPointerCapture(e.pointerId);   // 손가락이 사진 밖으로 나가도 계속 이벤트 받기
    const p = toImage(e);
    drawing = { x0: p.x, y0: p.y, x: p.x, y: p.y, w: 0, h: 0 };
  });
  rgWrap.addEventListener('pointermove', e => {
    if (!drawing) return;
    const p = toImage(e);
    drawing.x = Math.min(drawing.x0, p.x); drawing.y = Math.min(drawing.y0, p.y);
    drawing.w = Math.abs(p.x - drawing.x0); drawing.h = Math.abs(p.y - drawing.y0);
    renderBoxes();
  });
  rgWrap.addEventListener('pointerup', () => {
    if (!drawing) return;
    const r = rgImg.getBoundingClientRect();
    // 한 변이 화면에서 20px보다 작으면 실수로 보고 버림
    if (drawing.w * r.width >= 20 && drawing.h * r.height >= 20) {
      boxes.push({ x: drawing.x, y: drawing.y, w: drawing.w, h: drawing.h });
    }
    drawing = null; renderBoxes();
  });
  rgWrap.addEventListener('pointercancel', () => { drawing = null; renderBoxes(); });

  $('#rgClear').addEventListener('click', () => { boxes = []; renderBoxes(); });
  $('#rgCancel').addEventListener('click', closeEditor);
  document.addEventListener('keydown', e => { if (e.key === 'Escape' && !rg.hidden) closeEditor(); });
  $('#rgReset').addEventListener('click', () => {
    const it = editing; if (!it || !it.auto) return;
    it.full = it.auto.full; it.cropped = it.auto.cropped; it.ms = it.auto.ms; it.boxes = null;
    showResult(it); closeEditor(); toast('자동 결과로 되돌렸어요');
  });
  $('#rgApply').addEventListener('click', async () => {
    const it = editing; if (!it || !boxes.length) return;
    const btns = rg.querySelectorAll('button'); btns.forEach(b => b.disabled = true);
    $('#rgApply').textContent = '오리는 중…';
    const labels = ['모델 준비', '사진 읽기', ...boxes.map((_, n) => `사각형 ${n + 1}/${boxes.length}`), 'PNG 만들기'];
    const step = makeSteps($('#rgProgFill'), $('#rgProgText'), labels);
    $('#rgProg').hidden = false; $('#rgCount').hidden = true;
    try {
      const t0 = performance.now();
      await step(0);
      const m = await getModel(it.model);
      await step(1);
      const bmp = await decodeImage(it.file);
      const { W, H } = fitSize(bmp);
      // 0~1 비율 좌표 → 원본 이미지 픽셀 좌표
      const px = boxes.map(b => ({ x: b.x * bmp.width, y: b.y * bmp.height, w: b.w * bmp.width, h: b.h * bmp.height }));
      const mask = await regionMask(m, bmp, px, W, H, n => step(2 + n));
      await step(labels.length - 1);
      await composeResult(it, bmp, mask, W, H);
      bmp.close && bmp.close(); mask.width = mask.height = 0;
      it.ms = Math.round(performance.now() - t0);
      it.boxes = boxes.map(b => ({ ...b }));
      showResult(it); closeEditor();
      toast(`사각형 ${it.boxes.length}개로 다시 오렸어요`);
    } catch (e) {
      console.error(e);
      toast('다시 오리지 못했어요: ' + ((e && e.message) || e));
    } finally {
      btns.forEach(b => b.disabled = false);
      $('#rgApply').textContent = '적용';
      $('#rgProg').hidden = true; $('#rgCount').hidden = false;
      if (!rg.hidden) { renderBoxes(); $('#rgReset').disabled = !(editing && editing.boxes); }
    }
  });

  // createImageBitmap이 못 여는 형식(아이폰 HEIC 등)은 <img>로 디코딩해서 다시 시도
  async function decodeImage(file) {
    try { return await createImageBitmap(file, { imageOrientation: 'from-image' }); } catch (e) {}
    try { return await createImageBitmap(file); } catch (e) {}
    const url = URL.createObjectURL(file);
    try {
      const img = new Image(); img.decoding = 'async'; img.src = url;
      await img.decode();
      const c = document.createElement('canvas'); c.width = img.naturalWidth; c.height = img.naturalHeight;
      c.getContext('2d').drawImage(img, 0, 0);
      return c;   // drawImage에 그대로 쓸 수 있음
    } catch (e) {
      throw new Error('이 이미지 형식은 이 브라우저에서 열 수 없어요. JPG나 PNG로 바꿔서 올려 주세요');
    } finally { URL.revokeObjectURL(url); }
  }

  // ---------- 저장 ----------
  async function whiteBg(it) {
    if (bgMode !== 'white') return it.result;
    const bmp = await createImageBitmap(it.result); const c = document.createElement('canvas'); c.width = bmp.width; c.height = bmp.height;
    const g = c.getContext('2d'); g.fillStyle = '#fff'; g.fillRect(0, 0, c.width, c.height); g.drawImage(bmp, 0, 0);
    return await new Promise(r => c.toBlob(r, 'image/png'));
  }
  function handleSaveErr(e) {
    const code = e && e.code;
    if (code === 'declined') return;
    if (code === 'rate_limited') { toast('저장 창이 이미 열려 있어요. 잠시 후 다시 눌러 주세요'); return; }
    toast('저장하지 못했어요' + (code ? ' (' + code + ')' : ''));
  }
  async function saveOne(it) {
    if (!downloads || !it.result) return;
    try { await downloads.save({ filename: baseName(it.name) + '_nobg.png', data: await whiteBg(it) }); toast('저장했어요'); }
    catch (e) { handleSaveErr(e); }
  }
  $('#saveAll').addEventListener('click', async () => {
    const done = items.filter(i => i.result); if (!downloads || !done.length) return;
    if (done.length === 1) return saveOne(done[0]);
    try {
      const zip = new JSZip(); const used = new Set();
      for (const it of done) {
        let n = baseName(it.name) + '_nobg', k = 2; while (used.has(n)) n = baseName(it.name) + '_nobg_' + (k++); used.add(n);
        zip.file(n + '.png', await whiteBg(it));
      }
      const blob = await zip.generateAsync({ type: 'blob' });
      await downloads.save({ filename: 'nobg_' + done.length + '.zip', data: blob });
      toast('ZIP으로 저장했어요');
    } catch (e) { handleSaveErr(e); }
  });

  // ---------- 예시 카드 ----------
  let exampleCard = null;
  function showExample() {
    const it = { name: 'example_mug.jpg' };
    const c = makeCard(it, 'docs/example.jpg', true);
    c.out.src = 'docs/example_nobg.png'; c.out.hidden = false; c.busy.remove();
    c.meta.textContent = '480×360 · 이렇게 오려져요';
    c.acts.querySelector('button').addEventListener('click', removeExample, { once: true });
    grid.appendChild(c.root); exampleCard = c.root;
  }
  function removeExample() { if (exampleCard) { exampleCard.remove(); exampleCard = null; } }
  showExample();

  // ---------- 저장 기능 사용 가능 여부 ----------
  (async () => {
    if (window.claude && window.claude.use) {
      // Claude 아티팩트 안: 플랫폼의 저장 기능 사용
      try { downloads = await window.claude.use('downloads'); } catch (e) { downloads = null; }
    } else {
      // 일반 웹(GitHub Pages, 로컬 서버 등): 브라우저 기본 다운로드
      downloads = { async save({ filename, data }) {
        const url = URL.createObjectURL(data instanceof Blob ? data : new Blob([data]));
        const a = document.createElement('a'); a.href = url; a.download = filename;
        document.body.appendChild(a); a.click(); a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 4000);
        return { status: 'saved' };
      } };
    }
    $('#dlNote').hidden = !!downloads;
    items.forEach(i => { if (i.result) i.card.save.disabled = !downloads; });
    updateSaveAll();
  })();
})();
