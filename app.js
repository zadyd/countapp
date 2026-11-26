/* 点数 · 网页版
   模型：YOLO11n 单类「孔洞」检测器（ONNX，640 输入）
*/
(function () {
  'use strict';

  const S = 640;
  const $ = (id) => document.getElementById(id);

  const cv = $('cv');
  const ctx = cv.getContext('2d');
  const stage = $('stage');

  let session = null;
  let modelReady = false;
  let imgBitmap = null;
  let imgW = 0, imgH = 0;
  let points = [];
  let confThr = 0.55;
  let showNumbers = true;
  let boxW = 0.02;

  // ---------- 加载状态 ----------
  function setStatus(text, pct, showRetry) {
    const b = $('busy');
    if (text === null) { b.style.display = 'none'; return; }
    b.style.display = 'flex';
    $('busyText').textContent = text;
    const bar = $('bar');
    bar.style.display = (pct === undefined) ? 'none' : 'block';
    if (pct !== undefined) bar.firstElementChild.style.width = Math.round(pct * 100) + '%';
    $('btnRetry').style.display = showRetry ? 'inline-block' : 'none';
  }

  function showError(msg) {
    setStatus('加载失败\n' + msg, undefined, true);
    $('busyText').style.whiteSpace = 'pre-line';
  }

  // 带进度的下载
  async function fetchWithProgress(url, onProgress) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error('下载失败 HTTP ' + resp.status);
    const total = Number(resp.headers.get('content-length')) || 0;
    if (!resp.body || !total) return new Uint8Array(await resp.arrayBuffer());
    const reader = resp.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      // content-length 是压缩后大小，实际读到的是解压后字节，进度封顶 95%
      onProgress(Math.min(0.95, received / total));
    }
    const out = new Uint8Array(received);
    let off = 0;
    for (const c of chunks) { out.set(c, off); off += c.length; }
    return out;
  }

  // ---------- 模型 ----------
  async function initModel() {
    if (typeof ort === 'undefined') throw new Error('推理引擎未加载');
    ort.env.wasm.wasmPaths = './';
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
    ort.env.logLevel = 'error';
    setStatus('加载模型…', 0);
    const bytes = await fetchWithProgress('./model.onnx', (p) => setStatus('加载模型…', p));

    // 手机上图优化极慢（wasm 环境），用 basic 能快一个数量级；本场景精度不受影响
    setStatus('初始化推理引擎…', undefined);
    let slow = 0;
    const slowTimer = setInterval(() => {
      slow += 1;
      if (slow === 1) setStatus('首次加载较慢，请稍候…', undefined);
      if (slow === 3) setStatus('还在初始化（约需 30~60 秒）…', undefined);
      if (slow === 6) setStatus('初始化时间过长，可点重新加载重试', undefined, true);
    }, 10000);

    try {
      session = await ort.InferenceSession.create(bytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'basic'
      });
    } finally {
      clearInterval(slowTimer);
    }
    modelReady = true;
    setStatus(null);
  }

  async function runModel(bitmap) {
    const w = bitmap.width || bitmap.naturalWidth;
    const h = bitmap.height || bitmap.naturalHeight;
    const off = document.createElement('canvas');
    off.width = S; off.height = S;
    const octx = off.getContext('2d', { willReadFrequently: true });
    octx.fillStyle = 'rgb(114,114,114)';
    octx.fillRect(0, 0, S, S);
    const p = letterboxParams(w, h, S);
    octx.imageSmoothingQuality = 'high';
    octx.drawImage(bitmap, p.padX, p.padY, p.dw, p.dh);

    const px = octx.getImageData(0, 0, S, S).data;
    const plane = S * S;
    const data = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i++) {
      const j = i * 4;
      data[i] = px[j] / 255;
      data[plane + i] = px[j + 1] / 255;
      data[2 * plane + i] = px[j + 2] / 255;
    }
    const tensor = new ort.Tensor('float32', data, [1, 3, S, S]);
    const feeds = {};
    feeds[session.inputNames[0]] = tensor;
    const t0 = performance.now();
    const out = await session.run(feeds);
    const ms = performance.now() - t0;
    const res = out[session.outputNames[0]];
    const r = decodeOutput(res.data, res.dims[2], confThr, 0.45, w, h, S);
    return { points: r.points, ms };
  }

  // ---------- 渲染 ----------
  function layout() {
    if (!imgBitmap) return;
    const pad = 8;
    const availW = stage.clientWidth - pad * 2;
    const availH = stage.clientHeight - pad * 2;
    const s = Math.min(availW / imgW, availH / imgH);
    const dw = Math.round(imgW * s), dh = Math.round(imgH * s);
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    cv.width = Math.round(dw * dpr);
    cv.height = Math.round(dh * dpr);
    cv.style.width = dw + 'px';
    cv.style.height = dh + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw(dw, dh);
  }

  function draw(dw, dh) {
    ctx.clearRect(0, 0, dw, dh);
    ctx.drawImage(imgBitmap, 0, 0, dw, dh);
    const r = Math.max(7, (boxW || 0.02) * dw * 1.35);
    const ordered = sortPoints(points);
    for (let i = 0; i < ordered.length; i++) {
      const p = ordered[i];
      const cx = p.x * dw, cy = p.y * dh;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(52,199,89,.18)';
      ctx.fill();
      ctx.strokeStyle = '#34c759';
      ctx.lineWidth = 1.6;
      ctx.stroke();
      if (showNumbers && ordered.length <= 400) {
        ctx.fillStyle = '#fff';
        ctx.font = `600 ${Math.max(8, r * 0.85)}px -apple-system,sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(String(i + 1), cx, cy);
      }
    }
  }

  function sortPoints(pts) {
    const sorted = pts.slice().sort((a, b) => a.y - b.y);
    const out = [];
    let band = [], top = -1;
    const tol = 0.025;
    for (const p of sorted) {
      if (top < 0 || p.y - top <= tol) {
        if (top < 0) top = p.y;
        band.push(p);
      } else {
        out.push(...band.sort((a, b) => a.x - b.x));
        band = [p]; top = p.y;
      }
    }
    out.push(...band.sort((a, b) => a.x - b.x));
    return out;
  }

  function updateBadge() {
    $('num').textContent = points.length;
    $('badge').style.display = imgBitmap ? 'flex' : 'none';
  }

  // ---------- 主流程 ----------
  async function handleBitmap(bitmap) {
    imgBitmap = bitmap;
    imgW = bitmap.width || bitmap.naturalWidth;
    imgH = bitmap.height || bitmap.naturalHeight;
    $('empty').style.display = 'none';
    cv.style.display = 'block';
    setStatus('识别中…', undefined);
    try {
      if (!modelReady) await initModel();
      setStatus('识别中…', undefined);
      const r = await runModel(bitmap);
      points = r.points;
      boxW = points.length ? median(points.map(p => p.w)) : 0.02;
      $('engine').textContent = 'AI';
      setStatus(null);
      $('hint').textContent = points.length
        ? `耗时 ${r.ms.toFixed(0)}ms · 点空白处补漏，点标记删除`
        : '没识别到，试试调整灵敏度';
    } catch (e) {
      console.error(e);
      setStatus(null);
      $('hint').textContent = '识别失败：' + (e.message || e);
      points = [];
    }
    layout();
    updateBadge();
  }

  function median(arr) {
    const a = arr.slice().sort((x, y) => x - y);
    return a[Math.floor(a.length / 2)] || 0.02;
  }

  async function fileToBitmap(file) {
    if (!file) return null;
    try {
      return await createImageBitmap(file);
    } catch (e) {
      // iOS 上 HEIC 等格式 createImageBitmap 可能失败，回退到 <img>
      return await new Promise((res, rej) => {
        const url = URL.createObjectURL(file);
        const im = new Image();
        im.onload = () => { URL.revokeObjectURL(url); res(im); };
        im.onerror = () => { URL.revokeObjectURL(url); rej(new Error('图片无法解码')); };
        im.src = url;
      });
    }
  }

  async function fromFile(file) {
    if (!file) return;
    try {
      const bmp = await fileToBitmap(file);
      if (bmp) await handleBitmap(bmp);
    } catch (e) {
      $('hint').textContent = '图片读取失败：' + (e.message || e);
    }
  }

  // ---------- 交互 ----------
  $('btnCam').addEventListener('click', () => $('fileCam').click());
  $('btnLib').addEventListener('click', () => $('fileLib').click());
  $('fileCam').addEventListener('change', (e) => fromFile(e.target.files[0]));
  $('fileLib').addEventListener('change', (e) => fromFile(e.target.files[0]));

  cv.addEventListener('click', (e) => {
    if (!imgBitmap) return;
    const r = cv.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width;
    const y = (e.clientY - r.top) / r.height;
    const hitR = Math.max(0.022, (boxW || 0.02) * 1.1);
    let best = -1, bestD = 1e9;
    points.forEach((p, i) => {
      const d = Math.hypot(p.x - x, p.y - y);
      if (d < bestD) { bestD = d; best = i; }
    });
    if (best >= 0 && bestD <= hitR) points.splice(best, 1);
    else points.push({ x, y, w: boxW || 0.02 });
    layout();
    updateBadge();
  });

  $('btnNum').addEventListener('click', () => {
    showNumbers = !showNumbers;
    $('btnNum').classList.toggle('off', !showNumbers);
    if (imgBitmap) layout();
  });

  $('btnAdj').addEventListener('click', () => { $('panel').style.display = 'block'; });
  $('btnClose').addEventListener('click', () => { $('panel').style.display = 'none'; });
  $('conf').addEventListener('input', (e) => {
    confThr = parseFloat(e.target.value);
    $('confVal').textContent = confThr.toFixed(2);
  });
  $('conf').addEventListener('change', async () => {
    if (imgBitmap) await handleBitmap(imgBitmap);
  });

  $('btnRetry').addEventListener('click', () => {
    location.replace(location.pathname + '?r=' + Date.now());
  });

  window.addEventListener('resize', () => { if (imgBitmap) layout(); });

  // ---------- 启动 ----------
  (async function boot() {
    try {
      await initModel();
    } catch (e) {
      console.error('模型加载失败', e);
      showError(e.message || String(e));
      return;
    }

    const q = new URLSearchParams(location.search);
    if (q.has('demo')) {
      try {
        const r = await fetch('./demo.jpg');
        const blob = await r.blob();
        await handleBitmap(await createImageBitmap(blob));
      } catch (e) { console.error(e); }
    }

    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('./sw.js').catch(() => {});
    }
  })();
})();
