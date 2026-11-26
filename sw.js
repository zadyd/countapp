// 缓存策略：
//   大文件（模型 / wasm）→ 缓存优先（省流量，首次下载后永久可用）
//   其他（html / js / json）→ 网络优先（保证更新一定生效，失败才回退缓存）
const CACHE = 'count-v5';
const BIG = /\.(onnx|wasm)$/;
const PRECACHE = ['./model.onnx', './ort-wasm-simd.wasm'];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE)
      .then((c) => Promise.all(PRECACHE.map((u) => c.add(u).catch(() => {}))))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  if (e.request.method !== 'GET') return;
  const url = new URL(e.request.url);
  if (url.origin !== location.origin) return;

  if (BIG.test(url.pathname)) {
    // 大文件：缓存优先
    e.respondWith(
      caches.match(e.request).then((hit) => hit || fetch(e.request).then((resp) => {
        const copy = resp.clone();
        caches.open(CACHE).then((c) => c.put(e.request, copy)).catch(() => {});
        return resp;
      }))
    );
    return;
  }

  // 其他：网络优先，失败回退缓存
  e.respondWith(
    fetch(e.request).then((resp) => {
      const copy = resp.clone();
      caches.open(CACHE).then((c) => c.put(e.request, copy)).catch(() => {});
      return resp;
    }).catch(() => caches.match(e.request).then((hit) => hit || Response.error()))
  );
});
