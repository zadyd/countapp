// 纯函数核心：letterbox 参数、输出解码、NMS。浏览器和 Node 测试共用。

/** 计算 letterbox 参数：把 WxH 等比缩放进 SxS 并居中留边 */
function letterboxParams(W, H, S) {
  const scale = S / Math.max(W, H);
  const dw = Math.round(W * scale);
  const dh = Math.round(H * scale);
  return { scale, dw, dh, padX: Math.round((S - dw) / 2), padY: Math.round((S - dh) / 2) };
}

/** 贪心 NMS，boxes 为 [x1,y1,x2,y2] */
function nms(boxes, scores, iouThr) {
  const order = scores.map((s, i) => i).sort((a, b) => scores[b] - scores[a]);
  const keep = [];
  const suppressed = new Uint8Array(boxes.length);
  for (let k = 0; k < order.length; k++) {
    const i = order[k];
    if (suppressed[i]) continue;
    keep.push(i);
    const bi = boxes[i];
    const ai = (bi[2] - bi[0]) * (bi[3] - bi[1]);
    for (let m = k + 1; m < order.length; m++) {
      const j = order[m];
      if (suppressed[j]) continue;
      const bj = boxes[j];
      const xx1 = Math.max(bi[0], bj[0]);
      const yy1 = Math.max(bi[1], bj[1]);
      const xx2 = Math.min(bi[2], bj[2]);
      const yy2 = Math.min(bi[3], bj[3]);
      const inter = Math.max(0, xx2 - xx1) * Math.max(0, yy2 - yy1);
      const aj = (bj[2] - bj[0]) * (bj[3] - bj[1]);
      if (inter / (ai + aj - inter + 1e-9) > iouThr) suppressed[j] = 1;
    }
  }
  return keep;
}

/**
 * 解码 YOLO 输出。
 * data: Float32Array，形状 [1, 4+nc, A]（本项目 nc=1），坐标是 640 输入下的像素值
 * 返回 { points: [{x,y,w}], count }，x/y 为归一化到原图的坐标
 */
function decodeOutput(data, numAnchors, confThr, iouThr, W, H, S) {
  const { scale, padX, padY } = letterboxParams(W, H, S);
  const boxes = [];
  const scores = [];
  for (let a = 0; a < numAnchors; a++) {
    const conf = data[4 * numAnchors + a];
    if (conf < confThr) continue;
    const cx = data[0 * numAnchors + a];
    const cy = data[1 * numAnchors + a];
    const w = data[2 * numAnchors + a];
    const h = data[3 * numAnchors + a];
    boxes.push([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]);
    scores.push(conf);
  }
  if (!boxes.length) return { points: [], count: 0 };
  const keep = nms(boxes, scores, iouThr);
  const points = [];
  for (const i of keep) {
    const b = boxes[i];
    const cx = (b[0] + b[2]) / 2;
    const cy = (b[1] + b[3]) / 2;
    const px = (cx - padX) / scale;
    const py = (cy - padY) / scale;
    if (px < 0 || py < 0 || px > W || py > H) continue;
    points.push({ x: px / W, y: py / H, w: (b[2] - b[0]) / scale / W, conf: scores[i] });
  }
  return { points, count: points.length };
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { letterboxParams, nms, decodeOutput };
}
