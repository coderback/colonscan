/* Runs the polyp model in the visitor's browser, on the eight public samples only.
   Nothing here touches the saved results, so they stay usable if this fails. */
(function () {
  'use strict';

  var SIZE = 256;
  var MEAN = [0.485, 0.456, 0.406];
  var STD = [0.229, 0.224, 0.225];
  var WATCHDOG_MS = 5 * 60 * 1000;

  var $ = function (id) { return document.getElementById(id); };
  var samples = window.SAVED_RESULTS;
  var runBtn = $('live-run');
  var cancelBtn = $('live-cancel');
  var progress = $('live-progress');
  var status = $('live-status');
  var liveScores = $('live-scores');
  var detail = $('live-detail');
  var layer = $('live-layer');
  var tPred = $('t-pred');
  var tLive = $('t-live');
  var tLiveRow = $('t-live-row');

  var worker = null;
  var ready = false;
  var busy = false;
  var threads = 1;
  var calls = {};
  var nextCall = 1;
  var watchdog = null;
  var results = {};

  function current() {
    return samples[window.COLONO_CURRENT || 0];
  }

  function mb(bytes) {
    return (bytes / 1e6).toFixed(1);
  }

  /* --- Preprocessing: the same result as demo/api/app.py, without a server ---
     A canvas resize would use a different filter, so this does the resize itself.
     It matches cv2.resize(INTER_LINEAR) exactly, including its fixed-point rounding. */
  function axis(src) {
    var scale = src / SIZE;
    var from = new Int32Array(SIZE);
    var to = new Int32Array(SIZE);
    var a0 = new Int32Array(SIZE);
    var a1 = new Int32Array(SIZE);
    for (var d = 0; d < SIZE; d++) {
      var f = Math.fround((d + 0.5) * scale - 0.5);
      var s = Math.floor(f);
      f = Math.fround(f - s);
      if (s < 0) { f = 0; s = 0; }
      if (s >= src - 1) { f = 0; s = src - 1; }
      from[d] = s;
      to[d] = Math.min(s + 1, src - 1);
      a0[d] = Math.round(Math.fround(1 - f) * 2048);
      a1[d] = Math.round(f * 2048);
    }
    return { from: from, to: to, a0: a0, a1: a1 };
  }

  function resizeRgb(rgba, w, h) {
    var xs = axis(w);
    var ys = axis(h);
    var out = new Uint8Array(SIZE * SIZE * 3);
    var rows = {};
    function row(y) {
      if (rows[y]) return rows[y];
      var line = new Int32Array(SIZE * 3);
      for (var x = 0; x < SIZE; x++) {
        var p = (y * w + xs.from[x]) * 4;
        var q = (y * w + xs.to[x]) * 4;
        for (var c = 0; c < 3; c++) line[x * 3 + c] = rgba[p + c] * xs.a0[x] + rgba[q + c] * xs.a1[x];
      }
      rows[y] = line;
      return line;
    }
    for (var dy = 0; dy < SIZE; dy++) {
      var r0 = row(ys.from[dy]);
      var r1 = row(ys.to[dy]);
      var b0 = ys.a0[dy];
      var b1 = ys.a1[dy];
      var base = dy * SIZE * 3;
      for (var i = 0; i < SIZE * 3; i++) {
        var v = (((r0[i] >> 4) * b0) >> 16) + (((r1[i] >> 4) * b1) >> 16);
        v = (v + 2) >> 2;
        out[base + i] = v < 0 ? 0 : v > 255 ? 255 : v;
      }
    }
    return out;
  }

  function normalise(rgb) {
    var n = SIZE * SIZE;
    var tensor = new Float32Array(n * 3);
    for (var c = 0; c < 3; c++) {
      for (var i = 0; i < n; i++) {
        tensor[c * n + i] = (rgb[i * 3 + c] / 255 - MEAN[c]) / STD[c];
      }
    }
    return tensor;
  }

  async function decode(url) {
    var res = await fetch(url);
    if (!res.ok) throw new Error('a sample file returned HTTP ' + res.status);
    var bitmap = await createImageBitmap(await res.blob(), {
      colorSpaceConversion: 'none',
      premultiplyAlpha: 'none',
    });
    var canvas = document.createElement('canvas');
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    var ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0);
    var data = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
    if (bitmap.close) bitmap.close();
    return data;
  }

  async function preprocess(id) {
    var img = await decode('samples/' + id + '.jpg');
    var rgb = resizeRgb(img.data, img.width, img.height);
    return { rgb: rgb, tensor: normalise(rgb), width: img.width, height: img.height };
  }

  async function loadTruth(id) {
    var img = await decode('samples/' + id + '_truth256.png');
    var truth = new Uint8Array(SIZE * SIZE);
    for (var i = 0; i < truth.length; i++) truth[i] = img.data[i * 4] > 127 ? 1 : 0;
    return truth;
  }

  function score(pred, truth) {
    var inter = 0, union = 0, total = 0, area = 0;
    for (var i = 0; i < pred.length; i++) {
      var p = pred[i], t = truth[i];
      if (p && t) inter++;
      if (p || t) union++;
      total += p + t;
      area += p;
    }
    return {
      iou: union ? inter / union : null,
      dice: total ? (2 * inter) / total : null,
      area: (area / pred.length) * 100,
    };
  }

  /* --- Worker --- */
  function armWatchdog() {
    clearTimeout(watchdog);
    watchdog = setTimeout(function () {
      stopWorker(new Error('the device stopped responding'));
    }, WATCHDOG_MS);
  }

  function stopWorker(reason) {
    clearTimeout(watchdog);
    if (worker) worker.terminate();
    worker = null;
    ready = false;
    Object.keys(calls).forEach(function (id) {
      calls[id].reject(reason);
      delete calls[id];
    });
  }

  function startWorker() {
    worker = new Worker('live-worker.js');
    worker.onmessage = function (e) {
      var m = e.data;
      var call = calls[m.id];
      if (!call) return;
      armWatchdog();
      if (m.done) {
        clearTimeout(watchdog);
        delete calls[m.id];
        if (m.error) {
          var err = new Error(m.error);
          err.stage = m.stage;
          call.reject(err);
        } else {
          call.resolve(m.result);
        }
      } else if (call.onUpdate) {
        call.onUpdate(m);
      }
    };
    worker.onerror = function (e) {
      if (e.preventDefault) e.preventDefault();
      var err = new Error('the runtime could not start');
      err.stage = 'session';
      stopWorker(err);
    };
  }

  function call(type, payload, transfer, onUpdate) {
    return new Promise(function (resolve, reject) {
      var id = nextCall++;
      calls[id] = { resolve: resolve, reject: reject, onUpdate: onUpdate };
      payload.id = id;
      payload.type = type;
      armWatchdog();
      worker.postMessage(payload, transfer || []);
    });
  }

  function preferredThreads() {
    if (!window.crossOriginIsolated) return 1;
    return Math.min(4, Math.ceil((navigator.hardwareConcurrency || 2) / 2));
  }

  async function loadOnce(info, wanted) {
    startWorker();
    var result = await call('load', {
      url: 'models/' + info.file,
      bytes: info.bytes,
      sha256: info.sha256,
      threads: wanted,
    }, [], function (m) {
      if (m.phase === 'session') {
        progress.removeAttribute('value');
        status.textContent = 'Starting the model on your device.';
      } else {
        progress.max = m.total;
        progress.value = m.progress;
        status.textContent = 'Downloading the model: ' + mb(m.progress) + ' of ' + mb(m.total) + ' MB.';
      }
    });
    threads = result.threads;
    ready = true;
  }

  async function loadModel() {
    var res = await fetch('models/model.json');
    if (!res.ok) throw new Error('the model details returned HTTP ' + res.status);
    var info = await res.json();
    status.textContent = 'Downloading the model: 0.0 of ' + mb(info.bytes) + ' MB.';
    progress.hidden = false;
    progress.max = info.bytes;
    progress.value = 0;
    var wanted = preferredThreads();
    try {
      await loadOnce(info, wanted);
    } catch (err) {
      if (err.cancelled || err.stage !== 'session' || wanted === 1) throw err;
      // Some devices can't start threaded WebAssembly. One thread is slower but works.
      stopWorker(err);
      await loadOnce(info, 1);
    }
    progress.hidden = true;
  }

  /* --- Drawing and display --- */
  function drawMask(result, width, height) {
    layer.width = width;
    layer.height = height;
    var ctx = layer.getContext('2d');
    var image = ctx.createImageData(width, height);
    var inside = new Uint8Array(width * height);
    var x, y;
    for (y = 0; y < height; y++) {
      var my = Math.floor((y * SIZE) / height);
      for (x = 0; x < width; x++) {
        inside[y * width + x] = result.mask[my * SIZE + Math.floor((x * SIZE) / width)];
      }
    }
    var R = 2;
    for (y = 0; y < height; y++) {
      for (x = 0; x < width; x++) {
        if (!inside[y * width + x]) continue;
        var edge = x < R || y < R || x >= width - R || y >= height - R;
        for (var dy = -R; !edge && dy <= R; dy++) {
          for (var dx = -R; !edge && dx <= R; dx++) {
            if (!inside[(y + dy) * width + x + dx]) edge = true;
          }
        }
        var o = (y * width + x) * 4;
        image.data[o] = edge ? 26 : 124;
        image.data[o + 1] = edge ? 214 : 255;
        image.data[o + 2] = edge ? 60 : 107;
        image.data[o + 3] = edge ? 255 : 120;
      }
    }
    ctx.putImageData(image, 0, 0);
  }

  function setCheck(box, value) {
    box.checked = value;
    box.dispatchEvent(new Event('change'));
  }

  function formatMs(ms) {
    return Math.round(ms).toLocaleString('en-GB') + ' ms';
  }

  function render() {
    var sample = current();
    var r = results[sample.id];
    tLiveRow.hidden = !r;
    liveScores.hidden = !r;
    detail.hidden = !r;
    if (r) {
      drawMask(r, sample.width, sample.height);
      $('l-iou').textContent = r.iou.toFixed(3);
      $('l-dice').textContent = r.dice.toFixed(3);
      $('l-area').textContent = r.area.toFixed(1) + '%';
      $('l-ms').textContent = formatMs(r.ms);
      detail.textContent = 'Measured on this device, for the model run alone: CPU through WebAssembly, ' +
        r.threads + (r.threads === 1 ? ' thread' : ' threads') + '. Timing varies with your device and what else it is doing.';
      setCheck(tLive, true);
      setCheck(tPred, false);
    } else {
      setCheck(tLive, false);
      setCheck(tPred, true);
    }
    if (!busy) {
      runBtn.textContent = r ? 'Run again on this image' : ready ? 'Run on this image' : 'Run on this image (about 44 MB download)';
    }
  }

  tLive.addEventListener('change', function () {
    layer.hidden = !tLive.checked;
  });

  function setBusy(value) {
    busy = value;
    runBtn.disabled = value;
    cancelBtn.hidden = !value;
    if (value) {
      runBtn.textContent = 'Working';
      cancelBtn.focus();
    } else {
      progress.hidden = true;
    }
  }

  async function run() {
    if (busy) return;
    var sample = current();
    setBusy(true);
    status.classList.remove('is-error');
    try {
      if (!ready) await loadModel();
      status.textContent = 'Running the model on your device.';
      var input = await preprocess(sample.id);
      var truth = await loadTruth(sample.id);
      var out = await call('run', { tensor: input.tensor, size: SIZE }, [input.tensor.buffer]);
      var mask = new Uint8Array(out.probs.length);
      for (var i = 0; i < mask.length; i++) mask[i] = out.probs[i] > 0.5 ? 1 : 0;
      var s = score(mask, truth);
      results[sample.id] = {
        id: sample.id,
        mask: mask,
        iou: s.iou,
        dice: s.dice,
        area: s.area,
        ms: out.ms,
        threads: threads,
      };
      status.textContent = current().id === sample.id
        ? 'Done. The green mask was computed on your device.'
        : 'Done. Image ' + (samples.indexOf(sample) + 1) + ' has a live result. Choose it to see the mask.';
    } catch (err) {
      progress.hidden = true;
      if (err.cancelled) {
        status.textContent = 'Stopped. The saved results are still shown.';
      } else {
        stopWorker(err);
        status.classList.add('is-error');
        status.textContent = 'The live model could not run here: ' + err.message + '. The saved results are still shown, and you can try again.';
        window.ColonoLive.lastError = err.message;
      }
    } finally {
      var wasFocused = document.activeElement === cancelBtn;
      setBusy(false);
      render();
      if (wasFocused || document.activeElement === document.body) runBtn.focus();
    }
  }

  function unsupported() {
    return !(window.WebAssembly && window.Worker && window.fetch && window.createImageBitmap && window.ReadableStream);
  }

  document.addEventListener('colono:sample', function () {
    status.classList.remove('is-error');
    if (!busy) status.textContent = '';
    render();
  });

  cancelBtn.addEventListener('click', function () {
    var err = new Error('cancelled');
    err.cancelled = true;
    stopWorker(err);
  });

  runBtn.addEventListener('click', run);

  // Read-only view of the results, so the numbers on the page can be checked.
  window.ColonoLive = { results: results, preprocess: preprocess, lastError: null };

  if (unsupported()) {
    runBtn.disabled = true;
    runBtn.textContent = 'Not available in this browser';
    status.textContent = 'This browser cannot run the live model. The saved results are still shown.';
    return;
  }
  render();
})();
