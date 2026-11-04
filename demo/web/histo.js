/* Shows the patch classifier's saved results, and runs it in the visitor's browser on the
   eight public test patches only. Works apart from the polyp model, with its own worker. */
(function () {
  'use strict';

  var SIZE = 224;
  var MEAN = [0.485, 0.456, 0.406];
  var STD = [0.229, 0.224, 0.225];
  var LABELS = ['Benign tissue', 'Adenocarcinoma'];

  var $ = function (id) { return document.getElementById(id); };
  var samples = window.HISTO_RESULTS;
  var summary = window.HISTO_SUMMARY;
  var base = $('h-base');
  var cam = $('h-cam');
  var showCam = $('h-show-cam');
  var thumbs = $('h-thumbs');
  var runBtn = $('h-run');
  var cancelBtn = $('h-cancel');
  var progress = $('h-progress');
  var status = $('h-status');

  var current = 0;
  var worker = null;
  var ready = false;
  var busy = false;
  var threads = 1;
  var info = null;
  var calls = {};
  var nextCall = 1;
  var live = {};

  function pct(p) {
    if (p > 0.999) return 'over 99.9%';
    if (p < 0.001) return 'under 0.1%';
    return (p * 100).toFixed(1) + '%';
  }

  function verdict(probs) {
    var i = probs[1] > probs[0] ? 1 : 0;
    return { label: LABELS[i], confidence: pct(probs[i]) };
  }

  function mb(bytes) {
    return Math.round(bytes / 1e6);
  }

  function showLive() {
    var r = live[current];
    $('h-live-scores').hidden = !r;
    $('h-live-detail').hidden = !r;
    if (!r) return;
    var v = verdict(r.probs);
    $('h-l-label').textContent = v.label;
    $('h-l-conf').textContent = v.confidence;
    $('h-l-ms').textContent = Math.round(r.ms) + ' ms';
    $('h-live-detail').textContent = 'Measured on this device, for the model run alone: CPU through WebAssembly, ' +
      r.threads + (r.threads === 1 ? ' thread' : ' threads') + '. Timing varies with your device and what else it is doing.';
  }

  function setRunLabel() {
    if (ready) runBtn.textContent = live[current] ? 'Run again on this patch' : 'Run on this patch';
    else runBtn.textContent = 'Run on this patch' + (info ? ' (about ' + mb(info.bytes) + ' MB download)' : '');
  }

  function pick(i) {
    current = i;
    var r = samples[i];
    base.src = 'samples/' + r.id + '.jpg';
    base.alt = 'Colon tissue patch ' + (i + 1) + ' from the public LC25000 dataset, labelled ' + LABELS[r.truth].toLowerCase();
    cam.src = 'samples/' + r.id + '_cam.jpg';
    var v = verdict(r.probs);
    $('h-truth').textContent = LABELS[r.truth];
    $('h-s-label').textContent = v.label;
    $('h-s-conf').textContent = v.confidence;
    Array.prototype.forEach.call(thumbs.children, function (b, j) {
      b.setAttribute('aria-pressed', j === i ? 'true' : 'false');
    });
    status.textContent = '';
    status.classList.remove('is-error');
    showLive();
    setRunLabel();
  }

  samples.forEach(function (r, i) {
    var b = document.createElement('button');
    b.type = 'button';
    b.className = 'thumb';
    b.setAttribute('aria-label', 'Show patch ' + (i + 1) + ', labelled ' + LABELS[r.truth].toLowerCase());
    var img = document.createElement('img');
    img.src = 'samples/' + r.id + '_224.png';
    img.alt = '';
    img.loading = 'lazy';
    b.appendChild(img);
    b.addEventListener('click', function () { if (!busy) pick(i); });
    thumbs.appendChild(b);
  });

  showCam.addEventListener('change', function () { cam.hidden = !showCam.checked; });

  /* The page reads the 224 pixel PNG, which the export wrote with the same resize as the
     PyTorch code, so no browser resize filter is involved. */
  async function tensorFor(r) {
    var res = await fetch('samples/' + r.id + '_224.png');
    if (!res.ok) throw new Error('the patch could not be loaded');
    var bitmap = await createImageBitmap(await res.blob(), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
    var canvas = document.createElement('canvas');
    canvas.width = canvas.height = SIZE;
    var ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0);
    var rgba = ctx.getImageData(0, 0, SIZE, SIZE).data;
    var out = new Float32Array(3 * SIZE * SIZE);
    for (var p = 0; p < SIZE * SIZE; p++) {
      for (var c = 0; c < 3; c++) {
        out[c * SIZE * SIZE + p] = (rgba[p * 4 + c] / 255 - MEAN[c]) / STD[c];
      }
    }
    return out;
  }

  function stopWorker(reason) {
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
      var pending = calls[m.id];
      if (!pending) return;
      if (m.done) {
        delete calls[m.id];
        if (m.error) {
          var err = new Error(m.error);
          err.stage = m.stage;
          pending.reject(err);
        } else {
          pending.resolve(m.result);
        }
      } else if (pending.onUpdate) {
        pending.onUpdate(m);
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
      worker.postMessage(payload, transfer || []);
    });
  }

  async function loadOnce(wanted) {
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
        status.textContent = 'Downloading the model: ' + (m.progress / 1e6).toFixed(1) + ' of ' + (m.total / 1e6).toFixed(1) + ' MB.';
      }
    });
    threads = result.threads;
    ready = true;
  }

  async function loadModel() {
    var wanted = window.crossOriginIsolated ? Math.min(4, Math.ceil((navigator.hardwareConcurrency || 2) / 2)) : 1;
    progress.hidden = false;
    try {
      await loadOnce(wanted);
    } catch (err) {
      if (err.cancelled || err.stage !== 'session' || wanted === 1) throw err;
      // Some devices can't start threaded WebAssembly. One thread is slower but works.
      stopWorker(err);
      await loadOnce(1);
    }
  }

  async function run() {
    busy = true;
    runBtn.disabled = true;
    cancelBtn.hidden = false;
    status.classList.remove('is-error');
    var i = current;
    try {
      if (!ready) await loadModel();
      status.textContent = 'Classifying the patch on your device.';
      progress.hidden = true;
      var tensor = await tensorFor(samples[i]);
      var result = await call('run', { tensor: tensor, size: SIZE }, [tensor.buffer]);
      live[i] = { probs: Array.from(result.probs), ms: result.ms, threads: threads };
      status.textContent = 'Done. This result was computed on your device.';
      showLive();
    } catch (err) {
      progress.hidden = true;
      status.textContent = err.cancelled ? 'Cancelled.' : 'The live run failed: ' + err.message + '. The saved result above is unaffected.';
      status.classList.toggle('is-error', !err.cancelled);
    }
    busy = false;
    runBtn.disabled = false;
    cancelBtn.hidden = true;
    setRunLabel();
  }

  runBtn.addEventListener('click', run);
  cancelBtn.addEventListener('click', function () {
    var err = new Error('cancelled');
    err.cancelled = true;
    stopWorker(err);
  });

  $('h-total').textContent = summary.test_correct + ' of ' + summary.test_total;
  pick(0);
  fetch('models/histo.json').then(function (res) { return res.json(); }).then(function (json) {
    info = json;
    setRunLabel();
  }).catch(function () {
    runBtn.disabled = true;
    status.textContent = 'The live run is not available right now. The saved results still work.';
  });
})();
