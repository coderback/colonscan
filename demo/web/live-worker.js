/* Downloads and runs the model off the main thread, so the page stays responsive. */
'use strict';
importScripts('vendor/ort-1.30.0/ort.wasm.min.js');

var session = null;

function fail(stage, err) {
  var e = new Error((err && err.message) || String(err));
  e.stage = stage;
  return e;
}

async function download(m, id) {
  var res;
  try {
    res = await fetch(m.url);
  } catch (err) {
    throw fail('download', 'the model could not be downloaded');
  }
  if (!res.ok || !res.body) throw fail('download', 'the model download returned HTTP ' + res.status);
  var bytes = new Uint8Array(m.bytes);
  var reader = res.body.getReader();
  var got = 0;
  var last = 0;
  for (;;) {
    var part;
    try {
      part = await reader.read();
    } catch (err) {
      throw fail('download', 'the model download was interrupted');
    }
    if (part.done) break;
    if (got + part.value.length > bytes.length) throw fail('download', 'the model file has an unexpected size');
    bytes.set(part.value, got);
    got += part.value.length;
    var now = performance.now();
    if (now - last > 100) {
      last = now;
      postMessage({ id: id, progress: got, total: bytes.length });
    }
  }
  if (got !== bytes.length) throw fail('download', 'the model download was incomplete');
  postMessage({ id: id, progress: got, total: bytes.length });
  if (self.crypto && crypto.subtle) {
    var digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
    var hex = Array.prototype.map.call(digest, function (b) { return b.toString(16).padStart(2, '0'); }).join('');
    if (hex !== m.sha256) throw fail('download', 'the model file failed its checksum');
  }
  return bytes;
}

var handlers = {
  load: async function (m, id) {
    var bytes = await download(m, id);
    postMessage({ id: id, phase: 'session' });
    ort.env.wasm.wasmPaths = new URL('vendor/ort-1.30.0/', self.location.href).href;
    ort.env.wasm.numThreads = m.threads;
    ort.env.logLevel = 'error';
    try {
      session = await ort.InferenceSession.create(bytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
    } catch (err) {
      throw fail('session', err);
    }
    return { threads: ort.env.wasm.numThreads, version: ort.env.versions.web };
  },
  run: async function (m) {
    var input = new ort.Tensor('float32', m.tensor, [1, 3, m.size, m.size]);
    var started = performance.now();
    var out;
    try {
      out = await session.run({ image: input });
    } catch (err) {
      throw fail('run', err);
    }
    var ms = performance.now() - started;
    var probs = new Float32Array(out.probability.data);
    return { probs: probs, ms: ms, transfer: [probs.buffer] };
  },
};

self.onmessage = function (e) {
  var m = e.data;
  handlers[m.type](m, m.id).then(
    function (result) {
      var transfer = result.transfer || [];
      delete result.transfer;
      postMessage({ id: m.id, done: true, result: result }, transfer);
    },
    function (err) {
      postMessage({ id: m.id, done: true, error: err.message, stage: err.stage || m.type });
    }
  );
};
