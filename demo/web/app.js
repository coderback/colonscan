(function () {
  'use strict';
  var results = window.SAVED_RESULTS;
  var $ = function (id) { return document.getElementById(id); };
  var thumbs = $('thumbs');

  function pick(i) {
    var r = results[i];
    var base = $('img-base'), pred = $('img-pred'), truth = $('img-truth');
    [base, pred, truth].forEach(function (el) { el.width = r.width; el.height = r.height; });
    base.src = 'samples/' + r.id + '.jpg';
    base.alt = 'Colonoscopy image ' + (i + 1) + ' from the public Kvasir-SEG dataset, showing a polyp';
    pred.src = 'samples/' + r.id + '_pred.png';
    truth.src = 'samples/' + r.id + '_truth.png';
    $('s-iou').textContent = r.iou.toFixed(3);
    $('s-dice').textContent = r.dice.toFixed(3);
    $('s-area').textContent = r.polyp_area_pct.toFixed(1) + '%';
    Array.prototype.forEach.call(thumbs.children, function (b, j) {
      b.setAttribute('aria-pressed', j === i ? 'true' : 'false');
    });
    window.COLONO_CURRENT = i;
    document.dispatchEvent(new CustomEvent('colono:sample'));
  }

  results.forEach(function (r, i) {
    var b = document.createElement('button');
    b.type = 'button';
    b.className = 'thumb';
    b.setAttribute('aria-label', 'Show sample ' + (i + 1));
    var img = document.createElement('img');
    img.src = 'samples/' + r.id + '.jpg';
    img.alt = '';
    img.loading = 'lazy';
    b.appendChild(img);
    b.onclick = function () { pick(i); };
    thumbs.appendChild(b);
  });

  function syncLayers() {
    $('img-pred').hidden = !$('t-pred').checked;
    $('img-truth').hidden = !$('t-truth').checked;
  }
  $('t-pred').onchange = syncLayers;
  $('t-truth').onchange = syncLayers;
  syncLayers();
  pick(0);
})();
