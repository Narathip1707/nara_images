/* Nara Images — DIP exam prep.
   Everything on screen is built from /api/functions, so adding a function to
   backend/registry.py is enough — no changes needed here. */

const $ = (id) => document.getElementById(id);

const S = {
  catalog: [],
  chapters: {},
  byId: {},
  images: [],
  image: null,
  ops: [],            // [{fn, params}] — the pipeline
  activeIdx: -1,      // which op's params/code/steps are showing
  last: null,
  variant: 'loop',
  reqSeq: 0,          // guards against out-of-order responses while dragging
};

/* ───────────────────────── boot ───────────────────────── */
async function boot() {
  const [fns, imgs] = await Promise.all([
    fetch('/api/functions').then(r => r.json()),
    fetch('/api/images').then(r => r.json()),
  ]);
  S.catalog = fns.functions;
  S.chapters = fns.chapters;
  S.catalog.forEach(f => S.byId[f.id] = f);
  S.images = imgs.images;

  renderImages();
  renderFunctions();
  wire();
  restorePanels();

  const first = S.images.find(i => i.name === 'lena_color_256.png') || S.images[0];
  if (first) selectImage(first.name);
  else noImages();
}

function noImages() {
  $('compare').classList.remove('on');
  $('stageMsg').className = 'stage-msg empty';
  $('stageMsg').innerHTML =
    '<b>ลากรูปอะไรก็ได้มาวางตรงนี้</b> เพื่อเริ่มใช้งาน<br>' +
    'หรือกดปุ่ม <b>อัปโหลด</b> มุมซ้ายบน<br><br>' +
    '<span class="dim">เรียนวิชานี้อยู่และมีโฟลเดอร์วิชาในเครื่อง? ' +
    'รัน <code>get_images.bat</code> เพื่อดึงรูปทดสอบ 34 รูปมาใส่ให้ครบ</span>';
}

/* ───────────────────────── images ───────────────────────── */
function renderImages() {
  const q = $('imageSearch').value.trim().toLowerCase();
  const list = S.images.filter(i => !q || i.name.toLowerCase().includes(q));
  $('imageList').innerHTML = list.map(thumbHTML).join('') ||
    '<p style="color:var(--dim);font-size:12px">ไม่พบรูปที่ค้นหา</p>';
  bindThumbs($('imageList'));
  renderSuggested();
}

function thumbHTML(i) {
  const label = i.name.replace('_uploads/', '');
  return `<div class="thumb ${S.image === i.name ? 'sel' : ''}" data-name="${esc(i.name)}"
            title="${esc(label)} — ${i.w}×${i.h} ${i.color ? 'สี' : 'เทา'}">
            <img src="/api/thumb/${encodeURI(i.name)}" loading="lazy" alt="">
            <span>${esc(label)}</span></div>`;
}

function bindThumbs(root) {
  root.querySelectorAll('.thumb').forEach(t =>
    t.onclick = () => selectImage(t.dataset.name));
}

function renderSuggested() {
  const spec = activeSpec();
  const names = spec ? (spec.good_for || []).filter(n => S.images.some(i => i.name === n)) : [];
  if (!names.length) { $('suggested').classList.add('hidden'); return; }
  $('suggested').classList.remove('hidden');
  $('suggestedList').innerHTML = names
    .map(n => thumbHTML(S.images.find(i => i.name === n))).join('');
  bindThumbs($('suggestedList'));
}

function selectImage(name) {
  S.image = name;
  renderImages();
  run();
}

/* ───────────────────────── functions ───────────────────────── */
function renderFunctions() {
  const q = $('fnSearch').value.trim().toLowerCase();
  const starOnly = $('starOnly').checked;
  const out = [];
  for (const [ch, title] of Object.entries(S.chapters)) {
    const fns = S.catalog.filter(f =>
      String(f.chapter) === ch &&
      (!starOnly || f.stars === 5) &&
      (!q || f.name.toLowerCase().includes(q) || (f.why || '').toLowerCase().includes(q)));
    if (!fns.length) continue;
    out.push(`<div class="chapter"><h3 class="rule">${esc(title)}</h3>` + fns.map(f =>
      `<button class="fn ${isActive(f.id) ? 'sel' : ''}" data-fn="${f.id}" title="${esc(f.why || '')}">
         <span class="nm">${esc(f.name)}</span>
         <span class="st">${'★'.repeat(f.stars)}</span>
       </button>`).join('') + '</div>');
  }
  $('fnList').innerHTML = out.join('') ||
    '<p style="color:var(--dim);font-size:12px">ไม่พบฟังก์ชันที่ค้นหา</p>';
  $('fnList').querySelectorAll('.fn').forEach(b =>
    b.onclick = () => pickFunction(b.dataset.fn));
}

const isActive = (id) => S.ops[S.activeIdx] && S.ops[S.activeIdx].fn === id;
const activeSpec = () => S.ops[S.activeIdx] ? S.byId[S.ops[S.activeIdx].fn] : null;

function defaults(spec) {
  const p = {};
  // Deep-copy object defaults. A `bands` default is an array, and a reference copy
  // would make this op's params THE SAME array as the registry's default — and as
  // every other op built from the same spec. One in-place edit would corrupt all of
  // them at once.
  (spec.params || []).forEach(x => p[x.key] =
    (x.default && typeof x.default === 'object') ? structuredClone(x.default) : x.default);
  return p;
}

function pickFunction(id) {
  const spec = S.byId[id];
  if (S.activeIdx >= 0 && S.activeIdx < S.ops.length) {
    S.ops[S.activeIdx] = { fn: id, params: defaults(spec) };   // replace the active step
  } else {
    S.ops.push({ fn: id, params: defaults(spec) });
    S.activeIdx = S.ops.length - 1;
  }
  afterOpsChange();
}

function afterOpsChange() {
  renderFunctions();
  renderPipeline();
  renderSuggested();
  renderNotes();
  loadCode();
  syncPickBtn();
  run();
}

/* ───────────────────────── pipeline ───────────────────────── */
function renderPipeline() {
  const parts = S.ops.map((op, i) => {
    const spec = S.byId[op.fn];
    const ms = S.last && S.last.applied[i] ? `<span class="ms">${S.last.applied[i].ms}ms</span>` : '';
    return `<div class="pstep ${i === S.activeIdx ? 'active' : ''}" data-i="${i}">
              <span class="nm">${esc(spec.name)}</span>${ms}
              <button class="x" data-del="${i}" title="ลบขั้นนี้">×</button>
            </div>`;
  });
  const add = `<button class="ghost padd" id="addStep">+ ต่อฟังก์ชัน</button>`;
  $('pipeline').innerHTML = parts.length
    ? parts.join('<span class="parrow">→</span>') + '<span class="parrow">→</span>' + add
    : '<span style="color:var(--dim);font-size:12.5px">กดฟังก์ชันทางขวาเพื่อเริ่ม</span>';

  $('pipeline').querySelectorAll('.pstep').forEach(el => el.onclick = (e) => {
    if (e.target.dataset.del !== undefined) return;
    S.activeIdx = +el.dataset.i;
    renderFunctions(); renderPipeline(); renderSuggested(); renderNotes(); loadCode(); renderSteps();
  });
  $('pipeline').querySelectorAll('[data-del]').forEach(b => b.onclick = (e) => {
    e.stopPropagation();
    S.ops.splice(+b.dataset.del, 1);
    S.activeIdx = Math.min(S.activeIdx, S.ops.length - 1);
    afterOpsChange();
  });
  const addBtn = $('addStep');
  if (addBtn) addBtn.onclick = () => {
    S.activeIdx = S.ops.length;   // next pick appends instead of replacing
    renderFunctions(); renderPipeline();
    toast('เลือกฟังก์ชันถัดไปจากแถบขวา');
  };
}

/* ───────────────────────── params ───────────────────────── */
function paramsHTML(spec, op, idx) {
  if (!spec.params || !spec.params.length) return '';
  const rows = spec.params.map(p => {
    const v = op.params[p.key] ?? p.default;
    if (p.type === 'choice') {
      return `<div class="param"><label>${esc(p.label)}</label>
        <select data-i="${idx}" data-k="${p.key}">${p.options.map(([val, lb]) =>
          `<option value="${esc(String(val))}" ${String(val) === String(v) ? 'selected' : ''}>${esc(lb)}</option>`
        ).join('')}</select></div>`;
    }
    if (p.type === 'bands') {
      const bands = Array.isArray(v) ? v : [];
      return `<div class="param"><label>${esc(p.label)}</label>
        <div class="bands" data-i="${idx}" data-k="${p.key}">
          ${bands.map(([lo, hi], bi) => `<div class="band" data-b="${bi}">
            <i class="sw" style="background:hsl(${lo},90%,50%)"></i>
            <input type="number" class="bl" min="0" max="360" step="1" value="${lo}">
            <span class="dash">–</span>
            <input type="number" class="bh" min="0" max="360" step="1" value="${hi}">
            <span class="deg">°</span>
            ${lo > hi ? '<span class="wrap-tag" title="ช่วงนี้วนผ่าน 0° (สีแดง)">↻</span>' : ''}
            <button class="x" data-bdel="${bi}" title="ลบช่วงนี้">×</button></div>`).join('') ||
            '<p class="dim">ยังไม่ได้เลือกสี — กดปุ่มดูดสีแล้วคลิกบนภาพ หรือเพิ่มช่วงด้านล่าง</p>'}
          <div class="band-add">
            <button class="ghost badd" ${bands.length >= (p.max_bands || 6) ? 'disabled' : ''}>+ เพิ่มช่วง</button>
            <select class="bpreset"><option value="">เพิ่มสีสำเร็จรูป…</option>
              ${(p.presets || []).map(([, lb, lo, hi]) =>
                `<option value="${lo},${hi}">${esc(lb)} (${lo}–${hi}°)</option>`).join('')}
            </select>
          </div>
        </div></div>`;
    }
    const step = p.step ?? (p.type === 'float' ? 0.1 : 1);
    return `<div class="param">
      <label><span>${esc(p.label)}</span><b data-out="${idx}-${p.key}">${v}</b></label>
      <input type="range" data-i="${idx}" data-k="${p.key}" data-t="${p.type}"
             min="${p.min}" max="${p.max}" step="${step}" value="${v}"></div>`;
  }).join('');
  return `<div class="params">${rows}</div>`;
}

function bindParams(root) {
  root.querySelectorAll('input[type=range]').forEach(r => {
    const commit = (live) => {
      let v = r.dataset.t === 'float' ? parseFloat(r.value) : parseInt(r.value, 10);
      if (r.dataset.t === 'odd_int' && v % 2 === 0) v += 1;
      S.ops[+r.dataset.i].params[r.dataset.k] = v;
      const out = root.querySelector(`[data-out="${r.dataset.i}-${r.dataset.k}"]`);
      if (out) out.textContent = v;
      if (!live) run();
    };
    r.oninput = () => commit(true);
    r.onchange = () => commit(false);
  });
  root.querySelectorAll('select[data-k]').forEach(s => s.onchange = () => {
    const raw = s.value;
    const n = Number(raw);
    S.ops[+s.dataset.i].params[s.dataset.k] = (raw !== '' && !isNaN(n)) ? n : raw;
    run();
  });
  root.querySelectorAll('.bands').forEach(el => {
    const i = +el.dataset.i, k = el.dataset.k;
    // Always assign a NEW array — never push/splice in place. run() -> paint() ->
    // renderSteps() -> paramsHTML re-reads S.ops[i].params, so this is all the
    // re-render plumbing needed.
    const set = (v) => { S.ops[i].params[k] = v; run(); };
    const readRows = () => [...el.querySelectorAll('.band')].map(b => [
      clampDeg(b.querySelector('.bl').value), clampDeg(b.querySelector('.bh').value)]);
    el.querySelectorAll('input[type=number]').forEach(n => n.onchange = () => set(readRows()));
    el.querySelectorAll('[data-bdel]').forEach(b => b.onclick = () =>
      set((S.ops[i].params[k] || []).filter((_, j) => j !== +b.dataset.bdel)));
    el.querySelector('.badd').onclick = () => set([...(S.ops[i].params[k] || []), [0, 40]]);
    el.querySelector('.bpreset').onchange = (e) => {
      if (!e.target.value) return;
      set([...(S.ops[i].params[k] || []), e.target.value.split(',').map(Number)]);
    };
  });
}

const clampDeg = (v) => Math.min(360, Math.max(0, Math.round(Number(v) || 0)));

/* ───────────────────────── run ───────────────────────── */
async function run() {
  if (!S.image) return;
  const seq = ++S.reqSeq;
  $('stage').classList.add('busy');
  try {
    const res = await fetch('/api/apply', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: S.image, ops: S.ops }),
    });
    if (seq !== S.reqSeq) return;   // a newer request already went out
    const data = await res.json();
    if (!res.ok) { showError(data.detail || 'เกิดข้อผิดพลาด'); return; }
    S.last = data;
    paint(data);
  } catch (e) {
    if (seq === S.reqSeq) showError('ติดต่อ server ไม่ได้: ' + e.message);
  } finally {
    if (seq === S.reqSeq) $('stage').classList.remove('busy');
  }
}

function showError(msg) {
  $('compare').classList.remove('on');
  $('stageMsg').className = 'stage-msg err';
  $('stageMsg').textContent = msg;
  $('downloadBtn').disabled = true;
}

function paint(d) {
  $('imgBefore').src = d.before;
  $('imgAfter').src = d.after;
  $('compare').classList.add('on');
  $('stageMsg').textContent = '';
  $('stageMsg').className = 'stage-msg';
  $('downloadBtn').disabled = false;

  fitImage(d.shape);

  const names = d.applied.map(a => a.name).join(' → ');
  const total = d.applied.reduce((s, a) => s + a.ms, 0);
  $('viewerTitle').innerHTML = names
    ? `${esc(S.image.replace('_uploads/', ''))} <span style="color:var(--dim)">→ ${esc(names)}
       · ${d.shape.w}×${d.shape.h} · ${total.toFixed(0)}ms</span>`
    : `${esc(S.image.replace('_uploads/', ''))} <span style="color:var(--dim)">· ${d.shape.w}×${d.shape.h}
       — ยังไม่ได้เลือกฟังก์ชัน</span>`;

  drawHist($('histBefore'), d.hist_before);
  drawHist($('histAfter'), d.hist_after);
  renderPipeline();
  renderSteps();
}

/* ───────────────────────── histogram canvas ───────────────────────── */
const COLORS = { 'ความถี่': '#4da3ff', 'แดง': '#ff6b6b', 'เขียว': '#6fd08c', 'น้ำเงิน': '#5aa9ff' };

function drawHist(cv, series) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth || 480, h = 150;
  cv.width = w * dpr; cv.height = h * dpr;
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);

  const pad = { l: 4, r: 4, t: 6, b: 14 };
  const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;

  const all = Object.values(series).flat();
  const max = Math.max(...all, 1);

  g.strokeStyle = '#2a323d'; g.lineWidth = 1;
  [0, 64, 128, 192, 255].forEach(v => {
    const x = pad.l + (v / 255) * iw;
    g.beginPath(); g.moveTo(x, pad.t); g.lineTo(x, pad.t + ih); g.stroke();
    g.fillStyle = '#6b7784'; g.font = '9px Consolas'; g.textAlign = 'center';
    g.fillText(v, x, h - 3);
  });

  for (const [name, vals] of Object.entries(series)) {
    const c = COLORS[name] || '#4da3ff';
    g.beginPath();
    g.moveTo(pad.l, pad.t + ih);
    vals.forEach((v, i) => g.lineTo(pad.l + (i / 255) * iw, pad.t + ih - (v / max) * ih));
    g.lineTo(pad.l + iw, pad.t + ih);
    g.closePath();
    g.fillStyle = c + '33'; g.fill();
    g.strokeStyle = c; g.lineWidth = 1.2; g.stroke();
  }
}

/* ───────────────────────── steps ───────────────────────── */
function renderSteps() {
  const pane = $('tab-steps');
  if (!S.last || !S.last.steps.length) {
    pane.innerHTML = '<p style="color:var(--dim)">เลือกฟังก์ชันเพื่อดูการคำนวณทีละขั้น</p>';
    return;
  }
  const spec = activeSpec();
  const why = spec ? `<div class="why"><b>ใช้เมื่อไหร่:</b> ${esc(spec.why || '')}</div>` : '';

  pane.innerHTML = why + S.last.steps.map((st, si) => {
    const params = S.ops[si] ? paramsHTML(S.byId[st.fn], S.ops[si], si) : '';
    return `<div class="step-block">
      <div class="step-title rule">${si + 1}. ${esc(st.title)}</div>
      ${st.formula ? `<div class="formula">${esc(st.formula)}</div>` : ''}
      ${params}
      ${st.items.map(itemHTML).join('')}
    </div>`;
  }).join('');

  bindParams(pane);
  pane.querySelectorAll('canvas[data-series]').forEach(cv => {
    const s = JSON.parse(cv.dataset.series);
    const marker = cv.dataset.marker ? JSON.parse(cv.dataset.marker) : null;
    drawSeries(cv, s, marker);
  });
}

function itemHTML(it) {
  if (it.type === 'text') return `<div class="step-text">${esc(it.body)}</div>`;
  if (it.type === 'value')
    return `<div class="kv"><span class="k">${esc(it.label)}</span><span class="v">${esc(String(it.value))}</span>
            ${it.note ? `<span class="n">${esc(it.note)}</span>` : ''}</div>`;
  if (it.type === 'matrix')
    return `${lbl(it.label)}<table class="mx">${it.rows.map(r =>
      `<tr>${r.map(v => `<td>${v}</td>`).join('')}</tr>`).join('')}</table>${note(it.note)}`;
  if (it.type === 'table') {
    const hl = new Set(it.highlight || []);
    return `${lbl(it.label)}<table class="dt">
      <thead><tr>${it.columns.map(c => `<th>${esc(c)}</th>`).join('')}</tr></thead>
      <tbody>${it.rows.map((r, i) =>
        `<tr class="${hl.has(i) ? 'hl' : ''}">${r.map(v => `<td>${esc(String(v))}</td>`).join('')}</tr>`
      ).join('')}</tbody></table>${note(it.note)}`;
  }
  if (it.type === 'chart') {
    const legend = Object.keys(it.series).map(n =>
      `<span><i style="background:${COLORS[n] || '#4da3ff'}"></i>${esc(n)}</span>`).join('');
    return `${lbl(it.label)}<div class="chartwrap">
      <canvas data-series='${esc(JSON.stringify(it.series))}'
              ${it.marker ? `data-marker='${esc(JSON.stringify(it.marker))}'` : ''}></canvas>
      <div class="legend">${legend}</div></div>${note(it.note)}`;
  }
  return '';
}

const lbl = (t) => t ? `<div class="step-label">${esc(t)}</div>` : '';
const note = (t) => t ? `<div class="step-note">${esc(t)}</div>` : '';

function drawSeries(cv, series, marker) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth || 460, h = 120;
  cv.width = w * dpr; cv.height = h * dpr;
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);

  const pad = 6, iw = w - pad * 2, ih = h - pad * 2;
  const all = Object.values(series).flat();
  const max = Math.max(...all, 1e-9), min = Math.min(...all, 0);
  const span = (max - min) || 1;

  for (const [name, vals] of Object.entries(series)) {
    const c = COLORS[name] || '#4da3ff';
    const n = vals.length;
    g.beginPath();
    vals.forEach((v, i) => {
      const x = pad + (n > 1 ? (i / (n - 1)) * iw : iw / 2);
      const y = pad + ih - ((v - min) / span) * ih;
      i ? g.lineTo(x, y) : g.moveTo(x, y);
    });
    g.strokeStyle = c; g.lineWidth = 1.4; g.stroke();
  }

  if (marker && marker.index >= 0) {
    const n = Math.max(...Object.values(series).map(v => v.length));
    const x = pad + (n > 1 ? (marker.index / (n - 1)) * iw : iw / 2);
    g.strokeStyle = '#ffd166'; g.lineWidth = 1.5; g.setLineDash([3, 3]);
    g.beginPath(); g.moveTo(x, pad); g.lineTo(x, pad + ih); g.stroke();
    g.setLineDash([]);
    g.fillStyle = '#ffd166'; g.font = 'bold 10px Consolas';
    g.textAlign = x > w / 2 ? 'right' : 'left';
    g.fillText(marker.label, x + (x > w / 2 ? -4 : 4), pad + 9);
  }
}

/* ───────────────────────── notes ───────────────────────── */
function renderNotes() {
  const spec = activeSpec();
  const pane = $('tab-notes');
  if (!spec) { pane.innerHTML = '<p style="color:var(--dim)">เลือกฟังก์ชันก่อน</p>'; return; }
  const notes = spec.notes || [];
  pane.innerHTML =
    `<div class="why"><b>${esc(spec.name)}</b> — ${esc(spec.why || '')}
       <div style="color:var(--dim);margin-top:4px;font-size:12px">
         สูตร: <code>${esc(spec.formula || '')}</code></div></div>` +
    (notes.length
      ? notes.map(n => `<div class="note ${n.level}">${n.level === 'warn' ? '<b>⚠️ ระวัง:</b> ' : '<b>💡 </b>'}${esc(n.text)}</div>`).join('')
      : '<p style="color:var(--dim)">ไม่มีข้อควรระวังพิเศษสำหรับฟังก์ชันนี้</p>');
}

/* ───────────────────────── code ───────────────────────── */
async function loadCode() {
  const spec = activeSpec();
  if (!spec) { $('codeBody').innerHTML = '<code>เลือกฟังก์ชันก่อน</code>'; return; }
  const d = await fetch('/api/code/' + spec.id).then(r => r.json()).catch(() => ({ code: {} }));
  S.codeCache = d.code || {};
  showCode();
}

function showCode() {
  const c = S.codeCache || {};
  const txt = c[S.variant];
  document.querySelectorAll('.cvar').forEach(b => {
    b.classList.toggle('active', b.dataset.variant === S.variant);
    b.disabled = !c[b.dataset.variant];
    b.style.opacity = c[b.dataset.variant] ? 1 : .4;
  });
  $('codeBody').innerHTML = `<code>${esc(txt ||
    (S.variant === 'cv2'
      ? 'ฟังก์ชันนี้ไม่มีคำสั่งสำเร็จรูปของ cv2 ที่ตรงตัว — ต้องเขียนลูปเอง'
      : 'ยังไม่มีโค้ดสำหรับฟังก์ชันนี้'))}</code>`;
}

/* ───────────────────────── wiring ───────────────────────── */
function wire() {
  $('imageSearch').oninput = renderImages;
  $('fnSearch').oninput = renderFunctions;
  $('starOnly').onchange = renderFunctions;
  $('helpBtn').onclick = () => $('help').classList.toggle('hidden');
  $('pickBtn').onclick = () => setPicking(!$('compare').classList.contains('picking'));

  document.querySelectorAll('[data-collapse]').forEach(b =>
    b.onclick = () => setCollapsed(b.dataset.collapse, true));
  document.querySelectorAll('[data-expand]').forEach(b =>
    b.onclick = () => setCollapsed(b.dataset.expand, false));
  $('focusBtn').onclick = toggleFocus;

  document.addEventListener('keydown', (e) => {
    // don't steal keys from the search boxes or the hue number inputs
    const t = e.target.tagName;
    if (t === 'INPUT' || t === 'TEXTAREA' || t === 'SELECT' || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === '[') { toggleCollapsed('imagePanel'); e.preventDefault(); }
    else if (e.key === ']') { toggleCollapsed('fnPanel'); e.preventDefault(); }
    else if (e.key === '\\') { toggleFocus(); e.preventDefault(); }
  });

  document.querySelectorAll('.tab').forEach(t => t.onclick = () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.toggle('active', x === t));
    ['steps', 'code', 'notes'].forEach(n =>
      $('tab-' + n).classList.toggle('hidden', n !== t.dataset.tab));
  });

  document.querySelectorAll('.cvar').forEach(b => b.onclick = () => {
    S.variant = b.dataset.variant;
    showCode();
  });

  $('copyCode').onclick = async () => {
    const txt = (S.codeCache || {})[S.variant];
    if (!txt) return toast('ไม่มีโค้ดให้คัดลอก', true);
    try {
      await navigator.clipboard.writeText(txt);
      toast('คัดลอกโค้ดแล้ว');
    } catch { toast('คัดลอกไม่สำเร็จ — เลือกข้อความแล้วกด Ctrl+C', true); }
  };

  $('modeCompare').onclick = () => setMode(true);
  $('modeAfter').onclick = () => setMode(false);

  $('downloadBtn').onclick = () => {
    if (!S.last) return;
    const a = document.createElement('a');
    a.href = S.last.after;
    const tag = S.ops.map(o => o.fn).join('-') || 'original';
    a.download = `${S.image.replace('_uploads/', '').replace(/\.[^.]+$/, '')}_${tag}.png`;
    a.click();
  };

  $('uploadInput').onchange = (e) => e.target.files[0] && upload(e.target.files[0]);
  document.addEventListener('dragover', e => e.preventDefault());
  document.addEventListener('drop', e => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) upload(f);
  });

  initHandle();
  window.addEventListener('resize', () => {
    if (S.last) { drawHist($('histBefore'), S.last.hist_before); drawHist($('histAfter'), S.last.hist_after); }
    renderSteps();
  });
}

function setMode(compare) {
  $('modeCompare').classList.toggle('active', compare);
  $('modeAfter').classList.toggle('active', !compare);
  $('compare').classList.toggle('after-only', !compare);
}

async function upload(file) {
  if (!file) return;
  if (!/^image\//.test(file.type) && !/\.(png|jpe?g|bmp|tiff?)$/i.test(file.name))
    return toast('ไฟล์นี้ไม่ใช่รูปภาพ', true);

  toast('กำลังอัปโหลด ' + file.name + '…');
  try {
    // raw body, not FormData -- keeps python-multipart out of the dependency list
    const r = await fetch('/api/upload', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/octet-stream',
        'X-Filename': encodeURIComponent(file.name),
      },
      body: file,
    });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) return toast(d.detail || ('อัปโหลดไม่สำเร็จ (HTTP ' + r.status + ')'), true);

    S.images = (await fetch('/api/images').then(x => x.json())).images;
    $('imageSearch').value = '';
    selectImage(d.name);
    toast('อัปโหลด ' + file.name + ' แล้ว');
  } catch (e) { toast('อัปโหลดไม่สำเร็จ: ' + e.message, true); }
}

function initHandle() {
  const wrap = $('compare'), handle = $('handle'), before = $('beforeWrap'), img = $('imgBefore');
  let drag = false;

  const move = (clientX) => {
    const r = wrap.getBoundingClientRect();
    const pct = Math.max(0, Math.min(100, ((clientX - r.left) / r.width) * 100));
    before.style.width = pct + '%';
    handle.style.left = pct + '%';
    img.style.width = r.width + 'px';   // clipped layer must stay aligned with the base
  };

  const sync = () => { img.style.width = wrap.getBoundingClientRect().width + 'px'; };
  img.addEventListener('load', sync);
  window.addEventListener('resize', () => { sync(); fitImage(S.last && S.last.shape); });

  // The wipe listens on the whole wrap, not just the handle, so an eyedropper click
  // would start a drag too. Guard by class rather than capture-phase + stopPropagation:
  // both listeners sit on `wrap` itself, and at-target listeners fire in registration
  // order regardless of the capture flag — so ordering would decide the winner.
  const down = (e) => {
    if (wrap.classList.contains('picking')) return;
    drag = true; move(e.clientX ?? e.touches[0].clientX); e.preventDefault();
  };
  handle.addEventListener('mousedown', down);
  wrap.addEventListener('mousedown', down);
  wrap.addEventListener('click', (e) => {
    if (wrap.classList.contains('picking')) pickAt(e.clientX, e.clientY);
  });
  handle.addEventListener('touchstart', down, { passive: false });
  window.addEventListener('mousemove', e => drag && move(e.clientX));
  window.addEventListener('touchmove', e => drag && move(e.touches[0].clientX), { passive: false });
  window.addEventListener('mouseup', () => drag = false);
  window.addEventListener('touchend', () => drag = false);
}

/* ───────────────────────── image fit ───────────────────────── */
/* The stage is ~1000px wide, so a 256px test image was rendering at 4x: it swallowed
   the page and pushed the histogram and the step-by-step panel below the fold, which
   are the whole point of the tool. CSS cannot fix this — it does not know the natural
   size — but the backend tells us, in d.shape.

   Cap on HEIGHT, not on a flat upscale multiplier. "Keep the histogram on screen" is
   the actual goal, and a multiplier only reaches it by accident: a flat 2x capped a
   256px image at 512px, well under the 1074px stage, so the cap bound instead of the
   stage and widening the stage (focus mode) changed nothing at all. A height cap
   scales with the window, uses the room focus mode frees up, and still guarantees
   ~38vh is left for everything below. The 3x ceiling is only a backstop against
   absurd blowups of tiny images — image-rendering:pixelated is deliberate here
   (seeing the pixels IS the lesson), but past ~3x you get bigger blocks, not detail. */
function fitImage(shape) {
  if (!shape) return;
  const byUpscale = shape.w * 3;
  const byHeight = (shape.w / shape.h) * window.innerHeight * IMG_VH;
  $('compare').style.maxWidth = Math.round(Math.min(byUpscale, byHeight)) + 'px';
}
// Measured, not guessed. At 0.62 the histogram started at y=938 in a 1050px window and
// was cut off — which defeats the point of capping at all. 0.52 still missed by 5px at
// 1440x900. 0.50 keeps the image and both histograms fully on screen at 900, 1050 and
// 1200px tall windows.
const IMG_VH = 0.50;

/* ───────────────────────── collapsible panels ───────────────────────── */
const PANELS = { imagePanel: 'c-images', fnPanel: 'c-fns' };

function setCollapsed(id, on) {
  $(id).classList.toggle('collapsed', on);
  document.querySelector('main').classList.toggle(PANELS[id], on);
  try { localStorage.setItem('nara.' + id, on ? '1' : '0'); } catch (e) { /* private mode */ }
  syncFocusBtn();
}

const isCollapsed = (id) => $(id).classList.contains('collapsed');

function toggleCollapsed(id) { setCollapsed(id, !isCollapsed(id)); }

function syncFocusBtn() {
  const both = isCollapsed('imagePanel') && isCollapsed('fnPanel');
  $('focusBtn').classList.toggle('active', both);
  $('focusBtn').textContent = both ? 'เลิกโฟกัส' : 'โฟกัส';
}

function toggleFocus() {
  const on = !(isCollapsed('imagePanel') && isCollapsed('fnPanel'));
  setCollapsed('imagePanel', on);
  setCollapsed('fnPanel', on);
}

function restorePanels() {
  for (const id of Object.keys(PANELS)) {
    let saved = null;
    try { saved = localStorage.getItem('nara.' + id); } catch (e) { /* private mode */ }
    if (saved === '1') setCollapsed(id, true);
  }
  syncFocusBtn();
}

/* ───────────────────────── eyedropper ───────────────────────── */
function syncPickBtn() {
  const spec = activeSpec();
  const on = !!(spec && spec.pickable);
  $('pickBtn').hidden = !on;
  if (!on) setPicking(false);
}

function setPicking(on) {
  $('compare').classList.toggle('picking', on);
  $('pickBtn').classList.toggle('active', on);
  if (on) toast('คลิกบนภาพตรงสีที่ต้องการ — คลิกซ้ำได้เรื่อยๆ เพื่อเลือกหลายสี');
}

async function pickAt(clientX, clientY) {
  const i = S.activeIdx, spec = activeSpec();
  if (!spec || !spec.pickable || !S.image) return;

  // The displayed image is an unrotated, aspect-preserved scale of the backend's
  // working array (_read downsizes to MAX_SIDE, _png_b64 encodes that same array),
  // so naturalWidth IS the backend's width — the mapping is 1:1, no scale factor.
  const img = $('imgAfter'), r = img.getBoundingClientRect();
  if (!r.width || !r.height) return;
  const x = Math.floor((clientX - r.left) / r.width * img.naturalWidth);
  const y = Math.floor((clientY - r.top) / r.height * img.naturalHeight);

  try {
    const res = await fetch('/api/pick', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      // ops = the pipeline PREFIX: sample what this op actually sees, not the final
      // result (which a previous color_select may have already blacked out).
      body: JSON.stringify({ image: S.image, ops: S.ops.slice(0, i), x, y, width: 20 }),
    });
    const d = await res.json();
    if (!res.ok) { toast(d.detail || 'ดูดสีไม่สำเร็จ', true); return; }
    if (d.weak) {
      toast(`พิกเซลนี้ S=${d.s} V=${d.v} ต่ำเกินไป (สีเทา/ดำ) — Hue ไม่มีความหมาย ` +
            'ลองคลิกที่สีสดๆ', true);
      return;
    }
    const p = (spec.params || []).find(q => q.type === 'bands');
    if (p) {
      const cur = S.ops[i].params[p.key] || [];
      if (cur.length >= (p.max_bands || 6)) { toast('เลือกได้สูงสุด ' + (p.max_bands || 6) + ' ช่วง', true); return; }
      S.ops[i].params[p.key] = [...cur, d.band];   // append, so clicking blue then orange builds both
    } else {
      S.ops[i].params.low = d.band[0];
      S.ops[i].params.high = d.band[1];
    }
    toast(`ดูดสี RGB(${d.rgb.join(', ')}) → H=${d.h}° → เลือกช่วง ${d.band[0]}–${d.band[1]}°`);
    run();
  } catch (e) { toast('ดูดสีไม่สำเร็จ: ' + e.message, true); }
}

let toastTimer;
function toast(msg, isErr) {
  const t = $('toast');
  t.textContent = msg;
  t.className = 'toast' + (isErr ? ' err' : '');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add('hidden'), 2200);
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

boot();
