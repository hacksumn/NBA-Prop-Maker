/* charts.js — dependency-free SVG / HTML charts for the Command Center. */
(function () {
  'use strict';
  let uid = 0;
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const isNum = (x) => typeof x === 'number' && isFinite(x);

  /* ---------- formatters ---------- */
  const F = {
    esc,
    pct: (x, d = 1) => (isNum(x) ? (x * 100).toFixed(d) + '%' : '–'),
    pts: (x, d = 1) => (isNum(x) ? (x >= 0 ? '+' : '−') + Math.abs(x * 100).toFixed(d) + ' pts' : '–'),
    u: (x, d = 2) => (isNum(x) ? (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(d) + 'u' : '–'),
    n: (x, d = 0) => (isNum(x) ? Number(x).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d }) : '–'),
    odds: (x) => (isNum(x) ? (x > 0 ? '+' : '') + Math.round(x) : '–'),
    money: (x) => (isNum(x) ? (x < 0 ? '−$' : '$') + Math.abs(x).toLocaleString(undefined, { maximumFractionDigits: 0 }) : '–'),
    sgnMoney: (x) => (isNum(x) ? (x >= 0 ? '+$' : '−$') + Math.abs(x).toLocaleString(undefined, { maximumFractionDigits: 0 }) : '–'),
    date: (s) => { if (!s) return '–'; const p = String(s).slice(0, 10).split('-'); return p.length === 3 ? MONTHS[+p[1] - 1] + ' ' + +p[2] : s; },
    dateY: (s) => { if (!s) return '–'; const p = String(s).slice(0, 10).split('-'); return p.length === 3 ? MONTHS[+p[1] - 1] + ' ' + +p[2] + ", '" + p[0].slice(2) : s; },
    tone: (x) => (!isNum(x) || x === 0 ? '' : x > 0 ? 't-win' : 't-loss'),
    ago: (h) => (!isNum(h) ? '–' : h < 1 ? Math.round(h * 60) + 'm ago' : h < 48 ? h.toFixed(1) + 'h ago' : Math.round(h / 24) + 'd ago'),
  };

  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888';

  /* ---------- scales ---------- */
  function niceTicks(min, max, count) {
    count = count || 5;
    const span = max - min || 1, raw = span / count, mag = Math.pow(10, Math.floor(Math.log10(raw))), err = raw / mag;
    const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
    const start = Math.floor(min / step) * step, out = [];
    for (let v = start; v <= max + step * 0.5; v += step) out.push(+v.toFixed(10));
    return out;
  }

  /* ---------- tooltip ---------- */
  const tipEl = () => document.getElementById('tip');
  function placeTip(x, y) {
    const t = tipEl(); if (!t) return;
    const r = t.getBoundingClientRect();
    let nx = x + 16, ny = y + 16;
    if (nx + r.width > innerWidth - 8) nx = x - r.width - 16;
    if (ny + r.height > innerHeight - 8) ny = y - r.height - 16;
    t.style.left = Math.max(6, nx) + 'px'; t.style.top = Math.max(6, ny) + 'px';
  }
  function showTip(html, x, y) { const t = tipEl(); if (!t) return; t.innerHTML = html; t.classList.add('on'); placeTip(x, y); }
  function hideTip() { const t = tipEl(); if (t) t.classList.remove('on'); }
  document.addEventListener('pointerover', (e) => { const el = e.target.closest && e.target.closest('[data-tip]'); if (el) showTip(el.getAttribute('data-tip'), e.clientX, e.clientY); });
  document.addEventListener('pointermove', (e) => { const t = tipEl(); if (t && t.classList.contains('on') && e.target.closest && e.target.closest('[data-tip]')) placeTip(e.clientX, e.clientY); });
  document.addEventListener('pointerout', (e) => { const to = e.relatedTarget; if (!to || !to.closest || !to.closest('[data-tip]')) { if (!(e.target.closest && e.target.closest('svg.chart .hit'))) hideTip(); } });

  /* ---------- responsive mount ---------- */
  const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver((entries) => {
    entries.forEach((en) => {
      const el = en.target, w = Math.round(en.contentRect.width);
      // redraw from a timer, not inside the callback (drawing resizes the element and trips the browser's
      // loop guard) and not from requestAnimationFrame (paused or throttled in hidden/emulated tabs)
      if (el.__draw && w > 0 && w !== el.__w) { el.__w = w; setTimeout(() => { el.__draw(!el.__animated); el.__animated = true; }, 0); }
    });
  }) : null;
  function mount(el, draw) {
    el.__draw = draw; el.__w = Math.round(el.clientWidth); el.__animated = false;
    if (el.__w > 0) { draw(true); el.__animated = true; }
    if (ro) ro.observe(el);
  }

  const svgOpen = (W, H) => `<svg class="chart" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img">`;

  /* ---------- line chart ---------- */
  function line(el, cfg) {
    const id = 'lg' + (++uid);
    mount(el, (first) => {
      const W = Math.max(280, el.clientWidth), H = cfg.height || 250;
      const m = { l: cfg.yw || 46, r: 16, t: 14, b: 26 };
      const iw = W - m.l - m.r, ih = H - m.t - m.b;
      const S = cfg.series, n = Math.max.apply(null, S.map((s) => s.data.length));
      const ys = [];
      S.forEach((s) => s.data.forEach((d) => { if (d && isNum(d.y)) ys.push(d.y); }));
      (cfg.refs || []).forEach((r) => ys.push(r.y));
      if (cfg.zero !== false) ys.push(0);
      if (!ys.length) { el.innerHTML = '<div class="empty">No data yet.</div>'; return; }
      let y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
      if (y0 === y1) { y0 -= 1; y1 += 1; }
      const pad = (y1 - y0) * 0.06; y0 -= pad; y1 += pad;
      if (isNum(cfg.yMin)) y0 = cfg.yMin; if (isNum(cfg.yMax)) y1 = cfg.yMax;
      const ticks = niceTicks(y0, y1, cfg.ticks || 4);
      y0 = Math.min(y0, ticks[0]); y1 = Math.max(y1, ticks[ticks.length - 1]);
      const X = (i) => m.l + (n <= 1 ? iw / 2 : (i * iw) / (n - 1));
      const Y = (v) => m.t + ih - ((v - y0) / (y1 - y0)) * ih;
      const yfmt = cfg.yFmt || ((v) => F.n(v, Math.abs(v) < 10 ? 1 : 0));
      let out = svgOpen(W, H) + '<defs>';
      S.forEach((s, k) => {
        out += `<linearGradient id="${id}a${k}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${s.color}" stop-opacity=".34"/><stop offset="1" stop-color="${s.color}" stop-opacity="0"/></linearGradient>`;
      });
      const zy = Y(Math.max(y0, Math.min(0, y1)));
      if (cfg.split) {
        out += `<clipPath id="${id}up"><rect x="0" y="0" width="${W}" height="${zy}"/></clipPath><clipPath id="${id}dn"><rect x="0" y="${zy}" width="${W}" height="${H}"/></clipPath>`;
      }
      out += '</defs><g class="grid">';
      ticks.forEach((t) => { out += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(t)}" y2="${Y(t)}"/><text x="${m.l - 8}" y="${Y(t) + 3.5}" text-anchor="end">${esc(yfmt(t))}</text>`; });
      out += '</g>';
      if (y0 < 0 && y1 > 0) out += `<line class="zero" x1="${m.l}" x2="${W - m.r}" y1="${Y(0)}" y2="${Y(0)}"/>`;
      (cfg.refs || []).forEach((r) => {
        out += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(r.y)}" y2="${Y(r.y)}" stroke="${r.color || css('--amber')}" stroke-dasharray="5 4" opacity=".8"/>`;
        if (r.label) out += `<text x="${W - m.r - 4}" y="${Y(r.y) - 5}" text-anchor="end" style="fill:${r.color || css('--amber')}">${esc(r.label)}</text>`;
      });
      const labels = cfg.labels || [];
      if (labels.length) {
        const want = Math.max(2, Math.min(cfg.xTicks || 6, n, Math.floor(iw / 72))), seen = new Set();
        let prevText = null;
        for (let k = 0; k < want; k++) {
          const i = want === 1 ? 0 : Math.round((k * (n - 1)) / (want - 1));
          if (seen.has(i)) continue; seen.add(i);
          const text = labels[i] || '';
          if (text === prevText) continue; prevText = text; // several points can share one date label
          out += `<text x="${X(i)}" y="${H - 7}" text-anchor="${k === 0 ? 'start' : k === want - 1 ? 'end' : 'middle'}">${esc(text)}</text>`;
        }
      }
      const cls = first ? 'draw' : '';
      S.forEach((s, k) => {
        let d = '', started = false, lastX = 0, firstX = 0;
        s.data.forEach((p, i) => {
          if (!p || !isNum(p.y)) { started = false; return; }
          d += (started ? 'L' : 'M') + X(i).toFixed(1) + ' ' + Y(p.y).toFixed(1); if (!started) firstX = X(i); started = true; lastX = X(i);
        });
        if (!d) return;
        const base = Y(Math.max(y0, Math.min(0, y1)));
        const area = s.area ? d + `L${lastX.toFixed(1)} ${base}L${firstX.toFixed(1)} ${base}Z` : '';
        if (cfg.split && k === 0) {
          const up = s.color, dn = cfg.splitColor || css('--loss');
          [['up', up], ['dn', dn]].forEach(([c, col]) => {
            out += `<g clip-path="url(#${id}${c})">`;
            if (s.area) out += `<path class="${first ? 'fadein' : ''}" d="${area}" fill="${col}" fill-opacity=".17"/>`;
            out += `<path class="${cls}" pathLength="1" d="${d}" fill="none" stroke="${col}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round" style="filter:drop-shadow(0 0 6px ${col})"/></g>`;
          });
        } else {
          if (s.area) out += `<path class="${first ? 'fadein' : ''}" d="${area}" fill="url(#${id}a${k})"/>`;
          out += `<path class="${cls}" pathLength="1" d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.width || 2.2}" stroke-linejoin="round" stroke-linecap="round" ${s.dashed ? 'stroke-dasharray="5 4"' : ''} style="filter:drop-shadow(0 0 5px ${s.color}66)"/>`;
        }
        const lp = [...s.data].reverse().find((p) => p && isNum(p.y));
        if (lp) { const li = s.data.lastIndexOf(lp); const col = cfg.split && k === 0 ? (lp.y >= 0 ? s.color : (cfg.splitColor || css('--loss'))) : s.color; out += `<circle cx="${X(li)}" cy="${Y(lp.y)}" r="9" fill="${col}" opacity=".18"/><circle cx="${X(li)}" cy="${Y(lp.y)}" r="3.6" fill="${col}" stroke="${css('--panel-solid')}" stroke-width="1.5"/>`; }
      });
      out += `<line class="xh" x1="0" x2="0" y1="${m.t}" y2="${m.t + ih}" stroke="${css('--text')}" opacity="0" stroke-dasharray="3 3"/>`;
      S.forEach((s, k) => { out += `<circle class="xd" data-k="${k}" r="4.5" fill="${s.color}" stroke="${css('--panel-solid')}" stroke-width="2" opacity="0"/>`; });
      out += `<rect class="hit" x="${m.l}" y="${m.t}" width="${iw}" height="${ih}" fill="transparent"/></svg>`;
      el.innerHTML = out;
      const svg = el.querySelector('svg'), hit = svg.querySelector('.hit'), xh = svg.querySelector('.xh'), dots = svg.querySelectorAll('.xd');
      const move = (e) => {
        const r = svg.getBoundingClientRect(), px = ((e.clientX - r.left) / r.width) * W;
        const i = Math.max(0, Math.min(n - 1, Math.round(((px - m.l) / iw) * (n - 1))));
        xh.setAttribute('x1', X(i)); xh.setAttribute('x2', X(i)); xh.setAttribute('opacity', '.5');
        dots.forEach((d) => { const s = S[+d.getAttribute('data-k')], p = s.data[i]; if (p && isNum(p.y)) { d.setAttribute('cx', X(i)); d.setAttribute('cy', Y(p.y)); d.setAttribute('opacity', '1'); } else d.setAttribute('opacity', '0'); });
        showTip(cfg.tip ? cfg.tip(i) : '', e.clientX, e.clientY);
      };
      hit.addEventListener('pointermove', move);
      hit.addEventListener('pointerleave', () => { xh.setAttribute('opacity', '0'); dots.forEach((d) => d.setAttribute('opacity', '0')); hideTip(); });
    });
  }

  /* ---------- signed bars (daily units) / diverging W-L bars ---------- */
  function bars(el, cfg) {
    mount(el, (first) => {
      const W = Math.max(280, el.clientWidth), H = cfg.height || 220;
      const m = { l: cfg.yw || 40, r: 10, t: 12, b: 26 };
      const iw = W - m.l - m.r, ih = H - m.t - m.b, D = cfg.data, n = D.length;
      if (!n) { el.innerHTML = '<div class="empty">No data yet.</div>'; return; }
      const top = cfg.diverge ? D.map((d) => d.up || 0) : D.map((d) => d.value);
      const bot = cfg.diverge ? D.map((d) => -(d.down || 0)) : D.map((d) => d.value);
      let y1 = Math.max(0, ...top), y0 = Math.min(0, ...bot);
      if (y1 === y0) y1 = y0 + 1;
      const ticks = niceTicks(y0, y1, cfg.ticks || 4);
      y0 = Math.min(y0, ticks[0]); y1 = Math.max(y1, ticks[ticks.length - 1]);
      const Y = (v) => m.t + ih - ((v - y0) / (y1 - y0)) * ih, step = iw / n, bw = Math.max(2, Math.min(34, step * 0.7));
      const yfmt = cfg.yFmt || ((v) => F.n(v, 0));
      let out = svgOpen(W, H) + '<g class="grid">';
      ticks.forEach((t) => { out += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(t)}" y2="${Y(t)}"/><text x="${m.l - 7}" y="${Y(t) + 3.5}" text-anchor="end">${esc(yfmt(t))}</text>`; });
      out += `</g><line class="zero" x1="${m.l}" x2="${W - m.r}" y1="${Y(0)}" y2="${Y(0)}"/>`;
      const pos = cfg.pos || css('--win'), neg = cfg.neg || css('--loss');
      D.forEach((d, i) => {
        const cx = m.l + step * i + step / 2, x = cx - bw / 2, anim = first ? `style="transform-origin:${cx}px ${Y(0)}px;animation:growY .7s ${Math.min(i * 12, 500)}ms cubic-bezier(.2,.8,.2,1) both"` : '';
        if (cfg.diverge) {
          const hu = Y(0) - Y(d.up || 0), hd = Y(-(d.down || 0)) - Y(0);
          if (d.up) out += `<rect x="${x}" y="${Y(d.up)}" width="${bw}" height="${hu}" rx="3" fill="${pos}" ${anim}/>`;
          if (d.down) out += `<rect x="${x}" y="${Y(0)}" width="${bw}" height="${hd}" rx="3" fill="${neg}" ${anim}/>`;
        } else {
          const y = Y(Math.max(d.value, 0)), h = Math.max(1.5, Math.abs(Y(d.value) - Y(0)));
          out += `<rect x="${x}" y="${d.value >= 0 ? y : Y(0)}" width="${bw}" height="${h}" rx="3" fill="${d.color || (d.value >= 0 ? pos : neg)}" ${anim}/>`;
        }
        out += `<rect x="${m.l + step * i}" y="${m.t}" width="${step}" height="${ih}" fill="transparent" data-tip="${esc(d.tip || '')}"/>`;
      });
      const want = Math.max(2, Math.min(cfg.xTicks || 7, n, Math.floor(iw / 62)));
      for (let k = 0; k < want; k++) {
        const i = want === 1 ? 0 : Math.round((k * (n - 1)) / (want - 1));
        out += `<text x="${m.l + step * i + step / 2}" y="${H - 7}" text-anchor="${k === 0 ? 'start' : k === want - 1 ? 'end' : 'middle'}">${esc(D[i].label || '')}</text>`;
      }
      el.innerHTML = out + '</svg>';
    });
  }

  /* ---------- sparkline ---------- */
  function spark(vals, o) {
    o = o || {};
    const w = o.w || 84, h = o.h || 30, v = vals.filter(isNum);
    if (v.length < 2) return '';
    let mn = Math.min(...v), mx = Math.max(...v); if (mn === mx) { mn -= 1; mx += 1; }
    const pts = v.map((y, i) => [(i / (v.length - 1)) * (w - 4) + 2, h - 3 - ((y - mn) / (mx - mn)) * (h - 6)]);
    const d = pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join('');
    const col = o.color || css('--accent-2'), last = pts[pts.length - 1], gid = 'sp' + (++uid);
    return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" class="spark"><defs><linearGradient id="${gid}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${col}" stop-opacity=".35"/><stop offset="1" stop-color="${col}" stop-opacity="0"/></linearGradient></defs><path d="${d}L${w - 2} ${h}L2 ${h}Z" fill="url(#${gid})"/><path d="${d}" fill="none" stroke="${col}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><circle cx="${last[0]}" cy="${last[1]}" r="2.6" fill="${col}"/></svg>`;
  }

  /* ---------- ring gauge with confidence arc ---------- */
  function ring(o) {
    const S = o.size || 132, r = S / 2 - 14, c = 2 * Math.PI * r, cx = S / 2, cy = S / 2;
    const v = Math.max(0, Math.min(1, o.value || 0)), col = o.color || css('--win'), id = 'rg' + (++uid);
    const arc = (a, b, rr, w, color, op) => {
      const circ = 2 * Math.PI * rr; return `<circle cx="${cx}" cy="${cy}" r="${rr}" fill="none" stroke="${color}" stroke-width="${w}" stroke-linecap="round" stroke-dasharray="${(b - a) * circ} ${circ}" stroke-dashoffset="${-a * circ}" transform="rotate(-90 ${cx} ${cy})" opacity="${op}"/>`;
    };
    let s = `<svg class="chart" style="width:${S}px;height:${S}px" viewBox="0 0 ${S} ${S}"><defs><linearGradient id="${id}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="${col}"/><stop offset="1" stop-color="${o.color2 || css('--accent-2')}"/></linearGradient></defs>`;
    s += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${css('--line')}" stroke-width="11"/>`;
    s += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="url(#${id})" stroke-width="11" stroke-linecap="round" stroke-dasharray="${v * c} ${c}" transform="rotate(-90 ${cx} ${cy})" style="filter:drop-shadow(0 0 8px ${col}88)"><animate attributeName="stroke-dasharray" from="0 ${c}" to="${v * c} ${c}" dur="1.1s" fill="freeze" calcMode="spline" keySplines=".2 .8 .2 1" keyTimes="0;1"/></circle>`;
    if (o.ci && isNum(o.ci[0]) && isNum(o.ci[1])) s += arc(o.ci[0], o.ci[1], r + 8.5, 3, css('--text'), 0.55);
    if (isNum(o.ref)) { const a = o.ref * 2 * Math.PI - Math.PI / 2; s += `<line x1="${cx + (r - 9) * Math.cos(a)}" y1="${cy + (r - 9) * Math.sin(a)}" x2="${cx + (r + 9) * Math.cos(a)}" y2="${cy + (r + 9) * Math.sin(a)}" stroke="${css('--amber')}" stroke-width="2.4" stroke-linecap="round"/>`; }
    s += `<text x="${cx}" y="${cy + 4}" text-anchor="middle" style="font-size:${S * 0.2}px;font-weight:780;fill:${css('--text')};letter-spacing:-.03em">${esc(o.text)}</text>`;
    if (o.sub) s += `<text x="${cx}" y="${cy + S * 0.17 + 6}" text-anchor="middle" style="font-size:10.5px">${esc(o.sub)}</text>`;
    return s + '</svg>';
  }

  /* ---------- forest plot (estimate + CI per row) ---------- */
  function forest(el, cfg) {
    mount(el, () => {
      const rows = cfg.rows, rh = cfg.rowH || 34, W = Math.max(300, el.clientWidth), lw = cfg.labelW || Math.min(170, W * 0.34);
      const m = { l: lw, r: 78, t: 8, b: 26 }, iw = W - m.l - m.r, H = m.t + rows.length * rh + m.b;
      if (!rows.length) { el.innerHTML = '<div class="empty">No data yet.</div>'; return; }
      let lo = cfg.domain ? cfg.domain[0] : Math.min(...rows.map((r) => (isNum(r.lo) ? r.lo : r.est))), hi = cfg.domain ? cfg.domain[1] : Math.max(...rows.map((r) => (isNum(r.hi) ? r.hi : r.est)));
      if (!cfg.domain) { const p = (hi - lo) * 0.08 || 0.05; lo -= p; hi += p; if (isNum(cfg.ref)) { lo = Math.min(lo, cfg.ref - 0.02); hi = Math.max(hi, cfg.ref + 0.02); } }
      const X = (v) => m.l + ((v - lo) / (hi - lo)) * iw, ticks = niceTicks(lo, hi, 5), fmt = cfg.fmt || ((v) => F.pct(v, 0));
      let out = svgOpen(W, H) + '<g class="grid">';
      ticks.filter((t) => t >= lo && t <= hi).forEach((t) => { out += `<line x1="${X(t)}" x2="${X(t)}" y1="${m.t}" y2="${H - m.b}"/><text x="${X(t)}" y="${H - 8}" text-anchor="middle">${esc(fmt(t))}</text>`; });
      out += '</g>';
      if (isNum(cfg.ref)) out += `<line x1="${X(cfg.ref)}" x2="${X(cfg.ref)}" y1="${m.t}" y2="${H - m.b}" stroke="${css('--amber')}" stroke-dasharray="5 4"/>` + (cfg.refLabel ? `<text x="${X(cfg.ref) + 4}" y="${m.t + 9}" style="fill:${css('--amber')}">${esc(cfg.refLabel)}</text>` : '');
      rows.forEach((r, i) => {
        const cy = m.t + i * rh + rh / 2, col = r.color || (isNum(r.lo) && isNum(cfg.ref) ? (r.lo > cfg.ref ? css('--win') : r.hi < cfg.ref ? css('--loss') : css('--accent-2')) : css('--accent-2'));
        out += `<text x="${m.l - 10}" y="${cy + 1}" text-anchor="end" style="fill:${css('--text')};font-weight:650;font-size:11.5px">${esc(r.label)}</text>`;
        if (r.sub) out += `<text x="${m.l - 10}" y="${cy + 13}" text-anchor="end" style="font-size:10px">${esc(r.sub)}</text>`;
        if (isNum(r.lo) && isNum(r.hi)) out += `<line x1="${X(r.lo)}" x2="${X(r.hi)}" y1="${cy}" y2="${cy}" stroke="${col}" stroke-width="3.2" stroke-linecap="round" opacity=".55"/>`;
        out += `<circle cx="${X(r.est)}" cy="${cy}" r="6" fill="${col}" stroke="${css('--panel-solid')}" stroke-width="2" style="filter:drop-shadow(0 0 6px ${col}88)"/>`;
        out += `<text x="${W - m.r + 10}" y="${cy + 4}" style="fill:${css('--text')};font-weight:700">${esc(r.value != null ? r.value : fmt(r.est))}</text>`;
        out += `<rect x="0" y="${cy - rh / 2}" width="${W}" height="${rh}" fill="transparent" data-tip="${esc(r.tip || '')}"/>`;
      });
      el.innerHTML = out + '</svg>';
    });
  }

  /* ---------- scatter / reliability ---------- */
  function scatter(el, cfg) {
    mount(el, () => {
      const W = Math.max(280, el.clientWidth), H = cfg.height || 300, m = { l: 44, r: 14, t: 12, b: 36 };
      const iw = W - m.l - m.r, ih = H - m.t - m.b, [x0, x1] = cfg.xDomain, [y0, y1] = cfg.yDomain;
      const X = (v) => m.l + ((v - x0) / (x1 - x0)) * iw, Y = (v) => m.t + ih - ((v - y0) / (y1 - y0)) * ih, fmt = cfg.fmt || ((v) => F.pct(v, 0));
      let out = svgOpen(W, H) + '<g class="grid">';
      niceTicks(y0, y1, 5).filter((t) => t >= y0 && t <= y1).forEach((t) => { out += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(t)}" y2="${Y(t)}"/><text x="${m.l - 7}" y="${Y(t) + 3.5}" text-anchor="end">${esc(fmt(t))}</text>`; });
      niceTicks(x0, x1, 5).filter((t) => t >= x0 && t <= x1).forEach((t) => { out += `<line x1="${X(t)}" x2="${X(t)}" y1="${m.t}" y2="${H - m.b}"/><text x="${X(t)}" y="${H - m.b + 15}" text-anchor="middle">${esc(fmt(t))}</text>`; });
      out += '</g>';
      if (cfg.diag) out += `<line x1="${X(Math.max(x0, y0))}" y1="${Y(Math.max(x0, y0))}" x2="${X(Math.min(x1, y1))}" y2="${Y(Math.min(x1, y1))}" stroke="${css('--amber')}" stroke-dasharray="5 4" opacity=".85"/>`;
      if (cfg.xLabel) out += `<text x="${m.l + iw / 2}" y="${H - 3}" text-anchor="middle">${esc(cfg.xLabel)}</text>`;
      if (cfg.yLabel) out += `<text transform="translate(11 ${m.t + ih / 2}) rotate(-90)" text-anchor="middle">${esc(cfg.yLabel)}</text>`;
      cfg.groups.forEach((g) => {
        if (g.line) { const pts = g.points.slice().sort((a, b) => a.x - b.x).map((p, i) => (i ? 'L' : 'M') + X(p.x).toFixed(1) + ' ' + Y(p.y).toFixed(1)).join(''); out += `<path d="${pts}" fill="none" stroke="${g.color}" stroke-width="1.8" opacity=".6"/>`; }
        g.points.forEach((p) => { out += `<circle class="pt" cx="${X(p.x)}" cy="${Y(p.y)}" r="${p.r || 4.5}" fill="${p.color || g.color}" fill-opacity=".72" stroke="${p.color || g.color}" stroke-width="1.4" data-tip="${esc(p.tip || '')}"/>`; });
      });
      el.innerHTML = out + '</svg>';
    });
  }

  /* ---------- calendar heat map (HTML grid) ---------- */
  function calendar(el, days) {
    if (!days.length) { el.innerHTML = '<div class="empty">No graded days yet.</div>'; return; }
    const map = {}; days.forEach((d) => { map[d.date] = d; });
    const parse = (s) => { const p = s.split('-'); return new Date(Date.UTC(+p[0], +p[1] - 1, +p[2])); };
    const start = parse(days[0].date), end = parse(days[days.length - 1].date);
    const first = new Date(start); first.setUTCDate(first.getUTCDate() - first.getUTCDay());
    const maxU = Math.max(1, ...days.map((d) => Math.abs(d.units || 0))), win = css('--win'), loss = css('--loss');
    // grow the squares when the season is short so the card is not mostly empty
    const weeks = Math.ceil(((end - first) / 86400000 + 1) / 7), avail = Math.max(300, el.clientWidth || 800);
    const cs = Math.max(15, Math.min(34, Math.floor((avail - 8) / weeks) - 4)), pitch = cs + 4;
    let cells = '', months = '', col = 0, lastMonth = -1, dayCount = 0;
    for (let t = new Date(first); t <= end; t.setUTCDate(t.getUTCDate() + 1)) {
      const iso = t.toISOString().slice(0, 10), d = map[iso], dow = t.getUTCDay();
      if (dow === 0) { col++; if (t.getUTCMonth() !== lastMonth) { months += `<span style="left:${(col - 1) * pitch}px">${MONTHS[t.getUTCMonth()]}</span>`; lastMonth = t.getUTCMonth(); } }
      if (t < start) { cells += '<div></div>'; continue; }
      if (!d) { cells += `<div class="cell" style="opacity:.55" data-tip="${esc('<b>' + F.dateY(iso) + '</b><br><span class=k>No graded picks</span>')}"></div>`; continue; }
      const mag = d.priced ? Math.min(1, Math.abs(d.units) / maxU) : Math.min(1, Math.abs(d.w - d.l) / 6), c = (d.priced ? d.units : d.w - d.l) >= 0 ? win : loss;
      const a = 0.28 + mag * 0.72;
      cells += `<div class="cell" style="background:${c};opacity:${a.toFixed(2)}" data-tip="${esc(`<b>${F.dateY(iso)}</b><br>${d.w}-${d.l}${d.p ? '-' + d.p : ''} · <b class="${F.tone(d.units)}">${d.priced ? F.u(d.units) : 'unpriced'}</b>`)}"></div>`;
      dayCount++;
    }
    el.innerHTML = `<div style="--cs:${cs}px"><div class="calmonths" style="width:${(col + 1) * pitch}px">${months}</div><div class="cal">${cells}</div></div>`;
  }

  /* ---------- horizontal bars (HTML) ---------- */
  function hbars(rows, o) {
    o = o || {};
    const max = o.max || Math.max(1e-9, ...rows.map((r) => Math.abs(r.value)));
    return '<div class="hbars">' + rows.map((r) => {
      const w = Math.min(100, (Math.abs(r.value) / max) * 100), col = r.color || (r.value >= 0 ? css('--win') : css('--loss'));
      return `<div class="hbar" data-tip="${esc(r.tip || '')}"><div class="lab">${esc(r.label)}${r.sub ? `<small>${esc(r.sub)}</small>` : ''}</div><div class="trk"><i style="left:0;width:${w}%;background:linear-gradient(90deg,${col}cc,${col})"></i></div><div class="v ${r.cls || ''}">${esc(r.text)}</div></div>`;
    }).join('') + '</div>';
  }

  /* ---------- circular geometry helpers ---------- */
  const polar = (cx, cy, r, a) => [cx + r * Math.cos(a), cy + r * Math.sin(a)];
  const pt = (p) => p[0].toFixed(2) + ' ' + p[1].toFixed(2);
  const TAU = Math.PI * 2, TOP = -Math.PI / 2;
  /** Annular sector (r0 = 0 gives a filled pie wedge). Angles in radians, clockwise from 12 o'clock = TOP. */
  function sectorPath(cx, cy, r0, r1, a0, a1) {
    a1 = Math.min(a1, a0 + TAU - 0.0005);
    const large = a1 - a0 > Math.PI ? 1 : 0, p0 = polar(cx, cy, r1, a0), p1 = polar(cx, cy, r1, a1);
    if (r0 <= 0) return `M${cx} ${cy}L${pt(p0)}A${r1} ${r1} 0 ${large} 1 ${pt(p1)}Z`;
    const q1 = polar(cx, cy, r0, a1), q0 = polar(cx, cy, r0, a0);
    return `M${pt(p0)}A${r1} ${r1} 0 ${large} 1 ${pt(p1)}L${pt(q1)}A${r0} ${r0} 0 ${large} 0 ${pt(q0)}Z`;
  }
  function arcStroke(cx, cy, r, a0, a1) {
    a1 = Math.min(a1, a0 + TAU - 0.0005);
    const large = a1 - a0 > Math.PI ? 1 : 0;
    return `M${pt(polar(cx, cy, r, a0))}A${r} ${r} 0 ${large} 1 ${pt(polar(cx, cy, r, a1))}`;
  }
  const palette = () => [css('--accent'), css('--accent-2'), css('--violet'), css('--win'), css('--amber'), '#ff7eb6', '#7be0c3', '#8ea2ff', '#f59e0b', '#94a3b8'];

  /* ---------- pie / donut with legend, hover lift and centre label ---------- */
  function pie(el, cfg) {
    mount(el, (first) => {
      const W = Math.max(200, el.clientWidth), stack = cfg.legend === 'bottom' || W < 380;
      const size = Math.round(Math.min(cfg.size || 210, stack ? W - 8 : Math.max(140, W * 0.52)));
      const cx = size / 2, cy = size / 2, R = size / 2 - 7, hole = cfg.hole == null ? 0.62 : cfg.hole, r0 = R * hole;
      const S = cfg.slices.filter((s) => isNum(s.value) && s.value > 0), total = S.reduce((a, s) => a + s.value, 0);
      if (!total) { el.innerHTML = '<div class="empty">Nothing to chart yet.</div>'; return; }
      const pal = palette(), gap = S.length > 1 ? 0.018 : 0, vf = cfg.valueFmt || ((v) => F.n(v));
      let a = TOP, svg = `<svg class="chart pie" viewBox="0 0 ${size} ${size}" width="${size}" height="${size}" role="img">`;
      const items = [];
      S.forEach((s, i) => {
        const frac = s.value / total, span = frac * TAU, col = s.color || pal[i % pal.length];
        const a0 = a + gap, a1 = a + span - gap; a += span;
        const tip = s.tip || `<b>${esc(s.label)}</b><br>${esc(vf(s.value))} · ${(frac * 100).toFixed(1)}%`;
        svg += `<path class="sl" d="${sectorPath(cx, cy, r0, R, a0, Math.max(a1, a0 + 0.01))}" fill="${col}" stroke="${css('--panel-solid')}" stroke-width="1.5" style="transform-origin:${cx}px ${cy}px;${first ? `animation:sweep .7s ${i * 70}ms cubic-bezier(.2,.8,.2,1) backwards` : ''}" data-tip="${esc(tip)}"/>`;
        if (cfg.labels !== false && frac >= 0.07) {
          const mid = (a0 + a1) / 2, lr = r0 > 0 ? (r0 + R) / 2 : R * 0.64, p = polar(cx, cy, lr, mid);
          svg += `<text x="${p[0].toFixed(1)}" y="${(p[1] + 4).toFixed(1)}" text-anchor="middle" style="fill:#fff;font-size:${size < 170 ? 10 : 11.5}px;font-weight:750;pointer-events:none;paint-order:stroke;stroke:rgba(0,0,0,.35);stroke-width:2px">${Math.round(frac * 100)}%</text>`;
        }
        items.push({ s, col, frac });
      });
      if (r0 > 0 && cfg.center) {
        svg += `<text x="${cx}" y="${cy + (cfg.center.sub ? 2 : 8)}" text-anchor="middle" style="font-size:${Math.round(size * 0.17)}px;font-weight:780;fill:${css('--text')};letter-spacing:-.03em">${esc(cfg.center.text)}</text>`;
        if (cfg.center.sub) svg += `<text x="${cx}" y="${cy + size * 0.12}" text-anchor="middle" style="font-size:10.5px">${esc(cfg.center.sub)}</text>`;
      }
      svg += '</svg>';
      const legend = items.map(({ s, col, frac }) => `<div class="lg" data-tip="${esc(s.tip || `<b>${esc(s.label)}</b><br>${esc(vf(s.value))} · ${(frac * 100).toFixed(1)}%`)}"><i style="background:${col}"></i><span class="lab">${s.icon || ''}${esc(s.label)}</span><b>${esc(s.text != null ? s.text : vf(s.value))}</b><em>${(frac * 100).toFixed(0)}%</em></div>`).join('');
      el.innerHTML = `<div class="piewrap ${stack ? 'stack' : ''}"><div class="piebox">${svg}</div><div class="pielegend">${legend}</div></div>`;
    });
  }

  /* ---------- semicircle dial with zones, needle and reference tick ---------- */
  function gauge(o) {
    const W = o.size || 230, r = W / 2 - 20, cx = W / 2, cy = r + 18, H = cy + (o.sub ? 50 : 40);
    const lo = o.min, hi = o.max, clamp = (v) => Math.max(lo, Math.min(hi, v)), frac = (v) => (clamp(v) - lo) / (hi - lo);
    const ang = (v) => Math.PI + frac(v) * Math.PI, id = 'gg' + (++uid);
    const zones = o.zones || [{ from: lo, to: hi, color: css('--accent-2') }];
    const valCol = isNum(o.value) ? (zones.find((z) => o.value >= z.from && o.value <= z.to) || zones[zones.length - 1]).color : css('--faint');
    let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" style="width:100%;max-width:${W}px;margin:0 auto" role="img"><defs><filter id="${id}"><feGaussianBlur stdDeviation="3"/></filter></defs>`;
    s += `<path d="${arcStroke(cx, cy, r, Math.PI, TAU)}" fill="none" stroke="${css('--line')}" stroke-width="15" stroke-linecap="round"/>`;
    zones.forEach((z) => { s += `<path d="${arcStroke(cx, cy, r, ang(z.from), ang(z.to) - 0.004)}" fill="none" stroke="${z.color}" stroke-width="15" opacity=".26"/>`; });
    if (isNum(o.value)) {
      const dA = ang(o.value), d = arcStroke(cx, cy, r, Math.PI, Math.max(dA, Math.PI + 0.02));
      s += `<path d="${d}" fill="none" stroke="${valCol}" stroke-width="15" stroke-linecap="round" opacity=".5" filter="url(#${id})"/>`;
      s += `<path d="${d}" fill="none" stroke="${valCol}" stroke-width="15" stroke-linecap="round" pathLength="1" style="stroke-dasharray:1;stroke-dashoffset:1;animation:draw 1.1s cubic-bezier(.3,.7,.2,1) forwards"/>`;
    }
    // minor ticks
    for (let i = 0; i <= 20; i++) { const a = Math.PI + (i / 20) * Math.PI, p0 = polar(cx, cy, r - 14, a), p1 = polar(cx, cy, r - (i % 5 === 0 ? 21 : 18), a); s += `<line x1="${p0[0].toFixed(1)}" y1="${p0[1].toFixed(1)}" x2="${p1[0].toFixed(1)}" y2="${p1[1].toFixed(1)}" stroke="${css('--faint')}" stroke-width="${i % 5 === 0 ? 1.6 : 1}" opacity=".7"/>`; }
    if (isNum(o.ref)) {
      const a = ang(o.ref), p0 = polar(cx, cy, r - 12, a), p1 = polar(cx, cy, r + 11, a), pl = polar(cx, cy, r + 21, a);
      s += `<line x1="${p0[0].toFixed(1)}" y1="${p0[1].toFixed(1)}" x2="${p1[0].toFixed(1)}" y2="${p1[1].toFixed(1)}" stroke="${css('--amber')}" stroke-width="3" stroke-linecap="round"/>`;
      if (o.refLabel) s += `<text x="${pl[0].toFixed(1)}" y="${(pl[1] + 3).toFixed(1)}" text-anchor="middle" style="fill:${css('--amber')};font-size:9.5px;font-weight:700">${esc(o.refLabel)}</text>`;
    }
    if (isNum(o.value)) {
      const deg = (frac(o.value) * 180) - 90;
      s += `<g style="transform-origin:${cx}px ${cy}px;--a:${deg}deg;animation:needle 1.2s cubic-bezier(.2,.9,.25,1.05) both"><path d="M${cx - 4} ${cy}L${cx} ${cy - r + 24}L${cx + 4} ${cy}Z" fill="${css('--text')}"/></g>`;
    }
    s += `<circle cx="${cx}" cy="${cy}" r="8" fill="${css('--panel-solid')}" stroke="${css('--text')}" stroke-width="2.5"/>`;
    s += `<text x="${cx - r}" y="${cy + 16}" text-anchor="middle" style="font-size:10px">${esc(o.minLabel != null ? o.minLabel : lo)}</text><text x="${cx + r}" y="${cy + 16}" text-anchor="middle" style="font-size:10px">${esc(o.maxLabel != null ? o.maxLabel : hi)}</text>`;
    s += `<text x="${cx}" y="${cy + 33}" text-anchor="middle" style="font-size:${Math.round(W * 0.115)}px;font-weight:790;fill:${valCol};letter-spacing:-.03em">${esc(o.text)}</text>`;
    if (o.sub) s += `<text x="${cx}" y="${cy + 47}" text-anchor="middle" style="font-size:10.5px">${esc(o.sub)}</text>`;
    return s + '</svg>';
  }

  /* ---------- concentric progress rings ("activity rings") with legend ---------- */
  function rings(o) {
    const size = o.size || 200, cx = size / 2, cy = size / 2, th = o.thickness || 13, gapR = 4, pal = palette(), R0 = size / 2 - 8;
    let s = `<svg class="chart" viewBox="0 0 ${size} ${size}" style="width:100%;max-width:${size}px" role="img">`, lg = '';
    o.rows.forEach((row, i) => {
      const r = R0 - i * (th + gapR), col = row.color || pal[i % pal.length], v = Math.max(0, Math.min(1, row.value || 0));
      s += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${css('--line')}" stroke-width="${th}"/>`;
      s += `<path d="${arcStroke(cx, cy, r, TOP, TOP + Math.max(v, 0.004) * TAU)}" fill="none" stroke="${col}" stroke-width="${th}" stroke-linecap="round" pathLength="1" style="stroke-dasharray:1;stroke-dashoffset:1;animation:draw ${0.9 + i * 0.2}s cubic-bezier(.3,.7,.2,1) forwards;filter:drop-shadow(0 0 5px ${col}77)" data-tip="${esc(row.tip || `<b>${esc(row.label)}</b><br>${esc(row.text || '')}`)}"/>`;
      if (isNum(row.ref)) { const a = TOP + row.ref * TAU, p0 = polar(cx, cy, r - th / 2 - 1, a), p1 = polar(cx, cy, r + th / 2 + 1, a); s += `<line x1="${p0[0].toFixed(1)}" y1="${p0[1].toFixed(1)}" x2="${p1[0].toFixed(1)}" y2="${p1[1].toFixed(1)}" stroke="${css('--amber')}" stroke-width="2.4"/>`; }
      lg += `<div class="lg" data-tip="${esc(row.tip || '')}"><i style="background:${col}"></i><span class="lab">${esc(row.label)}</span><b>${esc(row.text || '')}</b></div>`;
    });
    if (o.center) s += `<text x="${cx}" y="${cy + 5}" text-anchor="middle" style="font-size:${Math.round(size * 0.12)}px;font-weight:780;fill:${css('--text')}">${esc(o.center)}</text>`;
    return `<div class="piewrap"><div class="piebox">${s}</svg></div><div class="pielegend">${lg}</div></div>`;
  }

  /* ---------- small progress ring (KPI-sized) ---------- */
  function miniRing(v, o) {
    o = o || {}; const S = o.size || 64, r = S / 2 - 6, cx = S / 2, col = o.color || css('--accent-2'), f = Math.max(0, Math.min(1, v || 0));
    return `<svg class="chart" viewBox="0 0 ${S} ${S}" style="width:${S}px;height:${S}px" role="img"><circle cx="${cx}" cy="${cx}" r="${r}" fill="none" stroke="${css('--line')}" stroke-width="6"/><path d="${arcStroke(cx, cx, r, TOP, TOP + Math.max(f, 0.01) * TAU)}" fill="none" stroke="${col}" stroke-width="6" stroke-linecap="round" pathLength="1" style="stroke-dasharray:1;stroke-dashoffset:1;animation:draw 1s ease forwards;filter:drop-shadow(0 0 4px ${col}88)"/><text x="${cx}" y="${cx + 4}" text-anchor="middle" style="font-size:${S * 0.22}px;font-weight:760;fill:${css('--text')}">${esc(o.text != null ? o.text : Math.round(f * 100) + '%')}</text></svg>`;
  }

  /* extra keyframes used by bars(), pie() and gauge() */
  const st = document.createElement('style');
  st.textContent = '@keyframes growY{from{transform:scaleY(0)}to{transform:scaleY(1)}}@keyframes sweep{from{opacity:0;transform:scale(.82) rotate(-28deg)}to{opacity:1;transform:none}}@keyframes needle{from{transform:rotate(-90deg)}to{transform:rotate(var(--a))}}';
  document.head.appendChild(st);

  window.F = F; window.C = { line, bars, spark, ring, forest, scatter, calendar, hbars, pie, gauge, rings, miniRing, palette, showTip, hideTip, css, niceTicks };
})();
