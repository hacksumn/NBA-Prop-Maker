/* app.js — Command Center views. Data arrives pre-aggregated in window.__DASH__ (see build_dashboard.py). */
(function () {
  'use strict';
  const D = window.__DASH__ || {};
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  const esc = F.esc;
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : v; } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* storage may be blocked */ } },
  };
  const state = {
    scope: store.get('nba.scope', 'season'), view: 'overview', eq: 'units', daily: 'wl', slipMode: 'unique',
    run: -1, feat: null, lane: 'holdout', clvKind: 'picks', teamMode: 'teams',
    ex: { q: '', prop: '', dir: '', res: '', era: '', sort: 'd', desc: true, page: 0 },
  };
  if (!D.scopes || !D.scopes[state.scope]) state.scope = 'season';
  const scopeData = () => D.scopes[state.scope];
  const TODAY = D.today || {}, SLIPS = D.slips || {}, GATES = D.gates || {}, CLV = D.clv || {}, MODELS = D.models || {}, HEALTH = D.health || {};
  const ERA_LABEL = {}; (D.eras || []).forEach((e) => { ERA_LABEL[e.key] = e.label; });

  /* ---------- team logos ----------
     Logos embedded by the builder (D.logos, data: URIs) work offline and are stored once, in a generated
     stylesheet. A team without an embedded logo falls back to the official NBA CDN image, and to a
     text badge when that cannot load either. */
  const TEAMS = D.teams || {};
  const LOGOS = D.logos || {};
  const logoUrl = (id) => `https://cdn.nba.com/logos/nba/${id}/global/L/logo.svg`;
  (function injectLogoStyles() {
    const rules = Object.keys(LOGOS).filter((a) => /^[A-Z]{3}$/.test(a) && /^data:image\/svg\+xml;base64,[A-Za-z0-9+/=]+$/.test(LOGOS[a]))
      .map((a) => `.tlogo.e-${a} i{background-image:url("${LOGOS[a]}")}`);
    if (!rules.length) return;
    const st = document.createElement('style');
    st.textContent = rules.join('\n');
    document.head.appendChild(st);
  })();
  const hasEmbedded = (a) => Object.prototype.hasOwnProperty.call(LOGOS, a) && /^[A-Z]{3}$/.test(a);
  function logo(abbr, size) {
    if (!abbr) return '';
    const t = TEAMS[abbr] || {}, s = size || 24, emb = hasEmbedded(abbr);
    const pic = emb ? `<i role="img" aria-label="${esc(abbr)}"></i>`
      : (t.id ? `<img src="${logoUrl(t.id)}" alt="${esc(abbr)}" loading="lazy" decoding="async" referrerpolicy="no-referrer">` : '');
    return `<span class="tlogo${emb ? ' e-' + abbr : ''}${emb || t.id ? '' : ' nologo'}" style="--s:${s}px;--c:${t.color || '#777'}" data-tip="${esc('<b>' + (t.name || abbr) + '</b>')}">${pic}<b>${esc(abbr)}</b></span>`;
  }
  const team = (abbr, size) => (abbr ? `<span class="tm">${logo(abbr, size || 22)}<span class="ab">${esc(abbr)}</span></span>` : '<span class="t-muted">–</span>');
  // image errors do not bubble, so listen in the capture phase and flip the badge on
  document.addEventListener('error', (e) => { const t = e.target; if (t && t.tagName === 'IMG' && t.parentNode && t.parentNode.classList && t.parentNode.classList.contains('tlogo')) t.parentNode.classList.add('nologo'); }, true);

  /* ---------- optional wallpaper (embedded by the builder as a data: URI; see style.css) ---------- */
  const WALL = (typeof D.wallpaper === 'string' && /^data:image\/(jpeg|png|webp);base64,[A-Za-z0-9+/=]+$/.test(D.wallpaper)) ? D.wallpaper : null;
  // header logo (scripts/dashboard_assets/brand.*): only a plain base64 image data URI is ever used; anything else keeps the text title
  const BRAND = (typeof D.brand === 'string' && /^data:image\/(jpeg|png|webp);base64,[A-Za-z0-9+/=]+$/.test(D.brand)) ? D.brand : null;
  if (WALL) document.documentElement.style.setProperty('--wall', `url("${WALL}")`);
  let wallOn = store.get('nba.wall', 'on') !== 'off';
  function applyWall() {
    document.documentElement.dataset.wall = WALL && wallOn ? 'on' : 'off';
    const b = document.getElementById('wallbtn');
    if (b) b.setAttribute('aria-pressed', String(wallOn));
  }

  /* ---------- icons ---------- */
  const ICON = {
    overview: '<rect x="3" y="3" width="7" height="9" rx="2"/><rect x="14" y="3" width="7" height="5" rx="2"/><rect x="14" y="12" width="7" height="9" rx="2"/><rect x="3" y="16" width="7" height="5" rx="2"/>',
    today: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.2"/>',
    perf: '<path d="M3 17l6-6 4 4 8-9"/><path d="M15 6h6v6"/>',
    explorer: '<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/>',
    gates: '<path d="M3 5h18l-7 8v6l-4 2v-8z"/>',
    clv: '<path d="M3 12h4l3-8 4 16 3-8h4"/>',
    models: '<path d="M9 3h6M10 3v6l-5 9a2 2 0 0 0 2 3h10a2 2 0 0 0 2-3l-5-9V3"/><path d="M8 15h8"/>',
    slips: '<path d="M6 3h12v18l-3-2-3 2-3-2-3 2z"/><path d="M9 8h6M9 12h6"/>',
    health: '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/><path d="M9 12l2 2 4-4"/>',
    decisions: '<path d="M5 4h11a3 3 0 0 1 3 3v13H8a3 3 0 0 1-3-3z"/><path d="M8 4v13"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    moon: '<path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/>',
    image: '<rect x="3" y="4" width="18" height="16" rx="3"/><circle cx="9" cy="10" r="1.8"/><path d="M21 16.5l-5-5-9.5 8.5"/>',
  };
  const svgi = (n) => `<svg viewBox="0 0 24 24">${ICON[n] || ''}</svg>`;

  /* ---------- 3D side-menu icons ----------
     Pure CSS + SVG (no library). Each icon is a stack of identical solid shapes at different depths: a dark
     extrusion (8 slices), a gradient face and a gloss highlight, inside a perspective box. Because the slices are
     real layers in 3D space, tilting the icon shows true side depth. %M% is the main paint and %A% the accent
     (white detail on the face, dark on the slices, hidden on the gloss). c = [light, base] gradient, d = extrusion. */
  const ICON3D = {
    overview: { c: ['#ff5c5c', '#ff2a2a'], d: '#6b0707', s: '<rect x="3" y="3" width="8.2" height="10.4" rx="2.4" fill="%M%"/><rect x="12.8" y="3" width="8.2" height="6.2" rx="2.4" fill="%M%"/><rect x="12.8" y="10.8" width="8.2" height="10.2" rx="2.4" fill="%M%"/><rect x="3" y="15" width="8.2" height="6" rx="2.4" fill="%M%"/>' },
    today: { c: ['#ffffff', '#cfcfcf'], d: '#4a4a4a', s: '<circle cx="12" cy="12" r="9.6" fill="%M%"/><circle cx="12" cy="12" r="6.6" fill="%A%"/><circle cx="12" cy="12" r="3.9" fill="%M%"/><circle cx="12" cy="12" r="1.5" fill="%A%"/>' },
    perf: { c: ['#ff5c5c', '#ff2a2a'], d: '#6b0707', s: '<rect x="3" y="13.2" width="4.4" height="7.8" rx="1.5" fill="%M%"/><rect x="9.8" y="8.6" width="4.4" height="12.4" rx="1.5" fill="%M%"/><rect x="16.6" y="3" width="4.4" height="18" rx="1.5" fill="%M%"/>' },
    explorer: { c: ['#ffffff', '#cfcfcf'], d: '#4a4a4a', s: '<circle cx="10.4" cy="10.4" r="6.6" fill="none" stroke="%M%" stroke-width="3.2"/><path d="M15.4 15.4l5 5" fill="none" stroke="%M%" stroke-width="3.6" stroke-linecap="round"/><path d="M7.4 9.4a3.4 3.4 0 0 1 2.6-2.1" fill="none" stroke="%A%" stroke-width="1.5" stroke-linecap="round"/>' },
    gates: { c: ['#ff5c5c', '#ff2a2a'], d: '#6b0707', s: '<path d="M3.2 4.2h17.6L14.2 12v6.6l-4.4 2.2V12z" fill="%M%" stroke="%M%" stroke-width="1.6" stroke-linejoin="round"/>' },
    clv: { c: ['#ffffff', '#cfcfcf'], d: '#4a4a4a', s: '<path d="M2.6 12.8h4.2l2.6-7.4 4.6 13.2 2.6-5.8h4.8" fill="none" stroke="%M%" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>' },
    models: { c: ['#ff5c5c', '#ff2a2a'], d: '#6b0707', s: '<path d="M9.4 3h5.2v5.4l4.7 8.3a3.1 3.1 0 0 1-2.7 4.6H7.4a3.1 3.1 0 0 1-2.7-4.6l4.7-8.3z" fill="%M%" stroke="%M%" stroke-width="1" stroke-linejoin="round"/><rect x="8" y="2" width="8" height="2.6" rx="1.3" fill="%M%"/><path d="M7.2 15.4h9.6" fill="none" stroke="%A%" stroke-width="2" stroke-linecap="round"/><circle cx="10.6" cy="18.2" r="1" fill="%A%"/><circle cx="14" cy="17.6" r=".8" fill="%A%"/>' },
    slips: { c: ['#ffffff', '#cfcfcf'], d: '#4a4a4a', s: '<path d="M6.6 2.6h10.8A1.6 1.6 0 0 1 19 4.2v17.2l-2.5-1.7-2.2 1.7-2.3-1.7-2.3 1.7-2.2-1.7L5 21.4V4.2a1.6 1.6 0 0 1 1.6-1.6z" fill="%M%" stroke="%M%" stroke-width=".8" stroke-linejoin="round"/><path d="M8.6 8h6.8M8.6 11.6h6.8M8.6 15.2h4" fill="none" stroke="%A%" stroke-width="1.8" stroke-linecap="round"/>' },
    health: { c: ['#ff5c5c', '#ff2a2a'], d: '#6b0707', s: '<path d="M12 2.4l8 3v6.4c0 4.9-3.3 8.4-8 9.8-4.7-1.4-8-4.9-8-9.8V5.4z" fill="%M%" stroke="%M%" stroke-width="1.2" stroke-linejoin="round"/><path d="M8 12.2l2.9 2.9 5.2-5.7" fill="none" stroke="%A%" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>' },
    decisions: { c: ['#ffffff', '#cfcfcf'], d: '#4a4a4a', s: '<path d="M6 2.8h11.6A1.4 1.4 0 0 1 19 4.2v14.6a1.4 1.4 0 0 1-1.4 1.4H7a3 3 0 0 1-3-3V4.8a2 2 0 0 1 2-2z" fill="%M%" stroke="%M%" stroke-width=".8" stroke-linejoin="round"/><path d="M8 2.8v17.4" fill="none" stroke="%A%" stroke-width="1.6"/><path d="M11 8h5M11 11.4h3.6" fill="none" stroke="%A%" stroke-width="1.8" stroke-linecap="round"/>' },
  };
  const darker = (hex, k) => '#' + [1, 3, 5].map((i) => Math.round(parseInt(hex.slice(i, i + 2), 16) * k).toString(16).padStart(2, '0')).join('');
  const layer3d = (shapes, main, accent, cls, style) => `<span class="sl ${cls}" style="${style}"><svg viewBox="0 0 24 24" focusable="false">${shapes.replace(/%M%/g, main).replace(/%A%/g, accent)}</svg></span>`;
  function defs3d() {   // one hidden SVG holds every gradient (shared by all icons)
    const grads = Object.keys(ICON3D).map((k) => `<linearGradient id="g3-${k}" gradientUnits="userSpaceOnUse" x1="4" y1="3" x2="20" y2="21"><stop offset="0" stop-color="${ICON3D[k].c[0]}"/><stop offset="1" stop-color="${ICON3D[k].c[1]}"/></linearGradient>`).join('');
    return `<svg class="defs3d" width="0" height="0" aria-hidden="true" focusable="false"><defs>${grads}<linearGradient id="g3-gloss" gradientUnits="userSpaceOnUse" x1="4" y1="2" x2="13" y2="16"><stop offset="0" stop-color="#fff" stop-opacity=".85"/><stop offset=".55" stop-color="#fff" stop-opacity=".12"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient></defs></svg>`;
  }
  function icon3d(name, index) {
    const d = ICON3D[name];
    if (!d) return `<span class="ico" aria-hidden="true">${svgi(name)}</span>`;   // unknown icon: flat fallback, never blank
    const SLICES = 8;
    let slices = '';
    for (let k = SLICES; k >= 1; k--) {   // deepest first, so painting order matches depth
      const dark = k <= 3 ? d.d : darker(d.d, 0.72);
      slices += layer3d(d.s, dark, dark, 'ex', `--z:${(-k * 1.35).toFixed(2)}px`);
    }
    const face = layer3d(d.s, `url(#g3-${name})`, '#ffffff', 'face', '--z:2px');
    const gloss = layer3d(d.s, 'url(#g3-gloss)', 'none', 'gloss', '--z:2.6px');
    return `<span class="ico" aria-hidden="true" style="--i:${index};--glow:${d.c[1]}"><span class="i3d"><span class="spin"><span class="tilt">${slices}${face}${gloss}</span></span></span><i class="ground"></i></span>`;
  }

  /* ---------- tiny templating helpers ---------- */
  const kpi = (o) => `<div class="kpi ${o.tone || 'info'}"><div class="lab">${esc(o.label)}</div><div class="val ${o.vcls || ''}">${
    o.valHtml != null ? o.valHtml : (o.count != null && isFinite(o.count) ? `<span data-count="${o.count}" data-fmt="${o.fmt}" data-dec="${o.dec || 0}">${esc(o.val)}</span>` : esc(o.val))
  }${o.unit ? `<small>${esc(o.unit)}</small>` : ''}</div><div class="sub">${o.sub || ''}</div>${o.spark || ''}</div>`;
  const panel = (o) => `<section class="panel ${o.cls || ''}"><div class="head"><div><h2><span class="dot"></span>${esc(o.title)}</h2>${o.sub ? `<div class="psub">${o.sub}</div>` : ''}</div><div class="tools">${o.tools || ''}</div></div>${o.body || ''}</section>`;
  const seg = (id, opts, val) => `<div class="seg" data-seg="${id}">${opts.map((o) => `<button data-v="${esc(o[0])}" aria-pressed="${o[0] === val}">${esc(o[1])}</button>`).join('')}</div>`;
  function onSeg(root, id, cb) {
    const el = $(`[data-seg="${id}"]`, root); if (!el) return;
    el.addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; $$('button', el).forEach((x) => x.setAttribute('aria-pressed', String(x === b))); cb(b.dataset.v); });
  }
  const chip = (t, k) => `<span class="chip ${k || ''}">${esc(t)}</span>`;
  const rec = (s) => `<span class="t-win">${s.w}</span><span class="sep">–</span><span class="t-loss">${s.l}</span>${s.p ? `<span class="sep">–</span><span class="t-amber">${s.p}</span>` : ''}`;
  const resChip = (r) => r === 'WIN' ? chip('WIN', 'win') : r === 'LOSS' ? chip('LOSS', 'loss') : r === 'PUSH' ? chip('PUSH', 'amber') : r === 'DNP' ? chip('DNP', '') : chip('PENDING', 'info pulse');
  const STALE_HOURS = 26;   // the page is rebuilt once a day; older than this means a rebuild was missed
  const unavail = (name) => `<div class="empty">${esc(name)} is unavailable. Check the warnings banner and the builder output.</div>`;
  const hitBar = (s) => {
    if (s.hit == null) return '<span class="t-muted">–</span>';
    const col = s.hit >= 0.5 ? 'var(--win)' : 'var(--loss)';
    return `<span class="barcell"><span class="minibar"><i style="width:${(s.hit * 100).toFixed(0)}%;background:${col}"></i></span><span class="num">${F.pct(s.hit)}</span></span>`;
  };
  const ciTxt = (s) => (s.lo != null ? `${F.pct(s.lo, 0)}–${F.pct(s.hi, 0)}` : '–');
  const lowN = (n, k) => (n < (k || 30) ? ' <span class="chip amber" style="padding:1px 6px;font-size:10px">low n</span>' : '');
  const tipRows = (title, rows) => `<b>${esc(title)}</b>` + rows.map((r) => `<br><span class="k">${esc(r[0])}</span> ${r[1]}`).join('');

  function countUp(root) {
    const reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
    $$('[data-count]', root).forEach((el) => {
      const to = parseFloat(el.dataset.count), kind = el.dataset.fmt, dec = +(el.dataset.dec || 0);
      const fmt = (v) => (kind === 'pct' ? F.pct(v, dec) : kind === 'u' ? F.u(v, dec) : kind === 'pts' ? F.pts(v, dec) : kind === 'money' ? F.sgnMoney(v) : F.n(v, dec));
      if (reduce || !isFinite(to)) { el.textContent = fmt(to); return; }
      const t0 = performance.now(), dur = 850;
      const tick = (t) => { const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3); el.textContent = fmt(to * e); if (k < 1) requestAnimationFrame(tick); else el.textContent = fmt(to); };
      requestAnimationFrame(tick);
    });
  }

  /* =====================================================================
     OVERVIEW
     ===================================================================== */
  function vOverview(root) {
    const sc = scopeData(), s = sc.summary, ser = sc.series, days = ser.days;
    const edge = s.priced_hit != null && s.break_even != null ? s.priced_hit - s.break_even : null;
    const g = ser.green_days, r = ser.red_days;
    const pricedNote = state.scope === 'season' ? `${s.priced_n} priced bets` : `priced subset only (${s.priced_n} bets)`;
    const kp = [
      kpi({ label: 'Record', valHtml: rec(s), tone: 'info', sub: `${F.pct(s.hit)} hit · ${s.dnp} DNP · ${s.pending} pending`, spark: C.spark(days.map((d) => d.cum_w - d.cum_l), { color: C.css('--accent-2') }) }),
      kpi({ label: 'Net units', count: s.units, fmt: 'u', dec: 2, val: F.u(s.units), vcls: F.tone(s.units), tone: (s.units || 0) >= 0 ? 'win' : 'loss', sub: `flat 1u at logged prices · ${pricedNote}`, spark: C.spark(days.map((d) => d.cum_units), { color: (s.units || 0) >= 0 ? C.css('--win') : C.css('--loss') }) }),
      kpi({ label: 'ROI per bet', count: s.roi, fmt: 'pct', dec: 1, val: F.pct(s.roi), vcls: F.tone(s.roi), tone: (s.roi || 0) >= 0 ? 'win' : 'loss', sub: `avg model ${F.pct(s.avg_model_prob, 0)} vs market ${F.pct(s.avg_no_vig, 0)}` }),
      kpi({ label: 'Edge vs break-even', count: edge, fmt: 'pts', dec: 1, val: F.pts(edge), vcls: F.tone(edge), tone: (edge || 0) >= 0 ? 'violet' : 'loss', sub: `hit ${F.pct(s.priced_hit)} vs break-even ${F.pct(s.break_even)}` }),
      kpi({ label: 'Drawdown', val: F.u(-ser.current_drawdown), vcls: ser.current_drawdown > 0.01 ? 't-loss' : 't-win', tone: 'amber', sub: `max ${ser.max_drawdown.toFixed(2)}u from peak` }),
      kpi({ label: 'Green days', valHtml: `<span class="t-win">${g}</span><small>/ ${g + r + ser.even_days}</small>`, tone: 'win', sub: `${r} red · ${ser.even_days} even${ser.streak.type ? ` · now ${ser.streak.type}${ser.streak.len}` : ''}` }),
    ];
    const isSeason = state.scope === 'season';
    root.innerHTML = `
      <div class="kpis stagger">${kp.join('')}</div>
      <div class="grid stagger">
        ${panel({ title: 'Dial deck', cls: 'c12', sub: `${esc(sc.label)} · needles show where each number sits against its break-even; amber ticks mark break-even`, body: dialDeck(sc) })}
      </div>
      <div class="grid stagger">
        ${panel({ title: 'Equity curve', cls: 'c8', sub: `${esc(sc.label)} · cumulative flat 1u at logged prices (priced bets only)`, tools: seg('eq', [['units', 'Units'], ['hit', 'Hit rate'], ['dd', 'Drawdown']], state.eq), body: '<div id="ov-eq"></div>' })}
        ${panel({ title: 'Hit rate', cls: 'c4', sub: 'Ring = observed. Outer arc = 95% Wilson interval. Amber tick = average break-even.', body: `
          <div class="ringwrap"><div id="ov-ring"></div><div>
            <div class="stat-row" style="margin:0"><div><b class="t-win">${s.w}</b><span>wins</span></div><div><b class="t-loss">${s.l}</b><span>losses</span></div></div>
            <div class="stat-row"><div><b>${s.p}</b><span>push</span></div><div><b>${s.dnp}</b><span>DNP</span></div></div>
          </div></div>
          <div class="hr"></div><div class="note">${s.n ? `With <b>${s.n}</b> graded picks the interval is <b>${ciTxt(s)}</b>. ${s.lo != null && s.break_even != null ? (s.lo > s.break_even ? 'The whole interval sits above break-even.' : 'The interval still includes break-even, so luck cannot be ruled out yet.') : ''}` : 'No graded picks in this scope yet.'}</div>` })}
      </div>
      <div class="grid stagger">
        ${panel({ title: 'Daily results', cls: 'c6', sub: 'Wins up, losses down. Hover for the slate.', tools: seg('daily', [['wl', 'W / L'], ['units', 'Units']], state.daily), body: '<div id="ov-daily"></div>' })}
        ${panel({ title: 'Lane scoreboard', cls: 'c6', sub: `${esc(sc.label)} · rings = hit rate (amber tick = break-even), bars = units`, body: `<div id="ov-lane-rings"></div><div class="hr"></div>${laneBars(sc.by_prop)}` })}
      </div>
      <div class="grid stagger">
        ${panel({ title: 'Result mix', cls: 'c4', sub: 'Every pick in scope by outcome', body: '<div id="ov-mix"></div>' })}
        ${panel({ title: 'Picks by lane', cls: 'c4', sub: 'Share of graded picks', body: '<div id="ov-lanepie"></div>' })}
        ${panel({ title: 'Picks by team', cls: 'c4', sub: 'Team of the player picked (graded)', body: '<div id="ov-teampie"></div>' })}
      </div>
      <div class="grid stagger">
        ${panel({ title: 'What the data is saying', cls: 'c7', sub: 'Every sentence is computed from your files at build time.', body: `<div class="insights">${(D.insights || []).map((i) => `<div class="insight ${i.tone}"><div class="ic">${i.tone === 'good' ? '✓' : i.tone === 'bad' ? '✕' : i.tone === 'warn' ? '!' : 'i'}</div><div><b>${esc(i.title)}</b><p>${esc(i.text)}</p></div></div>`).join('') || unavail('Insights')}</div>` })}
        ${panel({ title: 'Era ladder', cls: 'c5', sub: 'Hit rate by era with 95% intervals. Line = −110 break-even (52.4%).', body: '<div id="ov-era"></div><div class="note" style="margin-top:8px">Backfilled rows were reconstructed after the fact (DEC-036) and are not a forward record. Eras with different books have different break-evens.</div>' })}
      </div>
      <div class="grid stagger">
        ${panel({ title: 'Latest card', cls: 'c12', sub: `${esc(F.dateY(TODAY.card_date))} · ${TODAY.n || 0} picks · ${TODAY.pending || 0} awaiting grades`, tools: '<button class="btn sm" data-go="today">Open card →</button>', body: miniCard() })}
      </div>`;
    drawEq(); drawDaily();
    onSeg(root, 'eq', (v) => { state.eq = v; drawEq(); });
    onSeg(root, 'daily', (v) => { state.daily = v; drawDaily(); });
    $('#ov-ring').innerHTML = C.ring({ value: s.hit || 0, ci: [s.lo, s.hi], ref: s.break_even, text: F.pct(s.hit), sub: `${s.n} graded`, color: C.css('--win'), size: 150 });
    const lanes = sc.by_prop_only || [];
    $('#ov-lane-rings').innerHTML = lanes.length ? C.rings({ size: 190, rows: lanes.slice(0, 5).map((r, i) => ({ label: `${r.prop} · ${r.w}-${r.l}`, value: r.hit, ref: r.break_even, text: F.pct(r.hit, 0), tip: tipRows(r.prop, [['Record', `${r.w}-${r.l}`], ['Hit', `${F.pct(r.hit)} (CI ${ciTxt(r)})`], ['Break-even', F.pct(r.break_even)], ['Units', F.u(r.units)]]) })), center: F.pct(s.hit, 0) }) : '<div class="empty">No lanes yet.</div>';
    C.pie($('#ov-mix'), { size: 190, center: { text: String(s.n), sub: 'graded' }, slices: [
      { label: 'Wins', value: s.w, color: C.css('--win') }, { label: 'Losses', value: s.l, color: C.css('--loss') },
      { label: 'Push', value: s.p, color: C.css('--amber') }, { label: 'DNP (void)', value: s.dnp, color: C.css('--faint') },
      { label: 'Pending', value: s.pending, color: C.css('--accent-2') }] });
    C.pie($('#ov-lanepie'), { size: 190, hole: 0, legend: 'bottom', slices: lanes.map((r) => ({ label: `${r.prop}`, value: r.n, text: `${r.w}-${r.l}`, tip: tipRows(r.prop, [['Graded', r.n], ['Record', `${r.w}-${r.l}`], ['Hit', F.pct(r.hit)], ['Units', F.u(r.units)]]) })) });
    const tl = (sc.teams_all || []).filter((t) => t.n > 0), top = tl.slice(0, 7), rest = tl.slice(7);
    C.pie($('#ov-teampie'), { size: 190, center: { text: String(tl.length), sub: 'teams' }, slices: [...top.map((t) => ({ label: t.team, icon: logo(t.team, 18), value: t.n, text: `${t.w}-${t.l}`, tip: tipRows((TEAMS[t.team] || {}).name || t.team, [['Graded', t.n], ['Record', `${t.w}-${t.l}`], ['Units', F.u(t.units)]]) })), ...(rest.length ? [{ label: `${rest.length} others`, value: rest.reduce((a, t) => a + t.n, 0), text: `${rest.reduce((a, t) => a + t.w, 0)}-${rest.reduce((a, t) => a + t.l, 0)}`, color: C.css('--faint') }] : [])] });
    const eras = (D.eras || []).filter((e) => e.n > 0);
    C.forest($('#ov-era'), { rows: eras.map((e) => ({ label: e.label, sub: `n=${e.n} · ${e.w}-${e.l}`, est: e.hit, lo: e.lo, hi: e.hi, tip: tipRows(e.label, [['Record', `${e.w}-${e.l}`], ['Hit', F.pct(e.hit)], ['95% CI', ciTxt(e)], ['Note', esc(e.note)]]) })), domain: [0.3, 0.9], ref: 0.5238, rowH: 40, labelW: 150 });
    countUp(root);
  }
  const dial = (label, gaugeSvg, note) => `<div class="dial"><div class="dl">${esc(label)}</div>${gaugeSvg}<div class="dn">${note || ''}</div></div>`;
  function dialDeck(sc) {
    const s = sc.summary, ser = sc.series, be = s.break_even, red = C.css('--loss'), amb = C.css('--amber'), grn = C.css('--win');
    const edge = s.priced_hit != null && be != null ? s.priced_hit - be : null, days = ser.green_days + ser.red_days + ser.even_days;
    const hitZones = be != null ? [{ from: 0.3, to: be, color: red }, { from: be, to: Math.min(0.8, be + 0.04), color: amb }, { from: Math.min(0.8, be + 0.04), to: 0.8, color: grn }] : undefined;
    return `<div class="dials">
      ${dial('Hit rate', C.gauge({ min: 0.3, max: 0.8, value: s.hit, ref: be, refLabel: be != null ? 'B/E' : null, zones: hitZones, text: F.pct(s.hit), sub: `${s.n} graded`, minLabel: '30%', maxLabel: '80%' }), `95% interval ${ciTxt(s)}`)}
      ${dial('ROI per bet', C.gauge({ min: -0.3, max: 0.3, value: s.roi, ref: 0, refLabel: '0', zones: [{ from: -0.3, to: 0, color: red }, { from: 0, to: 0.3, color: grn }], text: F.pct(s.roi), sub: `${s.priced_n} priced bets`, minLabel: '−30%', maxLabel: '+30%' }), 'flat 1u at the logged price')}
      ${dial('Edge vs break-even', C.gauge({ min: -0.12, max: 0.12, value: edge, ref: 0, refLabel: '0', zones: [{ from: -0.12, to: 0, color: red }, { from: 0, to: 0.12, color: grn }], text: F.pts(edge), sub: 'priced hit − break-even', minLabel: '−12', maxLabel: '+12' }), `hit ${F.pct(s.priced_hit)} vs ${F.pct(be)}`)}
      ${dial('Green days', C.gauge({ min: 0, max: 1, value: days ? ser.green_days / days : null, ref: 0.5, refLabel: '50%', zones: [{ from: 0, to: 0.5, color: red }, { from: 0.5, to: 1, color: grn }], text: days ? `${ser.green_days}/${days}` : '–', sub: 'profitable slates', minLabel: '0', maxLabel: 'all' }), `${ser.red_days} red · ${ser.even_days} even`)}
    </div>`;
  }
  function laneBars(rows) {
    if (!rows || !rows.length) return '<div class="empty">No lanes yet.</div>';
    const priced = rows.some((r) => r.units != null);
    return C.hbars(rows.slice(0, 8).map((r) => ({
      label: `${r.prop} ${r.direction}`, sub: `${r.w}-${r.l} · ${F.pct(r.hit, 0)} hit`, value: priced ? (r.units || 0) : (r.hit || 0) - 0.5,
      text: priced ? F.u(r.units) : F.pct(r.hit), cls: priced ? F.tone(r.units) : '',
      tip: tipRows(`${r.prop} ${r.direction}`, [['Record', `${r.w}-${r.l}`], ['Hit', `${F.pct(r.hit)} (CI ${ciTxt(r)})`], ['Units', F.u(r.units)], ['ROI', F.pct(r.roi)]]),
    })));
  }
  function drawEq() {
    const ser = scopeData().series, days = ser.days, el = $('#ov-eq'); if (!el) return;
    if (!days.length) { el.innerHTML = '<div class="empty">No graded days in this scope.</div>'; return; }
    const labels = days.map((d) => F.date(d.date)), sp = (d) => `<b>${F.dateY(d.date)}</b><br>Day ${d.w}-${d.l}${d.p ? '-' + d.p : ''} · <b class="${F.tone(d.units)}">${F.u(d.units)}</b>`;
    if (state.eq === 'hit') {
      const be = scopeData().summary.break_even;
      C.line(el, { height: 270, labels, yFmt: (v) => F.pct(v, 0), zero: false, refs: be ? [{ y: be, label: 'break-even', color: C.css('--amber') }] : [], series: [{ color: C.css('--accent-2'), area: true, data: days.map((d) => ({ y: d.cum_hit })) }], tip: (i) => sp(days[i]) + `<br><span class="k">Cumulative hit</span> <b>${F.pct(days[i].cum_hit)}</b> (${days[i].cum_w}-${days[i].cum_l})` });
    } else if (state.eq === 'dd') {
      C.line(el, { height: 270, labels, yFmt: (v) => v.toFixed(1) + 'u', series: [{ color: C.css('--loss'), area: true, data: days.map((d) => ({ y: -d.dd })) }], tip: (i) => sp(days[i]) + `<br><span class="k">Below peak</span> <b class="t-loss">${F.u(-days[i].dd)}</b>` });
    } else {
      C.line(el, { height: 270, labels, split: true, yFmt: (v) => v.toFixed(1) + 'u', series: [{ color: C.css('--win'), area: true, data: days.map((d) => ({ y: d.cum_units })) }], tip: (i) => sp(days[i]) + `<br><span class="k">Cumulative</span> <b class="${F.tone(days[i].cum_units)}">${F.u(days[i].cum_units)}</b><br><span class="k">Record</span> ${days[i].cum_w}-${days[i].cum_l}` });
    }
  }
  function drawDaily() {
    const days = scopeData().series.days, el = $('#ov-daily'); if (!el) return;
    const tipFn = (d) => tipRows(F.dateY(d.date), [['Record', `${d.w}-${d.l}${d.p ? '-' + d.p : ''}`], ['Units', `<b class="${F.tone(d.units)}">${d.priced ? F.u(d.units) : 'unpriced'}</b>`]]);
    if (state.daily === 'units') C.bars(el, { height: 250, yFmt: (v) => v.toFixed(0) + 'u', data: days.map((d) => ({ label: F.date(d.date), value: d.units, tip: tipFn(d) })) });
    else C.bars(el, { height: 250, diverge: true, data: days.map((d) => ({ label: F.date(d.date), up: d.w, down: d.l, tip: tipFn(d) })) });
  }
  function miniCard() {
    const rows = (TODAY.picks || []);
    if (!rows.length) return '<div class="empty">No card on file.</div>';
    return `<div class="tablewrap"><table><thead><tr><th>Player</th><th>Pick</th><th class="n">Price</th><th class="n">EV</th><th class="n">Model</th><th class="n">Market</th><th>Status</th></tr></thead><tbody>${rows.slice().sort((a, b) => (b.conf || 0) - (a.conf || 0)).map((p) => `<tr><td class="pl"><span class="tm">${logo(p.team, 24)}${esc(p.player)}</span>${p.opp ? ` <span class="vs">VS</span> ${logo(p.opp, 18)}` : ''}</td><td>${esc(p.dir)} ${p.line} ${esc(p.prop)}</td><td class="n">${F.odds(p.price)}</td><td class="n ${F.tone(p.ev)}">${F.pct(p.ev)}</td><td class="n">${F.pct(p.mp, 0)}</td><td class="n">${F.pct(p.nv, 0)}</td><td>${!p.result && p.prov ? chip('prov. ' + p.prov.result, p.prov.result === 'WIN' ? 'win' : p.prov.result === 'LOSS' ? 'loss' : 'amber') : resChip(p.result)}</td></tr>`).join('')}</tbody></table></div>`;
  }

  /* =====================================================================
     TODAY'S CARD
     ===================================================================== */
  function tipTime(iso) { if (!iso) return null; const d = new Date(iso); return isNaN(d) ? null : d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }); }
  function pickCard(p) {
    const shown = p.result || (p.prov && p.prov.result) || null, isProv = !p.result && !!p.prov;
    const cls = shown === 'WIN' ? 'W' : shown === 'LOSS' ? 'L' : shown === 'PUSH' ? 'P' : '';
    const t = tipTime(p.tip), gcol = { A: 'win', B: 'info', C: 'amber', D: '' }[p.grade] || '';
    const moved = p.latest_price != null && p.price != null && p.latest_price !== p.price;
    return `<div class="pcard ${cls}">
      <div class="row1"><div style="display:flex;gap:11px;align-items:center;min-width:0">${logo(p.team, 44)}<div style="min-width:0"><div class="who">${esc(p.player)}</div><div class="matchup" style="display:flex;align-items:center;gap:5px;flex-wrap:wrap">${esc(p.team || '')}${p.opp ? ` <span class="vs">VS</span> ${logo(p.opp, 16)} ${esc(p.opp)}` : ''}${t ? ' · tip ' + t : ''}</div></div></div>
        <div style="display:flex;gap:5px;flex-wrap:wrap;justify-content:flex-end">${p.grade ? chip('Grade ' + p.grade, gcol) : ''}${p.tier ? chip(p.tier, 'violet') : ''}</div></div>
      <div class="bet"><span class="dir">${esc(p.dir)}</span><span class="big">${p.line}</span><span class="stat">${esc(p.prop)}</span><span class="px ${p.price > 0 ? 't-win' : ''}">${F.odds(p.price)}</span></div>
      <div class="dual">
        <div class="r"><span>Model</span><div class="track"><i style="width:${Math.min(100, (p.mp || 0) * 100)}%;background:linear-gradient(90deg,var(--violet),var(--accent-2))"></i>${p.nv != null ? `<b style="left:${p.nv * 100}%" title="Market no-vig"></b>` : ''}</div><span class="num">${F.pct(p.mp, 0)}</span></div>
        <div class="r"><span>Edge (EV)</span><div class="track"><i style="width:${Math.min(100, Math.max(2, (p.ev || 0) * 250))}%;background:${(p.ev || 0) >= 0 ? 'var(--win)' : 'var(--loss)'}"></i></div><span class="num ${F.tone(p.ev)}">${F.pct(p.ev, 1)}</span></div>
        <div class="r"><span>Confidence</span><div class="track"><i style="width:${Math.min(100, p.conf || 0)}%;background:var(--accent)"></i></div><span class="num">${p.conf != null ? p.conf.toFixed(0) : '–'}</span></div>
      </div>
      <div class="foot">${p.pred != null ? chip('Proj ' + p.pred, 'info') : ''}${p.l10 != null ? chip('L10 ' + p.l10) : ''}${p.sigma != null ? chip('σ ' + p.sigma.toFixed(2)) : ''}${p.meta != null ? chip('Meta ' + p.meta.toFixed(0)) : ''}${moved ? chip(`now ${F.odds(p.latest_price)}`, 'amber') : ''}${p.in_latest ? '' : chip('not in latest run', 'amber')}</div>
      ${shown ? `<div class="result ${shown === 'WIN' ? 't-win' : shown === 'LOSS' ? 't-loss' : 't-amber'}" ${isProv ? 'data-tip="Provisional: graded by the CLV report from final box scores. The official grade lands in picks_history.csv next morning and is the only one counted in records."' : ''}>${isProv ? 'PROVISIONAL ' : ''}${shown}${(isProv ? p.prov.actual : p.actual) != null ? ' · ' + (isProv ? p.prov.actual : p.actual) : ''}</div>` : ''}
    </div>`;
  }
  const legChips = (legs) => `<div class="legs">${(legs || []).map((l) => chip(`${l.player} ${l.dir || ''} ${l.line} ${l.prop}${l.price != null ? ' (' + F.odds(l.price) + ')' : ''}`, 'accent')).join('')}</div>`;
  function vToday(root) {
    const picks = TODAY.picks || [];
    const stat = TODAY.pending ? chip(`${TODAY.pending} awaiting grades`, 'info pulse') : chip('all graded', 'win');
    const w = picks.filter((p) => p.result === 'WIN').length, l = picks.filter((p) => p.result === 'LOSS').length;
    const order = { conf: (a, b) => (b.conf || 0) - (a.conf || 0), ev: (a, b) => (b.ev || 0) - (a.ev || 0), tip: (a, b) => String(a.tip || 'z').localeCompare(String(b.tip || 'z')) };
    root.innerHTML = `
      <div class="kpis stagger">
        ${kpi({ label: 'Card date', val: F.dateY(TODAY.card_date), tone: 'info', sub: `built ${esc((TODAY.build_date || '').slice(0, 10))} · ${stat.replace(/<[^>]+>/g, '')}` })}
        ${kpi({ label: 'Picks on card', count: TODAY.n || 0, fmt: 'int', val: String(TODAY.n || 0), tone: 'violet', sub: `${picks.filter((p) => p.prop === 'AST').length} AST · ${picks.filter((p) => p.prop === 'STL').length} STL · ${picks.filter((p) => p.prop === 'BLK').length} BLK · 100% UNDER` })}
        ${kpi({ label: 'Graded so far', valHtml: `<span class="t-win">${w}</span><span class="sep">–</span><span class="t-loss">${l}</span>`, tone: 'win', sub: `${TODAY.pending || 0} pending officially${TODAY.provisional && TODAY.provisional.n ? ` · <b>${TODAY.provisional.w}-${TODAY.provisional.l}</b> provisional (${TODAY.provisional.n} final box scores in)` : ''}` })}
        ${kpi({ label: 'Avg EV at price', val: F.pct(avg(picks.map((p) => p.ev))), tone: 'amber', sub: `avg model ${F.pct(avg(picks.map((p) => p.mp)), 0)} vs market ${F.pct(avg(picks.map((p) => p.nv)), 0)}` })}
      </div>
      ${(TODAY.dropped_from_latest || []).length ? `<div class="banner warn"><b>Heads up</b><span>${(TODAY.dropped_from_latest || []).map(esc).join(', ')} ${TODAY.dropped_from_latest.length > 1 ? 'were' : 'was'} published earlier today but is not in the latest run's card (picks_history is append-only; picks_latest reflects the newest run). Verify the live line before betting.</span></div>` : ''}
      <div class="grid stagger">
        ${panel({ title: 'Tonight\'s slate', cls: 'c12', sub: `${(TODAY.games || []).length} games on the card · picks per game and how they are settling`, body: gamesHtml() })}
      </div>
      <div class="grid stagger">
        ${panel({ title: 'Card by prop', cls: 'c4', body: '<div id="td-pie"></div>' })}
        ${panel({ title: 'Model vs market', cls: 'c4', sub: 'Average model probability on the card; amber tick = average FanDuel no-vig', body: `<div id="td-gauge"></div>` })}
        ${panel({ title: 'Settlement', cls: 'c4', sub: 'Official grades plus provisional box-score results', body: '<div id="td-ring" style="text-align:center"></div>' })}
      </div>
      <div class="grid"><div class="panel c12">${`<div class="head"><div><h2><span class="dot"></span>The card</h2><div class="psub">${picks.length} UNDER picks · marker on the model bar = FanDuel no-vig market probability</div></div><div class="tools">${seg('csort', [['conf', 'Confidence'], ['ev', 'EV'], ['tip', 'Tip time']], 'conf')}</div></div><div class="cards" id="td-cards"></div>`}</div></div>
      <div class="grid stagger">
        ${panel({ title: 'Betslips from the latest run', cls: 'c5', sub: 'Priced at logged FanDuel odds · joint probability vs break-even', body: (SLIPS.latest || []).length ? `<div style="display:grid;gap:10px">${SLIPS.latest.map((s) => `<div class="slip"><div class="top2"><b>${esc(s.type)}</b><span>${chip('$' + (s.wager || 0) + ' → $' + (s.potential || 0), 'accent')}</span></div>${legChips(s.legs)}<div class="note">Joint prob <b>${F.pct(s.joint)}</b> vs break-even <b>${F.pct(s.be)}</b> · EV <b class="${F.tone(s.ev)}">${F.pct(s.ev)}</b>/unit</div></div>`).join('')}</div>` : '<div class="empty">No betslips on the latest run.</div>' })}
        ${panel({ title: 'Gate board', cls: 'c7', sub: 'Every candidate the latest run considered, and what stopped it', body: gateBoard() })}
      </div>`;
    const draw = (k) => { $('#td-cards').innerHTML = picks.slice().sort(order[k]).map(pickCard).join('') || '<div class="empty">No picks on file for this date.</div>'; };
    draw('conf'); onSeg(root, 'csort', draw);
    const byProp = {}; picks.forEach((p) => { byProp[p.prop] = (byProp[p.prop] || 0) + 1; });
    C.pie($('#td-pie'), { size: 180, center: { text: String(picks.length), sub: 'picks' }, slices: Object.entries(byProp).map(([k, v]) => ({ label: k + ' UNDER', value: v })) });
    const mp = avg(picks.map((p) => p.mp)), nv = avg(picks.map((p) => p.nv));
    $('#td-gauge').innerHTML = C.gauge({ min: 0.4, max: 0.8, value: mp, ref: nv, refLabel: 'market', zones: nv != null ? [{ from: 0.4, to: nv, color: C.css('--amber') }, { from: nv, to: 0.8, color: C.css('--win') }] : undefined, text: F.pct(mp, 1), sub: `market ${F.pct(nv, 1)} · avg EV ${F.pct(avg(picks.map((p) => p.ev)))}`, minLabel: '40%', maxLabel: '80%' });
    const settled = picks.filter((p) => p.result || p.prov).length;
    $('#td-ring').innerHTML = C.ring({ value: picks.length ? settled / picks.length : 0, text: `${settled}/${picks.length}`, sub: 'settled', size: 160, color: C.css('--accent-2'), color2: C.css('--violet') });
    countUp(root);
  }
  function gamesHtml() {
    const g = TODAY.games || [];
    if (!g.length) return '<div class="empty">No games on the card.</div>';
    return `<div class="games">${g.map((x) => {
      const t = tipTime(x.tip), off = x.w + x.l, prov = x.pw + x.pl;
      return `<div class="game"><div class="tms"><div class="side">${logo(x.a, 52)}<span>${esc(x.a)}</span></div><span class="vs">VS</span><div class="side">${logo(x.b, 52)}<span>${esc(x.b)}</span></div></div>
        <div class="meta"><span>${t ? 'Tip ' + t : 'Tip time n/a'}</span><span style="display:flex;gap:5px;flex-wrap:wrap;justify-content:flex-end">${chip(x.n + (x.n === 1 ? ' pick' : ' picks'), 'accent')}${off ? chip(`${x.w}-${x.l} official`, x.w >= x.l ? 'win' : 'loss') : ''}${prov ? chip(`${x.pw}-${x.pl} prov.`, x.pw >= x.pl ? 'win' : 'loss') : ''}${x.pending ? chip(`${x.pending} live`, 'info pulse') : ''}</span></div></div>`;
    }).join('')}</div>`;
  }
  const avg =(a) => { const v = a.filter((x) => x != null && isFinite(x)); return v.length ? v.reduce((x, y) => x + y, 0) / v.length : null; };
  function gateBoard() {
    const b = GATES.board || [];
    if (!b.length) return '<div class="empty">No candidate board on file.</div>';
    const passed = b.filter((x) => x.passed).length;
    return `<div class="legend"><span><i style="background:var(--win)"></i>${passed} passed all gates</span><span><i style="background:var(--loss)"></i>${b.length - passed} blocked</span></div><div class="tablewrap" style="max-height:430px"><table><thead><tr><th>Player</th><th>Pick</th><th class="n">Price</th><th class="n">EV</th><th>Verdict</th></tr></thead><tbody>${b.slice().sort((x, y) => (y.passed - x.passed) || ((y.ev || 0) - (x.ev || 0))).map((x) => `<tr><td class="pl"><span class="tm">${logo(x.team, 22)}${esc(x.player)}</span> <span class="vs">VS</span> ${logo(x.opp, 16)}</td><td>${esc(x.dir)} ${x.line} ${esc(x.prop)}</td><td class="n">${F.odds(x.price)}</td><td class="n ${F.tone(x.ev)}">${F.pct(x.ev)}</td><td>${x.passed ? chip('Published', 'win') : `<span data-tip="${esc(x.failure || '')}">${chip(x.category || 'Blocked', 'loss')}</span> <span class="sub">${esc((x.failure || '').slice(0, 44))}</span>`}</td></tr>`).join('')}</tbody></table></div>`;
  }

  /* =====================================================================
     PERFORMANCE
     ===================================================================== */
  function bucketTable(rows, label, cmp) {
    if (!rows.length) return '<div class="empty">Not enough priced picks.</div>';
    return `<div class="tablewrap"><table class="compact"><thead><tr><th>${esc(label)}</th><th class="n">W-L</th><th class="n">Hit</th>${cmp ? `<th class="n">${esc(cmp.h)}</th>` : ''}<th class="n">Units</th></tr></thead><tbody>${rows.map((r) => `<tr><td class="pl">${esc(r.label)}${lowN(r.n, 20)}</td><td class="n">${r.w}-${r.l}</td><td class="n">${hitBar(r)}</td>${cmp ? `<td class="n">${F.pct(cmp.f(r), 0)}</td>` : ''}<td class="n ${F.tone(r.units)}">${F.u(r.units)}</td></tr>`).join('')}</tbody></table></div>`;
  }
  function teamTiles(rows, key) {
    if (!rows.length) return '<div class="empty">No teams in this scope yet.</div>';
    const sorted = rows.slice().sort((a, b) => ((b.units == null ? -99 : b.units) - (a.units == null ? -99 : a.units)) || (b.n - a.n));
    return `<div class="teamgrid">${sorted.map((r) => {
      const ab = r[key], nm = (TEAMS[ab] || {}).name || ab, u = r.units;
      return `<div class="tile ${u > 0.005 ? 'pos' : u < -0.005 ? 'neg' : ''}" data-tip="${esc(tipRows(nm, [['Record', `${r.w}-${r.l}`], ['Hit', `${F.pct(r.hit)} (CI ${ciTxt(r)})`], ['Units', F.u(u)], ['ROI', F.pct(r.roi)]]))}">${logo(ab, 46)}<div class="rc">${r.w}–${r.l}</div><div class="un ${F.tone(u)}">${u == null ? 'unpriced' : F.u(u)}</div><div class="sb">${F.pct(r.hit, 0)} · n=${r.n}</div></div>`;
    }).join('')}</div>`;
  }
  function vPerf(root) {
    const sc = scopeData(), s = sc.summary, ser = sc.series;
    root.innerHTML = `
      <div class="grid stagger">
        ${panel({ title: 'Calendar', cls: 'c12', sub: `${esc(sc.label)} · each square is a slate: green = profitable, red = losing, brighter = bigger`, body: '<div id="pf-cal" style="overflow-x:auto"></div>' })}
      </div>
      <div class="grid stagger">
        ${panel({ title: 'Team board', cls: 'c12', sub: 'Every team in scope as a tile: record, flat-unit result and hit rate. Sorted by units. Hover for detail.', tools: seg('tmode', [['teams', 'Teams we picked'], ['opps', 'Opponents we picked against']], state.teamMode), body: '<div id="pf-teams"></div>' })}
      </div>
      <div class="grid stagger">
        ${panel({ title: 'Bets by price band', cls: 'c4', sub: 'Graded priced bets, hover for hit rate and units', body: '<div id="pf-pie-price"></div>' })}
        ${panel({ title: 'Bets by model-probability band', cls: 'c4', sub: 'Calibrated side probability at pick time', body: '<div id="pf-pie-prob"></div>' })}
        ${panel({ title: 'Bets by EV band', cls: 'c4', sub: 'Expected value per unit at the offered price', body: '<div id="pf-pie-ev"></div>' })}
      </div>
      <div class="grid stagger">
        ${panel({ title: 'Rolling 20-pick hit rate', cls: 'c6', sub: 'Smooths day-to-day noise. Dashed = average break-even.', body: '<div id="pf-roll"></div>' })}
        ${panel({ title: 'Cumulative units', cls: 'c6', sub: 'Priced picks only, flat 1u', body: '<div id="pf-units"></div>' })}
      </div>
      <div class="grid stagger">
        ${panel({ title: 'Lanes: prop × direction', cls: 'c12', body: `<div class="tablewrap"><table><thead><tr><th>Lane</th><th class="n">Graded</th><th class="n">Record</th><th class="n">Hit rate</th><th class="n">95% CI</th><th class="n">Avg model</th><th class="n">Avg market</th><th class="n">Avg EV</th><th class="n">Units</th><th class="n">ROI</th></tr></thead><tbody>${sc.by_prop.map((r) => `<tr><td class="pl">${esc(r.prop)} <span class="chip ${r.direction === 'UNDER' ? 'accent' : 'info'}">${esc(r.direction)}</span></td><td class="n">${r.n}${lowN(r.n)}</td><td class="n">${r.w}-${r.l}</td><td class="n">${hitBar(r)}</td><td class="n t-muted">${ciTxt(r)}</td><td class="n">${F.pct(r.avg_model_prob, 0)}</td><td class="n">${F.pct(r.avg_no_vig, 0)}</td><td class="n">${F.pct(r.avg_ev, 1)}</td><td class="n ${F.tone(r.units)}">${F.u(r.units)}</td><td class="n ${F.tone(r.roi)}">${F.pct(r.roi)}</td></tr>`).join('')}</tbody></table></div>` })}
      </div>
      <div class="grid stagger">
        ${panel({ title: 'By price paid', cls: 'c4', sub: 'FanDuel American price on the pick', body: bucketTable(sc.by_price, 'Price band', { h: 'Break-even', f: (r) => r.break_even }) })}
        ${panel({ title: 'By model probability', cls: 'c4', sub: 'Calibrated side probability at pick time', body: bucketTable(sc.by_prob, 'Model p', { h: 'Predicted', f: (r) => r.avg_model_prob }) })}
        ${panel({ title: 'By expected value', cls: 'c4', sub: 'EV per unit at the offered price', body: bucketTable(sc.by_ev, 'EV band') })}
      </div>
      <div class="grid stagger">
        ${panel({ title: 'Lines that keep showing up', cls: 'c4', sub: 'Prop + line combos with 4+ graded picks', body: sc.by_line.length ? `<div class="tablewrap"><table><thead><tr><th>Line</th><th class="n">Record</th><th class="n">Hit</th><th class="n">Units</th></tr></thead><tbody>${sc.by_line.map((r) => `<tr><td class="pl">${esc(r.prop)} ${r.line}</td><td class="n">${r.w}-${r.l}</td><td class="n">${hitBar(r)}</td><td class="n ${F.tone(r.units)}">${F.u(r.units)}</td></tr>`).join('')}</tbody></table></div>` : '<div class="empty">Not enough repeats yet.</div>' })}
        ${panel({ title: 'Players (3+ picks)', cls: 'c4', sub: 'Most-picked players and how they did', body: sc.players.length ? `<div class="tablewrap" style="max-height:380px"><table><thead><tr><th>Player</th><th class="n">Record</th><th class="n">Units</th></tr></thead><tbody>${sc.players.map((r) => `<tr><td class="pl"><span class="tm">${logo(r.team, 22)}${esc(r.player)}</span></td><td class="n">${r.w}-${r.l}</td><td class="n ${F.tone(r.units)}">${F.u(r.units)}</td></tr>`).join('')}</tbody></table></div>` : '<div class="empty">No player has 3 picks yet.</div>' })}
        ${panel({ title: 'Teams (4+ picks)', cls: 'c4', sub: 'Team of the player picked', body: sc.teams.length ? `<div class="tablewrap" style="max-height:380px"><table><thead><tr><th>Team</th><th class="n">Record</th><th class="n">Units</th></tr></thead><tbody>${sc.teams.map((r) => `<tr><td class="pl">${team(r.team, 24)}</td><td class="n">${r.w}-${r.l}</td><td class="n ${F.tone(r.units)}">${F.u(r.units)}</td></tr>`).join('')}</tbody></table></div>` : '<div class="empty">Not enough team data.</div>' })}
      </div>
      <div class="grid stagger">
        ${panel({ title: 'By month', cls: 'c12', body: `<div class="tablewrap"><table><thead><tr><th>Month</th><th class="n">Graded</th><th class="n">Record</th><th class="n">Hit rate</th><th class="n">Units (priced)</th><th class="n">DNP</th><th class="n">Pending</th></tr></thead><tbody>${sc.by_month.map((r) => `<tr><td class="pl">${esc(r.month)}</td><td class="n">${r.n}</td><td class="n">${r.w}-${r.l}</td><td class="n">${hitBar(r)}</td><td class="n ${F.tone(r.units)}">${r.priced_n ? F.u(r.units) : '–'}</td><td class="n">${r.dnp}</td><td class="n">${r.pending}</td></tr>`).join('')}</tbody></table></div>` })}
      </div>`;
    C.calendar($('#pf-cal'), ser.days);
    const drawTeams = () => { $('#pf-teams').innerHTML = state.teamMode === 'opps' ? teamTiles(sc.opps || [], 'opponent') : teamTiles(sc.teams_all || [], 'team'); };
    drawTeams(); onSeg(root, 'tmode', (v) => { state.teamMode = v; drawTeams(); });
    const bucketSlices = (rows) => rows.map((r) => ({ label: r.label, value: r.n, text: `${r.w}-${r.l}`, tip: tipRows(r.label, [['Bets', r.n], ['Record', `${r.w}-${r.l}`], ['Hit', F.pct(r.hit)], ['Units', `<b class="${F.tone(r.units)}">${F.u(r.units)}</b>`]]) }));
    C.pie($('#pf-pie-price'), { size: 180, hole: 0, legend: 'bottom', slices: bucketSlices(sc.by_price) });
    C.pie($('#pf-pie-prob'), { size: 180, center: { text: String(sc.by_prob.reduce((a, r) => a + r.n, 0)), sub: 'bets' }, legend: 'bottom', slices: bucketSlices(sc.by_prob) });
    C.pie($('#pf-pie-ev'), { size: 180, center: { text: String(sc.by_ev.reduce((a, r) => a + r.n, 0)), sub: 'bets' }, legend: 'bottom', slices: bucketSlices(sc.by_ev) });
    const rl = ser.rolling20, be = s.break_even;
    if (rl.length) C.line($('#pf-roll'), { height: 230, zero: false, yFmt: (v) => F.pct(v, 0), refs: be ? [{ y: be, label: 'break-even', color: C.css('--amber') }] : [], labels: rl.map((p) => F.date(p.date)), series: [{ color: C.css('--violet'), area: true, data: rl.map((p) => ({ y: p.hit })) }], tip: (i) => tipRows(F.dateY(rl[i].date), [['Pick #', rl[i].i], ['Rolling 20', `<b>${F.pct(rl[i].hit)}</b>`]]) });
    else $('#pf-roll').innerHTML = '<div class="empty">Needs 10+ graded picks.</div>';
    C.line($('#pf-units'), { height: 230, split: true, labels: ser.days.map((d) => F.date(d.date)), yFmt: (v) => v.toFixed(0) + 'u', series: [{ color: C.css('--win'), area: true, data: ser.days.map((d) => ({ y: d.cum_units })) }], tip: (i) => tipRows(F.dateY(ser.days[i].date), [['Day', `${ser.days[i].w}-${ser.days[i].l} · ${F.u(ser.days[i].units)}`], ['Cumulative', `<b class="${F.tone(ser.days[i].cum_units)}">${F.u(ser.days[i].cum_units)}</b>`]]) });
  }

  /* =====================================================================
     PICK EXPLORER
     ===================================================================== */
  const COLS = [
    ['d', 'Date', false], ['pl', 'Player', false], ['tm', 'Team', false], ['op', 'Opp', false], ['pr', 'Prop', false], ['di', 'Dir', false], ['ln', 'Line', true],
    ['px', 'Price', true], ['ev', 'EV', true], ['mp', 'Model', true], ['nv', 'Market', true], ['cf', 'Conf', true], ['sg', 'Grade', false],
    ['rs', 'Result', false], ['ac', 'Actual', true], ['u', 'Units', true], ['er', 'Era', false],
  ];
  function filteredPicks() {
    const f = state.ex, q = f.q.trim().toLowerCase();
    let rows = (D.picks || []).filter((p) => (!q || (p.pl + ' ' + (p.tm || '') + ' ' + (p.op || '')).toLowerCase().includes(q)) && (!f.prop || p.pr === f.prop) && (!f.dir || p.di === f.dir) && (!f.era || p.er === f.era) && (!f.res || (f.res === 'PENDING' ? p.rs == null : p.rs === f.res)));
    const k = f.sort, dir = f.desc ? -1 : 1;
    rows.sort((a, b) => { const x = a[k], y = b[k]; if (x == null && y == null) return 0; if (x == null) return 1; if (y == null) return -1; return (x < y ? -1 : x > y ? 1 : 0) * dir; });
    return rows;
  }
  function exSummary(rows) {
    const w = rows.filter((r) => r.rs === 'WIN').length, l = rows.filter((r) => r.rs === 'LOSS').length;
    const pr = rows.filter((r) => (r.rs === 'WIN' || r.rs === 'LOSS') && r.dc != null), u = pr.reduce((a, r) => a + (r.u || 0), 0);
    return { n: rows.length, w, l, hit: w + l ? w / (w + l) : null, units: pr.length ? u : null, priced: pr.length };
  }
  function vExplorer(root) {
    const props = [...new Set((D.picks || []).map((p) => p.pr))].sort();
    root.innerHTML = `
      <div class="grid"><div class="panel c12">
        <div class="head"><div><h2><span class="dot"></span>Pick explorer</h2><div class="psub">Every pick ever logged. Filter, sort, and export what you see.</div></div>
          <div class="tools"><button class="btn sm" id="ex-csv">Export CSV</button></div></div>
        <div class="field" style="margin-bottom:12px">
          <input type="search" id="ex-q" placeholder="Search player / team…" value="${esc(state.ex.q)}" style="min-width:210px">
          <select id="ex-prop"><option value="">All props</option>${props.map((p) => `<option ${state.ex.prop === p ? 'selected' : ''}>${p}</option>`).join('')}</select>
          <select id="ex-dir"><option value="">Over + Under</option><option ${state.ex.dir === 'UNDER' ? 'selected' : ''}>UNDER</option><option ${state.ex.dir === 'OVER' ? 'selected' : ''}>OVER</option></select>
          <select id="ex-res"><option value="">Any result</option>${['WIN', 'LOSS', 'PUSH', 'DNP', 'PENDING'].map((r) => `<option ${state.ex.res === r ? 'selected' : ''}>${r}</option>`).join('')}</select>
          <select id="ex-era"><option value="">All eras</option>${(D.eras || []).map((e) => `<option value="${e.key}" ${state.ex.era === e.key ? 'selected' : ''}>${esc(e.label)}</option>`).join('')}</select>
        </div>
        <div id="ex-sum" class="legend" style="margin:0 0 10px"></div>
        <div class="tablewrap" style="max-height:640px"><table id="ex-table"></table></div>
        <div class="field" style="justify-content:space-between;margin-top:10px"><span class="note" id="ex-pg"></span><span><button class="btn sm" id="ex-prev">← Prev</button> <button class="btn sm" id="ex-next">Next →</button></span></div>
      </div></div>`;
    const PER = 40;
    const draw = () => {
      const rows = filteredPicks(), s = exSummary(rows), pages = Math.max(1, Math.ceil(rows.length / PER));
      state.ex.page = Math.min(state.ex.page, pages - 1);
      $('#ex-sum', root).innerHTML = `<span><b>${F.n(s.n)}</b> picks</span><span><b class="t-win">${s.w}</b>–<b class="t-loss">${s.l}</b> graded</span><span>hit <b>${F.pct(s.hit)}</b></span><span>units <b class="${F.tone(s.units)}">${s.units == null ? '–' : F.u(s.units)}</b> on ${s.priced} priced</span>`;
      const page = rows.slice(state.ex.page * PER, state.ex.page * PER + PER);
      $('#ex-table', root).innerHTML = `<thead><tr>${COLS.map((c) => `<th class="sort ${c[2] ? 'n' : ''}" data-k="${c[0]}">${c[1]}${state.ex.sort === c[0] ? `<span class="arr">${state.ex.desc ? '▼' : '▲'}</span>` : ''}</th>`).join('')}</tr></thead><tbody>${page.map((p) => `<tr><td>${esc(p.d)}</td><td class="pl">${esc(p.pl)}</td><td>${team(p.tm, 20)}</td><td>${team(p.op, 20)}</td><td>${esc(p.pr)}</td><td><span class="chip ${p.di === 'UNDER' ? 'accent' : 'info'}">${esc(p.di)}</span></td><td class="n">${p.ln}</td><td class="n">${F.odds(p.px)}</td><td class="n ${F.tone(p.ev)}">${p.ev != null ? F.pct(p.ev) : '–'}</td><td class="n">${p.mp != null ? F.pct(p.mp, 0) : '–'}</td><td class="n">${p.nv != null ? F.pct(p.nv, 0) : '–'}</td><td class="n">${p.cf != null ? p.cf.toFixed(0) : '–'}</td><td>${esc(p.sg || '')}</td><td>${resChip(p.rs)}</td><td class="n">${p.ac != null ? p.ac : '–'}</td><td class="n ${F.tone(p.u)}">${p.u != null ? F.u(p.u) : '–'}</td><td><span class="sub">${esc(ERA_LABEL[p.er] || p.er)}</span></td></tr>`).join('') || `<tr><td colspan="${COLS.length}"><div class="empty">No picks match.</div></td></tr>`}</tbody>`;
      $('#ex-pg', root).textContent = `Page ${state.ex.page + 1} of ${pages} · ${F.n(rows.length)} rows`;
      $$('#ex-table th.sort', root).forEach((th) => th.addEventListener('click', () => { const k = th.dataset.k; if (state.ex.sort === k) state.ex.desc = !state.ex.desc; else { state.ex.sort = k; state.ex.desc = true; } state.ex.page = 0; draw(); }));
    };
    const bind = (id, key) => $(id, root).addEventListener('input', (e) => { state.ex[key] = e.target.value; state.ex.page = 0; draw(); });
    bind('#ex-q', 'q'); bind('#ex-prop', 'prop'); bind('#ex-dir', 'dir'); bind('#ex-res', 'res'); bind('#ex-era', 'era');
    $('#ex-prev', root).onclick = () => { state.ex.page = Math.max(0, state.ex.page - 1); draw(); };
    $('#ex-next', root).onclick = () => { state.ex.page++; draw(); };
    $('#ex-csv', root).onclick = () => {
      const rows = filteredPicks(), head = COLS.map((c) => c[1]).join(',');
      const body = rows.map((p) => COLS.map((c) => { const v = p[c[0]]; return v == null ? '' : '"' + String(v).replace(/"/g, '""') + '"'; }).join(',')).join('\n');
      const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([head + '\n' + body], { type: 'text/csv' })); a.download = 'picks_filtered.csv'; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    };
    draw();
  }

  /* =====================================================================
     GATES & CANDIDATES
     ===================================================================== */
  const BUCKET_LABEL = { published: ['Published to the card', 'win'], blocked_price: ['Blocked by price gate', 'amber'], blocked_other: ['Blocked by other gates', 'violet'] };
  function vGates(root) {
    if (!GATES.buckets) { root.innerHTML = panel({ title: 'Gates', cls: 'c12', body: unavail('Candidate history') }); return; }
    const kp = GATES.buckets.map((b) => kpi({ label: BUCKET_LABEL[b.bucket][0], valHtml: rec(b), tone: BUCKET_LABEL[b.bucket][1] === 'win' ? 'win' : BUCKET_LABEL[b.bucket][1] === 'amber' ? 'amber' : 'violet', sub: `${F.pct(b.hit)} hit · <b class="${F.tone(b.units)}">${F.u(b.units)}</b> · ROI ${F.pct(b.roi)}` })).join('');
    const br = GATES.brier;
    root.innerHTML = `
      <div class="kpis stagger">${kp}${br ? kpi({ label: 'Brier: model vs market', valHtml: `<span class="${br.model < br.market ? 't-win' : 't-loss'}">${br.model.toFixed(4)}</span><small>vs ${br.market.toFixed(4)}</small>`, tone: br.model < br.market ? 'win' : 'loss', sub: `${br.n} graded candidates · lower is better` }) : ''}</div>
      <div class="grid stagger">
        ${panel({ title: 'Candidate flow', cls: 'c4', sub: 'Where every logged candidate ended up', body: '<div id="gt-pie-bucket"></div>' })}
        ${panel({ title: 'Why candidates were stopped', cls: 'c4', sub: 'Primary reason per candidate', body: '<div id="gt-pie-reason"></div>' })}
        ${panel({ title: 'Published ROI dial', cls: 'c4', sub: 'Flat 1u at the recorded price for candidates that reached the card', body: '<div id="gt-gauge"></div>' })}
      </div>
      <div class="grid stagger">
        ${panel({ title: 'Which gate stopped what', cls: 'c7', sub: `Candidates ${esc(GATES.dates[0])} → ${esc(GATES.dates[1])}, graded at the recorded FanDuel price. A blocked bucket that wins is money the gate cost; one that loses is money it saved.`, body: `<div class="tablewrap"><table><thead><tr><th>Reason</th><th class="n">Graded</th><th class="n">Record</th><th class="n">Hit</th><th class="n">Units</th><th class="n">ROI</th></tr></thead><tbody>${(GATES.categories || []).map((c) => `<tr><td class="pl">${esc(c.category)}${lowN(c.n, 20)}</td><td class="n">${c.n}</td><td class="n">${c.w}-${c.l}</td><td class="n">${hitBar(c)}</td><td class="n ${F.tone(c.units)}">${F.u(c.units)}</td><td class="n ${F.tone(c.roi)}">${F.pct(c.roi)}</td></tr>`).join('')}</tbody></table></div><div class="note" style="margin-top:10px">Samples are tiny (a few slates). DEC-071/073 loosened the volatility, meta and EV gates on similar evidence; re-review is scheduled around <b>2026-11-30</b>.</div>` })}
        ${panel({ title: 'Is the model calibrated?', cls: 'c5', sub: 'Predicted vs realised hit rate on graded candidates. On the dashed line = perfectly calibrated.', body: '<div id="gt-rel"></div><div class="legend"><span><i style="background:var(--violet)"></i>Model probability</span><span><i style="background:var(--accent-2)"></i>Market no-vig</span></div>' })}
      </div>
      <div class="grid stagger">
        ${panel({ title: 'Cumulative units by bucket', cls: 'c8', sub: 'How each bucket would have performed at flat 1u', body: '<div id="gt-cum"></div><div class="legend"><span><i style="background:var(--win)"></i>Published</span><span><i style="background:var(--amber)"></i>Blocked: price</span><span><i style="background:var(--violet)"></i>Blocked: other</span></div>' })}
        ${panel({ title: 'Gate experiment (DEC-071/073)', cls: 'c4', sub: 'Alternative thresholds measured forward on the same candidates', body: gateExperiment() })}
      </div>`;
    const bk = {}; GATES.buckets.forEach((b) => { bk[b.bucket] = b; });
    const bcol = { published: C.css('--accent-2'), blocked_price: C.css('--loss'), blocked_other: C.css('--violet') };
    C.pie($('#gt-pie-bucket'), { size: 180, hole: 0, legend: 'bottom', slices: GATES.buckets.map((b) => ({ label: BUCKET_LABEL[b.bucket][0], value: b.n + b.p + b.dnp + b.pending, text: `${b.w}-${b.l}`, color: bcol[b.bucket], tip: tipRows(BUCKET_LABEL[b.bucket][0], [['Candidates', b.n + b.p + b.dnp + b.pending], ['Graded record', `${b.w}-${b.l}`], ['Units', F.u(b.units)]]) })) });
    C.pie($('#gt-pie-reason'), { size: 180, center: { text: String(GATES.rows), sub: 'candidates' }, legend: 'bottom', slices: (GATES.categories || []).map((c) => ({ label: c.category, value: c.n + c.p + c.dnp + c.pending, text: `${c.w}-${c.l}`, tip: tipRows(c.category, [['Candidates', c.n + c.p + c.dnp + c.pending], ['Record', `${c.w}-${c.l}`], ['Units', F.u(c.units)]]) })) });
    const pub = bk.published;
    $('#gt-gauge').innerHTML = pub ? C.gauge({ min: -0.4, max: 0.4, value: pub.roi, ref: 0, refLabel: '0', zones: [{ from: -0.4, to: 0, color: C.css('--loss') }, { from: 0, to: 0.4, color: C.css('--win') }], text: F.pct(pub.roi), sub: `${pub.w}-${pub.l} · ${F.u(pub.units)}`, minLabel: '−40%', maxLabel: '+40%' }) + `<div class="note" style="text-align:center;margin-top:4px">Small sample (${pub.n} graded): the needle will swing a lot until more slates settle.</div>` : '<div class="empty">No published candidates.</div>';
    const rel = GATES.reliability || [];
    C.scatter($('#gt-rel'), { height: 290, xDomain: [0.4, 0.8], yDomain: [0.3, 1], diag: true, xLabel: 'predicted probability', yLabel: 'actual hit rate', groups: [
      { color: C.css('--violet'), line: true, points: rel.filter((r) => r.n >= 8).map((r) => ({ x: r.pred, y: r.act, r: 4 + Math.min(9, Math.sqrt(r.n)), tip: tipRows(r.label, [['Candidates', r.n], ['Model predicted', F.pct(r.pred)], ['Market implied', F.pct(r.mkt)], ['Actual', `<b>${F.pct(r.act)}</b>`]]) })) },
      { color: C.css('--faint'), points: rel.filter((r) => r.n < 8).map((r) => ({ x: r.pred, y: r.act, r: 3.5, tip: tipRows(r.label + ' (too few to trust)', [['Candidates', r.n], ['Model predicted', F.pct(r.pred)], ['Actual', F.pct(r.act)]]) })) },
      { color: C.css('--accent-2'), points: rel.map((r) => ({ x: r.mkt, y: r.act, r: 3.5, tip: tipRows(r.label + ' (market)', [['Market implied', F.pct(r.mkt)], ['Actual', F.pct(r.act)]]) })) },
    ] });
    let cum = { published: 0, blocked_price: 0, blocked_other: 0 };
    const series = (k) => GATES.by_day.map((d) => { cum[k] += d[k].units; return { y: cum[k] }; });
    const sP = series('published'); cum = { published: 0, blocked_price: 0, blocked_other: 0 };
    const sB = series('blocked_price'); cum = { published: 0, blocked_price: 0, blocked_other: 0 };
    const sO = series('blocked_other');
    C.line($('#gt-cum'), { height: 260, labels: GATES.by_day.map((d) => F.date(d.date)), yFmt: (v) => v.toFixed(0) + 'u', series: [{ color: C.css('--accent-2'), area: true, data: sP }, { color: C.css('--loss'), data: sB }, { color: C.css('--violet'), data: sO }], tip: (i) => tipRows(F.dateY(GATES.by_day[i].date), [['Published', `${GATES.by_day[i].published.w}-${GATES.by_day[i].published.l} · ${F.u(sP[i].y)} cum`], ['Blocked: price', `${GATES.by_day[i].blocked_price.w}-${GATES.by_day[i].blocked_price.l} · ${F.u(sB[i].y)} cum`], ['Blocked: other', `${GATES.by_day[i].blocked_other.w}-${GATES.by_day[i].blocked_other.l} · ${F.u(sO[i].y)} cum`]]) });
    countUp(root);
  }
  function gateExperiment() {
    const g = MODELS.gate_experiment; if (!g || !g.comparisons) return '<div class="empty">No experiment file.</div>';
    const rows = Object.entries(g.comparisons);
    return `<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px">${chip(g.status, g.status === 'collecting' ? 'info pulse' : 'win')}${chip(`${g.dates} date(s)`)}${chip(`${g.contracts} contracts`)}${g.review_eligible ? chip('review eligible', 'win') : chip('not review-eligible yet', 'amber')}</div><div class="tablewrap"><table><thead><tr><th>Variant</th><th class="n">n</th><th class="n">Wins</th><th class="n">Units</th></tr></thead><tbody>${rows.map(([k, v]) => `<tr><td class="pl">${esc(k.replace(/_/g, ' '))}</td><td class="n">${v.n}</td><td class="n">${v.wins}</td><td class="n ${F.tone(v.profit_units)}">${F.u(v.profit_units)}</td></tr>`).join('')}</tbody></table></div><div class="note" style="margin-top:8px">Started ${esc(g.start || '')}; ${g.excluded || 0} run(s) excluded for scoring-source drift.</div>`;
  }

  /* =====================================================================
     MARKET & CLV
     ===================================================================== */
  function vClv(root) {
    const pk = (CLV.picks || {}), o = pk.overall;
    if (!o) { root.innerHTML = panel({ title: 'CLV', cls: 'c12', body: unavail('CLV summary') }); return; }
    const ml = (CLV.model_leans || {}).overall;
    const groups = Object.entries(pk.groups || {});
    root.innerHTML = `
      <div class="kpis stagger">
        ${kpi({ label: 'Beat the close', count: o.beat_close_rate, fmt: 'pct', dec: 0, val: F.pct(o.beat_close_rate, 0), tone: 'amber', sub: `${o.rows_with_exact_close} picks with an exact-line close` })}
        ${kpi({ label: 'Price moved our way', count: o.moved_toward_side_rate, fmt: 'pct', dec: 0, val: F.pct(o.moved_toward_side_rate, 0), tone: 'info', sub: `${F.pct(o.moved_away_rate, 0)} moved against us` })}
        ${kpi({ label: 'Avg no-vig move', val: F.pts(o.mean_no_vig_move), vcls: F.tone(o.mean_no_vig_move), tone: 'violet', sub: 'bet → close, our side' })}
        ${kpi({ label: 'Avg EV vs close', val: F.pct(o.mean_ev_vs_close), vcls: F.tone(o.mean_ev_vs_close), tone: 'loss', sub: 'baseline ≈ −4.5% at −110 (the vig)' })}
        ${kpi({ label: 'Realised ROI', val: F.pct(o.roi_units_per_bet), vcls: F.tone(o.roi_units_per_bet), tone: 'win', sub: `${o.graded} graded · hit ${F.pct(o.hit_rate)}` })}
        ${kpi({ label: 'Close captured', val: `${o.median_close_minutes_before_tip}`, unit: 'min', tone: 'info', sub: 'median before tip' })}
      </div>
      <div class="grid stagger">
        ${panel({ title: 'Beat-the-close dial', cls: 'c4', sub: 'Share of picks whose price beat the closing price', body: '<div id="cv-g1"></div>' })}
        ${panel({ title: 'Market direction dial', cls: 'c4', sub: 'Share of picks where the no-vig price moved toward our side (50% = coin flip)', body: '<div id="cv-g2"></div>' })}
        ${panel({ title: 'Close quality', cls: 'c4', sub: 'How many picks have a usable closing price', body: '<div id="cv-pie"></div>' })}
      </div>
      <div class="grid stagger">
        ${panel({ title: 'Distribution of EV vs the close', cls: 'c6', sub: `Exact-line closes only (n=${o.rows_with_exact_close}); the end bins pool anything beyond ±10%. Right of zero = we beat the closing price.`, body: '<div id="cv-hist"></div>' })}
        ${panel({ title: 'Bet price vs closing price', cls: 'c6', sub: 'No-vig probability of our side. Above the dashed line = the market moved toward us.', body: '<div id="cv-sc"></div><div class="legend"><span><i style="background:var(--win)"></i>Won</span><span><i style="background:var(--loss)"></i>Lost</span><span><i style="background:var(--faint)"></i>Pending</span></div>' })}
      </div>
      <div class="grid stagger">
        ${panel({ title: 'By stat', cls: 'c8', body: `<div class="tablewrap"><table><thead><tr><th>Stat</th><th class="n">Picks</th><th class="n">Exact close</th><th class="n">Beat close</th><th class="n">Moved toward</th><th class="n">Avg no-vig move</th><th class="n">Hit</th><th class="n">ROI</th></tr></thead><tbody>${groups.map(([k, v]) => `<tr><td class="pl">${esc(k.toUpperCase())}</td><td class="n">${v.rows}</td><td class="n">${v.rows_with_exact_close}</td><td class="n">${F.pct(v.beat_close_rate, 0)}</td><td class="n">${F.pct(v.moved_toward_side_rate, 0)}</td><td class="n ${F.tone(v.mean_no_vig_move)}">${F.pts(v.mean_no_vig_move)}</td><td class="n">${F.pct(v.hit_rate)}</td><td class="n ${F.tone(v.roi_units_per_bet)}">${F.pct(v.roi_units_per_bet)}</td></tr>`).join('')}${ml ? `<tr><td class="pl t-muted">All model leans</td><td class="n">${CLV.model_leans.all_rows}</td><td class="n">${ml.rows_with_exact_close}</td><td class="n">${F.pct(ml.beat_close_rate, 0)}</td><td class="n">${F.pct(ml.moved_toward_side_rate, 0)}</td><td class="n ${F.tone(ml.mean_no_vig_move)}">${F.pts(ml.mean_no_vig_move)}</td><td class="n">${F.pct(ml.hit_rate)}</td><td class="n ${F.tone(ml.roi_units_per_bet)}">${F.pct(ml.roi_units_per_bet)}</td></tr>` : ''}</tbody></table></div>` })}
        ${panel({ title: 'How to read this', cls: 'c4', body: `<div class="note"><b>CLV is the market's verdict on a price.</b> A bet that beats the closing line was a better price than what the sharpest information ended up supporting.<br><br>
          <b>The −4.5% trap.</b> EV-vs-close multiplies the closing no-vig probability by your bet odds, so at −110 a bet that never moved already reads about −4.5%. Judge it against that baseline, and lean on <b>price moved our way</b> and <b>avg no-vig move</b>.<br><br>
          <b>Sample caveats.</b> ${pk.close_quality_counts ? `${pk.close_quality_counts.no_later_snapshot} of ${pk.all_rows} picks have no later snapshot and ${pk.close_quality_counts.line_moved} moved to a different line, so only ${pk.close_quality_counts.exact_line} are exact-close. ` : ''}Closes are the nearest T-5 / T-60 capture, not the true last price.</div>` })}
      </div>`;
    $('#cv-g1').innerHTML = C.gauge({ min: 0, max: 0.6, value: o.beat_close_rate, ref: 0.5, refLabel: '50%', zones: [{ from: 0, to: 0.25, color: C.css('--loss') }, { from: 0.25, to: 0.4, color: C.css('--amber') }, { from: 0.4, to: 0.6, color: C.css('--win') }], text: F.pct(o.beat_close_rate, 0), sub: `${o.rows_with_exact_close} exact closes`, minLabel: '0', maxLabel: '60%' });
    $('#cv-g2').innerHTML = C.gauge({ min: 0, max: 1, value: o.moved_toward_side_rate, ref: 0.5, refLabel: '50%', zones: [{ from: 0, to: 0.45, color: C.css('--loss') }, { from: 0.45, to: 0.55, color: C.css('--amber') }, { from: 0.55, to: 1, color: C.css('--win') }], text: F.pct(o.moved_toward_side_rate, 0), sub: `${F.pct(o.moved_away_rate, 0)} moved against us`, minLabel: '0', maxLabel: '100%' });
    const cq = pk.close_quality_counts || {};
    C.pie($('#cv-pie'), { size: 170, center: { text: String(pk.all_rows || 0), sub: 'picks' }, legend: 'bottom', slices: [
      { label: 'Exact-line close', value: cq.exact_line, color: C.css('--accent-2') }, { label: 'No later snapshot', value: cq.no_later_snapshot, color: C.css('--faint') }, { label: 'Line moved at close', value: cq.line_moved, color: C.css('--loss') }] });
    const h = CLV.hist || [];
    C.bars($('#cv-hist'), { height: 250, yFmt: (v) => v.toFixed(0), data: h.map((b) => ({ label: `${(b.lo * 100).toFixed(0)}%`, value: b.n, color: b.hi <= 0 ? C.css('--loss') : C.css('--win'), tip: tipRows(`${(b.lo * 100).toFixed(0)}% to ${(b.hi * 100).toFixed(0)}%`, [['Picks', b.n]]) })), xTicks: 6 });
    C.scatter($('#cv-sc'), { height: 280, xDomain: [0.3, 0.8], yDomain: [0.3, 0.8], diag: true, xLabel: 'no-vig probability at bet time', yLabel: 'at the close', groups: [{ color: C.css('--faint'), points: (CLV.scatter || []).map((p) => ({ x: p.b, y: p.c, color: p.res === 'WIN' ? C.css('--win') : p.res === 'LOSS' ? C.css('--loss') : C.css('--faint'), r: 5, tip: tipRows(p.stat, [['Bet', F.pct(p.b)], ['Close', F.pct(p.c)], ['Move', F.pts(p.c - p.b)], ['Result', p.res || 'pending']]) })) }] });
    countUp(root);
  }

  /* =====================================================================
     MODEL LAB
     ===================================================================== */
  function vModels(root) {
    const props = MODELS.props || [];
    if (!props.length) { root.innerHTML = panel({ title: 'Model Lab', cls: 'c12', body: unavail('Model metrics') }); return; }
    const live = new Set(((D.scopes.season.by_prop_only) || []).map((r) => r.prop));
    const lanes = MODELS.lanes || {}, cal = (MODELS.calibration || {}).rows || [], cand = MODELS.candidate || {}, bt = MODELS.backtest || {};
    const featProps = Object.keys(MODELS.features || {});
    if (!state.feat || !featProps.includes(state.feat)) state.feat = featProps.includes('AST') ? 'AST' : featProps[0];
    root.innerHTML = `
      <div class="grid stagger">
        ${panel({ title: 'Production models: what the evidence says', cls: 'c12', sub: 'Real-line out-of-fold accuracy, meta-classifier AUC, and the always-UNDER check. Live lanes are the props that actually published this season.', body: `<div class="tablewrap"><table><thead><tr><th>Prop</th><th>Live this season</th><th class="n">Real lines</th><th class="n">OOF hit rate</th><th class="n">Meta AUC</th><th class="n">Features</th><th class="n">MAE</th><th class="n">Audit n</th><th class="n">Model</th><th class="n">Always-UNDER</th><th class="n">Edge</th><th>Audit</th></tr></thead><tbody>${props.map((p) => `<tr><td class="pl">${esc(p.prop)}</td><td>${live.has(p.prop) ? chip('LIVE', 'win') : chip('off', '')}</td><td class="n">${F.n(p.n_real)}</td><td class="n">${hitBar({ hit: p.real_hit })}</td><td class="n ${p.meta_auc > 0.6 ? 't-win' : ''}">${p.meta_auc != null ? p.meta_auc.toFixed(3) : '–'}</td><td class="n">${p.n_feat ?? '–'}<span class="t-muted">/${p.n_feat_orig ?? '–'}</span></td><td class="n">${p.mae != null ? p.mae.toFixed(2) : '–'}</td><td class="n">${p.audit_n ?? '–'}</td><td class="n">${p.audit_model != null ? p.audit_model.toFixed(1) + '%' : '–'}</td><td class="n">${p.audit_under != null ? p.audit_under.toFixed(1) + '%' : '–'}</td><td class="n ${F.tone(p.audit_edge)}">${p.audit_edge != null ? (p.audit_edge >= 0 ? '+' : '−') + Math.abs(p.audit_edge).toFixed(1) + ' pts' : '–'}</td><td>${p.audit_status ? chip(p.audit_status, p.audit_status === 'PASS' ? 'win' : 'loss') : '<span class="t-muted">–</span>'}</td></tr>`).join('')}</tbody></table></div><div class="note" style="margin-top:10px">Meta AUC ≈ 0.50 means the classifier cannot rank winners from losers; only STL (${fmtAuc('STL')}) and BLK (${fmtAuc('BLK')}) clear 0.6, which matches the UNDER-only lane policy. The audit compares against always betting UNDER on pick history (as of ${esc(String(MODELS.audit_ts || '').slice(0, 10))}).</div>` })}
      </div>
      ${shadowSection()}
      <div class="grid stagger">
        ${panel({ title: 'Meta-classifier AUC dials', cls: 'c12', sub: 'How well each prop\'s meta-classifier ranks winners from losers. 0.50 (amber tick) is a coin flip; only STL and BLK clear 0.60, which is why the live card is limited to UNDER lanes.', body: `<div class="dial-sm">${props.map((p) => dial(p.prop + (live.has(p.prop) ? ' · LIVE' : ''), C.gauge({ size: 170, min: 0.45, max: 0.7, value: p.meta_auc, ref: 0.5, zones: [{ from: 0.45, to: 0.5, color: C.css('--loss') }, { from: 0.5, to: 0.6, color: C.css('--amber') }, { from: 0.6, to: 0.7, color: C.css('--win') }], text: p.meta_auc != null ? p.meta_auc.toFixed(3) : '–', minLabel: '.45', maxLabel: '.70' }), `${F.n(p.n_real)} real lines`)).join('')}</div>` })}
      </div>
      <div class="grid stagger">
        ${panel({ title: 'UNDER-lane selection lift', cls: 'c6', sub: lanes.caveat ? esc(lanes.caveat) : '', tools: seg('lane', [['development', 'Development'], ['holdout', 'Holdout']], state.lane), body: '<div id="ml-lane"></div><div class="note" id="ml-lane-note"></div>' })}
        ${panel({ title: 'Feature importance', cls: 'c6', sub: 'Top drivers of the selected-feature model', tools: `<select id="ml-feat">${featProps.map((f) => `<option ${f === state.feat ? 'selected' : ''}>${f}</option>`).join('')}</select>`, body: '<div id="ml-feat-bars"></div>' })}
      </div>
      <div class="grid stagger">
        ${panel({ title: 'Calibration on the holdout', cls: 'c6', sub: `Brier score and expected calibration error on ${(MODELS.calibration && MODELS.calibration.split) ? esc(MODELS.calibration.split.holdout_date_min + ' → ' + MODELS.calibration.split.holdout_date_max) : 'the holdout'}. Lower is better.`, body: `<div class="tablewrap"><table><thead><tr><th>Prop</th><th class="n">n</th><th class="n">Brier raw</th><th class="n">Brier cal.</th><th class="n">ECE raw</th><th class="n">ECE cal.</th><th>Calibration</th></tr></thead><tbody>${cal.map((c) => { const d = c.brier_raw - c.brier_cal; return `<tr><td class="pl">${esc(c.prop)}</td><td class="n">${c.n}</td><td class="n">${c.brier_raw.toFixed(4)}</td><td class="n">${c.brier_cal.toFixed(4)}</td><td class="n">${c.ece_raw.toFixed(3)}</td><td class="n">${c.ece_cal.toFixed(3)}</td><td>${d > 0.0001 ? chip('helps', 'win') : d < -0.0001 ? chip('hurts', 'loss') : chip('neutral')}</td></tr>`; }).join('')}</tbody></table></div>` })}
        ${panel({ title: 'Shadow candidate vs production', cls: 'c6', sub: cand.id ? `${esc(cand.id)} · frozen future holdout since ${esc(cand.start || '')} · <b>not promoted</b>` : '', body: candidatePanel(cand) })}
      </div>
      <div class="grid stagger">
        ${panel({ title: 'Cut-off backtest: top-N daily picks', cls: 'c6', sub: bt.days ? `${bt.days} days, ${esc(bt.start)} → ${esc(bt.end)}, models retrained before the window` : '', body: '<div id="ml-bt"></div><div class="note">Older backtest on PrizePicks-era lines. Useful for ranking quality, not for forward expectations.</div>' })}
        ${panel({ title: 'Production integrity', cls: 'c6', body: integrityPanel() })}
      </div>`;
    const drawLane = () => {
      const blk = (lanes.results || {})[state.lane] || {}, pt = blk.per_target || {}, cb = blk.combined;
      const rows = Object.entries(pt).map(([k, v]) => ({ label: k.toUpperCase(), sub: `n=${v.n} · hit ${F.pct(v.hit_rate, 0)}`, est: v.lift, lo: v.lift_ci95_low, hi: v.lift_ci95_high, value: F.pts(v.lift, 0), tip: tipRows(k.toUpperCase(), [['Picks', v.n], ['Hit', F.pct(v.hit_rate)], ['Matched random UNDER', F.pct(v.matched_random_under_rate)], ['Lift', `<b>${F.pts(v.lift)}</b>`], ['95% CI', `${F.pts(v.lift_ci95_low)} to ${F.pts(v.lift_ci95_high)}`]]) }));
      if (cb) rows.unshift({ label: 'LIVE LANES', sub: `AST+BLK+STL · n=${cb.n}`, est: cb.lift, lo: cb.lift_ci95_low, hi: cb.lift_ci95_high, value: F.pts(cb.lift, 0), color: C.css('--accent'), tip: tipRows('Production lanes combined', [['Picks', cb.n], ['Hit', F.pct(cb.hit_rate)], ['Matched random', F.pct(cb.matched_random_under_rate)], ['Lift', `<b>${F.pts(cb.lift)}</b>`]]) });
      C.forest($('#ml-lane'), { rows, domain: [-0.2, 0.6], ref: 0, refLabel: 'no lift', fmt: (v) => (v * 100).toFixed(0) + 'pt', rowH: 40, labelW: 130 });
      const per = state.lane === 'holdout' ? lanes.hold : lanes.dev;
      $('#ml-lane-note').innerHTML = per ? `Top pick per prop per date vs a date/prop-matched random UNDER, ${per.dates} dates (${esc(per.date_min)} → ${esc(per.date_max)}). Positive lift with a CI above 0 is the evidence behind the ranking policy.` : '';
    };
    drawShadowForest();
    drawLane(); onSeg(root, 'lane', (v) => { state.lane = v; drawLane(); });
    const drawFeat = () => {
      const f = (MODELS.features || {})[state.feat] || [], mx = Math.max(...f.map((x) => x.v), 1e-9);
      $('#ml-feat-bars').innerHTML = C.hbars(f.map((x) => ({ label: x.f, value: x.v, max: mx, color: C.css('--violet'), text: (x.v * 100).toFixed(1) + '%', tip: `<b>${esc(x.f)}</b><br><span class="k">Importance</span> ${(x.v * 100).toFixed(2)}%` })));
    };
    drawFeat(); $('#ml-feat', root).addEventListener('input', (e) => { state.feat = e.target.value; drawFeat(); });
    if (bt.tops) C.bars($('#ml-bt'), { height: 220, yFmt: (v) => (v * 100).toFixed(0) + '%', pos: C.css('--accent-2'), data: bt.tops.map((t) => ({ label: `Top ${t.k}`, value: t.hit, tip: tipRows(`Top ${t.k} per day`, [['Record', `${t.w}-${t.l}`], ['Hit', F.pct(t.hit)]]) })), xTicks: 4 });
  }
  function fmtAuc(p) { const r = (MODELS.props || []).find((x) => x.prop === p); return r && r.meta_auc != null ? r.meta_auc.toFixed(2) : '–'; }
  function candidatePanel(c) {
    if (!c || !c.results) return '<div class="empty">No candidate holdout file.</div>';
    const all = c.results.find((r) => r.prop === 'ALL'), per = c.results.filter((r) => r.prop !== 'ALL');
    const dates = all ? all.dates : 0, need = c.min_dates || 15, pctDone = Math.min(100, (dates / need) * 100);
    const rowsDone = c.min_rows ? Math.min(1, (c.rows || 0) / c.min_rows) : 0;
    return `<div style="display:flex;gap:26px;flex-wrap:wrap;margin-bottom:12px"><div class="ringrow">${C.miniRing(pctDone / 100, { size: 76, text: `${dates}/${need}`, color: C.css('--accent') })}<div class="rt"><b>Dates</b><span>${need} needed before review</span></div></div><div class="ringrow">${C.miniRing(rowsDone, { size: 76, text: rowsDone >= 1 ? '✓' : `${F.n(c.rows)}`, color: C.css('--accent-2') })}<div class="rt"><b>Contracts</b><span>${F.n(c.rows)} of ${c.min_rows} minimum</span></div></div></div>
      <div class="tablewrap"><table><thead><tr><th>Prop</th><th class="n">Rows</th><th class="n">Candidate</th><th class="n">Production</th><th class="n">Always-UNDER</th><th class="n">vs prod.</th></tr></thead><tbody>${[...(all ? [all] : []), ...per].map((r) => `<tr><td class="pl">${esc(r.prop)}</td><td class="n">${r.rows}</td><td class="n">${F.pct(r.cand)}</td><td class="n">${F.pct(r.prod)}</td><td class="n">${F.pct(r.under)}</td><td class="n ${F.tone(r.vs_prod)}" data-tip="${esc('95% CI ' + F.pts(r.vs_prod_ci && r.vs_prod_ci[0]) + ' to ' + F.pts(r.vs_prod_ci && r.vs_prod_ci[1]))}">${F.pts(r.vs_prod)}</td></tr>`).join('')}</tbody></table></div>
      <div class="note" style="margin-top:8px">Nothing is promoted unless it clears ${need} dates and ${c.min_rows} rows <i>and</i> beats production and always-UNDER with a positive interval.</div>`;
  }
  /* ---------- shadow models (DEC-109 market-anchored probability, DEC-108 UNDER lanes): paper only, never published ---------- */
  const llTxt = (v) => (v != null && isFinite(v) ? v.toFixed(4) : '–');
  const llDiff = (v) => (v != null && isFinite(v) ? (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(4) : '–');
  // lower log loss is better: a negative difference whose interval stays below zero is the only clearly good case
  const llTone = (b) => (b && b.hi != null && b.hi < 0 ? 't-win' : b && b.lo != null && b.lo > 0 ? 't-loss' : 't-amber');
  const progBar = (n, need) => `<span class="barcell"><span class="minibar"><i style="width:${Math.min(100, (n / need) * 100).toFixed(0)}%;background:var(--accent)"></i></span><span class="num">${F.n(n)}/${need}</span></span>`;

  function shadowMarketPanel(m) {
    if (!m || !m.installed) return panel({ title: 'Shadow model: market-anchored probability', cls: 'c12', body: '<div class="empty">The shadow model is not installed (output/market_probability_shadow.json is missing).</div>' });
    const ev = m.evaluation;
    const chips = [chip(m.status === 'shadow_only' ? 'shadow only' : String(m.status || 'unknown'), 'info'), chip(`trained through ${m.trained_through || '–'}`), chip(`${F.n(m.training_rows)} fit rows · ${F.n(m.training_dates)} dates`)];
    if (ev) chips.push(ev.promotion_eligible ? chip('promotion eligible', 'win') : chip('not promotion-eligible', 'amber'));
    if (m.evaluation_hash_ok === true) chips.push(chip('evidence file verified', 'win'));
    if (m.evaluation_hash_ok === false) chips.push(chip('evidence file changed since install', 'loss'));
    if (!ev) return panel({ title: 'Shadow model: market-anchored probability', cls: 'c8', body: `<div class="chips-row" style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px">${chips.join('')}</div><div class="empty">Its evaluation file could not be read. Check the warnings banner.</div>` });
    const tests = [
      ['Frozen prior-season fit', ev.frozen, 'Fit once on last season, scored on this season\'s forward offers.'],
      ['Expanding, prior dates only', ev.walk, 'Refit before each date using only earlier whole dates.'],
      ['Same rows · anchored model', ev.anchored_same_rows, 'Offers that also have an archived incumbent probability; anchored model vs the market.'],
      ['Same rows · incumbent (calibrated)', ev.incumbent_same_rows, 'The same offers scored by the current production probability, for comparison.'],
    ].filter((t) => t[1]);
    const sub = (t) => `<div class="t-muted" style="font-size:11px;font-weight:400">${t}</div>`;
    const rowHtml = tests.map(([label, b, tip]) => `<tr><td class="pl" data-tip="${esc(tip)}">${esc(label)}${sub(`${F.n(b.rows)} offers · ${F.n(b.dates)} dates`)}</td><td class="n">${llTxt(b.market)}</td><td class="n">${llTxt(b.model)}</td><td class="n ${llTone(b)}">${llDiff(b.diff)}${sub(`${llDiff(b.lo)} to ${llDiff(b.hi)}`)}</td><td class="n">${b.under_picks == null ? '–' : b.under_picks === 0 ? '<span class="t-muted">none qualify</span>' : `${F.n(b.under_picks)}${sub(`ROI ${F.pct(b.under_roi)}`)}`}</td></tr>`).join('');
    const forestNote = (ev.by_prop || []).length ? '<div class="psub" style="margin:14px 0 2px">Change in log loss vs the market by prop (×1000, expanding fit). Left of the dashed line is better than the market.</div><div id="sh-forest"></div>' : '';
    return panel({
      title: 'Shadow model: market-anchored probability', cls: 'c8',
      sub: `Starts from FanDuel's no-vig price and learns only a small correction. Scored on offers from ${esc(F.dateY(ev.forward_start))} to ${esc(F.dateY(ev.data_end))}. Paper only; the live card does not use it.`,
      body: `<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px">${chips.join('')}</div>
        <div class="tablewrap"><table><thead><tr><th>Test</th><th class="n">Market</th><th class="n">Model</th><th class="n" data-tip="Model minus market, with the 95% interval from resampling whole dates">Difference · 95% CI</th><th class="n" data-tip="UNDER picks the model would take at +1% EV or better at the offered price">UNDERs at +1% EV</th></tr></thead><tbody>${rowHtml}</tbody></table></div>
        <div class="note" style="margin-top:8px">Log loss, lower is better. The gain over the market is real but tiny (about 0.001), the intervals only just clear zero, and no UNDER reaches +1% EV at the offered price, so this is a better probability estimate, not a betting edge. ${esc(ev.promotion_reason || '')}</div>${forestNote}`,
    });
  }

  function shadowProspectivePanel(m) {
    if (!m || !m.installed) return '';
    const pro = m.prospective || {}, need = m.min_binary || 300, needD = m.min_dates || 30;
    let body;
    if (!pro.present) {
      body = '<div class="empty">No prospective rows yet. The next FanDuel prediction run starts the log, and each offer is graded by the morning run after its game.</div>';
    } else {
      const dates = pro.dates || 0, bin = pro.binary || 0;
      const vs = (pro.vs_market || []).map((v) => `<tr><td class="pl">${esc(v.label)}</td><td class="n">${F.n(v.n)}</td><td class="n ${v.diff != null ? (v.diff < 0 ? 't-win' : v.diff > 0 ? 't-loss' : '') : ''}">${llDiff(v.diff)}</td></tr>`).join('');
      body = `<div style="display:flex;gap:26px;flex-wrap:wrap;margin-bottom:12px"><div class="ringrow">${C.miniRing(Math.min(1, dates / needD), { size: 76, text: `${dates}/${needD}`, color: C.css('--accent') })}<div class="rt"><b>Settled dates</b><span>${needD} needed before review</span></div></div><div class="ringrow">${C.miniRing(Math.min(1, bin / need), { size: 76, text: F.n(bin), color: C.css('--accent-2') })}<div class="rt"><b>Settled offers</b><span>${F.n(need)} needed before review</span></div></div></div>
        <div class="stat-row"><div><b>${F.n(pro.rows)}</b><span>offers logged</span></div><div><b>${F.n(pro.pending)}</b><span>awaiting grade</span></div><div><b>${F.n((pro.push || 0) + (pro.dnp || 0))}</b><span>push / DNP</span></div></div>
        ${bin ? `<div class="tablewrap" style="margin-top:10px"><table><thead><tr><th>Log loss vs market</th><th class="n">Settled rows</th><th class="n">Difference</th></tr></thead><tbody>${vs}</tbody></table></div>` : '<div class="note" style="margin-top:10px">The comparison with the market appears once offers settle.</div>'}
        ${pro.fallbacks ? `<div class="note" style="margin-top:8px">${F.n(pro.fallbacks)} offer(s) lacked enough history and used the market price unchanged.</div>` : ''}`;
    }
    return panel({
      title: 'Prospective record', cls: 'c4', sub: pro.present && pro.first ? `Offers from ${esc(F.dateY(pro.first))} to ${esc(F.dateY(pro.last))}` : 'Collecting from the next prediction run',
      body: `${body}<div class="note" style="margin-top:10px">Needs ${needD} settled dates <i>and</i> ${F.n(need)} settled offers before a review, and reaching them promotes nothing. The retrospective numbers on the left do not count toward this record.</div>`,
    });
  }

  function shadowLanesPanel(L) {
    const need = (L && L.min_binary) || 300, lanes = (L && L.lanes) || [];
    const rows = lanes.map((l) => `<tr><td class="pl">${esc(l.label)}<div class="t-muted" style="font-size:11px;font-weight:400">${esc(l.rule)}</div></td><td class="n">${F.n(l.rows)}</td><td class="n">${progBar(l.binary || 0, need)}</td><td class="n">${F.n(l.dates)}</td><td class="n">${l.binary ? hitBar({ hit: l.hit }) : '<span class="t-muted">–</span>'}</td><td class="n ${l.binary ? F.tone(l.units) : ''}">${l.binary ? F.u(l.units) : '–'}</td><td class="n ${l.binary ? F.tone(l.roi) : ''}">${l.binary ? F.pct(l.roi) : '–'}</td><td class="n t-muted">${F.n(l.push)} · ${F.n(l.dnp)} · ${F.n(l.pending)}</td></tr>`).join('');
    return panel({
      title: 'Shadow UNDER lanes · paper only', cls: 'c12',
      sub: L && L.present ? 'The first eligible FanDuel offer per player, prop and day, at the exact recorded price. Nothing here is published, bet or used for training.' : 'Collecting from the next prediction run. Nothing here is published, bet or used for training.',
      body: `<div class="tablewrap"><table><thead><tr><th>Lane</th><th class="n">Logged</th><th class="n">Settled (win/loss)</th><th class="n">Dates</th><th class="n">Hit</th><th class="n">Paper units</th><th class="n">ROI</th><th class="n">Push · DNP · pending</th></tr></thead><tbody>${rows || '<tr><td colspan="8" class="empty">No lanes are defined.</td></tr>'}</tbody></table></div>
        <div class="note" style="margin-top:8px">About 40 combinations were screened to find these two, so their earlier results may be luck; only rows logged from now on count. The ${need}-outcome review minimum applies to each lane on its own (they are not pooled) and reaching it is not authorization to bet.</div>`,
    });
  }

  function shadowSection() {
    const S = MODELS.shadow || {};
    return `<div class="grid stagger">${shadowMarketPanel(S.market)}${shadowProspectivePanel(S.market)}</div>
      <div class="grid stagger">${shadowLanesPanel(S.lanes)}</div>`;
  }

  function drawShadowForest() {
    const el = $('#sh-forest'), ev = ((MODELS.shadow || {}).market || {}).evaluation;
    if (!el || !ev) return;
    const rows = (ev.by_prop || []).filter((p) => p.diff != null).map((p) => {
      const est = p.diff * 1000, lo = p.lo != null ? p.lo * 1000 : null, hi = p.hi != null ? p.hi * 1000 : null;
      return { label: p.prop, sub: `${F.n(p.rows)} offers`, est, lo, hi, value: (est > 0 ? '+' : est < 0 ? '−' : '') + Math.abs(est).toFixed(2),
        color: hi != null && hi < 0 ? C.css('--win') : lo != null && lo > 0 ? C.css('--loss') : C.css('--accent-2'),
        tip: tipRows(p.prop, [['Offers', F.n(p.rows)], ['Market', llTxt(p.market)], ['Model', llTxt(p.model)], ['Difference', `<b>${llDiff(p.diff)}</b>`], ['95% CI', `${llDiff(p.lo)} to ${llDiff(p.hi)}`]]) };
    });
    if (!rows.length) return;
    const lo = Math.min(0, ...rows.map((r) => (r.lo != null ? r.lo : r.est))), hi = Math.max(0, ...rows.map((r) => (r.hi != null ? r.hi : r.est))), pad = (hi - lo) * 0.08 || 0.5;
    C.forest(el, { rows, domain: [lo - pad, hi + pad], ref: 0, refLabel: 'market', fmt: (v) => v.toFixed(1), rowH: 30, labelW: 120 });
  }
  function integrityPanel() {
    const m = HEALTH.maintenance || {}, rc = HEALTH.reconciliation || {};
    return `<div class="stat-row"><div><b class="${m.production_unchanged ? '' : 't-loss'}">${m.production_unchanged ? 'Locked' : 'Changed'}</b><span>production bundle</span></div><div><b>${m.artifacts_hashed ?? '–'}</b><span>artifacts hashed</span></div><div><b>${esc(m.status || '–')}</b><span>${esc(m.mode || 'maintenance')} job</span></div></div>
      <div class="note">Last training maintenance ended <b>${esc(String(m.ended_at || '–').replace('T', ' ').slice(0, 16))}</b>. The Sunday job trains into an isolated candidates folder and verifies every production file hash is unchanged (DEC-057).<br><br>Grade reconciliation checked <b>${F.n(rc.checked)}</b> stored grades: <b class="${rc.current_conflicts ? 't-amber' : 't-win'}">${rc.current_conflicts ?? 0} conflicts</b>, ${rc.outcome_conflicts ?? 0} that flip a W/L outcome.</div>`;
  }

  /* =====================================================================
     BETSLIPS
     ===================================================================== */
  function vSlips(root) {
    if (!SLIPS.raw) { root.innerHTML = panel({ title: 'Betslips', cls: 'c12', body: unavail('Betslip history') }); return; }
    const s = SLIPS[state.slipMode];
    root.innerHTML = `
      <div class="banner warn"><b>Double-count check</b><span>${SLIPS.duplicates} of ${SLIPS.raw.n} logged slips repeat the same legs under a second label (e.g. "Power Play" and "UNDER-Only"). <b>Unique</b> counts each leg set once. If both labels were really staked, switch to <b>All logged</b>.</span></div>
      <div class="field" style="margin-bottom:12px">${seg('smode', [['unique', 'Unique slips'], ['raw', 'All logged rows']], state.slipMode)}<span class="note">$20 stake per slip · net = payout − stake on settled slips</span></div>
      <div class="kpis stagger">
        ${kpi({ label: 'Slip record', valHtml: rec(s), tone: 'info', sub: `${F.pct(s.hit)} hit · ${s.pending} pending` })}
        ${kpi({ label: 'Net P&L', count: s.net, fmt: 'money', val: F.sgnMoney(s.net), vcls: F.tone(s.net), tone: s.net >= 0 ? 'win' : 'loss', sub: `on ${F.money(s.staked)} staked` })}
        ${kpi({ label: 'ROI', count: s.roi, fmt: 'pct', dec: 0, val: F.pct(s.roi, 0), vcls: F.tone(s.roi), tone: 'violet', sub: 'dominated by PrizePicks multipliers' })}
        ${kpi({ label: 'Slips logged', val: F.n(s.n), tone: 'amber', sub: `${SLIPS.raw.n} raw · ${SLIPS.unique.n} unique` })}
      </div>
      <div class="grid stagger">
        ${panel({ title: 'Slip results', cls: 'c4', sub: state.slipMode === 'unique' ? 'Unique slips' : 'All logged rows', body: '<div id="sl-pie-res"></div>' })}
        ${panel({ title: 'Slips by type', cls: 'c4', sub: 'Logged rows per slip label', body: '<div id="sl-pie-type"></div>' })}
        ${panel({ title: 'ROI dial', cls: 'c4', sub: 'Net dollars over dollars staked on settled slips', body: '<div id="sl-gauge"></div>' })}
      </div>
      <div class="grid stagger">
        ${panel({ title: 'Cumulative P&L', cls: 'c8', sub: 'Unique slips by slate date', body: '<div id="sl-cum"></div>' })}
        ${panel({ title: 'By era', cls: 'c4', sub: 'Unique slips only', body: `<div class="tablewrap"><table><thead><tr><th>Era</th><th class="n">Rec.</th><th class="n">Net</th></tr></thead><tbody>${(SLIPS.by_era || []).map((e) => `<tr><td class="pl">${esc(ERA_LABEL[e.era] || e.era)}${lowN(e.n, 20)}</td><td class="n">${e.w}-${e.l}</td><td class="n ${F.tone(e.net)}">${F.sgnMoney(e.net)}</td></tr>`).join('')}</tbody></table></div><div class="note" style="margin-top:8px">PrizePicks pays fixed multipliers (e.g. 3× for a 2-pick Power Play), so a modest hit rate can still print a large ROI. The backfilled era is reconstructed, not live.</div>` })}
      </div>
      <div class="grid stagger">
        ${panel({ title: 'By slip type', cls: 'c5', body: `<div class="tablewrap"><table><thead><tr><th>Type</th><th class="n">n</th><th class="n">Record</th><th class="n">Hit</th><th class="n">Net</th></tr></thead><tbody>${(SLIPS.by_type || []).map((t) => `<tr><td class="pl">${esc(t.slip_type)}</td><td class="n">${t.n}</td><td class="n">${t.w}-${t.l}</td><td class="n">${F.pct(t.hit, 0)}</td><td class="n ${F.tone(t.net)}">${F.sgnMoney(t.net)}</td></tr>`).join('')}</tbody></table></div>` })}
        ${panel({ title: 'Recent slips', cls: 'c7', sub: 'Newest first', body: `<div style="display:grid;gap:9px;max-height:520px;overflow:auto">${(SLIPS.recent || []).slice().reverse().slice(0, 24).map((r) => `<div class="slip"><div class="top2"><span><b>${esc(r.type)}</b> <span class="t-muted">· ${esc(F.dateY(r.date))}</span></span><span style="display:flex;gap:6px">${r.payout != null ? chip(F.money(r.payout) + ' back', '') : ''}${resChip(r.result)}</span></div>${legChips(r.legs)}</div>`).join('')}</div>` })}
      </div>`;
    const d = SLIPS.daily || [];
    C.line($('#sl-cum'), { height: 260, split: true, labels: d.map((x) => F.date(x.date)), yFmt: (v) => F.money(v), series: [{ color: C.css('--win'), area: true, data: d.map((x) => ({ y: x.cum })) }], tip: (i) => tipRows(F.dateY(d[i].date), [['Slate', `${F.sgnMoney(d[i].net)} · ${d[i].n} slip(s)`], ['Cumulative', `<b class="${F.tone(d[i].cum)}">${F.sgnMoney(d[i].cum)}</b>`]]) });
    C.pie($('#sl-pie-res'), { size: 180, center: { text: String(s.n), sub: 'slips' }, legend: 'bottom', slices: [
      { label: 'Won', value: s.w, color: C.css('--win') }, { label: 'Lost', value: s.l, color: C.css('--loss') },
      { label: 'Push', value: s.p, color: C.css('--amber') }, { label: 'Pending', value: s.pending, color: C.css('--accent-2') }] });
    C.pie($('#sl-pie-type'), { size: 180, hole: 0, legend: 'bottom', slices: (SLIPS.by_type || []).map((t) => ({ label: t.slip_type, value: t.n, text: `${t.w}-${t.l}`, tip: tipRows(t.slip_type, [['Rows', t.n], ['Record', `${t.w}-${t.l}`], ['Net', F.sgnMoney(t.net)]]) })) });
    $('#sl-gauge').innerHTML = C.gauge({ min: -1, max: 2, value: s.roi, ref: 0, refLabel: '0', zones: [{ from: -1, to: 0, color: C.css('--loss') }, { from: 0, to: 2, color: C.css('--win') }], text: F.pct(s.roi, 0), sub: `${F.sgnMoney(s.net)} on ${F.money(s.staked)}`, minLabel: '−100%', maxLabel: '+200%' });
    onSeg(root, 'smode', (v) => { state.slipMode = v; vSlips(root); });
    countUp(root);
  }

  /* =====================================================================
     SYSTEM HEALTH
     ===================================================================== */
  const STATUS_TONE = { ok: { c: 'win', t: 't-win' }, warn: { c: 'amber', t: 't-amber' }, fail: { c: 'loss', t: 't-loss' }, late: { c: 'info', t: 't-info' } };
  function ageChip(h) { return h == null ? chip('–') : h < 14 ? chip(F.ago(h), 'win') : h < 36 ? chip(F.ago(h), 'amber') : chip(F.ago(h), 'loss'); }
  function vHealth(root) {
    const pl = HEALTH.pipeline || {}, runs = pl.recent_runs || [], data = HEALTH.data || {}, git = HEALTH.git || {}, flags = HEALTH.flags || [];
    const withSteps = runs.filter((r) => r.steps);
    if (state.run < 0 || state.run >= withSteps.length) state.run = Math.max(0, withSteps.length - 1); // newest by default
    const credits = pl.credits || [], lastCredit = credits.length ? credits[credits.length - 1] : null;
    const lastRun = runs[runs.length - 1];
    root.innerHTML = `
      <div class="kpis stagger">
        ${kpi({ label: 'Last pipeline run', val: lastRun ? ({ ok: 'OK', warn: 'WARN', fail: 'FAIL', late: 'LATE' }[lastRun.status] || lastRun.status) : '–', vcls: lastRun ? STATUS_TONE[lastRun.status].t : '', tone: lastRun && lastRun.status === 'fail' ? 'loss' : lastRun && lastRun.status === 'late' ? 'info' : 'win', sub: lastRun ? `${esc(lastRun.start)} · ${lastRun.warn} warn · ${lastRun.err} err` : 'no logs' })}
        ${kpi({ label: 'Last box score', val: F.dateY((data.nba_data || {}).last_date), tone: 'info', sub: `${F.n((data.nba_data || {}).rows)} rows · ${F.n((data.nba_data || {}).players)} players` })}
        ${kpi({ label: 'Lines archive', val: F.n((data.lines || {}).rows), tone: 'violet', sub: `through ${F.dateY((data.lines || {}).last_date)}` })}
        ${kpi({ label: 'Odds API credits', count: lastCredit && lastCredit.credits, fmt: 'int', val: lastCredit ? F.n(lastCredit.credits) : '–', tone: lastCredit && lastCredit.credits < 2000 ? 'loss' : 'amber', sub: lastCredit ? `as of ${esc(lastCredit.t)}` : 'not logged' })}
        ${kpi({ label: 'Production models', val: (HEALTH.maintenance || {}).production_unchanged ? 'Locked' : '–', vcls: '', tone: 'win', sub: `${(HEALTH.maintenance || {}).artifacts_hashed || '–'} artifacts hashed` })}
        ${kpi({ label: 'Repo', val: esc(git.commit || '–'), tone: 'info', sub: `${esc(git.branch || '')} · ${git.dirty ?? '?'} uncommitted` })}
      </div>
      ${flags.map((f) => `<div class="banner bad"><b>Integrity</b><span>${esc(f.text)} ${esc(f.hint || '')}</span></div>`).join('')}
      <div class="grid stagger">
        ${panel({ title: 'Pipeline run', cls: 'c12', sub: 'Step timings and everything the step logged at WARNING or above', tools: `<select id="hl-run">${withSteps.map((r, i) => `<option value="${i}" ${i === state.run ? 'selected' : ''}>${esc(r.start)} · ${r.status}</option>`).reverse().join('')}</select>`, body: '<div id="hl-steps"></div>' })}
      </div>
      <div class="grid stagger">
        ${panel({ title: 'Odds API credits', cls: 'c4', sub: 'Remaining credits vs the highest reading in the last 45 days', body: '<div id="hl-gauge"></div>' })}
        ${panel({ title: 'Lines archive by source', cls: 'c4', sub: `${F.n((data.lines || {}).rows)} rows in historical_lines.csv`, body: '<div id="hl-src"></div>' })}
        ${panel({ title: 'Box scores by season', cls: 'c4', sub: `${F.n((data.nba_data || {}).rows)} player-game rows`, body: '<div id="hl-seasons"></div>' })}
      </div>
      <div class="grid stagger">
        ${panel({ title: 'Run history', cls: 'c5', sub: `Last ${runs.length} runs parsed from logs/run_*.log`, body: `<div class="tablewrap" style="max-height:340px"><table><thead><tr><th>Started</th><th>Status</th><th class="n">Warn</th><th class="n">Err</th><th class="n">Min</th></tr></thead><tbody>${runs.slice().reverse().map((r) => `<tr><td>${esc(r.start)}</td><td>${chip(r.status, STATUS_TONE[r.status].c)}</td><td class="n">${r.warn}</td><td class="n">${r.err}</td><td class="n">${(r.secs / 60).toFixed(0)}</td></tr>`).join('')}</tbody></table></div>` })}
        ${panel({ title: 'Odds API credits', cls: 'c7', sub: 'Remaining provider credits, each time the pipeline logged them', body: '<div id="hl-credits"></div>' })}
      </div>
      <div class="grid stagger">
        ${panel({ title: 'File freshness', cls: 'c7', sub: 'Ring fills as a file ages toward 48 hours', body: `<div class="tablewrap"><table><thead><tr><th></th><th>File</th><th>Modified</th><th>Age</th></tr></thead><tbody>${(HEALTH.files || []).map((f) => `<tr><td>${C.miniRing(f.age_h == null ? 0 : Math.max(0.03, 1 - Math.min(f.age_h / 48, 1)), { size: 34, text: '', color: f.age_h == null ? C.css('--faint') : f.age_h < 14 ? C.css('--accent-2') : f.age_h < 36 ? C.css('--violet') : C.css('--loss') })}</td><td class="pl">${esc(f.file)}</td><td class="t-muted">${esc(String(f.mtime || '–').replace('T', ' '))}</td><td>${ageChip(f.age_h)}</td></tr>`).join('')}</tbody></table></div>` })}
        ${panel({ title: 'Grade reconciliation', cls: 'c5', sub: 'Stored grades that disagree with the current box score', body: reconTable() })}
      </div>`;
    const drawRun = () => {
      const r = withSteps[state.run]; if (!r) { $('#hl-steps').innerHTML = '<div class="empty">No run detail.</div>'; return; }
      $('#hl-steps').innerHTML = `${r.note ? `<div class="banner warn" style="margin-bottom:10px"><b>Note</b><span>${esc(r.note)}</span></div>` : ''}<div class="timeline">${r.steps.map((s) => `<div class="step ${s.err ? 'fail' : s.warn ? 'warn' : ''} ${(s.msgs || []).length ? 'wide' : ''}"><div class="sid">STEP ${esc(s.id)}</div><div class="snm">${esc(s.name)}</div><div class="sd">${s.secs >= 90 ? (s.secs / 60).toFixed(1) + ' min' : s.secs + ' s'}${s.err ? ` · <span class="t-loss">${s.err} err</span>` : ''}${s.warn ? ` · <span class="t-amber">${s.warn} warn</span>` : ''}</div>${(s.msgs || []).map((m) => `<div class="msg ${m.level}">${esc(m.text)}</div>`).join('')}</div>`).join('')}</div>`;
    };
    const peak = credits.length ? Math.max(...credits.map((c) => c.credits)) : 0, cr = lastCredit ? lastCredit.credits : null;
    $('#hl-gauge').innerHTML = C.gauge({ min: 0, max: peak || 1, value: cr, zones: [{ from: 0, to: (peak || 1) * 0.1, color: C.css('--loss') }, { from: (peak || 1) * 0.1, to: (peak || 1) * 0.3, color: C.css('--amber') }, { from: (peak || 1) * 0.3, to: peak || 1, color: C.css('--win') }], text: cr != null ? F.n(cr) : '–', sub: peak ? `${F.pct(cr / peak, 0)} of the ${F.n(peak)} peak` : '', minLabel: '0', maxLabel: F.n(peak) });
    const srcs = Object.entries((data.lines || {}).by_source || {}).sort((a, b) => b[1] - a[1]), srcTop = srcs.slice(0, 6), srcRest = srcs.slice(6);
    C.pie($('#hl-src'), { size: 170, center: { text: String(srcs.length), sub: 'sources' }, legend: 'bottom', slices: [...srcTop.map(([k, v]) => ({ label: k.replace(/^odds_api_/, '').replace(/_/g, ' '), value: v })), ...(srcRest.length ? [{ label: `${srcRest.length} others`, value: srcRest.reduce((a, [, v]) => a + v, 0), color: C.css('--faint') }] : [])] });
    C.pie($('#hl-seasons'), { size: 170, hole: 0, legend: 'bottom', slices: Object.entries((data.nba_data || {}).seasons || {}).map(([k, v]) => ({ label: k, value: v })) });
    drawRun(); $('#hl-run', root).addEventListener('input', (e) => { state.run = +e.target.value; drawRun(); });
    if (credits.length > 1) C.line($('#hl-credits'), { height: 210, zero: false, labels: credits.map((c) => F.date(c.t.slice(0, 10))), yFmt: (v) => F.n(v / 1000, 1) + 'k', series: [{ color: C.css('--accent'), area: true, data: credits.map((c) => ({ y: c.credits })) }], tip: (i) => tipRows(credits[i].t, [['Credits', `<b>${F.n(credits[i].credits)}</b>`]]) });
    else $('#hl-credits').innerHTML = '<div class="empty">Not enough credit readings in the logs.</div>';
    countUp(root);
  }
  function reconTable() {
    const rc = HEALTH.reconciliation || {};
    if (!rc.conflicts || !rc.conflicts.length) return '<div class="empty">No conflicts. Stored grades match the box score.</div>';
    return `<div class="tablewrap"><table><thead><tr><th>File</th><th>Date</th><th>Player</th><th class="n">Stored</th><th class="n">Box</th><th>Result</th></tr></thead><tbody>${rc.conflicts.map((c) => `<tr><td class="t-muted">${esc((c.source_file || '').replace('_history.csv', ''))}</td><td>${esc(c.game_date)}</td><td class="pl">${esc(c.player)} <span class="sub">${esc(c.prop)}</span></td><td class="n">${c.stored_actual}</td><td class="n">${c.observed_actual}</td><td>${c.stored_result === c.observed_result ? chip('same', '') : chip(`${c.stored_result}→${c.observed_result}`, 'loss')}</td></tr>`).join('')}</tbody></table></div>`;
  }

  /* =====================================================================
     DECISIONS & TASKS
     ===================================================================== */
  function vDecisions(root) {
    const tasks = (D.tasks || []).filter((t) => !t.done), done = (D.tasks || []).length - tasks.length;
    const urgent = tasks.filter((t) => /NEEDS JAKE|BLOCKER/i.test(t.tag || '')), rest = tasks.filter((t) => !urgent.includes(t));
    const card = (t, u) => `<div class="todo ${u ? 'urgent' : ''}"><div>${t.tag ? chip(t.tag.length > 34 ? t.tag.slice(0, 32) + '…' : t.tag, u ? 'amber' : 'info') : chip('task')}</div><p>${esc(t.text)}</p></div>`;
    root.innerHTML = `
      <div class="grid stagger">
        ${panel({ title: 'Needs your decision', cls: 'c5', sub: `${urgent.length} urgent · ${tasks.length} open · ${done} done in the NOW list of TASKS.md`, body: `<div style="display:grid;gap:9px;max-height:700px;overflow:auto">${urgent.map((t) => card(t, true)).join('')}${rest.slice(0, 14).map((t) => card(t, false)).join('') || ''}${!tasks.length ? '<div class="empty">Nothing open.</div>' : ''}</div>` })}
        ${panel({ title: 'Decision log', cls: 'c7', sub: `${(D.decisions || []).length} decisions parsed from DECISIONS.md. This is the project's memory: why every rule exists.`, tools: '<input type="search" id="dc-q" placeholder="Filter decisions…">', body: '<div class="tl" id="dc-list"></div>' })}
      </div>`;
    const draw = () => {
      const q = ($('#dc-q', root).value || '').toLowerCase();
      const rows = (D.decisions || []).filter((d) => !q || (d.id + ' ' + d.title).toLowerCase().includes(q));
      $('#dc-list', root).innerHTML = rows.map((d) => `<div class="item"><div><span class="id">${esc(d.id)}</span> <span class="t-muted">${esc(d.date || '')}</span></div><div class="t">${esc(d.title)}</div></div>`).join('') || '<div class="empty">No match.</div>';
    };
    $('#dc-q', root).addEventListener('input', draw); draw();
  }

  /* =====================================================================
     SHELL: nav, header, routing, theme
     ===================================================================== */
  const VIEWS = [
    { id: 'overview', label: 'Overview', icon: 'overview', render: vOverview, scoped: true },
    { id: 'today', label: 'Today', icon: 'today', render: vToday },
    { id: 'perf', label: 'Results', icon: 'perf', render: vPerf, scoped: true },
    { id: 'explorer', label: 'Picks', icon: 'explorer', render: vExplorer },
    { id: 'gates', label: 'Gates', icon: 'gates', render: vGates },
    { id: 'clv', label: 'Market', icon: 'clv', render: vClv },
    { id: 'models', label: 'Models', icon: 'models', render: vModels },
    { id: 'slips', label: 'Slips', icon: 'slips', render: vSlips },
    { id: 'health', label: 'Health', icon: 'health', render: vHealth },
    { id: 'decisions', label: 'Memory', icon: 'decisions', render: vDecisions },
  ];
  const dirty = new Set();
  function logoSvg() { return '<svg viewBox="0 0 32 32" fill="none" stroke="url(#lg)" stroke-width="2" stroke-linecap="round"><defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#ff2a2a"/><stop offset="1" stop-color="#ffffff"/></linearGradient></defs><circle cx="16" cy="16" r="11"/><path d="M5 16h22M16 5v22M8 8c5 4 5 12 0 16M24 8c-5 4-5 12 0 16"/></svg>'; }
  function renderShell() {
    const flagBad = (HEALTH.flags || []).length || ((HEALTH.pipeline || {}).last_run || {}).err;
    $('#rail').innerHTML = `${defs3d()}<div class="logo" title="Jake's NBA Dashboard">${logoSvg()}</div><nav class="nav">${VIEWS.map((v, i) => `<button data-view="${v.id}" title="${v.label}">${icon3d(v.icon, i)}<span class="flat" aria-hidden="true">${svgi(v.icon)}</span><span>${v.label}</span>${v.id === 'health' && flagBad ? '<i class="badge-dot"></i>' : ''}</button>`).join('')}</nav><div class="spacer"></div>`;
    // the icon tilts toward the pointer (skipped when the OS asks for reduced motion)
    if (!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches)) {
      $$('.nav button', $('#rail')).forEach((b) => {
        const tilt = $('.tilt', b);
        if (!tilt) return;
        b.addEventListener('pointermove', (e) => {
          const r = b.getBoundingClientRect();
          const x = (e.clientX - r.left) / r.width - 0.5, y = (e.clientY - r.top) / r.height - 0.5;
          tilt.style.setProperty('--ty', (x * 80).toFixed(1) + 'deg');
          tilt.style.setProperty('--tx', (-y * 70).toFixed(1) + 'deg');
        });
        b.addEventListener('pointerleave', () => { tilt.style.removeProperty('--ty'); tilt.style.removeProperty('--tx'); });
      });
    }
    $('#rail').addEventListener('click', (e) => { const b = e.target.closest('[data-view]'); if (b) go(b.dataset.view); });
    const meta = D.meta || {}, sd = D.scopes[state.scope].summary;
    // how old is this snapshot? (a static page cannot refresh itself, so say so when a daily rebuild was missed)
    const builtAt = meta.generated_at ? new Date(meta.generated_at) : null;
    const ageH = builtAt && !isNaN(builtAt.getTime()) ? Math.max(0, (Date.now() - builtAt.getTime()) / 36e5) : null;
    const ageTxt = ageH == null ? '' : ageH < 1 ? 'under 1 hour ago' : ageH < 48 ? Math.round(ageH) + ' hours ago' : Math.floor(ageH / 24) + ' days ago';
    const titleHtml = BRAND ? `<h1 class="brandmark"><img src="${BRAND}" alt="Dime City Data" decoding="async"></h1>` : `<h1><span class="grad">Jake's NBA</span> Dashboard</h1>`;
    $('#top').innerHTML = `<div class="${BRAND ? 'titleblock' : ''}">${titleHtml}<div class="sub">Built ${esc((meta.generated_at || '').replace('T', ' '))}${ageTxt ? ' (' + ageTxt + ')' : ''} ·${F.n(meta.total_picks)} picks logged ${esc(F.dateY(meta.first_pick))} → ${esc(F.dateY(meta.last_pick))}${meta.git && meta.git.commit ? ' · ' + esc(meta.git.branch) + '@' + esc(meta.git.commit) : ''}</div></div><div class="grow"></div>
      <div id="scopeseg">${seg('scope', [['season', '2026-27 season'], ['live', 'All live'], ['all', 'Everything']], state.scope)}</div>
      <span class="chip ${TODAY.pending ? 'info pulse' : 'win'}"><i class="pip"></i>${TODAY.pending ? TODAY.pending + ' picks live' : 'card settled'}</span>
      ${WALL ? `<button class="iconbtn" id="wallbtn" title="Show or hide the wallpaper" aria-label="Show or hide the wallpaper" aria-pressed="${wallOn}">${svgi('image')}</button>` : ''}
      <button class="iconbtn" id="themebtn" title="Toggle theme" aria-label="Toggle theme"></button>`;
    onSeg($('#top'), 'scope', (v) => { state.scope = v; store.set('nba.scope', v); VIEWS.forEach((x) => { if (x.scoped) dirty.add(x.id); }); show(state.view, true); });
    $('#themebtn').onclick = () => setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
    if (WALL) $('#wallbtn').onclick = () => { wallOn = !wallOn; store.set('nba.wall', wallOn ? 'on' : 'off'); applyWall(); };
    applyWall();
    setTheme(store.get('nba.theme', 'dark'), true);
    const w = (meta.warnings || []);
    const staleBanner = ageH != null && ageH >= STALE_HOURS
      ? `<div class="banner bad"><b>Out of date</b><span>This page was built ${esc(ageTxt)}, so results, tonight's card and injury context may have changed since. Rebuild it with run_dashboard.bat; if the daily rebuild keeps failing, check the pipeline log.</span></div>` : '';
    $('#warnings').innerHTML = staleBanner + (w.length ? `<div class="banner warn"><b>${w.length} section${w.length > 1 ? 's' : ''} degraded</b><span>${w.map(esc).join(' · ')}</span></div>` : '');
  }
  function setTheme(t, init) {
    document.documentElement.dataset.theme = t; store.set('nba.theme', t);
    $('#themebtn').innerHTML = svgi(t === 'dark' ? 'sun' : 'moon');
    if (!init) { VIEWS.forEach((v) => dirty.add(v.id)); show(state.view, true); }
  }
  const rendered = {};
  function show(id, force) {
    const v = VIEWS.find((x) => x.id === id) || VIEWS[0];
    state.view = v.id;
    $$('[data-view]', $('#rail')).forEach((b) => b.setAttribute('aria-current', String(b.dataset.view === v.id)));
    $('#scopeseg').style.display = v.scoped ? '' : 'none';
    $$('.view').forEach((x) => x.classList.remove('on'));
    let el = $('#view-' + v.id);
    if (!el) { el = document.createElement('div'); el.className = 'view'; el.id = 'view-' + v.id; $('#views').appendChild(el); }
    // show the view BEFORE rendering it: charts measure their container, and a display:none
    // container has no width, which would leave them blank until a resize event
    el.classList.add('on');
    if (force || dirty.has(v.id) || !rendered[v.id]) {
      try { v.render(el); } catch (e) { console.error(e); el.innerHTML = `<div class="banner bad"><b>View error</b><span>${esc(v.label)}: ${esc(e.message)}</span></div>`; }
      rendered[v.id] = true; dirty.delete(v.id);
    }
    if (location.hash.slice(1) !== v.id) history.replaceState(null, '', '#' + v.id);
    $$('[data-go]', el).forEach((b) => { b.onclick = () => go(b.dataset.go); });
    window.scrollTo({ top: 0, behavior: 'instant' in window ? 'instant' : 'auto' });
  }
  function go(id) { show(id); }
  window.addEventListener('keydown', (e) => {
    if (e.target.closest && e.target.closest('input,select,textarea') || e.ctrlKey || e.metaKey || e.altKey) return;
    const i = '1234567890'.indexOf(e.key); if (i >= 0 && VIEWS[i]) go(VIEWS[i].id);
  });
  window.addEventListener('hashchange', () => { const h = location.hash.slice(1); if (VIEWS.some((v) => v.id === h) && h !== state.view) show(h); });
  renderShell();
  show(VIEWS.some((v) => v.id === location.hash.slice(1)) ? location.hash.slice(1) : 'overview');
})();
