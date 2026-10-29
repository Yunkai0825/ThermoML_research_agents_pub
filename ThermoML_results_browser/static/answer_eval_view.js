/**
 * answer_eval_view.js -- the Result panel's Eval switch.
 *
 * Draws an answer's saved evaluation over its own result.md: each reviewed
 * claim (eval/claim_review.json) tinted by its category, each compared value
 * (eval/evaluation.json) marked by its outcome, and in grey the lines whose
 * compared values no reviewed claim holds. Selecting a highlight shows its
 * review and where its numbers came from in the side panel, above the list of
 * related DOI:block ids. Eval starts on whenever the answer has an
 * evaluation; the switch shows the plain answer.
 *
 * Usage: const view = ThermoMLAnswerEval.create(card, {renderMd, linkify});
 *        view.show(resultMd, evaluation);  // evaluation: {numbers, claims} or null
 * ``card`` holds an ``.eval-switch`` input, a ``.result-md`` body and an
 * ``.eval-side`` column (styles: answer_eval_view.css).
 */
(function () {
  'use strict';

  const OUTCOME = { exact: 'Exact', near_miss: 'Near miss', miss: 'Miss' };
  const BADGE = { exact: 'text-bg-success', near_miss: 'text-bg-warning', miss: 'text-bg-danger' };
  const NUMERIC_GROUP = { property_value: 'Property', measurement_condition: 'Conditions',
                          stored_uncertainty: 'Stored uncertainty', pipeline_artifact: 'Pipeline artifact' };
  const CATEGORY = { database_verified: 'Database verified', domain_knowledge: 'Domain knowledge',
                     incorrect: 'Incorrect', unresolved: 'Unresolved' };
  const CATEGORY_HELP = {
    database_verified: 'Matches the original ThermoML records: system, property, phase, composition, conditions, units and source.',
    domain_knowledge: 'Correct physical chemistry, a supported literature fact, or a valid conversion, calculation or interpretation.',
    incorrect: 'The value, definition, scope, attribution or inference is demonstrably wrong.',
    unresolved: 'An open empirical assertion the review could not settle from the evidence it had.'
  };
  const EVIDENCE_LABEL = {
    manual_physical_chemistry_assessment: 'Assessment', original_database_audit: 'Original database audit',
    domain_reference: 'Domain reference', original_record_locators: 'Original records',
    direct_line_check: 'Direct line check', original_database: 'Original database',
    external_primary_or_definition: 'Primary source or definition'
  };
  // Markers put into result.md before it is rendered and turned into elements after. A value: U+E000, its index in
  // U+E010..U+E019 digits, U+E001, the value, U+E002. A claim: U+2E4C, its index, U+2E4E, the claim, U+2E4F (Unicode
  // punctuation, so bold and italics next to a claim's ends parse as they do without it).
  const V_OPEN = '', V_MID = '', V_CLOSE = '';
  const C_OPEN = '⹌', C_MID = '⹎', C_CLOSE = '⹏';
  const MARKER = /[--⹌⹎⹏]/;
  const MARKERS = /[--⹌⹎⹏]/g;
  const ENCODED = /%EE%80%(?:8[0-2]|9[0-9])|%E2%B9%8[CEF]/gi;
  const TOKEN = /([-]+)||⹌([-]+)⹎|⹏/g;
  const LIST_MARKER = /^\s*(?:\d+[.)]|[-*+])\s/;
  const LINE_PREFIX = /^[ \t]*(?:>[ \t]?)*[ \t]*(?:#{1,6}[ \t]+|(?:[-*+]|\d+[.)])[ \t]+(?:\[[ xX]\][ \t]+)?)?/;
  const NO_WRAP = /^(TABLE|THEAD|TBODY|TFOOT|TR|COLGROUP|UL|OL|DL)$/;
  const CLOSE_BUTTON = '<button type="button" class="btn-close btn-sm ms-auto ev-clear" aria-label="Clear the selection"></button>';
  const SIDE_HTML =
    '<div class="eval-side-inner">' +
      '<div class="p-2 border-bottom small ev-legend">' +
        '<div class="ev-legend-row ev-legend-claims"><span class="text-muted">Claims:</span>' +
          ' <span class="ev-sw ev-cat-database_verified ms-1"></span> Database verified' +
          ' <span class="ev-sw ev-cat-domain_knowledge ms-2"></span> Domain knowledge' +
          ' <span class="ev-sw ev-cat-incorrect ms-2"></span> Incorrect' +
          ' <span class="ev-sw ev-cat-unresolved ms-2"></span> Unresolved</div>' +
        '<div class="ev-legend-row ev-legend-numbers"><span class="text-muted">Numbers:</span>' +
          ' <span class="ev-sw ev-sw-exact ms-1"></span> Exact' +
          ' <span class="ev-sw ev-sw-near_miss ms-2"></span> Near miss' +
          ' <span class="ev-sw ev-sw-miss ms-2"></span> Miss' +
          ' <span class="ev-sw ev-sw-grey ms-2"></span> <span class="ev-grey-label"></span></div>' +
        '<div class="ev-status mt-2 text-muted"></div>' +
      '</div>' +
      '<div class="p-2 ev-selected"></div>' +
      '<div class="p-2 border-top ev-related"></div>' +
    '</div>';

  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }
  // Eval files are written by agents: their outcome and category words are looked up, never trusted as markup or classes.
  function own(map, key) { return Object.prototype.hasOwnProperty.call(map, key) ? map[key] : undefined; }
  function cls(s) { return String(s == null ? '' : s).replace(/[^A-Za-z0-9_-]/g, '_'); }
  function outcomeLabel(o) { return own(OUTCOME, o) || String(o); }
  function categoryLabel(c) { return own(CATEGORY, c) || String(c); }
  function linkifyUrls(s) {
    return esc(s).replace(/https?:\/\/[^\s<>"')\]]+/g, url => '<a href="' + url + '" target="_blank" rel="noopener">' + url + '</a>');
  }
  function pfx(u) {
    return window.__pfx__ ? window.__pfx__(u) : u;
  }
  function encodeIndex(n) { return Array.from(String(n), d => String.fromCharCode(0xE010 + Number(d))).join(''); }
  function decodeIndex(s) { return Number(Array.from(s, c => c.charCodeAt(0) - 0xE010).join('')); }

  function paperLink(doi, block) {
    const href = pfx('/paper/' + encodeURI(doi) + '#block-' + block);
    return '<a href="' + esc(href) + '" target="_blank" rel="noopener">' + esc(doi + ':' + block) + '</a>';
  }

  // final_reference comes as a list of points, one point, or "NA: ..." text.
  function sources(ref) {
    if (ref == null || ref === '') return [];
    if (typeof ref === 'string') return [{ na: ref }];
    return (Array.isArray(ref) ? ref : [ref]).filter(r => r && typeof r === 'object' && r.doi).map(function (r) {
      const block = typeof r.block === 'number' || /^\d+$/.test(String(r.block)) ? 'PROPblock_' + r.block : String(r.block);
      const point = r.point == null || r.point === '' ? '' : (/^\d+$/.test(String(r.point)) ? 'BLKpoint_' + r.point : String(r.point));
      return { doi: r.doi, block: block, point: point, stored: r.stored_value, unit: r.stored_unit,
               converted: r.converted_stored_value };
    });
  }

  // The raw text of a line or a claim, its compared values marked.
  function quoteHtml(raw, startColumn, inside) {
    const chars = Array.from(raw);
    let html = '', at = 0;
    inside.slice().sort((a, b) => a.column - b.column).forEach(function (v) {
      const from = v.column - startColumn, to = v.end_column - startColumn;
      html += esc(chars.slice(at, from).join(''));
      html += '<mark class="ev-val ev-' + cls(v.outcome) + '">' + esc(chars.slice(from, to).join('')) + '</mark>';
      at = to;
    });
    return html + esc(chars.slice(at).join(''));
  }

  function valueCard(v, index) {
    let html = '<div class="ev-value" data-card="' + index + '">';
    html += '<div class="d-flex align-items-center gap-1 flex-wrap"><mark class="ev-val ev-' + cls(v.outcome) + ' fw-semibold">' + esc(v.text) + '</mark>';
    html += '<span class="badge ' + (own(BADGE, v.outcome) || 'text-bg-secondary') + '">' + esc(outcomeLabel(v.outcome)) + '</span>';
    if (String(v.misassigned).toLowerCase() === 'true') html += '<span class="badge text-bg-dark" title="The number matches, but the answer gives it to the wrong system, state or source">Misassigned</span>';
    if (own(NUMERIC_GROUP, v.group)) html += '<span class="badge text-bg-light border">' + esc(NUMERIC_GROUP[v.group]) + '</span>';
    html += '</div>';
    const what = [v.property, v.claimed_unit ? 'unit ' + v.claimed_unit : ''].filter(Boolean).map(esc).join(' &middot; ');
    if (what) html += '<div class="text-muted small mt-1">' + what + '</div>';
    if (v.display && v.display.reviewed_grouped_literal) {
      html += '<div class="small mt-1">Read in the answer as <strong>' + esc(v.display.reviewed_grouped_literal) + '</strong></div>';
    }
    if (v.group === 'pipeline_artifact') {
      const checks = [['numeric_outcome', 'Number'], ['condition_outcome', 'Conditions'], ['source_outcome', 'Source'],
                      ['unit_outcome', 'Units'], ['basis_outcome', 'Basis'], ['raw_numeric_outcome', 'Raw numeral']];
      html += '<div class="small mt-1">' + checks.filter(x => v[x[0]] != null)
        .map(x => esc(x[1] + ': ' + outcomeLabel(v[x[0]]))).join(' &middot; ') + '</div>';
      if (v.precision_policy && v.precision_policy.applied) {
        html += '<div class="small mt-1">Fitting precision: default 4 significant digits (ROUND_HALF_UP); match: ' + esc(v.precision_policy.matched_by) + '</div>';
        html += '<details><summary>Numerical precision comparison</summary><ul class="ev-evidence mb-0 ps-3">' + evidenceItem(v.precision_policy) + '</ul></details>';
      }
      const artifact = v.artifact_reference || (typeof v.final_reference === 'object' ? v.final_reference : null);
      const curve = v.curve_verification || (artifact && artifact.curve_verification);
      if (curve) {
        html += '<div class="small mt-1"><strong>Reviewer-calculated verification of saved curve</strong><div>Calculated reference: ' +
          esc(curve.calculated_reference_value) + ' ' + esc(curve.calculated_reference_unit || '') + '</div>';
        html += '<div>Source-precision interval: ' + esc((curve.source_precision_interval || []).join(' to ')) + '</div>';
        if (artifact) html += '<div>Native coefficient input (' + esc(artifact.key || 'coefficient') + '): ' + esc(artifact.stored_value) + ' ' + esc(artifact.stored_unit || '') + '</div>';
        html += '<div class="text-muted">The calculated reference is not a stored pipeline scalar; original coefficient inputs are retained.</div>';
        html += '<details><summary>Calculation provenance and precision</summary><ul class="ev-evidence mb-0 ps-3">' + evidenceItem(curve) + '</ul></details></div>';
      }
      html += '<div class="small mt-1"><span class="text-muted">Pipeline evidence:</span>';
      if (artifact) html += '<ul class="ev-evidence mb-0 ps-3">' + evidenceItem(artifact) + '</ul>';
      else html += ' ' + esc(typeof v.final_reference === 'string' ? v.final_reference : 'NA; no stored artifact recorded');
      if (v.artifact_search) html += '<details><summary>Artifact search</summary><ul class="ev-evidence mb-0 ps-3">' + evidenceItem(v.artifact_search) + '</ul></details>';
      html += '</div>';
    }
    const refs = sources(v.final_reference);
    if (v.group !== 'pipeline_artifact' || refs.some(r => r.doi)) {
    html += '<div class="small mt-1"><span class="text-muted">Came from:</span>';
    if (!refs.length) html += ' <span class="text-muted">no stored source recorded</span>';
    refs.forEach(function (r) {
      if (r.na) { html += '<div class="text-muted">' + esc(r.na) + '</div>'; return; }
      html += '<div>' + paperLink(r.doi, r.block) + (r.point ? ' &middot; ' + esc(r.point) : '') + '</div>';
      const stored = [r.stored, r.unit].filter(x => x != null && x !== '').join(' ');
      if (stored) {
        html += '<div class="text-muted">stored ' + esc(stored) +
          (r.converted != null && r.converted !== '' && String(r.converted) !== String(r.stored) ? ' (converted ' + esc(r.converted) + ')' : '') + '</div>';
      }
    });
    html += '</div>';
    }
    if (v.band_check && typeof v.band_check === 'object') {
      const b = v.band_check;
      html += '<div class="small mt-1"><span class="text-muted">Checked as a range:</span> ' +
        esc([b.interval_outcome, b.contained != null ? 'contained ' + b.contained : ''].filter(Boolean).join(', ')) + '</div>';
    }
    if (v.reason) html += '<div class="small mt-1"><span class="text-muted">Reason:</span> ' + esc(v.reason) + '</div>';
    return html + '</div>';
  }

  function humanize(key) {
    return EVIDENCE_LABEL[key] || String(key).replace(/_/g, ' ').replace(/^./, c => c.toUpperCase());
  }

  // One evidence item of a claim, whatever its shape: text, or an object whose fields are listed.
  function evidenceItem(item) {
    if (item == null || item === '') return '';
    if (typeof item !== 'object') return '<li>' + linkifyUrls(String(item)) + '</li>';
    const kind = item.type || item.kind;
    let html = '<li>' + (kind ? '<div class="fw-semibold">' + esc(humanize(kind)) + '</div>' : '');
    Object.keys(item).forEach(function (key) {
      if (key === 'type' || key === 'kind') return;
      const value = item[key];
      let shown;
      if (key === 'locators' && Array.isArray(value)) {
        shown = value.slice(0, 6).map(l => esc([l.doi, l.block != null ? 'block ' + l.block : '', l.point_1based != null ? 'point ' + l.point_1based : ''].filter(Boolean).join(' '))).join('; ') +
          (value.length > 6 ? ' and ' + (value.length - 6) + ' more' : '');
      } else if (key === 'records' && Array.isArray(value)) {
        shown = value.map(r => esc([r.doi, r.status].filter(Boolean).join(' · ')) + (r.reason ? ': ' + esc(r.reason) : '')).join('<br>');
      } else if (Array.isArray(value) && value.every(v => typeof v !== 'object')) {
        shown = value.map(v => linkifyUrls(String(v))).join(', ');
      } else if (value !== null && typeof value === 'object') {
        shown = '<code>' + esc(JSON.stringify(value)) + '</code>';
      } else {
        shown = linkifyUrls(String(value));
      }
      html += '<div><span class="text-muted">' + esc(key.replace(/_/g, ' ')) + ':</span> ' + shown + '</div>';
    });
    return html + '</li>';
  }

  function create(card, options) {
    const renderMd = options.renderMd;
    const linkify = options.linkify || function () {};
    const resultEl = card.querySelector('.result-md');
    const side = card.querySelector('.eval-side');
    side.innerHTML = SIDE_HTML;
    const sideSelected = side.querySelector('.ev-selected');
    const sideRelated = side.querySelector('.ev-related');
    const sideStatus = side.querySelector('.ev-status');
    const evalSwitch = card.querySelector('.eval-switch');
    const switchWrap = evalSwitch.closest('.eval-switch-wrap') || evalSwitch;
    let current = null;   // the answer shown: {result, has_numbers, excerpts, related, claims}
    let values = [];      // its compared values: {x: excerpt index, v: value}
    let owner = [];       // per compared value: the index of the reviewed claim it is in, or -1
    let spans = [];       // per claim: the marked span [s, e) of its line (code points), or null when not placed
    let evalOn = false;

    function categoryOf(k) { const c = current.claims[k]; return c.status === 'unresolved' ? 'unresolved' : c.category; }

    // Put markers around every reviewed claim and every compared value at its recorded line and columns (1-based code
    // points, end exclusive) of result.md itself. A claim's marked span leaves out the line's own syntax (a list
    // marker, a quote or heading mark, a table row's outer pipes). A value that is not there, or is a list number, is
    // left unmarked and counted.
    function withMarkers(md) {
      const lines = md.split('\n');
      const events = new Map();
      function add(line, pos, kind, text) {
        if (!events.has(line)) events.set(line, new Map());
        const at = events.get(line);
        if (!at.has(pos)) at.set(pos, { closeV: [], closeC: [], openC: [], openV: [] });
        at.get(pos)[kind].push(text);
      }
      // A line is used only as the evaluated result.md has it (the page may show some lines rewritten).
      const shownAsEvaluated = n => own(current.source, String(n)) === lines[n - 1];
      spans = (current.claims || []).map(function (c, k) {
        const line = lines[c.line - 1] || '';
        const chars = Array.from(line);
        let s = c.column - 1, e = c.end_column - 1;
        if (!shownAsEvaluated(c.line) || chars.slice(s, e).join('') !== c.text) return null;
        const row = /^\s*\|/.test(line);
        if (row) {
          while (s < e && /[|\s]/.test(chars[s])) s += 1;
          while (e > s && /[|\s]/.test(chars[e - 1])) e -= 1;
        } else {
          const prefix = Array.from(line.match(LINE_PREFIX)[0]).length;
          if (s < prefix) s = Math.min(prefix, e);
        }
        // A marker between white space (or the line's edge) and "~" would let a "~" there end or start a strikethrough
        // the answer does not have; such an edge moves past the tildes.
        const blank = i => i < 0 || i >= chars.length || /\s/.test(chars[i]);
        if (blank(s - 1)) while (s < e && chars[s] === '~') s += 1;
        if (blank(e)) while (e > s && chars[e - 1] === '~') e -= 1;
        if (s >= e) return null;
        add(c.line, s, 'openC', C_OPEN + encodeIndex(k) + C_MID);
        add(c.line, e, 'closeC', C_CLOSE);
        return { s: s, e: e, wholeRow: row && c.text.trim() === line.replace(/\r$/, '').trim() };
      });
      values.forEach(function (item, index) {
        const line = current.excerpts[item.x].line;
        const text = lines[line - 1] || '';
        const chars = Array.from(text);
        const marker = text.match(LIST_MARKER);
        const markerEnd = marker ? Array.from(marker[0]).length : 0;
        const from = item.v.column - 1, to = item.v.end_column - 1;
        if (!shownAsEvaluated(line) || chars.slice(from, to).join('') !== item.v.text || from < markerEnd) return;
        add(line, from, 'openV', V_OPEN + encodeIndex(index) + V_MID);
        add(line, to, 'closeV', V_CLOSE);
      });
      events.forEach(function (at, line) {
        const chars = Array.from(lines[line - 1] || '');
        let out = '';
        for (let p = 0; p <= chars.length; p += 1) {
          const ev = at.get(p);
          if (ev) out += ev.closeV.join('') + ev.closeC.join('') + ev.openC.join('') + ev.openV.join('');
          if (p < chars.length) out += chars[p];
        }
        lines[line - 1] = out;
      });
      return lines.join('\n');
    }

    // Turn the markers into elements: a value's characters into <mark>, a claim's text into spans. Both may cross
    // element boundaries (a claim that starts inside bold text; a table row's cells). Markers that reached an
    // attribute (a link target) are removed.
    function placeMarkers(root) {
      root.querySelectorAll('*').forEach(function (el) {
        Array.from(el.attributes).forEach(function (a) {
          const clean = a.value.replace(MARKERS, '').replace(ENCODED, '');
          if (clean !== a.value) el.setAttribute(a.name, clean);
        });
      });
      const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
      const nodes = [];
      while (walker.nextNode()) nodes.push(walker.currentNode);
      const placed = { values: new Set(), claims: new Set() };
      let claim = -1, value = -1;
      nodes.forEach(function (node) {
        const text = node.nodeValue;
        if (claim < 0 && value < 0 && !MARKER.test(text)) return;
        const wrap = !NO_WRAP.test(node.parentNode.nodeName);
        const frag = document.createDocumentFragment();
        function emit(piece) {
          const clean = piece.replace(MARKERS, '');
          if (!clean) return;
          let el = document.createTextNode(clean);
          if (wrap && value >= 0) {
            const mark = document.createElement('mark');
            mark.className = 'ev-val ev-' + cls(values[value].v.outcome);
            mark.dataset.v = value;
            mark.title = outcomeLabel(values[value].v.outcome) + ': select to see where it came from';
            mark.appendChild(el);
            el = mark;
            placed.values.add(value);
          }
          if (wrap && claim >= 0) {
            const span = document.createElement('span');
            span.className = 'ev-claim ev-cat-' + cls(categoryOf(claim));
            span.dataset.claim = claim;
            span.title = categoryLabel(categoryOf(claim)) + ' claim: select to see the review';
            span.appendChild(el);
            el = span;
            placed.claims.add(claim);
          }
          frag.appendChild(el);
        }
        let last = 0, m;
        TOKEN.lastIndex = 0;
        while ((m = TOKEN.exec(text))) {
          emit(text.slice(last, m.index));
          if (m[1] !== undefined) value = decodeIndex(m[1]);
          else if (m[2] !== undefined) claim = decodeIndex(m[2]);
          else if (m[0] === V_CLOSE) value = -1;
          else claim = -1;
          last = TOKEN.lastIndex;
        }
        emit(text.slice(last));
        node.parentNode.replaceChild(frag, node);
      });
      // A claim that is a whole table row tints the row.
      spans.forEach(function (span, k) {
        if (!span || !span.wholeRow) return;
        const parts = Array.from(root.querySelectorAll('span.ev-claim[data-claim="' + k + '"]'));
        const row = parts.length && parts[0].closest('tr');
        if (!row || parts.some(p => p.closest('tr') !== row)) return;
        row.classList.add('ev-claim-row', 'ev-cat-' + cls(categoryOf(k)));
        row.dataset.claim = k;
        parts.forEach(p => { p.className = 'ev-claim-part'; });
      });
      return placed;
    }

    // Highlight, in grey, the answer line of an excerpt whose compared values no reviewed claim holds: a table row
    // whole, otherwise the text of its block between the line breaks around the value, leaving reviewed claims as
    // they are (a paragraph can hold several source lines; a list item's own text, not its sub-list).
    function wrapLine(block, mark, x) {
      const nodes = [];
      (function walk(node) {
        node.childNodes.forEach(function (child) {
          if (child.nodeType === 3) nodes.push(child);
          else if (child.nodeType === 1 && !/^(UL|OL|TABLE)$/.test(child.tagName)) walk(child);
        });
      })(block);
      const starts = [];
      let total = '';
      nodes.forEach(function (n) { starts.push(total.length); total += n.nodeValue; });
      const k = nodes.findIndex(n => mark.contains(n));
      if (k < 0) return;
      const from = total.lastIndexOf('\n', starts[k] - 1) + 1;
      let to = total.indexOf('\n', starts[k]);
      if (to < 0) to = total.length;
      nodes.forEach(function (n, j) {
        const s = starts[j], e = s + n.nodeValue.length;
        const a = Math.max(from, s), b = Math.min(to, e);
        if (a >= b || n.parentNode.closest('[data-claim]')) return;
        let target = n;
        if (b < e) target.splitText(b - s);
        if (a > s) target = target.splitText(a - s);
        const span = document.createElement('span');
        span.className = 'ev-line';
        span.dataset.ex = x;
        target.parentNode.insertBefore(span, target);
        span.appendChild(target);
      });
    }

    function placeLines(root) {
      let shown = 0;
      current.excerpts.forEach(function (excerpt, x) {
        const free = values.map((item, i) => i).filter(i => values[i].x === x && owner[i] < 0);
        const mark = free.map(i => root.querySelector('mark.ev-val[data-v="' + i + '"]')).find(Boolean);
        if (!mark) return;
        shown += 1;
        const row = mark.closest('tr');
        if (row) { row.classList.add('ev-row'); row.dataset.ex = x; return; }
        wrapLine(mark.closest('li, p, h1, h2, h3, h4, h5, h6, blockquote, dd, dt, pre, td, th') || root, mark, x);
      });
      return shown;
    }

    function clearMarks() {
      resultEl.querySelectorAll('.ev-sel').forEach(el => el.classList.remove('ev-sel'));
      resultEl.querySelectorAll('mark.ev-focus').forEach(el => el.classList.remove('ev-focus'));
    }

    // A new selection is read from its top; a selected value's card is then scrolled to.
    function focusValue(index) {
      side.querySelector('.eval-side-inner').scrollTop = 0;
      if (index == null) return;
      const mark = resultEl.querySelector('mark.ev-val[data-v="' + index + '"]');
      if (mark) mark.classList.add('ev-focus');
      const valueCardEl = sideSelected.querySelector('[data-card="' + index + '"]');
      if (valueCardEl) { valueCardEl.classList.add('ev-focus'); valueCardEl.scrollIntoView({ block: 'nearest' }); }
    }

    // A grey line: the compared values on it that no reviewed claim holds.
    function selectLine(x, focusIndex) {
      clearMarks();
      resultEl.querySelectorAll('[data-ex="' + x + '"]').forEach(el => el.classList.add('ev-sel'));
      const excerpt = current.excerpts[x];
      const mine = values.map((item, i) => i).filter(i => values[i].x === x && owner[i] < 0);
      const raw = (current.result.split('\n')[excerpt.line - 1] || '').replace(/\r$/, '');
      let html = '<div class="d-flex align-items-center gap-1 mb-1"><span class="fw-semibold">Line ' + esc(excerpt.line) + '</span>' +
        (current.claims ? '<span class="text-muted small">no reviewed claim holds these numbers</span>' : '') + CLOSE_BUTTON + '</div>';
      html += '<div class="ev-quote">' + quoteHtml(raw, 1, mine.map(i => values[i].v)) + '</div>';
      mine.forEach(i => { html += valueCard(values[i].v, i); });
      const inClaims = values.map((item, i) => i).filter(i => values[i].x === x && owner[i] >= 0).map(i => values[i].v.text);
      if (inClaims.length) html += '<div class="small text-muted mt-2">Also on this line, inside a reviewed claim: ' + esc(inClaims.join(', ')) + '</div>';
      const others = excerpt.values.filter(v => v.group === 'outside').map(v => v.text);
      if (others.length) html += '<div class="small text-muted mt-2">Also on this line, not compared: ' + esc(others.join(', ')) + '</div>';
      sideSelected.innerHTML = html;
      focusValue(focusIndex);
    }

    // A reviewed claim: its category, the review's reason and evidence, then the compared values inside it.
    function selectClaim(k, focusIndex) {
      clearMarks();
      resultEl.querySelectorAll('[data-claim="' + k + '"]').forEach(el => el.classList.add('ev-sel'));
      const c = current.claims[k];
      const cat = categoryOf(k);
      const mine = values.map((item, i) => i).filter(i => owner[i] === k);
      let html = '<div class="ev-cat-' + cls(cat) + '">';
      html += '<div class="d-flex align-items-center gap-1 mb-1"><span class="badge ev-badge-cat">' + esc(categoryLabel(cat)) + '</span>' +
        '<span class="fw-semibold">Line ' + esc(c.line) + '</span>' + CLOSE_BUTTON + '</div>';
      html += '<div class="small text-muted mb-1">' + esc(own(CATEGORY_HELP, cat) || '') + ' <span class="text-nowrap">(' + esc(c.claim_id) + ')</span></div>';
      html += '<div class="ev-quote ev-quote-claim">' + quoteHtml(c.text, c.column, mine.map(i => values[i].v)) + '</div>';
      if (c.reason) html += '<div class="small mt-2"><span class="text-muted">Review:</span> ' + linkifyUrls(c.reason) + '</div>';
      const evidence = Array.isArray(c.evidence) ? c.evidence.filter(e => e != null && e !== '') : (c.evidence == null ? [] : [c.evidence]);
      if (evidence.length) {
        html += '<div class="small mt-2"><div class="text-muted">Evidence (' + evidence.length + ')</div><ul class="ev-evidence mb-0 ps-3">' +
          evidence.map(evidenceItem).join('') + '</ul></div>';
      }
      html += '</div>';
      html += '<div class="mt-3 small"><span class="fw-semibold">Numbers in this claim</span> <span class="text-muted">from the numerical validation</span></div>';
      if (!current.has_numbers) html += '<div class="small text-muted">This answer has no numerical validation.</div>';
      else if (!mine.length) html += '<div class="small text-muted">No compared number in this claim.</div>';
      mine.forEach(i => { html += valueCard(values[i].v, i); });
      sideSelected.innerHTML = html;
      focusValue(focusIndex);
    }

    function clearSelection() {
      clearMarks();
      sideSelected.innerHTML = '<div class="text-muted">Select a highlight to see its review and where its numbers came from.</div>';
    }

    function relatedHtml() {
      if (!current.has_numbers) {
        return '<div class="fw-semibold mb-1">Related DOI:block ids</div><div class="text-muted">This answer has no numerical validation, so it has no list of compared sources.</div>';
      }
      const items = current.related.slice().sort((a, b) => b.values - a.values || a.doi.localeCompare(b.doi) || a.block - b.block);
      let html = '<div class="fw-semibold mb-1">Related DOI:block ids (' + items.length + ')</div><ul class="list-unstyled mb-0">';
      items.forEach(function (r) {
        html += '<li class="mb-1">' + paperLink(r.doi, 'PROPblock_' + r.block) + ' <span class="text-muted">' +
          (r.values ? r.values + (r.values === 1 ? ' value' : ' values') : 'claims or coverage') + '</span>' +
          (r.omitted ? ' <span class="badge text-bg-light border" title="Relevant to the question but left out of the answer">omitted</span>' : '') + '</li>';
      });
      return html + '</ul>';
    }

    function statusHtml(lines, placed) {
      let html = '';
      if (current.claims) {
        const counts = {};
        current.claims.forEach((c, k) => { const cat = categoryOf(k); counts[cat] = (counts[cat] || 0) + 1; });
        const parts = ['database_verified', 'domain_knowledge', 'incorrect', 'unresolved'].filter(cat => counts[cat])
          .map(cat => counts[cat] + ' ' + CATEGORY[cat].toLowerCase());
        const missing = current.claims.length - placed.claims.size;
        html += '<div>Claim review: ' + current.claims.length + ' claims (' + parts.join(', ') + ')' +
          (missing ? '; ' + missing + ' could not be placed on the answer' : '') + '</div>';
      } else {
        html += '<div>Claim review: none for this answer</div>';
      }
      if (current.has_numbers) {
        const inClaims = values.filter((item, i) => owner[i] >= 0).length;
        const missing = values.length - placed.values.size;
        html += '<div>Numerical validation: ' + placed.values.size + ' compared values' +
          (current.claims ? ', ' + inClaims + ' of them inside reviewed claims, the rest on ' + lines + ' grey lines' : ' on ' + lines + ' grey lines') +
          (missing ? '; ' + missing + ' could not be placed on the answer' : '') + '</div>';
        Object.keys(NUMERIC_GROUP).forEach(function (group) {
          const tally = { exact: 0, near_miss: 0, miss: 0 };
          values.filter(item => item.v.group === group).forEach(item => {
            if (Object.prototype.hasOwnProperty.call(tally, item.v.outcome)) tally[item.v.outcome] += 1;
          });
          html += '<div>' + esc(NUMERIC_GROUP[group]) + ': ' + tally.exact + ' Exact, ' +
            tally.near_miss + ' Near miss, ' + tally.miss + ' Miss</div>';
        });
      } else {
        html += '<div>Numerical validation: none for this answer</div>';
      }
      return html;
    }

    function render() {
      card.classList.toggle('eval-on', evalOn);
      side.classList.toggle('d-none', !evalOn);
      if (!evalOn) {
        resultEl.innerHTML = renderMd(current.result);
        linkify(resultEl);
        return;
      }
      resultEl.innerHTML = renderMd(withMarkers(current.result));
      const placed = placeMarkers(resultEl);
      const lines = placeLines(resultEl);
      linkify(resultEl);
      side.querySelector('.ev-legend-claims').classList.toggle('d-none', !current.claims);
      side.querySelector('.ev-legend-numbers').classList.toggle('d-none', !current.has_numbers);
      side.querySelector('.ev-grey-label').textContent = current.claims ? 'Numbers no reviewed claim holds' : 'Evaluated line';
      sideStatus.innerHTML = statusHtml(lines, placed);
      clearSelection();
      sideRelated.innerHTML = relatedHtml();
    }

    // Show an answer (the page's own result text) with its saved evaluation, if any; Eval starts on when there is one.
    function show(resultMd, evaluation) {
      const numbers = evaluation && evaluation.numbers;
      const claims = evaluation && Array.isArray(evaluation.claims) ? evaluation.claims : null;
      current = { result: resultMd || '', has_numbers: !!numbers, excerpts: numbers ? numbers.excerpts || [] : [],
                  related: numbers ? numbers.related || [] : [], claims: claims,
                  source: (evaluation && evaluation.lines) || {} };
      values = [];
      current.excerpts.forEach(function (excerpt, x) {
        (excerpt.values || []).forEach(function (v) { if (v.group !== 'outside') values.push({ x: x, v: v }); });
      });
      const reviewed = claims || [];
      owner = values.map(function (item) {
        const line = current.excerpts[item.x].line;
        return reviewed.findIndex(c => c.line === line && c.column <= item.v.column && item.v.end_column <= c.end_column);
      });
      const title = [numbers ? 'Numerical validation: eval/evaluation.json' : '',
                     claims ? 'Claim review: eval/claim_review.json' : ''].filter(Boolean).join('\n');
      switchWrap.title = title || 'No evaluation for this answer';
      evalSwitch.disabled = !title;
      evalSwitch.checked = !!title;
      evalOn = !!title;
      render();
    }

    evalSwitch.addEventListener('change', function () { evalOn = evalSwitch.checked; render(); });
    resultEl.addEventListener('click', function (event) {
      if (!evalOn || event.target.closest('a')) return;
      const mark = event.target.closest('mark.ev-val');
      if (mark) {
        const i = Number(mark.dataset.v);
        if (owner[i] >= 0) selectClaim(owner[i], i); else selectLine(values[i].x, i);
        return;
      }
      const claim = event.target.closest('[data-claim]');
      if (claim) { selectClaim(Number(claim.dataset.claim)); return; }
      const line = event.target.closest('[data-ex]');
      if (line) selectLine(Number(line.dataset.ex));
    });
    sideSelected.addEventListener('click', function (event) { if (event.target.closest('.ev-clear')) clearSelection(); });

    return { show: show };
  }

  window.ThermoMLAnswerEval = { create: create };
})();
