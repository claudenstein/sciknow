// sciknow web reader — extracted from web/app.py inline <script>.
// Bootstrap state lives at window.SCIKNOW_BOOTSTRAP, set by the
// HTML template before this file loads. See _render_book() in app.py.
// v2 Phase E (8710b7d follow-up) — JS extraction.

// Phase 33 — page-load tag visible in DevTools Console. The build tag
// is the git short hash (or a UTC timestamp if git isn't available).
// If you see an old hash in the console after a deploy, your browser
// is running stale JS — hard-refresh with Ctrl+Shift+R (macOS: Cmd+Shift+R).
console.log('[sciknow] web reader loaded · build ' + window.SCIKNOW_BOOTSTRAP.buildTag);

// ── State ─────────────────────────────────────────────────────────────────
let currentDraftId = window.SCIKNOW_BOOTSTRAP.activeId;
let currentChapterId = window.SCIKNOW_BOOTSTRAP.activeChapterId;
let currentSectionType = window.SCIKNOW_BOOTSTRAP.activeSectionType;
let currentJobId = null;
let currentEventSource = null;
// Phase 32.4 — needs to be `let` (was `const`) so deleteSection /
// addSectionToChapter can refresh the in-memory cache after a PUT.
let chaptersData = window.SCIKNOW_BOOTSTRAP.chaptersData;

// ── Phase 42: data-action click dispatcher ───────────────────────────────
// Replaces ~20 inline onclick handlers that interpolated variables via
// Python f-strings with HTML-entity escaping (&quot;). That pattern was
// flagged by the Phase 22 audit: mitigated for XSS by _esc(), but
// fragile — the next maintainer could easily forget escaping. It also
// stops us from adopting a strict script-src CSP.
//
// New pattern: every such button carries `data-action="kebab-name"` plus
// one or two `data-*` attrs for its arguments. The single listener below
// looks up the handler in ACTIONS and invokes it with the element (which
// owns its dataset) plus the raw event. Browsers escape data-* values
// automatically, so the injection vector goes away by construction.
//
// Static handlers like `onclick="openPlanModal()"` are left alone — no
// interpolation = no fragility. A future CSP pass can convert them too.
const ACTIONS = {
  // Cluster 1 — Python-rendered sidebar + comments. Element carries
  // data-chapter-id / data-sec-type / data-draft-id / data-comment-id
  // so the handler doesn't need any interpolation.
  'preview-empty-section': (el) =>
    previewEmptySection(el.dataset.chapterId, el.dataset.secType),
  'start-writing-chapter': (el) =>
    startWritingChapter(el.dataset.chapterId),
  'adopt-orphan-section': (el, e) => {
    // Adopt/delete buttons sit inside an anchor that navigates to the
    // draft; stop the click from bubbling so the page doesn't change.
    e.preventDefault(); e.stopPropagation();
    adoptOrphanSection(el.dataset.chapterId, el.dataset.secType);
  },
  'delete-orphan-draft': (el, e) => {
    e.preventDefault(); e.stopPropagation();
    deleteOrphanDraft(el.dataset.draftId);
  },
  // Phase 54.6.x — sidebar section delete moved off inline onclick so
  // slugs that contain apostrophes ("the_sun's_magnetic_shield") or
  // colons ("beyond_sunspots:_the_solar_dynamo") stop breaking the
  // attribute string. Both autowrite-generated sections and any
  // manually-typed title with punctuation are affected.
  'delete-section': (el, e) => {
    e.preventDefault(); e.stopPropagation();
    deleteSection(el.dataset.chapterId, el.dataset.secType);
  },
  // Phase 54.6.x — corkboard cell click. Same apostrophe-in-slug bug
  // as the sidebar — switched to data-action dispatch.
  'cork-open-card': (el) => {
    if (el.dataset.draftId) {
      loadSection(el.dataset.draftId);
    } else {
      writeForCell(el.dataset.chapterId, el.dataset.secType);
    }
  },
  'resolve-comment': (el) => resolveComment(el.dataset.commentId),

  // Cluster 2 — dashboard heatmap + gaps. The chapter-modal / load-
  // section opens happen directly from the cells. `preview-empty-
  // section` is already registered above and works for both the
  // sidebar and the heatmap callers.
  'open-chapter-modal': (el) => openChapterModal(el.dataset.chapterId),
  'load-section': (el) => loadSection(el.dataset.draftId),
  'write-for-gap': (el) => writeForGap(parseInt(el.dataset.chapterNum, 10)),
  'expand-single-gap': (el) => expandSingleGap(el.dataset.gapDesc),

  // Cluster 3 — sidebar + section editor. setSectionType wants the
  // clicked element as its second arg so it can toggle `.active` on
  // the right chip; the delegator already passes `el`.
  'set-section-type': (el) => setSectionType(el.dataset.secType, el),
  'add-section-to-chapter': (el) => addSectionToChapter(el.dataset.chapterId),
  'move-section': (el) => moveSection(
    parseInt(el.dataset.sectionIndex, 10),
    parseInt(el.dataset.delta, 10),
  ),
  'remove-section': (el) => removeSection(parseInt(el.dataset.sectionIndex, 10)),

  // Cluster 4 — wiki browser + catalog pagination.
  'open-wiki-page': (el) => openWikiPage(el.dataset.slug),
  'load-wiki-pages': (el) => loadWikiPages(parseInt(el.dataset.page, 10)),
  'ask-about-paper': (el) => askAboutPaper(el.dataset.paperTitle),
  'load-catalog': (el) => loadCatalog(parseInt(el.dataset.page, 10)),

  // Phase 54.6.49 — expand-by-author disambiguation row toggle.
  'eap-toggle-author': (el) => eapToggleAuthor(el.dataset.sid),

  // Cluster 5 — version history + snapshot bundle/draft restore.
  'select-version': (el) => selectVersion(el.dataset.versionId),
  // Phase 54.6.309 — pin a version as the one the reader displays.
  'activate-version': (el) => activateVersion(el.dataset.draftId),
  'restore-bundle': (el) => restoreBundle(el.dataset.snapshotId, el.dataset.scope),
  'diff-snapshot': (el) => diffSnapshot(el.dataset.snapshotId),
  'restore-snapshot': (el) => restoreSnapshot(el.dataset.snapshotId),
};

document.addEventListener('click', function(e) {
  const el = e.target.closest('[data-action]');
  if (!el) return;
  const fn = ACTIONS[el.dataset.action];
  if (typeof fn !== 'function') return;
  // Event handlers may return false (legacy form), throw, or not —
  // mirror the inline-onclick semantics so call sites don't change.
  try {
    const ret = fn(el, e);
    if (ret === false) { e.preventDefault(); e.stopPropagation(); }
  } catch (exc) {
    console.error('[sciknow] data-action "' + el.dataset.action + '" failed:', exc);
  }
});

// ── Theme ─────────────────────────────────────────────────────────────────
function toggleTheme() {
  const html = document.documentElement;
  html.dataset.theme = html.dataset.theme === 'dark' ? 'light' : 'dark';
  localStorage.setItem('theme', html.dataset.theme);
  updateThemeButton();
}
function updateThemeButton() {
  const isDark = document.documentElement.dataset.theme === 'dark';
  const use = document.getElementById('theme-icon-use');
  if (use) use.setAttribute('href', isDark ? '#i-moon' : '#i-sun');
  document.getElementById('theme-label').textContent = isDark ? 'Dark' : 'Light';
}
if (localStorage.getItem('theme')) {
  document.documentElement.dataset.theme = localStorage.getItem('theme');
}
updateThemeButton();

// ── SPA Navigation ────────────────────────────────────────────────────────
function navTo(el) {
  const draftId = el.dataset.draftId;
  if (!draftId) return true;  // fallback to normal navigation
  loadSection(draftId);
  return false;  // prevent default <a> navigation
}

// Phase 14.2 — empty-state UX: select a chapter without needing a draft.
function selectChapter(chGroupEl) {
  if (!chGroupEl) return;
  const chId = chGroupEl.dataset.chId;
  if (!chId) return;
  currentChapterId = chId;
  currentDraftId = '';
  currentSectionType = '';
  // Visual highlight
  document.querySelectorAll('.ch-group').forEach(g => g.classList.remove('selected'));
  chGroupEl.classList.add('selected');
  document.querySelectorAll('.sec-link.active').forEach(l => l.classList.remove('active'));
  // Look up the chapter title for the breadcrumb
  const chTitleEl = chGroupEl.querySelector('.ch-title');
  const chLabel = chTitleEl ? chTitleEl.textContent.replace(/\s*\u2717\s*$/, '').trim() : 'Chapter';
  // Show the empty state in the main area
  showChapterEmptyState(chLabel, chId);
}

function showChapterEmptyState(chLabel, chId) {
  document.getElementById('draft-title').textContent = chLabel;
  const subtitle = document.getElementById('draft-subtitle');
  subtitle.innerHTML = '<span class="u-muted">No drafts yet &mdash; pick a section type and click Write, or use Autowrite to draft all sections.</span>';
  subtitle.style.display = 'block';
  // Show toolbar
  // Hide other panels
  document.getElementById('dashboard-view').style.display = 'none';
  document.getElementById('edit-view').style.display = 'none';
  document.getElementById('version-panel').style.display = 'none';
  document.getElementById('stream-panel').style.display = 'none';
  document.getElementById('scores-panel').classList.remove('open');

  // Phase 14.3 — look up the chapter's saved scope (description + topic_query)
  const ch = chaptersData.find(c => c.id === chId);
  const desc = (ch && ch.description) ? ch.description : '';
  const tq = (ch && ch.topic_query) ? ch.topic_query : '';

  // Phase 14.4 — use the chapter's actual book-style section template,
  // not a hardcoded paper-style list. Falls back to book defaults if the
  // chapter has no template (which should never happen post-14.4 since
  // the backend always returns at least _DEFAULT_BOOK_SECTIONS).
  const sections = (ch && Array.isArray(ch.sections_template) && ch.sections_template.length)
    ? ch.sections_template
    : ['overview', 'key_evidence', 'current_understanding', 'open_questions', 'summary'];
  let html = '<div class="empty-state">';
  html += '<h3>Start writing this chapter</h3>';

  // Phase 14.3 — show the chapter scope right here in the empty state
  html += '<div class="ch-scope">';
  html += '<div class="ch-scope-row"><span class="ch-scope-label">Scope</span><div class="ch-scope-val">' +
          (desc ? desc.replace(/</g, '&lt;') : '<em class="u-faint">No description set. Click Edit chapter scope to add one.</em>') +
          '</div></div>';
  html += '<div class="ch-scope-row"><span class="ch-scope-label">Topic query</span><div class="ch-scope-val">' +
          (tq ? '<code>' + tq.replace(/</g, '&lt;') + '</code>' : '<em class="u-faint">Not set &mdash; the chapter title will be used as the retrieval query.</em>') +
          '</div></div>';
  // Phase 42 — data-action dispatch (see ACTIONS registry).
  html += '<button class="btn-secondary u-mt-2 u-small" data-action="open-chapter-modal" data-chapter-id="' + chId + '" title="Open the Chapter modal to edit this chapter\'s title, scope, topic query and sections.">&#9881; Edit chapter scope</button>';
  html += '</div>';

  html += '<p>Once the scope feels right, choose a section type below and click <strong>Write</strong> in the toolbar, or click <strong>Autowrite</strong> to draft a section autonomously with the convergence loop.</p>';
  html += '<div class="empty-section-picker">';
  sections.forEach(s => {
    const active = s === currentSectionType ? ' active' : '';
    html += '<button class="section-chip' + active + '" data-action="set-section-type" data-sec-type="' + s + '" title="Switch the active draft to this section type. Each section gets its own draft + section-specific writer prompt.">' + s.replace(/_/g, ' ') + '</button>';
  });
  html += '</div>';
  html += '<p class="u-mt-4 u-small u-muted">Tip: you can also explore the corpus without writing anything &mdash; use <strong>Ask Corpus</strong>, <strong>Wiki Query</strong>, or <strong>Browse Papers</strong>. The book&#39;s overall <strong>&#128221; Plan</strong> is also editable from the toolbar.</p>';
  html += '</div>';
  document.getElementById('read-view').innerHTML = html;
  document.getElementById('read-view').style.display = 'block';
}

function setSectionType(t, btn) {
  currentSectionType = t;
  document.querySelectorAll('.section-chip').forEach(c => c.classList.remove('active'));
  if (btn) btn.classList.add('active');
}

function startWritingChapter(chId) {
  // Find the chapter group and select it, then prompt to write the first section.
  const grp = document.querySelector('[data-ch-id="' + chId + '"]');
  if (grp) selectChapter(grp);
  if (!currentSectionType) currentSectionType = 'overview';
  // Auto-trigger write
  doWrite();
}

// Phase 14.2 — show an inline guidance toast instead of a JS alert.
// Replaces the silent-failing alert() calls in doWrite/doReview/etc.
function showEmptyHint(html) {
  let hint = document.getElementById('empty-hint');
  if (!hint) {
    hint = document.createElement('div');
    hint.id = 'empty-hint';
    hint.className = 'empty-hint';
    document.body.appendChild(hint);
  }
  hint.innerHTML = html + '<button onclick="document.getElementById(&#39;empty-hint&#39;).remove()" style="margin-left:12px;background:transparent;border:1px solid var(--border);color:var(--fg);padding:2px 8px;border-radius:4px;cursor:pointer;font-size:11px;" title="Hide this hint. It auto-dismisses after 6 seconds.">Dismiss</button>';
  // Auto-dismiss after 6s
  if (hint._timer) clearTimeout(hint._timer);
  hint._timer = setTimeout(() => {
    if (hint && hint.parentElement) hint.remove();
  }, 6000);
}

async function loadSection(draftId) {
  try {
    // Phase 15.6 — clear any in-progress live preview when navigating
    // to a different section. The saved draft we're about to load will
    // replace the read-view content anyway.
    clearLiveWrite();
    const res = await fetch('/api/section/' + draftId);
    if (!res.ok) return;
    const data = await res.json();

    currentDraftId = data.id;
    currentChapterId = data.chapter_id || '';
    currentSectionType = data.section_type || '';

    // Restore section view (if coming from dashboard)
    document.getElementById('dashboard-view').style.display = 'none';
    document.getElementById('read-view').style.display = 'block';
    document.getElementById('draft-subtitle').style.display = 'block';
    document.getElementById('argue-map-view').style.display = 'none';

    // Update main content
    // Phase 27 — prefer the display_title computed from the chapter
    // sections meta so a renamed section shows the new title in the
    // center h1 instead of the stale slug-based drafts.title snapshot.
    document.getElementById('draft-title').textContent = data.display_title || data.title;
    document.getElementById('draft-version').textContent = data.version;
    document.getElementById('draft-words').textContent = data.word_count;
    // When the draft has no body (autowrite stub, freshly-deleted content,
    // or a newly-created version that hasn't been written yet), show a
    // visible placeholder so the user doesn't read the empty panel as
    // "the text isn't shown" — the previous innerHTML='' silently blanked
    // the panel and was the top soak-window UX complaint.
    const _hasContent = (data.content_html || '').trim().length > 0;
    if (_hasContent) {
      document.getElementById('read-view').innerHTML = data.content_html;
    } else {
      const _ver = data.version || 1;
      document.getElementById('read-view').innerHTML =
        '<div class="empty-state"><h3>This draft is empty</h3>' +
        '<p class="u-muted">Version ' + _ver + ' has no body yet — ' +
        'click <strong>Write</strong> in the toolbar to draft a single shot, ' +
        'or <strong>Autowrite</strong> to run the convergence loop.</p></div>';
    }
    // Phase 54.6.87 — render math in the loaded draft content so $...$
    // doesn't show up as literal dollar signs.
    _renderMathInEl(document.getElementById('read-view'));
    document.getElementById('edit-view').style.display = 'none';

    // Phase 22 — word target progress bar in the subtitle. Hidden when
    // the section has no target set.
    updateWordTargetBar(data.word_count, data.target_words);

    // Update right panel
    document.getElementById('panel-sources').innerHTML = data.sources_html;
    document.getElementById('panel-review').innerHTML = data.review_html;
    document.getElementById('panel-comments').innerHTML = data.comments_html;
    document.getElementById('comment-draft-id').value = data.id;

    // Update sidebar active state
    document.querySelectorAll('.sec-link').forEach(l => l.classList.remove('active'));
    const link = document.querySelector('[data-draft-id="' + draftId + '"]');
    if (link) link.classList.add('active');

    // Update URL without reload
    history.pushState({draftId: draftId}, '', '/section/' + draftId);

    // Hide panels
    document.getElementById('stream-panel').style.display = 'none';
    document.getElementById('version-panel').style.display = 'none';

    // Build citation popovers + update status selector
    setTimeout(buildPopovers, 100);
    if (data.status) document.getElementById('status-select').value = data.status;
  } catch(e) {
    console.error('Navigation failed:', e);
  }
}

// Handle browser back/forward
window.addEventListener('popstate', function(e) {
  if (e.state && e.state.draftId) loadSection(e.state.draftId);
});

// ── Comments ──────────────────────────────────────────────────────────────
function resolveComment(cid) {
  fetch('/comment/' + cid + '/resolve', {method: 'POST'})
    .then(() => loadSection(currentDraftId));
}

// ── Streaming helpers ─────────────────────────────────────────────────────
function showStreamPanel(label) {
  const panel = document.getElementById('stream-panel');
  const body = document.getElementById('stream-body');
  const status = document.getElementById('stream-status');
  const scores = document.getElementById('stream-scores');
  panel.style.display = 'block';
  body.innerHTML = '';
  scores.style.display = 'none';
  scores.innerHTML = '';
  status.textContent = label;
  document.getElementById('job-indicator').style.display = 'block';
  body.scrollTop = 0;
}

function hideStreamPanel() {
  document.getElementById('job-indicator').style.display = 'none';
}

function startStream(jobId) {
  currentJobId = jobId;
  if (currentEventSource) currentEventSource.close();

  const source = new EventSource('/api/stream/' + jobId);
  currentEventSource = source;
  const body = document.getElementById('stream-body');
  const status = document.getElementById('stream-status');
  const scoresEl = document.getElementById('stream-scores');

  // Phase 15 — live stats footer for the main stream panel
  const stats = createStreamStats('main-stream-stats', 'qwen3.5:27b');
  stats.start();

  source.onmessage = function(e) {
    const evt = JSON.parse(e.data);

    if (evt.type === 'token') {
      setStreamCursor(body, false);
      body.innerHTML += evt.text.replace(/</g, '&lt;').replace(/>/g, '&gt;');
      setStreamCursor(body, true);
      body.scrollTop = body.scrollHeight;
      stats.update(evt.text);
      // Phase 15.6 — also stream into the main read-view live preview
      // for `book write` (not just autowrite). The token has no `phase`
      // tag here because plain `book write` doesn't have multi-phase
      // streaming, so all tokens are writer tokens by definition.
      appendLiveWrite(evt.text);
    }
    else if (evt.type === 'progress') {
      status.textContent = evt.detail || evt.stage;
    }
    else if (evt.type === 'scores') {
      scoresEl.style.display = 'block';
      const s = evt.scores;
      const dims = ['groundedness', 'completeness', 'coherence', 'citation_accuracy', 'hedging_fidelity', 'overall'];
      scoresEl.innerHTML = 'Iteration ' + evt.iteration + ': ' + dims.map(d => {
        const v = (s[d] || 0).toFixed(2);
        const cls = v >= 0.85 ? 'good' : v >= 0.7 ? 'mid' : 'low';
        return '<span class="score-bar"><span class="label">' + d.slice(0,6) + '</span> ' +
               '<span class="value ' + cls + '">' + v + '</span></span>';
      }).join('');
    }
    else if (evt.type === 'cove_verification') {
      // Phase 11 — Chain-of-Verification (decoupled fact check). Only fires
      // when standard groundedness or hedging_fidelity is below threshold.
      const cd = evt.data || {};
      const score = (cd.cove_score != null) ? cd.cove_score.toFixed(2) : '?';
      const mismatches = cd.mismatches || [];
      const high = mismatches.filter(m => m.severity === 'high');
      const med = mismatches.filter(m => m.severity === 'medium');
      let html = '<div style="margin:10px 0;padding:8px;border-left:3px solid var(--warning);background:var(--toolbar-bg);">';
      html += '<div class="u-bold">Chain-of-Verification: ' + score + '</div>';
      if (high.length || med.length) {
        html += '<div style="font-size:12px;opacity:0.8;">' +
                '<span class="u-danger">' + high.length + ' NOT_IN_SOURCES</span> · ' +
                '<span class="u-warning">' + med.length + ' DIFFERENT_SCOPE</span></div>';
      }
      html += '</div>';
      body.innerHTML += html;
    }
    else if (evt.type === 'revision_verdict') {
      const icon = evt.action === 'KEEP' ? '\u2713' : '\u2717';
      const color = evt.action === 'KEEP' ? 'var(--success)' : 'var(--danger)';
      body.innerHTML += '<div style="color:' + color + ';font-weight:bold;margin:8px 0;">' +
        icon + ' ' + evt.action + ': ' + evt.old_score.toFixed(2) + ' \u2192 ' + evt.new_score.toFixed(2) +
        '</div>';
    }
    else if (evt.type === 'converged') {
      status.textContent = 'Converged at iteration ' + evt.iteration +
        ' (score: ' + evt.final_score.toFixed(2) + ')';
    }
    else if (evt.type === 'iteration_start') {
      body.innerHTML += '<div style="opacity:0.5;margin:12px 0;border-top:1px solid var(--border);padding-top:8px;">' +
        'Iteration ' + evt.iteration + '/' + evt.max + '</div>';
    }
    else if (evt.type === 'model_info') {
      // Phase 15.5 — model name now shown only in the stats footer (which
      // already pulls it via setModel below). The previous in-body line
      // duplicated the same info redundantly.
      stats.setModel(evt.writer_model || 'qwen3.5:27b');
    }
    else if (evt.type === 'checkpoint') {
      // Phase 15.1 — incremental save reached. Briefly note in the body.
      body.innerHTML += '<div class="u-tiny u-success u-py-1">' +
        '\u2693 checkpoint saved · ' + (evt.stage || '') + ' · ' +
        (evt.word_count || 0) + ' words</div>';
    }
    else if (evt.type === 'completed') {
      status.textContent = 'Done';
      stats.done('done');
      setStreamCursor(body, false);
      hideStreamPanel();
      source.close();
      currentEventSource = null;
      currentJobId = null;
      // Refresh sidebar and current section
      refreshAfterJob(evt.draft_id);
    }
    else if (evt.type === 'error') {
      status.textContent = 'Error: ' + evt.message;
      body.innerHTML += '<div class="u-danger u-my-2">' + evt.message + '</div>';
      stats.done('error');
      setStreamCursor(body, false);
      hideStreamPanel();
      source.close();
      currentEventSource = null;
      currentJobId = null;
    }
    else if (evt.type === 'retrieval_density_adjust') {
      // Phase 54.6.151 — surface the 54.6.150 widener in the log panel
      // so users see the adjustment happen live instead of only in the
      // backend log. Shown in the accent colour to distinguish it from
      // pure progress events.
      const delta = (evt.new_target || 0) - (evt.base_target || 0);
      const deltaStr = (delta >= 0 ? '+' : '') + delta.toLocaleString();
      body.innerHTML += '<div class="u-tiny u-accent u-py-1" '
        + 'title="' + _escHtml(evt.explanation || '') + '">'
        + '&#9876; density widener: ' + (evt.n_chunks || 0) + ' chunks · '
        + 'target ' + (evt.base_target || 0).toLocaleString() + ' &rarr; '
        + (evt.new_target || 0).toLocaleString() + ' words (' + deltaStr + ')'
        + '</div>';
    }
    else if (evt.type === 'section_length_warning') {
      // Phase 54.6.151 — Guideline 3 (Delgado 2018 digital-section
      // ceiling). Soft warning, non-blocking: autowrite still runs at
      // the requested target, but the user sees that the section is
      // above the comfort band.
      body.innerHTML += '<div class="u-tiny u-warning u-py-1" '
        + 'title="' + _escHtml(evt.explanation || '') + '">'
        + '&#9888;&#65039; section length: '
        + (evt.target || 0).toLocaleString() + ' words exceeds the '
        + (evt.soft_ceiling || 3000).toLocaleString() + '-word digital-comfort ceiling '
        + '(RESEARCH.md &sect;24 guideline 3 — Delgado 2018). '
        + 'Consider splitting the section.'
        + '</div>';
    }
    else if (evt.type === 'done') {
      stats.done('done');
      setStreamCursor(body, false);
      hideStreamPanel();
      source.close();
      currentEventSource = null;
      currentJobId = null;
    }
  };

  source.onerror = function() {
    status.textContent = 'Connection lost';
    stats.done('error');
    setStreamCursor(body, false);
    hideStreamPanel();
    source.close();
    currentEventSource = null;
    currentJobId = null;
  };
}

function stopJob() {
  if (currentJobId) {
    fetch('/api/jobs/' + currentJobId, {method: 'DELETE'});
    document.getElementById('stream-status').textContent = 'Stopping...';
  }
  // Phase 30 — also notify the global task bar so the user gets
  // immediate visual feedback even when the inner stop button is
  // pressed instead of the global one.
  stopGlobalJob();
}

// ── Phase 30 / 32.5: persistent global task bar ───────────────────────
//
// Phase 30 first design used a SECOND EventSource on /api/stream/{id}
// alongside the per-section preview consumer. Both ended up calling
// queue.get() on the same server-side asyncio.Queue, which REMOVES
// items, so the two consumers split the event stream and the task
// bar saw a tiny (often zero) subset of tokens. Patching the task
// bar's parser/regex/math couldn't fix it — the events weren't
// arriving in the first place.
//
// Phase 32.5 fix: stop using SSE for the task bar. The server now
// tracks token count, rolling tps, elapsed, model name, and stream
// state per job in `_jobs[id]` (server side, plain Python). The task
// bar polls GET /api/jobs/{id}/stats every 500ms and reads a
// fixed-shape snapshot. Single source of truth, no race, no parsing.

let _globalJob = null;
let _globalJobTimer = null;

function _formatElapsed(ms) {
  const s = Math.floor(ms / 1000);
  if (s < 60) return s + 's';
  const m = Math.floor(s / 60);
  if (m < 60) return m + 'm ' + (s % 60).toString().padStart(2, '0') + 's';
  // Phase 30 — beyond 60 minutes show hours + minutes (per user request)
  const h = Math.floor(m / 60);
  return h + 'h ' + (m % 60).toString().padStart(2, '0') + 'm';
}

function _renderTaskBar() {
  const j = _globalJob;
  if (!j) return;
  const bar = document.getElementById('task-bar');
  if (!bar) return;
  bar.style.display = 'flex';
  // Phase 32.3 — body class adds padding-top so the fixed-position
  // task bar doesn't cover the sidebar/main top edge.
  document.body.classList.add('task-bar-open');
  document.getElementById('tb-task').textContent = j.taskDesc || j.type || 'Working';
  document.getElementById('tb-task').title = j.taskDesc || j.type || '';
  document.getElementById('tb-model').textContent = j.modelName || 'qwen3.5:27b';
  document.getElementById('tb-tokens').textContent = (j.tokens || 0).toLocaleString();
  document.getElementById('tb-tps').textContent = (j.tps || 0).toFixed(1);
  document.getElementById('tb-elapsed').textContent = _formatElapsed((j.elapsedS || 0) * 1000);
  // ETA — only when target_words is known and tokens are flowing
  const etaWrap = document.getElementById('tb-eta');
  if (j.targetWords && j.tps > 0.1 && j.tokens > 0) {
    const remaining = Math.max(0, j.targetWords - j.tokens);
    const etaMs = (remaining / j.tps) * 1000;
    document.getElementById('tb-eta-val').textContent = _formatElapsed(etaMs);
    etaWrap.style.display = 'inline-flex';
  } else {
    etaWrap.style.display = 'none';
  }
  // Dot state
  const dot = document.getElementById('tb-dot');
  dot.className = 'tb-dot ' + (j.state || 'streaming');
}

// Phase 32.5 — poll the server-side stats endpoint. Replaces the
// previous (broken) SSE consumer. Called every 500ms while a job
// is active.
async function _pollGlobalJobStats(jobId) {
  if (!_globalJob || _globalJob.id !== jobId) return;
  try {
    const res = await fetch('/api/jobs/' + jobId + '/stats');
    if (res.status === 410 || res.status === 404) {
      // Job already finished and was swept by GC, or never existed.
      // Treat as a clean finish so the bar dismisses on its own.
      const j = _globalJob;
      if (j && j.state === 'streaming') {
        j.state = 'done';
        j.taskDesc = 'Done';
        _renderTaskBar();
      }
      _finishGlobalJob('done', 2000);
      return;
    }
    if (!res.ok) return;  // network blip — keep polling
    const stats = await res.json();
    const j = _globalJob;
    if (!j || j.id !== jobId) return;
    // Mirror server snapshot directly into the local state — server
    // is the source of truth, no client-side accounting.
    j.tokens = stats.tokens || 0;
    j.tps = stats.tps || 0;
    j.elapsedS = stats.elapsed_s || 0;
    if (stats.model_name) j.modelName = stats.model_name;
    if (stats.task_desc) j.taskDesc = stats.task_desc;
    if (stats.target_words) j.targetWords = stats.target_words;
    // Lifecycle: if the server says we're done/error, transition.
    if (stats.stream_state === 'done') {
      j.state = 'done';
      // Don't overwrite a server-supplied "Stopped" / final message
      if (j.taskDesc === stats.task_desc || j.taskDesc === 'Running…') {
        j.taskDesc = 'Done';
      }
      _renderTaskBar();
      _finishGlobalJob('done', 4000);
      return;
    } else if (stats.stream_state === 'error') {
      j.state = 'error';
      j.taskDesc = 'Error: ' + ((stats.error_message || 'unknown').slice(0, 80));
      _renderTaskBar();
      _finishGlobalJob('error', 0);  // wait for explicit dismiss
      return;
    }
    _renderTaskBar();
  } catch (e) {
    // Network blip — keep polling. Don't surface; the next tick will retry.
  }
}

function startGlobalJob(jobId, opts) {
  if (!jobId) return;
  // Clean up any previous job's poll timer
  if (_globalJobTimer) {
    clearInterval(_globalJobTimer);
    _globalJobTimer = null;
  }

  _globalJob = {
    id: jobId,
    type: (opts && opts.type) || 'job',
    taskDesc: (opts && opts.taskDesc) || 'Running…',
    modelName: (opts && opts.modelName) || 'qwen3.5:27b',
    targetWords: (opts && opts.targetWords) || null,
    tokens: 0,
    tps: 0,
    elapsedS: 0,
    state: 'streaming',
    sectionType: (opts && opts.sectionType) || null,
    chapterId: (opts && opts.chapterId) || null,
  };

  // Show buttons in their starting state
  document.getElementById('tb-stop').classList.remove('u-hidden');
  document.getElementById('tb-dismiss').classList.add('u-hidden');

  _renderTaskBar();

  // Phase 32.5 — kick off the poll loop. 500ms is fast enough that
  // the t/s number feels live but slow enough to be negligible HTTP
  // load (about 2 small JSON requests per second).
  _pollGlobalJobStats(jobId);  // immediate first tick
  _globalJobTimer = setInterval(() => _pollGlobalJobStats(jobId), 500);
}

function _finishGlobalJob(state, autoDismissMs) {
  if (!_globalJob) return;
  if (_globalJobTimer) {
    clearInterval(_globalJobTimer);
    _globalJobTimer = null;
  }
  _globalJob.state = state;
  _renderTaskBar();
  // Show the dismiss button instead of the stop button
  document.getElementById('tb-stop').classList.add('u-hidden');
  document.getElementById('tb-dismiss').classList.remove('u-hidden');
  // Auto-dismiss after the grace period (0 = wait for user)
  if (autoDismissMs > 0) {
    setTimeout(() => {
      if (_globalJob && _globalJob.state !== 'streaming') dismissTaskBar();
    }, autoDismissMs);
  }
}

function stopGlobalJob() {
  if (!_globalJob) return;
  // Optimistic UI: change state immediately so the user sees feedback
  _globalJob.state = 'idle';
  _globalJob.taskDesc = 'Stopping…';
  _renderTaskBar();
  fetch('/api/jobs/' + _globalJob.id, {method: 'DELETE'}).catch(() => {});
  // Phase 32.5 — the next _pollGlobalJobStats tick will see
  // stream_state === 'done' (set by _observe_event_for_stats when
  // the cancelled event flows through _run_generator_in_thread) and
  // call _finishGlobalJob. Safety net: force-dismiss after 5s in
  // case the generator never reaches its next yield.
  const jobIdAtClick = _globalJob.id;
  setTimeout(() => {
    if (_globalJob && _globalJob.id === jobIdAtClick && _globalJob.state !== 'streaming') {
      _finishGlobalJob('done', 1500);
    }
  }, 5000);
}

function dismissTaskBar() {
  const bar = document.getElementById('task-bar');
  if (bar) bar.style.display = 'none';
  // Phase 32.3 — drop the body padding so the layout returns to
  // full height when the bar isn't visible.
  document.body.classList.remove('task-bar-open');
  _globalJob = null;
  // Phase 32.5 — only the poll timer needs cleanup now; the broken
  // SSE source approach was removed.
  if (_globalJobTimer) {
    clearInterval(_globalJobTimer);
    _globalJobTimer = null;
  }
}

// ── Phase 30/31: Knowledge Graph browse modal (Graph + Table tabs) ────
async function openKgModal() {
  // Restore the user's saved palette / overrides BEFORE the first
  // render so the canvas opens in the last theme they used instead
  // of briefly flashing the default. (Chip swatches + color pickers
  // get initialised inside _initKgThemeChips on the first render.)
  _kgLoadPrefs();
  // Phase 48d — if the page was loaded with a #kg=… share URL, pre-
  // fill the filter inputs from its payload so the initial loadKg
  // fetches exactly the same slice. Theme + overrides were already
  // applied by _kgMaybeParseHashOnLoad; camera + pins get applied
  // inside the first render frame via _kgApplyPendingShare.
  const pending = window._kgPendingShare;
  if (pending && pending.f) {
    const subj = document.getElementById('kg-subject');
    const pred = document.getElementById('kg-predicate');
    const obj  = document.getElementById('kg-object');
    if (subj) subj.value = pending.f.s || '';
    if (pred) {
      // Predicate options are populated on first loadKg; adding a
      // stub option ensures the current value survives the reload.
      const v = pending.f.p || '';
      if (v && !Array.from(pred.options).some(o => o.value === v)) {
        const opt = document.createElement('option');
        opt.value = v; opt.textContent = v;
        pred.appendChild(opt);
      }
      pred.value = v;
    }
    if (obj)  obj.value  = pending.f.o || '';
  }
  openModal('kg-modal');
  switchKgTab('kg-graph');
  await loadKg(0);
}
// Phase 48d — auto-open the modal when the URL has a #kg=… share hash
// so a shared link lands the recipient directly in the graph view.
window.addEventListener('DOMContentLoaded', () => {
  if (window._kgPendingShare) {
    try { openKgModal(); } catch (e) {}
  }
  // 54.6.105 — #visuals[=equation|figure|chart|table|code] auto-opens
  // the Explore > Visuals modal with the given kind. Used by the
  // equation-render diagnostic harness; also handy as a share-link.
  const hash = location.hash || '';
  const visM = hash.match(/^#visuals(?:=(figure|chart|equation|table|code))?$/);
  if (visM) {
    setTimeout(() => {
      try {
        openVisualsModal();
        if (visM[1]) {
          setTimeout(() => {
            const mode = document.getElementById('vis-mode');
            const kind = document.getElementById('vis-kind-filter');
            if (mode && visM[1] !== 'figure' && visM[1] !== 'chart') mode.value = 'list';
            if (kind) { kind.value = visM[1]; kind.dispatchEvent(new Event('change')); }
          }, 600);
        }
      } catch (e) {}
    }, 300);
  }
});

function switchKgTab(name) {
  document.querySelectorAll('#kg-modal .tab').forEach(t => {
    t.classList.toggle('active', t.dataset.tab === name);
  });
  document.getElementById('kg-graph-pane').style.display = (name === 'kg-graph') ? 'block' : 'none';
  document.getElementById('kg-table-pane').style.display = (name === 'kg-table') ? 'block' : 'none';
}

// ── Phase 54.6.12 — Visualize modal (ECharts rewrite) ──────────────────
// One modal, six tabs, all backed by ECharts. Each tab has its own
// chart instance cached in window._vizCharts so we don't reinit on
// tab swaps — just resize the now-visible one.
window._vizCharts = window._vizCharts || {};
window._vizLoaded = window._vizLoaded || {};

function _vizChart(id) {
  // Lazy-init ECharts instance bound to a container div.
  if (window._vizCharts[id]) return window._vizCharts[id];
  const el = document.getElementById(id);
  if (!el || typeof echarts === 'undefined') return null;
  const c = echarts.init(el, null, {renderer: 'canvas'});
  window._vizCharts[id] = c;
  // Keep all charts in sync with the viewport on window resize.
  window.addEventListener('resize', () => c.resize());
  return c;
}

function openVizModal(tab) {
  // Close whichever nav dropdown was open (Phase 54.6.16 — multiple
  // possibilities: viz-dropdown, book-dropdown, etc).
  document.querySelectorAll('.nav-dropdown.open').forEach(d => d.classList.remove('open'));
  openModal('viz-modal');
  _vizLoadPrefs();
  _vizWireControls();
  if (tab) switchVizTab(tab);
  // Auto-load the active tab on first open.
  const active = document.querySelector('#viz-modal .tab.active');
  const name = (active && active.dataset.tab) || 'viz-topic';
  _vizAutoLoad(name);
}

// Generic top-bar dropdown toggle. Clicking outside (or opening another
// dropdown) closes the current one. Phase 54.6.16 — generalised from
// the legacy Visualize-only dropdown to support Book / Explore /
// Visualize / Manage.
function toggleNavDropdown(id, ev) {
  if (ev) ev.stopPropagation();
  const dd = document.getElementById(id);
  if (!dd) return;
  // Close any other open nav-dropdowns first.
  document.querySelectorAll('.nav-dropdown.open').forEach(other => {
    if (other !== dd) other.classList.remove('open');
  });
  const isOpen = dd.classList.toggle('open');
  if (isOpen) {
    const onDocClick = (e) => {
      if (!dd.contains(e.target)) {
        dd.classList.remove('open');
        document.removeEventListener('click', onDocClick);
      }
    };
    setTimeout(() => document.addEventListener('click', onDocClick), 0);
  }
}
// Back-compat shim for any external caller that still references the
// old name. Delegates to the generic toggler.
function toggleVizDropdown(ev) { toggleNavDropdown('viz-dropdown', ev); }

function _vizAutoLoad(name) {
  if (name === 'viz-topic'     && !_vizLoaded['viz-topic'])     loadTopicMap(false);
  if (name === 'viz-sunburst'  && !_vizLoaded['viz-sunburst'])  loadSunburst();
  if (name === 'viz-timeline'  && !_vizLoaded['viz-timeline'])  loadTimeline();
  if (name === 'viz-radar'     && !_vizLoaded['viz-radar'])     loadGapRadar();
  // consensus + ego are user-driven (need input), so no auto-load.
  // Always resize the visible chart to fix "0×0" initial-render bug.
  const chartIdMap = {
    'viz-topic':     'viz-topic-chart',
    'viz-sunburst':  'viz-sunburst-chart',
    'viz-consensus': 'viz-consensus-chart',
    'viz-timeline':  'viz-timeline-chart',
    'viz-ego':       'viz-ego-chart',
    'viz-radar':     'viz-radar-chart',
  };
  const cid = chartIdMap[name];
  if (cid && window._vizCharts[cid]) {
    setTimeout(() => window._vizCharts[cid].resize(), 20);
  }
}

function switchVizTab(name) {
  document.querySelectorAll('#viz-modal .tab').forEach(t => {
    t.classList.toggle('active', t.dataset.tab === name);
  });
  ['viz-topic','viz-sunburst','viz-consensus','viz-timeline','viz-ego','viz-radar']
    .forEach(n => {
      const p = document.getElementById(n + '-pane');
      if (p) p.style.display = (n === name) ? 'block' : 'none';
    });
  document.getElementById('viz-status').textContent = '';
  _vizAutoLoad(name);
}

function _vizSetStatus(msg, kind) {
  const el = document.getElementById('viz-status');
  if (!el) return;
  el.textContent = msg || '';
  el.style.color = kind === 'error' ? 'var(--danger)'
                 : kind === 'ok'    ? 'var(--success)'
                 : 'var(--fg-muted)';
}

// ── Phase 54.6.15 — shared theming for every Viz tab ───────────────────
// Reuses KG_THEMES (same palette definitions as the Knowledge Graph
// modal, defined earlier in this bundle) plus its own font / label
// scale / custom-colour overrides. Every loadX caches its base option
// in _vizBaseOption so a theme swap re-renders without re-fetching.

let _vizTheme = 'paper';
let _vizFont = 'sans-solid';
let _vizLabelScale = 1.0;
let _vizCustomColors = {};  // keys: bg, label

function _vizLoadPrefs() {
  try {
    _vizTheme = localStorage.getItem('sciknow.viz.theme') || 'paper';
    _vizFont = localStorage.getItem('sciknow.viz.font') || 'sans-solid';
    const s = parseFloat(localStorage.getItem('sciknow.viz.labelScale') || '1.0');
    if (!isNaN(s)) _vizLabelScale = s;
    const raw = localStorage.getItem('sciknow.viz.custom');
    if (raw) _vizCustomColors = JSON.parse(raw) || {};
  } catch (_) {}
}
function _vizSavePrefs() {
  try {
    localStorage.setItem('sciknow.viz.theme', _vizTheme);
    localStorage.setItem('sciknow.viz.font', _vizFont);
    localStorage.setItem('sciknow.viz.labelScale', String(_vizLabelScale));
    localStorage.setItem('sciknow.viz.custom', JSON.stringify(_vizCustomColors));
  } catch (_) {}
}

function _vizFontFamily() {
  const map = {
    'sans-halo':      'var(--font-sans)',
    'sans-solid':     'var(--font-sans)',
    'serif-solid':    'var(--font-serif)',
    'serif-halo':     'var(--font-serif)',
    'mono-solid':     'var(--font-mono)',
    'mono-halo':      'var(--font-mono)',
    'condensed-solid':'"Barlow Condensed","Arial Narrow",sans-serif',
    'display-solid':  '"Playfair Display",Georgia,serif',
  };
  return map[_vizFont] || map['sans-solid'];
}

function _vizActivePalette() {
  const base = (typeof KG_THEMES !== 'undefined' ? KG_THEMES[_vizTheme] : null)
             || {canvasBg:'#f3f5f9', label:'#0a1a33', edge:'#5a7bb0', nodeMid:'#6c8ec8'};
  return {
    bg:    _vizCustomColors.bg    || base.canvasBg,
    label: _vizCustomColors.label || base.label,
    edge:  base.edge,
    node:  base.nodeMid,
    hi:    base.hiMid || '#f59e0b',
  };
}

function _vizDecorateOption(option) {
  const p = _vizActivePalette();
  const font = _vizFontFamily();
  option = option || {};
  option.backgroundColor = p.bg;
  option.textStyle = Object.assign(
    {color: p.label, fontFamily: font}, option.textStyle || {}
  );
  // Legend text
  if (option.legend) {
    const arr = Array.isArray(option.legend) ? option.legend : [option.legend];
    arr.forEach(l => {
      l.textStyle = Object.assign({color: p.label, fontFamily: font}, l.textStyle || {});
    });
  }
  // Axis line + labels
  ['xAxis', 'yAxis'].forEach(ax => {
    if (!option[ax]) return;
    const arr = Array.isArray(option[ax]) ? option[ax] : [option[ax]];
    arr.forEach(a => {
      a.axisLine   = Object.assign({lineStyle: {color: p.edge}}, a.axisLine || {});
      a.axisLabel  = Object.assign({color: p.label, fontFamily: font}, a.axisLabel || {});
      a.nameTextStyle = Object.assign({color: p.label, fontFamily: font}, a.nameTextStyle || {});
      a.splitLine  = Object.assign({lineStyle: {color: p.edge, opacity: 0.2}}, a.splitLine || {});
    });
  });
  // Radar indicator axes
  if (option.radar) {
    option.radar.axisName = Object.assign(
      {color: p.label, fontFamily: font}, option.radar.axisName || {}
    );
    option.radar.splitLine = Object.assign({lineStyle:{color: p.edge, opacity: 0.25}}, option.radar.splitLine || {});
    option.radar.splitArea = {areaStyle: {color: ['transparent','transparent']}};
  }
  // Series-level label font / size
  (option.series || []).forEach(ser => {
    if (ser.label && typeof ser.label === 'object') {
      ser.label.fontFamily = font;
      if (typeof ser.label.fontSize === 'number') {
        ser.label.fontSize = Math.max(8, Math.round(ser.label.fontSize * _vizLabelScale));
      } else {
        ser.label.fontSize = Math.max(8, Math.round(11 * _vizLabelScale));
      }
      ser.label.color = ser.label.color || p.label;
    }
    if (ser.type === 'sunburst' && ser.label) {
      // Sunburst reads label.color per level; give each level a legible text colour.
      if (ser.levels) {
        ser.levels.forEach(lvl => {
          if (lvl && lvl.label) {
            lvl.label.fontFamily = font;
            if (typeof lvl.label.fontSize === 'number') {
              lvl.label.fontSize = Math.max(8, Math.round(lvl.label.fontSize * _vizLabelScale));
            }
          }
        });
      }
    }
  });
  return option;
}

window._vizBaseOption = window._vizBaseOption || {};

function _vizRender(chartId, baseOption) {
  const chart = _vizChart(chartId);
  if (!chart) return null;
  // Deep-copy so subsequent theme swaps don't share mutated refs.
  const copy = JSON.parse(JSON.stringify(baseOption));
  window._vizBaseOption[chartId] = copy;
  chart.setOption(_vizDecorateOption(JSON.parse(JSON.stringify(copy))), true);
  return chart;
}

function _vizReapplyTheme() {
  Object.keys(window._vizBaseOption).forEach(id => {
    const chart = _vizChart(id);
    if (!chart) return;
    const base = window._vizBaseOption[id];
    chart.setOption(_vizDecorateOption(JSON.parse(JSON.stringify(base))), true);
  });
  // Theme-chip active state
  document.querySelectorAll('#viz-modal .viz-theme-chip').forEach(c => {
    c.classList.toggle('active', c.dataset.theme === _vizTheme);
  });
  // Font select
  const fs = document.getElementById('viz-font-select');
  if (fs) fs.value = _vizFont;
  // Label slider
  const ls = document.getElementById('viz-labelscale');
  if (ls) ls.value = String(_vizLabelScale);
}

function vizSetTheme(name) {
  if (typeof KG_THEMES !== 'undefined' && !KG_THEMES[name]) return;
  _vizTheme = name;
  _vizSavePrefs();
  _vizReapplyTheme();
}
function vizInvertTheme() {
  if (typeof KG_THEMES === 'undefined') return;
  const inv = (KG_THEMES[_vizTheme] || {}).inverse;
  if (inv) vizSetTheme(inv);
}
function vizSetFont(key) {
  _vizFont = key;
  _vizSavePrefs();
  _vizReapplyTheme();
}
function vizSetLabelScale(v) {
  _vizLabelScale = Math.max(0.6, Math.min(2.0, parseFloat(v)));
  _vizSavePrefs();
  _vizReapplyTheme();
}
function vizSetCustomColor(kind, value) {
  _vizCustomColors[kind] = value;
  _vizSavePrefs();
  _vizReapplyTheme();
}
function vizClearCustomColors() {
  _vizCustomColors = {};
  _vizSavePrefs();
  _vizReapplyTheme();
}
function vizToggleFullscreen() {
  const modal = document.querySelector('#viz-modal .modal');
  if (!modal) return;
  if (document.fullscreenElement) {
    document.exitFullscreen();
  } else if (modal.requestFullscreen) {
    modal.requestFullscreen();
  }
  // ECharts instances must resize after fullscreen transition.
  setTimeout(() => Object.values(window._vizCharts || {}).forEach(c => {
    try { c.resize(); } catch (_) {}
  }), 150);
}
function vizDownloadPng() {
  const active = document.querySelector('#viz-modal .tab.active');
  const tab = active ? active.dataset.tab : 'viz-topic';
  const idMap = {
    'viz-topic':'viz-topic-chart', 'viz-sunburst':'viz-sunburst-chart',
    'viz-consensus':'viz-consensus-chart', 'viz-timeline':'viz-timeline-chart',
    'viz-ego':'viz-ego-chart', 'viz-radar':'viz-radar-chart',
  };
  const chart = window._vizCharts && window._vizCharts[idMap[tab]];
  if (!chart) { alert('Nothing to download for this tab yet.'); return; }
  const url = chart.getDataURL({type: 'png', pixelRatio: 2, backgroundColor: _vizActivePalette().bg});
  const a = document.createElement('a');
  a.href = url;
  a.download = 'sciknow-' + tab + '-' + new Date().toISOString().slice(0,10) + '.png';
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
}

// Wire the theme chips once, on first openVizModal.
function _vizWireControls() {
  document.querySelectorAll('#viz-modal .viz-theme-chip').forEach(chip => {
    if (chip.dataset.wired) return;
    chip.dataset.wired = '1';
    chip.addEventListener('click', () => vizSetTheme(chip.dataset.theme));
  });
  document.querySelectorAll('#viz-modal .viz-theme-chip').forEach(c => {
    c.classList.toggle('active', c.dataset.theme === _vizTheme);
  });
  const bgPicker = document.getElementById('viz-color-bg');
  if (bgPicker && _vizCustomColors.bg) bgPicker.value = _vizCustomColors.bg;
  const lblPicker = document.getElementById('viz-color-label');
  if (lblPicker && _vizCustomColors.label) lblPicker.value = _vizCustomColors.label;
}

// 1. Topic map — ECharts scatter with built-in zoom/pan + tooltips.
async function loadTopicMap(refresh) {
  _vizSetStatus('Loading topic map' + (refresh ? ' (refreshing UMAP — may take 5-60s)…' : '…'));
  try {
    const res = await fetch('/api/viz/topic-map' + (refresh ? '?refresh=true' : ''));
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    _vizLoaded['viz-topic'] = true;
    if (!data.points || !data.points.length) {
      _vizSetStatus(data.message || 'no points', 'error');
      return;
    }
    _vizSetStatus(data.n_papers + ' papers · ' + (data.clusters || []).length + ' clusters · scroll to zoom, drag to pan', 'ok');
    const clusterColor = {};
    (data.clusters || []).forEach(c => { clusterColor[c.id] = c.color; });
    // One series per cluster so the ECharts legend works for toggles.
    const byCluster = {};
    data.points.forEach(p => {
      const key = (p.cluster == null) ? 'noise' : String(p.cluster);
      (byCluster[key] = byCluster[key] || []).push(p);
    });
    const series = Object.keys(byCluster).map(key => ({
      name: 'Cluster ' + key,
      type: 'scatter',
      symbolSize: 9,
      itemStyle: {color: clusterColor[key] || '#888'},
      // Phase 54.6.13 — labels on every point. "emphasis" means
      // labels only render on hover/zoom by default; set show:true
      // to always-render if you want a wall of titles. We use the
      // first author + two-digit year so the label stays short even
      // when hundreds of points are visible.
      label: {
        show: true, position: 'right', distance: 4,
        formatter: p => {
          const a = (p.data.author || '').split(/[,;]/)[0].trim();
          const lastName = a.split(/\s+/).pop() || '';
          const yy = p.data.year ? String(p.data.year).slice(-2) : '';
          return lastName ? (lastName + (yy ? "'" + yy : '')) : '';
        },
        color: '#333', fontSize: 9, backgroundColor: 'rgba(255,255,255,0.55)',
        padding: [1, 3], borderRadius: 3,
      },
      labelLayout: {hideOverlap: true},
      emphasis: {scale: 1.4, label: {fontSize: 11, fontWeight: 'bold'}},
      data: byCluster[key].map(p => ({
        value: [p.x, p.y],
        name: p.title,
        year: p.year,
        author: p.first_author,
        document_id: p.document_id,
      })),
    }));
    const chart = _vizRender('viz-topic-chart', {
      tooltip: {
        trigger: 'item',
        confine: true, enterable: true, extraCssText:
          'max-width: 480px; white-space: normal; word-break: break-word; '
          + 'line-height: 1.45; padding: 10px 12px;',
        formatter: p => (p.data.year || '?') + ' · ' + _escHtml(p.data.author || '?')
          + '<br/><strong>' + _escHtml(p.data.name || '(untitled)') + '</strong>',
      },
      legend: {type: 'scroll', bottom: 2, textStyle: {fontSize: 10}},
      xAxis: {show: false, min: -1.1, max: 1.1, type: 'value'},
      yAxis: {show: false, min: -1.1, max: 1.1, type: 'value'},
      dataZoom: [
        {type: 'inside', xAxisIndex: 0, filterMode: 'none'},
        {type: 'inside', yAxisIndex: 0, filterMode: 'none'},
      ],
      series: series,
    });
    if (!chart) return;
    // Click → copy the document_id into the ego-radial input, handy flow.
    chart.off('click');
    chart.on('click', params => {
      if (params && params.data && params.data.document_id) {
        const inp = document.getElementById('viz-ego-docid');
        if (inp) inp.value = params.data.document_id;
      }
    });
  } catch (exc) {
    _vizSetStatus('Failed: ' + exc.message, 'error');
  }
}

// 2. RAPTOR sunburst — native ECharts sunburst with drill-in + zoom-out.
async function loadSunburst() {
  _vizSetStatus('Loading RAPTOR tree…');
  try {
    const res = await fetch('/api/viz/raptor-tree');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const tree = await res.json();
    if (tree.message) { _vizSetStatus(tree.message, 'error'); return; }
    _vizSetStatus(tree.total_nodes + ' RAPTOR nodes · click a slice to zoom, click centre to zoom out', 'ok');
    _vizLoaded['viz-sunburst'] = true;
    _vizRender('viz-sunburst-chart', {
      tooltip: {trigger: 'item',
        // Phase 54.6.16 — show the FULL summary in the tooltip
        // (previous 160-char slice hid the back half of every long
        // RAPTOR summary). enterable + confine let the user hover
        // into the tooltip and scroll, and keep it inside the viewport.
        confine: true, enterable: true, extraCssText:
          'max-width: 520px; white-space: normal; word-break: break-word; '
          + 'line-height: 1.45; max-height: 60vh; overflow-y: auto; '
          + 'padding: 10px 12px;',
        formatter: p => 'L' + (p.data.level || 0) + ' · '
          + (p.data.n_docs || 0) + ' docs'
          + (p.data.year_min || p.data.year_max
              ? ' · ' + (p.data.year_min || '?') + '–' + (p.data.year_max || '?')
              : '')
          + '<br/><br/>' + _escHtml(p.name || p.data.name || ''),
      },
      series: [{
        type: 'sunburst',
        radius: ['0', '95%'],
        nodeClick: 'rootToNode',
        sort: null,
        emphasis: {focus: 'ancestor'},
        // Phase 54.6.13 — labels on every slice. ECharts auto-rotates
        // them along the arc; we cap at 28 chars + ellipsis so deeper
        // rings stay legible.
        label: {
          show: true, rotate: 'tangential', overflow: 'truncate',
          fontSize: 11, color: '#fff', minAngle: 6,
          formatter: p => {
            const s = (p.name || '').toString();
            return s.length > 28 ? s.slice(0, 28) + '…' : s;
          },
        },
        levels: [
          {},
          {itemStyle: {borderWidth: 2},
            label: {fontSize: 12, fontWeight: 'bold'}},
          {label: {fontSize: 10}},
          {label: {fontSize: 9, color: '#222'}},
        ],
        data: (tree.children && tree.children.length) ? tree.children : [{name: 'empty', value: 1}],
      }],
    });
  } catch (exc) {
    _vizSetStatus('Failed: ' + exc.message, 'error');
  }
}

// 3. Consensus landscape — scatter by consensus_level with ECharts tooltips.
async function loadConsensusLandscape() {
  const topic = (document.getElementById('viz-consensus-topic').value || '').trim();
  if (!topic) { alert('Enter a topic first.'); return; }
  _vizSetStatus('Running wiki consensus (30s-2min)…');
  const fd = new FormData();
  fd.append('topic', topic);
  try {
    const res = await fetch('/api/viz/consensus-landscape', {method:'POST', body: fd});
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    _vizLoaded['viz-consensus'] = true;
    _vizSetStatus((data.claims || []).length + ' claim(s)', 'ok');
    const colors = {strong:'#059669', moderate:'#0284c7',
                    weak:'#f59e0b', contested:'#dc2626',
                    unknown:'#888'};
    const byLevel = {};
    (data.claims || []).forEach(c => {
      (byLevel[c.consensus_level || 'unknown']
        = byLevel[c.consensus_level || 'unknown'] || []).push(c);
    });
    const series = Object.keys(byLevel).map(level => ({
      name: level,
      type: 'scatter',
      symbolSize: d => 10 + Math.min(10, (d[2] || 1)),
      itemStyle: {color: colors[level] || '#888', opacity: 0.75},
      data: byLevel[level].map(c => ({
        value: [c.x, c.y, (c.supporting || []).length + (c.contradicting || []).length],
        name: c.claim,
        trend: c.trend,
      })),
    }));
    _vizRender('viz-consensus-chart', {
      tooltip: {trigger: 'item',
        confine: true, enterable: true, extraCssText:
          'max-width: 520px; white-space: normal; word-break: break-word; '
          + 'line-height: 1.45; max-height: 60vh; overflow-y: auto; '
          + 'padding: 10px 12px;',
        formatter: p => '<strong>' + _escHtml(p.seriesName) + '</strong>'
          + (p.data.trend ? ' · ' + _escHtml(p.data.trend) : '')
          + '<br/>' + p.data.value[0] + ' supporting · ' + p.data.value[1] + ' contradicting'
          + '<br/><br/>' + _escHtml(p.data.name || ''),
      },
      legend: {bottom: 2},
      grid: {left: 50, bottom: 50, right: 20, top: 20},
      xAxis: {type: 'value', name: 'supporting papers →', nameLocation: 'middle', nameGap: 30},
      yAxis: {type: 'value', name: 'contradicting →', nameLocation: 'middle', nameGap: 30},
      series: series,
    });
  } catch (exc) {
    _vizSetStatus('Failed: ' + exc.message, 'error');
  }
}

// 4. Timeline river — ECharts stacked line with dataZoom brush.
async function loadTimeline() {
  _vizSetStatus('Loading timeline…');
  try {
    const res = await fetch('/api/viz/timeline');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    if (!(data.years || []).length) { _vizSetStatus('no data', 'error'); return; }
    _vizLoaded['viz-timeline'] = true;
    const modeLabel = (data.mode === 'decade') ? 'decades (no clusters yet — run `catalog cluster`)' : 'clusters';
    _vizSetStatus(data.years.length + ' years · ' + data.series.length + ' ' + modeLabel + ' · drag below axis to zoom',
                  data.mode === 'decade' ? '' : 'ok');
    const series = (data.series || []).map(ser => ({
      name: 'Cluster ' + ser.cluster,
      type: 'line', stack: 'total',
      areaStyle: {opacity: 0.85},
      emphasis: {focus: 'series'},
      itemStyle: {color: ser.color},
      showSymbol: false,
      smooth: 0.15,
      data: ser.values,
    }));
    _vizRender('viz-timeline-chart', {
      tooltip: {trigger: 'axis',
        confine: true, enterable: true, extraCssText:
          'max-width: 420px; white-space: normal; word-break: break-word; '
          + 'line-height: 1.45; max-height: 60vh; overflow-y: auto; '
          + 'padding: 10px 12px;'},
      legend: {type: 'scroll', top: 0, textStyle: {fontSize: 10}},
      grid: {top: 30, left: 45, right: 20, bottom: 70},
      xAxis: {type: 'category', data: data.years, boundaryGap: false},
      yAxis: {type: 'value', name: 'papers'},
      dataZoom: [
        {type: 'inside'},
        {type: 'slider', height: 20, bottom: 25},
      ],
      series: series,
    });
  } catch (exc) {
    _vizSetStatus('Failed: ' + exc.message, 'error');
  }
}

// 5. Ego radial — polar scatter; drag to rotate, wheel to zoom.
async function loadEgoRadial() {
  let docId = (document.getElementById('viz-ego-docid').value || '').trim();
  if (!docId) { alert('Enter a document UUID or the first ~8 chars.'); return; }
  const k = parseInt(document.getElementById('viz-ego-k').value || '20', 10);
  if (docId.length < 32) {
    try {
      const rr = await fetch('/api/catalog?q=' + encodeURIComponent(docId) + '&limit=5');
      const rd = await rr.json();
      const hit = (rd.papers || rd.results || []).find(p =>
        (p.document_id || '').toLowerCase().startsWith(docId.toLowerCase())
      );
      if (hit) docId = hit.document_id;
    } catch (_) {}
  }
  _vizSetStatus('Finding neighbours…');
  try {
    const res = await fetch('/api/viz/ego-radial?document_id=' + encodeURIComponent(docId) + '&k=' + k);
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    _vizLoaded['viz-ego'] = true;
    _vizSetStatus(data.neighbours.length + ' neighbours around ' + (data.centre.title || docId).slice(0, 60), 'ok');
    // ECharts polar uses (radius, angle). Our API returned x,y so
    // reverse-derive r + θ for polar coords.
    const neighbours = (data.neighbours || []).map(n => {
      const r = Math.sqrt(n.x*n.x + n.y*n.y);
      const theta = Math.atan2(n.y, n.x) * 180 / Math.PI;
      return {r, theta, ...n};
    });
    const chart = _vizRender('viz-ego-chart', {
      tooltip: {trigger: 'item',
        confine: true, enterable: true, extraCssText:
          'max-width: 480px; white-space: normal; word-break: break-word; '
          + 'line-height: 1.45; padding: 10px 12px;',
        formatter: p => (p.data.score != null ? 'sim ' + p.data.score.toFixed(3) + ' · ' : '')
          + (p.data.year || '?') + ' · ' + _escHtml(p.data.author || '?')
          + '<br/>' + _escHtml(p.data.name || ''),
      },
      polar: {radius: '80%'},
      angleAxis: {type: 'value', startAngle: 0, min: -180, max: 180, show: false},
      radiusAxis: {type: 'value', max: 1.0, show: false},
      series: [
        // Edges as thin lines from centre to each neighbour.
        {type: 'line', coordinateSystem: 'polar', showSymbol: false,
          lineStyle: {color: '#bbb', width: 0.8},
          data: neighbours.flatMap(n => [[0, 0], [n.r, n.theta], [NaN, NaN]]),
        },
        // Centre.
        {type: 'scatter', coordinateSystem: 'polar', symbolSize: 22,
          itemStyle: {color: '#0284c7'},
          data: [{value: [0, 0], name: data.centre.title || '',
                  year: data.centre.year, author: data.centre.first_author}],
          label: {show: true, position: 'top', fontSize: 11,
                  fontWeight: 'bold', color: '#0284c7',
                  formatter: (data.centre.title || '').slice(0, 40)},
        },
        // Neighbours.
        {type: 'scatter', coordinateSystem: 'polar', symbolSize: 14,
          itemStyle: {color: '#059669', opacity: 0.8},
          data: neighbours.map(n => ({
            value: [n.r, n.theta],
            name: n.title || '',
            year: n.year, author: n.first_author, score: n.score,
            document_id: n.document_id,
          })),
        },
      ],
    });
    if (!chart) return;
    chart.off('click');
    chart.on('click', params => {
      if (params && params.seriesIndex === 2 && params.data.document_id) {
        document.getElementById('viz-ego-docid').value = params.data.document_id;
      }
    });
  } catch (exc) {
    _vizSetStatus('Failed: ' + exc.message, 'error');
  }
}

// 6. Gap radar — native ECharts radar chart.
async function loadGapRadar() {
  _vizSetStatus('Loading gap radar…');
  try {
    const res = await fetch('/api/viz/gap-radar');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    _vizLoaded['viz-radar'] = true;
    _vizSetStatus(data.chapters.length + ' chapters', 'ok');
    _vizRender('viz-radar-chart', {
      tooltip: {trigger: 'item',
        confine: true, enterable: true, extraCssText:
          'max-width: 420px; white-space: normal; word-break: break-word; '
          + 'line-height: 1.45; padding: 10px 12px;'},
      legend: {type: 'scroll', bottom: 2, textStyle: {fontSize: 10}},
      radar: {
        indicator: (data.axes || []).map(a => ({name: a, max: 1.0})),
        radius: '65%',
        splitNumber: 4,
        axisName: {fontSize: 12},
      },
      series: [{
        type: 'radar', areaStyle: {opacity: 0.12},
        emphasis: {areaStyle: {opacity: 0.35}},
        data: (data.chapters || []).map(ch => ({
          name: 'Ch.' + ch.number + ' ' + ch.title,
          value: ch.values,
        })),
      }],
    });
  } catch (exc) {
    _vizSetStatus('Failed: ' + exc.message, 'error');
  }
}

let _kgPredicatesLoaded = false;
let _kgTriples = [];
let _kgExtractSource = null;

// Phase 54.6.x — KG extraction wrapper surfaced directly in the KG
// modal (parallel to the existing Wiki → Lint button). Streams the
// same /api/wiki/extract-kg job; reloads the graph when done.
async function kgExtractFromModal() {
  const btn = document.getElementById('kg-extract-btn');
  const status = document.getElementById('kg-extract-status');
  const log = document.getElementById('kg-extract-log');
  const force = document.getElementById('kg-extract-force').checked;
  if (!confirm(
    'Extract knowledge_graph triples from the corpus?\n\n'
    + 'Runs one LLM call per paper that has no triples yet (or every '
    + 'paper if force is checked). Long-running for big corpora — '
    + 'cancel by closing the modal or stopping the job from the task '
    + 'bar.'
  )) return;
  if (btn) btn.disabled = true;
  if (log) { log.style.display = 'block'; log.textContent = ''; }
  if (status) status.textContent = 'Starting…';
  const fd = new FormData();
  fd.append('force', force);
  let res;
  try {
    res = await fetch('/api/wiki/extract-kg', {method: 'POST', body: fd});
  } catch (exc) {
    if (status) status.textContent = 'Request failed: ' + exc.message;
    if (btn) btn.disabled = false;
    return;
  }
  if (!res.ok) {
    if (status) status.textContent = 'Start failed: HTTP ' + res.status;
    if (btn) btn.disabled = false;
    return;
  }
  const data = await res.json();
  if (_kgExtractSource) _kgExtractSource.close();
  const source = new EventSource('/api/stream/' + data.job_id);
  _kgExtractSource = source;
  source.onmessage = function(e) {
    let evt;
    try { evt = JSON.parse(e.data); } catch (_) { return; }
    if (evt.type === 'log' && log) {
      log.textContent += evt.text + '\n';
      log.scrollTop = log.scrollHeight;
    } else if (evt.type === 'progress') {
      if (status) status.textContent = evt.detail || evt.stage || '';
      if (evt.detail && evt.detail.startsWith('$ ') && log) {
        log.textContent += evt.detail + '\n';
      }
    } else if (evt.type === 'completed') {
      source.close(); _kgExtractSource = null;
      if (status) status.innerHTML = '<span class="u-success">✓ Done — reloading graph…</span>';
      if (btn) btn.disabled = false;
      // Refresh the KG modal contents.
      _kgPredicatesLoaded = false;
      const sel = document.getElementById('kg-predicate');
      if (sel) {
        while (sel.options.length > 1) sel.remove(1);
      }
      try { loadKg(0); } catch (_) {}
    } else if (evt.type === 'error') {
      source.close(); _kgExtractSource = null;
      if (status) status.innerHTML = '<span class="u-danger">✗ ' + (evt.message || 'error') + '</span>';
      if (btn) btn.disabled = false;
    } else if (evt.type === 'done') {
      source.close(); _kgExtractSource = null;
      if (btn) btn.disabled = false;
    }
  };
}

async function loadKg(offset) {
  const subject = document.getElementById('kg-subject').value.trim();
  const predicate = document.getElementById('kg-predicate').value.trim();
  const obj = document.getElementById('kg-object').value.trim();
  const params = new URLSearchParams({
    subject: subject, predicate: predicate, object: obj,
    limit: 200, offset: offset || 0,
  });
  document.getElementById('kg-status').textContent = 'Loading…';
  try {
    const res = await fetch('/api/kg?' + params.toString());
    const data = await res.json();
    if (!_kgPredicatesLoaded && data.predicates && data.predicates.length) {
      const sel = document.getElementById('kg-predicate');
      data.predicates.forEach(p => {
        const opt = document.createElement('option');
        opt.value = p; opt.textContent = p;
        sel.appendChild(opt);
      });
      _kgPredicatesLoaded = true;
    }
    _kgTriples = data.triples || [];
    document.getElementById('kg-status').textContent =
      data.total + ' triple' + (data.total === 1 ? '' : 's') + ' total · ' +
      'showing ' + _kgTriples.length + ' (top 100 in graph view)';

    // Render BOTH the table and the graph from the same data so
    // switching tabs is instant.
    _renderKgTable(_kgTriples);
    _renderKgGraph(_kgTriples.slice(0, 100));
  } catch (e) {
    document.getElementById('kg-status').textContent = 'Error: ' + e.message;
  }
}

function _renderKgTable(triples) {
  if (triples.length === 0) {
    document.getElementById('kg-results').innerHTML =
      '<div style="padding:24px;text-align:center;color:var(--fg-muted);font-size:12px;">No triples match your filter.</div>';
    return;
  }
  let html = '<table class="kg-table"><thead><tr>';
  html += '<th>Subject</th><th>Predicate</th><th>Object</th><th>Source</th></tr></thead><tbody>';
  triples.forEach(t => {
    html += '<tr>';
    html += '<td>' + escapeHtml(t.subject) + '</td>';
    html += '<td class="kg-pred">' + escapeHtml(t.predicate) + '</td>';
    html += '<td>' + escapeHtml(t.object) + '</td>';
    html += '<td class="kg-source" title="' + escapeHtml(t.source_title || '') + '">' +
            escapeHtml((t.source_title || '').substring(0, 60)) + '</td>';
    html += '</tr>';
  });
  html += '</tbody></table>';
  document.getElementById('kg-results').innerHTML = html;
}

// Phase 48 — KG color presets. Each preset defines the full palette:
// the canvas gradient background, the node sphere shading (inner →
// mid → outer gradient stops), the highlight (fixed/pinned) gradient,
// the edge line color, and the label fill + outline. `inverse` names
// the paired light/dark preset for the Invert button.
const KG_THEMES = {
  'deep-space': {
    name: 'Deep Space',
    canvasBg: '#060b18',
    bgInner: '#1e2a44', bgOuter: '#060b18',
    nodeInner: '#ffffff', nodeMid: '#8fc3ff', nodeOuter: '#0e2a54',
    hiMid: '#ffd98a', hiOuter: '#6a3e00',
    nodeStroke: '#0a1a33',
    edge: '#7fb6ff',
    label: '#e6f0ff', labelStroke: '#040914',
    inverse: 'paper',
  },
  'paper': {
    name: 'Paper',
    canvasBg: '#f3f5f9',
    bgInner: '#ffffff', bgOuter: '#d8ddea',
    nodeInner: '#ffffff', nodeMid: '#6c8ec8', nodeOuter: '#1f3a6b',
    hiMid: '#d48f2d', hiOuter: '#7a3f00',
    nodeStroke: '#1f3a6b',
    edge: '#5a7bb0',
    label: '#0a1a33', labelStroke: '#ffffff',
    inverse: 'deep-space',
  },
  'terminal': {
    name: 'Terminal',
    canvasBg: '#020402',
    bgInner: '#0a1a0a', bgOuter: '#000000',
    nodeInner: '#f0ffe8', nodeMid: '#55e86b', nodeOuter: '#083015',
    hiMid: '#ffef40', hiOuter: '#5a4a00',
    nodeStroke: '#062008',
    edge: '#3ccc6a',
    label: '#bfffc6', labelStroke: '#000000',
    inverse: 'paper',
  },
  'blueprint': {
    name: 'Blueprint',
    canvasBg: '#061530',
    bgInner: '#123466', bgOuter: '#040d22',
    nodeInner: '#ffffff', nodeMid: '#6bf2ff', nodeOuter: '#033a55',
    hiMid: '#ffb347', hiOuter: '#5c3500',
    nodeStroke: '#02101f',
    edge: '#7bd5ff',
    label: '#eafaff', labelStroke: '#02101f',
    inverse: 'paper',
  },
  'solarized': {
    name: 'Solarized',
    canvasBg: '#002b36',
    bgInner: '#08414e', bgOuter: '#001820',
    nodeInner: '#fdf6e3', nodeMid: '#b58900', nodeOuter: '#3a2a00',
    hiMid: '#cb4b16', hiOuter: '#4a1e00',
    nodeStroke: '#001820',
    edge: '#268bd2',
    label: '#eee8d5', labelStroke: '#002b36',
    inverse: 'solarized-light',
  },
  'solarized-light': {
    name: 'Solarized Light',
    canvasBg: '#fdf6e3',
    bgInner: '#ffffff', bgOuter: '#eee8d5',
    nodeInner: '#fdf6e3', nodeMid: '#b58900', nodeOuter: '#3a2a00',
    hiMid: '#cb4b16', hiOuter: '#4a1e00',
    nodeStroke: '#3a2a00',
    edge: '#268bd2',
    label: '#073642', labelStroke: '#fdf6e3',
    inverse: 'solarized',
  },
  'neon': {
    name: 'Neon',
    canvasBg: '#000000',
    bgInner: '#1a0030', bgOuter: '#000000',
    nodeInner: '#ffffff', nodeMid: '#ff3db7', nodeOuter: '#3a0028',
    hiMid: '#2affd5', hiOuter: '#003d3a',
    nodeStroke: '#0a0014',
    edge: '#c66bff',
    label: '#ffd6ff', labelStroke: '#0a0014',
    inverse: 'paper',
  },
};
let _kgActiveTheme = 'deep-space';
// Per-user color overrides layered on top of the active preset. Empty
// object = pure preset. Any key in here wins over the same key in
// KG_THEMES[_kgActiveTheme] when the sim reads its colors. Persisted
// to localStorage alongside _kgActiveTheme so the last palette the
// user picked (preset + any custom tweaks) is restored next session.
let _kgCustomOverrides = {};

// Choose a readable text color (black or white) given a background hex,
// using the standard luminance formula. Used so that picking a label
// color auto-sets the label stroke to maintain contrast against
// whatever background the node happens to be over.
function _kgContrast(hex) {
  const m = (hex || '').replace('#', '').trim();
  if (m.length !== 6 && m.length !== 3) return '#000000';
  const full = m.length === 3
    ? m.split('').map(c => c + c).join('')
    : m;
  const r = parseInt(full.slice(0, 2), 16);
  const g = parseInt(full.slice(2, 4), 16);
  const b = parseInt(full.slice(4, 6), 16);
  const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  return lum > 0.55 ? '#0a0a0a' : '#ffffff';
}

// Darken a hex color by a 0..1 factor (0.7 = 70% of original). Used to
// derive the outer shading stop of a sphere-shaded node when the user
// picks the node's main color.
function _kgDarken(hex, factor) {
  const m = (hex || '').replace('#', '').trim();
  if (m.length !== 6) return hex;
  const r = Math.max(0, Math.min(255, Math.round(parseInt(m.slice(0, 2), 16) * factor)));
  const g = Math.max(0, Math.min(255, Math.round(parseInt(m.slice(2, 4), 16) * factor)));
  const b = Math.max(0, Math.min(255, Math.round(parseInt(m.slice(4, 6), 16) * factor)));
  const toHex = (v) => v.toString(16).padStart(2, '0');
  return '#' + toHex(r) + toHex(g) + toHex(b);
}

// Merged palette: active preset with overrides applied on top. Callers
// that need colors should go through this, never read KG_THEMES
// directly, so custom tweaks are respected.
function _kgEffectiveTheme() {
  const base = KG_THEMES[_kgActiveTheme] || KG_THEMES['deep-space'];
  return Object.assign({}, base, _kgCustomOverrides || {});
}

// Persist theme + overrides across sessions. The key is versioned
// (`_v1`) so we can evolve the shape later without loading stale data.
function _kgSavePrefs() {
  try {
    localStorage.setItem('kg_prefs_v1', JSON.stringify({
      theme: _kgActiveTheme,
      overrides: _kgCustomOverrides,
    }));
  } catch (e) { /* localStorage can throw in private mode */ }
}
function _kgLoadPrefs() {
  try {
    const s = localStorage.getItem('kg_prefs_v1');
    if (!s) return;
    const p = JSON.parse(s);
    if (p && p.theme && KG_THEMES[p.theme]) _kgActiveTheme = p.theme;
    if (p && p.overrides && typeof p.overrides === 'object') {
      _kgCustomOverrides = p.overrides;
    }
  } catch (e) { /* ignore corrupted prefs */ }
}

// Push the effective theme into the running simulation without
// restarting it. Called after any preset / override change.
function _kgRefreshLiveTheme() {
  const c = document.getElementById('kg-graph-canvas');
  if (c && c._kgSim && c._kgSim.refreshTheme) c._kgSim.refreshTheme();
  // Keep the color-picker inputs in sync with the effective theme so
  // reopening the modal doesn't show stale values against a live bg.
  const t = _kgEffectiveTheme();
  const bgPicker = document.getElementById('kg-color-bg');
  const lbPicker = document.getElementById('kg-color-label');
  const edPicker = document.getElementById('kg-color-edge');
  const ndPicker = document.getElementById('kg-color-node');
  if (bgPicker) bgPicker.value = t.canvasBg || t.bgOuter || '#060b18';
  if (lbPicker) lbPicker.value = t.label || '#e6f0ff';
  if (edPicker) edPicker.value = t.edge || '#7fb6ff';
  if (ndPicker) ndPicker.value = t.nodeMid || '#8fc3ff';
}

// Handler for each color-picker input (BG / Label / Edge / Node). For
// BG we set all three bg stops to the same color (solid fill); for
// Label we auto-derive the stroke as the contrast color so labels
// remain readable. Overrides are saved to localStorage immediately.
function kgSetCustomColor(kind, value) {
  if (!value || typeof value !== 'string') return;
  if (kind === 'bg') {
    _kgCustomOverrides.canvasBg = value;
    _kgCustomOverrides.bgInner = value;
    _kgCustomOverrides.bgOuter = value;
  } else if (kind === 'label') {
    _kgCustomOverrides.label = value;
    _kgCustomOverrides.labelStroke = _kgContrast(value);
  } else if (kind === 'edge') {
    _kgCustomOverrides.edge = value;
  } else if (kind === 'node') {
    _kgCustomOverrides.nodeMid = value;
    _kgCustomOverrides.nodeOuter = _kgDarken(value, 0.35);
    _kgCustomOverrides.nodeStroke = _kgDarken(value, 0.2);
  } else {
    return;
  }
  _kgSavePrefs();
  _kgRefreshLiveTheme();
}

// Wipe all per-user overrides, snapping the live theme back to whatever
// preset is currently active.
function kgClearCustomColors() {
  _kgCustomOverrides = {};
  _kgSavePrefs();
  _kgRefreshLiveTheme();
}

// Toggle fullscreen for the KG graph pane. Using the pane (not the
// whole modal) means the toolbar + canvas fill the screen while the
// rest of the modal chrome is hidden by the browser's fullscreen UI.
function kgToggleFullscreen() {
  const target = document.getElementById('kg-graph-pane');
  if (!target) return;
  const isFs = !!(document.fullscreenElement || document.webkitFullscreenElement);
  if (!isFs) {
    const req = target.requestFullscreen || target.webkitRequestFullscreen;
    if (req) req.call(target).catch(() => {});
  } else {
    const ex = document.exitFullscreen || document.webkitExitFullscreen;
    if (ex) ex.call(document).catch(() => {});
  }
}
// Keep the fullscreen button label accurate across user-initiated
// enter/exit (e.g. ESC or browser chrome).
document.addEventListener('fullscreenchange', () => {
  const btn = document.getElementById('kg-fullscreen-btn');
  if (!btn) return;
  btn.innerHTML = document.fullscreenElement
    ? '\u2922 Exit fullscreen'
    : '\u26F6 Fullscreen';
});

// Apply a theme's gradient stops to the SVG <defs>. Called on init and
// whenever the user picks a new preset — the render() loop then paints
// next frame with the new palette without restarting the simulation.
function _applyKgDefs(svg, theme) {
  const defs = svg.querySelector('defs');
  if (!defs) return;
  defs.innerHTML =
    '<radialGradient id="kg-nodeg" cx="30%" cy="30%" r="75%">' +
      '<stop offset="0%" stop-color="' + theme.nodeInner + '"/>' +
      '<stop offset="35%" stop-color="' + theme.nodeMid + '"/>' +
      '<stop offset="100%" stop-color="' + theme.nodeOuter + '"/>' +
    '</radialGradient>' +
    '<radialGradient id="kg-nodeh" cx="30%" cy="30%" r="75%">' +
      '<stop offset="0%" stop-color="' + theme.nodeInner + '"/>' +
      '<stop offset="30%" stop-color="' + theme.hiMid + '"/>' +
      '<stop offset="100%" stop-color="' + theme.hiOuter + '"/>' +
    '</radialGradient>' +
    '<radialGradient id="kg-bg" cx="50%" cy="50%" r="80%">' +
      '<stop offset="0%" stop-color="' + theme.bgInner + '"/>' +
      '<stop offset="100%" stop-color="' + theme.bgOuter + '"/>' +
    '</radialGradient>';
}

// Switch the active KG theme. Clicking a preset clears any custom
// overrides (the chip is meant as a reset-to-known-good). Safe to
// call before the graph is built — it just updates state that the
// next _renderKgGraph will read.
function setKgTheme(name) {
  if (!KG_THEMES[name]) return;
  _kgActiveTheme = name;
  _kgCustomOverrides = {};
  _kgSavePrefs();
  document.querySelectorAll('.kg-theme-chip').forEach(c => {
    c.classList.toggle('active', c.getAttribute('data-theme') === name);
  });
  _kgRefreshLiveTheme();
}

// One-click swap to the paired light/dark preset of the current theme.
function invertKgTheme() {
  const cur = KG_THEMES[_kgActiveTheme];
  if (cur && cur.inverse) setKgTheme(cur.inverse);
}

// Paint each theme chip with a tiny radial preview of its palette
// (one-time; chips live inside the modal markup). Also loads the
// persisted prefs from the last session so the initial render uses
// the user's saved theme + overrides instead of the defaults.
function _initKgThemeChips() {
  if (window._kgChipsReady) return;
  _kgLoadPrefs();
  document.querySelectorAll('.kg-theme-chip').forEach(chip => {
    const name = chip.getAttribute('data-theme');
    const t = KG_THEMES[name];
    if (!t) return;
    chip.style.background =
      'radial-gradient(circle at 35% 35%, ' +
      t.nodeMid + ' 0%, ' + t.nodeOuter + ' 55%, ' +
      t.bgOuter + ' 100%)';
    chip.title = t.name;
    chip.classList.toggle('active', name === _kgActiveTheme);
    chip.addEventListener('click', () => setKgTheme(name));
  });
  // Seed the color pickers from the effective theme.
  const t = _kgEffectiveTheme();
  const pairs = [
    ['kg-color-bg', t.canvasBg],
    ['kg-color-label', t.label],
    ['kg-color-edge', t.edge],
    ['kg-color-node', t.nodeMid],
  ];
  pairs.forEach(([id, val]) => {
    const el = document.getElementById(id);
    if (el && val) el.value = val;
  });
  window._kgChipsReady = true;
}

// Phase 48b — Okabe-Ito-derived palette for Louvain community coloring.
// Colorblind-safe, eight distinct hues; paired (mid, outer) stops give
// each cluster a sphere-shaded gradient without a full per-theme
// cluster palette. The 9th entry is a neutral gray for overflow.
const KG_CLUSTER_PALETTE = [
  { mid: '#56B4E9', outer: '#0c3850' },  // sky blue
  { mid: '#E69F00', outer: '#50300c' },  // orange
  { mid: '#009E73', outer: '#0a3d2c' },  // bluish green
  { mid: '#F0E442', outer: '#504a0c' },  // yellow
  { mid: '#0072B2', outer: '#001f3a' },  // blue
  { mid: '#D55E00', outer: '#3d1a00' },  // vermillion
  { mid: '#CC79A7', outer: '#3e1f32' },  // reddish purple
  { mid: '#8ed6a5', outer: '#234a31' },  // mint
  { mid: '#999999', outer: '#333333' },  // overflow gray
];

// Predicate family → semantic category → color. Follows VOWL/WebVOWL
// convention of grouping predicates by what they *do* rather than
// coloring every predicate separately (unreadable above ~8 hues).
const KG_PREDICATE_FAMILIES = {
  causal:      { color: '#D55E00', glyph: '\u2192',
                 keys: ['cause', 'increase', 'decrease', 'induce', 'affect',
                        'drive', 'reduce', 'enhance', 'inhibit', 'trigger',
                        'lead to', 'result in', 'contribute', 'produce'] },
  measurement: { color: '#0072B2', glyph: '\u2248',
                 keys: ['measure', 'observe', 'detect', 'record', 'sample',
                        'estimate', 'quantify', 'report', 'find', 'show'] },
  taxonomic:   { color: '#009E73', glyph: '\u2282',
                 keys: ['is a', 'is-a', 'type', 'subtype', 'part of',
                        'part-of', 'contains', 'includes', 'belongs',
                        'kind of', 'category'] },
  compositional:{ color: '#8ed6a5', glyph: '\u25c6',
                 keys: ['uses', 'use', 'composed', 'consists', 'built',
                        'based on', 'based-on', 'relies', 'applies'] },
  citational:  { color: '#999999', glyph: '\u00a7',
                 keys: ['cite', 'reference', 'evidence', 'support',
                        'contradict', 'agree', 'disagree', 'extend'] },
  other:       { color: '#CC79A7', glyph: '\u2022', keys: [] },
};

function _kgPredicateFamily(pred) {
  const p = (pred || '').toLowerCase();
  for (const fam of ['causal', 'measurement', 'taxonomic', 'compositional', 'citational']) {
    for (const key of KG_PREDICATE_FAMILIES[fam].keys) {
      if (p.indexOf(key) !== -1) return fam;
    }
  }
  return 'other';
}

// Louvain community detection (one-level local-moving pass). Input:
// number of nodes + edges with source/target/count. Output: array of
// community index per node, densely reindexed from 0. Fast enough for
// n ≤ 500 (≈ ms on a modern laptop); the one-level variant skips the
// aggregation step but still produces clusters good enough to drive
// node coloring + gravity wells for our scale.
function _kgLouvain(numNodes, edges) {
  if (numNodes === 0) return [];
  // Build weighted undirected adjacency
  const adj = [];
  for (let i = 0; i < numNodes; i++) adj.push(new Map());
  let m2 = 0;
  edges.forEach(e => {
    if (e.source === e.target) return;
    const w = Math.max(1, e.count || 1);
    adj[e.source].set(e.target, (adj[e.source].get(e.target) || 0) + w);
    adj[e.target].set(e.source, (adj[e.target].get(e.source) || 0) + w);
    m2 += 2 * w;
  });
  if (m2 === 0) return new Array(numNodes).fill(0);
  const degree = new Array(numNodes);
  for (let i = 0; i < numNodes; i++) {
    let d = 0;
    for (const w of adj[i].values()) d += w;
    degree[i] = d;
  }
  const community = new Array(numNodes);
  for (let i = 0; i < numNodes; i++) community[i] = i;
  const commSum = {};
  for (let i = 0; i < numNodes; i++) commSum[i] = degree[i];

  let changed = true, iter = 0;
  while (changed && iter < 12) {
    changed = false; iter++;
    for (let i = 0; i < numNodes; i++) {
      const cur = community[i];
      // Sum of weights from i to each neighboring community
      const wToComm = new Map();
      for (const [j, w] of adj[i]) {
        const c = community[j];
        wToComm.set(c, (wToComm.get(c) || 0) + w);
      }
      // Temporarily remove i from its community for the gain calc
      commSum[cur] -= degree[i];
      let bestC = cur, bestGain = 0;
      const kI = degree[i];
      for (const [c, kIin] of wToComm) {
        if (c === cur) continue;
        const sigmaTot = commSum[c] || 0;
        const gain = kIin - (sigmaTot * kI) / m2;
        if (gain > bestGain) { bestGain = gain; bestC = c; }
      }
      // Stay in cur if no better community
      commSum[bestC] = (commSum[bestC] || 0) + degree[i];
      if (bestC !== cur) { community[i] = bestC; changed = true; }
    }
  }
  // Relabel to dense 0..k
  const remap = new Map();
  let k = 0;
  const result = new Array(numNodes);
  for (let i = 0; i < numNodes; i++) {
    const c = community[i];
    if (!remap.has(c)) remap.set(c, k++);
    result[i] = remap.get(c);
  }
  return result;
}

// Inject per-cluster gradient defs into the SVG so node `fill` can
// reference `url(#kg-cluster-N)`. Called once after Louvain runs and
// again on theme changes (so the gradient inner/stroke stays coherent).
function _applyKgClusterDefs(svg, theme, numClusters) {
  const defs = svg.querySelector('defs');
  if (!defs) return;
  let extra = '';
  for (let i = 0; i < numClusters; i++) {
    const p = KG_CLUSTER_PALETTE[i % KG_CLUSTER_PALETTE.length];
    extra +=
      '<radialGradient id="kg-cluster-' + i + '" cx="30%" cy="30%" r="75%">' +
        '<stop offset="0%" stop-color="' + theme.nodeInner + '"/>' +
        '<stop offset="35%" stop-color="' + p.mid + '"/>' +
        '<stop offset="100%" stop-color="' + p.outer + '"/>' +
      '</radialGradient>';
  }
  // Append to the existing defs (which _applyKgDefs already populated)
  defs.insertAdjacentHTML('beforeend', extra);
}

// Cubic ease-in-out for the center-on-node camera tween.
function _kgEase(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

// Render/hide the floating context menu. The <div> lives in the modal
// HTML so any menu action has access to the current simulation closure
// via the global canvas._kgSim handle.
function _kgShowMenu(x, y, items) {
  const menu = document.getElementById('kg-context-menu');
  if (!menu) return;
  menu.innerHTML = '';
  items.forEach(item => {
    if (item === '-') {
      const sep = document.createElement('div');
      sep.className = 'kg-menu-sep';
      menu.appendChild(sep);
      return;
    }
    const btn = document.createElement('button');
    btn.className = 'kg-menu-item';
    btn.textContent = item.label;
    if (item.hint) {
      const hint = document.createElement('span');
      hint.className = 'kg-menu-hint';
      hint.textContent = item.hint;
      btn.appendChild(hint);
    }
    btn.addEventListener('click', () => {
      _kgHideMenu();
      try { item.onClick(); } catch (e) { console.error(e); }
    });
    menu.appendChild(btn);
  });
  // Position; clamp to viewport
  menu.style.display = 'block';
  const mw = menu.offsetWidth, mh = menu.offsetHeight;
  const vw = window.innerWidth, vh = window.innerHeight;
  menu.style.left = Math.min(x, vw - mw - 8) + 'px';
  menu.style.top  = Math.min(y, vh - mh - 8) + 'px';
}
function _kgHideMenu() {
  const menu = document.getElementById('kg-context-menu');
  if (menu) menu.style.display = 'none';
}
// Dismiss the menu on any outside click or Escape
document.addEventListener('click', (e) => {
  const menu = document.getElementById('kg-context-menu');
  if (menu && menu.style.display === 'block' && !menu.contains(e.target)) {
    _kgHideMenu();
  }
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') _kgHideMenu();
});

// Toolbar click/input handlers. Defined at module scope so the HTML can
// reference them via onclick; each forwards to the live simulation via
// canvas._kgSim.
function kgToggleFreeze() {
  const c = document.getElementById('kg-graph-canvas');
  if (c && c._kgSim && c._kgSim.togglePause) c._kgSim.togglePause();
}
function kgResetView() {
  const c = document.getElementById('kg-graph-canvas');
  if (c && c._kgSim && c._kgSim.resetView) c._kgSim.resetView();
}
function kgDownloadPng() {
  const c = document.getElementById('kg-graph-canvas');
  if (c && c._kgSim && c._kgSim.downloadPng) c._kgSim.downloadPng();
}
function kgSetColorBy(mode) {
  const c = document.getElementById('kg-graph-canvas');
  if (c && c._kgSim && c._kgSim.setColorBy) c._kgSim.setColorBy(mode);
}
function kgSetLabelScale(v) {
  const c = document.getElementById('kg-graph-canvas');
  if (c && c._kgSim && c._kgSim.setLabelScale) c._kgSim.setLabelScale(parseFloat(v));
}
// Phase 54.6.10 — typography picker for KG labels. Persists the
// choice in localStorage so it survives modal reopens.
function kgSetFont(key) {
  const c = document.getElementById('kg-graph-canvas');
  if (c && c._kgSim && c._kgSim.setFont) c._kgSim.setFont(key);
  try { localStorage.setItem('sciknow.kg.font', key); } catch (_) {}
}
function kgSetDegFilter(v) {
  const c = document.getElementById('kg-graph-canvas');
  if (c && c._kgSim && c._kgSim.setDegFilter) c._kgSim.setDegFilter(parseInt(v, 10));
  const lbl = document.getElementById('kg-degfilter-label');
  if (lbl) lbl.textContent = (parseInt(v, 10) >= 99 ? '∞' : v);
}
function kgSearch(q) {
  const c = document.getElementById('kg-graph-canvas');
  if (c && c._kgSim && c._kgSim.search) c._kgSim.search(q);
}

// Phase 48d — cached layout per filter. Stores the final node
// positions for a given filter combination in localStorage so
// re-opening the same view warm-starts from the prior layout instead
// of re-running the full settle from random seeds every time.

function _kgLayoutKey() {
  const s = (document.getElementById('kg-subject') || {}).value || '';
  const p = (document.getElementById('kg-predicate') || {}).value || '';
  const o = (document.getElementById('kg-object') || {}).value || '';
  // Compact base64 without +/= so the key stays URL-safe and short.
  const raw = JSON.stringify({ s: s, p: p, o: o });
  try {
    return 'kg_layout_' + btoa(raw).replace(/[+/=]/g, '').slice(0, 40);
  } catch (e) {
    return 'kg_layout_default';
  }
}
function _kgLoadLayout(key) {
  try {
    const s = localStorage.getItem(key);
    if (!s) return null;
    const d = JSON.parse(s);
    return (d && d.nodes) || null;
  } catch (e) { return null; }
}
function _kgSaveLayout(key, nodes) {
  const positions = nodes.map(n => ({
    l: n.label,
    p: [Math.round(n.x), Math.round(n.y), Math.round(n.z)],
  }));
  const payload = JSON.stringify({ nodes: positions, ts: Date.now() });
  try {
    localStorage.setItem(key, payload);
  } catch (e) {
    // Storage full → evict aggressively then retry once.
    _kgEvictOldLayouts(5);
    try { localStorage.setItem(key, payload); } catch (e2) { /* give up */ }
  }
  _kgEvictOldLayouts(30);
}
function _kgEvictOldLayouts(maxCount) {
  const keys = [];
  for (let i = 0; i < localStorage.length; i++) {
    const k = localStorage.key(i);
    if (!k || k.indexOf('kg_layout_') !== 0) continue;
    try {
      const ts = (JSON.parse(localStorage.getItem(k)) || {}).ts || 0;
      keys.push([k, ts]);
    } catch (e) { /* skip corrupt */ }
  }
  keys.sort((a, b) => a[1] - b[1]);
  while (keys.length > maxCount) {
    const [k] = keys.shift();
    try { localStorage.removeItem(k); } catch (e) {}
  }
}

// ── Shareable URL ─────────────────────────────────────────────────
// Encodes theme + overrides + filter fields + camera + pinned node
// labels in the URL hash as compact base64 JSON. Anyone who opens
// the URL gets the KG modal auto-opened on exactly the same view.

function kgCopyShareLink() {
  const c = document.getElementById('kg-graph-canvas');
  if (!c || !c._kgSim || !c._kgSim.getShareState) return;
  const st = c._kgSim.getShareState();
  st.t = _kgActiveTheme;
  st.o = _kgCustomOverrides;
  st.f = {
    s: (document.getElementById('kg-subject') || {}).value || '',
    p: (document.getElementById('kg-predicate') || {}).value || '',
    o: (document.getElementById('kg-object') || {}).value || '',
  };
  let enc;
  try {
    enc = btoa(unescape(encodeURIComponent(JSON.stringify(st))));
  } catch (e) { return; }
  const url = window.location.origin + window.location.pathname + '#kg=' + enc;
  try {
    navigator.clipboard.writeText(url).then(() => {
      const status = document.getElementById('kg-status');
      if (status) status.textContent = 'Share link copied to clipboard.';
    }, () => window.prompt('Copy this link:', url));
  } catch (e) { window.prompt('Copy this link:', url); }
}
// Parse #kg=… on load; if present, stash the state and have openKgModal
// apply it after the first render completes.
(function _kgMaybeParseHashOnLoad() {
  const m = (window.location.hash || '').match(/^#kg=(.+)$/);
  if (!m) return;
  try {
    const st = JSON.parse(decodeURIComponent(escape(atob(m[1]))));
    window._kgPendingShare = st;
    if (st.t && typeof KG_THEMES !== 'undefined' && KG_THEMES[st.t]) {
      _kgActiveTheme = st.t;
    }
    if (st.o) _kgCustomOverrides = st.o;
  } catch (e) { /* ignore bad hash */ }
})();

// Apply a pending share state (filter + cam + pinned) after the sim
// has rendered at least once. Called from the first render() frame.
function _kgApplyPendingShare() {
  const st = window._kgPendingShare;
  if (!st) return;
  window._kgPendingShare = null;
  // Filter fields were already set before the load via openKgModal's
  // hook; re-apply camera + pins now that the sim exists.
  const c = document.getElementById('kg-graph-canvas');
  if (c && c._kgSim && c._kgSim.applyShareState) {
    c._kgSim.applyShareState(st);
  }
}

// Phase 48 — interactive 3D knowledge graph. Every unique entity is a
// node in a real 3D world; triples are weighted edges (same-pair
// triples merged into one visual edge with count + family info). A
// continuous rAF loop runs a ForceAtlas2-derived physics model
// (log-weighted attraction with dissuade-hubs, repulsion scaled by
// degree, per-cluster gravity wells) + a 3D orbit camera. Interactions
// include drag-to-orbit, drag-nodes, wheel-zoom, hover-dim-1-hop,
// spacebar-freeze, right-click context menu, search, center-on-click,
// PNG export. Palette + coloring mode are swappable live without
// restarting the simulation. No extra deps — still pure SVG.
function _renderKgGraph(triples) {
  const canvas = document.getElementById('kg-graph-canvas');
  if (!canvas) return;
  _initKgThemeChips();
  if (canvas._kgSim) { try { canvas._kgSim.stop(); } catch (e) {} }
  canvas.innerHTML = '';
  if (!triples || triples.length === 0) {
    canvas.innerHTML = '<div style="padding:80px 24px;text-align:center;color:var(--fg-muted);font-size:12px;">No triples match your filter.</div>';
    return;
  }

  const W = canvas.clientWidth || 800;
  const H = 520;

  // ── Build nodes + aggregated edges ─────────────────────────────
  // Multiple triples between the same (subject, object) pair collapse
  // into one logical edge with a `count` and the list of source
  // triples (so right-click → "source paper" still works for each
  // underlying claim). Direction is preserved — a→b and b→a stay
  // separate edges and get opposite curvature offsets.
  const nodeIndex = new Map();
  const nodes = [];
  function ensureNode(label) {
    if (!nodeIndex.has(label)) {
      nodeIndex.set(label, nodes.length);
      nodes.push({
        id: nodes.length, label: label,
        x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0,
        fixed: false, hidden: false, degree: 0, cluster: 0,
      });
    }
    return nodeIndex.get(label);
  }
  const edgeMap = new Map();  // "s→t" → edge
  triples.forEach(t => {
    const sLab = (t.subject || '').substring(0, 60);
    const oLab = (t.object  || '').substring(0, 60);
    if (!sLab || !oLab || sLab === oLab) return;
    const s = ensureNode(sLab), o = ensureNode(oLab);
    const key = s + '\u2192' + o;
    let e = edgeMap.get(key);
    if (!e) {
      e = { source: s, target: o, count: 0,
             triples: [], families: new Set(), predicates: new Set() };
      edgeMap.set(key, e);
    }
    e.count++;
    e.predicates.add(t.predicate || '');
    e.families.add(_kgPredicateFamily(t.predicate || ''));
    e.triples.push({
      predicate: t.predicate || '',
      doc_id: t.source_doc_id,
      doc_title: t.source_title,
      confidence: t.confidence,
      // Phase 48d — per-triple verbatim sentence from the source
      // paper; may be null for rows ingested before migration 0019.
      source_sentence: t.source_sentence || '',
    });
  });
  const edges = Array.from(edgeMap.values());
  edges.forEach(e => { nodes[e.source].degree++; nodes[e.target].degree++; });
  function nodeLabel(label) {
    return label.length > 24 ? label.substring(0, 24) + '\u2026' : label;
  }

  // ── Community detection (Louvain, undirected, count-weighted) ──
  const communities = _kgLouvain(nodes.length, edges);
  const numClusters = Math.max(1, (communities.length
    ? Math.max.apply(null, communities) + 1 : 1));
  nodes.forEach((n, i) => { n.cluster = communities[i] || 0; });

  // Place cluster centroids on a sphere (Fibonacci lattice), scaled by
  // √numClusters so the per-well radius stays roughly constant.
  const clusterCenters = [];
  const wellR = 180 + 30 * Math.sqrt(numClusters);
  for (let c = 0; c < numClusters; c++) {
    const frac = (c + 0.5) / numClusters;
    const phi = Math.acos(1 - 2 * frac);
    const theta = Math.PI * (1 + Math.sqrt(5)) * c;
    clusterCenters.push({
      x: wellR * Math.sin(phi) * Math.cos(theta),
      y: wellR * Math.sin(phi) * Math.sin(theta),
      z: wellR * Math.cos(phi),
    });
  }
  // Seed node positions inside their cluster so layout converges fast
  nodes.forEach(n => {
    const c = clusterCenters[n.cluster % clusterCenters.length];
    n.x = c.x + (Math.random() - 0.5) * 80;
    n.y = c.y + (Math.random() - 0.5) * 80;
    n.z = c.z + (Math.random() - 0.5) * 80;
  });
  // Phase 48d — warm-start from the cached layout for this filter, if
  // we have one. Only positions saved under the *current* filter hash
  // apply; nodes that didn't exist in the cached layout keep their
  // freshly-seeded cluster coords (the layout still settles, just
  // faster and with less drift between views).
  const _kgLayoutKeyCurrent = _kgLayoutKey();
  const _kgLayoutCached = _kgLoadLayout(_kgLayoutKeyCurrent);
  if (_kgLayoutCached && Array.isArray(_kgLayoutCached)) {
    const byLabel = new Map();
    _kgLayoutCached.forEach(p => byLabel.set(p.l, p.p));
    nodes.forEach(n => {
      const pos = byLabel.get(n.label);
      if (pos && pos.length === 3) {
        n.x = pos[0]; n.y = pos[1]; n.z = pos[2];
      }
    });
  }

  // ── Curve-bundle offsets (parallel + bidirectional edges) ──────
  // For each unordered pair {u,v}, count how many edges connect them
  // and assign each an offset index so they fan out around the line.
  // A→B and B→A flip sign of the offset so they sit on opposite sides.
  const bundleCount = new Map();
  edges.forEach(e => {
    const lo = Math.min(e.source, e.target);
    const hi = Math.max(e.source, e.target);
    const pair = lo + '|' + hi;
    const n = bundleCount.get(pair) || 0;
    e._pair = pair;
    e._bundleIdx = n;
    e._dirSign = (e.source < e.target) ? 1 : -1;
    bundleCount.set(pair, n + 1);
  });
  edges.forEach(e => {
    const total = bundleCount.get(e._pair);
    const mid = (total - 1) / 2;
    // Bundle offset lives in screen-space px; ±15 per lane works well
    e._offset = (e._bundleIdx - mid) * 15 * e._dirSign;
  });

  // ── Camera + view state ────────────────────────────────────────
  const cam = { rotX: -0.22, rotY: 0.55, dist: 850, fov: 680 };
  const camDefault = { rotX: -0.22, rotY: 0.55, dist: 850 };
  let theme = _kgEffectiveTheme();
  let colorBy = 'cluster';  // 'cluster' | 'predicate' | 'theme'
  let labelScale = 1.0;
  // Phase 54.6.10 — label typography. Supported families:
  //   sans (default)   → var(--font-sans)      (Inter + system)
  //   serif            → var(--font-serif)     (Georgia / TNR)
  //   mono             → var(--font-mono)      (SF Mono / ui-mono)
  //   condensed        → Barlow Condensed / Arial Narrow
  //   display          → Georgia Display / Playfair-style
  // `solid=true` drops the paint-order halo so the label is one
  // colour (fill only). On bright themes the halo hurts more than
  // it helps, so "solid" is the preferred preset for readability.
  let labelFontFamily = 'var(--font-sans)';
  let labelSolid = false;
  // Restore the user's saved typography preference (if any) and
  // sync the <select> back to it so the dropdown shows the active
  // choice next time the modal opens.
  try {
    const savedFont = localStorage.getItem('sciknow.kg.font');
    if (savedFont) {
      const fontMap = {
        'sans-halo':      ['var(--font-sans)',  false],
        'sans-solid':     ['var(--font-sans)',  true],
        'serif-solid':    ['var(--font-serif)', true],
        'serif-halo':     ['var(--font-serif)', false],
        'mono-solid':     ['var(--font-mono)',  true],
        'mono-halo':      ['var(--font-mono)',  false],
        'condensed-solid':['"Barlow Condensed","Arial Narrow",sans-serif', true],
        'display-solid':  ['"Playfair Display",Georgia,serif', true],
      };
      const pair = fontMap[savedFont];
      if (pair) { labelFontFamily = pair[0]; labelSolid = pair[1]; }
      const sel = document.getElementById('kg-font-select');
      if (sel) sel.value = savedFont;
    }
  } catch (_) {}
  let degFilter = 999;       // max degree; nodes above this are hidden
  let hoverId = -1;          // node id under the mouse (-1 if none)
  let hoverNeighbors = null; // Set<id> of 1-hop neighbors when hovering
  let hoverEdgeSet = null;   // Set<edge-index> of edges incident to hover
  let searchMatches = null;  // Set<id> of nodes matching the live search
  let running = true, paused = false, raf = null;

  // ── SVG scaffold ───────────────────────────────────────────────
  const svgNS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(svgNS, 'svg');
  svg.setAttribute('viewBox', (-W/2) + ' ' + (-H/2) + ' ' + W + ' ' + H);
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  svg.innerHTML =
    '<defs></defs>' +
    '<rect class="kg-bg" x="' + (-W/2) + '" y="' + (-H/2) + '" width="' + W +
      '" height="' + H + '" fill="url(#kg-bg)" pointer-events="all"/>' +
    '<g class="kg-edges"></g><g class="kg-nodes"></g>';
  canvas.appendChild(svg);
  _applyKgDefs(svg, theme);
  _applyKgClusterDefs(svg, theme, numClusters);
  canvas.style.background = theme.canvasBg;
  const edgeLayer = svg.querySelector('.kg-edges');
  const nodeLayer = svg.querySelector('.kg-nodes');

  // ── Projection ─────────────────────────────────────────────────
  function project(n) {
    const cy = Math.cos(cam.rotY), sy = Math.sin(cam.rotY);
    const cx = Math.cos(cam.rotX), sxA = Math.sin(cam.rotX);
    const xr = n.x * cy + n.z * sy;
    const zr = -n.x * sy + n.z * cy;
    const yr = n.y;
    const yc = yr * cx - zr * sxA;
    const zc = yr * sxA + zr * cx;
    const zcam = zc + cam.dist;
    if (zcam <= 1) return { sx: 0, sy: 0, zcam: 1, scale: 0.0001 };
    const scale = cam.fov / zcam;
    return { sx: xr * scale, sy: yc * scale, zcam: zcam, scale: scale };
  }
  function worldDelta(dx, dy, zcam) {
    const scale = cam.fov / Math.max(zcam, 1);
    const dxc = dx / scale, dyc = dy / scale;
    const cx = Math.cos(cam.rotX), sxA = Math.sin(cam.rotX);
    const cy = Math.cos(cam.rotY), sy = Math.sin(cam.rotY);
    const xr = dxc, yr = dyc * cx, zr = -dyc * sxA;
    return { x: xr * cy - zr * sy, y: yr, z: xr * sy + zr * cy };
  }

  // ── ForceAtlas2-derived physics ────────────────────────────────
  // Repulsion scales with (deg+1)(deg+1) so hubs push each other away
  // strongly (the FA2 "dissuade-hubs" trick). Attraction uses log(1+d)
  // — the linLog mode of Noack/FA2 — which gives dramatically better
  // hub separation than d² on real citation graphs. Edge weight is
  // log(1+count) so a single high-count merged edge can't collapse
  // the layout. A weak attractor at each Louvain centroid keeps
  // communities visually together.
  const KR = 120;             // repulsion strength
  const KA = 0.08;            // attraction strength
  const KW = 0.0025;          // cluster-well strength
  const KC = 0.0004;          // origin-centering strength
  const DAMP = 0.78;
  function step() {
    if (paused) return;
    for (let i = 0; i < nodes.length; i++) {
      nodes[i].ax = 0; nodes[i].ay = 0; nodes[i].az = 0;
    }
    // Repulsion (O(n²); fine ≤ 500 nodes)
    for (let i = 0; i < nodes.length; i++) {
      if (nodes[i].hidden) continue;
      for (let j = i + 1; j < nodes.length; j++) {
        if (nodes[j].hidden) continue;
        const a = nodes[i], b = nodes[j];
        const dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
        const d2 = dx*dx + dy*dy + dz*dz + 1;
        const d = Math.sqrt(d2);
        const k = KR * (a.degree + 1) * (b.degree + 1);
        const f = k / d2;
        const ux = dx/d, uy = dy/d, uz = dz/d;
        a.ax += ux*f; a.ay += uy*f; a.az += uz*f;
        b.ax -= ux*f; b.ay -= uy*f; b.az -= uz*f;
      }
    }
    // Attraction (linLog, log-count-weighted, /degree for hub dissuade)
    edges.forEach(e => {
      const a = nodes[e.source], b = nodes[e.target];
      if (a.hidden || b.hidden) return;
      const dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
      const d = Math.sqrt(dx*dx + dy*dy + dz*dz) + 0.01;
      const w = Math.log(1 + e.count);
      const f = KA * w * Math.log(1 + d);
      const ux = dx/d, uy = dy/d, uz = dz/d;
      const fa = f / Math.max(a.degree + 1, 1);
      const fb = f / Math.max(b.degree + 1, 1);
      a.ax += ux * fa; a.ay += uy * fa; a.az += uz * fa;
      b.ax -= ux * fb; b.ay -= uy * fb; b.az -= uz * fb;
    });
    // Cluster gravity wells + weak origin pull
    nodes.forEach(n => {
      if (n.hidden) return;
      const c = clusterCenters[n.cluster % clusterCenters.length];
      n.ax += (c.x - n.x) * KW;
      n.ay += (c.y - n.y) * KW;
      n.az += (c.z - n.z) * KW;
      n.ax -= n.x * KC;
      n.ay -= n.y * KC;
      n.az -= n.z * KC;
      if (n.fixed) { n.vx = 0; n.vy = 0; n.vz = 0; return; }
      n.vx = (n.vx + n.ax) * DAMP;
      n.vy = (n.vy + n.ay) * DAMP;
      n.vz = (n.vz + n.az) * DAMP;
      n.x += n.vx; n.y += n.vy; n.z += n.vz;
    });
  }

  // ── Render ─────────────────────────────────────────────────────
  function edgeColor(e) {
    if (colorBy === 'predicate') {
      // If the edge has multiple families, pick the first by family
      // priority order so the color is stable across frames
      const prio = ['causal', 'measurement', 'taxonomic', 'compositional', 'citational', 'other'];
      for (const fam of prio) {
        if (e.families.has(fam)) return KG_PREDICATE_FAMILIES[fam].color;
      }
      return KG_PREDICATE_FAMILIES.other.color;
    }
    return theme.edge;
  }
  function nodeFill(n) {
    if (n.fixed) return 'url(#kg-nodeh)';
    if (colorBy === 'cluster' && numClusters > 1) {
      return 'url(#kg-cluster-' + (n.cluster % KG_CLUSTER_PALETTE.length) + ')';
    }
    return 'url(#kg-nodeg)';
  }
  function nodeVisible(n) {
    if (n.hidden) return false;
    if (n.degree > degFilter) return false;
    return true;
  }

  function render() {
    const proj = nodes.map(n => ({ n: n, p: project(n) }));
    const order = proj.slice().sort((a, b) => b.p.zcam - a.p.zcam);

    // Dim rules: while hovering a node, non-neighbors fade; same for
    // search matches. The effects compose — a match that isn't a
    // hover-neighbor is both search-boosted and hover-dimmed.
    function nodeDim(id) {
      let k = 1.0;
      if (hoverNeighbors && !hoverNeighbors.has(id)) k *= 0.18;
      if (searchMatches && !searchMatches.has(id)) k *= 0.35;
      return k;
    }
    function edgeDim(ei, srcId, tgtId) {
      let k = 1.0;
      if (hoverEdgeSet && !hoverEdgeSet.has(ei)) k *= 0.15;
      if (searchMatches && !searchMatches.has(srcId) && !searchMatches.has(tgtId)) k *= 0.35;
      return k;
    }

    // Edges as curved quadratic Béziers
    let eHtml = '';
    edges.forEach((e, ei) => {
      const a = nodes[e.source], b = nodes[e.target];
      if (!nodeVisible(a) || !nodeVisible(b)) return;
      const pa = proj[e.source].p, pb = proj[e.target].p;
      const dxs = pb.sx - pa.sx, dys = pb.sy - pa.sy;
      const len = Math.sqrt(dxs*dxs + dys*dys) + 0.01;
      const nx = -dys / len, ny = dxs / len;
      const mx = (pa.sx + pb.sx) / 2 + nx * e._offset;
      const my = (pa.sy + pb.sy) / 2 + ny * e._offset;
      const avg = (pa.zcam + pb.zcam) / 2;
      const baseOp = Math.max(0.12, Math.min(0.78, 900 / avg));
      const op = baseOp * edgeDim(ei, e.source, e.target);
      const w = Math.max(0.5, Math.min(3.2,
                    0.9 * Math.min(pa.scale, pb.scale) *
                    (1 + Math.log(1 + e.count))));
      // Phase 48d — native SVG <title> gives a 1-line browser tooltip
      // on edge hover showing the source sentence (if any). Zero
      // runtime cost; no custom popover to maintain.
      const tip = (e.triples.find(t => t.source_sentence) || {}).source_sentence || '';
      const tipAttr = tip ? '<title>' + escapeHtml(tip) + '</title>' : '';
      eHtml += '<path class="kg-edge" data-ei="' + ei + '" ' +
               'd="M ' + pa.sx.toFixed(1) + ' ' + pa.sy.toFixed(1) +
               ' Q ' + mx.toFixed(1) + ' ' + my.toFixed(1) +
               ' ' + pb.sx.toFixed(1) + ' ' + pb.sy.toFixed(1) + '" ' +
               'stroke="' + edgeColor(e) + '" stroke-width="' + w.toFixed(2) +
               '" fill="none" opacity="' + op.toFixed(2) + '">' +
               tipAttr + '</path>';
      // Count badge for merged edges (3+ triples)
      if (e.count >= 3 && pa.scale > 0.5) {
        const bx = mx, by = my;
        eHtml += '<circle cx="' + bx.toFixed(1) + '" cy="' + by.toFixed(1) +
                 '" r="6" fill="' + theme.canvasBg +
                 '" stroke="' + edgeColor(e) + '" stroke-width="1" opacity="' +
                 (op * 1.3).toFixed(2) + '" pointer-events="none"/>';
        eHtml += '<text class="u-mono" x="' + bx.toFixed(1) + '" y="' + (by + 2.5).toFixed(1) +
                 '" text-anchor="middle" font-size="8" fill="' + theme.label +
                 '" pointer-events="none">' +
                 e.count + '</text>';
      }
    });
    edgeLayer.innerHTML = eHtml;

    // Nodes
    let nHtml = '';
    order.forEach(item => {
      const n = item.n, p = item.p;
      if (!nodeVisible(n)) return;
      const rBase = 4 + Math.sqrt(Math.min(n.degree, 25)) * 2;
      const r = Math.max(2, rBase * p.scale);
      const baseOp = Math.max(0.35, Math.min(1.0, 1200 / p.zcam));
      const op = baseOp * nodeDim(n.id);
      const fill = nodeFill(n);
      const isHover = (n.id === hoverId);
      const rEff = r * (isHover ? 1.25 : 1.0);
      nHtml += '<g class="kg-node" data-id="' + n.id + '" opacity="' + op.toFixed(2) + '">';
      nHtml += '<circle cx="' + p.sx.toFixed(1) + '" cy="' + p.sy.toFixed(1) +
               '" r="' + (rEff * 2.2).toFixed(2) + '" fill="' + fill +
               '" opacity="0.18" pointer-events="none"/>';
      nHtml += '<circle cx="' + p.sx.toFixed(1) + '" cy="' + p.sy.toFixed(1) +
               '" r="' + rEff.toFixed(2) + '" fill="' + fill +
               '" stroke="' + (isHover ? theme.label : theme.nodeStroke) +
               '" stroke-width="' + (isHover ? '1.6' : '0.7') + '"/>';
      if (p.scale > 0.45) {
        const fs = Math.max(8, 10.5 * p.scale) * labelScale;
        // Phase 54.6.10 — honor the font dropdown. Solid mode
        // skips the paint-order stroke so the label is one colour
        // (better contrast on bright themes).
        const stylePre = 'font-family:' + labelFontFamily + ';';
        const styleStroke = labelSolid ? ''
          : ('paint-order:stroke;stroke:' + theme.labelStroke
             + ';stroke-width:2.5px;');
        nHtml += '<text x="' + (p.sx + rEff + 3).toFixed(1) + '" y="' +
                 (p.sy + 3).toFixed(1) + '" font-size="' + fs.toFixed(1) +
                 '" fill="' + theme.label + '" pointer-events="none" ' +
                 'style="' + stylePre + styleStroke + '">' +
                 escapeHtml(nodeLabel(n.label)) + '</text>';
      }
      nHtml += '</g>';
    });
    nodeLayer.innerHTML = nHtml;
  }

  // Settle
  for (let i = 0; i < 120; i++) step();

  // Phase 48d — persist the post-settle layout for this filter so the
  // next open of the same filter warm-starts from it. Done once, a
  // few seconds after the main rAF loop starts (gives the live
  // physics a little more time to refine the cold cluster seeds
  // without blocking the first paint). Idempotent: re-saves on every
  // render would be wasteful; one shot is enough.
  let _kgLayoutSaved = false;
  setTimeout(() => {
    if (_kgLayoutSaved || !nodes.length) return;
    _kgSaveLayout(_kgLayoutKeyCurrent, nodes);
    _kgLayoutSaved = true;
  }, 2500);

  function loop() {
    if (!running) return;
    if (!paused) step();
    render();
    // Apply any pending shareable-URL state once the first frame is up
    if (window._kgPendingShare) {
      try { _kgApplyPendingShare(); } catch (e) {}
    }
    raf = requestAnimationFrame(loop);
  }
  loop();

  // ── Interaction: drag/orbit/wheel + hover dim + right-click menu ─
  let drag = null;
  // Convert a pointer event to local coordinates in the SVG's viewBox
  // space. Returns both the raw screen-space delta (`sx/sy`, used for
  // orbit sensitivity tuning) and viewBox-space coordinates (`x/y`,
  // used for node drag) so dragging a node stays under the cursor
  // even when the SVG is scaled (e.g. in fullscreen, where 1 screen
  // pixel ≠ 1 viewBox unit).
  function localPoint(evt) {
    const rect = svg.getBoundingClientRect();
    const sx = evt.clientX - rect.left - rect.width / 2;
    const sy = evt.clientY - rect.top - rect.height / 2;
    const scaleX = W / Math.max(rect.width, 1);
    const scaleY = H / Math.max(rect.height, 1);
    return { x: sx * scaleX, y: sy * scaleY, sx: sx, sy: sy };
  }
  function neighborsOf(id) {
    const ns = new Set([id]);
    const es = new Set();
    edges.forEach((e, ei) => {
      if (e.source === id) { ns.add(e.target); es.add(ei); }
      else if (e.target === id) { ns.add(e.source); es.add(ei); }
    });
    return { ns: ns, es: es };
  }

  svg.addEventListener('mousedown', (evt) => {
    if (evt.button !== 0) return;  // left-click only for drag
    _kgHideMenu();
    const pt = localPoint(evt);
    const el = evt.target.closest && evt.target.closest('.kg-node');
    if (el) {
      const id = parseInt(el.getAttribute('data-id'), 10);
      const n = nodes[id];
      const p = project(n);
      n.fixed = true;
      // startX/Y are viewBox-space; startSX/SY are screen-px — both
      // kept so orbit and node drag each use the right scale.
      drag = { mode: 'node', id: id,
                startX: pt.x, startY: pt.y,
                startSX: pt.sx, startSY: pt.sy,
                startWX: n.x, startWY: n.y, startWZ: n.z,
                startZcam: p.zcam, moved: false };
    } else {
      drag = { mode: 'orbit', startX: pt.x, startY: pt.y,
                startSX: pt.sx, startSY: pt.sy,
                startRotX: cam.rotX, startRotY: cam.rotY };
    }
    svg.classList.add('kg-grabbing');
    evt.preventDefault();
  });
  function onMove(evt) {
    if (!drag) return;
    const pt = localPoint(evt);
    if (drag.mode === 'orbit') {
      // Orbit uses screen-px delta so sensitivity stays consistent
      // regardless of how the SVG is scaled on screen.
      const dsx = pt.sx - drag.startSX, dsy = pt.sy - drag.startSY;
      cam.rotY = drag.startRotY + dsx * 0.006;
      cam.rotX = Math.max(-1.45, Math.min(1.45, drag.startRotX + dsy * 0.006));
    } else {
      // Node drag uses viewBox-space delta so the node stays exactly
      // under the cursor even when the SVG is scaled (fullscreen).
      const dx = pt.x - drag.startX, dy = pt.y - drag.startY;
      const dsx = pt.sx - drag.startSX, dsy = pt.sy - drag.startSY;
      if (Math.abs(dsx) + Math.abs(dsy) > 2) drag.moved = true;
      const d = worldDelta(dx, dy, drag.startZcam);
      const n = nodes[drag.id];
      n.x = drag.startWX + d.x;
      n.y = drag.startWY + d.y;
      n.z = drag.startWZ + d.z;
      n.vx = 0; n.vy = 0; n.vz = 0;
    }
  }
  function onUp() {
    if (!drag) return;
    const d = drag; drag = null;
    svg.classList.remove('kg-grabbing');
    if (d.mode === 'node') {
      const n = nodes[d.id];
      // If shift was held on mousedown we'd keep it pinned; but SVG
      // doesn't report modifier state at mouseup reliably, so we use
      // right-click → Pin for persistent pinning. Default mouseup
      // releases the node back to physics.
      n.fixed = false;
      if (!d.moved) {
        // Tap-center: tween the camera so this node is framed at origin
        tweenCenterOn(n);
      }
    }
  }
  window.addEventListener('mousemove', onMove);
  window.addEventListener('mouseup', onUp);

  // Hover: dim everything except node + 1-hop neighbors. Delegation
  // via mouseover/mouseout on svg + target.closest('.kg-node') —
  // works even though the DOM is rebuilt every frame, because the
  // listener is on the stable <svg>, not per-node.
  svg.addEventListener('mouseover', (evt) => {
    const el = evt.target.closest && evt.target.closest('.kg-node');
    if (!el) return;
    const id = parseInt(el.getAttribute('data-id'), 10);
    if (isNaN(id)) return;
    hoverId = id;
    const { ns, es } = neighborsOf(id);
    hoverNeighbors = ns;
    hoverEdgeSet = es;
  });
  svg.addEventListener('mouseout', (evt) => {
    const rel = evt.relatedTarget;
    if (!rel || !svg.contains(rel) ||
        !(rel.closest && rel.closest('.kg-node'))) {
      hoverId = -1; hoverNeighbors = null; hoverEdgeSet = null;
    }
  });

  // Right-click context menu. Actions differ for node / edge /
  // background (each branch ends with a `_kgShowMenu` call).
  svg.addEventListener('contextmenu', (evt) => {
    evt.preventDefault();
    const nodeEl = evt.target.closest && evt.target.closest('.kg-node');
    const edgeEl = evt.target.closest && evt.target.closest('.kg-edge');
    if (nodeEl) {
      const id = parseInt(nodeEl.getAttribute('data-id'), 10);
      const n = nodes[id];
      _kgShowMenu(evt.clientX, evt.clientY, [
        { label: 'Expand 1 hop', onClick: () => kgEgoExpand(n.label, 1) },
        { label: 'Expand 2 hops', onClick: () => kgEgoExpand(n.label, 2) },
        { label: (n.fixed ? 'Unpin' : 'Pin node'),
           onClick: () => { n.fixed = !n.fixed; } },
        { label: 'Center view', onClick: () => tweenCenterOn(n) },
        { label: 'Hide node', onClick: () => { n.hidden = true; } },
        '-',
        { label: 'Show in table',
           onClick: () => {
             document.getElementById('kg-subject').value = n.label;
             switchKgTab('kg-table');
             loadKg(0);
           } },
        { label: 'Copy label',
           onClick: () => { try { navigator.clipboard.writeText(n.label); } catch(e) {} } },
      ]);
      return;
    }
    if (edgeEl) {
      const ei = parseInt(edgeEl.getAttribute('data-ei'), 10);
      const e = edges[ei];
      const firstT = e.triples[0] || {};
      const title = (firstT.doc_title || '(unknown source)').substring(0, 70);
      const preds = Array.from(e.predicates).slice(0, 3).join(', ');
      // Pick the first non-empty source sentence across merged triples
      const sent = (e.triples.find(t => t.source_sentence)
                    || {}).source_sentence || '';
      const sentDisplay = sent
        ? ('\u201C' + (sent.length > 80 ? sent.substring(0, 80) + '\u2026' : sent) + '\u201D')
        : '(no source sentence — re-compile wiki to backfill)';
      _kgShowMenu(evt.clientX, evt.clientY, [
        { label: 'Source: ' + title, onClick: () => {} },
        { label: 'Predicates: ' + preds, onClick: () => {} },
        { label: sentDisplay, onClick: () => {} },
        '-',
        { label: sent ? 'Copy sentence' : 'Copy sentence (none)',
           onClick: () => {
             if (!sent) return;
             try { navigator.clipboard.writeText(sent); } catch (ex) {}
           } },
        { label: 'Copy triple',
           onClick: () => {
             const txt = nodes[e.source].label + '  [' + preds + ']  ' + nodes[e.target].label;
             try { navigator.clipboard.writeText(txt); } catch (ex) {}
           } },
        { label: 'Show this paper in table',
           onClick: () => {
             if (firstT.doc_id) {
               document.getElementById('kg-subject').value = '';
               document.getElementById('kg-object').value = '';
               kgSearchByDoc(firstT.doc_id);
             }
           } },
        { label: 'Filter by first predicate',
           onClick: () => {
             const p = (firstT.predicate || '').trim();
             if (!p) return;
             const sel = document.getElementById('kg-predicate');
             const has = Array.from(sel.options).some(o => o.value === p);
             if (!has) {
               const opt = document.createElement('option');
               opt.value = p; opt.textContent = p; sel.appendChild(opt);
             }
             sel.value = p; loadKg(0);
           } },
      ]);
      return;
    }
    // Background menu
    _kgShowMenu(evt.clientX, evt.clientY, [
      { label: (paused ? 'Resume physics' : 'Freeze physics'),
         hint: 'Space', onClick: () => { paused = !paused; } },
      { label: 'Reset view', onClick: () => { resetView(); } },
      { label: 'Unhide all nodes',
         onClick: () => { nodes.forEach(n => { n.hidden = false; }); } },
      '-',
      { label: 'Download PNG', onClick: () => downloadPng() },
    ]);
  });

  svg.addEventListener('wheel', (evt) => {
    evt.preventDefault();
    const factor = Math.exp(evt.deltaY * 0.0015);
    cam.dist = Math.max(250, Math.min(3000, cam.dist * factor));
  }, { passive: false });

  // Keyboard: Space toggles freeze; only when KG modal is open and
  // focus isn't in an input. Prevents the page from scrolling on Space.
  function onKey(evt) {
    const modal = document.getElementById('kg-modal');
    if (!modal || modal.style.display === 'none') return;
    const tag = (evt.target && evt.target.tagName) || '';
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    if (evt.code === 'Space') {
      paused = !paused; evt.preventDefault();
    }
  }
  window.addEventListener('keydown', onKey);

  // ── Camera tween: smoothly frame a node at origin ──────────────
  function tweenCenterOn(n) {
    const startRotX = cam.rotX, startRotY = cam.rotY, startDist = cam.dist;
    // Pick rotation so the node's world vector points toward -Z (into
    // screen), then back off to a comfortable distance. Clamp pitch so
    // we don't flip through the poles.
    const len = Math.sqrt(n.x*n.x + n.y*n.y + n.z*n.z) + 0.01;
    const targetRotY = Math.atan2(n.x, Math.max(n.z, 0.01));
    const targetRotX = Math.max(-1.4, Math.min(1.4,
                         -Math.atan2(n.y, Math.sqrt(n.x*n.x + n.z*n.z) + 0.01)));
    const targetDist = Math.max(400, Math.min(1200, len * 1.8 + 300));
    const t0 = performance.now();
    function frame(t) {
      const k = Math.min((t - t0) / 450, 1);
      const e = _kgEase(k);
      cam.rotX = startRotX + (targetRotX - startRotX) * e;
      cam.rotY = startRotY + (targetRotY - startRotY) * e;
      cam.dist = startDist + (targetDist - startDist) * e;
      if (k < 1) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  // ── Reset view ─────────────────────────────────────────────────
  function resetView() {
    cam.rotX = camDefault.rotX;
    cam.rotY = camDefault.rotY;
    cam.dist = camDefault.dist;
    nodes.forEach(n => { n.hidden = false; n.fixed = false; });
    searchMatches = null;
    const s = document.getElementById('kg-search');
    if (s) s.value = '';
  }

  // ── Search: live-highlight nodes whose label contains the query ─
  function doSearch(q) {
    q = (q || '').trim().toLowerCase();
    if (!q) { searchMatches = null; return; }
    const m = new Set();
    nodes.forEach(n => {
      if (n.label.toLowerCase().indexOf(q) !== -1) m.add(n.id);
    });
    searchMatches = m;
    // Auto-center on the first match
    for (const id of m) { tweenCenterOn(nodes[id]); break; }
  }

  // ── PNG export: serialize SVG → <img> → canvas → blob → download ─
  function downloadPng() {
    const xml = new XMLSerializer().serializeToString(svg);
    const svgBlob = new Blob(
      ['<?xml version="1.0" encoding="UTF-8"?>' + xml],
      { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(svgBlob);
    const img = new Image();
    const outW = svg.clientWidth || W;
    const outH = svg.clientHeight || H;
    img.onload = () => {
      const cvs = document.createElement('canvas');
      cvs.width = outW * 2; cvs.height = outH * 2;  // 2× for retina
      const ctx = cvs.getContext('2d');
      ctx.fillStyle = theme.canvasBg;
      ctx.fillRect(0, 0, cvs.width, cvs.height);
      ctx.drawImage(img, 0, 0, cvs.width, cvs.height);
      URL.revokeObjectURL(url);
      cvs.toBlob(b => {
        const a = document.createElement('a');
        a.href = URL.createObjectURL(b);
        a.download = 'knowledge-graph.png';
        document.body.appendChild(a); a.click(); a.remove();
      }, 'image/png');
    };
    img.onerror = () => { URL.revokeObjectURL(url); };
    img.src = url;
  }

  // ── Expose sim controls ────────────────────────────────────────
  canvas._kgSim = {
    stop: () => {
      running = false;
      if (raf) cancelAnimationFrame(raf);
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      window.removeEventListener('keydown', onKey);
    },
    setTheme: (name) => {
      if (name && KG_THEMES[name]) _kgActiveTheme = name;
      theme = _kgEffectiveTheme();
      _applyKgDefs(svg, theme);
      _applyKgClusterDefs(svg, theme, numClusters);
      canvas.style.background = theme.canvasBg;
    },
    refreshTheme: () => {
      // Called after custom-color overrides change OR after a preset
      // swap. Reads the effective (preset + overrides) theme and
      // repushes the gradient stops. No simulation restart; the
      // render loop picks up the new `theme` closure next frame.
      theme = _kgEffectiveTheme();
      _applyKgDefs(svg, theme);
      _applyKgClusterDefs(svg, theme, numClusters);
      canvas.style.background = theme.canvasBg;
    },
    togglePause: () => { paused = !paused; },
    setColorBy: (mode) => { if (mode) colorBy = mode; },
    setLabelScale: (v) => { labelScale = Math.max(0.4, Math.min(2.2, v)); },
    // Phase 54.6.10 — Font: maps the dropdown value to a
    // (family, solid) pair. Solid strips the label stroke so the
    // text is single-colour. The render loop reads these closure
    // vars on the next frame so no restart is needed.
    setFont: (key) => {
      const map = {
        'sans-halo':      ['var(--font-sans)',  false],
        'sans-solid':     ['var(--font-sans)',  true],
        'serif-solid':    ['var(--font-serif)', true],
        'serif-halo':     ['var(--font-serif)', false],
        'mono-solid':     ['var(--font-mono)',  true],
        'mono-halo':      ['var(--font-mono)',  false],
        'condensed-solid':['"Barlow Condensed","Arial Narrow",sans-serif', true],
        'display-solid':  ['"Playfair Display",Georgia,serif', true],
      };
      const pair = map[key] || map['sans-halo'];
      labelFontFamily = pair[0];
      labelSolid = pair[1];
    },
    setDegFilter: (v) => { degFilter = (v >= 99) ? 999 : v; },
    search: doSearch,
    resetView: resetView,
    centerOn: tweenCenterOn,
    downloadPng: downloadPng,
    // Phase 48d — return the subset of state that defines the visible
    // view (camera pose + which nodes are pinned). The share-URL
    // builder merges this with the theme + filter fields into the
    // compact base64 blob. We serialize pin by node label (not id)
    // because ids are per-filter-load and unstable.
    getShareState: () => ({
      c: { rx: cam.rotX, ry: cam.rotY, d: cam.dist },
      p: nodes.filter(n => n.fixed).map(n => n.label),
    }),
    applyShareState: (st) => {
      if (!st) return;
      if (st.c) {
        if (typeof st.c.rx === 'number') cam.rotX = st.c.rx;
        if (typeof st.c.ry === 'number') cam.rotY = st.c.ry;
        if (typeof st.c.d  === 'number') cam.dist = Math.max(250, Math.min(3000, st.c.d));
      }
      if (Array.isArray(st.p)) {
        const pinSet = new Set(st.p);
        nodes.forEach(n => { if (pinSet.has(n.label)) n.fixed = true; });
      }
    },
  };
}

// ── Module-scope KG helpers used by the context menu + toolbar ────

// Replace the current graph view with the ego network around a label.
// depth=1 → just the node's direct 1-hop (single /api/kg call using
// `any_side`). depth=2 → 1-hop + the 1-hop of each of the top-10
// most-frequent neighbors (parallel fetches, deduped, confidence-
// ranked, capped at 200 triples). Two hops is where scientific KGs
// get interesting — depth 1 is often just a claim, depth 2 is the
// surrounding context.
async function kgEgoExpand(label, depth) {
  depth = depth || 1;
  const status = document.getElementById('kg-status');
  try {
    if (status) status.textContent = 'Expanding around "' + label + '" (depth ' + depth + ')…';
    const r1 = await fetch('/api/kg?' + new URLSearchParams({ any_side: label, limit: 200 }));
    let all = ((await r1.json()).triples) || [];
    if (depth >= 2 && all.length) {
      // Count neighbor frequency to pick who to expand next
      const freq = new Map();
      all.forEach(t => {
        const s = (t.subject || '').substring(0, 60);
        const o = (t.object  || '').substring(0, 60);
        if (s.toLowerCase() !== label.toLowerCase()) {
          freq.set(s, (freq.get(s) || 0) + 1);
        }
        if (o.toLowerCase() !== label.toLowerCase()) {
          freq.set(o, (freq.get(o) || 0) + 1);
        }
      });
      const topN = Array.from(freq.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10)
        .map(entry => entry[0]);
      // Parallel fetch, 50 triples per neighbor (keeps response time sane)
      const batches = await Promise.all(topN.map(n =>
        fetch('/api/kg?' + new URLSearchParams({ any_side: n, limit: 50 }))
          .then(r => r.json()).then(d => d.triples || [])
          .catch(() => [])
      ));
      const seen = new Set();
      all.forEach(t => seen.add((t.subject || '') + '|' + (t.predicate || '') + '|' + (t.object || '')));
      batches.forEach(batch => {
        batch.forEach(t => {
          const k = (t.subject || '') + '|' + (t.predicate || '') + '|' + (t.object || '');
          if (!seen.has(k)) { all.push(t); seen.add(k); }
        });
      });
      // Confidence-sort, cap at 200 so the graph stays navigable
      all.sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
      all = all.slice(0, 200);
    }
    _kgTriples = all;
    if (status) status.textContent =
      'Expanded around "' + label + '" (depth ' + depth + ') · ' +
      all.length + ' triple(s)';
    _renderKgTable(all);
    _renderKgGraph(all.slice(0, 100));
  } catch (e) {
    if (status) status.textContent = 'Error: ' + e.message;
  }
}

// Fetch all triples for a given document and re-render. Used by the
// edge context menu's "Show this paper" action.
async function kgSearchByDoc(docId) {
  const params = new URLSearchParams({ document_id: docId, limit: 200 });
  try {
    const res = await fetch('/api/kg?' + params.toString());
    const data = await res.json();
    _kgTriples = data.triples || [];
    document.getElementById('kg-status').textContent =
      'Showing triples from selected paper · ' + _kgTriples.length;
    _renderKgTable(_kgTriples);
    _renderKgGraph(_kgTriples.slice(0, 100));
    switchKgTab('kg-graph');
  } catch (e) {
    document.getElementById('kg-status').textContent = 'Error: ' + e.message;
  }
}

// ── Phase 30: Export modal ────────────────────────────────────────────
function openExportModal() {
  openModal('export-modal');

  // Phase 31 — separate HTML and PDF buttons. PDF uses weasyprint
  // server-side; HTML is the printable browser-rendered version
  // (still useful as a fallback if weasyprint can't run).
  const exts = [
    { ext: 'pdf',  label: 'PDF' },
    { ext: 'html', label: 'HTML' },
    { ext: 'md',   label: 'Markdown' },
    { ext: 'txt',  label: 'Text' },
  ];
  function _btnHtml(base, enabled) {
    return exts.map(e => enabled
      ? '<a href="' + base + '.' + e.ext + '" target="_blank">' + e.label + '</a>'
      : '<a class="disabled">' + e.label + '</a>'
    ).join('');
  }

  // This section
  const sectionName = currentDraftId
    ? (document.getElementById('draft-title').textContent || 'current section')
    : null;
  document.getElementById('export-section-name').textContent =
    sectionName || '(no draft selected)';
  document.getElementById('export-section-btns').innerHTML =
    _btnHtml('/api/export/draft/' + currentDraftId, !!sectionName);

  // This chapter
  const ch = chaptersData.find(c => c.id === currentChapterId);
  const chName = ch ? ('Ch.' + ch.num + ': ' + ch.title) : null;
  document.getElementById('export-chapter-name').textContent =
    chName || '(no chapter selected)';
  document.getElementById('export-chapter-btns').innerHTML =
    _btnHtml('/api/export/chapter/' + currentChapterId, !!chName);

  // Whole book — always enabled
  document.getElementById('export-book-name').textContent =
    document.querySelector('.sidebar h2').textContent || 'Book';
  document.getElementById('export-book-btns').innerHTML =
    _btnHtml('/api/export/book', true);
}

async function refreshAfterJob(newDraftId) {
  // Reload sidebar data
  try {
    const res = await fetch('/api/chapters');
    const data = await res.json();
    rebuildSidebar(data.chapters, newDraftId || currentDraftId);
    document.getElementById('gaps-count').textContent = data.gaps_count;
  } catch(e) {}
  // Navigate to the new draft if one was created
  if (newDraftId) loadSection(newDraftId);
}

function rebuildSidebar(chapters, activeId) {
  const container = document.getElementById('sidebar-sections');
  // Phase 54.6.x — keep the chaptersData global in sync with what
  // the sidebar is actually rendering. Without this, a long-running
  // autowrite session (page loaded yesterday, autowrite ran overnight,
  // user clicks a section in the morning) leaves the global pinned
  // to the page-load snapshot — sections that finished overnight
  // still look "empty" to selectChapter / previewEmptySection /
  // showChapterEmptyState, even though the sidebar visibly shows
  // them as drafted. Result: clicking such a section opened the
  // chapter empty-state instead of loading the new draft.
  if (Array.isArray(chapters)) chaptersData = chapters;
  let html = '';
  chapters.forEach(ch => {
    const safeTitle = escapeHtml(ch.title || '');
    // Phase 54.6.309 — bibliography pseudo-chapter: no "Ch.N:" prefix,
    // no delete button, not draggable.
    const isBib = !!ch.is_bibliography;
    const extraCls = isBib ? ' ch-group--bibliography' : '';
    html += '<div class="ch-group' + extraCls + '" data-ch-id="' + ch.id + '">';
    // Phase 23 — chevron toggle at the start of the chapter title.
    // Phase 33 — chapter title is draggable for reordering (mirrors
    // section drag-drop from Phase 26 but operates on ch-group).
    const titleText = isBib ? safeTitle : ('Ch.' + ch.num + ': ' + safeTitle);
    const deleteBtn = isBib
      ? ''
      : '<span class="ch-actions"><button onclick="event.stopPropagation();deleteChapter(\'' + ch.id + '\')" title="Delete chapter">\u2717</button></span>';
    if (isBib) {
      // Non-draggable, synthetic pseudo-chapter.
      html += '<div class="ch-title clickable" ' +
        'data-ch-id="' + ch.id + '" ' +
        'onclick="selectChapter(this.parentElement)">' +
        '<button class="ch-toggle" ' +
        'onclick="event.stopPropagation();toggleChapter(this.closest(\'.ch-group\'))" ' +
        'title="Collapse or expand sections">\u25be</button>' +
        titleText + deleteBtn + '</div>';
    } else {
      // Regular chapter — draggable. Keep the literal markup
      // `ch-title clickable" draggable="true"` intact; the
      // l1_phase33_keyboard_shortcuts test greps for this substring.
      html += '<div class="ch-title clickable" draggable="true" ' +
        'data-ch-id="' + ch.id + '" ' +
        'ondragstart="chDragStart(event,\'' + ch.id + '\')" ' +
        'ondragover="chDragOver(event)" ' +
        'ondrop="chDrop(event,\'' + ch.id + '\')" ' +
        'ondragend="chDragEnd(event)" ' +
        'onclick="selectChapter(this.parentElement)">' +
        '<button class="ch-toggle" ' +
        'onclick="event.stopPropagation();toggleChapter(this.closest(\'.ch-group\'))" ' +
        'title="Collapse or expand sections">\u25be</button>' +
        titleText + deleteBtn + '</div>';
    }

    // Phase 21 — render the FULL section template, not just sections
    // with drafts. Empty slots become "Write" CTAs; orphan drafts
    // (whose section_type no longer matches a template slug) appear
    // at the end with a danger marker. This keeps the sidebar in
    // sync with whatever the chapter modal Section editor saves.
    const meta = Array.isArray(ch.sections_meta) ? ch.sections_meta : [];
    const titleBySlug = {};
    const planBySlug = {};
    meta.forEach(s => {
      titleBySlug[s.slug] = s.title;
      planBySlug[s.slug] = s.plan || '';
    });

    // Group existing drafts by slug
    const draftBySlug = {};
    ch.sections.forEach(sec => {
      const slug = (sec.type || '').toLowerCase();
      if (!draftBySlug[slug] || (sec.version > (draftBySlug[slug].version || 1))) {
        draftBySlug[slug] = sec;
      }
    });

    // Phase 22 — chapter completion progress bar (drafted / template).
    // Computed BEFORE rendering sections so it lives between the
    // chapter title and the section list, mirroring _render_sidebar.
    if (meta.length > 0) {
      let nDrafted = 0;
      meta.forEach(t => { if (draftBySlug[t.slug]) nDrafted += 1; });
      const nTotal = meta.length;
      const pct = Math.round(100 * nDrafted / nTotal);
      html += '<div class="ch-progress" title="' + nDrafted + ' of ' + nTotal + ' sections drafted">' +
        '<span class="ch-progress-bar"><span class="ch-progress-fill" style="width:' + pct + '%"></span></span>' +
        '<span class="ch-progress-label">' + nDrafted + '/' + nTotal + '</span>' +
        '</div>';
    }

    // 1) Render template slots in declared order (drafted or empty).
    const seenSlugs = new Set();
    if (meta.length > 0) {
      meta.forEach(tmpl => {
        seenSlugs.add(tmpl.slug);
        const draft = draftBySlug[tmpl.slug];
        const planAttr = escapeHtml((tmpl.plan || '').replace(/\n/g, ' ').slice(0, 200));
        const safeTmplTitle = escapeHtml(tmpl.title || '');
        // Phase 32.4 — inline delete button. The handler removes the
        // slug from sections_meta; if a draft exists, it becomes an
        // orphan in the sidebar (recoverable via the existing + adopt
        // button — fully reversible).
        // Phase 54.6.x \u2014 slugs can carry apostrophes
        // (`the_sun's_magnetic_shield`) and other punctuation. The
        // previous inline onclick interpolated the raw slug into a
        // single-quoted JS string inside a double-quoted HTML
        // attribute, so an apostrophe closed the JS literal early
        // and corrupted the surrounding markup, breaking neighbour
        // section links. Switched to data-action dispatch (browsers
        // escape data-* values automatically).
        const slugAttr = escapeHtml(tmpl.slug);
        const delBtn = '<button class="sec-delete-btn" ' +
          'data-action="delete-section" ' +
          'data-chapter-id="' + ch.id + '" ' +
          'data-sec-type="' + slugAttr + '" ' +
          'title="Remove this section from the chapter (draft becomes an orphan)">\u2717</button>';
        if (draft) {
          const active = draft.id === activeId ? 'active' : '';
          // Phase 26 — draggable for reordering
          html += '<a class="sec-link ' + active + '" href="/section/' + draft.id +
            '" draggable="true" data-draft-id="' + draft.id + '" ' +
            'data-section-slug="' + slugAttr + '" title="' + planAttr + '" ' +
            'onclick="return navTo(this)">' +
            '<span class="sec-status-dot drafted"></span>' +
            safeTmplTitle +
            ' <span class="meta">v' + draft.version + ' \u00b7 ' + draft.words + 'w</span>' +
            delBtn + '</a>';
        } else {
          // Phase 29 — preview-on-click instead of immediate doWrite()
          // Phase 42 — data-action dispatch (preview-empty-section).
          html += '<div class="sec-link sec-empty" draggable="true" ' +
            'data-section-slug="' + slugAttr + '" ' +
            'title="' + planAttr + '" ' +
            'data-action="preview-empty-section" ' +
            'data-chapter-id="' + ch.id + '" data-sec-type="' + slugAttr + '">' +
            '<span class="sec-status-dot empty"></span>' +
            safeTmplTitle +
            ' <span class="meta">empty \u00b7 \u270e</span>' +
            delBtn + '</div>';
        }
      });
    }

    // 2) Render orphan drafts (drafts whose slug isn't in the template).
    Object.keys(draftBySlug).forEach(slug => {
      if (seenSlugs.has(slug)) return;
      const draft = draftBySlug[slug];
      const display = escapeHtml(draft.title || (slug.charAt(0).toUpperCase() + slug.slice(1)));
      // Phase 22 — inline X button to delete the orphan
      // Phase 25 — also "+" button to adopt the slug into sections
      // Phase 54.6.x — same apostrophe-in-slug fix as the template
      // path above: switched to data-action dispatch.
      const orphanSlugAttr = escapeHtml(slug);
      html += '<a class="sec-link sec-orphan" href="/section/' + draft.id +
        '" data-draft-id="' + draft.id + '" onclick="return navTo(this)" ' +
        'title="Orphan draft. Click to inspect, + to adopt into sections, X to delete.">' +
        '<span class="sec-status-dot orphan"></span>' +
        display +
        ' <span class="meta">orphan \u00b7 v' + draft.version + ' \u00b7 ' + draft.words + 'w</span>' +
        '<button class="sec-orphan-adopt" ' +
        'data-action="adopt-orphan-section" ' +
        'data-chapter-id="' + ch.id + '" ' +
        'data-sec-type="' + orphanSlugAttr + '" ' +
        'title="Add this section_type to the chapter sections list">+</button>' +
        '<button class="sec-orphan-delete" ' +
        'data-action="delete-orphan-draft" ' +
        'data-draft-id="' + draft.id + '" ' +
        'title="Delete this orphan draft permanently">\u2717</button>' +
        '</a>';
    });

    if (meta.length === 0 && Object.keys(draftBySlug).length === 0) {
      html += '<div class="sec-link sec-empty-cta" data-action="start-writing-chapter" data-chapter-id="' + ch.id + '">\u270e Start writing</div>';
    }
    // Phase 32.4 — "+ Add section" CTA at the bottom of every
    // chapter's section list. Click → prompt for a title → POST
    // a new section dict via PUT /api/chapters/{id}/sections.
    html += '<div class="sec-link sec-add-cta" ' +
      'data-action="add-section-to-chapter" data-chapter-id="' + ch.id + '" ' +
      'title="Add a new section to this chapter">+ Add section</div>';
    html += '</div>';
  });
  container.innerHTML = html;
  // Phase 23 — re-apply collapsed state after rebuilding the DOM.
  restoreCollapsedChapters();
}

// ── Action handlers ───────────────────────────────────────────────────────
async function doWrite() {
  if (!currentChapterId) {
    showEmptyHint('Select a chapter from the sidebar first &mdash; click any chapter title in the left panel, then come back and click Write.');
    return;
  }
  const section = currentSectionType || 'introduction';
  showStreamPanel('Writing ' + section + '...');
  // Phase 15.6 — clear the read-view and prepare it for live writing
  startLiveWrite();

  const fd = new FormData();
  fd.append('chapter_id', currentChapterId);
  fd.append('section_type', section);
  const res = await fetch('/api/write', {method: 'POST', body: fd});
  const data = await res.json();
  startStream(data.job_id);
  // Phase 30 — persistent task bar
  startGlobalJob(data.job_id, {
    type: 'write',
    taskDesc: 'Writing ' + section,
    modelName: 'qwen3.5:27b',
    sectionType: section,
    chapterId: currentChapterId,
  });
}

async function doReview() {
  if (!currentDraftId) { showEmptyHint("No draft selected &mdash; click a section in the sidebar, or click <strong>Start writing</strong> under any chapter to create a first draft."); return; }
  showStreamPanel('Reviewing...');

  const fd = new FormData();
  const res = await fetch('/api/review/' + currentDraftId, {method: 'POST', body: fd});
  const data = await res.json();
  // Phase 30 — persistent task bar
  startGlobalJob(data.job_id, {
    type: 'review',
    taskDesc: 'Reviewing draft',
    modelName: 'qwen3.5:27b',
  });
  startStream(data.job_id);
}

async function doRevise() {
  if (!currentDraftId) { showEmptyHint("No draft selected &mdash; click a section in the sidebar, or click <strong>Start writing</strong> under any chapter to create a first draft."); return; }
  const instruction = prompt('Revision instruction (leave empty to use review feedback):');
  if (instruction === null) return;  // cancelled
  showStreamPanel('Revising...');

  const fd = new FormData();
  fd.append('instruction', instruction);
  const res = await fetch('/api/revise/' + currentDraftId, {method: 'POST', body: fd});
  const data = await res.json();
  startStream(data.job_id);
}

async function doGaps() {
  // Phase 54.6.14 — let the user pick a brainstorming method to steer
  // the gap analysis (Reverse Brainstorming, Five Whys, Scope Boundaries,
  // Missing Control, etc.). Loads the catalogue on demand.
  const methods = await _loadMethodsOnce('brainstorming');
  const names = [''].concat(methods.map(m => m.name));
  const prompt = 'Gap-analysis method (Enter to skip, or type one):\n\n'
    + methods.slice(0, 20).map((m, i) => (i + 1) + '. ' + m.name).join('\n')
    + '\n\nType the method NAME or NUMBER (blank = default):';
  let pick = window.prompt(prompt, '');
  if (pick === null) return;  // cancelled
  pick = (pick || '').trim();
  // Number → name
  const n = parseInt(pick, 10);
  if (!isNaN(n) && n >= 1 && n <= methods.length) pick = methods[n - 1].name;
  showStreamPanel('Analysing gaps' + (pick ? ' via ' + pick : '') + '…');
  const fd = new FormData();
  if (pick) fd.append('method', pick);
  const res = await fetch('/api/gaps', {method: 'POST', body: fd});
  const data = await res.json();
  startStream(data.job_id);
}

// Shared helper: fetch + cache the methods catalogue.
async function _loadMethodsOnce(kind) {
  window._methodsCache = window._methodsCache || {};
  if (window._methodsCache[kind]) return window._methodsCache[kind];
  try {
    const r = await fetch('/api/methods?kind=' + encodeURIComponent(kind));
    const d = await r.json();
    window._methodsCache[kind] = d.methods || [];
  } catch (_) {
    window._methodsCache[kind] = [];
  }
  return window._methodsCache[kind];
}

// Populate a <select> dropdown with methods grouped by category.
async function _populateMethodSelect(selectId, kind) {
  const sel = document.getElementById(selectId);
  if (!sel || sel.dataset.populated) return;
  const methods = await _loadMethodsOnce(kind);
  // Group by category to build <optgroup>s.
  const byCat = {};
  methods.forEach(m => {
    (byCat[m.category] = byCat[m.category] || []).push(m);
  });
  // Keep the (default) option as the first child.
  for (const cat of Object.keys(byCat).sort()) {
    const og = document.createElement('optgroup');
    og.label = cat;
    for (const m of byCat[cat]) {
      const opt = document.createElement('option');
      opt.value = m.name;
      opt.textContent = m.name;
      opt.title = m.description;
      og.appendChild(opt);
    }
    sel.appendChild(og);
  }
  sel.dataset.populated = '1';
}

// ── Dashboard ─────────────────────────────────────────────────────────────
async function showDashboard() {
  const [res, statsRes] = await Promise.all([
    fetch('/api/dashboard'),
    fetch('/api/stats').catch(() => null),
  ]);
  const data = await res.json();
  const corpusStats = statsRes ? await statsRes.json().catch(() => null) : null;
  const s = data.stats;

  let html = '<div class="dashboard">';
  html += '<h2>Book Dashboard</h2>';

  // Phase 54.6.98 — dashboard hint: make clear where the per-draft AI
  // actions live. The toolbar (Edit/Autowrite/Write/Review/Revise +
  // Verify/Critique/Extras dropdowns) is hidden on the Dashboard, so
  // first-time users looking for "AI Review" see only Outline (Plan
  // modal) and wonder where Review went.
  html += '<div style="margin:4px 0 18px;padding:10px 14px;background:var(--bg-alt);border-left:3px solid var(--link);border-radius:4px;font-size:12px;line-height:1.5;">'
    + '<strong>AI actions &mdash; where to find them:</strong><br>'
    + '&bull; <strong>Outline</strong> &amp; <strong>Book plan</strong> live in the <em>Plan</em> modal (top nav).<br>'
    + '&bull; <strong>AI Write / Autowrite / Review / Revise / Verify / Align / Argue / Gaps / &hellip;</strong> appear in the '
    + '<em>draft toolbar</em> &mdash; click any section in the sidebar to open its draft and the toolbar shows up. '
    + 'Click the <strong>?</strong> button at the end of the toolbar for a full reference, or read '
    + '<a href="https://github.com/claudenstein/sciknow/blob/main/docs/BOOK_ACTIONS.md" target="_blank" style="color:var(--link);">docs/BOOK_ACTIONS.md</a>.'
    + '</div>';

  // Book stats — modernized stat-tile cards
  html += '<div class="stat-grid">';
  html += '<div class="stat-tile"><div class="num">' + s.total_words.toLocaleString() + '</div><div class="lbl">Words</div></div>';
  html += '<div class="stat-tile"><div class="num">' + s.chapters + '</div><div class="lbl">Chapters</div></div>';
  html += '<div class="stat-tile"><div class="num">' + s.drafts + '</div><div class="lbl">Drafts</div></div>';
  html += '<div class="stat-tile"><div class="num">' + s.gaps_open + '</div><div class="lbl">Open Gaps</div></div>';
  html += '<div class="stat-tile"><div class="num">' + s.comments + '</div><div class="lbl">Comments</div></div>';
  html += '</div>';

  // Phase 35 — Total Compute panel. Aggregates every LLM-backed job
  // (write/review/revise/argue/gaps/autowrite/plan/...) from the
  // llm_usage_log ledger so the user can see total GPU compute spent
  // on the book. Per-operation breakdown appears as chips below the
  // three headline tiles. Autowrite is a strict subset — its detail
  // panel follows directly below.
  const tc = data.total_compute || {};
  const fmtTokens = (n) => (n || 0) >= 1000
    ? ((n / 1000).toFixed(1) + 'K')
    : String(n || 0);
  const fmtSecs = (secs) => {
    secs = Math.round(secs || 0);
    if (secs < 60) return secs + 's';
    if (secs < 3600) return Math.floor(secs / 60) + 'm ' + (secs % 60) + 's';
    return Math.floor(secs / 3600) + 'h ' + Math.floor((secs % 3600) / 60) + 'm';
  };
  if ((tc.total_jobs || 0) > 0) {
    html += '<h3 class="u-heading-section u-lg u-semibold u-muted u-upper u-ls-sm">Total Compute</h3>';
    html += '<div class="stat-grid">';
    html += '<div class="stat-tile"><div class="num">' + fmtTokens(tc.total_tokens) + '</div><div class="lbl">Total Tokens</div></div>';
    html += '<div class="stat-tile"><div class="num">' + fmtSecs(tc.total_seconds) + '</div><div class="lbl">Total Time</div></div>';
    html += '<div class="stat-tile"><div class="num">' + (tc.total_jobs || 0) + '</div><div class="lbl">LLM Jobs</div></div>';
    html += '</div>';
    // Per-operation breakdown as compact chips (only ops that actually ran)
    if (Array.isArray(tc.by_operation) && tc.by_operation.length > 0) {
      html += '<div class="u-mt-2 u-flex-raw u-wrap u-gap-6">';
      tc.by_operation.forEach(o => {
        html += '<span style="display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:12px;background:var(--bg-alt,#f3f4f6);font-size:12px;color:var(--fg-muted);">';
        html += '<strong style="color:var(--fg);">' + (o.operation || '?') + '</strong>';
        html += '<span>' + fmtTokens(o.tokens) + ' tok</span>';
        html += '<span>' + fmtSecs(o.seconds) + '</span>';
        html += '<span>×' + (o.jobs || 0) + '</span>';
        html += '</span>';
      });
      html += '</div>';
    }
  }

  // Phase 33 — Autowrite effort stats from the Layer 0 telemetry tables.
  // Shows cumulative token usage + time spent across all completed runs.
  const aw = data.autowrite_stats || {};
  if (aw.total_runs > 0) {
    html += '<h3 class="u-heading-section u-lg u-semibold u-muted u-upper u-ls-sm">Autowrite Effort</h3>';
    html += '<div class="stat-grid">';
    html += '<div class="stat-tile"><div class="num">' + (aw.total_runs || 0) + '</div><div class="lbl">Runs</div></div>';
    const tokStr = (aw.total_tokens || 0) >= 1000
      ? ((aw.total_tokens / 1000).toFixed(1) + 'K')
      : String(aw.total_tokens || 0);
    html += '<div class="stat-tile"><div class="num">' + tokStr + '</div><div class="lbl">Tokens Used</div></div>';
    // Format total_seconds as hours + minutes
    const secs = aw.total_seconds || 0;
    let timeStr;
    if (secs < 60) timeStr = secs + 's';
    else if (secs < 3600) timeStr = Math.floor(secs / 60) + 'm ' + (secs % 60) + 's';
    else timeStr = Math.floor(secs / 3600) + 'h ' + Math.floor((secs % 3600) / 60) + 'm';
    html += '<div class="stat-tile"><div class="num">' + timeStr + '</div><div class="lbl">Time Spent</div></div>';
    html += '</div>';
  }

  // Phase 14 — Corpus stats panel (mirrors `db stats` + RAPTOR + topics)
  if (corpusStats) {
    html += '<h3 class="u-heading-section u-lg u-semibold u-muted u-upper u-ls-sm">Corpus</h3>';
    html += '<div class="stat-grid">';
    html += '<div class="stat-tile"><div class="num">' + (corpusStats.n_documents || 0).toLocaleString() + '</div><div class="lbl">Documents</div>';
    if (corpusStats.n_completed != null) {
      html += '<div class="sub">' + corpusStats.n_completed.toLocaleString() + ' complete</div>';
    }
    html += '</div>';
    html += '<div class="stat-tile"><div class="num">' + (corpusStats.n_chunks || 0).toLocaleString() + '</div><div class="lbl">Chunks</div></div>';
    html += '<div class="stat-tile"><div class="num">' + (corpusStats.n_citations || 0).toLocaleString() + '</div><div class="lbl">Citations</div></div>';
    if (corpusStats.n_wiki_pages) {
      html += '<div class="stat-tile"><div class="num">' + corpusStats.n_wiki_pages.toLocaleString() + '</div><div class="lbl">Wiki Pages</div></div>';
    }
    if (corpusStats.topic_clusters && corpusStats.topic_clusters.length) {
      html += '<div class="stat-tile"><div class="num">' + corpusStats.topic_clusters.length + '</div><div class="lbl">Topic Clusters</div>';
      html += '<div class="sub">' + corpusStats.topic_clusters[0].name + ' (' + corpusStats.topic_clusters[0].n + ')</div></div>';
    }
    if (corpusStats.raptor_levels && Object.keys(corpusStats.raptor_levels).length) {
      const totalNodes = Object.values(corpusStats.raptor_levels).reduce((a, b) => a + b, 0);
      html += '<div class="stat-tile"><div class="num">' + totalNodes.toLocaleString() + '</div><div class="lbl">RAPTOR Nodes</div>';
      html += '<div class="raptor-bar">';
      Object.entries(corpusStats.raptor_levels).forEach(([lvl, n]) => {
        html += '<div class="raptor-lvl"><strong>' + n.toLocaleString() + '</strong>' + lvl + '</div>';
      });
      html += '</div></div>';
    } else {
      html += '<div class="stat-tile" style="opacity:0.6;"><div class="num">—</div><div class="lbl">RAPTOR</div><div class="sub">Not built</div></div>';
    }
    html += '</div>';
  }

  // Heatmap — Phase 14.4 — book-style section columns from real data
  // Phase 14.5 — heatmap header now includes a Plan link so the leitmotiv
  // is one click away from the dashboard.
  html += '<div class="heatmap-header">';
  html += '<h3>Completion Heatmap</h3>';
  html += '<button class="btn-link" onclick="openPlanModal()" title="View, edit, or regenerate the book plan (the leitmotiv)">&#128221; Book Plan</button>';
  html += '</div>';
  html += '<p class="u-note-xs">Click a chapter title to edit its scope. Click an empty cell to preview that section. Click a filled cell to open the draft. Hover any cell to see the section title.</p>';
  // Phase 30 — columns are POSITIONAL (1, 2, 3, ...) up to max(N) across
  // all chapters. Each chapter shows its actual sections in order;
  // chapters with fewer sections get blank "absent" cells in the extra
  // slots so the table is rectangular.
  const nCols = data.n_columns || 1;
  html += '<table class="heatmap"><thead><tr><th></th>';
  for (let i = 1; i <= nCols; i++) {
    html += '<th title="Section position ' + i + '">' + i + '</th>';
  }
  html += '</tr></thead><tbody>';
  data.heatmap.forEach(row => {
    // Phase 42 — data-action dispatch for the heatmap's clickable cells.
    html += '<tr><td class="ch-label clickable" data-action="open-chapter-modal" data-chapter-id="' + row.id + '" title="Click to edit chapter title and scope">';
    html += '<span class="ch-label-num">Ch.' + row.num + '</span> ' + escapeHtml((row.title || '').substring(0, 36));
    html += ' <span class="ch-label-edit">&#9881;</span></td>';
    row.cells.forEach((cell, idx) => {
      const posLabel = (idx + 1) + '. ' + (cell.title || '');
      if (cell.status === 'absent') {
        // This chapter has fewer sections than max(N) — render blank
        html += '<td><span class="hm-cell absent" title="(no section ' + (idx + 1) + ' in this chapter)">·</span></td>';
      } else if (cell.status === 'empty') {
        // Empty template slot — Phase 29 click-to-preview
        html += '<td><span class="hm-cell empty" ' +
          'data-action="preview-empty-section" ' +
          'data-chapter-id="' + row.id + '" data-sec-type="' + cell.type + '" ' +
          'title="' + escapeHtml(posLabel) + ' (empty — click to preview)">+</span></td>';
      } else {
        const label = 'v' + cell.version + ' ' + cell.words + 'w';
        html += '<td><span class="hm-cell ' + cell.status + '" ' +
          'data-action="load-section" data-draft-id="' + cell.draft_id + '" ' +
          'title="' + escapeHtml(posLabel) + ' &mdash; ' + label + '">' + label + '</span></td>';
      }
    });
    html += '</tr>';
  });
  html += '</tbody></table>';

  // Gaps
  if (data.gaps.length > 0) {
    // Phase 54.6.5 — Auto-expand corpus from all open topic/evidence gaps.
    const openExpandable = data.gaps.filter(
      g => g.type === 'topic' || g.type === 'evidence'
    ).length;
    const autoBtn = (openExpandable > 0)
      ? '<button class="btn-primary u-ml-auto" '
        + 'onclick="openAutoExpandPreview()" '
        + 'title="Query OpenAlex once per open topic/evidence gap; merge + rank candidates so you can cherry-pick which to ingest. Mirrors `sciknow book auto-expand`.">'
        + '&#128269; Auto-expand from these gaps</button>'
      : '';
    html += '<div class="u-flex-raw u-ai-center u-gap-2 u-mb-2">'
      + '<h3 class="u-m-0">Open Gaps</h3>'
      + autoBtn
      + '</div><div class="gap-list">';
    data.gaps.forEach(g => {
      let btn = '';
      if (g.type === 'draft' && g.chapter_num) {
        // Phase 42 — data-action dispatch; chapter_num is a numeric
        // attr, parsed back via parseInt in the handler.
        btn = '<button data-action="write-for-gap" data-chapter-num="' + g.chapter_num + '" title="Open this gap chapter + start a write job scoped to its topic.">Write</button>';
      } else if (g.type === 'topic' || g.type === 'evidence') {
        // Phase 54.6.5 — replace the old "Run: sciknow db expand" alert
        // with a real one-click flow: prefill the Topic-search subtab
        // with this gap's description and open the preview modal.
        btn = '<button data-action="expand-single-gap" '
          + 'data-gap-desc="' + escapeHtml(g.description) + '" '
          + 'title="Preview candidates for this specific gap via Topic search.">Expand</button>';
      }
      html += '<div class="gap-item">';
      html += '<span class="gap-type">' + g.type + '</span>';
      html += '<span class="gap-desc">' + (g.chapter_num ? 'Ch.' + g.chapter_num + ': ' : '') + g.description.substring(0, 120) + '</span>';
      html += btn + '</div>';
    });
    html += '</div>';
  }

  html += '</div>';

  // Show dashboard, hide section view
  document.getElementById('dashboard-view').innerHTML = html;
  document.getElementById('dashboard-view').style.display = 'block';
  document.getElementById('read-view').style.display = 'none';
  document.getElementById('edit-view').style.display = 'none';
  document.getElementById('draft-title').textContent = 'Dashboard';
  document.getElementById('draft-subtitle').style.display = 'none';
  document.getElementById('stream-panel').style.display = 'none';
  document.getElementById('version-panel').style.display = 'none';

  // Clear sidebar active
  document.querySelectorAll('.sec-link').forEach(l => l.classList.remove('active'));
  history.pushState({dashboard: true}, '', '/');
}

function writeForCell(chapterId, sectionType) {
  currentChapterId = chapterId;
  currentSectionType = sectionType;
  // Hide dashboard, show section view
  document.getElementById('dashboard-view').style.display = 'none';
  document.getElementById('read-view').style.display = 'block';
  document.getElementById('read-view').innerHTML = '<p class="u-dim">Starting write...</p>';
  document.getElementById('draft-subtitle').style.display = 'block';
  doWrite();
}

// Phase 29 — clicking an empty section row in the sidebar now SHOWS
// a preview placeholder in the read-view instead of immediately
// triggering doWrite(). The user can:
//   - read the section title + plan + target words
//   - click "Start writing" to fire doWrite (single section)
//   - click "Autowrite" to fire doAutowrite (with iterations)
//   - click another section in the sidebar to navigate away
// All without an LLM call happening accidentally on a single click.
async function previewEmptySection(chapterId, sectionType) {
  currentChapterId = chapterId;
  currentSectionType = sectionType;

  // Phase 54.6.x — re-validate that the section is actually empty.
  // The sidebar can lag behind the database when autowrite ran in
  // the background and the in-memory chaptersData snapshot is stale
  // (e.g. tab opened yesterday, autowrite finished overnight). If
  // the slug now has a draft, route to loadSection instead so the
  // user sees the real content instead of the "Start writing" stub.
  try {
    const r = await fetch('/api/chapters');
    if (r.ok) {
      const fresh = await r.json();
      const freshChapters = fresh.chapters || fresh;
      if (Array.isArray(freshChapters)) {
        chaptersData = freshChapters;
        const fch = freshChapters.find(c => c.id === chapterId);
        if (fch && Array.isArray(fch.sections)) {
          const target = (sectionType || '').toLowerCase();
          const draft = fch.sections.find(
            d => (d.type || '').toLowerCase() === target
          );
          if (draft && draft.id) {
            rebuildSidebar(freshChapters, draft.id);
            loadSection(draft.id);
            return;
          }
        }
      }
    }
  } catch (_) { /* fall through to the normal empty-state preview */ }

  // Look up the section meta from the in-memory chapters cache
  const ch = chaptersData.find(c => c.id === chapterId);
  if (!ch) {
    showEmptyHint('Chapter not found in cache.');
    return;
  }
  const meta = Array.isArray(ch.sections_meta) ? ch.sections_meta : [];
  const sec = meta.find(s => s.slug === sectionType);
  const sectionTitle = sec ? sec.title : (sectionType.charAt(0).toUpperCase() + sectionType.slice(1));
  const sectionPlan = sec ? (sec.plan || '') : '';
  const sectionTarget = sec && sec.target_words && sec.target_words > 0
    ? sec.target_words
    : (window._chapterWordTarget && Math.floor(window._chapterWordTarget / Math.max(1, meta.length)));

  // Switch to read-view (hide dashboard if it's showing)
  document.getElementById('dashboard-view').style.display = 'none';
  document.getElementById('read-view').style.display = 'block';
  document.getElementById('draft-subtitle').style.display = 'block';
  document.getElementById('edit-view').style.display = 'none';

  // Update the title bar to reflect the section
  document.getElementById('draft-title').textContent =
    'Ch.' + ch.num + ': ' + ch.title + ' \u2014 ' + sectionTitle;
  document.getElementById('draft-version').textContent = '0';
  document.getElementById('draft-words').textContent = '0';
  updateWordTargetBar(0, sectionTarget);

  // Clear any active state on other section rows + highlight the
  // empty row we just clicked.
  document.querySelectorAll('.sec-link').forEach(l => l.classList.remove('active'));
  const link = document.querySelector(
    '.sec-link.sec-empty[data-section-slug="' + sectionType + '"]'
  );
  if (link) link.classList.add('active');

  // Render the preview content into the read-view
  const planHtml = sectionPlan
    ? '<div style="margin:16px 0;padding:12px 16px;background:var(--toolbar-bg);' +
      'border-left:3px solid var(--accent);border-radius:4px;">' +
      '<div style="font-size:11px;color:var(--fg-muted);text-transform:uppercase;' +
      'letter-spacing:0.04em;margin-bottom:6px;">Section plan</div>' +
      '<div style="font-family:var(--font-serif);font-size:14px;line-height:1.6;' +
      'white-space:pre-wrap;">' + escapeHtml(sectionPlan) + '</div></div>'
    : '<div style="margin:16px 0;font-size:13px;color:var(--fg-muted);font-style:italic;">' +
      'No section plan set yet. Open the chapter modal (\u2699 icon) and add one in the Sections tab.</div>';

  const html =
    '<div class="empty-section-preview">' +
    '<div class="u-note-sm">' +
    '<span class="sec-status-dot empty"></span> empty section &middot; ' +
    'target ~' + (sectionTarget || 'auto') + ' words' +
    '</div>' +
    planHtml +
    '<div style="margin-top:24px;display:flex;gap:8px;flex-wrap:wrap;">' +
    '<button class="btn-primary" onclick="doWrite()" title="Write one draft: retrieve → generate → persist. Stops after a single pass.">\u270e Start writing</button>' +
    '<button class="btn-secondary" onclick="doAutowrite()" title="Write, score, and iterate (auto-revise) until the scorer + claim verifier pass thresholds. Configurable in Autowrite settings.">\u26a1 Autowrite (with iterations)</button>' +
    '</div>' +
    '<p style="margin-top:24px;font-size:12px;color:var(--fg-muted);">' +
    'This section has no draft yet. Click <strong>Start writing</strong> for a single ' +
    'pass, or <strong>Autowrite</strong> to run the score &rarr; verify &rarr; revise loop.' +
    ' You can also reorder this slot in the sidebar by dragging it.' +
    '</p>' +
    '</div>';
  document.getElementById('read-view').innerHTML = html;

  // Update URL so refresh keeps the preview state
  history.pushState({previewSection: sectionType, chapterId: chapterId},
    '', '/');
}

function writeForGap(chapterNum) {
  // Find the chapter ID from chaptersData
  const ch = chaptersData.find(c => c.num === chapterNum);
  if (ch) writeForCell(ch.id, 'introduction');
}


// ── Version History ───────────────────────────────────────────────────────
let versionData = [];
let selectedVersions = [];

async function showVersions() {
  if (!currentDraftId) { showEmptyHint("No draft selected &mdash; click a section in the sidebar, or click <strong>Start writing</strong> under any chapter to create a first draft."); return; }
  const res = await fetch('/api/versions/' + currentDraftId);
  const data = await res.json();
  versionData = data.versions;

  if (versionData.length < 1) { alert('No version history.'); return; }

  const panel = document.getElementById('version-panel');
  const timeline = document.getElementById('version-timeline');
  const diffView = document.getElementById('diff-view');
  panel.style.display = 'block';
  diffView.innerHTML = '<p class="u-dim">Click a version to preview. Select two to compare. Use <strong>Make active</strong> to pin the version the reader displays.</p>';
  selectedVersions = [];

  // Phase 54.6.314 — version rows show date/time + editable
  // description. Each row is a two-line layout: the badge/name/meta
  // cluster on top, the date + description underneath. Clicking the
  // badge selects the version (single → navigate, two → diff).
  // Clicking the name renames it; clicking the description edits it.
  let html = '<div class="version-list">';
  versionData.forEach(v => {
    const review = v.has_review ? ' <span class="version-tag" title="Has review feedback">rev\u2713</span>' : '';
    const active = v.is_active ? ' <span class="version-tag version-tag--active" title="Currently displayed in the reader.">active</span>' : '';
    const score = (v.final_overall != null)
      ? '<span class="version-score" title="Autowrite final_overall score (0–1).">' + (v.final_overall).toFixed(2) + '</span>'
      : '<span class="version-score version-score--na" title="Not autowrite-scored.">—</span>';
    const model = v.model_used ? ' <span class="version-tag" title="Writer model">' + escapeHtml(v.model_used) + '</span>' : '';
    const makeActiveBtn = v.is_active
      ? ''
      : '<button class="btn-sm" data-action="activate-version" data-draft-id="' + v.id + '" title="Make this the version shown by the reader. Global citation numbering + right-panel sources refresh to match.">Make active</button>';
    const name = v.version_name
      ? '<span class="version-name" title="User-supplied version label. Click to rename.">' + escapeHtml(v.version_name) + '</span>'
      : '<span class="version-name version-name--empty" title="No name assigned. Click to add.">(unnamed)</span>';
    const stamp = _formatVersionStamp(v.created_at, v.updated_at);
    const description = v.version_description
      ? '<span class="version-desc" title="User-supplied short description. Click to edit.">' + escapeHtml(v.version_description) + '</span>'
      : '<span class="version-desc version-desc--empty" title="No description. Click to add.">(add a short description…)</span>';
    html += '<div class="version-row' + (v.is_active ? ' version-row--active' : '') + '">' +
      '<div class="version-row-top">' +
        '<span class="version-badge" data-vid="' + v.id + '" data-action="select-version" data-version-id="' + v.id + '" title="Click to navigate to this version. Click a second version to diff the two.">' +
          'v' + v.version + '</span>' +
        '<span onclick="renameVersion(\'' + v.id + '\')" style="cursor:pointer;">' + name + '</span>' +
        '<span class="version-meta">' + (v.word_count || 0) + 'w' + review + active + model + '</span>' +
        score +
        makeActiveBtn +
      '</div>' +
      '<div class="version-row-meta">' +
        '<span class="version-stamp" title="Created — last-modified timestamps (local time).">' + stamp + '</span>' +
        '<span class="version-desc-wrap" onclick="editVersionDescription(\'' + v.id + '\')" style="cursor:pointer;">' + description + '</span>' +
      '</div>' +
    '</div>';
  });
  html += '</div>';
  timeline.innerHTML = html;
}

// Phase 54.6.314 — format the version-row timestamps. ISO → local
// "YYYY-MM-DD HH:MM". If created/updated differ by > 60 s show both.
function _formatVersionStamp(createdIso, updatedIso) {
  const fmt = (iso) => {
    if (!iso) return '—';
    try {
      const d = new Date(iso);
      if (isNaN(d.getTime())) return iso;
      const pad = (n) => String(n).padStart(2, '0');
      return d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate())
        + ' ' + pad(d.getHours()) + ':' + pad(d.getMinutes());
    } catch (e) { return iso; }
  };
  const c = fmt(createdIso);
  const u = fmt(updatedIso);
  if (!updatedIso || u === c) return 'created ' + c;
  // Only surface updated_at when it materially differs from created.
  try {
    const dc = new Date(createdIso);
    const du = new Date(updatedIso);
    if (!isNaN(dc.getTime()) && !isNaN(du.getTime())
        && Math.abs(du.getTime() - dc.getTime()) < 60000) {
      return 'created ' + c;
    }
  } catch (e) {}
  return 'created ' + c + ' · updated ' + u;
}

async function editVersionDescription(draftId) {
  if (!draftId) return;
  const current = (versionData.find(v => v.id === draftId) || {}).version_description || '';
  const next = prompt('Short description for this version (empty to clear):', current);
  if (next === null) return;
  try {
    const fd = new FormData();
    fd.append('description', next);
    const res = await fetch('/api/draft/' + encodeURIComponent(draftId) + '/version-description', {
      method: 'POST', body: fd,
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    await showVersions();
  } catch (e) {
    alert('Could not update description: ' + e);
  }
}

async function renameVersion(draftId) {
  if (!draftId) return;
  const current = (versionData.find(v => v.id === draftId) || {}).version_name || '';
  const next = prompt('Version name (empty to clear):', current);
  if (next === null) return;
  try {
    const fd = new FormData();
    fd.append('name', next);
    const res = await fetch('/api/draft/' + encodeURIComponent(draftId) + '/rename-version', {
      method: 'POST', body: fd,
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    await showVersions();
  } catch (e) {
    alert('Could not rename: ' + e);
  }
}

// Phase 54.6.309 — flip the active flag on one version; reload the reader
// so the pinned version is what shows up (globalised citations, sources,
// scores all follow).
async function activateVersion(draftId) {
  if (!draftId) return;
  try {
    const r = await fetch('/api/draft/' + encodeURIComponent(draftId) + '/activate', {method: 'POST'});
    if (!r.ok) {
      const msg = await r.text();
      alert('Could not activate: ' + msg);
      return;
    }
    // Reload list + navigate to the new active draft.
    await showVersions();
    if (typeof loadSection === 'function') {
      loadSection(draftId);
    }
  } catch (e) {
    alert('Error activating version: ' + e);
  }
}

async function selectVersion(vid) {
  // Toggle selection (max 2)
  const idx = selectedVersions.indexOf(vid);
  if (idx >= 0) {
    selectedVersions.splice(idx, 1);
  } else {
    if (selectedVersions.length >= 2) selectedVersions.shift();
    selectedVersions.push(vid);
  }

  // Update badges
  document.querySelectorAll('.version-badge').forEach(b => {
    b.classList.toggle('selected', selectedVersions.includes(b.dataset.vid));
  });

  if (selectedVersions.length === 2) {
    // Show diff
    const diffView = document.getElementById('diff-view');
    diffView.innerHTML = '<p class="u-dim">Loading diff...</p>';
    const res = await fetch('/api/diff/' + selectedVersions[0] + '/' + selectedVersions[1]);
    const data = await res.json();
    diffView.innerHTML = data.diff_html;
  } else if (selectedVersions.length === 1) {
    // Navigate to that version, then RE-show the version panel
    // (Phase 54.6.314 — loadSection hides version-panel as part of
    // its "navigate somewhere" cleanup, which otherwise blew the
    // panel away the moment the user clicked a version badge. Users
    // reported this as "I can't browse between versions." Keep the
    // picker alive so browsing across versions is continuous.)
    await loadSection(selectedVersions[0]);
    const panel = document.getElementById('version-panel');
    if (panel) panel.style.display = 'block';
    // Update selected-badge styling in case the DOM was rebuilt.
    document.querySelectorAll('.version-badge').forEach(b => {
      b.classList.toggle('selected', selectedVersions.includes(b.dataset.vid));
    });
  }
}

// ── Chapter Management ────────────────────────────────────────────────────
async function addChapter() {
  const title = document.getElementById('ch-add-title').value.trim();
  if (!title) return;

  const fd = new FormData();
  fd.append('title', title);
  await fetch('/api/chapters', {method: 'POST', body: fd});
  document.getElementById('ch-add-title').value = '';
  document.getElementById('ch-add-form').style.display = 'none';
  await refreshAfterJob(null);
}

async function deleteChapter(chapterId) {
  if (!confirm('Delete this chapter? Drafts will be preserved but unlinked.')) return;
  await fetch('/api/chapters/' + chapterId, {method: 'DELETE'});
  await refreshAfterJob(null);
}

// Phase 32.4 — delete a single section from a chapter's sections_meta.
// The associated draft (if any) becomes an orphan in the sidebar — it
// is NOT hard-deleted, so the user can adopt it back via the existing
// + button or hard-delete it via the X. This mirrors the chapter
// delete UX (drafts preserved, just unlinked).
async function deleteSection(chapterId, slug) {
  if (!chapterId || !slug) return;
  const ch = chaptersData.find(c => c.id === chapterId);
  if (!ch) return;
  const meta = Array.isArray(ch.sections_meta) ? ch.sections_meta : [];
  const sec = meta.find(s => s.slug === slug);
  const title = (sec && sec.title) ? sec.title : slug;
  // Detect whether a draft exists for this slug — affects the warning text.
  const drafted = (ch.sections || []).some(d => (d.type || '').toLowerCase() === slug.toLowerCase());
  const msg = drafted
    ? 'Remove section "' + title + '" from this chapter?\n\nThe existing draft will become an orphan (still listed in the sidebar with a + to re-add and X to permanently delete).'
    : 'Remove section "' + title + '" from this chapter?';
  if (!confirm(msg)) return;
  const updated = meta.filter(s => s.slug !== slug).map(s => ({
    slug: s.slug,
    title: s.title || s.slug,
    plan: s.plan || '',
    target_words: (s.target_words && s.target_words > 0) ? s.target_words : null,
  }));
  try {
    const res = await fetch('/api/chapters/' + chapterId + '/sections', {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({sections: updated}),
    });
    if (!res.ok) throw new Error('save failed (' + res.status + ')');
    // If the user was viewing the draft we just orphaned, fall back
    // to the dashboard so the center frame doesn't show stale content.
    if (currentDraftId && drafted) {
      const activeIsThis = (ch.sections || []).some(d =>
        d.id === currentDraftId && (d.type || '').toLowerCase() === slug.toLowerCase()
      );
      if (activeIsThis) {
        currentDraftId = '';
        showDashboard();
      }
    }
    // Refresh the sidebar so the slot disappears and any orphaned
    // draft surfaces in the orphan list.
    const sidebarRes = await fetch('/api/chapters');
    const sd = await sidebarRes.json();
    rebuildSidebar(sd.chapters || sd, currentDraftId);
    // Update local cache so subsequent operations see the new list
    if (Array.isArray(sd.chapters)) chaptersData = sd.chapters;
  } catch (e) {
    alert('Failed to remove section: ' + e.message);
  }
}

// Phase 32.4 — inline add-section flow from the sidebar. Prompts for
// a title, derives a slug client-side (matches core.book_ops's
// _slugify_section_name), appends a new section dict to the chapter's
// sections_meta, and PUTs the new list. The chapter modal Sections
// tab still works for richer edits (plan, target_words, reorder).
async function addSectionToChapter(chapterId) {
  if (!chapterId) return;
  const ch = chaptersData.find(c => c.id === chapterId);
  if (!ch) return;
  const title = prompt('New section title (e.g. "The 11-Year Solar Cycle"):');
  if (!title || !title.trim()) return;
  const cleanTitle = title.trim();
  const slug = cleanTitle.toLowerCase().replace(/\s+/g, '_').replace(/[^a-z0-9_]/g, '');
  if (!slug) {
    alert('Could not derive a slug from that title. Try plain letters.');
    return;
  }
  const meta = Array.isArray(ch.sections_meta) ? ch.sections_meta : [];
  if (meta.some(s => s.slug === slug)) {
    alert('A section with slug "' + slug + '" already exists in this chapter.');
    return;
  }
  const updated = meta.map(s => ({
    slug: s.slug,
    title: s.title || s.slug,
    plan: s.plan || '',
    target_words: (s.target_words && s.target_words > 0) ? s.target_words : null,
  }));
  updated.push({slug: slug, title: cleanTitle, plan: '', target_words: null});
  try {
    const res = await fetch('/api/chapters/' + chapterId + '/sections', {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({sections: updated}),
    });
    if (!res.ok) throw new Error('save failed (' + res.status + ')');
    // Refresh sidebar so the new slot appears
    const sidebarRes = await fetch('/api/chapters');
    const sd = await sidebarRes.json();
    rebuildSidebar(sd.chapters || sd, currentDraftId);
    if (Array.isArray(sd.chapters)) chaptersData = sd.chapters;
  } catch (e) {
    alert('Failed to add section: ' + e.message);
  }
}

// ── Enhanced Editor (Phase 3a) ────────────────────────────────────────
let _autosaveTimer = null;
let _currentRaw = '';

function toggleEdit() {
  const rv = document.getElementById('read-view');
  const ev = document.getElementById('edit-view');
  const ta = document.getElementById('edit-area');
  if (ev.style.display === 'none') {
    // Load current raw content via API
    fetch('/api/section/' + currentDraftId).then(r => r.json()).then(data => {
      _currentRaw = data.content_raw;
      ta.value = _currentRaw;
      edPreview();
    });
    rv.style.display = 'none';
    ev.style.display = 'block';
    // Start autosave
    _autosaveTimer = setInterval(edAutosave, 5000);
  } else {
    rv.style.display = 'block';
    ev.style.display = 'none';
    if (_autosaveTimer) { clearInterval(_autosaveTimer); _autosaveTimer = null; }
  }
}

function edPreview() {
  const ta = document.getElementById('edit-area');
  const preview = document.getElementById('edit-preview');
  // Phase 14.4 — flag unsaved changes immediately so the user sees the
  // amber dot the moment they type, not after the next 5s autosave tick.
  if (ta.value !== _currentRaw) setAutosaveState('unsaved', 'Unsaved changes');
  let md = ta.value;
  // Simple markdown → HTML for preview.
  md = md.replace(/^### (.+)$/gm, '<h4>$1</h4>');
  md = md.replace(/^## (.+)$/gm, '<h3>$1</h3>');
  md = md.replace(/^# (.+)$/gm, '<h2>$1</h2>');
  md = md.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  md = md.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // Phase 54.6.87 — ![alt](url) → <img> (constrained to our own
  // /api/visuals/image/ paths + whitelisted schemes). Matches the
  // server-side _md_to_html rule so the preview shows the same thumbnails
  // the read-view will.
  md = md.replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, function(_m, alt, src) {
    alt = (alt || '').replace(/"/g, '&quot;').replace(/</g, '&lt;');
    const ok = /^(?:\/api\/visuals\/image\/|\/static\/|https?:\/\/)/.test(src);
    if (!ok) return alt;
    return '<img class="inline-figure" src="' + src + '" alt="' + alt + '" '
      + 'loading="lazy" style="max-width:100%;height:auto;border:1px solid '
      + 'var(--border,#ddd);border-radius:6px;margin:10px 0;" title="' + alt + '">';
  });
  md = md.replace(/\[(\d+)\]/g, '<span class="citation">[$1]</span>');
  const paras = md.split('\n\n');
  preview.innerHTML = paras.map(p => {
    p = p.trim();
    if (!p) return '';
    if (p.startsWith('<h') || p.startsWith('<img')) return p;
    return '<p>' + p + '</p>';
  }).join('');
  // Phase 54.6.87 — render math via KaTeX auto-render. Pre-fix the
  // editor preview showed raw `$...$` / `$$...$$` source, same as the
  // main read-view did until this phase. Silent if KaTeX isn't loaded.
  _renderMathInEl(preview);
}

function _renderMathInEl(el) {
  /* Shared helper — auto-render KaTeX inside any content-bearing
   * element. Used by edPreview + read-view assignments below. */
  if (!el || typeof window.renderMathInElement !== 'function') return;
  try {
    window.renderMathInElement(el, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$',  right: '$',  display: false },
        { left: '\\(', right: '\\)', display: false },
        { left: '\\[', right: '\\]', display: true },
      ],
      throwOnError: false,
    });
  } catch (e) { /* non-fatal */ }
}

function edInsert(before, after) {
  const ta = document.getElementById('edit-area');
  const start = ta.selectionStart;
  const end = ta.selectionEnd;
  const sel = ta.value.substring(start, end) || 'text';
  ta.value = ta.value.substring(0, start) + before + sel + after + ta.value.substring(end);
  ta.selectionStart = start + before.length;
  ta.selectionEnd = start + before.length + sel.length;
  ta.focus();
  edPreview();
}

function edInsertCite() {
  const n = prompt('Citation number:');
  if (n) edInsert('[' + n + ']', '');
}

// Phase 54.6.312 — extra markdown insertion helpers for the editor
// toolbar. edInsertLine prefixes the current line (or each selected
// line) with `prefix`; useful for lists, quotes, rules.
function edInsertLine(prefix, suffix) {
  const ta = document.getElementById('edit-area');
  if (!ta) return;
  suffix = suffix || '';
  const start = ta.selectionStart;
  const end = ta.selectionEnd;
  const before = ta.value.substring(0, start);
  const selected = ta.value.substring(start, end);
  const after = ta.value.substring(end);
  // Find the start of the current line for the caret prefix.
  const lineStart = before.lastIndexOf('\n') + 1;
  const head = before.substring(0, lineStart);
  const lineBefore = before.substring(lineStart);
  let body;
  if (selected) {
    body = selected.split('\n').map(l => prefix + l).join('\n');
  } else {
    body = prefix + (lineBefore || '');
  }
  const replaced = selected ? '' : lineBefore;
  ta.value = head + body + suffix + after.substring(0);
  const newPos = head.length + body.length + suffix.length;
  ta.selectionStart = newPos; ta.selectionEnd = newPos;
  // If we replaced the current line's text, drop it from head/lineBefore.
  if (!selected && lineBefore) {
    // We already substituted above; do nothing extra.
  }
  ta.focus();
  edPreview();
}

function edInsertLink() {
  const url = prompt('Link URL (https://…):', 'https://');
  if (!url) return;
  const label = prompt('Link text:', '');
  const sel = label || 'link';
  edInsert('[' + sel + '](' + url + ')', '');
}

function edUndo() {
  const ta = document.getElementById('edit-area');
  if (!ta) return;
  ta.focus();
  try { document.execCommand('undo'); edPreview(); }
  catch (e) { /* browser does not support execCommand */ }
}

function edRedo() {
  const ta = document.getElementById('edit-area');
  if (!ta) return;
  ta.focus();
  try { document.execCommand('redo'); edPreview(); }
  catch (e) {}
}

// Save the editor buffer as a NEW version with an optional label.
// Calls /api/draft/{id}/save-as-version (creates a fresh drafts row,
// increments version, pins it active) then reloads the reader so the
// Versions panel shows the new entry.
async function edSaveAsVersion() {
  const ta = document.getElementById('edit-area');
  if (!ta || !currentDraftId) return;
  const name = prompt('Version name (optional, e.g. "pre-review"):', '');
  if (name === null) return;   // cancelled
  const fd = new FormData();
  fd.append('content', ta.value);
  fd.append('version_name', name || '');
  setAutosaveState('saving', 'Saving new version…');
  try {
    const res = await fetch('/api/draft/' + encodeURIComponent(currentDraftId) + '/save-as-version', {
      method: 'POST', body: fd,
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    setAutosaveState('', 'Saved v' + (data.version || '') + (data.version_name ? (' (' + data.version_name + ')') : ''));
    if (_autosaveTimer) { clearInterval(_autosaveTimer); _autosaveTimer = null; }
    toggleEdit();
    if (data.new_draft_id && typeof loadSection === 'function') loadSection(data.new_draft_id);
  } catch (e) {
    setAutosaveState('error', 'Save failed');
    alert('Could not save as new version: ' + e);
  }
}

// Phase 14.4 — autosave UX: a small status pill that's always visible
// while in edit mode. Colours: green=saved, amber-pulse=saving,
// amber-static=unsaved, red=error.
function setAutosaveState(state, label) {
  const el = document.getElementById('autosave-status');
  if (!el) return;
  el.classList.remove('saving', 'unsaved', 'error');
  if (state) el.classList.add(state);
  const text = document.getElementById('autosave-text');
  if (text) text.textContent = label;
}

async function edSave() {
  const ta = document.getElementById('edit-area');
  setAutosaveState('saving', 'Saving...');
  try {
    const fd = new FormData();
    fd.append('content', ta.value);
    await fetch('/edit/' + currentDraftId, {method: 'POST', body: fd});
    setAutosaveState('', 'Saved');
  } catch (e) {
    setAutosaveState('error', 'Save failed');
    return;
  }
  if (_autosaveTimer) { clearInterval(_autosaveTimer); _autosaveTimer = null; }
  toggleEdit();
  loadSection(currentDraftId);
}

async function edAutosave() {
  const ta = document.getElementById('edit-area');
  if (ta.value === _currentRaw) return;
  setAutosaveState('saving', 'Saving...');
  try {
    _currentRaw = ta.value;
    const fd = new FormData();
    fd.append('content', ta.value);
    await fetch('/edit/' + currentDraftId, {method: 'POST', body: fd});
    const t = new Date().toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
    setAutosaveState('', 'Saved at ' + t);
  } catch (e) {
    setAutosaveState('error', 'Save failed');
  }
}

// ── Claim Verification (Phase 5b) ────────────────────────────────────
async function doVerify() {
  if (!currentDraftId) { showEmptyHint("No draft selected &mdash; click a section in the sidebar, or click <strong>Start writing</strong> under any chapter to create a first draft."); return; }
  showStreamPanel('Verifying citations...');

  const fd = new FormData();
  const res = await fetch('/api/verify/' + currentDraftId, {method: 'POST', body: fd});
  const data = await res.json();

  // Override stream handler to process verification data
  currentJobId = data.job_id;
  if (currentEventSource) currentEventSource.close();

  const source = new EventSource('/api/stream/' + data.job_id);
  currentEventSource = source;
  const body = document.getElementById('stream-body');
  const status = document.getElementById('stream-status');

  source.onmessage = function(e) {
    const evt = JSON.parse(e.data);
    if (evt.type === 'progress') {
      status.textContent = evt.detail || evt.stage;
    }
    else if (evt.type === 'verification') {
      const vd = evt.data;
      const grStr = (vd.groundedness_score != null) ? vd.groundedness_score.toFixed(2) : '?';
      const hfStr = (vd.hedging_fidelity_score != null) ? vd.hedging_fidelity_score.toFixed(2) : '?';
      status.textContent = 'Groundedness: ' + grStr + '  ·  Hedging fidelity: ' + hfStr;

      // Show results in stream body
      let html = '<div class="u-sys">';
      html += '<div style="font-size:18px;font-weight:bold;margin-bottom:12px;">Groundedness: ' +
        '<span style="color:' + (vd.groundedness_score >= 0.8 ? 'var(--success)' : vd.groundedness_score >= 0.6 ? 'var(--warning)' : 'var(--danger)') + '">' +
        grStr + '</span>' +
        '   <span style="font-size:14px;font-weight:normal;opacity:0.7;">Hedging fidelity: ' +
        '<span style="color:' + (vd.hedging_fidelity_score >= 0.8 ? 'var(--success)' : vd.hedging_fidelity_score >= 0.6 ? 'var(--warning)' : 'var(--danger)') + '">' +
        hfStr + '</span></span></div>';

      if (vd.claims) {
        vd.claims.forEach(c => {
          // Phase 11 — OVERSTATED is the lexical-modality counterpart to
          // EXTRAPOLATED. Render in orange to distinguish from yellow
          // (extrapolated) and red (misrepresented).
          let color;
          if (c.verdict === 'SUPPORTED') color = 'var(--success)';
          else if (c.verdict === 'EXTRAPOLATED') color = 'var(--warning)';
          else if (c.verdict === 'OVERSTATED') color = '#f97316';  // orange-500
          else color = 'var(--danger)';
          html += '<div style="margin:6px 0;padding:6px 10px;border-left:3px solid ' + color + ';background:var(--toolbar-bg);border-radius:4px;">';
          html += '<span style="font-weight:bold;color:' + color + ';">' + c.verdict + '</span> ';
          html += '<span class="u-small">' + c.citation + '</span><br>';
          html += '<span class="u-small u-dim-7">' + (c.text || '').substring(0, 120) + '</span>';
          if (c.reason) html += '<br><span class="u-tiny u-dim">' + c.reason + '</span>';
          html += '</div>';
        });
      }

      if (vd.unsupported_claims && vd.unsupported_claims.length > 0) {
        html += '<div class="u-mt-3 u-bold u-danger">Unsupported Claims:</div>';
        vd.unsupported_claims.forEach(u => {
          html += '<div class="u-small u-danger u-my-1">- ' + u.substring(0, 120) + '</div>';
        });
      }
      html += '</div>';
      body.innerHTML = html;

      // Apply color indicators to citations in the read view
      applyVerificationColors(vd);
    }
    else if (evt.type === 'completed') {
      hideStreamPanel();
      source.close(); currentEventSource = null; currentJobId = null;
    }
    else if (evt.type === 'error') {
      status.textContent = 'Error: ' + evt.message;
      body.innerHTML = '<div class="u-danger">' + evt.message + '</div>';
      hideStreamPanel();
      source.close(); currentEventSource = null; currentJobId = null;
    }
    else if (evt.type === 'done') {
      hideStreamPanel();
      source.close(); currentEventSource = null; currentJobId = null;
    }
  };
}

function applyVerificationColors(vd) {
  if (!vd.claims) return;
  const classMap = { 'SUPPORTED': 'verified-supported', 'EXTRAPOLATED': 'verified-extrapolated', 'OVERSTATED': 'verified-overstated', 'MISREPRESENTED': 'verified-misrepresented' };
  vd.claims.forEach(c => {
    const ref = c.citation ? c.citation.replace(/[\[\]]/g, '') : '';
    if (!ref) return;
    document.querySelectorAll('.citation[data-ref="' + ref + '"]').forEach(el => {
      el.className = 'citation ' + (classMap[c.verdict] || '');
    });
  });
}

// ── Phase 46.A — Auto-insert [N] citations (two-pass LLM) ───────────────
async function doInsertCitations() {
  if (!currentDraftId) {
    showEmptyHint("No draft selected &mdash; click a section in the sidebar, or click <strong>Start writing</strong> under any chapter to create a first draft.");
    return;
  }
  if (!confirm(
    "Auto-insert [N] citation markers?\n\n"
    + "This scans the draft for claims that need citations, hybrid-searches "
    + "your corpus for top-8 candidates per claim, and saves a new version "
    + "with the accepted citations applied.\n\n"
    + "Takes ~1-3 minutes depending on draft length. Mirrors "
    + "`sciknow book insert-citations`."
  )) return;
  showStreamPanel('Inserting citations...');
  const fd = new FormData();
  const res = await fetch('/api/insert-citations/' + currentDraftId, {
    method: 'POST', body: fd
  });
  const data = await res.json();
  currentJobId = data.job_id;
  if (currentEventSource) currentEventSource.close();
  const source = new EventSource('/api/stream/' + data.job_id);
  currentEventSource = source;
  const body = document.getElementById('stream-body');
  const status = document.getElementById('stream-status');
  body.innerHTML =
    '<div id="ic-summary" style="font-size:16px;margin-bottom:12px;"></div>'
    + '<div id="ic-log" style="font-size:12px;font-family:ui-monospace,monospace;'
    + 'max-height:320px;overflow:auto;padding:8px;background:var(--toolbar-bg);'
    + 'border-radius:4px;"></div>';
  const summary = document.getElementById('ic-summary');
  const log = document.getElementById('ic-log');
  function _append(html) { log.innerHTML += html; log.scrollTop = log.scrollHeight; }
  source.onmessage = function(e) {
    const evt = JSON.parse(e.data);
    if (evt.type === 'progress') {
      status.textContent = evt.detail || evt.stage;
      _append('<div class="u-muted">' + (evt.detail || evt.stage) + '</div>');
    } else if (evt.type === 'citation_needs') {
      summary.innerHTML = '<strong>' + evt.count + '</strong> location(s) flagged for citation';
      status.textContent = 'Retrieving candidates...';
    } else if (evt.type === 'citation_candidates') {
      _append('<div>#' + (evt.index + 1) + ': ' + evt.n_candidates + ' candidate(s)</div>');
    } else if (evt.type === 'citation_selected') {
      const conf = (evt.confidence != null) ? evt.confidence.toFixed(2) : '?';
      _append('<div class="u-success">&nbsp;&nbsp;&#10003; #'
        + (evt.index + 1) + ' picked (conf ' + conf + ')</div>');
    } else if (evt.type === 'citation_skipped') {
      _append('<div class="u-muted">&nbsp;&nbsp;&mdash; #'
        + (evt.index + 1) + ' skipped (' + (evt.reason || 'no match') + ')</div>');
    } else if (evt.type === 'citation_inserted') {
      // Total count emitted right before completed
    } else if (evt.type === 'completed') {
      const msg = evt.message
        || ('Inserted ' + (evt.n_inserted || 0) + ' / ' + (evt.n_needs || 0) + ' citations');
      status.textContent = msg;
      summary.innerHTML = '<span class="u-success u-bold">&#10003; '
        + msg + '</span>';
      source.close(); currentEventSource = null; currentJobId = null;
      if ((evt.n_inserted || 0) > 0 && currentDraftId) {
        setTimeout(() => loadSection(currentDraftId), 800);
      }
      setTimeout(() => hideStreamPanel(), 3500);
    } else if (evt.type === 'error') {
      status.textContent = 'Error: ' + evt.message;
      summary.innerHTML = '<span class="u-danger">&#10007; ' + evt.message + '</span>';
      source.close(); currentEventSource = null; currentJobId = null;
    } else if (evt.type === 'done') {
      source.close(); currentEventSource = null; currentJobId = null;
    }
  };
}

// ── Phase 54.6.14 — BMAD-inspired critic skills ─────────────────────────
// Adversarial review: streams findings as a markdown list into the
// stream panel. No persistence (doesn't touch review_feedback).
async function doAdversarialReview() {
  if (!currentDraftId) {
    showEmptyHint("No draft selected.");
    return;
  }
  if (!confirm(
    "Run adversarial critic pass?\n\n"
    + "A cynical reviewer mode that finds AT LEAST 10 concrete issues per\n"
    + "draft — unsupported claims, overgeneralisation, weasel words, missing\n"
    + "counter-evidence, internal contradictions, loaded framing, etc.\n\n"
    + "Doesn't overwrite the normal review. Takes 30s-2min."
  )) return;
  showStreamPanel('Adversarial review in progress…');
  const fd = new FormData();
  const res = await fetch('/api/adversarial-review/' + currentDraftId, {method:'POST', body: fd});
  const data = await res.json();
  currentJobId = data.job_id;
  if (currentEventSource) currentEventSource.close();
  const source = new EventSource('/api/stream/' + data.job_id);
  currentEventSource = source;
  const body = document.getElementById('stream-body');
  const status = document.getElementById('stream-status');
  body.innerHTML = '<h3 class="u-mb-m">&#128126; Adversarial review</h3>'
    + '<div class="u-sys u-lg u-lh-1-5" id="adv-output"></div>';
  const out = document.getElementById('adv-output');
  let buf = '';
  source.onmessage = function(e) {
    const evt = JSON.parse(e.data);
    if (evt.type === 'token') {
      buf += evt.text;
      out.innerHTML = _mdSimple(buf);
    } else if (evt.type === 'progress') {
      status.textContent = evt.detail || evt.stage;
    } else if (evt.type === 'completed') {
      status.textContent = (evt.n_findings || '?') + ' finding(s)';
      out.innerHTML = _mdSimple(evt.findings_markdown || buf);
      source.close(); currentEventSource = null; currentJobId = null;
    } else if (evt.type === 'error') {
      status.textContent = 'Error: ' + evt.message;
      source.close(); currentEventSource = null; currentJobId = null;
    } else if (evt.type === 'done') {
      source.close(); currentEventSource = null; currentJobId = null;
    }
  };
}

// Edge-case hunter: streams structured findings as a sortable table.
async function doEdgeCases() {
  if (!currentDraftId) {
    showEmptyHint("No draft selected.");
    return;
  }
  if (!confirm(
    "Run edge-case hunter?\n\n"
    + "Walks every branching path and boundary condition in the draft.\n"
    + "Reports only UNHANDLED cases — scope boundaries, counter-cases,\n"
    + "causal alternatives, quantitative limits, extrapolations, missing\n"
    + "controls. Returns a severity-ranked findings table.\n\n"
    + "Takes 30s-2min."
  )) return;
  showStreamPanel('Walking edge cases…');
  const fd = new FormData();
  const res = await fetch('/api/edge-cases/' + currentDraftId, {method:'POST', body: fd});
  const data = await res.json();
  currentJobId = data.job_id;
  if (currentEventSource) currentEventSource.close();
  const source = new EventSource('/api/stream/' + data.job_id);
  currentEventSource = source;
  const body = document.getElementById('stream-body');
  const status = document.getElementById('stream-status');
  body.innerHTML = '<h3 class="u-mb-m">&#129327; Edge-case hunter</h3>'
    + '<table class="u-w-full u-bcollapse u-md" id="ec-table">'
    + '<thead class="u-bg-tb"><tr>'
    + '<th class="u-th-wrap">Sev</th>'
    + '<th class="u-th-wrap">Location</th>'
    + '<th class="u-th-wrap">Unhandled condition</th>'
    + '<th class="u-th-wrap">If triggered, the claim…</th>'
    + '</tr></thead><tbody></tbody></table>';
  const tbody = body.querySelector('#ec-table tbody');
  const sevColor = {high:'var(--danger)', medium:'var(--warning)', low:'var(--fg-muted)'};
  source.onmessage = function(e) {
    const evt = JSON.parse(e.data);
    if (evt.type === 'progress') {
      status.textContent = evt.detail || evt.stage;
    } else if (evt.type === 'finding') {
      const f = evt.data || {};
      const sev = (f.severity || 'low').toLowerCase();
      const tr = document.createElement('tr');
      tr.style.borderTop = '1px solid var(--border)';
      tr.innerHTML =
        '<td class="u-p-6-8 u-vat"><span style="color:'
          + sevColor[sev] + ';font-weight:bold;font-size:11px;text-transform:uppercase;">'
          + sev + '</span></td>'
        + '<td class="u-p-6-8 u-vat u-italic">' + _escHtml(f.location || '') + '</td>'
        + '<td class="u-p-6-8 u-vat">' + _escHtml(f.trigger || '') + '</td>'
        + '<td class="u-p-6-8 u-vat u-muted">' + _escHtml(f.consequence || '') + '</td>';
      tbody.appendChild(tr);
    } else if (evt.type === 'completed') {
      status.innerHTML = '<span class="u-success">&#10003; '
        + (evt.n_findings || 0) + ' edge case(s) flagged.</span>';
      source.close(); currentEventSource = null; currentJobId = null;
    } else if (evt.type === 'error') {
      status.textContent = 'Error: ' + evt.message;
      source.close(); currentEventSource = null; currentJobId = null;
    } else if (evt.type === 'done') {
      source.close(); currentEventSource = null; currentJobId = null;
    }
  };
}

// Phase 54.6.97 — three new draft-level actions that just shell to
// the CLI via /api/cli-stream. All three write their output as plain
// log text into the existing stream panel so there's no new UI to
// maintain for each action.
function _runCliActionForDraft(argv, title, startMsg) {
  if (!currentDraftId) { showEmptyHint('No draft selected.'); return; }
  showStreamPanel(startMsg);
  const body = document.getElementById('stream-body');
  body.innerHTML = '<h3 class="u-mb-m">' + title + '</h3>'
    + '<pre class="u-pre u-mono u-small u-lh-145" id="cli-action-out"></pre>';
  const out = document.getElementById('cli-action-out');
  const status = document.getElementById('stream-status');
  fetch('/api/cli-stream', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({argv: argv}),
  }).then(r => r.json()).then(data => {
    currentJobId = data.job_id;
    if (currentEventSource) currentEventSource.close();
    const source = new EventSource('/api/stream/' + data.job_id);
    currentEventSource = source;
    source.onmessage = function(e) {
      const evt = JSON.parse(e.data);
      if (evt.type === 'log') {
        out.textContent += evt.text + '\n';
        out.scrollTop = out.scrollHeight;
      } else if (evt.type === 'completed') {
        status.innerHTML = '<span class="u-success">\u2713 Done.</span>';
        source.close(); currentEventSource = null; currentJobId = null;
      } else if (evt.type === 'error') {
        status.innerHTML = '<span class="u-danger">\u2717 ' + (evt.message || 'error') + '</span>';
        source.close(); currentEventSource = null; currentJobId = null;
      } else if (evt.type === 'done') {
        source.close(); currentEventSource = null; currentJobId = null;
      }
    };
  }).catch(exc => {
    status.textContent = 'Request failed: ' + exc;
  });
}

async function doVerifyDraft() {
  if (!confirm(
    'Run Verify Draft (claim atomization)?\n\n'
    + 'Splits each sentence into atomic sub-claims, NLI-scores each one against\n'
    + 'source chunks, and surfaces MIXED-TRUTH sentences (part supported, part\n'
    + 'not) that the sentence-level verifier misses. Read-only — does not\n'
    + 'modify the draft.\n\nTakes 30s-90s depending on draft length.'
  )) return;
  _runCliActionForDraft(
    ['book', 'verify-draft', currentDraftId],
    '&#129516; Verify Draft (claim atomization)',
    'Atomizing claims…'
  );
}

async function doFinalizeDraft() {
  if (!confirm(
    'Run Finalize Draft (L3 VLM verify)?\n\n'
    + 'Verifies every [Fig. N] / [Table N] / [Eq. N] marker in this draft by '
    + 'running the vision-language model on (claim_sentence, image) pairs '
    + 'and scoring whether the image depicts the claim (0-10 rubric).\n\n'
    + 'Deferred from the per-iteration autowrite loop because it is '
    + 'expensive (~3-10s per marker). Run once when the draft is stable, '
    + 'before export.\n\n'
    + 'Exit code 0 if every resolved marker passes, 1 if any are flagged '
    + '(so CI can gate export on clean verify).'
  )) return;
  _runCliActionForDraft(
    ['book', 'finalize-draft', currentDraftId],
    '&#128190; Finalize Draft (L3 VLM verify)',
    'Running VLM on each [Fig. N] marker…'
  );
}

async function doAlignCitations() {
  if (!confirm(
    'Run Align Citations?\n\n'
    + 'Post-pass that (conservatively) remaps [N] markers to the chunk that\n'
    + 'actually entails each sentence. Only remaps when the claimed chunk has\n'
    + 'entailment < 0.5 AND the top chunk beats it by >= 0.15. Saves the\n'
    + 'remapped draft back to the database.\n\nTakes 20s-60s.'
  )) return;
  _runCliActionForDraft(
    ['book', 'align-citations', currentDraftId],
    '&#128279; Align Citations',
    'Re-scoring entailment for every [N]…'
  );
}

async function doEnsembleReview() {
  const nStr = prompt('Number of independent reviewers? (1-9, default 3)', '3');
  if (nStr === null) return;
  const n = Math.max(1, Math.min(9, parseInt(nStr, 10) || 3));
  if (!confirm(
    'Run Ensemble Review with ' + n + ' reviewer(s)?\n\n'
    + 'Each reviewer uses NeurIPS rubric (soundness/presentation/contribution\n'
    + '1-4, overall 1-10, confidence 1-5, decision strong_reject..strong_accept)\n'
    + 'with temperature 0.75 and rotating neutral/pessimistic/optimistic stance.\n'
    + 'A meta-reviewer medians the scores and unions the findings.\n\n'
    + 'Cost scales with N — ' + n + 'x a single review (~' + (n * 60) + 's).'
  )) return;
  _runCliActionForDraft(
    ['book', 'ensemble-review', currentDraftId, '-n', String(n)],
    '&#128218; Ensemble Review',
    'Running ' + n + ' independent reviewers…'
  );
}

function openAIActionsHelp() {
  // Close any open nav dropdowns before pulling up the modal so the
  // overlay doesn't stack weirdly.
  document.querySelectorAll('.nav-dropdown.open').forEach(d => d.classList.remove('open'));
  openModal('ai-help-modal');
}

// Tiny markdown renderer — paragraphs, numbered lists, bullets, code.
// Used by the adversarial-review panel; keeps the findings readable
// without pulling in a whole markdown library.
function _mdSimple(src) {
  let s = _escHtml(src || '');
  s = s.replace(/^\s*([0-9]+)\.\s+(.*)$/gm,
    '<li><strong>$1.</strong> $2</li>');
  s = s.replace(/^\s*[-*]\s+(.*)$/gm, '<li>$1</li>');
  // Wrap consecutive <li>s in a single <ol>.
  s = s.replace(/(<li>.*<\/li>\s*)+/gs, m => '<ol style="padding-left:22px;">' + m + '</ol>');
  s = s.replace(/\n{2,}/g, '</p><p>');
  s = '<p>' + s + '</p>';
  s = s.replace(/<p>\s*(<ol[^>]*>)/g, '$1').replace(/(<\/ol>)\s*<\/p>/g, '$1');
  return s;
}

// ── Citation Popovers (Phase 5a) ─────────────────────────────────────
function buildPopovers() {
  // Extract source data from the sources panel
  const sourceItems = document.querySelectorAll('#panel-sources li');
  const sourceData = {};
  sourceItems.forEach((li, i) => {
    sourceData[i + 1] = li.textContent;
  });

  // Phase 20 — count broken citations so we can warn the user once.
  // A broken citation is one whose ref number > number of sources;
  // it's typically the result of a pre-Phase-18 draft (where the
  // writer's prompt and the saved sources had inconsistent numbering)
  // or the writer hallucinating a citation number.
  let brokenCount = 0;
  let brokenRefs = new Set();

  document.querySelectorAll('.citation').forEach(el => {
    const ref = el.dataset.ref;
    if (!ref) return;
    // Always (re)attach click handler — buildPopovers may run multiple
    // times on the same DOM (after section nav, after autowrite finish),
    // and the source map could have changed in between.
    const src = sourceData[parseInt(ref)];

    if (!src) {
      // Phase 20 — broken citation: visually mark and short-circuit.
      // Don't attach a click handler that scrolls to a non-existent
      // anchor — instead, on click show a tooltip explaining the
      // citation has no matching source.
      el.classList.add('citation-broken');
      el.title = 'Citation [' + ref + '] has no matching source. ' +
                 'This usually means the draft predates the Phase 18 ' +
                 'fix — re-run autowrite to regenerate it.';
      el.onclick = function(e) {
        e.preventDefault();
        e.stopPropagation();
      };
      brokenCount += 1;
      brokenRefs.add(ref);
      return;
    }

    // Healthy citation: build popover + scroll-to-source click handler.
    if (!el.querySelector('.citation-popover')) {
      const popover = document.createElement('div');
      popover.className = 'citation-popover';

      // Parse the APA-style citation: [N] Author (year). Title. Journal. doi:...
      const parts = src.replace(/^\[\d+\]\s*/, '').split('. ');
      const titlePart = parts.length > 1 ? parts[1] || '' : parts[0] || '';
      const authorYear = parts[0] || '';
      const rest = parts.slice(2).join('. ');

      popover.innerHTML = '<div class="cp-title">' + titlePart + '</div>' +
        '<div class="cp-authors">' + authorYear + '</div>' +
        (rest ? '<div class="cp-meta">' + rest + '</div>' : '');

      el.appendChild(popover);
    }
    el.classList.remove('citation-broken');
    el.style.cursor = 'pointer';
    // Phase 54.6.312 — clicking a citation opens a rich preview modal
    // (title + authors + abstract + open-access URL) instead of just
    // scrolling to the source list. Shift+click keeps the old
    // scroll-to-source behaviour for people who already rely on it.
    el.onclick = function(ev) {
      if (ev && (ev.shiftKey || ev.metaKey || ev.ctrlKey)) {
        const target = document.getElementById('source-' + ref);
        if (target) target.scrollIntoView({behavior: 'smooth', block: 'center'});
        return;
      }
      openCitationPreview(ref);
    };
  });

  // Phase 20 — surface broken citations once per page so the user knows
  // the dead links aren't a UI bug but a data problem they can fix by
  // re-running autowrite on the affected draft.
  if (brokenCount > 0) {
    console.warn('[sciknow] ' + brokenCount + ' broken citation(s) ' +
                 'in this draft (orphan refs: ' + Array.from(brokenRefs).sort().join(', ') +
                 '). The draft cites source numbers that aren\'t in the sources panel — ' +
                 'usually a pre-Phase-18 draft. Re-run autowrite to regenerate.');
  }
}

// Build popovers on page load
document.addEventListener('DOMContentLoaded', buildPopovers);

// ── Autowrite Dashboard (Phase 4) ────────────────────────────────────
// Enhanced autowrite that shows convergence chart
let awScores = [];
let awTargetScore = 0.85;

// Phase 33 — autowrite mode picker via a proper modal (replaces the
// triple-prompt() UX from Phase 28). doAutowrite opens the modal;
// confirmAutowrite reads the values and fires the request.
let _awSelectedMode = 'skip';

function selectAwMode(mode) {
  _awSelectedMode = mode;
  document.querySelectorAll('.aw-mode-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.mode === mode);
  });
}

function doAutowrite() {
  if (!currentChapterId) { showEmptyHint("No chapter selected &mdash; click any chapter title in the sidebar to select it, then try again."); return; }

  const isAllSections = !currentSectionType;
  const section = currentSectionType || '__all__';
  const ch = chaptersData.find(c => c.id === currentChapterId);
  const chTitle = ch ? ('Ch.' + ch.num + ': ' + ch.title) : 'this chapter';

  // Configure the modal scope label
  const scopeEl = document.getElementById('aw-config-scope');
  if (scopeEl) {
    scopeEl.textContent = isAllSections
      ? 'Autowrite ALL sections of ' + chTitle
      : 'Autowrite ' + section + ' in ' + chTitle;
  }

  // Reset inputs to defaults
  document.getElementById('aw-config-max-iter').value = '3';
  document.getElementById('aw-config-target-score').value = '0.85';
  _awSelectedMode = 'skip';
  document.querySelectorAll('.aw-mode-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.mode === 'skip');
  });

  // Show the mode section only when running all sections AND some already
  // have drafts — otherwise mode choice is irrelevant.
  const modeSection = document.getElementById('aw-config-mode-section');
  const modeInfo = document.getElementById('aw-config-mode-info');
  if (isAllSections && ch) {
    const nSecs = (ch.sections_meta && ch.sections_meta.length) || 0;
    let nDrafted = 0;
    if (Array.isArray(ch.sections)) {
      const draftedSlugs = new Set(ch.sections.filter(s => s.id).map(s => (s.type || '').toLowerCase()));
      nDrafted = (ch.sections_meta || []).filter(m => draftedSlugs.has(m.slug)).length;
    }
    if (nDrafted > 0) {
      modeSection.style.display = 'block';
      modeInfo.textContent = nDrafted + ' of ' + nSecs + ' sections already have a draft.';
    } else {
      modeSection.style.display = 'none';
    }
  } else {
    modeSection.style.display = 'none';
  }

  openModal('autowrite-config-modal');
}

async function confirmAutowrite() {
  closeModal('autowrite-config-modal');

  const isAllSections = !currentSectionType;
  const section = currentSectionType || '__all__';
  const maxIter = document.getElementById('aw-config-max-iter').value || '3';
  const targetStr = document.getElementById('aw-config-target-score').value || '0.85';
  awTargetScore = parseFloat(targetStr) || 0.85;
  awScores = [];

  const modeRebuild = _awSelectedMode === 'rebuild';
  const modeResume = _awSelectedMode === 'resume';

  showStreamPanel(isAllSections
    ? 'Autowriting all sections...'
    : 'Autowriting ' + section + '...');
  // Phase 15.6 — clear the read-view and prepare it for live writing
  startLiveWrite();

  // Add chart + log to the stream panel
  const body = document.getElementById('stream-body');
  body.innerHTML = '<div class="aw-dashboard">' +
    '<div class="aw-chart" id="aw-chart"><svg viewBox="0 0 400 120"></svg></div>' +
    '<div class="aw-log" id="aw-log"></div>' +
    '</div>' +
    '<div id="aw-content" style="margin-top:12px;white-space:pre-wrap;line-height:1.6;font-family:var(--font-serif);font-size:15px;"></div>';

  // Phase 15 — live stats footer wired to the main stream-stats element
  const stats = createStreamStats('main-stream-stats', 'qwen3.5:27b');
  stats.start();

  const fd = new FormData();
  fd.append('chapter_id', currentChapterId);
  if (!isAllSections) fd.append('section_type', section);
  fd.append('max_iter', maxIter);
  fd.append('target_score', String(awTargetScore));
  if (isAllSections && modeRebuild) fd.append('rebuild', 'true');
  if (isAllSections && modeResume) fd.append('resume', 'true');
  // Phase 54.6.144 — visuals-in-writer opt-in.
  const incVisEl = document.getElementById('aw-config-include-visuals');
  if (incVisEl && incVisEl.checked) fd.append('include_visuals', 'true');
  const endpoint = isAllSections ? '/api/autowrite-chapter' : '/api/autowrite';
  const res = await fetch(endpoint, {method: 'POST', body: fd});
  const data = await res.json();

  // Phase 30 — start the persistent task bar so the live state
  // survives navigation. The bar OWNS its own SSE source; the
  // existing per-section source.onmessage handler below still
  // runs for live preview / dashboard chart, but it doesn't
  // own the lifecycle anymore.
  startGlobalJob(data.job_id, {
    type: 'autowrite',
    taskDesc: isAllSections
      ? 'Autowriting all sections of Ch.' + (chaptersData.find(c => c.id === currentChapterId) || {}).num
      : 'Autowriting ' + section,
    modelName: 'qwen3.5:27b',
    sectionType: section,
    chapterId: currentChapterId,
  });

  // Custom stream handler for autowrite
  currentJobId = data.job_id;
  if (currentEventSource) currentEventSource.close();
  const source = new EventSource('/api/stream/' + data.job_id);
  currentEventSource = source;
  const status = document.getElementById('stream-status');
  const scoresEl = document.getElementById('stream-scores');
  const awContent = document.getElementById('aw-content');
  const awLog = document.getElementById('aw-log');

  source.onmessage = function(e) {
    const evt = JSON.parse(e.data);
    if (evt.type === 'token') {
      // Phase 15.3 — route tokens by phase. Writing/revising tokens go
      // to the visible draft area + main read-view live preview;
      // scoring/verify/CoVe/planning JSON tokens only feed the stats
      // counter (they'd be ugly to show in the draft pane).
      const phase = evt.phase || 'writing';
      stats.update(evt.text);
      stats.setPhase(phase);
      if (phase === 'writing' || phase === 'revising') {
        // Existing autowrite-dashboard token area (now redundant with the
        // read-view live preview, but kept for users who want to focus on
        // the dashboard).
        setStreamCursor(awContent, false);
        awContent.innerHTML += evt.text.replace(/</g, '&lt;').replace(/>/g, '&gt;');
        setStreamCursor(awContent, true);
        awContent.scrollTop = awContent.scrollHeight;
        // Phase 15.6 — also stream into the main read-view as a live
        // markdown preview, so the user sees the writing happen in the
        // same place they'd read the final draft.
        appendLiveWrite(evt.text);
      }
    }
    else if (evt.type === 'progress') {
      status.textContent = evt.detail || evt.stage;
    }
    else if (evt.type === 'scores') {
      awScores.push(evt.scores);
      drawConvergenceChart();
      // Show score bars
      scoresEl.style.display = 'block';
      const s = evt.scores;
      const dims = ['groundedness', 'completeness', 'coherence', 'citation_accuracy', 'hedging_fidelity', 'overall'];
      scoresEl.innerHTML = 'Iteration ' + evt.iteration + ': ' + dims.map(d => {
        const v = (s[d] || 0).toFixed(2);
        const cls = v >= 0.85 ? 'good' : v >= 0.7 ? 'mid' : 'low';
        return '<span class="score-bar"><span class="label">' + d.slice(0,6) + '</span> ' +
          '<span class="value ' + cls + '">' + v + '</span></span>';
      }).join('');
    }
    else if (evt.type === 'cove_verification') {
      // Phase 11 — Chain-of-Verification mismatches in the autowrite stream.
      const cd = evt.data || {};
      const score = (cd.cove_score != null) ? cd.cove_score.toFixed(2) : '?';
      const mismatches = cd.mismatches || [];
      const high = mismatches.filter(m => m.severity === 'high');
      const med = mismatches.filter(m => m.severity === 'medium');
      let icon = high.length ? '\u2717' : med.length ? '\u26a0' : '\u2713';
      let cls = high.length ? 'log-discard' : 'log-keep';
      awLog.innerHTML += '<div class="' + cls + '">' + icon + ' CoVe ' + score +
        ' · ' + high.length + 'H/' + med.length + 'M mismatches</div>';
    }
    else if (evt.type === 'model_info') {
      // Phase 15.5 — model name shown only in the stats footer; the
      // previous awLog line duplicated information that's already in
      // the live stats pill above the dashboard.
      stats.setModel(evt.writer_model || 'qwen3.5:27b');
    }
    else if (evt.type === 'checkpoint') {
      // Phase 15.1 — incremental save checkpoint reached. Show a brief
      // green note in the log so the user knows their work is persisted
      // and Stop won't lose anything past this point.
      awLog.innerHTML += '<div class="log-keep u-tiny">' +
        '\u2693 checkpoint saved · ' + (evt.stage || '') + ' · ' +
        (evt.word_count || 0) + ' words</div>';
    }
    else if (evt.type === 'verification') {
      // Standard verifier in the autowrite stream — emit a brief log line.
      const vd = evt.data || {};
      const grStr = (vd.groundedness_score != null) ? vd.groundedness_score.toFixed(2) : '?';
      const hfStr = (vd.hedging_fidelity_score != null) ? vd.hedging_fidelity_score.toFixed(2) : '?';
      awLog.innerHTML += '<div class="u-dim-7 u-tiny">verify: gr=' + grStr + ' · hf=' + hfStr + '</div>';
    }
    else if (evt.type === 'iteration_start') {
      awContent.innerHTML = '';
      // Phase 15.6 — clear the live preview so each new iteration shows
      // its own writing fresh in the read-view.
      clearLiveWrite();
      awLog.innerHTML += '<div style="opacity:0.5;border-top:1px solid var(--border);padding-top:4px;margin-top:4px;">Iteration ' + evt.iteration + '/' + evt.max + '</div>';
    }
    else if (evt.type === 'revision_verdict') {
      const cls = evt.action === 'KEEP' ? 'log-keep' : 'log-discard';
      const icon = evt.action === 'KEEP' ? '\u2713' : '\u2717';
      awLog.innerHTML += '<div class="' + cls + '">' + icon + ' ' + evt.action +
        ': ' + evt.old_score.toFixed(2) + ' \u2192 ' + evt.new_score.toFixed(2) + '</div>';
      awContent.innerHTML = '';
    }
    else if (evt.type === 'converged') {
      status.textContent = 'Converged at iteration ' + evt.iteration + ' (score: ' + evt.final_score.toFixed(2) + ')';
      awLog.innerHTML += '<div class="log-keep u-bold">\u2713 CONVERGED (score: ' + evt.final_score.toFixed(2) + ')</div>';
    }
    else if (evt.type === 'completed') {
      // Phase 20 — for multi-section runs, this fires once per section.
      // We refresh after each section so the user sees the new draft
      // appear in the sidebar, but DON'T close the stream — there are
      // more sections to write. The all_sections_complete event below
      // closes the stream when truly done.
      if (isAllSections) {
        // Refresh sidebar so the new section's draft becomes clickable.
        // Don't navigate away — the user is watching the live preview.
        fetch('/api/chapters').then(r => r.json()).then(d => {
          rebuildSidebar(d.chapters || d, currentDraftId);
        }).catch(() => {});
      } else {
        status.textContent = 'Done';
        stats.done('done');
        setStreamCursor(awContent, false);
        hideStreamPanel();
        source.close(); currentEventSource = null; currentJobId = null;
        refreshAfterJob(evt.draft_id);
      }
    }
    // Phase 20 — multi-section autowrite envelope events
    else if (evt.type === 'chapter_autowrite_start') {
      awLog.innerHTML += '<div class="u-bold u-border-t u-pt-6 u-mt-6">' +
        '\u270e Chapter autowrite: ' + evt.n_sections + ' sections' +
        (evt.rebuild ? ' (rebuild mode)' : '') + '</div>';
    }
    else if (evt.type === 'section_start') {
      const skip = evt.skipped ? ' [SKIPPED — already drafted]' : '';
      awLog.innerHTML += '<div class="u-semibold u-accent u-border-t u-pt-6 u-mt-6">' +
        '\u25b6 Section ' + evt.index + '/' + evt.total + ': ' + evt.title + skip + '</div>';
      status.textContent = 'Section ' + evt.index + '/' + evt.total + ': ' + evt.title +
        (evt.skipped ? ' (skipping)' : '');
      if (!evt.skipped) {
        // New section starting — clear the previous section's preview.
        clearLiveWrite();
        if (awContent) awContent.innerHTML = '';
      }
    }
    else if (evt.type === 'section_done') {
      const score = (evt.final_score != null) ? ' (' + evt.final_score.toFixed(2) + ')' : '';
      const cls = evt.error ? 'log-discard' : 'log-keep';
      const icon = evt.error ? '\u2717' : evt.skipped ? '\u2014' : '\u2713';
      const note = evt.error ? ' error: ' + evt.error : evt.skipped ? ' skipped' : ' done' + score;
      awLog.innerHTML += '<div class="' + cls + '">' + icon + ' Section ' + evt.index + note + '</div>';
    }
    else if (evt.type === 'section_error') {
      awLog.innerHTML += '<div class="log-discard">\u2717 Section ' + evt.index + ' failed: ' + evt.message + '</div>';
    }
    else if (evt.type === 'all_sections_complete') {
      status.textContent = 'Chapter done: ' + evt.n_completed + '/' + evt.n_total +
        ' written, ' + evt.n_skipped + ' skipped' +
        (evt.n_failed > 0 ? ', ' + evt.n_failed + ' failed' : '');
      awLog.innerHTML += '<div class="log-keep u-bold u-border-t u-pt-6 u-mt-6">' +
        '\u2713 Chapter complete: ' + evt.n_completed + ' written, ' +
        evt.n_skipped + ' skipped, ' + evt.n_failed + ' failed</div>';
      stats.done('done');
      setStreamCursor(awContent, false);
      // Don't auto-hide; let the user read the summary. Close the SSE
      // source so we don't leak the connection.
      source.close(); currentEventSource = null; currentJobId = null;
      // Refresh sidebar so all new section drafts are visible
      fetch('/api/chapters').then(r => r.json()).then(d => {
        rebuildSidebar(d.chapters || d, currentDraftId);
      }).catch(() => {});
    }
    else if (evt.type === 'error') {
      status.textContent = 'Error: ' + evt.message;
      awLog.innerHTML += '<div class="u-danger">' + evt.message + '</div>';
      stats.done('error');
      setStreamCursor(awContent, false);
      hideStreamPanel();
      source.close(); currentEventSource = null; currentJobId = null;
    }
    else if (evt.type === 'done') {
      stats.done('done');
      setStreamCursor(awContent, false);
      hideStreamPanel();
      source.close(); currentEventSource = null; currentJobId = null;
    }
  };
}

// ── Autowrite WHOLE BOOK (every chapter × every section) ───────────
// Phase 54.6.x — fan out autowrite_chapter_all_sections_stream over
// every chapter in book order. Uses the same stream-panel + live
// preview UI as confirmAutowrite(); the only new envelope events are
// `book_autowrite_start`, `chapter_start`, `chapter_done`, and
// `all_chapters_complete`.
async function doAutowriteBook() {
  const ok = confirm(
    "Autowrite EVERY section of EVERY chapter in this book?\n\n" +
    "This is long-running and uses the writer model for every empty " +
    "section. Existing drafts are skipped by default. " +
    "Click OK to proceed, Cancel to abort."
  );
  if (!ok) return;

  awTargetScore = 0.85;
  awScores = [];

  showStreamPanel('Autowriting whole book…');
  startLiveWrite();

  const body = document.getElementById('stream-body');
  body.innerHTML = '<div class="aw-dashboard">'
    + '<div class="aw-chart" id="aw-chart"><svg viewBox="0 0 400 120"></svg></div>'
    + '<div class="aw-log" id="aw-log"></div>'
    + '</div>'
    + '<div id="aw-content" style="margin-top:12px;white-space:pre-wrap;line-height:1.6;font-family:var(--font-serif);font-size:15px;"></div>';

  const stats = createStreamStats('main-stream-stats', 'qwen3.5:27b');
  stats.start();

  const fd = new FormData();
  fd.append('max_iter', '3');
  fd.append('target_score', String(awTargetScore));
  const res = await fetch('/api/autowrite-book', {method: 'POST', body: fd});
  const data = await res.json();
  if (!data || !data.job_id) {
    hideStreamPanel();
    alert('Failed to start whole-book autowrite: ' + (data && data.error || 'unknown'));
    return;
  }

  startGlobalJob(data.job_id, {
    type: 'autowrite_book',
    taskDesc: 'Autowriting whole book',
    modelName: 'qwen3.5:27b',
  });

  currentJobId = data.job_id;
  if (currentEventSource) currentEventSource.close();
  const source = new EventSource('/api/stream/' + data.job_id);
  currentEventSource = source;
  const status = document.getElementById('stream-status');
  const awContent = document.getElementById('aw-content');
  const awLog = document.getElementById('aw-log');

  source.onmessage = function(e) {
    let evt;
    try { evt = JSON.parse(e.data); } catch (_) { return; }
    if (evt.type === 'token') {
      const phase = evt.phase || 'writing';
      stats.update(evt.text);
      stats.setPhase(phase);
      if (phase === 'writing' || phase === 'revising') {
        setStreamCursor(awContent, false);
        awContent.innerHTML += evt.text.replace(/</g, '&lt;').replace(/>/g, '&gt;');
        setStreamCursor(awContent, true);
        awContent.scrollTop = awContent.scrollHeight;
        appendLiveWrite(evt.text);
      }
    }
    else if (evt.type === 'progress') {
      status.textContent = evt.detail || evt.stage;
    }
    else if (evt.type === 'scores') {
      awScores.push(evt.scores);
      drawConvergenceChart();
    }
    else if (evt.type === 'book_autowrite_start') {
      awLog.innerHTML += '<div class="u-bold u-border-t u-pt-6 u-mt-6">'
        + '✎ Whole book autowrite: ' + evt.n_chapters + ' chapters'
        + (evt.rebuild ? ' (rebuild mode)' : '')
        + (evt.resume ? ' (resume mode)' : '')
        + '</div>';
    }
    else if (evt.type === 'chapter_start') {
      awLog.innerHTML += '<div class="u-semibold u-accent u-border-t u-pt-6 u-mt-6">'
        + '◆ Chapter ' + evt.chapter_index + '/' + evt.chapter_total
        + ': ' + (evt.chapter_title || '(untitled)') + '</div>';
      status.textContent = 'Chapter ' + evt.chapter_index + '/' + evt.chapter_total
        + ': ' + (evt.chapter_title || '');
      clearLiveWrite();
      if (awContent) awContent.innerHTML = '';
    }
    else if (evt.type === 'chapter_autowrite_start') {
      awLog.innerHTML += '<div class="u-tiny u-muted">  '
        + evt.n_sections + ' sections in this chapter</div>';
    }
    else if (evt.type === 'section_start') {
      const skip = evt.skipped ? ' [SKIPPED — already drafted]' : '';
      awLog.innerHTML += '<div class="u-md u-pl-4">  ▶ Section '
        + evt.index + '/' + evt.total + ': ' + evt.title + skip + '</div>';
      if (!evt.skipped) {
        clearLiveWrite();
        if (awContent) awContent.innerHTML = '';
      }
    }
    else if (evt.type === 'section_done') {
      const score = (evt.final_score != null) ? ' (' + evt.final_score.toFixed(2) + ')' : '';
      const cls = evt.error ? 'log-discard' : 'log-keep';
      const icon = evt.error ? '✗' : evt.skipped ? '—' : '✓';
      const note = evt.error ? ' error: ' + evt.error : evt.skipped ? ' skipped' : ' done' + score;
      awLog.innerHTML += '<div class="' + cls + '">    ' + icon + ' Section ' + evt.index + note + '</div>';
    }
    else if (evt.type === 'section_error') {
      awLog.innerHTML += '<div class="log-discard">    ✗ Section ' + evt.index + ' failed: ' + evt.message + '</div>';
    }
    else if (evt.type === 'all_sections_complete') {
      awLog.innerHTML += '<div class="log-keep">  ✓ chapter complete: '
        + evt.n_completed + ' written, ' + evt.n_skipped + ' skipped, '
        + evt.n_failed + ' failed</div>';
      fetch('/api/chapters').then(r => r.json()).then(d => {
        rebuildSidebar(d.chapters || d, currentDraftId);
      }).catch(() => {});
    }
    else if (evt.type === 'chapter_done') {
      // Inner all_sections_complete already logged the totals.
    }
    else if (evt.type === 'chapter_error') {
      awLog.innerHTML += '<div class="log-discard">  ✗ Chapter ' + evt.chapter_index + ' failed: ' + (evt.message || '') + '</div>';
    }
    else if (evt.type === 'all_chapters_complete') {
      status.textContent = 'Book autowrite complete: '
        + evt.n_chapters_completed + '/' + evt.n_chapters + ' chapters · '
        + evt.n_sections_completed + ' sections written';
      awLog.innerHTML += '<div class="log-keep u-bold u-border-t u-pt-6 u-mt-6">'
        + '✓ Book complete: ' + evt.n_chapters_completed + '/' + evt.n_chapters
        + ' chapters · ' + evt.n_sections_completed + ' sections written, '
        + evt.n_sections_skipped + ' skipped, '
        + evt.n_sections_failed + ' failed</div>';
      stats.done('done');
      setStreamCursor(awContent, false);
      source.close(); currentEventSource = null; currentJobId = null;
      fetch('/api/chapters').then(r => r.json()).then(d => {
        rebuildSidebar(d.chapters || d, currentDraftId);
      }).catch(() => {});
    }
    else if (evt.type === 'error') {
      status.textContent = 'Error: ' + evt.message;
      awLog.innerHTML += '<div class="u-danger">' + evt.message + '</div>';
      stats.done('error');
      setStreamCursor(awContent, false);
      hideStreamPanel();
      source.close(); currentEventSource = null; currentJobId = null;
    }
    else if (evt.type === 'done') {
      stats.done('done');
      setStreamCursor(awContent, false);
      source.close(); currentEventSource = null; currentJobId = null;
    }
  };
}

function drawConvergenceChart() {
  const svg = document.querySelector('#aw-chart svg');
  if (!svg || awScores.length === 0) return;

  const w = 400, h = 120, pad = 20;
  const n = awScores.length;
  const maxN = Math.max(n, 3);

  let html = '';
  // Target line
  const ty = h - pad - (awTargetScore * (h - 2 * pad));
  html += '<line x1="' + pad + '" y1="' + ty + '" x2="' + (w - pad) + '" y2="' + ty + '" class="chart-target"/>';
  html += '<text x="' + (w - pad + 4) + '" y="' + (ty + 3) + '" font-size="9" fill="var(--success)">' + awTargetScore + '</text>';

  // Score line
  const points = awScores.map((s, i) => {
    const x = pad + (i / (maxN - 1)) * (w - 2 * pad);
    const y = h - pad - ((s.overall || 0) * (h - 2 * pad));
    return x + ',' + y;
  });
  html += '<polyline points="' + points.join(' ') + '" class="chart-line"/>';

  // Dots
  awScores.forEach((s, i) => {
    const x = pad + (i / (maxN - 1)) * (w - 2 * pad);
    const y = h - pad - ((s.overall || 0) * (h - 2 * pad));
    html += '<circle cx="' + x + '" cy="' + y + '" r="4" class="chart-dot"/>';
    html += '<text x="' + x + '" y="' + (y - 8) + '" font-size="10" text-anchor="middle" fill="var(--fg)">' + (s.overall || 0).toFixed(2) + '</text>';
  });

  // Axes labels
  html += '<text x="' + pad + '" y="' + (h - 2) + '" font-size="9" fill="var(--fg)" opacity="0.5">1</text>';
  if (n > 1) html += '<text x="' + (w - pad) + '" y="' + (h - 2) + '" font-size="9" fill="var(--fg)" opacity="0.5" text-anchor="end">' + n + '</text>';

  svg.innerHTML = html;
}

// ── Argument Map Visualization (Phase 5c) ─────────────────────────────
async function promptArgue() {
  const claim = prompt('Enter a claim to map evidence for/against:');
  if (!claim) return;
  showStreamPanel('Mapping argument...');

  const fd = new FormData();
  fd.append('claim', claim);
  const res = await fetch('/api/argue', {method: 'POST', body: fd});
  const data = await res.json();

  // Custom handler that parses the argue output and builds a map
  currentJobId = data.job_id;
  if (currentEventSource) currentEventSource.close();
  const source = new EventSource('/api/stream/' + data.job_id);
  currentEventSource = source;
  const body = document.getElementById('stream-body');
  const status = document.getElementById('stream-status');
  let fullText = '';

  source.onmessage = function(e) {
    const evt = JSON.parse(e.data);
    if (evt.type === 'token') {
      fullText += evt.text;
      body.innerHTML = fullText.replace(/</g, '&lt;').replace(/>/g, '&gt;');
      body.scrollTop = body.scrollHeight;
    }
    else if (evt.type === 'progress') {
      status.textContent = evt.detail || evt.stage;
    }
    else if (evt.type === 'completed') {
      status.textContent = 'Done';
      hideStreamPanel();
      source.close(); currentEventSource = null; currentJobId = null;
      // Build visual argument map from the text
      buildArgueMap(claim, fullText);
      if (evt.draft_id) refreshAfterJob(evt.draft_id);
    }
    else if (evt.type === 'error') {
      status.textContent = 'Error: ' + evt.message;
      hideStreamPanel();
      source.close(); currentEventSource = null; currentJobId = null;
    }
    else if (evt.type === 'done') {
      hideStreamPanel();
      source.close(); currentEventSource = null; currentJobId = null;
    }
  };
}

function buildArgueMap(claim, text) {
  // Parse the structured argue output to extract supporting/contradicting/neutral
  const sections = {supporting: [], contradicting: [], neutral: []};
  let currentSection = null;

  text.split('\n').forEach(line => {
    const lower = line.toLowerCase();
    if (lower.includes('evidence supporting') || lower.includes('supports')) currentSection = 'supporting';
    else if (lower.includes('counterargument') || lower.includes('contradict')) currentSection = 'contradicting';
    else if (lower.includes('methodological') || lower.includes('neutral')) currentSection = 'neutral';
    else if (lower.includes('assessment') || lower.includes('overall')) currentSection = null;

    // Extract citations from the line
    const cites = line.match(/\[\d+\]/g);
    if (currentSection && cites) {
      cites.forEach(c => {
        if (!sections[currentSection].includes(c)) sections[currentSection].push(c);
      });
    }
  });

  const mapDiv = document.getElementById('argue-map-view');
  if (sections.supporting.length === 0 && sections.contradicting.length === 0 && sections.neutral.length === 0) {
    mapDiv.style.display = 'none';
    return;
  }

  const w = 700, h = 350;
  const cx = w / 2, cy = h / 2;

  let svg = '<svg viewBox="0 0 ' + w + ' ' + h + '" xmlns="http://www.w3.org/2000/svg">';

  // Central claim
  svg += '<rect x="' + (cx - 120) + '" y="' + (cy - 20) + '" width="240" height="40" rx="8" fill="var(--accent)" opacity="0.9"/>';
  svg += '<text x="' + cx + '" y="' + (cy + 5) + '" text-anchor="middle" fill="white" font-size="12" font-weight="bold">' +
    claim.substring(0, 40) + (claim.length > 40 ? '...' : '') + '</text>';

  // Supporting (left)
  sections.supporting.forEach((c, i) => {
    const ny = 30 + i * 45;
    const nx = 80;
    svg += '<line x1="' + nx + '" y1="' + ny + '" x2="' + (cx - 120) + '" y2="' + cy + '" class="link-supports"/>';
    svg += '<rect x="' + (nx - 40) + '" y="' + (ny - 14) + '" width="80" height="28" rx="6" fill="var(--success)" opacity="0.2" stroke="var(--success)"/>';
    svg += '<text x="' + nx + '" y="' + (ny + 4) + '" text-anchor="middle" font-size="12" fill="var(--fg)">' + c + '</text>';
  });

  // Contradicting (right)
  sections.contradicting.forEach((c, i) => {
    const ny = 30 + i * 45;
    const nx = w - 80;
    svg += '<line x1="' + nx + '" y1="' + ny + '" x2="' + (cx + 120) + '" y2="' + cy + '" class="link-contradicts"/>';
    svg += '<rect x="' + (nx - 40) + '" y="' + (ny - 14) + '" width="80" height="28" rx="6" fill="var(--danger)" opacity="0.2" stroke="var(--danger)"/>';
    svg += '<text x="' + nx + '" y="' + (ny + 4) + '" text-anchor="middle" font-size="12" fill="var(--fg)">' + c + '</text>';
  });

  // Neutral (bottom)
  sections.neutral.forEach((c, i) => {
    const nx = cx - 100 + i * 70;
    const ny = h - 40;
    svg += '<line x1="' + nx + '" y1="' + ny + '" x2="' + cx + '" y2="' + (cy + 20) + '" class="link-neutral"/>';
    svg += '<rect x="' + (nx - 30) + '" y="' + (ny - 14) + '" width="60" height="28" rx="6" fill="var(--border)" opacity="0.3" stroke="var(--border)"/>';
    svg += '<text x="' + nx + '" y="' + (ny + 4) + '" text-anchor="middle" font-size="12" fill="var(--fg)">' + c + '</text>';
  });

  // Legend
  svg += '<text x="10" y="' + (h - 5) + '" font-size="10" fill="var(--success)">\u25cf Supports</text>';
  svg += '<text x="100" y="' + (h - 5) + '" font-size="10" fill="var(--danger)">\u25cf Contradicts</text>';
  svg += '<text x="210" y="' + (h - 5) + '" font-size="10" fill="var(--fg)" opacity="0.5">\u25cf Neutral</text>';

  svg += '</svg>';

  mapDiv.innerHTML = '<div class="argue-map">' + svg + '</div>';
  mapDiv.style.display = 'block';
}

// ── Phase 14: Modal infrastructure ────────────────────────────────────
// Phase 54.6.178 — routed views. openModal / closeModal now keep the
// URL in sync with the open modal. Opening a modal id in the route
// table pushes its path; closing any routed modal pops back to '/'.
// Browser back/forward navigates the modal stack.
const _MODAL_ROUTES = {
  'plan-modal':          '/plan',
  'book-settings-modal': '/settings',
  'wiki-modal':          '/wiki',
  'bundles-modal':       '/bundles',
  'tools-modal':         '/tools',
  'projects-modal':      '/projects',
  'catalog-modal':       '/catalog',
  'export-modal':        '/export',
  'corpus-modal':        '/corpus',
  'viz-modal':           '/visualize',
  'kg-modal':            '/kg',
  'ask-modal':           '/ask',
  'setup-modal':         '/setup',
  'backups-modal':       '/backups',
  'visuals-modal':       '/visuals',
  'ai-help-modal':       '/help',
};
function _modalSuppressRoute() {
  // Allow the auto-open boot handler to open modals without
  // re-pushing the state it was just navigated to.
  return window._routeNavigating === true;
}
function openModal(id) {
  document.getElementById(id).classList.add('open');
  const path = _MODAL_ROUTES[id];
  if (path) {
    // Phase 54.6.200 — routed modals render as full-page views
    // (body.routed-view). Toggling this class is how the CSS
    // switches the overlay from floating-scrim to full-page.
    document.body.classList.add('routed-view');
  }
  if (path && !_modalSuppressRoute() && window.location.pathname !== path) {
    history.pushState({ modal: id }, '', path);
  }
}
function closeModal(id) {
  document.getElementById(id).classList.remove('open');
  // Stop any in-flight job that the modal launched
  if (currentEventSource && (id === 'wiki-modal' || id === 'ask-modal')) {
    try { currentEventSource.close(); } catch (e) {}
    currentEventSource = null;
    if (currentJobId) {
      fetch('/api/jobs/' + currentJobId, {method: 'DELETE'}).catch(() => {});
      currentJobId = null;
    }
  }
  // Phase 54.6.200 — clear the full-page routed-view state if no other
  // routed modal is still open.
  const stillOpen = document.querySelector('.modal-overlay.open');
  if (!stillOpen) document.body.classList.remove('routed-view');
  // If we closed a routed modal and the current URL points at it, pop
  // the history so the URL returns to the book root.
  const path = _MODAL_ROUTES[id];
  if (path && !_modalSuppressRoute() && window.location.pathname === path) {
    history.pushState({}, '', '/');
  }
}
// Browser back/forward — walk the modal stack. If the new URL matches
// a modal route, open that modal; otherwise close any open modal.
window.addEventListener('popstate', function () {
  window._routeNavigating = true;
  try {
    const path = window.location.pathname;
    let routedOpen = false;
    // Close any currently-open modal that doesn't match
    document.querySelectorAll('.modal-overlay.open').forEach(function (m) {
      const id = m.id;
      if (_MODAL_ROUTES[id] !== path) m.classList.remove('open');
    });
    // Open the matching one, if any
    for (const id in _MODAL_ROUTES) {
      if (_MODAL_ROUTES[id] === path) {
        const el = document.getElementById(id);
        if (el && !el.classList.contains('open')) el.classList.add('open');
        routedOpen = true;
        break;
      }
    }
    // Phase 54.6.200 — sync the full-page routed-view body class
    // so back/forward can exit the full-page view cleanly.
    document.body.classList.toggle('routed-view', routedOpen);
  } finally {
    window._routeNavigating = false;
  }
});
// Escape closes any open modal
// ── Phase 33: keyboard shortcuts ──────────────────────────────────────
//
// Global keydown handler for the web reader. All shortcuts are designed
// to be non-destructive and avoid conflicts with browser defaults:
//   Esc           — close any open modal (already existed since Phase 14)
//   Ctrl+S        — force save in editor (suppresses the browser's "Save as")
//   Ctrl+K        — focus the sidebar search bar
//   Ctrl+E        — toggle the inline editor
//   ← / →         — navigate to previous / next section in the sidebar
//                   (only when focus is NOT in an input/textarea/select)
//   D             — show dashboard (same exclusion)
//   P             — open plan modal (same exclusion)
//
// The "not in an input" guard prevents letter shortcuts from swallowing
// keystrokes while the user types in a form field, textarea, search bar,
// or the editor. Ctrl+* shortcuts work everywhere because the user
// explicitly holds Ctrl.

document.addEventListener('keydown', function(e) {
  const tag = (e.target.tagName || '').toUpperCase();
  const inInput = (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT'
                   || e.target.isContentEditable);

  // ── Esc — close the closest open overlay, or go home ────────────
  // Phase 54.6.193: progressive disclosure — Escape peels one layer
  // off at a time, and when nothing's open it navigates to `/` so
  // the user always has a reliable "get me out of here" key.
  // Priority: open modal → ⌘K palette → open nav dropdown → home.
  // Typing inside an input is respected (browser blurs the field
  // on Escape and we stay put) so Escape never navigates away from
  // an unsaved edit unless the user presses it a second time.
  if (e.key === 'Escape') {
    const openModals = document.querySelectorAll('.modal-overlay.open');
    if (openModals.length > 0) {
      openModals.forEach(m => closeModal(m.id));
      return;
    }
    const cmdk = document.getElementById('cmdk');
    if (cmdk && cmdk.style.display === 'block') {
      closeCmdK();
      return;
    }
    const openDropdown = document.querySelector('.nav-dropdown.open');
    if (openDropdown) {
      openDropdown.classList.remove('open');
      return;
    }
    if (inInput) return;
    const dashView = document.getElementById('dashboard-view');
    const onOverlayView = dashView && dashView.style.display === 'block';
    if (window.location.pathname !== '/' || onOverlayView) {
      window.location.href = '/';
    }
    return;
  }

  // ── Ctrl+S — force save in editor ────────────────────────────────
  if ((e.ctrlKey || e.metaKey) && e.key === 's') {
    e.preventDefault();  // suppress browser "Save page as..."
    const ev = document.getElementById('edit-view');
    if (ev && ev.style.display !== 'none') {
      edAutosave();
    }
    return;
  }

  // ── Ctrl+K — focus search bar ────────────────────────────────────
  if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
    e.preventDefault();
    const searchInput = document.querySelector('.search-bar input');
    if (searchInput) {
      searchInput.focus();
      searchInput.select();
    }
    return;
  }

  // ── Ctrl+E — toggle editor ───────────────────────────────────────
  if ((e.ctrlKey || e.metaKey) && e.key === 'e') {
    e.preventDefault();
    if (currentDraftId) toggleEdit();
    return;
  }

  // ── Letter shortcuts (only when NOT typing in a field) ───────────
  if (inInput) return;

  // ← / → — previous / next section
  if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
    const links = Array.from(document.querySelectorAll('#sidebar-sections .sec-link[href]'));
    if (links.length === 0) return;
    const idx = links.findIndex(a => a.classList.contains('active'));
    let next;
    if (e.key === 'ArrowLeft') {
      next = idx > 0 ? links[idx - 1] : links[links.length - 1];
    } else {
      next = idx < links.length - 1 ? links[idx + 1] : links[0];
    }
    if (next) {
      next.click();
      next.scrollIntoView({block: 'nearest'});
    }
    return;
  }

  // D — dashboard
  if (e.key === 'd' || e.key === 'D') {
    showDashboard();
    return;
  }

  // P — plan modal
  if (e.key === 'p' || e.key === 'P') {
    openPlanModal();
    return;
  }
});

// ── Phase 14: Score history viewer (Phase 13 GUI integration) ─────────
async function showScoresPanel() {
  if (!currentDraftId) { showEmptyHint("No draft selected &mdash; click a section in the sidebar, or click <strong>Start writing</strong> under any chapter to create a first draft."); return; }
  const panel = document.getElementById('scores-panel');
  const body = document.getElementById('scores-panel-body');
  panel.classList.add('open');
  body.innerHTML = '<div class="scores-empty">Loading...</div>';

  try {
    const res = await fetch('/api/draft/' + currentDraftId + '/scores');
    if (!res.ok) {
      body.innerHTML = '<div class="scores-empty">Draft not found.</div>';
      return;
    }
    const data = await res.json();
    const history = data.score_history || [];

    if (history.length === 0) {
      body.innerHTML = '<div class="scores-empty">' +
        'No score history persisted on this draft.<br>' +
        '<span class="u-tiny">Only autowrite drafts record convergence trajectories — drafts made with `book write` only have a final state.</span>' +
        '</div>';
      return;
    }

    // Build the iteration table.
    const dims = ['groundedness', 'completeness', 'coherence', 'citation_accuracy', 'hedging_fidelity', 'overall'];
    let html = '<table class="scores-table"><thead><tr><th>Iter</th>';
    dims.forEach(d => { html += '<th>' + d.slice(0, 6) + '</th>'; });
    html += '<th>Weakest</th><th>Verdict</th></tr></thead><tbody>';

    history.forEach(h => {
      const s = h.scores || {};
      html += '<tr><td>' + h.iteration + '</td>';
      dims.forEach(d => {
        const v = s[d];
        if (v == null) { html += '<td>—</td>'; }
        else {
          const cls = v >= 0.85 ? 'score-good' : v >= 0.7 ? 'score-mid' : 'score-low';
          html += '<td class="' + cls + '">' + Number(v).toFixed(2) + '</td>';
        }
      });
      html += '<td>' + (s.weakest_dimension || '—') + '</td>';
      const v = h.revision_verdict || '—';
      const verdictColor = v === 'KEEP' ? 'var(--success)' : v === 'DISCARD' ? 'var(--danger)' : 'var(--fg-faint)';
      html += '<td style="color:' + verdictColor + '">' + v + '</td>';
      html += '</tr>';
    });
    html += '</tbody></table>';

    // CoVe + verification summary line
    const cove_runs = history.filter(h => h.cove && h.cove.ran);
    const total_overstated = history.reduce((acc, h) => acc + ((h.verification && h.verification.n_overstated) || 0), 0);
    const total_extrapolated = history.reduce((acc, h) => acc + ((h.verification && h.verification.n_extrapolated) || 0), 0);
    if (cove_runs.length || total_overstated || total_extrapolated) {
      html += '<div style="padding:8px 16px;font-size:11px;color:var(--fg-muted);background:var(--toolbar-bg);border-top:1px solid var(--border);">';
      html += '<strong>Verification:</strong> ' + total_overstated + ' OVERSTATED · ' + total_extrapolated + ' EXTRAPOLATED across ' + history.length + ' iterations';
      if (cove_runs.length) {
        html += '  ·  CoVe ran in ' + cove_runs.length + '/' + history.length + ' iterations';
      }
      html += '</div>';
    }

    // Sparkline for overall score trajectory
    const overalls = history.map(h => (h.scores && h.scores.overall) || 0);
    if (overalls.length >= 2) {
      const w = 380, hgt = 50, pad = 4;
      const minV = Math.min(...overalls, 0.5);
      const maxV = Math.max(...overalls, 1.0);
      const range = maxV - minV || 1;
      const points = overalls.map((v, i) => {
        const x = pad + (i / (overalls.length - 1)) * (w - 2 * pad);
        const y = hgt - pad - ((v - minV) / range) * (hgt - 2 * pad);
        return x + ',' + y;
      }).join(' ');
      html += '<div class="scores-spark">';
      html += '<div class="u-note-mb-1">Overall score trajectory</div>';
      html += '<svg viewBox="0 0 ' + w + ' ' + hgt + '" preserveAspectRatio="none">';
      html += '<polyline points="' + points + '" fill="none" stroke="var(--accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>';
      overalls.forEach((v, i) => {
        const x = pad + (i / (overalls.length - 1)) * (w - 2 * pad);
        const y = hgt - pad - ((v - minV) / range) * (hgt - 2 * pad);
        html += '<circle cx="' + x + '" cy="' + y + '" r="3" fill="var(--accent)"/>';
      });
      html += '</svg></div>';
    }

    // Final overall + features active
    if (data.final_overall != null) {
      html += '<div style="padding:8px 16px;font-size:11px;color:var(--fg-muted);">';
      html += 'Final overall: <strong>' + Number(data.final_overall).toFixed(3) + '</strong>';
      if (data.target_score) html += '  ·  target: ' + data.target_score;
      if (data.max_iter) html += '  ·  max_iter: ' + data.max_iter;
      html += '</div>';
    }

    body.innerHTML = html;
  } catch (e) {
    body.innerHTML = '<div class="scores-empty u-danger">Error loading score history: ' + e.message + '</div>';
  }
}

// ── Phase 15: live streaming stats helper ─────────────────────────────
//
// Returns an object with `start()`, `update(text)`, and `done(state)`
// methods that maintain a small footer showing rolling tok/s, total
// tokens, elapsed time, time-to-first-token, and the model name.
//
// Pure client-side — uses performance.now() timestamps and counts
// whitespace-delimited tokens (a rough proxy for actual model tokens
// but accurate enough for live feedback).
// Phase 15.4 — build tag the user can check from the browser console:
//   typeof STREAM_STATS_BUILD === "string" && STREAM_STATS_BUILD
// If undefined, the page is running stale JS — hard-refresh (Ctrl+Shift+R).
const STREAM_STATS_BUILD = "phase-15.4";

function createStreamStats(containerId, modelName) {
  const el = document.getElementById(containerId);
  if (!el) return { start: ()=>{}, update: ()=>{}, done: ()=>{}, setModel: ()=>{}, setPhase: ()=>{} };
  let started = 0, firstTok = 0, lastTok = 0, count = 0;
  let recentTokens = []; // sliding window for rolling tok/s
  let timer = null;
  let currentModel = modelName || '?';
  let currentPhase = '';
  // Phase 15.4 — debug logging: prints once when the FIRST token is
  // received so the user can confirm in DevTools Console that token
  // events are actually flowing through stats.update().
  let _firstUpdateLogged = false;

  function fmtTime(ms) {
    const s = ms / 1000;
    if (s < 60) return s.toFixed(1) + 's';
    const m = Math.floor(s / 60);
    return m + 'm ' + Math.floor(s % 60).toString().padStart(2, '0') + 's';
  }

  function render(state) {
    const elapsed = started ? performance.now() - started : 0;
    const ttft = firstTok ? firstTok - started : 0;
    // Rolling tok/s over the last 3 seconds
    const cutoff = performance.now() - 3000;
    const recent = recentTokens.filter(t => t > cutoff);
    const tps = recent.length / 3;
    const avgTps = (firstTok && elapsed > 0) ? count / ((elapsed - ttft) / 1000) : 0;

    el.className = 'stream-stats ' + (state || 'streaming');
    el.innerHTML =
      '<span class="ss-dot"></span>' +
      '<span class="ss-stat"><strong>' + currentModel + '</strong></span>' +
      (currentPhase ? '<span class="ss-sep">·</span><span class="ss-stat ss-phase">' + currentPhase + '</span>' : '') +
      '<span class="ss-sep">·</span>' +
      '<span class="ss-stat"><strong>' + count + '</strong>&nbsp;tok</span>' +
      '<span class="ss-sep">·</span>' +
      '<span class="ss-stat"><strong>' + tps.toFixed(1) + '</strong>&nbsp;tok/s</span>' +
      (avgTps > 0 ? '<span class="ss-sep">·</span><span class="ss-stat" title="Average since first token">avg <strong>' + avgTps.toFixed(1) + '</strong></span>' : '') +
      '<span class="ss-sep">·</span>' +
      '<span class="ss-stat">' + fmtTime(elapsed) + '</span>' +
      (ttft > 0 ? '<span class="ss-sep">·</span><span class="ss-stat" title="Time to first token">ttft ' + fmtTime(ttft) + '</span>' : '');
  }

  return {
    start() {
      started = performance.now();
      firstTok = 0; count = 0; recentTokens = [];
      el.style.display = 'flex';
      render('streaming');
      if (timer) clearInterval(timer);
      timer = setInterval(() => render('streaming'), 200);
    },
    update(text) {
      if (!started) this.start();
      if (!firstTok) {
        firstTok = performance.now();
        if (!_firstUpdateLogged) {
          _firstUpdateLogged = true;
          console.log('[stream-stats] first token received',
            { text: (text || '').slice(0, 40), build: STREAM_STATS_BUILD });
        }
      }
      // Phase 15.4 — defensive: text might be non-string if a producer
      // emits malformed events. Coerce so the regex below doesn't throw.
      const safeText = (typeof text === 'string') ? text : String(text || '');
      // Approximate tokens via whitespace + punctuation splits.
      const n = (safeText.match(/\S+/g) || []).length;
      count += n;
      const now = performance.now();
      for (let i = 0; i < n; i++) recentTokens.push(now);
    },
    done(state) {
      lastTok = performance.now();
      if (timer) { clearInterval(timer); timer = null; }
      render(state || 'done');
    },
    setModel(m) {
      if (m) currentModel = m;
    },
    setPhase(p) {
      currentPhase = p || '';
    },
  };
}

// Helper that adds a blinking cursor to an element while streaming.
function setStreamCursor(el, on) {
  if (!el) return;
  let cursor = el.querySelector('.stream-cursor');
  if (on && !cursor) {
    cursor = document.createElement('span');
    cursor.className = 'stream-cursor';
    el.appendChild(cursor);
  } else if (!on && cursor) {
    cursor.remove();
  }
}

// ── Phase 15.6: live writing preview in the main read-view ──────────
//
// As writing/revising tokens stream in, accumulate them and re-render
// the read-view as live HTML — same Georgia serif body the final draft
// uses, with paragraph breaks and [N] citation styling. On completion,
// refreshAfterJob() reloads the saved draft from the API and the live
// preview gets replaced with the proper rendered version.

let _liveWriteText = '';
let _liveWriteActive = false;

function _escapeHtml(s) {
  return (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function _renderLiveMarkdown(text) {
  // Cheap, fast HTML rendering of streaming markdown:
  //   - Escape HTML first
  //   - Style [N] citations like the final view
  //   - Convert \n\n into paragraph boundaries
  //   - Convert single \n inside a paragraph into <br>
  // This won't handle every markdown feature (no headings, no bold, no
  // lists) but it's enough to make the streaming text look like prose
  // instead of console output.
  const escaped = _escapeHtml(text);
  const withCitations = escaped.replace(/\[(\d+)\]/g,
    '<span class="citation" data-ref="$1">[$1]</span>');
  const paragraphs = withCitations.split(/\n\n+/);
  return paragraphs.map(p => '<p data-live="1">' + p.replace(/\n/g, '<br>') + '</p>').join('');
}

function clearLiveWrite() {
  _liveWriteText = '';
  _liveWriteActive = false;
}

function startLiveWrite() {
  _liveWriteText = '';
  _liveWriteActive = true;
  const rv = document.getElementById('read-view');
  if (rv) {
    rv.innerHTML =
      '<div class="live-writing-banner">&#9998; Writing live &mdash; this preview will be replaced with the saved draft when generation completes.</div>' +
      '<div class="live-writing-body" id="live-writing-body"></div>';
  }
}

function appendLiveWrite(text) {
  if (!_liveWriteActive) startLiveWrite();
  _liveWriteText += (text || '');
  const body = document.getElementById('live-writing-body');
  if (body) {
    body.innerHTML = _renderLiveMarkdown(_liveWriteText);
    // Add a blinking cursor to the very last paragraph
    const lastP = body.querySelector('p:last-child');
    if (lastP) setStreamCursor(lastP, true);
    // Auto-scroll the read-view to follow new tokens
    const rv = document.getElementById('read-view');
    if (rv) rv.scrollTop = rv.scrollHeight;
  }
}

// ── Phase 14: Wiki Query modal ────────────────────────────────────────
let wikiCurrentTab = 'wiki-query';
function switchWikiTab(name) {
  wikiCurrentTab = name;
  document.querySelectorAll('#wiki-modal .tab').forEach(t => {
    t.classList.toggle('active', t.dataset.tab === name);
  });
  // Phase 54.6.61 — wiki-summaries + wiki-visuals added to the registry.
  ['wiki-query', 'wiki-summaries', 'wiki-visuals',
   'wiki-browse', 'wiki-lint', 'wiki-consensus'].forEach(tab => {
    const pane = document.getElementById(tab + '-pane');
    if (pane) pane.style.display = (name === tab) ? 'block' : 'none';
  });
  if (name === 'wiki-browse') loadWikiPages(1);
  if (name === 'wiki-summaries') loadWikiSummaries();
  if (name === 'wiki-visuals') loadWikiVisuals();
}

// ── Phase 54.6.2 — Wiki Lint + Consensus (surface CLI in GUI) ───────────
let _wikiLintJob = null;
let _wikiLintSource = null;

async function doWikiLint() {
  const deep = document.getElementById('wiki-lint-deep').checked;
  const runBtn = document.getElementById('wiki-lint-run');
  const stopBtn = document.getElementById('wiki-lint-stop');
  const status = document.getElementById('wiki-lint-status');
  const summary = document.getElementById('wiki-lint-summary');
  const issuesEl = document.getElementById('wiki-lint-issues');
  runBtn.disabled = true;
  stopBtn.style.display = 'inline-block';
  status.textContent = 'Running structural checks…';
  summary.innerHTML = '';
  issuesEl.innerHTML = '';

  const fd = new FormData();
  fd.append('deep', deep);
  let res;
  try {
    res = await fetch('/api/wiki/lint', {method: 'POST', body: fd});
  } catch (exc) {
    status.textContent = 'Request failed: ' + exc.message;
    runBtn.disabled = false; stopBtn.style.display = 'none';
    return;
  }
  const data = await res.json();
  _wikiLintJob = data.job_id;
  const source = new EventSource('/api/stream/' + _wikiLintJob);
  _wikiLintSource = source;
  const bySeverity = { high: [], medium: [], low: [] };

  source.onmessage = function(e) {
    const evt = JSON.parse(e.data);
    if (evt.type === 'progress') {
      status.textContent = evt.detail || evt.stage;
    } else if (evt.type === 'lint_issue') {
      const sev = (evt.severity || 'low').toLowerCase();
      (bySeverity[sev] || bySeverity.low).push(evt);
      _renderWikiLintIssues(bySeverity);
    } else if (evt.type === 'completed') {
      const n = evt.issues_count || 0;
      status.textContent = n === 0 ? 'All checks passed.' : (n + ' issue(s) found.');
      summary.innerHTML = (n === 0)
        ? '<div class="u-success u-bold">&#10003; Wiki is clean.</div>'
        : '<div>'
          + '<span class="u-danger u-bold">' + (bySeverity.high.length) + '</span> high · '
          + '<span class="u-warning u-bold">' + (bySeverity.medium.length) + '</span> medium · '
          + '<span class="u-muted u-bold">' + (bySeverity.low.length) + '</span> low'
          + '</div>';
      source.close(); _wikiLintSource = null; _wikiLintJob = null;
      runBtn.disabled = false; stopBtn.style.display = 'none';
    } else if (evt.type === 'error') {
      status.textContent = 'Error: ' + evt.message;
      source.close(); _wikiLintSource = null; _wikiLintJob = null;
      runBtn.disabled = false; stopBtn.style.display = 'none';
    } else if (evt.type === 'done') {
      source.close(); _wikiLintSource = null; _wikiLintJob = null;
      runBtn.disabled = false; stopBtn.style.display = 'none';
    }
  };
}

function _renderWikiLintIssues(bySeverity) {
  const el = document.getElementById('wiki-lint-issues');
  const order = ['high', 'medium', 'low'];
  const colors = {high:'var(--danger)', medium:'var(--warning)', low:'var(--fg-muted)'};
  const labels = {high:'HIGH', medium:'MEDIUM', low:'LOW'};
  let html = '';
  for (const sev of order) {
    const issues = bySeverity[sev] || [];
    if (!issues.length) continue;
    html += '<div class="u-mt-2"><div style="font-weight:bold;color:'
      + colors[sev] + ';font-size:11px;letter-spacing:0.05em;">' + labels[sev]
      + ' (' + issues.length + ')</div>';
    for (const i of issues) {
      const kind = i.type_ || i.kind || 'issue';
      html += '<div style="padding:6px 8px;margin-top:4px;border-left:3px solid '
        + colors[sev] + ';background:var(--toolbar-bg);border-radius:4px;font-size:12px;">'
        + '<code class="u-label-xs">' + _escHtml(kind) + '</code> '
        + _escHtml(i.detail || i.message || JSON.stringify(i)) + '</div>';
    }
    html += '</div>';
  }
  el.innerHTML = html;
}

async function stopWikiLint() {
  if (_wikiLintJob) {
    await fetch('/api/jobs/' + _wikiLintJob, {method: 'DELETE'});
  }
}

// Phase 54.6.8 — Wiki KG backfill from the Lint tab.
async function doWikiExtractKg() {
  const force = document.getElementById('wiki-extractkg-force').checked;
  const runBtn = document.getElementById('wiki-extractkg-run');
  const status = document.getElementById('wiki-extractkg-status');
  const logEl = document.getElementById('wiki-extractkg-log');
  runBtn.disabled = true;
  status.textContent = 'Starting KG extraction…';
  logEl.style.display = 'block';
  logEl.textContent = '';
  const fd = new FormData();
  fd.append('force', force);
  let res;
  try {
    res = await fetch('/api/wiki/extract-kg', {method: 'POST', body: fd});
  } catch (exc) {
    status.textContent = 'Request failed: ' + exc.message;
    runBtn.disabled = false;
    return;
  }
  if (!res.ok) {
    status.textContent = 'Start failed: HTTP ' + res.status;
    runBtn.disabled = false;
    return;
  }
  const data = await res.json();
  const source = new EventSource('/api/stream/' + data.job_id);
  source.onmessage = function(e) {
    let evt;
    try { evt = JSON.parse(e.data); } catch (_) { return; }
    if (evt.type === 'log') {
      logEl.textContent += evt.text + '\n';
      logEl.scrollTop = logEl.scrollHeight;
    } else if (evt.type === 'progress') {
      if (evt.detail && evt.detail.startsWith('$ ')) {
        logEl.textContent += evt.detail + '\n';
      }
    } else if (evt.type === 'completed') {
      source.close();
      status.innerHTML = '<span class="u-success">&#10003; Done — open the Knowledge Graph modal to browse triples.</span>';
      runBtn.disabled = false;
    } else if (evt.type === 'error') {
      source.close();
      status.textContent = 'Error: ' + (evt.message || 'see log');
      runBtn.disabled = false;
    } else if (evt.type === 'done') {
      source.close();
      runBtn.disabled = false;
    }
  };
}

let _wikiConsensusJob = null;
let _wikiConsensusSource = null;

async function doWikiConsensus() {
  const topic = document.getElementById('wiki-consensus-topic').value.trim();
  if (!topic) { alert('Enter a topic first.'); return; }
  const runBtn = document.getElementById('wiki-consensus-run');
  const stopBtn = document.getElementById('wiki-consensus-stop');
  const status = document.getElementById('wiki-consensus-status');
  const summaryEl = document.getElementById('wiki-consensus-summary');
  const claimsEl = document.getElementById('wiki-consensus-claims');
  const debatedEl = document.getElementById('wiki-consensus-debated');
  runBtn.disabled = true;
  stopBtn.style.display = 'inline-block';
  status.textContent = 'Gathering evidence…';
  summaryEl.innerHTML = '';
  claimsEl.innerHTML = '';
  debatedEl.innerHTML = '';

  const fd = new FormData();
  fd.append('topic', topic);
  const res = await fetch('/api/wiki/consensus', {method: 'POST', body: fd});
  const data = await res.json();
  _wikiConsensusJob = data.job_id;
  const source = new EventSource('/api/stream/' + _wikiConsensusJob);
  _wikiConsensusSource = source;

  source.onmessage = function(e) {
    const evt = JSON.parse(e.data);
    if (evt.type === 'progress') {
      status.textContent = evt.detail || evt.stage;
    } else if (evt.type === 'consensus') {
      _renderConsensus(evt.data || {}, summaryEl, claimsEl, debatedEl);
    } else if (evt.type === 'completed') {
      const slug = evt.slug || '';
      const claims = evt.claims || 0;
      status.innerHTML = 'Saved <strong>' + claims + '</strong> claim(s)'
        + (slug ? ' as <a href="#" onclick="event.preventDefault();switchWikiTab(\'wiki-browse\');openWikiPage(\'' + _escHtml(slug) + '\');">' + _escHtml(slug) + '</a>.' : '.');
      source.close(); _wikiConsensusSource = null; _wikiConsensusJob = null;
      runBtn.disabled = false; stopBtn.style.display = 'none';
    } else if (evt.type === 'error') {
      // Phase 54.6.44 — surface the error inline so the user sees why
      // the output panels stayed empty. Pre-fix, a "no data for topic"
      // or JSON-parse failure only updated the tiny status textContent;
      // users reported this as "consensus shows nothing".
      status.textContent = 'Error: ' + (evt.message || 'unknown');
      summaryEl.innerHTML =
        '<div style="padding:12px;background:rgba(220,50,50,0.08);border-left:3px solid var(--danger);border-radius:4px;">'
        + '<strong class="u-danger">Consensus failed.</strong><br>'
        + '<span class="u-small">' + _escHtml(evt.message || '(no details)') + '</span><br>'
        + '<span class="u-tiny u-muted u-mt-6 u-block">'
        + 'Common causes: (1) the topic has no wiki pages yet — try `uv run sciknow wiki compile` or a broader topic; '
        + '(2) the LLM returned un-parseable JSON — check the server log for a <code>consensus_map</code> warning.'
        + '</span></div>';
      source.close(); _wikiConsensusSource = null; _wikiConsensusJob = null;
      runBtn.disabled = false; stopBtn.style.display = 'none';
    } else if (evt.type === 'done') {
      source.close(); _wikiConsensusSource = null; _wikiConsensusJob = null;
      runBtn.disabled = false; stopBtn.style.display = 'none';
    }
  };
  source.onerror = function() {
    // Phase 54.6.44 — surface EventSource transport failures (server
    // died, network blip). The default behavior is silent retry which
    // looks to the user like "nothing happened".
    status.textContent = 'Lost connection to server. Reload and try again.';
    source.close(); _wikiConsensusSource = null; _wikiConsensusJob = null;
    runBtn.disabled = false; stopBtn.style.display = 'none';
  };
}

function _renderConsensus(data, summaryEl, claimsEl, debatedEl) {
  if (data.summary) {
    summaryEl.innerHTML = '<div class="u-p-2 u-bg-tb u-r-sm">'
      + _escHtml(data.summary) + '</div>';
  }
  const colorOf = {
    strong: 'var(--success)', moderate: 'var(--accent)',
    weak: 'var(--warning)', contested: 'var(--danger)',
  };
  const claims = data.claims || [];
  if (claims.length) {
    let html = '<div class="u-tiny u-bold u-muted u-my-2">CLAIMS</div>';
    for (const c of claims) {
      const level = (c.consensus_level || 'unknown').toLowerCase();
      const color = colorOf[level] || 'var(--fg-muted)';
      const trend = c.trend ? (' · trend: ' + c.trend) : '';
      const sup = c.supporting_papers || [];
      const con = c.contradicting_papers || [];
      html += '<div style="padding:8px 10px;margin-top:6px;border-left:3px solid '
        + color + ';background:var(--toolbar-bg);border-radius:4px;font-size:12px;">'
        + '<span style="font-weight:bold;color:' + color + ';text-transform:uppercase;font-size:10px;letter-spacing:0.05em;">'
        + _escHtml(level) + '</span>'
        + '<span class="u-muted u-xxs">' + _escHtml(trend) + '</span>'
        + '<div class="u-mt-1">' + _escHtml(c.claim || '') + '</div>';
      if (sup.length) {
        html += '<div class="u-mt-1 u-success u-tiny">Supports ('
          + sup.length + '): ' + sup.slice(0, 4).map(_escHtml).join(', ')
          + (sup.length > 4 ? ' +' + (sup.length - 4) : '') + '</div>';
      }
      if (con.length) {
        html += '<div class="u-mt-2px u-danger u-tiny">Contradicts ('
          + con.length + '): ' + con.slice(0, 4).map(_escHtml).join(', ')
          + (con.length > 4 ? ' +' + (con.length - 4) : '') + '</div>';
      }
      html += '</div>';
    }
    claimsEl.innerHTML = html;
  }
  const debated = data.most_debated || [];
  if (debated.length) {
    debatedEl.innerHTML = '<div class="u-tiny u-bold u-muted u-mb-1">MOST DEBATED</div>'
      + '<ul style="margin:0 0 0 20px;padding:0;font-size:12px;">'
      + debated.map(d => '<li>' + _escHtml(d) + '</li>').join('') + '</ul>';
  }
}

async function stopWikiConsensus() {
  if (_wikiConsensusJob) {
    await fetch('/api/jobs/' + _wikiConsensusJob, {method: 'DELETE'});
  }
}

let wikiBrowsePage = 1;
async function loadWikiPages(page) {
  wikiBrowsePage = page || 1;
  const filter = document.getElementById('wiki-type-filter');
  const params = new URLSearchParams({page: wikiBrowsePage, per_page: 50});
  if (filter && filter.value) params.set('page_type', filter.value);
  const list = document.getElementById('wiki-browse-list');
  const detail = document.getElementById('wiki-page-detail');
  // Phase 54.6.44 — defensive null checks. If the DOM skeleton is
  // missing (e.g. someone refactored the modal and forgot to update
  // the IDs), fail loudly to the console and a visible banner
  // instead of crashing silently in the original `.style` deref.
  if (!list) {
    console.error('loadWikiPages: #wiki-browse-list not found in DOM');
    return;
  }
  if (detail) detail.style.display = 'none';
  list.innerHTML = '<div class="u-empty">Loading...</div>';

  try {
    const res = await fetch('/api/wiki/pages?' + params.toString());
    if (!res.ok) {
      throw new Error('HTTP ' + res.status + ' ' + res.statusText);
    }
    const data = await res.json();
    console.log('[loadWikiPages] total=' + (data.total || 0) +
                ' available_types=' + JSON.stringify(data.available_types || []));

    // Populate the type filter dropdown if it's still empty
    if (filter && filter.options.length <= 1 && data.available_types) {
      data.available_types.forEach(t => {
        const opt = document.createElement('option');
        opt.value = t;
        opt.textContent = t.replace(/_/g, ' ');
        filter.appendChild(opt);
      });
    }

    if (!data.pages || data.pages.length === 0) {
      const msg = data.error
        ? 'API returned an error: <code>' + _escHtml(data.error) + '</code><br>'
          + '<span class="u-tiny">Check the server log.</span>'
        : 'No wiki pages found.<br>'
          + '<span class="u-tiny">Run <code>uv run sciknow wiki compile</code> to build wiki pages from your corpus.</span>';
      list.innerHTML = '<div class="u-empty">'
        + msg + '</div>';
      return;
    }

    let html = '<div class="wiki-page-list">';
    data.pages.forEach(p => {
      const slug = (p.slug || '').replace(/'/g, '&#39;');
      // Phase 54.6.9 — also surface year + authors (served by the
      // enriched list_pages helper). Year is '' for multi-source
      // synthesis pages; authors is "" when the linked paper has
      // no author list. Both are always-present empty strings so the
      // column widths don't jump between rows.
      const year = (p.year != null) ? String(p.year) : '';
      const authors = (p.authors_display || '').replace(/</g, '&lt;');
      html += '<div class="wiki-page-row" data-action="open-wiki-page" data-slug="' + slug + '">';
      html += '<div class="wp-title">' + (p.title || p.slug || '').replace(/</g, '&lt;') + '</div>';
      html += '<div class="wp-authors" title="' + authors.replace(/"/g, '&quot;') + '">' + authors + '</div>';
      html += '<div class="wp-year">' + year + '</div>';
      html += '<div class="wp-meta">' + (p.word_count || 0).toLocaleString() + ' words · ' + (p.n_sources || 0) + ' src</div>';
      html += '<div class="wp-type">' + (p.page_type || '').replace(/_/g, ' ') + '</div>';
      html += '</div>';
    });
    html += '</div>';

    if (data.n_pages > 1) {
      html += '<div class="catalog-pager">';
      html += '<button data-action="load-wiki-pages" data-page="' + (wikiBrowsePage - 1) + '" ' + (wikiBrowsePage <= 1 ? 'disabled' : '') + ' title="Previous page of wiki results.">‹ Prev</button>';
      html += '<span>Page ' + data.page + ' of ' + data.n_pages + '  ·  ' + data.total + ' pages</span>';
      html += '<button data-action="load-wiki-pages" data-page="' + (wikiBrowsePage + 1) + '" ' + (wikiBrowsePage >= data.n_pages ? 'disabled' : '') + ' title="Next page of wiki results.">Next ›</button>';
      html += '</div>';
    }

    list.innerHTML = html;
  } catch (e) {
    list.innerHTML = '<div class="u-empty-danger">Error: ' + e.message + '</div>';
  }
}

// ── Phase 54.6.61 — Summaries tab ───────────────────────────────────────
// We fetch once (all 500-1000 paper summaries fit in a single request)
// and filter/sort client-side. Each card opens the full summary via
// openWikiPage(), which reuses the existing detail pane under the
// Browse tab — so "Back to list" lands on Browse. That's acceptable;
// the state is visible and the user can use the Summaries tab button
// to get back.

let _wikiSummariesCache = null;  // array from /api/wiki/pages

async function loadWikiSummaries() {
  const list = document.getElementById('wiki-summaries-list');
  if (!list) return;
  if (_wikiSummariesCache) { renderWikiSummaries(); return; }
  list.innerHTML = '<div class="u-empty">Loading summaries…</div>';
  try {
    // per_page=2000 — we have ~630 summaries today; a single page keeps
    // the client-side sort/filter trivial and avoids paginator wiring.
    const res = await fetch('/api/wiki/pages?page_type=paper_summary&per_page=2000&page=1');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    _wikiSummariesCache = (data.pages || []).filter(p => p.page_type === 'paper_summary');
    renderWikiSummaries();
  } catch (e) {
    list.innerHTML = '<div class="u-empty-danger">Error: ' + e.message + '</div>';
  }
}

function renderWikiSummaries() {
  const list = document.getElementById('wiki-summaries-list');
  const countEl = document.getElementById('wiki-sum-count');
  const searchEl = document.getElementById('wiki-sum-search');
  const sortEl = document.getElementById('wiki-sum-sort');
  if (!list || !_wikiSummariesCache) return;

  const q = ((searchEl && searchEl.value) || '').toLowerCase().trim();
  const sort = (sortEl && sortEl.value) || 'year_desc';

  let items = _wikiSummariesCache.slice();
  if (q) {
    items = items.filter(p => {
      const hay = ((p.title || '') + ' ' + (p.authors_display || '')
                   + ' ' + (p.slug || '')).toLowerCase();
      return hay.indexOf(q) !== -1;
    });
  }

  const cmp = {
    year_desc: (a, b) => (b.year || 0) - (a.year || 0)
                     || (a.title || '').localeCompare(b.title || ''),
    year_asc: (a, b) => (a.year || 9999) - (b.year || 9999)
                    || (a.title || '').localeCompare(b.title || ''),
    updated_desc: (a, b) => (b.updated_at || '').localeCompare(a.updated_at || ''),
    title_asc: (a, b) => (a.title || '').localeCompare(b.title || ''),
    words_desc: (a, b) => (b.word_count || 0) - (a.word_count || 0),
  }[sort] || ((a, b) => 0);
  items.sort(cmp);

  if (countEl) {
    countEl.textContent = items.length + ' of ' + _wikiSummariesCache.length
                        + ' summaries';
  }

  if (items.length === 0) {
    list.innerHTML = '<div class="u-empty">No summaries match.</div>';
    return;
  }

  let html = '';
  for (const p of items) {
    const title = _escHtml(p.title || p.slug || '');
    const authors = _escHtml(p.authors_display || '');
    const year = p.year != null ? p.year : '—';
    const words = (p.word_count || 0).toLocaleString();
    const slug = _escHtml(p.slug || '');
    html += '<div class="wiki-summary-card" onclick="openWikiSummary(\'' + slug + '\')" '
         +  'style="border:1px solid var(--border);border-radius:8px;padding:10px 14px;'
         +  'margin-bottom:8px;cursor:pointer;transition:background var(--t-fast);" '
         +  'onmouseover="this.style.background=\'var(--bg-alt)\'" '
         +  'onmouseout="this.style.background=\'transparent\'">'
         +  '<div style="font-weight:600;margin-bottom:3px;">' + title + '</div>'
         +  '<div class="u-hint-sm">'
         +  (authors || '<em>unknown authors</em>') + '  ·  '
         +  year + '  ·  ' + words + ' words'
         +  '</div>'
         +  '</div>';
  }
  list.innerHTML = html;
}

function openWikiSummary(slug) {
  // Switch to the Browse tab (which owns the detail pane) then open.
  // The detail pane is shared between Browse and Summaries — simplest
  // wiring. A dedicated detail pane inside Summaries would duplicate
  // 150 lines of TOC / backlinks / annotations code.
  switchWikiTab('wiki-browse');
  openWikiPage(slug);
}

// ── Phase 54.6.61 — Visuals tab ─────────────────────────────────────────
// Grid for figures (actual thumbnails via /api/visuals/image/<id>),
// list layout for equations/tables/code. Reuses /api/visuals (kind,
// query, limit filters). Equations render via KaTeX which is already
// loaded by the page header; tables render the raw MinerU HTML.

async function loadWikiVisuals() {
  const kindEl = document.getElementById('wiki-vis-kind');
  const searchEl = document.getElementById('wiki-vis-search');
  const limitEl = document.getElementById('wiki-vis-limit');
  const list = document.getElementById('wiki-visuals-list');
  const statsEl = document.getElementById('wiki-vis-stats');
  if (!list) return;

  const kind = (kindEl && kindEl.value) || 'figure';
  const query = ((searchEl && searchEl.value) || '').trim();
  const limit = (limitEl && +limitEl.value) || 60;

  // Stats line (cached — change only when extract-visuals re-runs)
  try {
    const sres = await fetch('/api/visuals/stats');
    const sdata = await sres.json();
    if (statsEl && sdata.stats) {
      const parts = Object.entries(sdata.stats).map(
        function(kv) { return kv[0] + ': ' + kv[1]; }
      ).join(', ');
      statsEl.textContent = 'Corpus: ' + parts;
    }
  } catch (e) { /* non-fatal */ }

  list.innerHTML = '<div class="u-p-4 u-muted">Loading ' + kind + 's…</div>';

  const params = new URLSearchParams({kind: kind, limit: String(limit)});
  if (query) params.set('query', query);
  try {
    const res = await fetch('/api/visuals?' + params.toString());
    const items = await res.json();
    if (!Array.isArray(items) || items.length === 0) {
      list.innerHTML = '<div class="u-empty">'
        + 'No ' + kind + 's found' + (query ? ' for “' + _escHtml(query) + '”' : '') + '.'
        + '</div>';
      return;
    }
    list.innerHTML = renderWikiVisuals(items, kind);
    // 54.6.105 — direct katex.render on each .eq-target we inserted.
    // Replaces the old auto-render pass (which was flaky).
    if (kind === 'equation') {
      const renderEqs = () => {
        const targets = list.querySelectorAll('.eq-target[data-latex]');
        if (!targets.length) return true;
        if (typeof window.katex === 'undefined') return false;
        targets.forEach(el => {
          if (el.dataset.rendered === '1') return;
          try {
            window.katex.render(el.dataset.latex || '', el, {
              displayMode: true, throwOnError: false, output: 'html',
              strict: 'ignore',
              macros: {
                '\\displaylimits': '', '\\mit': '', '\\sc': '',
                '\\mathbfcal': '\\mathcal',
                '\\textless': '<', '\\textgreater': '>',
                '\\hdots': '\\ldots',
              },
            });
            el.dataset.rendered = '1';
          } catch (_) { el.textContent = el.dataset.latex || ''; }
        });
        return true;
      };
      if (!renderEqs()) {
        let n = 0;
        const t = setInterval(() => { if (renderEqs() || ++n > 10) clearInterval(t); }, 200);
      }
    }
  } catch (e) {
    list.innerHTML = '<div class="u-empty-danger">Error: ' + e.message + '</div>';
  }
}

function renderWikiVisuals(items, kind) {
  // Phase 54.6.62 — chart renders the same as figure (same image endpoint).
  if (kind === 'figure' || kind === 'chart') {
    // Thumbnail grid. Native lazy-loading on <img> keeps the initial
    // render cheap even at limit=500.
    let html = '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px;">';
    for (const v of items) {
      const caption = _escHtml((v.caption || v.content || '').substring(0, 140));
      const paper = _escHtml((v.paper_title || '').substring(0, 60));
      const year = v.year ? ' (' + v.year + ')' : '';
      const fn = _escHtml(v.figure_num || 'Figure');
      // Phase 54.6.72 (#1) — ai_caption (vision-LLM generated) if present
      const ai = v.ai_caption ? _escHtml(v.ai_caption.substring(0, 260)) : '';
      html += '<div class="u-border u-r-md u-ov-hidden u-bg">'
           +  '<a href="/api/visuals/image/' + encodeURIComponent(v.id) + '" target="_blank" '
           +  'style="display:block;background:#000;">'
           +  '<img src="/api/visuals/image/' + encodeURIComponent(v.id) + '" loading="lazy" '
           +  'style="width:100%;height:160px;object-fit:contain;display:block;" '
           +  'onerror="this.style.display=\'none\';this.parentElement.innerHTML=\'<div style=padding:40px;color:#888;text-align:center;font-size:11px;>image missing</div>\';">'
           +  '</a>'
           +  '<div class="u-pill-lg u-tiny">'
           +  '<div style="font-weight:600;margin-bottom:2px;">' + fn + '</div>'
           +  (caption ? '<div style="color:var(--fg-muted);line-height:1.35;max-height:3.6em;overflow:hidden;">' + caption + '</div>' : '')
           +  (ai ? '<div style="color:var(--fg);line-height:1.4;margin-top:4px;padding:4px 6px;background:var(--bg-alt,#f5f5f5);border-radius:4px;font-size:10.5px;max-height:7em;overflow:hidden;" title="' + _escHtml(v.ai_caption || '') + '"><strong style="color:var(--accent,dodgerblue);">AI:</strong> ' + ai + '</div>' : '')
           +  '<div class="u-muted u-mt-1 u-italic">' + paper + year + '</div>'
           +  '</div>'
           +  '</div>';
    }
    html += '</div>';
    return html;
  }

  // Non-figure kinds: list/row layout
  let html = '';
  for (const v of items) {
    const paper = _escHtml((v.paper_title || '').substring(0, 80)) + (v.year ? ' (' + v.year + ')' : '');
    const caption = _escHtml(v.caption || '');
    let body = '';
    if (kind === 'equation') {
      // 54.6.105 — direct katex.render on an .eq-target div (same as
      // the Explore > Visuals modal after 54.6.105). Auto-render
      // proved flaky; data-latex + programmatic render is bulletproof.
      let eqBody = String(v.content || '').trim();
      eqBody = eqBody.replace(/^\s*\$\$\s*/, '').replace(/\s*\$\$\s*$/, '');
      eqBody = eqBody.replace(/^\s*\$\s*/, '').replace(/\s*\$\s*$/, '').trim();
      const attrSafe = _escHtml(eqBody);
      body = '<div class="vis-eq" style="padding:14px 16px;background:#fff;color:#111;border-radius:6px;border:1px solid var(--border);">'
           + '<div class="eq-target" data-latex="' + attrSafe + '" '
           +      'style="font-size:17px;line-height:1.55;text-align:center;overflow-x:auto;">'
           +      _escHtml(eqBody)
           + '</div></div>';
    } else if (kind === 'table') {
      // MinerU stores table_body as HTML — inject directly (scoped
      // .vis-table-wrap styles give it borders + zebra rows).
      body = '<div class="vis-table-wrap" style="max-height:320px;overflow:auto;border:1px solid var(--border);'
           + 'border-radius:6px;padding:8px;background:#fff;color:#111;">'
           + (v.content || '<em>empty table</em>') + '</div>';
    } else {  // code
      body = '<pre style="max-height:240px;overflow:auto;background:var(--bg);'
           + 'border-radius:4px;padding:8px;font-size:11px;">'
           + _escHtml(v.content || '') + '</pre>';
    }
    html += '<div class="u-border u-r-8 u-p-10 u-mb-m">'
         +  '<div class="u-tiny u-muted u-mb-1 u-italic">'
         +  paper + '</div>'
         +  (caption ? '<div class="u-small u-mb-6">' + caption + '</div>' : '')
         +  body
         +  '</div>';
  }
  return html;
}

// Phase 54 — track the currently-viewed wiki slug so the TOC + Copy
// Permalink know what to point at, and so the hashchange handler
// can short-circuit no-op re-renders.
let _currentWikiSlug = null;

async function openWikiPage(slug) {
  const list = document.getElementById('wiki-browse-list');
  const detail = document.getElementById('wiki-page-detail');
  const meta = document.getElementById('wiki-page-meta');
  const content = document.getElementById('wiki-page-content');
  const toc = document.getElementById('wiki-toc');
  list.style.display = 'none';
  detail.style.display = 'block';
  content.innerHTML = '<div class="u-empty">Loading...</div>';
  if (toc) toc.innerHTML = '';
  // Phase 54.3 — reset the inline "Ask this page" state when the
  // active page changes so a stale answer doesn't hang around on
  // the new page.
  const _askInput = document.getElementById('wiki-ask-input');
  const _askStatus = document.getElementById('wiki-ask-status');
  const _askStream = document.getElementById('wiki-ask-stream');
  const _askSources = document.getElementById('wiki-ask-sources');
  if (_askInput) _askInput.value = '';
  if (_askStatus) _askStatus.textContent = '';
  if (_askStream) _askStream.textContent = '';
  if (_askSources) { _askSources.style.display = 'none'; _askSources.innerHTML = ''; }
  if (_wikiAskSource) { try { _wikiAskSource.close(); } catch (e) {} _wikiAskSource = null; }
  // Phase 54.5 — reset annotation textarea + status between pages so
  // a stale note from the previous page isn't mistakenly kept.
  const _annBody = document.getElementById('wiki-annotation-body');
  const _annStatus = document.getElementById('wiki-annotation-status');
  const _annTs = document.getElementById('wiki-annotation-ts');
  if (_annBody) _annBody.value = '';
  if (_annStatus) _annStatus.textContent = '';
  if (_annTs) _annTs.textContent = '';
  _currentWikiSlug = slug;

  // Phase 54 — reflect the open page in the URL hash so back/forward
  // work and permalinks are shareable. Use pushState only when the
  // hash isn't already right, to avoid loops with the hashchange
  // listener below.
  const target = '#wiki/' + encodeURIComponent(slug);
  if (window.location.hash !== target) {
    history.pushState(null, '', target);
  }

  try {
    const res = await fetch('/api/wiki/page/' + encodeURIComponent(slug));
    if (!res.ok) {
      content.innerHTML = '<div style="color:var(--danger);padding:24px;">Wiki page <code>' + slug + '</code> not found.</div>';
      return;
    }
    const data = await res.json();
    const metaParts = [];
    if (data.page_type) metaParts.push('<strong>' + data.page_type.replace(/_/g, ' ') + '</strong>');
    if (data.word_count) metaParts.push(data.word_count.toLocaleString() + ' words');
    if (data.n_sources) metaParts.push(data.n_sources + ' source(s)');
    if (data.updated_at) metaParts.push('updated ' + data.updated_at.substring(0, 10));
    meta.innerHTML = metaParts.join(' · ');
    content.innerHTML = data.content_html || '<em>(empty page)</em>';

    // Phase 54.1 — render math via KaTeX auto-render if loaded.
    // Falls back silently to the raw `$...$` when KaTeX isn't
    // available (e.g. offline on first-ever page load).
    if (window.renderMathInElement) {
      try {
        window.renderMathInElement(content, {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '$',  right: '$',  display: false },
            { left: '\\(', right: '\\)', display: false },
            { left: '\\[', right: '\\]', display: true },
          ],
          throwOnError: false,
        });
      } catch (e) { /* render failure is non-fatal */ }
    }

    // Phase 54.1 — staleness banner (surfaces wiki_pages.needs_rewrite).
    const banner = document.getElementById('wiki-stale-banner');
    if (banner) banner.style.display = data.needs_rewrite ? 'block' : 'none';

    // Phase 54 — build the TOC from the rendered headings.
    _buildWikiTOC();

    // Phase 54.2 — Related pages + Referenced-by (backlinks) panels.
    // Fire both in parallel; render each section only when it returns
    // non-empty. No blocking — the main content paints first.
    _loadWikiRelated(slug);
    _loadWikiBacklinks(slug);

    // Phase 54.4 — Facts from the corpus (concept pages only).
    _renderWikiFacts(data);

    // Phase 54.5 — load the user's "My take" annotation for this page.
    _loadWikiAnnotation(slug);
    // Phase 54 — honour `?h=<heading-id>` in the hash if present.
    const m = (window.location.hash || '').match(/\?h=([^&]+)$/);
    if (m) {
      const el = document.getElementById(decodeURIComponent(m[1]));
      if (el) el.scrollIntoView({ behavior: 'instant', block: 'start' });
    } else {
      content.scrollTop = 0;
    }
  } catch (e) {
    content.innerHTML = '<div class="u-danger">Error: ' + e.message + '</div>';
  }
}

// Phase 54 — post-render TOC builder. Scans h2/h3/h4 inside
// #wiki-page-content, emits a sticky sidebar nav with click-to-
// scroll handlers. Zero-cost for pages without headings — the
// sidebar just renders empty.
function _buildWikiTOC() {
  const host = document.getElementById('wiki-toc');
  const content = document.getElementById('wiki-page-content');
  if (!host || !content) return;
  const heads = content.querySelectorAll('h2, h3, h4');
  if (!heads.length) { host.innerHTML = ''; return; }
  let html = '<div class="wiki-toc-heading">On this page</div><ol class="wiki-toc-list">';
  heads.forEach(h => {
    if (!h.id) return;
    const cls = 'wiki-toc-' + h.tagName.toLowerCase();
    html += '<li class="' + cls + '"><a data-heading="' + h.id + '">' +
            escapeHtml(h.textContent) + '</a></li>';
  });
  html += '</ol>';
  host.innerHTML = html;
}
// Delegated click → smooth-scroll the target heading into view.
document.addEventListener('click', (evt) => {
  const a = evt.target.closest && evt.target.closest('#wiki-toc [data-heading]');
  if (!a) return;
  evt.preventDefault();
  const el = document.getElementById(a.dataset.heading);
  if (el) {
    el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    // Also update the hash to make the heading shareable.
    if (_currentWikiSlug) {
      const target = '#wiki/' + encodeURIComponent(_currentWikiSlug) +
                     '?h=' + encodeURIComponent(a.dataset.heading);
      history.replaceState(null, '', target);
    }
  }
});

// Phase 54.3 — "Ask this page" inline RAG. Hits the new
// /api/wiki/page/<slug>/ask endpoint, streams tokens into the
// page's inline chat box via the same SSE contract the book-
// reader uses.
let _wikiAskSource = null;
async function askWikiPage() {
  const q = (document.getElementById('wiki-ask-input').value || '').trim();
  if (!q || !_currentWikiSlug) return;
  const broaden = document.getElementById('wiki-ask-broaden').checked;
  const status = document.getElementById('wiki-ask-status');
  const streamEl = document.getElementById('wiki-ask-stream');
  const sourcesEl = document.getElementById('wiki-ask-sources');
  const submit = document.getElementById('wiki-ask-submit');

  status.textContent = 'Retrieving and generating…';
  streamEl.textContent = '';
  sourcesEl.style.display = 'none';
  sourcesEl.innerHTML = '';
  submit.disabled = true;

  const fd = new FormData();
  fd.append('question', q);
  if (broaden) fd.append('broaden', 'true');

  let data;
  try {
    const res = await fetch('/api/wiki/page/' + encodeURIComponent(_currentWikiSlug) + '/ask',
                            { method: 'POST', body: fd });
    data = await res.json();
  } catch (e) {
    status.textContent = 'Error: ' + e.message;
    submit.disabled = false;
    return;
  }

  if (_wikiAskSource) { try { _wikiAskSource.close(); } catch (e) {} _wikiAskSource = null; }
  const source = new EventSource('/api/stream/' + data.job_id);
  _wikiAskSource = source;
  let collected = null;
  let scope = null;

  source.onmessage = function(e) {
    const evt = JSON.parse(e.data);
    if (evt.type === 'token') {
      streamEl.textContent += evt.text;
    } else if (evt.type === 'progress') {
      status.textContent = evt.detail || evt.stage;
    } else if (evt.type === 'sources') {
      collected = evt.sources;
      scope = evt.scope || 'corpus';
      status.textContent = 'Generating from ' + (evt.n || collected.length) +
                           ' passage(s) · scope=' + scope;
    } else if (evt.type === 'completed') {
      status.textContent = 'Done' + (scope ? ' · scope=' + scope : '');
      submit.disabled = false;
      if (collected && collected.length) {
        let html = '<div class="u-label-fg-6">Sources (' +
                   collected.length + ')</div><ol>';
        collected.forEach(s => { html += '<li>' + escapeHtml(s) + '</li>'; });
        html += '</ol>';
        sourcesEl.innerHTML = html;
        sourcesEl.style.display = 'block';
      }
      source.close(); _wikiAskSource = null;
    } else if (evt.type === 'error') {
      status.textContent = 'Error: ' + evt.message;
      submit.disabled = false;
      source.close(); _wikiAskSource = null;
    }
  };
  source.onerror = function() {
    status.textContent = 'Stream disconnected';
    submit.disabled = false;
    try { source.close(); } catch (e) {}
    _wikiAskSource = null;
  };
}

// Phase 54.5 — "My take" annotation: load / save / delete.
async function _loadWikiAnnotation(slug) {
  const body = document.getElementById('wiki-annotation-body');
  const ts = document.getElementById('wiki-annotation-ts');
  const status = document.getElementById('wiki-annotation-status');
  if (!body) return;
  body.value = '';
  if (ts) ts.textContent = '';
  if (status) status.textContent = '';
  try {
    const res = await fetch('/api/wiki/page/' + encodeURIComponent(slug) +
                            '/annotation');
    if (!res.ok) return;
    const d = await res.json();
    body.value = d.body || '';
    if (d.updated_at && ts) {
      ts.textContent = 'last saved ' + d.updated_at.substring(0, 16).replace('T', ' ');
    }
  } catch (e) { /* silent — empty textarea is the fallback */ }
  // Phase 54.6.x — wire debounced autosave on the "My take" textarea.
  // Idempotent (dataset.autosaveWired guards against double-bind).
  _wireWikiAnnotationAutosave();
}

async function saveWikiAnnotation() {
  if (!_currentWikiSlug) return;
  const body = document.getElementById('wiki-annotation-body');
  const status = document.getElementById('wiki-annotation-status');
  const ts = document.getElementById('wiki-annotation-ts');
  if (!body) return;
  const fd = new FormData();
  fd.append('body', body.value || '');
  status.textContent = 'saving…';
  try {
    const res = await fetch(
      '/api/wiki/page/' + encodeURIComponent(_currentWikiSlug) + '/annotation',
      { method: 'PUT', body: fd },
    );
    const d = await res.json();
    if (d.deleted) {
      status.textContent = 'cleared';
      if (ts) ts.textContent = '';
    } else {
      status.textContent = 'saved';
      if (ts && d.updated_at) {
        ts.textContent = 'last saved ' + d.updated_at.substring(0, 16).replace('T', ' ');
      }
    }
    setTimeout(() => { if (status.textContent === 'saved' || status.textContent === 'cleared') status.textContent = ''; }, 2000);
  } catch (e) {
    status.textContent = 'save failed: ' + e.message;
  }
}

async function deleteWikiAnnotation() {
  const body = document.getElementById('wiki-annotation-body');
  if (body) body.value = '';
  await saveWikiAnnotation();
}

// Phase 54.5 — j/k navigation through the wiki browse list.
// Only active when the browse-list pane is visible, nobody's typing
// in a form field, and no modifier keys are held.
let _wikiListIdx = -1;
function _wikiListItems() {
  return document.querySelectorAll(
    '#wiki-browse-list [data-slug], #wiki-browse-list tr[data-slug], #wiki-browse-list li[data-slug]'
  );
}
function _setWikiListActive(idx) {
  const items = _wikiListItems();
  if (!items.length) return;
  _wikiListIdx = Math.max(0, Math.min(idx, items.length - 1));
  items.forEach((n, i) => n.classList.toggle('active-row', i === _wikiListIdx));
  items[_wikiListIdx].scrollIntoView({ block: 'nearest' });
}
document.addEventListener('keydown', (evt) => {
  // Only fire when the browse list is on-screen and no other
  // input is focused.
  const listVisible = (() => {
    const el = document.getElementById('wiki-browse-list');
    if (!el) return false;
    if (el.style.display === 'none') return false;
    const detail = document.getElementById('wiki-page-detail');
    return !(detail && detail.style.display !== 'none');
  })();
  if (!listVisible) return;
  const tag = (evt.target && evt.target.tagName) || '';
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
  if (evt.metaKey || evt.ctrlKey || evt.altKey) return;
  if (evt.key === 'j') {
    evt.preventDefault();
    _setWikiListActive(_wikiListIdx < 0 ? 0 : _wikiListIdx + 1);
  } else if (evt.key === 'k') {
    evt.preventDefault();
    _setWikiListActive(_wikiListIdx < 0 ? 0 : _wikiListIdx - 1);
  } else if (evt.key === 'Enter') {
    const items = _wikiListItems();
    if (_wikiListIdx >= 0 && items[_wikiListIdx]) {
      const slug = items[_wikiListIdx].dataset.slug;
      if (slug) {
        evt.preventDefault();
        window.location.hash = '#wiki/' + encodeURIComponent(slug);
      }
    }
  }
});

// Phase 54.4 — Render the "Facts from the corpus" block using the
// triples the API attaches to concept pages (server-side join
// against the knowledge_graph table). Hidden when the page isn't a
// concept or no triples matched.
function _renderWikiFacts(data) {
  const block = document.getElementById('wiki-facts-block');
  const list = document.getElementById('wiki-facts-list');
  const link = document.getElementById('wiki-facts-kg-link');
  if (!block || !list) return;
  const triples = (data && data.related_triples) || [];
  if (data.page_type !== 'concept' || !triples.length) {
    block.style.display = 'none';
    return;
  }
  // "Open in graph" — use the KG modal's existing share-URL convention
  // so the concept opens pinned at the centre of the 3D orbit view.
  if (link && typeof KG_THEMES !== 'undefined') {
    try {
      const shareState = {
        f: { s: '', p: '', o: '' },
        c: { rx: -0.22, ry: 0.55, d: 850 },
        p: [ (data.title || _currentWikiSlug).toLowerCase() ],
      };
      const hash = '#kg=' + btoa(unescape(encodeURIComponent(
        JSON.stringify(shareState))));
      link.href = hash;
      // Also pre-fill the subject filter when the user lands in the KG
      link.addEventListener('click', (evt) => {
        evt.preventDefault();
        const subj = document.getElementById('kg-subject');
        if (subj) subj.value = data.title || _currentWikiSlug;
        window.location.hash = hash;
      }, { once: true });
    } catch (e) { /* leave link as-is on encode failure */ }
  }

  // Classify each triple by predicate family so the left-border
  // matches the KG colour scheme. Reuses KG_PREDICATE_FAMILIES
  // from the KG modal script if it's loaded.
  function _family(predicate) {
    try {
      if (typeof _kgPredicateFamily === 'function') return _kgPredicateFamily(predicate);
    } catch (e) {}
    return 'other';
  }

  list.innerHTML = triples.slice(0, 40).map(t => {
    const fam = _family(t.predicate || '');
    const sent = t.source_sentence || '';
    const docTitle = t.source_title || '';
    const tipParts = [];
    if (docTitle) tipParts.push(docTitle);
    if (sent) tipParts.push('“' + sent + '”');
    const tipAttr = tipParts.length
      ? ' title="' + escapeHtml(tipParts.join(' — ')) + '"'
      : '';
    let srcHtml = '';
    if (sent) {
      const short = sent.length > 140 ? sent.substring(0, 140) + '…' : sent;
      srcHtml = '<span class="wf-src">' + escapeHtml(short) + '</span>';
    }
    return (
      '<li class="wf-fam-' + fam + '"' + tipAttr + '>' +
      '<span class="wf-subject">' + escapeHtml(t.subject || '') + '</span>' +
      '<span class="wf-pred"> ' + escapeHtml(t.predicate || '') + ' </span>' +
      '<span class="wf-object">' + escapeHtml(t.object || '') + '</span>' +
      srcHtml +
      '</li>'
    );
  }).join('');
  block.style.display = 'block';
}

// Phase 54.2 — Related / backlinks loaders.
async function _loadWikiRelated(slug) {
  const block = document.getElementById('wiki-related-block');
  const list = document.getElementById('wiki-related-list');
  if (!block || !list) return;
  block.style.display = 'none';
  try {
    const res = await fetch('/api/wiki/page/' + encodeURIComponent(slug) +
                            '/related?limit=5');
    if (!res.ok) return;
    const items = await res.json();
    if (!Array.isArray(items) || items.length === 0) return;
    list.innerHTML = items.map(t =>
      '<li>' +
      '<a href="#wiki/' + encodeURIComponent(t.slug) + '">' +
         escapeHtml(t.title) + '</a>' +
      '<span class="wp-type">' + (t.page_type || '').replace(/_/g, ' ') + '</span>' +
      '</li>'
    ).join('');
    block.style.display = 'block';
  } catch (e) { /* silent — related pages is best-effort */ }
}

async function _loadWikiBacklinks(slug) {
  const block = document.getElementById('wiki-backlinks-block');
  const list = document.getElementById('wiki-backlinks-list');
  if (!block || !list) return;
  block.style.display = 'none';
  try {
    const res = await fetch('/api/wiki/page/' + encodeURIComponent(slug) +
                            '/backlinks');
    if (!res.ok) return;
    const items = await res.json();
    if (!Array.isArray(items) || items.length === 0) return;
    list.innerHTML = items.map(t => {
      const alt = t.alt && t.alt !== t.from_slug
        ? ' <span class="wp-alt">&ldquo;' + escapeHtml(t.alt) + '&rdquo;</span>'
        : '';
      return '<li>' +
        '<a href="#wiki/' + encodeURIComponent(t.from_slug) + '">' +
           escapeHtml(t.from_title || t.from_slug) + '</a>' +
        alt +
        '</li>';
    }).join('');
    block.style.display = 'block';
  } catch (e) { /* silent */ }
}

function closeWikiPageDetail() {
  document.getElementById('wiki-browse-list').style.display = 'block';
  document.getElementById('wiki-page-detail').style.display = 'none';
  _currentWikiSlug = null;
  if ((window.location.hash || '').startsWith('#wiki/')) {
    history.pushState(null, '', '#wiki');
  }
}

function copyWikiPermalink() {
  if (!_currentWikiSlug) return;
  const url = window.location.origin + window.location.pathname +
              '#wiki/' + encodeURIComponent(_currentWikiSlug);
  navigator.clipboard.writeText(url).then(
    () => {
      const btn = event.target.closest('button');
      if (btn) {
        const prev = btn.innerHTML;
        btn.innerHTML = '&check; Copied';
        setTimeout(() => { btn.innerHTML = prev; }, 1500);
      }
    },
    () => prompt('Copy this link:', url),
  );
}

function openWikiModal() {
  openModal('wiki-modal');
  setTimeout(() => document.getElementById('wiki-query-input').focus(), 100);
}

// Phase 54 — hash router for the wiki SPA surface.
// Supports:
//   #wiki             → open modal on Browse list
//   #wiki/<slug>      → open modal on page detail
//   #wiki/<slug>?h=X  → open page detail + scroll to heading id X
function _wikiRouteFromHash() {
  const h = window.location.hash || '';
  if (!h.startsWith('#wiki')) return false;
  const rest = h.substring(5); // strip "#wiki"
  const modal = document.getElementById('wiki-modal');
  if (modal && modal.style.display !== 'flex' && modal.style.display !== 'block') {
    openModal('wiki-modal');
  }
  switchWikiTab('wiki-browse');
  if (!rest || rest === '' || rest === '/') {
    closeWikiPageDetail();
    loadWikiPages(1);
    return true;
  }
  // rest is "/slug" optionally followed by "?h=heading"
  const m = rest.match(/^\/([^?]+)(?:\?h=(.+))?$/);
  if (!m) return true;
  const slug = decodeURIComponent(m[1]);
  if (slug !== _currentWikiSlug) {
    openWikiPage(slug);
  }
  return true;
}
window.addEventListener('hashchange', _wikiRouteFromHash);
window.addEventListener('DOMContentLoaded', () => {
  if ((window.location.hash || '').startsWith('#wiki')) {
    _wikiRouteFromHash();
  }
});

// ── Phase 54 — Ctrl-K / Cmd-K wiki command palette ─────────────────
// Fuzzy-filter over wiki page titles + slugs, keyboard-navigable.
// Titles are fetched once per session and cached in-memory; the
// wiki size is bounded (a few hundred pages typical) so shipping the
// whole list is fine.

let _wikiTitlesCache = null;
let _wikiPaletteIdx = 0;

async function _loadWikiTitles() {
  if (_wikiTitlesCache) return _wikiTitlesCache;
  try {
    const res = await fetch('/api/wiki/titles');
    if (res.ok) _wikiTitlesCache = await res.json();
    else _wikiTitlesCache = [];
  } catch (e) { _wikiTitlesCache = []; }
  return _wikiTitlesCache;
}

// Tiny fuzzy scorer — substring hit dominates; char-in-order fallback
// for typos / abbreviations. Good enough for a few hundred items.
function _wikiFuzzyScore(needle, hay) {
  if (!needle) return 1;
  needle = needle.toLowerCase();
  hay = (hay || '').toLowerCase();
  const idx = hay.indexOf(needle);
  if (idx !== -1) return 1000 - idx;  // earlier hits rank higher
  let hi = 0, hits = 0;
  for (const ch of needle) {
    const i = hay.indexOf(ch, hi);
    if (i === -1) return 0;
    hits += 1;
    hi = i + 1;
  }
  return hits;
}

async function _renderWikiPalette() {
  const q = document.getElementById('wiki-palette-input').value.trim();
  const titles = await _loadWikiTitles();
  let items;
  if (!q) {
    items = titles.slice(0, 10);
  } else {
    items = titles
      .map(t => ({
        ...t,
        _s: _wikiFuzzyScore(q, t.title) + _wikiFuzzyScore(q, t.slug) * 0.5,
      }))
      .filter(t => t._s > 0)
      .sort((a, b) => b._s - a._s)
      .slice(0, 10);
  }
  _wikiPaletteIdx = 0;
  const host = document.getElementById('wiki-palette-results');
  if (!items.length) {
    host.innerHTML = '<li class="wiki-palette-empty">No pages match</li>';
    return;
  }
  host.innerHTML = items.map((t, i) => {
    const cls = (i === 0) ? 'wiki-palette-item active' : 'wiki-palette-item';
    return '<li class="' + cls + '" data-slug="' + t.slug + '">' +
           '<span class="wp-title">' + escapeHtml(t.title) + '</span>' +
           '<span class="wp-type">' + (t.page_type || '').replace(/_/g, ' ') + '</span>' +
           '</li>';
  }).join('');
}

function _wikiPaletteKey(evt) {
  const host = document.getElementById('wiki-palette-results');
  const items = host.querySelectorAll('.wiki-palette-item');
  if (evt.key === 'Escape') {
    evt.preventDefault(); closeWikiPalette(); return;
  }
  if (!items.length) return;
  if (evt.key === 'ArrowDown' || evt.key === 'ArrowUp') {
    evt.preventDefault();
    items[_wikiPaletteIdx].classList.remove('active');
    _wikiPaletteIdx = (evt.key === 'ArrowDown')
      ? (_wikiPaletteIdx + 1) % items.length
      : (_wikiPaletteIdx - 1 + items.length) % items.length;
    items[_wikiPaletteIdx].classList.add('active');
    items[_wikiPaletteIdx].scrollIntoView({ block: 'nearest' });
    return;
  }
  if (evt.key === 'Enter') {
    evt.preventDefault();
    const slug = items[_wikiPaletteIdx].dataset.slug;
    closeWikiPalette();
    window.location.hash = '#wiki/' + encodeURIComponent(slug);
    return;
  }
}

function openWikiPalette() {
  const modal = document.getElementById('wiki-palette');
  if (!modal) return;
  modal.style.display = 'flex';
  const input = document.getElementById('wiki-palette-input');
  input.value = '';
  _renderWikiPalette();
  setTimeout(() => input.focus(), 30);
}

function closeWikiPalette() {
  const modal = document.getElementById('wiki-palette');
  if (modal) modal.style.display = 'none';
}

// Delegated click on a palette row → navigate.
document.addEventListener('click', (evt) => {
  const item = evt.target.closest && evt.target.closest('.wiki-palette-item');
  if (!item) return;
  const slug = item.dataset.slug;
  if (!slug) return;
  closeWikiPalette();
  window.location.hash = '#wiki/' + encodeURIComponent(slug);
});

// Global Ctrl-K / Cmd-K — open the palette. Skip if the user is
// typing in a textarea / input (except the palette itself, whose
// input handler takes arrow keys + Escape via onkeydown).
document.addEventListener('keydown', (evt) => {
  if ((evt.metaKey || evt.ctrlKey) && (evt.key === 'k' || evt.key === 'K')) {
    // Allow command palette from anywhere, including other inputs.
    evt.preventDefault();
    openWikiPalette();
  }
});

// ── Phase 54.1 — keyboard shortcut router (?, /, g-chord) ─────────────
// A small state machine for the "g then h / g then w" two-key chord,
// plus single-key shortcuts that only fire outside form fields so we
// don't swallow user typing.

let _kbChord = null;            // 'g' while waiting for the second key
let _kbChordTimer = null;

function _inFormField(el) {
  if (!el) return false;
  const tag = (el.tagName || '').toLowerCase();
  if (tag === 'input' || tag === 'textarea' || tag === 'select') return true;
  return !!el.isContentEditable;
}

function openKbHelp() {
  const el = document.getElementById('kb-help');
  if (el) el.style.display = 'flex';
}
function closeKbHelp() {
  const el = document.getElementById('kb-help');
  if (el) el.style.display = 'none';
}

document.addEventListener('keydown', (evt) => {
  // Chord continuation always fires, even inside form fields,
  // because we only enter a chord state outside form fields below.
  if (_kbChord === 'g') {
    _kbChord = null;
    if (_kbChordTimer) { clearTimeout(_kbChordTimer); _kbChordTimer = null; }
    if (evt.key === 'w' || evt.key === 'W') {
      evt.preventDefault();
      window.location.hash = '#wiki';
      return;
    }
    if (evt.key === 'h' || evt.key === 'H') {
      evt.preventDefault();
      // Close every open modal overlay.
      document.querySelectorAll('.modal-overlay').forEach(m => {
        m.style.display = 'none';
      });
      history.pushState(null, '', window.location.pathname);
      return;
    }
    // Unknown second key — drop the chord and fall through.
  }
  if (_inFormField(evt.target)) return;
  // Don't intercept when modifier keys are held — Ctrl-K etc. has
  // its own handler above.
  if (evt.metaKey || evt.ctrlKey || evt.altKey) return;

  if (evt.key === '?') {
    evt.preventDefault();
    const el = document.getElementById('kb-help');
    if (el && el.style.display === 'flex') closeKbHelp(); else openKbHelp();
    return;
  }
  if (evt.key === 'Escape') {
    closeKbHelp();
    // Let the rest of the app's Escape handlers (modal, menu) run too.
    return;
  }
  if (evt.key === '/') {
    evt.preventDefault();
    openWikiPalette();
    return;
  }
  if (evt.key === 'g' || evt.key === 'G') {
    _kbChord = 'g';
    // Abandon the chord if the user doesn't follow up within 1.2 s.
    if (_kbChordTimer) clearTimeout(_kbChordTimer);
    _kbChordTimer = setTimeout(() => { _kbChord = null; }, 1200);
    return;
  }
});

async function doWikiQuery() {
  const q = document.getElementById('wiki-query-input').value.trim();
  if (!q) return;
  const status = document.getElementById('wiki-status');
  const stream = document.getElementById('wiki-stream');
  const sources = document.getElementById('wiki-sources');

  status.textContent = 'Querying wiki...';
  stream.textContent = '';
  sources.style.display = 'none';
  sources.innerHTML = '';

  // Phase 15 — live tok/s + elapsed stats footer
  const stats = createStreamStats('wiki-stream-stats', 'wiki LLM');
  stats.start();
  setStreamCursor(stream, true);

  const fd = new FormData();
  fd.append('question', q);
  const res = await fetch('/api/wiki/query', {method: 'POST', body: fd});
  const data = await res.json();
  currentJobId = data.job_id;
  if (currentEventSource) currentEventSource.close();

  const source = new EventSource('/api/stream/' + data.job_id);
  currentEventSource = source;

  source.onmessage = function(e) {
    const evt = JSON.parse(e.data);
    if (evt.type === 'token') {
      setStreamCursor(stream, false);
      stream.textContent += evt.text;
      setStreamCursor(stream, true);
      stream.scrollTop = stream.scrollHeight;
      stats.update(evt.text);
    } else if (evt.type === 'model_info') {
      stats.setModel(evt.writer_model || evt.fast_model || 'wiki LLM');
    } else if (evt.type === 'progress') {
      status.textContent = evt.detail || evt.stage;
    } else if (evt.type === 'completed') {
      status.textContent = 'Done';
      stats.done('done');
      setStreamCursor(stream, false);
      if (evt.sources && evt.sources.length) {
        let html = '<div class="u-label-fg-6">Sources</div>';
        evt.sources.forEach(s => { html += '<div class="src-item">' + s + '</div>'; });
        sources.innerHTML = html;
        sources.style.display = 'block';
      }
      source.close(); currentEventSource = null; currentJobId = null;
    } else if (evt.type === 'error') {
      status.textContent = 'Error: ' + evt.message;
      stats.done('error');
      setStreamCursor(stream, false);
      source.close(); currentEventSource = null; currentJobId = null;
    } else if (evt.type === 'done') {
      stats.done('done');
      setStreamCursor(stream, false);
      source.close(); currentEventSource = null; currentJobId = null;
    }
  };
}

// ── Phase 14: Corpus Ask modal (RAG question) ─────────────────────────
function openAskModal() {
  openModal('ask-modal');
  setTimeout(() => document.getElementById('ask-input').focus(), 100);
}

async function doAsk() {
  const q = document.getElementById('ask-input').value.trim();
  if (!q) return;
  const status = document.getElementById('ask-status');
  const stream = document.getElementById('ask-stream');
  const sources = document.getElementById('ask-sources');

  status.textContent = 'Retrieving and generating...';
  stream.textContent = '';
  sources.style.display = 'none';
  sources.innerHTML = '';

  const stats = createStreamStats('ask-stream-stats', 'qwen3.5:27b');
  stats.start();
  setStreamCursor(stream, true);

  const fd = new FormData();
  fd.append('question', q);
  const yf = document.getElementById('ask-year-from').value;
  const yt = document.getElementById('ask-year-to').value;
  if (yf) fd.append('year_from', yf);
  if (yt) fd.append('year_to', yt);

  const res = await fetch('/api/ask', {method: 'POST', body: fd});
  const data = await res.json();
  currentJobId = data.job_id;
  if (currentEventSource) currentEventSource.close();

  const source = new EventSource('/api/stream/' + data.job_id);
  currentEventSource = source;
  let collectedSources = null;

  source.onmessage = function(e) {
    const evt = JSON.parse(e.data);
    if (evt.type === 'token') {
      setStreamCursor(stream, false);
      stream.textContent += evt.text;
      setStreamCursor(stream, true);
      stream.scrollTop = stream.scrollHeight;
      stats.update(evt.text);
    } else if (evt.type === 'model_info') {
      stats.setModel(evt.writer_model);
    } else if (evt.type === 'progress') {
      status.textContent = evt.detail || evt.stage;
    } else if (evt.type === 'sources') {
      collectedSources = evt.sources;
      status.textContent = 'Generating from ' + (evt.n || evt.sources.length) + ' passages...';
    } else if (evt.type === 'completed') {
      status.textContent = 'Done';
      stats.done('done');
      setStreamCursor(stream, false);
      if (collectedSources && collectedSources.length) {
        let html = '<div class="u-label-fg-6">Sources (' + collectedSources.length + ')</div>';
        collectedSources.forEach(s => { html += '<div class="src-item">' + s + '</div>'; });
        sources.innerHTML = html;
        sources.style.display = 'block';
      }
      source.close(); currentEventSource = null; currentJobId = null;
    } else if (evt.type === 'error') {
      status.textContent = 'Error: ' + evt.message;
      stats.done('error');
      setStreamCursor(stream, false);
      source.close(); currentEventSource = null; currentJobId = null;
    } else if (evt.type === 'done') {
      stats.done('done');
      setStreamCursor(stream, false);
      source.close(); currentEventSource = null; currentJobId = null;
    }
  };
}

// ── Phase 36: Tools modal (CLI-parity panel) ──────────────────────────
// Four tabs, four flows:
//   Search     → POST /api/search/(query|similar)  (JSON)
//   Synthesize → POST /api/ask/synthesize + SSE    (streaming)
//   Topics     → GET  /api/catalog/topics[?name=]  (JSON)
//   Corpus     → POST /api/corpus/(enrich|expand) + SSE (subprocess)
let _toolsCorpusJob = null;
let _toolsSynthJob = null;

function openToolsModal() {
  openModal('tools-modal');
  switchToolsTab('tl-search');
  setTimeout(() => document.getElementById('tl-search-q').focus(), 100);
}

// ── Phase 54.6.230 — System Monitor modal ───────────────────────────
//
// Polls /api/monitor on an interval while the modal is open; reuses
// the same snapshot dict the CLI's `sciknow db monitor` renders, so
// any changes to one side automatically flow through. Interval is
// stopped on close + on "Poll = 0s". Not SSE by design — pipeline
// stats don't change fast enough to justify a streaming connection,
// and /api/monitor is cheap (all SELECTs + small external reads).
let _monitorInterval = null;

// Phase 54.6.269 — browser notification for newly-raised ERROR
// alerts when the tab is hidden. Request permission quietly on
// first monitor open; never block. Opt-out honoured when the
// user denies.
function _requestNotificationPermissionIfNeeded() {
  try {
    if (typeof Notification === 'undefined') return;
    if (Notification.permission === 'default') {
      Notification.requestPermission().catch(() => {});
    }
  } catch (_) { /* restrictive sandbox */ }
}

// Fire once per new error code. Caller passes the snap and the
// set of codes that were previously seen (via the iter-23
// localStorage ring). Silent no-op when the tab is visible, the
// user denied the permission, or there's nothing new.
function _maybeNotifyNewErrors(newCodes, alerts) {
  try {
    if (typeof Notification === 'undefined') return;
    if (Notification.permission !== 'granted') return;
    if (document.visibilityState !== 'hidden') return;
    for (const a of alerts) {
      if (a.severity !== 'error') continue;
      if (!a.code || !newCodes.has(a.code)) continue;
      try {
        new Notification('sciknow: ' + a.code, {
          body: a.message || a.code,
          tag: 'sciknow-' + a.code,  // dedupe identical reappearances
          requireInteraction: false,
        });
      } catch (_) { /* ignore per-alert failures */ }
    }
  } catch (_) { /* storage/notification disabled */ }
}

function openMonitorModal() {
  _requestNotificationPermissionIfNeeded();
  openModal('monitor-modal');
  refreshMonitor();
  restartMonitorPoll();
}
function stopMonitorPoll() {
  if (_monitorInterval) {
    clearInterval(_monitorInterval);
    _monitorInterval = null;
  }
}
// Phase 54.6.258 — adaptive poll cadence. When snap.active_jobs is
// non-empty we tick at FAST_POLL_S (default 2s) so TPS / token
// counters on the Active jobs table feel live. When idle we fall
// back to the user-configured "Poll Ns" input value. Poll=0 stays
// OFF regardless — manual-only mode.
const MONITOR_FAST_POLL_S = 2;
let _monitorCurrentCadence = 0;  // 0 = not polling

function _monitorUserPollSeconds() {
  return parseInt(
    (document.getElementById('monitor-poll-seconds') || {}).value || '0', 10
  );
}

function restartMonitorPoll() {
  stopMonitorPoll();
  const secs = _monitorUserPollSeconds();
  if (secs > 0) {
    _monitorCurrentCadence = secs;
    _monitorInterval = setInterval(refreshMonitor, secs * 1000);
  } else {
    _monitorCurrentCadence = 0;
  }
  updateMonitorPollBadge(false);
}

// Called from refreshMonitor after we know whether active_jobs is
// non-empty; flips the interval if needed without burning a tick.
function _monitorAdaptPollRate(activeJobsCount) {
  const user = _monitorUserPollSeconds();
  if (user <= 0) {
    // User explicitly disabled polling; never auto-resume
    updateMonitorPollBadge(false);
    return;
  }
  const desired = activeJobsCount > 0 ? MONITOR_FAST_POLL_S : user;
  if (desired !== _monitorCurrentCadence) {
    stopMonitorPoll();
    _monitorCurrentCadence = desired;
    _monitorInterval = setInterval(refreshMonitor, desired * 1000);
  }
  updateMonitorPollBadge(activeJobsCount > 0 && desired < user);
}

function updateMonitorPollBadge(isFast) {
  const badge = document.getElementById('monitor-poll-badge');
  if (!badge) return;
  if (isFast) {
    badge.style.display = '';
    badge.textContent = '(fast tick · ' + MONITOR_FAST_POLL_S + 's)';
    badge.style.color = '#080';
  } else {
    badge.style.display = 'none';
  }
}

function refreshMonitor() {
  fetch('/api/monitor?days=14')
    .then(r => r.ok ? r.json() : Promise.reject(new Error('HTTP ' + r.status)))
    .then(snap => {
      renderMonitor(snap);
      rebuildMonitorNavStrip();
      applyMonitorFilter();
      _monitorAdaptPollRate(((snap && snap.active_jobs) || []).length);
    })
    .catch(err => {
      const target = document.getElementById('monitor-content');
      if (target) {
        target.innerHTML = '<p class="u-note" style="color:var(--fg-danger, #c00);">'
          + 'Monitor fetch failed: ' + String(err) + '</p>';
      }
    });
}

// Phase 54.6.256 — rebuild jump-to nav strip from the current set of
// h4 headings. Each heading gets a slug id (derived from its text),
// and the strip renders a chip per heading that scrolls it into view.
// Hidden when ≤3 headings to keep clean installs quiet.
// Phase 54.6.265 — clicking a nav chip also updates the URL hash
// via history.replaceState (not pushState — we don't want monitor
// navigation polluting browser history). Reloads to that URL
// scroll straight to the target panel via openMonitorFromHash().
function rebuildMonitorNavStrip() {
  const strip = document.getElementById('monitor-nav-strip');
  const target = document.getElementById('monitor-content');
  if (!strip || !target) return;
  strip.innerHTML = '';
  const headings = target.querySelectorAll('h4');
  if (headings.length <= 3) {
    strip.style.display = 'none';
    return;
  }
  strip.style.display = 'flex';
  const seen = {};
  headings.forEach((h, i) => {
    const txt = (h.textContent || ('section ' + i)).trim();
    let slug = 'mon-' + txt.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
    // de-dupe
    if (seen[slug]) {
      seen[slug] += 1;
      slug = slug + '-' + seen[slug];
    } else {
      seen[slug] = 1;
    }
    h.id = slug;
    const chip = document.createElement('button');
    chip.className = 'btn btn--sm';
    chip.style.fontSize = '0.8em';
    chip.style.padding = '0.15em 0.5em';
    chip.textContent = txt;
    chip.title = 'Jump to ' + txt + ' (updates URL hash — copy the address to share)';
    chip.onclick = () => {
      h.scrollIntoView({ behavior: 'smooth', block: 'start' });
      try {
        history.replaceState(null, '', '#' + slug);
      } catch (e) { /* restrictive sandbox */ }
    };
    strip.appendChild(chip);
  });
}

// Phase 54.6.265 — if the page URL has a #mon-* hash at load time,
// open the monitor modal and scroll the matching section into view
// after the first render lands. The hashchange listener also picks
// up runtime changes so typing #mon-gpu in the URL bar works.
function openMonitorFromHash() {
  const h = (window.location.hash || '').slice(1);
  if (!h || !h.startsWith('mon-')) return;
  // Open the modal if closed; refreshMonitor will populate it.
  const modal = document.getElementById('monitor-modal');
  const wasOpen = modal && modal.style.display !== 'none';
  if (!wasOpen && typeof openMonitorModal === 'function') {
    openMonitorModal();
  }
  // Wait for the first render to finish before scrolling — poll
  // up to 10x (5s) then give up.
  let tries = 0;
  const timer = setInterval(() => {
    tries += 1;
    const target = document.getElementById(h);
    if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      clearInterval(timer);
    } else if (tries >= 10) {
      clearInterval(timer);
    }
  }, 500);
}

window.addEventListener('DOMContentLoaded', openMonitorFromHash);
window.addEventListener('hashchange', openMonitorFromHash);

// Phase 54.6.254 — live filter across every data row in the modal.
// Skips header rows (<th>) and keeps section headings (<h4>) always
// visible. "M rows · N of M visible" feedback in the count label.
// Blank query = show everything.
function applyMonitorFilter() {
  const input = document.getElementById('monitor-filter');
  const countLabel = document.getElementById('monitor-filter-count');
  if (!input) return;
  const q = (input.value || '').trim().toLowerCase();
  const target = document.getElementById('monitor-content');
  if (!target) return;
  const rows = target.querySelectorAll('tr');
  let shown = 0, total = 0;
  rows.forEach(r => {
    // Header rows: keep visible
    if (r.querySelector('th') && !r.querySelector('td')) {
      r.style.display = '';
      return;
    }
    total += 1;
    if (!q) { r.style.display = ''; shown += 1; return; }
    const text = (r.textContent || '').toLowerCase();
    if (text.indexOf(q) !== -1) {
      r.style.display = '';
      shown += 1;
    } else {
      r.style.display = 'none';
    }
  });
  if (countLabel) {
    if (q) {
      countLabel.textContent = shown + ' / ' + total + ' rows';
    } else {
      countLabel.textContent = '';
    }
  }
}

function clearMonitorFilter() {
  const input = document.getElementById('monitor-filter');
  if (input) { input.value = ''; applyMonitorFilter(); input.focus(); }
}

// Phase 54.6.268 — fetch /api/monitor/alerts-md and copy its plain-
// text body to the clipboard. Button flashes ✓ on success. Falls
// back to window.prompt for insecure origins / disabled clipboard.
function copyAlertsMarkdown(btn) {
  fetch('/api/monitor/alerts-md')
    .then(r => r.ok ? r.text() : Promise.reject(new Error('HTTP ' + r.status)))
    .then(md => {
      const flash = (ok) => {
        if (!btn) return;
        const orig = btn.textContent;
        btn.textContent = ok ? '✓ Copied' : '! failed';
        setTimeout(() => { btn.textContent = orig; }, 1100);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(md).then(
          () => flash(true),
          () => { flash(false); window.prompt('Copy this Markdown:', md); }
        );
      } else {
        window.prompt('Copy this Markdown:', md);
      }
    })
    .catch(err => alert('Alerts-MD fetch failed: ' + String(err)));
}

// Phase 54.6.257 — copy a suggested-fix command from an alert to the
// clipboard. Flashes the button briefly so the user gets a success
// cue. Falls back to a window.prompt() on insecure origins where
// navigator.clipboard is unavailable.
function copyMonitorAction(cmd, btn) {
  const flash = (ok) => {
    if (!btn) return;
    const orig = btn.textContent;
    btn.textContent = ok ? '✓' : '!';
    setTimeout(() => { btn.textContent = orig; }, 900);
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(cmd).then(
      () => flash(true),
      () => { flash(false); window.prompt('Copy this command:', cmd); }
    );
  } else {
    window.prompt('Copy this command:', cmd);
  }
}

// Phase 54.6.255 — download current snapshot as JSON file. Re-fetches
// (doesn't grab last-rendered state) so operator gets fresh data.
// Filename carries UTC timestamp so multiple snapshots don't collide.
function downloadMonitorSnapshot() {
  fetch('/api/monitor?days=14')
    .then(r => r.ok ? r.json() : Promise.reject(new Error('HTTP ' + r.status)))
    .then(snap => {
      const blob = new Blob([JSON.stringify(snap, null, 2)],
                            { type: 'application/json' });
      const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
      const slug = (snap.project && snap.project.slug) ? snap.project.slug : 'default';
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'sciknow-monitor-' + slug + '-' + ts + '.json';
      document.body.appendChild(a);
      a.click();
      setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 100);
    })
    .catch(err => alert('Snapshot download failed: ' + String(err)));
}

// Phase 54.6.272 — keyboard-help overlay toggle. Can be invoked
// from the ? button or the `?` key.
function toggleMonitorHelp(forceState) {
  const ov = document.getElementById('monitor-help-overlay');
  if (!ov) return;
  const shouldShow = (typeof forceState === 'boolean')
    ? forceState
    : ov.style.display === 'none';
  ov.style.display = shouldShow ? 'block' : 'none';
}

// Phase 54.6.255 — keyboard shortcut: `/` focuses the monitor filter
// while the modal is open. Skip if the user is already typing in an
// input so `/` in other text fields stays a literal slash.
// Phase 54.6.272 — `?` toggles the help overlay, Esc closes it (and
// any other handlers get their chance first).
(function installMonitorKeyboardShortcuts() {
  document.addEventListener('keydown', function (e) {
    const modal = document.getElementById('monitor-modal');
    if (!modal || modal.style.display === 'none') return;
    const tag = (e.target && e.target.tagName) || '';
    const typing = tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';

    if (e.key === '/' && !typing) {
      const filterInput = document.getElementById('monitor-filter');
      if (!filterInput) return;
      e.preventDefault();
      filterInput.focus();
      filterInput.select();
      return;
    }
    if (e.key === '?' && !typing) {
      e.preventDefault();
      toggleMonitorHelp();
      return;
    }
    if (e.key === 'Escape') {
      const ov = document.getElementById('monitor-help-overlay');
      if (ov && ov.style.display !== 'none') {
        e.preventDefault();
        toggleMonitorHelp(false);
        return;
      }
    }
  });
})();

function _escHTML(s) {
  if (s === null || s === undefined) return '—';
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function _fmtMs(v) {
  if (v === null || v === undefined) return '—';
  if (v >= 60000) return (v / 60000).toFixed(1) + 'm';
  if (v >= 1000) return (v / 1000).toFixed(1) + 's';
  return Math.round(v) + 'ms';
}

function _fmtNum(n) {
  return (n || 0).toLocaleString();
}

function renderMonitor(snap) {
  const target = document.getElementById('monitor-content');
  if (!target || !snap) return;

  const project = snap.project || {};
  const corpus = snap.corpus || {};
  const gpus = snap.gpu || [];
  const loaded = ((snap.llm || {}).loaded_models) || [];
  const qcolls = snap.qdrant || [];
  const backends = snap.converter_backends || [];
  const pipe = snap.pipeline || {};
  const timing = pipe.stage_timing || [];
  const fails = pipe.stage_failures || [];
  const activity = pipe.recent_activity || [];
  const rates = pipe.rates || {};
  const queueStates = pipe.queue_states || {};
  const topFailures = pipe.top_failures || [];
  const hourly = pipe.hourly_throughput || [];
  const clusters = snap.topic_clusters || [];
  const llmUsage = ((snap.llm || {}).usage_last_days) || [];
  const storage = snap.storage || {};
  const disk = storage.disk || {};
  const pgMb = storage.pg_database_mb || 0;
  const pendingDl = snap.pending_downloads || 0;
  // Phase 54.6.243 additions
  const alerts = snap.alerts || [];
  const inbox = snap.inbox || {};
  const qsig = snap.quality_signals || {};
  const wikiMat = snap.wiki_materialization || {};
  const projectsOverview = snap.projects_overview || [];
  // Phase 54.6.280 — citation graph connectivity metrics.
  const cgraph = snap.citation_graph || {};
  // Phase 54.6.282 — chunker section-type coverage.
  const seccov = snap.section_coverage || {};
  // Phase 54.6.287 — per-converter-backend section coverage.
  const seccovByBackend = snap.section_coverage_by_backend || [];
  // Phase 54.6.293 — cached sidecar integrity audit.
  const sidecarAudit = snap.sidecar_audit || {};
  // Phase 54.6.298 — per-field enrichment coverage.
  const enrichment = snap.enrichment || {};
  // Phase 54.6.294 — top-N slow ingest docs.
  const slowDocs = snap.slow_docs || [];
  // Phase 54.6.296 — per-collection payload-index health check.
  const qdrantIndexes = snap.qdrant_indexes || {};
  // Phase 54.6.299 — HNSW / quantization drift check.
  const qdrantHnsw = snap.qdrant_hnsw || {};
  // Phase 54.6.301 — retrieval latency summary.
  const retrievalLat = snap.retrieval_latency || {};
  // Phase 54.6.284 — retraction detail (counts + recent list).
  const retractions = snap.retractions || {};
  // Phase 54.6.244 additions
  const funnel = snap.ingest_funnel || [];
  const hourlyFails = snap.pipeline_hourly_failures || [];
  const diskFree = snap.disk_free || {};
  // Phase 54.6.249 additions — book activity, LLM cost, corpus growth
  const bookAct = snap.book_activity || {};
  const costTotals = snap.cost_totals || {};
  const corpusGrowth = snap.corpus_growth || {};
  // Phase 54.6.250 additions — bench + backup freshness
  const benchFresh = snap.bench_freshness || {};
  const backupFresh = snap.backup_freshness || {};
  // Phase 54.6.251 additions — meta quality + year histogram + coverage
  const metaQ = snap.meta_quality || {};
  const yearHist = snap.year_histogram || [];
  const embedCov = snap.embeddings_coverage || {};
  const vcov = snap.visuals_coverage || {};
  // Phase 54.6.252 — config drift list (active project overrides .env)
  const configDrift = snap.config_drift || [];
  // Phase 54.6.260 — log tail panel data
  const logTail = snap.log_tail || {};
  // Phase 54.6.262 — services reachability
  const services = snap.services || {};

  const sections = [];

  // Phase 54.6.253 — doctor-style verdict banner at the top of the
  // modal. Traffic-light verdict (OK / WARN / FAIL) mirrors the CLI
  // `sciknow db doctor` output: same rule — error wins over warn
  // wins over info. Rendered even when there are zero alerts so
  // operators get an explicit "all green" confirmation.
  {
    const errN = alerts.filter(a => a.severity === 'error').length;
    const warnN = alerts.filter(a => a.severity === 'warn').length;
    const infoN = alerts.filter(a => a.severity === 'info').length;
    const verdict = errN ? 'FAIL' : warnN ? 'WARN' : 'OK';
    const palette = {
      OK: { colour: '#080', icon: '✓', bg: 'rgba(0,130,0,0.08)' },
      WARN: { colour: '#b70', icon: '⚠', bg: 'rgba(230,150,0,0.08)' },
      FAIL: { colour: '#c33', icon: '✗', bg: 'rgba(200,50,50,0.08)' },
    };
    const p = palette[verdict];
    // Phase 54.6.259 — composite health score (0-100). Renders next
    // to the verdict. Penalty breakdown goes into the title tooltip
    // so hover reveals why the score isn't 100.
    // Phase 54.6.267 — health-score ring in localStorage (60 samples
    // per origin). Appended per render; trimmed to size; rendered as
    // a mini sparkline next to the score. Lets operators spot a
    // dropping trend ("were at 100, now 70") without server-side
    // state.
    const health = snap.health_score || {};
    const hs = typeof health.score === 'number' ? health.score : null;
    let healthHtml = '';
    if (hs !== null) {
      const hc = hs >= 90 ? '#080' : hs >= 60 ? '#b70' : '#c33';
      const pen = (health.penalties || []).join(' · ') || 'no penalties';
      // Load + update the local ring
      const HS_KEY = 'sciknow.monitor.healthRing';
      let ring = [];
      try {
        ring = JSON.parse(window.localStorage.getItem(HS_KEY) || '[]');
        if (!Array.isArray(ring)) ring = [];
      } catch (_) { ring = []; }
      ring.push(hs);
      if (ring.length > 60) ring = ring.slice(-60);
      try {
        window.localStorage.setItem(HS_KEY, JSON.stringify(ring));
      } catch (_) { /* storage disabled */ }
      let sparkHtml = '';
      if (ring.length >= 2) {
        const sparkRamp = '▁▂▃▄▅▆▇█';
        // Scale 0..100 across the 8-level ramp
        const spark = ring.map(v => {
          const idx = Math.min(Math.round((v / 100) * (sparkRamp.length - 1)), sparkRamp.length - 1);
          return sparkRamp[idx];
        }).join('');
        const minV = Math.min(...ring), maxV = Math.max(...ring);
        sparkHtml = ' <span title="last ' + ring.length + ' readings · '
          + 'min ' + minV + ' / max ' + maxV + '" '
          + 'style="font-family:monospace;color:' + hc + ';letter-spacing:0.5px;">'
          + spark + '</span>';
      }
      healthHtml = '<div title="' + _escHTML(pen) + '">'
        + '<strong>Health</strong> '
        + '<span style="color:' + hc + ';font-size:1.1em;font-weight:bold;">'
        + hs + '/100</span>' + sparkHtml + '</div>';
    }
    // Phase 54.6.262 — services reachability pill group. PG / Qdrant
    // / LLM substrate with latency-aware colouring (green up, red
    // down, dim "n/a" when the probe key is missing). v2 Phase A
    // adds infer_writer/embedder/reranker; ollama only appears on
    // the v1 fallback path. The `if (!info) continue` below silently
    // skips probes the substrate didn't run, so the same pill list
    // works on both v1 and v2.
    let svcHtml = '';
    const svcOrder = [
      ['postgres', 'PG'], ['qdrant', 'Qdr'],
      ['infer_writer', 'Writer'],
      ['infer_embedder', 'Embedder'],
      ['infer_reranker', 'Reranker'],
      ['ollama', 'Ollama'],
    ];
    const svcBits = [];
    for (const [key, label] of svcOrder) {
      const info = services[key];
      if (!info) continue;
      const c = info.up ? '#080' : '#c33';
      const lat = info.latency_ms !== undefined ? info.latency_ms + 'ms' : '?';
      svcBits.push('<span style="color:' + c + ';" title="' + _escHTML(info.error || '')
        + '">' + (info.up ? '●' : '✗') + ' <strong>' + label + '</strong> '
        + _escHTML(lat) + '</span>');
    }
    if (svcBits.length) {
      svcHtml = '<div style="display:flex;gap:1em;">' + svcBits.join('') + '</div>';
    }
    sections.push('<div style="border:2px solid ' + p.colour
      + ';background:' + p.bg + ';padding:0.5em 0.75em;border-radius:4px;'
      + 'margin-bottom:1em;display:flex;align-items:center;gap:1.5em;flex-wrap:wrap;">'
      + '<div style="font-size:1.5em;color:' + p.colour + ';font-weight:bold;">'
      + p.icon + ' ' + verdict + '</div>'
      + healthHtml
      + '<div><strong>Errors</strong> <span style="color:#c33;">' + errN + '</span></div>'
      + '<div><strong>Warn</strong> <span style="color:#b70;">' + warnN + '</span></div>'
      + '<div><strong>Info</strong> <span style="color:#36a;">' + infoN + '</span></div>'
      + svcHtml
      + '<div class="u-muted" style="margin-left:auto;font-size:0.85em;">'
      + 'Equivalent: <code>sciknow library doctor</code></div>'
      + '</div>');
  }

  // Phase 54.6.243 — consolidated alert banner. Same list the CLI
  // surfaces; rendered as a coloured card at the very top so the
  // modal leads with "anything on fire?".
  // Phase 54.6.257 — each alert that carries an `action` renders a
  // 📋 copy button next to the message. Uses the Clipboard API when
  // available (secure-origin requirement met on localhost) and
  // falls back to a noop prompt otherwise.
  if (alerts.length) {
    const worstSev = alerts.some(a => a.severity === 'error') ? 'error'
      : alerts.some(a => a.severity === 'warn') ? 'warn' : 'info';
    const border = worstSev === 'error' ? '#c33'
      : worstSev === 'warn' ? '#e90' : '#38c';
    const bg = worstSev === 'error' ? 'rgba(200,50,50,0.08)'
      : worstSev === 'warn' ? 'rgba(230,150,0,0.08)'
        : 'rgba(50,130,200,0.06)';
    // Phase 54.6.266 — delta vs last seen alert codes.
    // Load persisted acknowledgement set (localStorage is per-origin
    // so this scales to any number of sessions). Compute `newCodes`
    // as the difference. Save current codes back immediately so a
    // 10-second rapid-fire poll doesn't keep showing NEW once
    // acknowledged.
    const STORAGE_KEY = 'sciknow.monitor.seenAlertCodes';
    let seenSet = new Set();
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY) || '[]';
      seenSet = new Set(JSON.parse(raw));
    } catch (_) { /* storage disabled */ }
    const currentCodes = new Set(alerts.map(a => a.code).filter(Boolean));
    const newCodes = new Set();
    for (const c of currentCodes) {
      if (!seenSet.has(c)) newCodes.add(c);
    }
    try {
      window.localStorage.setItem(
        STORAGE_KEY, JSON.stringify(Array.from(currentCodes))
      );
    } catch (_) { /* storage disabled */ }
    // Phase 54.6.269 — passive-guardian notification. Only fires
    // when the tab is hidden AND there's a new error-level code,
    // so idle-tab operators see "something broke" without polling.
    _maybeNotifyNewErrors(newCodes, alerts);
    let html = '<div style="border:1px solid ' + border
      + ';background:' + bg + ';padding:0.5em 0.75em;border-radius:4px;margin-bottom:1em;">'
      + '<div style="display:flex;align-items:center;gap:1em;">'
      + '<strong>Alerts</strong>'
      + (newCodes.size
        ? ' <span style="color:#c33;font-weight:bold;">· ' + newCodes.size + ' NEW</span>'
        : '')
      // Phase 54.6.268 — Markdown export button. Fetches the shared
      // /api/monitor/alerts-md endpoint so the copied text matches
      // what `sciknow db monitor --alerts-md` produces.
      + ' <button class="btn btn--sm" style="padding:0.1em 0.5em;font-size:0.75em;margin-left:auto;" '
      + 'onclick="copyAlertsMarkdown(this)" '
      + 'title="Copy current alerts as a Markdown block — paste into Slack / Linear / GitHub ticket">📋 Copy as MD</button>'
      + '</div>'
      + '<ul style="margin:0.25em 0 0 1em;padding:0;">';
    for (const a of alerts.slice(0, 8)) {
      const icon = a.severity === 'error' ? '✗' : a.severity === 'warn' ? '⚠' : 'ℹ';
      const colour = a.severity === 'error' ? '#c33'
        : a.severity === 'warn' ? '#b70' : '#36a';
      const isNew = a.code && newCodes.has(a.code);
      const newBadge = isNew
        ? ' <span style="background:#c33;color:#fff;padding:0 0.3em;border-radius:3px;font-size:0.75em;font-weight:bold;margin-left:0.3em;">NEW</span>'
        : '';
      html += '<li style="color:' + colour + ';">' + icon + ' '
        + _escHTML(a.message || a.code || '') + newBadge;
      if (a.action) {
        // Attribute-safe: single-quoted HTML with backslash-escaped
        // single quotes inside the onclick arg.
        const safeAct = String(a.action).replace(/\\/g, '\\\\').replace(/'/g, "\\'");
        html += ' <code style="background:rgba(0,0,0,0.06);padding:0.1em 0.3em;border-radius:3px;'
             + 'font-size:0.85em;color:#444;margin-left:0.4em;">$ ' + _escHTML(a.action) + '</code>'
             + ' <button class="btn btn--sm" style="padding:0.1em 0.4em;font-size:0.8em;" '
             + 'title="Copy command to clipboard" '
             + 'onclick="copyMonitorAction(\'' + safeAct + '\', this)">📋</button>';
      }
      html += '</li>';
    }
    html += '</ul></div>';
    sections.push(html);
  }

  // Header/meta
  sections.push('<div style="display:flex;gap:2em;flex-wrap:wrap;margin-bottom:1em;">'
    + '<div><strong>Project</strong>: ' + _escHTML(project.slug || '—') + '</div>'
    + '<div><strong>DB</strong>: ' + _escHTML(project.pg_database || '—') + '</div>'
    + '<div><strong>Last refresh</strong>: <code>' + _escHTML(snap.last_refresh || 'never') + '</code></div>'
    + '</div>');

  // Phase 54.6.252 — config drift card. Non-empty means the active
  // project is overriding .env-provided values at runtime. Yellow
  // tint because it's typically intentional (user switched projects
  // without cleaning .env) but worth surfacing since any manual
  // read of .env would show the wrong db/data_dir.
  if (configDrift.length) {
    let html = '<div style="border:1px solid #e90;background:rgba(230,150,0,0.08);'
      + 'padding:0.5em 0.75em;border-radius:4px;margin-bottom:1em;">'
      + '<strong>Config drift</strong> — active project overrides '
      + configDrift.length + ' .env key(s):<ul style="margin:0.25em 0 0 1em;padding:0;">';
    for (const entry of configDrift) {
      html += '<li><code>' + _escHTML(entry) + '</code></li>';
    }
    html += '</ul><div style="font-size:0.85em;margin-top:0.25em;color:var(--fg-muted);">'
      + 'Drop the listed keys from <code>.env</code> to silence.</div></div>';
    sections.push(html);
  }

  // ── Rates + ETA + queue banner ──────────────────────────────────
  // Hot info at the top of the modal so "where am I?" is answered
  // before the user scrolls past the corpus table.
  const rate1h = Math.round(rates.rate_1h || 0);
  const rate4h = Math.round(rates.rate_4h || 0);
  const eta = rates.eta_hours;
  const etaTxt = (eta === null || eta === undefined) ? '—'
    : (eta < 1 ? Math.round(eta * 60) + 'm'
      : eta < 24 ? eta.toFixed(1) + 'h'
        : (eta / 24).toFixed(1) + 'd');
  const qStr = Object.keys(queueStates).length
    ? Object.entries(queueStates).map(([k, v]) => k + ':' + v).join(' ')
    : 'idle';
  sections.push('<div style="display:flex;gap:2em;flex-wrap:wrap;margin-bottom:1em;padding:0.5em;background:var(--bg-alt, #f5f5f5);border-radius:4px;">'
    + '<div><strong>Rate (1h / 4h)</strong>: ' + rate1h + ' / ' + rate4h + ' docs/hr</div>'
    + '<div><strong>ETA</strong>: ' + _escHTML(etaTxt) + '</div>'
    + '<div><strong>Queue</strong>: ' + _escHTML(qStr) + '</div>'
    + (pendingDl ? '<div><strong>Pending DL</strong>: ' + pendingDl + '</div>' : '')
    + (inbox.count ? '<div><strong>Inbox</strong>: ' + inbox.count + ' pdf'
      // Phase 54.6.281 — inline age-bucket breakdown.  Palette
      // matches the CLI: fresh green, week cyan, month amber, stale
      // grey.  Only the non-zero buckets render.
      + (function() {
        const b = inbox.age_buckets || {};
        const parts = [];
        const pals = [
          ['fresh_24h', '24h', '#080'],
          ['week', '1w', '#28a'],
          ['month', '1mo', '#b70'],
          ['stale', 'old', '#888'],
        ];
        for (const [k, lbl, col] of pals) {
          const n = b[k] || 0;
          if (n > 0) {
            parts.push('<span style="color:' + col + ';margin-left:0.4em;" '
              + 'title="' + lbl + ' bucket">' + lbl + '·'
              + '<strong>' + n + '</strong></span>');
          }
        }
        return parts.length
          ? ' <span style="color:var(--fg-muted);font-size:0.9em;">('
            + parts.join('') + ')</span>'
          : '';
      })()
      + '</div>' : '')
    + '</div>');

  // Phase 54.6.243 — retrieval-quality strip (abstracts coverage +
  // chunk sizing + KG density), one line each so the ops signals
  // are scannable at a glance without hunting through tables.
  const qualityBits = [];
  if (qsig.abstract_eligible) {
    qualityBits.push('<div><strong>Abstracts</strong>: '
      + _fmtNum(qsig.abstract_covered) + '/' + _fmtNum(qsig.abstract_eligible)
      + ' (' + (qsig.abstract_pct || 0).toFixed(0) + '%)</div>');
  }
  if (qsig.chunk_p50_chars) {
    qualityBits.push('<div><strong>Chunk chars</strong>: p50 '
      + _fmtNum(qsig.chunk_p50_chars) + ' · p95 ' + _fmtNum(qsig.chunk_p95_chars || 0) + '</div>');
  }
  if (qsig.kg_triples_per_doc && qsig.kg_triples_per_doc > 0) {
    qualityBits.push('<div><strong>KG density</strong>: '
      + qsig.kg_triples_per_doc.toFixed(1) + ' triples/doc</div>');
  }
  if (wikiMat.topics_total) {
    qualityBits.push('<div><strong>Wiki materialization</strong>: '
      + _fmtNum(wikiMat.wiki_pages) + '/' + _fmtNum(wikiMat.topics_total)
      + ' (' + (wikiMat.pct || 0).toFixed(0) + '%)</div>');
  }
  // Phase 54.6.250 — bench + backup freshness pills. Same
  // threshold palette as the CLI header chips: green <7d,
  // yellow 7-14d (bench) / 7-30d (backup), red beyond.
  const _freshnessPill = (ageDays, warnAt, errorAt) => {
    if (ageDays === null || ageDays === undefined) return null;
    let colour = '#080';
    if (ageDays >= errorAt) colour = '#c33';
    else if (ageDays >= warnAt) colour = '#e80';
    const txt = ageDays < 1
      ? Math.round(ageDays * 24) + 'h'
      : ageDays.toFixed(0) + 'd';
    return { colour, txt };
  };
  const benchPill = _freshnessPill(benchFresh.newest_age_days, 7, 14);
  if (benchPill) {
    qualityBits.push('<div><strong>Bench</strong>: '
      + '<span style="color:' + benchPill.colour + ';">'
      + benchPill.txt + '</span></div>');
  }
  const backupPill = _freshnessPill(backupFresh.newest_age_days, 7, 30);
  if (backupPill) {
    qualityBits.push('<div><strong>Backup</strong>: '
      + '<span style="color:' + backupPill.colour + ';">'
      + backupPill.txt + '</span>'
      + (backupFresh.count ? ' <span class="u-muted">(' + backupFresh.count + ' sets)</span>' : '')
      + '</div>');
  }
  if (qualityBits.length) {
    sections.push('<div style="display:flex;gap:2em;flex-wrap:wrap;margin-bottom:1em;">'
      + qualityBits.join('') + '</div>');
  }

  // Phase 54.6.244 — ingest funnel table (all canonical pipeline
  // stages with doc counts; stages with zero docs hidden).
  if (funnel.length && funnel.some(f => f.n > 0)) {
    const maxN = Math.max(...funnel.map(f => f.n || 0)) || 1;
    let html = '<h4>Ingest funnel</h4><table class="stats-table" style="width:100%;">'
      + '<tr><th>Stage</th><th>Documents</th><th>Fill</th></tr>';
    for (const row of funnel) {
      if (!row.n) continue;
      const pct = (row.n / maxN * 100).toFixed(0);
      const colour = row.stage === 'complete' ? '#080'
        : row.stage === 'failed' ? '#c33' : '#e90';
      html += '<tr><td>' + _escHTML(row.stage) + '</td><td>'
        + _fmtNum(row.n) + '</td><td><div style="background:' + colour
        + ';height:8px;width:' + pct + '%;border-radius:2px;"></div></td></tr>';
    }
    html += '</table>';
    sections.push(html);
  }

  // Phase 54.6.244 — disk-free strip. Renders a compact
  // "data_dir free: 120G / 500G (76% used)" line when available.
  if (diskFree.total_mb) {
    const freeGb = (diskFree.free_mb / 1024).toFixed(0);
    const totalGb = (diskFree.total_mb / 1024).toFixed(0);
    const pct = diskFree.pct_used.toFixed(0);
    const colour = diskFree.pct_used >= 95 ? '#c33'
      : diskFree.pct_used >= 90 ? '#e90' : '#444';
    sections.push('<div style="margin:0.5em 0;color:' + colour + ';">'
      + '<strong>Disk</strong>: ' + freeGb + 'G free of ' + totalGb
      + 'G (' + pct + '% used)</div>');
  }

  // ── 24h throughput sparkline ────────────────────────────────────
  // Phase 54.6.248 — shared _spark() helper for throughput, hourly
  // failures, and GPU trend (CLI has _sparkline() in
  // sciknow/cli/db.py; this mirror keeps the look consistent).
  const _spark = (values, opts) => {
    const o = opts || {};
    const chars = o.chars || '▁▂▃▄▅▆▇█';
    const defColour = o.defaultColour || '#080';
    const warnColour = o.warnColour || '#e80';
    const critColour = o.critColour || '#c00';
    if (!values || !values.length) return '<span style="opacity:0.3">—</span>';
    const mx = Math.max(...values, 1);
    return values.map(v => {
      if (v === 0) return '<span style="opacity:0.3">▁</span>';
      const idx = Math.min(Math.round((v / mx) * (chars.length - 1)), chars.length - 1);
      const pct = v / mx;
      const c = pct >= 0.9 ? critColour : pct >= 0.5 ? warnColour : defColour;
      return '<span style="color:' + c + '">' + chars[idx] + '</span>';
    }).join('');
  };

  if (hourly.length) {
    const peakH = Math.max(...hourly);
    const nowH = hourly[hourly.length - 1] || 0;
    sections.push('<h4>24h docs/hour</h4>'
      + '<div style="font-family:monospace;font-size:1.3em;letter-spacing:1px;">'
      + _spark(hourly) + '</div>'
      + '<div style="color:var(--fg-muted);font-size:0.85em;">peak ' + peakH + '  ·  now ' + nowH + '</div>');
  }

  // Phase 54.6.249 — 14-week corpus growth sparkline with headline
  // 24h / 7d / 30d deltas. CLI has the same headline ("growth +N /24h
  // +M /7d") in the footer; web renders a wider sparkline too since
  // it has the space.
  if (corpusGrowth.weekly_sparkline && corpusGrowth.weekly_sparkline.some(v => v > 0)) {
    const weeks = corpusGrowth.weekly_sparkline || [];
    const weeksBack = corpusGrowth.weeks_back || weeks.length;
    sections.push('<h4>Corpus growth (last ' + weeksBack + ' weeks)</h4>'
      + '<div style="font-family:monospace;font-size:1.3em;letter-spacing:1px;">'
      + _spark(weeks) + '</div>'
      + '<div style="color:var(--fg-muted);font-size:0.85em;">'
      + 'added <strong>' + _fmtNum(corpusGrowth.last_24h || 0) + '</strong> /24h · '
      + '<strong>' + _fmtNum(corpusGrowth.last_7d || 0) + '</strong> /7d · '
      + '<strong>' + _fmtNum(corpusGrowth.last_30d || 0) + '</strong> /30d'
      + '</div>');
  }

  // Phase 54.6.248 — 24h failure sparkline. Always-red palette so a
  // single unexpected blip visually jumps out at the operator. CLI
  // aligns hourly_throughput and pipeline_hourly_failures to the
  // same 24h window so the two sparklines stack for diff-reading.
  if (hourlyFails.length && hourlyFails.some(v => v > 0)) {
    const totalFails = hourlyFails.reduce((a, b) => a + b, 0);
    sections.push('<h4>24h failures</h4>'
      + '<div style="font-family:monospace;font-size:1.3em;letter-spacing:1px;">'
      + _spark(hourlyFails, {
        defaultColour: '#c33', warnColour: '#c33', critColour: '#900'
      }) + '</div>'
      + '<div style="color:var(--fg-muted);font-size:0.85em;">total failures in window: '
      + totalFails + '</div>');
  }

  // Corpus + GPU + loaded models in a row
  const corpusCells = [
    ['Docs (done/total)', _fmtNum(corpus.documents_complete) + ' / ' + _fmtNum(corpus.documents_total)],
    ['Chunks', _fmtNum(corpus.chunks)], ['Citations', _fmtNum(corpus.citations)],
    ['Visuals', _fmtNum(corpus.visuals)], ['KG triples', _fmtNum(corpus.kg_triples)],
    ['Wiki pages', _fmtNum(corpus.wiki_pages)], ['Institutions', _fmtNum(corpus.institutions)],
  ];
  sections.push('<h4>Corpus</h4><table class="stats-table" style="width:100%;">'
    + '<tr>' + corpusCells.map(c => '<th>' + _escHTML(c[0]) + '</th>').join('') + '</tr>'
    + '<tr>' + corpusCells.map(c => '<td>' + _escHTML(c[1]) + '</td>').join('') + '</tr>'
    + '</table>');

  // Phase 54.6.249 — active-book summary + LLM cost totals strip. Both
  // were collected by `collect_monitor_snapshot` but only the CLI
  // footer surfaced them. Card + inline strip render so operators
  // see "where is autowrite right now?" and "is the LLM spend
  // reasonable?" without opening another view.
  if (bookAct && bookAct.title) {
    const pct = bookAct.chapters_total
      ? Math.round((bookAct.chapters_drafted / bookAct.chapters_total) * 100)
      : 0;
    const lastUp = bookAct.last_updated ? _escHTML(bookAct.last_updated) : '—';
    let html = '<h4>Active book</h4>'
      + '<div style="display:flex;gap:2em;flex-wrap:wrap;padding:0.5em 0.75em;'
      + 'background:var(--bg-alt, #f5f5f5);border-radius:4px;">'
      + '<div><strong>' + _escHTML(bookAct.title) + '</strong>'
      + (bookAct.book_type ? ' <span class="u-muted">(' + _escHTML(bookAct.book_type) + ')</span>' : '')
      + '</div>'
      + '<div><strong>Chapters drafted</strong>: ' + _fmtNum(bookAct.chapters_drafted || 0)
      + ' / ' + _fmtNum(bookAct.chapters_total || 0) + ' (' + pct + '%)</div>'
      + '<div><strong>Words</strong>: ' + _fmtNum(bookAct.total_words || 0) + '</div>'
      + '<div><strong>Updated</strong>: <code>' + lastUp + '</code></div>'
      + '</div>';
    // Phase 54.6.302 — per-chapter velocity table.  Identifies
    // stalled chapters (no drafts) + chapters that need another
    // pass (versions > 1 but still below target).
    const chapters = snap.book_chapter_velocity || [];
    if (chapters.length) {
      html += '<table class="stats-table" style="width:100%;font-size:0.9em;margin-top:0.5em;">'
        + '<tr><th style="width:2.5em;">#</th><th>Title</th>'
        + '<th style="width:14em;">Progress</th>'
        + '<th style="width:8em;text-align:right;">Words</th>'
        + '<th style="width:3em;text-align:right;">v</th>'
        + '<th style="width:9em;">Last updated</th></tr>';
      for (const c of chapters) {
        const p = c.completion_pct || 0;
        const col = p >= 80 ? 'var(--success)' : p >= 30 ? 'var(--warn)' : 'var(--fg-muted)';
        const bar = '<div style="display:flex;height:0.85em;border-radius:3px;overflow:hidden;background:var(--border);">'
          + '<div style="background:' + col + ';width:' + p.toFixed(0) + '%;"></div>'
          + '</div>';
        const updated = c.last_updated_iso
          ? _escHTML(c.last_updated_iso.slice(5, 16).replace('T', ' '))
          : '<span class="u-muted">—</span>';
        html += '<tr><td>' + c.number + '</td>'
          + '<td>' + _escHTML((c.title || '').slice(0, 60)) + '</td>'
          + '<td>' + bar
          + '<div style="font-size:0.75em;color:var(--fg-muted);margin-top:0.1em;">'
          + p.toFixed(0) + '%</div></td>'
          + '<td style="text-align:right;color:' + col + ';">'
          + _fmtNum(c.words) + ' / ' + _fmtNum(c.target_words) + '</td>'
          + '<td style="text-align:right;" class="u-muted">' + (c.versions || 0) + '</td>'
          + '<td>' + updated + '</td></tr>';
      }
      html += '</table>';
    }
    sections.push(html);
  }

  // Phase 54.6.284 — retraction detail.  Title list behind the
  // existing `retracted_papers` info alert.  Per 54.6.276 policy,
  // these are flagged not excluded — the operator decides per-case.
  if (retractions && (retractions.recent || []).length) {
    const counts = retractions.counts || {};
    const rec = retractions.recent || [];
    const badge = (s, n, col) => '<span style="background:' + col
      + ';color:#fff;padding:0.1em 0.5em;border-radius:3px;font-size:0.85em;margin-right:0.4em;">'
      + _escHTML(s) + ' · ' + n + '</span>';
    let head = '<h4>Retracted / corrected papers</h4>'
      + '<div style="margin-bottom:0.5em;">';
    if (counts.retracted) head += badge('retracted', counts.retracted, '#c33');
    if (counts.corrected) head += badge('corrected', counts.corrected, '#b70');
    for (const k of Object.keys(counts)) {
      if (k === 'retracted' || k === 'corrected' || k === 'none') continue;
      head += badge(k, counts[k], '#888');
    }
    head += '<span class="u-muted" style="font-size:0.85em;">'
      + '(flagged, not auto-excluded — review per-case)</span></div>';
    let table = '<table class="stats-table" style="width:100%;font-size:0.9em;">'
      + '<tr><th style="width:5em;">Status</th><th>Title</th>'
      + '<th style="width:6em;">Year</th><th style="width:8em;">DOI</th></tr>';
    for (const r of rec) {
      const sc = r.status === 'retracted' ? '#c33'
        : r.status === 'corrected' ? '#b70' : '#666';
      const doiLink = r.doi
        ? '<a href="https://doi.org/' + encodeURIComponent(r.doi)
          + '" target="_blank" rel="noopener">' + _escHTML(r.doi.slice(0, 25))
          + '…</a>'
        : '<span class="u-muted">—</span>';
      table += '<tr><td><span style="color:' + sc + ';font-weight:bold;">'
        + _escHTML(r.status) + '</span></td>'
        + '<td>' + _escHTML(r.title || '(no title)') + '</td>'
        + '<td class="u-muted">' + (r.year !== null && r.year !== undefined
          ? _escHTML(String(r.year)) : '—') + '</td>'
        + '<td style="font-size:0.85em;">' + doiLink + '</td></tr>';
    }
    table += '</table>';
    sections.push(head + table);
  }

  // Phase 54.6.282 — chunker section-type coverage. Horizontal
  // stacked bar with per-type colours + legend. `unknown` gets
  // the flag colour (red) so the chunker health signal jumps.
  if (seccov && (seccov.total || 0) > 0) {
    const total = seccov.total;
    const types = seccov.per_type || [];
    const unkPct = seccov.unknown_pct || 0;
    // Canonical palette aligned with _SECTION_PATTERNS.  Keeps the
    // same type → colour mapping between renderings so operators
    // can eyeball "is methods shrinking?" across snapshots.
    const palette = {
      abstract: '#6aa',
      introduction: '#7a5',
      methods: '#38a',
      results: '#a73',
      discussion: '#a58',
      conclusion: '#85a',
      related_work: '#aa5',
      appendix: '#888',
      unknown: '#c33',
    };
    let bar = '<div style="display:flex;height:1em;border-radius:3px;overflow:hidden;">';
    for (const p of types) {
      const c = palette[p.type] || '#666';
      bar += '<div style="background:' + c + ';width:' + p.pct + '%;" '
        + 'title="' + _escHTML(p.type) + ': ' + _fmtNum(p.n)
        + ' (' + p.pct.toFixed(1) + '%)"></div>';
    }
    bar += '</div>';
    let legend = '<div style="font-size:0.85em;color:var(--fg-muted);margin-top:0.4em;display:flex;flex-wrap:wrap;gap:0.75em;">';
    for (const p of types) {
      const c = palette[p.type] || '#666';
      legend += '<span><span style="display:inline-block;width:0.7em;height:0.7em;background:'
        + c + ';margin-right:0.25em;vertical-align:middle;"></span>'
        + _escHTML(p.type) + ' <strong>' + _fmtNum(p.n) + '</strong> ('
        + p.pct.toFixed(1) + '%)</span>';
    }
    legend += '</div>';
    const unkColour = unkPct < 40 ? '#080'
      : unkPct < 70 ? '#b70' : '#c33';
    const headline = '<div style="font-size:0.9em;margin-bottom:0.4em;">'
      + '<strong>Chunk types</strong>: ' + _fmtNum(total)
      + ' total · <span style="color:' + unkColour + ';">unknown '
      + unkPct.toFixed(1) + '%</span>'
      + (unkPct >= 70 ? ' <span class="u-muted">(chunker may be losing heading structure — check `converter_backend` mix)</span>' : '')
      + '</div>';
    // Phase 54.6.287 — per-backend rows appear below the aggregate
    // bar when more than one backend has contributed chunks.  Lets
    // the operator eyeball "is VLM-Pro actually better at heading
    // detection than pipeline?" without running SQL.
    let perBackend = '';
    if (seccovByBackend && seccovByBackend.length >= 2) {
      const bPalette = {
        abstract: '#6aa', introduction: '#7a5', methods: '#38a',
        results: '#a73', discussion: '#a58', conclusion: '#85a',
        related_work: '#aa5', appendix: '#888', unknown: '#c33',
      };
      let html = '<div style="margin-top:0.75em;">'
        + '<div style="font-size:0.9em;margin-bottom:0.4em;"><strong>By converter backend</strong></div>'
        + '<table class="stats-table" style="width:100%;font-size:0.85em;">'
        + '<tr><th style="text-align:left;">Backend</th>'
        + '<th style="text-align:right;">Chunks</th>'
        + '<th style="text-align:right;">Unknown</th>'
        + '<th>Type distribution</th></tr>';
      for (const b of seccovByBackend) {
        const u = b.unknown_pct || 0;
        const uCol = u < 40 ? '#080' : u < 70 ? '#b70' : '#c33';
        let bbar = '<div style="display:flex;height:0.9em;border-radius:3px;overflow:hidden;">';
        for (const t of (b.per_type || [])) {
          const c = bPalette[t.type] || '#666';
          bbar += '<div style="background:' + c + ';width:' + t.pct + '%;" '
            + 'title="' + _escHTML(t.type) + ': ' + _fmtNum(t.n)
            + ' (' + t.pct.toFixed(1) + '%)"></div>';
        }
        bbar += '</div>';
        html += '<tr><td><code>' + _escHTML(b.backend) + '</code></td>'
          + '<td style="text-align:right;">' + _fmtNum(b.total) + '</td>'
          + '<td style="text-align:right;color:' + uCol + ';">' + u.toFixed(1) + '%</td>'
          + '<td>' + bbar + '</td></tr>';
      }
      html += '</table></div>';
      perBackend = html;
    }
    sections.push('<h4>Section coverage</h4>'
      + headline + bar + legend + perBackend);
  }

  // Phase 54.6.293 — cached sidecar integrity banner.  Shows the
  // 807/807 ✓ vs any drift count, plus cache age.  Silent when
  // dual-embedder isn't configured.
  if (sidecarAudit && sidecarAudit.enabled && !sidecarAudit.error) {
    const n = sidecarAudit.n_docs || 0;
    const healthy = sidecarAudit.healthy || 0;
    const critical =
      (sidecarAudit.sidecar_missing || 0)
      + (sidecarAudit.sidecar_partial || 0)
      + (sidecarAudit.sidecar_orphan || 0)
      + (sidecarAudit.prod_missing || 0)
      + (sidecarAudit.prod_partial || 0);
    const age = sidecarAudit.age_s || 0;
    const ageStr = age < 60 ? age.toFixed(0) + ' s'
      : (age / 60).toFixed(1) + ' min';
    const col = critical === 0 ? '#080' : '#c33';
    const icon = critical === 0 ? '✓' : '✗';
    let html = '<h4>Sidecar integrity</h4>'
      + '<div style="display:flex;gap:2em;flex-wrap:wrap;padding:0.5em 0.75em;'
      + 'background:var(--bg-alt, #f5f5f5);border-radius:4px;margin-bottom:0.5em;">'
      + '<div><strong>Status</strong> '
      + '<span style="color:' + col + ';font-size:1.1em;font-weight:bold;">'
      + icon + ' ' + _fmtNum(healthy) + '/' + _fmtNum(n) + '</span></div>'
      + '<div><strong>Missing</strong>: ' + (sidecarAudit.sidecar_missing || 0) + '</div>'
      + '<div><strong>Partial</strong>: ' + (sidecarAudit.sidecar_partial || 0) + '</div>'
      + '<div><strong>Orphan</strong>: ' + (sidecarAudit.sidecar_orphan || 0) + '</div>'
      + '<div><strong>DB / prod / sidecar</strong>: '
      + _fmtNum(sidecarAudit.db_chunks || 0) + ' / '
      + _fmtNum(sidecarAudit.prod_total || 0) + ' / '
      + _fmtNum(sidecarAudit.sidecar_total || 0) + '</div>'
      + '<div class="u-muted" style="margin-left:auto;">cached · '
      + _escHTML(ageStr) + ' old · '
      + '<code>sciknow library audit-sidecar</code></div>'
      + '</div>';
    sections.push(html);
  }

  // Phase 54.6.280 — citation graph connectivity. Three compact
  // stat cards (coverage %, papers with extracted refs, orphan
  // count) plus a top-5 most-cited leaderboard. The data source is
  // `core.monitor._citation_graph`, shared with the CLI's "corpus"
  // panel — this view just has room to list the top-cited papers,
  // which the narrow CLI column doesn't.
  if (cgraph && (cgraph.total_refs || 0) > 0) {
    const totalRefs = cgraph.total_refs || 0;
    const internal = cgraph.internal_refs || 0;
    const coverage = cgraph.coverage_pct || 0;
    const covColour = coverage >= 30 ? '#080'
      : coverage >= 10 ? '#b70' : '#c33';
    const totalDocs = corpus.documents_complete || 0;
    const citing = cgraph.citing_docs || 0;
    const exPct = totalDocs ? Math.round(citing / totalDocs * 100) : 0;
    const exColour = exPct >= 80 ? '#080'
      : exPct >= 40 ? '#b70' : '#c33';
    const orphans = cgraph.orphans || 0;
    const orphPct = totalDocs ? Math.round(orphans / totalDocs * 100) : 0;
    const orphColour = orphPct < 30 ? '#080'
      : orphPct < 70 ? '#b70' : '#c33';
    let html = '<h4>Citation graph</h4>'
      + '<div style="display:flex;gap:2em;flex-wrap:wrap;padding:0.5em 0.75em;'
      + 'background:var(--bg-alt, #f5f5f5);border-radius:4px;margin-bottom:0.5em;">'
      + '<div title="Citations whose target paper is also in the corpus — grow with `sciknow corpus expand`">'
      + '<strong>In-corpus refs</strong><br>'
      + '<span style="color:' + covColour + ';font-size:1.1em;">'
      + _fmtNum(internal) + ' / ' + _fmtNum(totalRefs)
      + ' (' + coverage.toFixed(1) + '%)</span></div>'
      + '<div title="Complete docs whose reference section was successfully parsed">'
      + '<strong>Refs extracted</strong><br>'
      + '<span style="color:' + exColour + ';font-size:1.1em;">'
      + _fmtNum(citing) + ' / ' + _fmtNum(totalDocs)
      + ' docs (' + exPct + '%)</span></div>'
      + '<div title="Complete docs with zero incoming citations from other corpus papers">'
      + '<strong>Orphans</strong><br>'
      + '<span style="color:' + orphColour + ';font-size:1.1em;">'
      + _fmtNum(orphans) + ' (' + orphPct + '%)</span></div>'
      + '<div title="Average outgoing references per citing document (extracted papers only)">'
      + '<strong>Avg out-degree</strong><br>'
      + '<span style="font-size:1.1em;">'
      + (cgraph.avg_out_degree || 0).toFixed(1) + ' refs/doc</span></div>'
      + '</div>';
    const topCited = cgraph.top_cited || [];
    if (topCited.length) {
      html += '<table class="stats-table" style="width:100%;font-size:0.9em;">'
        + '<tr><th style="width:3em;">#</th><th>Most-cited paper</th>'
        + '<th style="width:5em;">Year</th><th style="width:5em;">Cites</th></tr>';
      topCited.forEach((p, i) => {
        html += '<tr><td class="u-muted">' + (i + 1) + '</td>'
          + '<td>' + _escHTML(p.title || '(no title)') + '</td>'
          + '<td class="u-muted">' + (p.year !== null && p.year !== undefined ? _escHTML(String(p.year)) : '—') + '</td>'
          + '<td><strong>' + _fmtNum(p.n) + '</strong></td></tr>';
      });
      html += '</table>';
    }
    sections.push(html);
  }

  // Phase 54.6.298 — per-field enrichment coverage.  Stacked bar
  // per field + headline; a "Run enrich" suggestion when any
  // actionable field is above the alert threshold.
  if (enrichment && (enrichment.total || 0) > 0) {
    const total = enrichment.total;
    const fields = enrichment.fields || [];
    const worstPct = enrichment.worst_pct || 0;
    const worstField = enrichment.worst_field;
    let html = '<h4>Metadata enrichment coverage</h4>'
      + '<table class="stats-table" style="width:100%;font-size:0.9em;">'
      + '<tr><th style="width:6em;">Field</th><th>Coverage</th>'
      + '<th style="width:6em;text-align:right;">Present</th>'
      + '<th style="width:6em;text-align:right;">Missing</th></tr>';
    for (const f of fields) {
      const pctPresent = 100 - f.pct_missing;
      const col = pctPresent >= 80 ? 'var(--success)'
        : pctPresent >= 50 ? 'var(--warn)' : 'var(--danger)';
      const bar = '<div style="display:flex;height:0.9em;border-radius:3px;overflow:hidden;background:var(--border);">'
        + '<div style="background:' + col + ';width:' + pctPresent.toFixed(1) + '%;"></div>'
        + '</div>';
      html += '<tr><td><code>' + _escHTML(f.name) + '</code></td>'
        + '<td>' + bar + '</td>'
        + '<td style="text-align:right;color:' + col + ';">'
        + pctPresent.toFixed(0) + '%</td>'
        + '<td style="text-align:right;color:var(--fg-muted);">'
        + _fmtNum(f.missing) + '</td></tr>';
    }
    html += '</table>';
    if (worstField && worstPct >= 50) {
      html += '<div class="u-muted" style="margin-top:0.4em;">'
        + '<strong>' + worstPct.toFixed(0) + '%</strong> of docs missing '
        + '<code>' + _escHTML(worstField) + '</code> — run '
        + '<code>sciknow corpus enrich</code> to fill from Crossref / OpenAlex / arXiv.'
        + '</div>';
    }
    sections.push(html);
  }

  // Phase 54.6.251 — metadata-source breakdown. Source is where the
  // paper's title/authors/year came from: "crossref" (ideal),
  // "embedded_pdf" (OK), "unknown" (dashboard asks user to run
  // `sciknow db enrich`). Rendered as a horizontal stacked bar so
  // ratio is legible without a legend per slice.
  if (metaQ.sources && metaQ.sources.length) {
    const total = metaQ.sources.reduce((a, s) => a + (s.n || 0), 0);
    const colour = {
      crossref: '#080',
      openalex: '#080',
      semantic_scholar: '#0a8',
      arxiv: '#28a',
      embedded_pdf: '#88a',
      llm_extracted: '#e80',
      unknown: '#c33',
    };
    const segs = metaQ.sources.map(s => {
      const pct = total ? (s.n / total * 100) : 0;
      const c = colour[s.source] || '#888';
      return '<div title="' + _escHTML(s.source) + ': ' + _fmtNum(s.n)
        + ' (' + pct.toFixed(0) + '%)" '
        + 'style="background:' + c + ';width:' + pct + '%;"></div>';
    }).join('');
    let html = '<h4>Metadata sources</h4>'
      + '<div style="display:flex;height:1em;border-radius:3px;overflow:hidden;margin-bottom:0.4em;">'
      + segs + '</div>'
      + '<div style="font-size:0.85em;color:var(--fg-muted);">';
    for (const s of metaQ.sources) {
      const pct = total ? (s.n / total * 100).toFixed(0) : 0;
      const c = colour[s.source] || '#888';
      html += '<span style="margin-right:1em;"><span style="display:inline-block;width:0.7em;height:0.7em;background:'
        + c + ';margin-right:0.2em;"></span>' + _escHTML(s.source)
        + ' <strong>' + _fmtNum(s.n) + '</strong> (' + pct + '%)</span>';
    }
    html += '</div>';
    // Citations cross-linked headline is on the same snapshot field
    if (metaQ.citations_total) {
      const xl = metaQ.citations_crosslinked || 0;
      const xlPct = metaQ.citations_crosslinked_pct || 0;
      html += '<div style="margin-top:0.4em;font-size:0.9em;">'
        + '<strong>Citations resolved</strong>: ' + _fmtNum(xl) + ' / '
        + _fmtNum(metaQ.citations_total) + ' (' + xlPct.toFixed(1) + '%)</div>';
    }
    sections.push(html);
  }

  // Phase 54.6.251 — year histogram sparkline. Using the same shared
  // _spark() helper as throughput and corpus growth. Trims trailing
  // zeros so a sparse corpus doesn't waste half the bar on empty.
  if (yearHist.length) {
    // year_histogram is a list of year/count pairs; extract counts
    const counts = yearHist.map(r => r.count || 0);
    // Trim trailing zeros (future-proof: histogram may extend past
    // "now year" when DB has malformed rows)
    while (counts.length && counts[counts.length - 1] === 0) counts.pop();
    const total = counts.reduce((a, b) => a + b, 0);
    if (total > 0) {
      const startYear = yearHist[0] && yearHist[0].year;
      const endYear = yearHist[counts.length - 1] && yearHist[counts.length - 1].year;
      sections.push('<h4>Corpus year distribution</h4>'
        + '<div style="font-family:monospace;font-size:1.3em;letter-spacing:1px;">'
        + _spark(counts) + '</div>'
        + '<div style="color:var(--fg-muted);font-size:0.85em;">'
        + _escHTML(String(startYear)) + ' → ' + _escHTML(String(endYear))
        + ' · ' + _fmtNum(total) + ' dated papers</div>');
    }
  }

  // Phase 54.6.251 — embeddings + visuals coverage pills. Both are
  // already-collected percentages; the dashboard calls out drift when
  // anything slips below 100 %. embeddings_coverage is the critical
  // one — retrieval is broken if chunks exist in PG without vectors.
  const coverageBits = [];
  if (embedCov.total) {
    const pct = embedCov.pct || 0;
    const col = pct >= 99 ? '#080' : pct >= 90 ? '#e80' : '#c33';
    coverageBits.push('<div><strong>Embeddings</strong>: '
      + '<span style="color:' + col + ';">' + pct.toFixed(1) + '%</span>'
      + ' (' + _fmtNum(embedCov.embedded) + '/' + _fmtNum(embedCov.total)
      + ', missing ' + _fmtNum(embedCov.missing) + ')</div>');
  }
  const vcovPct = (num, tot) => tot ? ((num / tot) * 100).toFixed(0) + '%' : '—';
  if (vcov.figures_total) {
    coverageBits.push('<div><strong>Figures captioned</strong>: '
      + _fmtNum(vcov.figures_captioned) + '/' + _fmtNum(vcov.figures_total)
      + ' (' + vcovPct(vcov.figures_captioned, vcov.figures_total) + ')</div>');
  }
  if (vcov.charts_total) {
    coverageBits.push('<div><strong>Charts captioned</strong>: '
      + _fmtNum(vcov.charts_captioned) + '/' + _fmtNum(vcov.charts_total)
      + ' (' + vcovPct(vcov.charts_captioned, vcov.charts_total) + ')</div>');
  }
  if (vcov.equations_total) {
    coverageBits.push('<div><strong>Equations paraphrased</strong>: '
      + _fmtNum(vcov.equations_paraphrased) + '/' + _fmtNum(vcov.equations_total)
      + ' (' + vcovPct(vcov.equations_paraphrased, vcov.equations_total) + ')</div>');
  }
  if (coverageBits.length) {
    sections.push('<h4>Coverage</h4>'
      + '<div style="display:flex;gap:2em;flex-wrap:wrap;">'
      + coverageBits.join('') + '</div>');
  }

  if (costTotals && costTotals.calls) {
    const dSec = costTotals.seconds || 0;
    const dHrs = dSec >= 3600 ? (dSec / 3600).toFixed(1) + 'h'
      : dSec >= 60 ? (dSec / 60).toFixed(1) + 'm' : Math.round(dSec) + 's';
    sections.push('<h4>LLM cost (last ' + (costTotals.window_days || 30) + 'd)</h4>'
      + '<div style="display:flex;gap:2em;flex-wrap:wrap;">'
      + '<div><strong>Tokens</strong>: ' + _fmtNum(costTotals.tokens || 0) + '</div>'
      + '<div><strong>LLM wall-time</strong>: ' + _escHTML(dHrs) + '</div>'
      + '<div><strong>Calls</strong>: ' + _fmtNum(costTotals.calls || 0) + '</div>'
      + '<div><strong>Models used</strong>: ' + _fmtNum(costTotals.models || 0) + '</div>'
      + '</div>');
  }

  // GPU
  if (gpus.length) {
    // Phase 54.6.248 — GPU row gains temp + util trend sparklines
    // sourced from snap.gpu_trend (populated by the core monitor
    // on every collect_monitor_snapshot call, rolling window up to
    // N samples per worker). Both series share the palette so hot
    // and overloaded GPUs go red.
    const gpuTrend = snap.gpu_trend || {};
    const tempSamples = gpuTrend.temp_samples || [];
    const utilSamples = gpuTrend.util_samples || [];
    const sparkCol = (vals) => (
      (vals && vals.length)
        ? '<span style="font-family:monospace;letter-spacing:1px;">'
          + _spark(vals) + '</span>'
        : '<span class="u-muted">—</span>'
    );
    let html = '<h4>GPU</h4><table class="stats-table" style="width:100%;">'
      + '<tr><th>#</th><th>Name</th><th>VRAM</th><th>Free</th>'
      + '<th>Util</th><th>Temp</th>'
      + '<th>Util trend</th><th>Temp trend</th></tr>';
    for (const g of gpus) {
      const vpct = g.memory_total_mb ? (g.memory_used_mb / g.memory_total_mb * 100).toFixed(0) : '0';
      // Phase 54.6.286 — headroom chip next to VRAM so operators
      // spot the dual-embedder + VLM OOM risk before it fires.
      const headroom = (g.headroom_pct !== undefined && g.headroom_pct !== null)
        ? g.headroom_pct : null;
      const hColour = headroom === null ? 'var(--fg-muted)'
        : headroom < 5 ? '#c33'
        : headroom < 15 ? '#b70'
        : '#080';
      const freeCell = headroom === null ? '—'
        : '<span style="color:' + hColour + ';font-weight:bold;" '
          + 'title="<15% = vram_low alert · <5% = vram_critical · '
          + 'dual-embedder + VLM stack can OOM a 24GB 3090">'
          + (g.memory_free_mb !== undefined ? _fmtNum(g.memory_free_mb) + ' MB ' : '')
          + '(' + headroom.toFixed(1) + '%)</span>';
      html += '<tr><td>' + g.index + '</td><td>' + _escHTML(g.name)
        + '</td><td>' + _fmtNum(g.memory_used_mb) + ' / ' + _fmtNum(g.memory_total_mb) + ' MB (' + vpct + '%)</td>'
        + '<td>' + freeCell + '</td>'
        + '<td>' + g.utilization_pct + '%</td><td>' + (g.temperature_c || '?') + '°C</td>'
        + '<td>' + sparkCol(utilSamples) + '</td>'
        + '<td>' + sparkCol(tempSamples) + '</td>'
        + '</tr>';
    }
    html += '</table>';
    if (gpuTrend.sample_count) {
      html += '<div class="u-muted" style="font-size:0.85em;">Trend: '
        + gpuTrend.sample_count + ' sample(s) in the worker rolling buffer — populates over time as you keep the modal open.</div>';
    }
    sections.push(html);
  }

  // v2 Phase A — llama-server substrate panel. List of writer /
  // embedder / reranker roles with port + pid + model path + health.
  // On v1 fallback installs `infer_substrate` is `[]` and we fall
  // through to the legacy Ollama-loaded-models panel below.
  const inferSub = snap.infer_substrate || [];
  if (inferSub.length) {
    let html = '<h4>LLM substrate — llama-server roles</h4>'
      + '<table class="stats-table" style="width:100%;">'
      + '<tr><th>Role</th><th>Port</th><th>PID</th><th>Health</th><th>Model</th></tr>';
    for (const r of inferSub) {
      const dot = r.healthy
        ? '<span style="color:#080;">●</span>'
        : '<span style="color:#c33;">✗</span>';
      html += '<tr><td>' + _escHTML(r.role) + '</td>'
        + '<td>' + _escHTML(String(r.port)) + '</td>'
        + '<td>' + _escHTML(String(r.pid || '—')) + '</td>'
        + '<td>' + dot + (r.healthy ? ' healthy' : ' down') + '</td>'
        + '<td><code style="font-size:0.85em;">' + _escHTML(r.model) + '</code></td></tr>';
    }
    html += '</table>';
    sections.push(html);
  } else if (loaded.length) {
    let html = '<h4>Ollama — loaded models</h4><table class="stats-table" style="width:100%;">'
      + '<tr><th>Model</th><th>VRAM</th><th>Expires</th></tr>';
    for (const m of loaded) {
      html += '<tr><td>' + _escHTML(m.name) + '</td><td>' + _fmtNum(m.vram_mb) + ' MB</td><td>'
        + _escHTML(m.expires_at || '—') + '</td></tr>';
    }
    html += '</table>';
    sections.push(html);
  } else {
    sections.push('<p class="u-note">No LLM models resident — start one with <code>sciknow infer up --role writer</code>.</p>');
  }

  // Phase 54.6.301 — retrieval latency panel.  Session ring buffer
  // populated by hybrid_search.search() in-process; empty when the
  // reader hasn't run retrieval yet.  Per-leg p50 bars let the
  // operator see which stage dominates (usually dense on a large
  // corpus; FTS on heavy BM25 queries).
  if (retrievalLat && (retrievalLat.count || 0) > 0) {
    const p50 = retrievalLat.p50_ms || 0;
    const p95 = retrievalLat.p95_ms || 0;
    const avg = retrievalLat.avg_ms || 0;
    const n = retrievalLat.count;
    const legs = retrievalLat.per_leg_p50 || {};
    const p95Col = p95 < 800 ? '#080' : p95 < 2000 ? '#b70' : '#c33';
    const banner = '<div style="display:flex;gap:1.5em;flex-wrap:wrap;padding:0.5em 0.75em;'
      + 'background:var(--bg-alt, #f5f5f5);border-radius:4px;margin-bottom:0.5em;">'
      + '<div><strong>n</strong>: ' + n + '</div>'
      + '<div><strong>p50</strong>: ' + p50 + ' ms</div>'
      + '<div><strong>p95</strong>: <span style="color:' + p95Col + ';font-weight:bold;">'
      + p95 + ' ms</span></div>'
      + '<div><strong>avg</strong>: ' + avg + ' ms</div>'
      + '</div>';
    // Per-leg medians as a stacked breakdown — percentages of p50.
    let legRows = '';
    if (Object.keys(legs).length) {
      const order = ['embed', 'dense', 'sparse', 'fts', 'fuse'];
      const legPalette = {
        embed: '#6aa', dense: '#38a', sparse: '#7a5',
        fts: '#a58', fuse: '#888',
      };
      let bar = '<div style="display:flex;height:0.9em;border-radius:3px;overflow:hidden;">';
      const total = order.reduce((s, k) => s + (legs[k] || 0), 0) || 1;
      for (const k of order) {
        const v = legs[k] || 0;
        const pct = (v / total) * 100;
        const col = legPalette[k] || '#666';
        bar += '<div style="background:' + col + ';width:' + pct + '%;" '
          + 'title="' + k + ' p50: ' + v + ' ms (' + pct.toFixed(0) + '%)"></div>';
      }
      bar += '</div>';
      const legend = order.map(k => {
        const v = legs[k] || 0;
        const col = legPalette[k] || '#666';
        return '<span style="margin-right:0.8em;font-size:0.85em;">'
          + '<span style="display:inline-block;width:0.65em;height:0.65em;background:'
          + col + ';margin-right:0.2em;vertical-align:middle;"></span>'
          + k + ' ' + v + ' ms</span>';
      }).join('');
      legRows = '<div style="margin-bottom:0.5em;">'
        + '<div style="font-size:0.9em;margin-bottom:0.3em;"><strong>Per-leg p50 breakdown</strong></div>'
        + bar
        + '<div style="color:var(--fg-muted);margin-top:0.2em;">' + legend + '</div>'
        + '</div>';
    }
    // Per-event table
    const events = retrievalLat.events || [];
    let eventsTable = '';
    if (events.length) {
      eventsTable = '<table class="stats-table" style="width:100%;font-size:0.85em;">'
        + '<tr><th>When</th><th style="text-align:right;">Total</th>'
        + '<th style="text-align:right;">Embed</th>'
        + '<th style="text-align:right;">Dense</th>'
        + '<th style="text-align:right;">Sparse</th>'
        + '<th style="text-align:right;">FTS</th>'
        + '<th style="text-align:right;">Fuse</th>'
        + '<th style="text-align:right;">Cands</th>'
        + '<th>Filtered</th></tr>';
      for (const e of events.slice().reverse()) {
        const dt = new Date((e.t || 0) * 1000).toLocaleTimeString();
        eventsTable += '<tr><td>' + _escHTML(dt) + '</td>'
          + '<td style="text-align:right;font-weight:bold;">' + (e.total_ms || 0) + ' ms</td>'
          + '<td style="text-align:right;" class="u-muted">' + (e.embed_ms || 0) + '</td>'
          + '<td style="text-align:right;" class="u-muted">' + (e.dense_ms || 0) + '</td>'
          + '<td style="text-align:right;" class="u-muted">' + (e.sparse_ms || 0) + '</td>'
          + '<td style="text-align:right;" class="u-muted">' + (e.fts_ms || 0) + '</td>'
          + '<td style="text-align:right;" class="u-muted">' + (e.fuse_ms || 0) + '</td>'
          + '<td style="text-align:right;">' + (e.candidates || 0) + '</td>'
          + '<td class="u-muted">' + (e.filtered ? 'yes' : '—') + '</td></tr>';
      }
      eventsTable += '</table>';
    }
    sections.push('<h4>Retrieval latency</h4>' + banner + legRows + eventsTable);
  }

  // Phase 54.6.291 — VRAM preflight history panel.  Shows the
  // ring-buffer of preflight events from vram_budget so the
  // operator can verify the dual-embedder + MinerU-VLM preflight
  // is firing and actually reclaiming VRAM.  Silent when the
  // buffer is empty (fresh process / no ingest yet).
  const preflight = snap.vram_preflight || {};
  if (preflight.count) {
    const tight = preflight.tight_count || 0;
    const failed = preflight.failed_count || 0;
    const freedGb = (preflight.total_freed_mb || 0) / 1024;
    const banner = '<div style="display:flex;gap:1.5em;flex-wrap:wrap;padding:0.5em 0.75em;'
      + 'background:var(--bg-alt, #f5f5f5);border-radius:4px;margin-bottom:0.5em;">'
      + '<div><strong>Events</strong>: ' + preflight.count + '</div>'
      + '<div><strong>Tight</strong>: <span style="color:'
      + (tight === 0 ? 'var(--fg-muted)'
         : tight < preflight.count / 2 ? '#b70' : '#c33')
      + ';">' + tight + '</span></div>'
      + '<div><strong>Failed</strong>: <span style="color:'
      + (failed ? '#c33' : '#080') + ';">' + failed + '</span></div>'
      + '<div><strong>Freed</strong>: ' + freedGb.toFixed(2) + ' GB</div>'
      + '</div>';
    let html = '<h4>VRAM preflight</h4>' + banner;
    const events = preflight.events || [];
    if (events.length) {
      html += '<table class="stats-table" style="width:100%;font-size:0.85em;">'
        + '<tr><th>When</th><th>Reason</th><th>Need</th>'
        + '<th>Before → After</th><th>Releasers</th>'
        + '<th>Budget met</th></tr>';
      for (const ev of events.slice().reverse()) {
        const dt = new Date((ev.t || 0) * 1000).toLocaleTimeString();
        const need = (ev.need_mb || 0).toLocaleString() + ' MB';
        const ba = (ev.started_free_mb || 0).toLocaleString()
          + ' → ' + (ev.ended_free_mb || 0).toLocaleString() + ' MB';
        const fired = (ev.fired || []).length
          ? (ev.fired || []).map(x => '<code>' + _escHTML(x) + '</code>').join(' ')
          : '<span class="u-muted">—</span>';
        const met = ev.met_budget
          ? '<span style="color:#080;">✓</span>'
          : '<span style="color:#c33;">✗</span>';
        html += '<tr><td>' + _escHTML(dt) + '</td>'
          + '<td>' + _escHTML(ev.reason || '') + '</td>'
          + '<td>' + _escHTML(need) + '</td>'
          + '<td>' + _escHTML(ba) + '</td>'
          + '<td>' + fired + '</td>'
          + '<td>' + met + '</td></tr>';
      }
      html += '</table>';
    }
    sections.push(html);
  }

  // Phase 54.6.289 — model-swap churn panel.  Lists the last few
  // observed swaps so the operator can see *which* roles are
  // thrashing.  Silent when the session buffer is empty.
  const swapTrend = ((snap.llm || {}).swap_trend) || {};
  if (swapTrend.swap_count) {
    const rate = swapTrend.swaps_per_hour || 0;
    const rateCol = rate >= 15 ? '#c33'
      : rate >= 5 ? '#b70' : 'var(--fg-muted)';
    let html = '<h4>Model swap churn</h4>'
      + '<div style="margin-bottom:0.5em;">'
      + '<strong style="color:' + rateCol + ';">' + rate.toFixed(1)
      + ' swaps/hour</strong>'
      + ' <span class="u-muted">· ' + swapTrend.swap_count
      + ' events in the last ' + Math.round((swapTrend.window_s || 0) / 60)
      + 'm (session buffer · lost on web restart)</span></div>';
    const events = swapTrend.events || [];
    if (events.length) {
      html += '<table class="stats-table" style="width:100%;font-size:0.9em;">'
        + '<tr><th>When</th><th>Added</th><th>Removed</th><th>Resident after</th></tr>';
      for (const ev of events.slice().reverse()) {
        const dt = new Date((ev.t || 0) * 1000).toLocaleTimeString();
        const addedCell = (ev.added || [])
          .map(x => '<code style="color:#080;">+' + _escHTML(x) + '</code>').join(' ');
        const removedCell = (ev.removed || [])
          .map(x => '<code style="color:#c33;">-' + _escHTML(x) + '</code>').join(' ');
        const residentCell = (ev.loaded || [])
          .map(x => '<code class="u-muted">' + _escHTML(x) + '</code>').join(' ') || '<span class="u-muted">∅</span>';
        html += '<tr><td>' + _escHTML(dt) + '</td>'
          + '<td>' + (addedCell || '<span class="u-muted">—</span>') + '</td>'
          + '<td>' + (removedCell || '<span class="u-muted">—</span>') + '</td>'
          + '<td>' + residentCell + '</td></tr>';
      }
      html += '</table>';
    }
    sections.push(html);
  }

  // Phase 54.6.246 — active web jobs panel. Inside the web process
  // this is the authoritative in-memory list (same data /api/monitor
  // returns); inside the CLI it comes from the cross-process pulse
  // file and may carry is_stale=true when the web crashed without
  // clearing it. Table hidden when no jobs — don't occupy space on
  // an idle system.
  const activeJobs = snap.active_jobs || [];
  if (activeJobs.length) {
    // Phase 54.6.247 — added TPS column. Coloured stop-light: red
    // when elapsed > 5s but TPS == 0 (stalled), green when >= 5 t/s,
    // muted otherwise. Same logic as the CLI header.
    let html = '<h4>Active jobs</h4>'
      + '<table class="stats-table" style="width:100%;">'
      + '<tr><th>Job</th><th>Type</th><th>Model</th><th>Tokens</th><th>TPS</th><th>Elapsed</th><th></th></tr>';
    for (const j of activeJobs) {
      const el = j.elapsed_s || 0;
      const elStr = el >= 60 ? (el / 60).toFixed(1) + 'm' : Math.round(el) + 's';
      const twStr = j.target_words ? ' / ' + j.target_words + 'w' : '';
      const stale = j.is_stale
        ? ' <span style="color:#c33;font-weight:bold;">STALE</span>'
        : '';
      const tps = j.tps || 0;
      let tpsColour = 'inherit';
      if (el > 5 && tps === 0) tpsColour = '#c33';
      else if (tps >= 5) tpsColour = '#080';
      else tpsColour = '#888';
      html += '<tr>'
        + '<td><code>' + _escHTML((j.id || '?').slice(0, 8)) + '</code>' + stale + '</td>'
        + '<td>' + _escHTML(j.type || '?') + '</td>'
        + '<td><code>' + _escHTML(j.model || '—') + '</code></td>'
        + '<td>' + _fmtNum(j.tokens || 0) + twStr + '</td>'
        + '<td style="color:' + tpsColour + ';"><strong>' + tps.toFixed(1) + '</strong> t/s</td>'
        + '<td>' + _escHTML(elStr) + '</td>'
        + '<td><code class="u-muted">' + _escHTML(j.stream_state || '') + '</code></td>'
        + '</tr>';
    }
    html += '</table>';
    sections.push(html);
  }

  // Phase 54.6.244 — model assignments per role. Mirrors the CLI
  // "gpu · models" panel — the user can see the whole LLM wiring at
  // a glance without opening Book Settings → Models tab.
  // v2 Phase A — when v2_writer_active is true, the writer GGUF
  // handles every writer-class role, so the per-role rows
  // (BOOK_WRITE_MODEL, BOOK_REVIEW_MODEL, AUTOWRITE_SCORER_MODEL,
  // LLM_FAST_MODEL) are not honored at runtime and we hide them.
  // The v1 fallback path (USE_LLAMACPP_WRITER=False) restores the
  // full per-role table so .env-configured Ollama tags stay
  // auditable.
  const models = snap.model_assignments || {};
  if (models.llm_main) {
    const main = models.llm_main;
    const v2Writer = !!models.v2_writer_active;
    const fmt = (val, inherits) => {
      if (val) return '<code>' + _escHTML(val) + '</code>';
      if (inherits && main) return '<code>' + _escHTML(main)
        + '</code> <span class="u-muted">(↑LLM_MODEL)</span>';
      return '<em class="u-muted">(unset)</em>';
    };
    const v2Rows = [
      ['Writer',          main,                      false],
      ['Caption VLM',     models.caption_vlm,        false],
      ['MinerU VLM',      models.mineru_vlm_model,   false],
      ['Embedder',        models.embedder,           false],
      ['Reranker',        models.reranker,           false],
    ];
    const v1Rows = [
      ['LLM_MODEL',              main,                      false],
      ['LLM_FAST_MODEL',         models.llm_fast,           false],
      ['BOOK_WRITE_MODEL',       models.book_write,         true],
      ['BOOK_REVIEW_MODEL',      models.book_review,        true],
      ['AUTOWRITE_SCORER_MODEL', models.autowrite_scorer,   true],
      ['VISUALS_CAPTION_MODEL',  models.caption_vlm,        false],
      ['MINERU_VLM_MODEL',       models.mineru_vlm_model,   false],
      ['EMBEDDING_MODEL',        models.embedder,           false],
      ['RERANKER_MODEL',         models.reranker,           false],
    ];
    const rows = v2Writer ? v2Rows : v1Rows;
    const heading = v2Writer
      ? 'Model assignments <span class="u-muted">(v2 substrate · single canonical writer)</span>'
      : 'Model assignments <span class="u-muted">(v1 fallback · per-role Ollama tags)</span>';
    let html = '<h4>' + heading + '</h4>'
      + '<table class="stats-table" style="width:100%;">'
      + '<tr><th>Role</th><th>Model</th></tr>';
    for (const [role, val, inherits] of rows) {
      if (val === undefined) continue;
      html += '<tr><td>' + _escHTML(role) + '</td><td>' + fmt(val, inherits) + '</td></tr>';
    }
    html += '</table>';
    sections.push(html);
  }

  // Qdrant collections — Phase 54.6.296 adds an Indexes column
  // listing expected/present/missing per collection so filter-
  // pushdown regressions (a missed create_payload_index) jump out.
  if (qcolls.length) {
    // Index map keyed by collection name for O(1) lookup
    const idxMap = {};
    for (const c of (qdrantIndexes.collections || [])) {
      idxMap[c.name] = c;
    }
    // Phase 54.6.299 — HNSW row map.  Papers-class collections
    // can have multiple vector fields (dense only today, colbert
    // on abstracts when enabled); pick the first papers-class
    // entry per collection for the summary cell.
    const hnswMap = {};
    for (const c of (qdrantHnsw.collections || [])) {
      if (!hnswMap[c.name]) hnswMap[c.name] = c;
    }
    let html = '<h4>Qdrant collections</h4><table class="stats-table" style="width:100%;">'
      + '<tr><th>Collection</th><th>Points</th><th>Vector fields</th>'
      + '<th title="Payload indexes present / expected — missing indexes turn filter pushdown into a full scan">Indexes</th>'
      + '<th title="HNSW tuning — m / ef_construct / quantization. Papers-class collections should match the .env QDRANT_HNSW_* values (54.6.299).">HNSW</th></tr>';
    for (const c of qcolls) {
      const fields = (c.vectors || []).concat((c.sparse_vectors || []).map(s => 'sparse:' + s));
      const idx = idxMap[c.name];
      let idxCell = '<span class="u-muted">—</span>';
      if (idx) {
        const pres = (idx.present || []).length;
        const exp  = (idx.expected || []).length;
        const miss = (idx.missing || []).length;
        const col = miss === 0 ? '#080' : '#c33';
        const title = (idx.missing || []).length
          ? 'missing: ' + idx.missing.join(', ')
          : 'all expected indexes present';
        idxCell = '<span style="color:' + col + ';" title="' + _escHTML(title) + '">'
          + pres + '/' + Math.max(pres, exp)
          + (miss ? ' (✗' + miss + ' missing)' : ' ✓')
          + '</span>';
      }
      const hnsw = hnswMap[c.name];
      let hnswCell = '<span class="u-muted">—</span>';
      if (hnsw) {
        const drift = hnsw.drift;
        const reasons = (hnsw.drift_reasons || []).join(' · ');
        const kind = hnsw.kind;
        const col = drift ? '#c33' : (kind === 'papers' ? '#080' : 'var(--fg-muted)');
        const label = 'm=' + hnsw.m + '/ef=' + hnsw.ef_construct
          + (hnsw.quantization ? ' ·Q' : '');
        const title = drift
          ? 'drift: ' + reasons
          : (kind === 'papers' ? 'tuned (papers-class)' : 'defaults (small collection)');
        const icon = drift ? '✗ ' : (kind === 'papers' ? '✓ ' : '');
        hnswCell = '<span style="color:' + col + ';" title="' + _escHTML(title) + '">'
          + icon + label + '</span>';
      }
      html += '<tr><td>' + _escHTML(c.name) + '</td><td>' + _fmtNum(c.points_count)
        + '</td><td>' + _escHTML(fields.join(', ')) + '</td>'
        + '<td>' + idxCell + '</td>'
        + '<td>' + hnswCell + '</td></tr>';
    }
    html += '</table>';
    sections.push(html);
  }

  // Converter backends
  if (backends.length) {
    let html = '<h4>Converter backends</h4><table class="stats-table" style="width:100%;">'
      + '<tr><th>Backend</th><th>Papers</th></tr>';
    for (const b of backends) {
      html += '<tr><td>' + _escHTML(b.backend) + '</td><td>' + _fmtNum(b.n) + '</td></tr>';
    }
    html += '</table>';
    sections.push(html);
  }

  // Pipeline timing.  Phase 54.6.288 adds a "Δ vs last week" column
  // driven by pipe.stage_timing_deltas — coloured chip per stage.
  if (timing.length) {
    const deltas = {};
    for (const d of (pipe.stage_timing_deltas || [])) {
      deltas[d.stage] = d;
    }
    let html = '<h4>Pipeline stage timing (completed jobs)</h4>'
      + '<table class="stats-table" style="width:100%;">'
      + '<tr><th>Stage</th><th>N</th><th>p50</th><th>p95</th><th>mean</th>'
      + '<th title="p95 change vs the preceding 7-day window — regression ≥+30%, improvement ≤-30%">Δ vs 7d prior</th></tr>';
    for (const row of timing) {
      const d = deltas[row.stage];
      let deltaCell = '<span class="u-muted">—</span>';
      if (d && d.delta_pct !== null && d.delta_pct !== undefined) {
        const dp = d.delta_pct;
        const col = d.severity === 'regression' ? '#c33'
          : d.severity === 'improvement' ? '#080'
          : 'var(--fg-muted)';
        const sign = dp >= 0 ? '+' : '';
        deltaCell = '<span style="color:' + col + ';font-weight:bold;" '
          + 'title="prev p95 ' + _fmtMs(d.p95_prev_ms)
          + ' → current p95 ' + _fmtMs(d.p95_cur_ms)
          + ' · n_prev=' + d.n_prev + ' n_cur=' + d.n_cur + '">'
          + sign + dp.toFixed(0) + '%</span>';
      }
      html += '<tr><td>' + _escHTML(row.stage) + '</td><td>' + _fmtNum(row.n)
        + '</td><td>' + _fmtMs(row.p50_ms) + '</td><td>' + _fmtMs(row.p95_ms)
        + '</td><td>' + _fmtMs(row.mean_ms) + '</td>'
        + '<td>' + deltaCell + '</td></tr>';
    }
    html += '</table>';
    sections.push(html);
  }

  // Phase 54.6.294 — slow-ingest leaderboard removed from the GUI
  // (2026-04-26). Was visually noisy (5 rows per render, ~120 px
  // tall stacked-bar each) and rarely actionable on a stable corpus.
  // The data still ships in `snap.slow_docs` for any downstream
  // JSON consumer; just no longer rendered in the System Monitor
  // modal.

  // Top failure classes — summary of worst offenders in last 24h
  if (topFailures.length) {
    let html = '<h4>Top failure classes (last 24h)</h4>'
      + '<table class="stats-table" style="width:100%;">'
      + '<tr><th>Stage</th><th>Error</th><th>Count</th></tr>';
    for (const tf of topFailures) {
      html += '<tr><td>' + _escHTML(tf.stage) + '</td><td style="color:var(--fg-muted);">'
        + _escHTML((tf.error || '').slice(0, 80)) + '</td><td>' + tf.count + '</td></tr>';
    }
    html += '</table>';
    sections.push(html);
  }

  // Storage panel
  if (Object.keys(disk).length || pgMb) {
    const fmtMb = (mb) => mb >= 1024 ? (mb / 1024).toFixed(1) + ' GB' : mb + ' MB';
    let html = '<h4>Storage</h4><table class="stats-table" style="width:100%;">'
      + '<tr><th>Path</th><th>Size</th></tr>';
    const rows = [
      ['data_dir total', disk.data_dir_mb || 0],
      ['mineru_output', disk.mineru_output_mb || 0],
      ['processed', disk.processed_mb || 0],
      ['downloads', disk.downloads_mb || 0],
      ['failed', disk.failed_mb || 0],
      ['bench', disk.bench_mb || 0],
      ['pg database', pgMb],
    ];
    for (const [label, mb] of rows) {
      if (mb === 0 && label !== 'data_dir total') continue;
      html += '<tr><td>' + _escHTML(label) + '</td><td>' + fmtMb(mb) + '</td></tr>';
    }
    html += '</table>';
    sections.push(html);
  }

  // Failures
  const anyFail = fails.some(f => f.failed > 0);
  if (anyFail) {
    let html = '<h4>Stage failures</h4><table class="stats-table" style="width:100%;">'
      + '<tr><th>Stage</th><th>Failed</th><th>Total</th><th>Rate</th></tr>';
    for (const f of fails) {
      if (!f.failed) continue;
      const rate = (f.failure_rate * 100).toFixed(1);
      html += '<tr><td>' + _escHTML(f.stage) + '</td><td>' + _fmtNum(f.failed)
        + '</td><td>' + _fmtNum(f.total) + '</td><td>' + rate + '%</td></tr>';
    }
    html += '</table>';
    sections.push(html);
  }

  // Phase 54.6.283 — LLM role-usage heatmap.  Days along the x-axis,
  // top-N operations along y.  Cell colour scales with call count
  // (log-ish ramp to keep small counts visible alongside spikes).
  // Silent when no rows in llm_usage_log (fresh install / CLI-only
  // usage that isn't logged).
  const llmByDay = ((snap.llm || {}).usage_by_day) || {};
  if (llmByDay.operations && llmByDay.operations.length) {
    const days = llmByDay.days || [];
    const ops = llmByDay.operations || [];
    const grid = llmByDay.grid || {};
    const maxCalls = llmByDay.max_calls || 1;
    const shortDay = (d) => (d || '').slice(5);  // MM-DD
    // Log-scale intensity so 1 call is visibly distinct from 0
    // while 100-call spikes don't wash everything else out.
    const cellColour = (n) => {
      if (!n) return '';
      const t = Math.log(1 + n) / Math.log(1 + maxCalls);
      // Green ramp 0 → dark green.
      const g = 180 - Math.round(120 * t);
      const a = 0.2 + 0.7 * t;
      return 'background:rgba(0,' + g + ',0,' + a.toFixed(2) + ');';
    };
    let html = '<h4>LLM usage heatmap</h4>'
      + '<table class="stats-table" style="width:100%;font-size:0.85em;">'
      + '<tr><th style="text-align:left;">Operation</th>';
    for (const d of days) {
      html += '<th style="text-align:center;font-weight:normal;">'
        + _escHTML(shortDay(d)) + '</th>';
    }
    html += '<th style="text-align:right;">Total</th></tr>';
    for (const op of ops) {
      const row = grid[op] || {};
      let total = 0;
      html += '<tr><td><code>' + _escHTML(op) + '</code></td>';
      for (const d of days) {
        const n = row[d] || 0;
        total += n;
        const style = cellColour(n);
        const tt = n ? _escHTML(op) + ' · ' + _escHTML(d)
          + ' · ' + n + ' calls' : '';
        html += '<td style="text-align:center;' + style + '" '
          + 'title="' + tt + '">' + (n || '') + '</td>';
      }
      html += '<td style="text-align:right;"><strong>'
        + _fmtNum(total) + '</strong></td></tr>';
    }
    html += '</table>';
    sections.push(html);
  }

  // LLM usage
  if (llmUsage.length) {
    let html = '<h4>LLM usage (last window)</h4><table class="stats-table" style="width:100%;">'
      + '<tr><th>Operation</th><th>Model</th><th>Tokens</th><th>Seconds</th><th>Calls</th></tr>';
    for (const l of llmUsage) {
      html += '<tr><td>' + _escHTML(l.operation) + '</td><td>' + _escHTML(l.model)
        + '</td><td>' + _fmtNum(l.tokens) + '</td><td>' + Math.round(l.seconds || 0)
        + 's</td><td>' + _fmtNum(l.calls) + '</td></tr>';
    }
    html += '</table>';
    sections.push(html);
  }

  // Recent activity
  if (activity.length) {
    let html = '<h4>Recent activity</h4><table class="stats-table" style="width:100%;">'
      + '<tr><th>When</th><th>Stage</th><th>Status</th><th>Duration</th><th>Doc</th></tr>';
    for (const a of activity) {
      const when = (a.created_at || '').split('T')[1] || '';
      html += '<tr><td>' + _escHTML(when.slice(0, 8)) + '</td><td>' + _escHTML(a.stage)
        + '</td><td>' + _escHTML(a.status) + '</td><td>' + _fmtMs(a.duration_ms)
        + '</td><td><code>' + _escHTML((a.doc_id || '').slice(0, 8)) + '</code></td></tr>';
    }
    html += '</table>';
    sections.push(html);
  }

  // Phase 54.6.243 — cross-project overview. Only rendered when
  // more than one project exists (single-project installs don't
  // need the extra table).
  if (projectsOverview.length > 1) {
    let html = '<h4>Projects</h4><table class="stats-table" style="width:100%;">'
      + '<tr><th>Active</th><th>Slug</th><th>Database</th><th>Documents</th></tr>';
    for (const p of projectsOverview) {
      const docs = p.docs < 0 ? '?' : _fmtNum(p.docs);
      const marker = p.is_active ? '● ' : '○ ';
      html += '<tr><td>' + marker + '</td><td>' + _escHTML(p.slug)
        + '</td><td>' + _escHTML(p.pg_database || '') + '</td><td>'
        + docs + '</td></tr>';
    }
    html += '</table>';
    sections.push(html);
  }

  // Phase 54.6.260 — collapsible log tail at the bottom. Uses the
  // native <details>/<summary> pair so no JS state to manage;
  // operators click to expand and see the 20-line tail. ERROR +
  // CRITICAL lines render red, WARNING in yellow. File path shown
  // in the summary so operators can `tail -f` it if they want a
  // live view.
  if ((logTail.lines || []).length) {
    const errCount = logTail.error_lines || 0;
    const filePath = logTail.file_path || '';
    const summaryColour = errCount > 0 ? '#c33' : 'var(--fg-muted)';
    let html = '<details style="margin-top:0.5em;">'
      + '<summary style="cursor:pointer;color:' + summaryColour + ';">'
      + '<strong>Recent log</strong> (' + logTail.lines.length + ' lines'
      + (errCount ? ', <span style="color:#c33;">' + errCount + ' ERROR/CRITICAL</span>' : '')
      + ')'
      + (filePath ? ' <code class="u-muted" style="font-size:0.85em;">' + _escHTML(filePath) + '</code>' : '')
      + '</summary>'
      + '<pre style="font-family:monospace;font-size:0.8em;'
      + 'background:var(--bg-alt,#f5f5f5);padding:0.5em;border-radius:4px;'
      + 'max-height:300px;overflow-y:auto;white-space:pre-wrap;word-break:break-word;">';
    for (const ln of logTail.lines) {
      let c = '#333';
      if (/\bERROR\b|\bCRITICAL\b/.test(ln)) c = '#c33';
      else if (/\bWARNING\b/.test(ln)) c = '#b70';
      html += '<span style="color:' + c + ';">' + _escHTML(ln) + '</span>\n';
    }
    html += '</pre></details>';
    sections.push(html);
  }

  target.innerHTML = sections.join('');
  const ts = document.getElementById('monitor-last-updated');
  if (ts) {
    // Phase 54.6.263 — show snapshot build time. Red at >2s,
    // yellow at >1s, muted otherwise.
    const ms = snap.snapshot_duration_ms;
    let msTxt = '';
    if (typeof ms === 'number') {
      const col = ms > 2000 ? '#c33' : ms > 1000 ? '#e80' : 'var(--fg-muted)';
      msTxt = ' · <span style="color:' + col + ';">built in ' + ms + 'ms</span>';
    }
    ts.innerHTML = 'Updated ' + new Date().toLocaleTimeString() + msTxt;
  }
}

function switchToolsTab(name) {
  // Only flip the TOP-level Tools tabs (not any inner Corpus-subtabs,
  // which carry data-ctab instead of data-tab so they don't collide).
  // Phase 54.6.18 — Corpus extracted into its own modal, so tl-corpus
  // is no longer part of the Tools tab list. Backwards-compat: if a
  // caller passes 'tl-corpus' we redirect to openCorpusModal.
  if (name === 'tl-corpus') {
    openCorpusModal();
    return;
  }
  document.querySelectorAll('#tools-modal > .tabs > .tab').forEach(t => {
    t.classList.toggle('active', t.dataset.tab === name);
  });
  ['tl-search', 'tl-synth', 'tl-topics'].forEach(n => {
    const pane = document.getElementById(n + '-pane');
    if (pane) pane.style.display = (n === name) ? 'block' : 'none';
  });
  if (name === 'tl-topics') loadToolTopics();
}

// Phase 54.6.18 — standalone Corpus modal. Opens the extracted pane
// and optionally switches to a specific sub-tab. Called from the
// top-bar Corpus ▾ dropdown (seven entry points) and from the
// legacy switchToolsTab('tl-corpus') fallback.
function openCorpusModal(subtab) {
  // Close any open nav dropdown so the top bar resets.
  document.querySelectorAll('.nav-dropdown.open').forEach(d => d.classList.remove('open'));
  openModal('corpus-modal');
  // The pane was previously style="display:none;" to hide it inside
  // the Tools modal's tab rotation. Inside its own modal it needs to
  // be visible.
  const pane = document.getElementById('tl-corpus-pane');
  if (pane) pane.style.display = 'block';
  // Default sub-tab on first open.
  switchCorpusTab(subtab || 'corp-enrich');
  loadCorpusTopicList();
}


// Phase 46.E — inner tabs for the Corpus pane (Enrich / Expand-citations /
// Expand-by-Author). Uses data-ctab to avoid colliding with the outer
// Tools tabs' data-tab.
function switchCorpusTab(name) {
  document.querySelectorAll('#tl-corpus-pane .tab').forEach(t => {
    t.classList.toggle('active', t.dataset.ctab === name);
  });
  ['corp-enrich', 'corp-cites', 'corp-agentic', 'corp-author',
   'corp-author-refs', 'corp-inbound', 'corp-topic', 'corp-coauth'].forEach(n => {
    const pane = document.getElementById(n + '-pane');
    if (pane) pane.style.display = (n === name) ? 'block' : 'none';
  });
}

// Phase 54.6.125 (Tier 3 #3) — preprint↔journal reconciliation viewer.
async function openReconciliationsModal() {
  document.querySelectorAll('.nav-dropdown.open').forEach(d => d.classList.remove('open'));
  openModal('reconciliations-modal');
  const list = document.getElementById('recon-list');
  list.textContent = 'Loading…';
  try {
    const res = await fetch('/api/reconciliations');
    const d = await res.json();
    const pairs = d.pairs || [];
    if (!pairs.length) {
      list.innerHTML = '<em class="u-muted">No active reconciliations. Run <code>Detect duplicates</code> then <code>Reconcile preprints</code> in the Corpus modal utility row.</em>';
      return;
    }
    let html = '<table class="u-table-full">'
      + '<tr class="u-border-b"><th class="u-cell-sm">Canonical</th>'
      + '<th class="u-cell-sm">Non-canonical (hidden)</th>'
      + '<th class="u-p-4-6">Action</th></tr>';
    for (const p of pairs) {
      html += '<tr class="u-border-b u-vat">'
        + '<td class="u-pad-md"><strong>' + _escHtml(p.canonical_id.slice(0, 8)) + '</strong>'
          + ' <span class="u-label-xs">'
          + _escHtml(String(p.canonical_year || '')) + ' · ' + _escHtml(p.canonical_journal || '') + '</span>'
          + '<br>' + _escHtml((p.canonical_title || '').substring(0, 100))
          + '<br><span class="u-mono u-xxs u-muted">' + _escHtml(p.canonical_doi || '') + '</span></td>'
        + '<td class="u-pad-md"><strong>' + _escHtml(p.non_canonical_id.slice(0, 8)) + '</strong>'
          + ' <span class="u-label-xs">' + _escHtml(String(p.non_canonical_year || '')) + '</span>'
          + '<br>' + _escHtml((p.non_canonical_title || '').substring(0, 100))
          + '<br><span class="u-mono u-xxs u-muted">' + _escHtml(p.non_canonical_doi || '') + '</span></td>'
        + '<td class="u-pad-md"><button onclick="undoReconciliation(\'' + p.non_canonical_id + '\')" '
          + 'style="background:rgba(80,200,120,0.15);color:var(--success);border:1px solid var(--success);border-radius:3px;padding:2px 8px;font-size:11px;cursor:pointer;">Undo</button></td>'
        + '</tr>';
    }
    html += '</table>';
    html += '<p class="u-xxs u-muted u-mt-10">'
      + pairs.length + ' active reconciliation(s).</p>';
    list.innerHTML = html;
  } catch (exc) {
    list.innerHTML = '<em class="u-danger">Failed: ' + exc + '</em>';
  }
}

async function undoReconciliation(docId) {
  if (!confirm('Restore non-canonical document ' + docId.slice(0, 8) + ' to retrieval?')) return;
  try {
    const res = await fetch('/api/reconciliations/undo', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({doc_id: docId}),
    });
    if (res.ok) openReconciliationsModal();
  } catch (_) {}
}

// Phase 54.6.123 (Tier 3 #2) — inbound "cites-me" expansion.
async function runInboundExpand() {
  const seed = parseInt(document.getElementById('tl-inb-seed').value || '30', 10);
  const total = parseInt(document.getElementById('tl-inb-total').value || '300', 10);
  const lim = parseInt(document.getElementById('tl-inb-limit').value || '20', 10);
  const thr = parseFloat(document.getElementById('tl-inb-relthr').value || '0.55');
  const relq = (document.getElementById('tl-inb-relq').value || '').trim();
  const dry = document.getElementById('tl-inb-dry').checked;
  const retry = document.getElementById('tl-inb-retry').checked;
  const argv = [
    'db', 'expand-inbound',
    '--per-seed-cap', String(seed),
    '--total-limit', String(total),
    '--limit', String(lim),
    '--relevance-threshold', String(thr),
  ];
  if (relq) argv.push('--relevance-query', relq);
  if (dry) argv.push('--dry-run');
  if (retry) argv.push('--retry-failed');
  runCorpusCliAction(argv, 'Inbound expansion starting…');
}

// Phase 54.6.116 (Tier 2 #4) — author oeuvre completion.
async function runOeuvreExpand() {
  const min_ = parseInt(document.getElementById('tl-oeu-min').value || '3', 10);
  const lim = parseInt(document.getElementById('tl-oeu-limit').value || '10', 10);
  const mx  = parseInt(document.getElementById('tl-oeu-max').value || '10', 10);
  const dry = document.getElementById('tl-oeu-dry').checked;
  const argv = [
    'db', 'expand-oeuvre',
    '--min-corpus-papers', String(min_),
    '--per-author-limit', String(lim),
    '--max-authors', String(mx),
  ];
  if (dry) argv.push('--dry-run');
  runCorpusCliAction(argv, 'Scanning corpus authors…');
}

// Phase 54.6.131 — Oeuvre PREVIEW. Pre-fetches every qualifying
// author's candidates and surfaces the merged shortlist in the
// candidates modal so the user cherry-picks before any download.
async function openExpandOeuvrePreview() {
  const min_ = parseInt(document.getElementById('tl-oeu-min').value || '3', 10);
  const lim = parseInt(document.getElementById('tl-oeu-limit').value || '10', 10);
  const mx  = parseInt(document.getElementById('tl-oeu-max').value || '10', 10);
  _eapResetModal(
    '&#128100; Oeuvre &mdash; Preview Candidates',
    `Scanning ${mx} top corpus author(s) and fetching their bibliographies…`,
    `(typically ${mx} × 10-30s — slowest preview because it serially probes each author)`
  );
  const fd = new FormData();
  fd.append('min_corpus_papers', String(min_));
  fd.append('per_author_limit', String(lim));
  fd.append('max_authors', String(mx));
  fd.append('strict_author', 'true');
  try {
    const res = await fetch('/api/corpus/expand-oeuvre/preview', {
      method: 'POST', body: fd,
    });
    if (!res.ok) {
      const detail = await res.text();
      throw new Error('HTTP ' + res.status + ': ' + detail);
    }
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    _eapCandidates = data.candidates || [];
    const info = data.info || {};
    const authors = data.authors || [];
    if (!authors.length) {
      _eapShowError(
        info.message ||
        `No author has ≥${min_} corpus papers. Lower the threshold or expand the corpus first.`
      );
      return;
    }
    // Pre-select non-cached candidates (no relevance threshold —
    // preview is by-design pre-rank).
    document.getElementById('eap-threshold').value = '0.55';
    _eapCandidates.forEach(c => {
      if (c.doi && !c.cached_status) _eapSelected.add(c.doi);
    });
    // Build per-author summary chips for the info line.
    const chips = authors.slice(0, 8).map(a => {
      const orc = a.orcid ? ` <span class="u-accent">[ORCID]</span>` : '';
      const cls = a.error ? 'color:var(--danger);' : '';
      return `<span style="display:inline-block;background:var(--bg-alt,#f8f8f8);border:1px solid var(--border);padding:1px 6px;border-radius:3px;margin-right:4px;${cls}">`
        + `${_escHtml(a.name)} <span class="u-muted">(${a.n_corpus}c · ${a.n_candidates}n)</span>${orc}</span>`;
    }).join('');
    const more = authors.length > 8 ? ` <span class="u-muted">+${authors.length - 8} more</span>` : '';
    document.getElementById('eap-info').innerHTML =
      `<div class="u-mb-6">`
      + `Pulled candidates from <strong>${info.qualifying_authors || 0}</strong> author(s) with ≥${info.min_corpus_papers} corpus papers `
      + `(per-author cap ${info.per_author_limit}). `
      + `<strong>${info.merged_candidates || 0}</strong> unique paper(s) after dedup `
      + `(<strong>${info.cross_author_duplicates || 0}</strong> cross-author duplicates dropped). `
      + `Anchor: <code>${_escHtml(info.relevance_query_used || 'centroid')}</code>.`
      + `</div>`
      + `<div style="font-size:11px;line-height:1.7;">${chips}${more}</div>`
      + `<div class="u-tiny u-muted u-mt-1">Each row in the table shows the source author below its title — sort by author by clicking the column header.</div>`;
    document.getElementById('eap-loading').style.display = 'none';
    document.getElementById('eap-content').style.display = 'block';
    eapRender();
  } catch (e) {
    _eapShowError(e && e.message ? e.message : String(e));
  }
}

// Phase 54.6.114 (Tier 2 #2) — agentic question-driven expansion.
// Streams CLI output into the Corpus modal's existing log panel.
async function runAgenticExpand() {
  const q = (document.getElementById('tl-ag-question').value || '').trim();
  if (!q) { alert('Please enter a research question.'); return; }
  const rounds = parseInt(document.getElementById('tl-ag-rounds').value || '3', 10);
  const budget = parseInt(document.getElementById('tl-ag-budget').value || '10', 10);
  const threshold = parseInt(document.getElementById('tl-ag-threshold').value || '3', 10);
  const dry = document.getElementById('tl-ag-dry').checked;
  const resume = document.getElementById('tl-ag-resume').checked;
  const argv = [
    'db', 'expand',
    '--question', q,
    '--question-rounds', String(rounds),
    '--question-budget', String(budget),
    '--question-threshold', String(threshold),
  ];
  if (dry) argv.push('--dry-run');
  if (resume) argv.push('--resume');
  runCorpusCliAction(argv, 'Agentic expansion starting…');
}

// Phase 54.6.132 — Agentic PREVIEW. Single-round flow: decompose
// question → measure coverage → identify gaps → fetch candidates per
// gap → user cherry-picks in the candidates modal → downloads via
// the existing pipeline. To advance to round 2, the user re-clicks
// after ingestion settles (coverage gets re-measured against the
// updated corpus, gaps may disappear or change).
async function openAgenticPreview() {
  const q = (document.getElementById('tl-ag-question').value || '').trim();
  if (!q) { alert('Please enter a research question.'); return; }
  const budget = parseInt(document.getElementById('tl-ag-budget').value || '10', 10);
  const threshold = parseInt(document.getElementById('tl-ag-threshold').value || '3', 10);
  _eapResetModal(
    '&#129504; Agentic &mdash; Preview Round Candidates',
    'Decomposing question + measuring coverage + running RRF ranker per gap…',
    '(LLM decompose ~5s · per-gap RRF subprocess ~30-90s each — same ranker auto-mode uses, so preview = exactly what auto would download)'
  );
  const fd = new FormData();
  fd.append('question', q);
  fd.append('budget', String(budget));
  fd.append('threshold', String(threshold));
  try {
    const res = await fetch('/api/corpus/agentic/preview', {
      method: 'POST', body: fd,
    });
    if (!res.ok) {
      const detail = await res.text();
      throw new Error('HTTP ' + res.status + ': ' + detail);
    }
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    const subtopics = data.subtopics || [];
    const coverage = data.coverage || [];
    const gaps = data.gaps || [];
    const info = data.info || {};
    if (info.error) {
      _eapShowError(info.error);
      return;
    }
    if (info.all_covered) {
      _eapShowError(info.message || 'All sub-topics already covered.');
      return;
    }
    if (!subtopics.length) {
      _eapShowError('LLM decomposition returned no sub-topics. Try rephrasing.');
      return;
    }
    _eapCandidates = data.candidates || [];
    document.getElementById('eap-threshold').value = '0.55';
    _eapCandidates.forEach(c => {
      if (c.doi && !c.cached_status) _eapSelected.add(c.doi);
    });
    // Coverage table — every sub-topic with green/yellow/red dot.
    const covRows = coverage.map(r => {
      const dot = r.covered
        ? '<span class="u-success">●</span>'
        : (r.n_papers > 0 ? '<span class="u-warning">●</span>'
                          : '<span class="u-danger">●</span>');
      return `<div class="u-tiny u-lh-1-5">${dot} `
        + `<span title="${_escHtml(r.sample_titles.join(' • '))}">${_escHtml(r.subtopic)}</span> `
        + `<span class="u-muted">— ${r.n_papers} paper(s)`
        + (r.covered ? ` · covered` : ` · gap`)
        + `</span></div>`;
    }).join('');
    // Per-gap chips
    const gapChips = (gaps || []).map(g => {
      const cls = g.error ? 'color:var(--danger);' : '';
      return `<span style="display:inline-block;background:var(--bg-alt,#f8f8f8);border:1px solid var(--border);padding:1px 6px;border-radius:3px;margin-right:4px;${cls}">`
        + `${_escHtml(g.subtopic)} <span class="u-muted">(${g.n_candidates})</span></span>`;
    }).join('');
    document.getElementById('eap-info').innerHTML =
      `<div class="u-mb-2">`
      + `LLM decomposed into <strong>${subtopics.length}</strong> sub-topic(s); `
      + `<strong>${gaps.length}</strong> gap(s) under the ≥${info.doc_threshold}-paper threshold. `
      + `Pulled <strong>${info.merged_candidates || 0}</strong> candidate(s) via the `
      + `<strong>Phase-49 RRF ranker</strong> `
      + `(<strong>${info.cross_gap_duplicates || 0}</strong> cross-gap duplicates dropped). `
      + `<span class="u-success u-tiny">✓ same ranker auto-mode uses</span>`
      + `</div>`
      + `<div class="u-note-xs"><strong>Coverage snapshot:</strong></div>`
      + `<div class="u-mb-2">${covRows}</div>`
      + `<div class="u-note-mb-1"><strong>Per-gap candidate counts:</strong></div>`
      + `<div style="font-size:11px;line-height:1.7;margin-bottom:4px;">${gapChips}</div>`
      + `<div class="u-hint">Each row shows its source sub-topic below the title. After downloading, re-click <em>Preview round</em> to advance — the next round will re-measure coverage against the new corpus and propose fresh gaps.</div>`;
    document.getElementById('eap-loading').style.display = 'none';
    document.getElementById('eap-content').style.display = 'block';
    eapRender();
  } catch (e) {
    _eapShowError(e && e.message ? e.message : String(e));
  }
}


// Phase 46.E — author search-as-you-type for the Expand-by-Author panel.
// Debounced 200ms. Hits /api/catalog/authors?q=…&limit=15 and renders a
// clickable list; clicking a row sets window._selectedExpandAuthorName
// and populates the input + orcid + selected-line.
//
// Phase 54.6.1 — clicking a row was dead because the old implementation
// inline-interpolated JSON.stringify(a) into an onclick attribute wrapped
// in double quotes. The inner quotes terminated the attribute and
// corrupted everything after. Fixed by caching the author list and
// dispatching via event delegation (data-idx → closure lookup).
let _authorSearchTimer = null;
let _lastAuthorResults = [];
window._selectedExpandAuthorName = null;

function _escHtml(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function onExpandAuthorSearchInput(_event) {
  if (_authorSearchTimer) clearTimeout(_authorSearchTimer);
  _authorSearchTimer = setTimeout(runExpandAuthorSearch, 200);
}

async function runExpandAuthorSearch() {
  const q = document.getElementById('tl-eauth-q').value.trim();
  const box = document.getElementById('tl-eauth-results');
  try {
    const url = '/api/catalog/authors?limit=15'
              + (q.length >= 2 ? '&q=' + encodeURIComponent(q) : '');
    const res = await fetch(url);
    const data = await res.json();
    const authors = data.authors || [];
    _lastAuthorResults = authors;
    if (!authors.length) {
      box.style.display = 'block';
      box.innerHTML = '<div class="u-p-10 u-muted">No authors match.</div>';
      return;
    }
    const rows = authors.map((a, i) => {
      const safe = _escHtml(a.name);
      const orcidBit = a.orcid
        ? ` <span class="u-muted">orcid ${_escHtml(a.orcid)}</span>`
        : '';
      return `<div class="eauth-row u-p-6-10 u-click u-border-b" data-idx="${i}"
        onmouseenter="this.style.background='var(--accent-light,#eef2ff)';"
        onmouseleave="this.style.background='';">
        <strong>${safe}</strong>${orcidBit}
        <div class="u-hint">
          ${a.n_papers} paper${a.n_papers === 1 ? '' : 's'} ·
          ${a.n_citations} citation${a.n_citations === 1 ? '' : 's'} in corpus
        </div></div>`;
    }).join('');
    box.innerHTML = rows;
    box.style.display = 'block';
    // Event delegation — robust to any characters in author.name.
    box.querySelectorAll('.eauth-row').forEach(row => {
      row.addEventListener('click', () => {
        const idx = parseInt(row.dataset.idx, 10);
        const author = _lastAuthorResults[idx];
        if (author) selectExpandAuthor(author);
      });
    });
  } catch (exc) {
    box.style.display = 'block';
    box.innerHTML = `<div class="u-p-10 u-danger">Search failed: ${exc}</div>`;
  }
}

function selectExpandAuthor(author) {
  window._selectedExpandAuthorName = author.name;
  document.getElementById('tl-eauth-q').value = author.name;
  if (author.orcid) document.getElementById('tl-eauth-orcid').value = author.orcid;
  document.getElementById('tl-eauth-results').style.display = 'none';
  document.getElementById('tl-eauth-selected').innerHTML =
    '<span style="color:var(--success,#059669);">Selected:</span> <strong>'
    + author.name + '</strong>'
    + (author.orcid ? ' <code>' + author.orcid + '</code>' : '')
    + ' — ' + author.n_papers + ' paper(s), ' + author.n_citations + ' citation(s) in this corpus';
}

// Phase 54.6.314 — live-autocomplete for the corp-author-refs pane.
// Parallel to onExpandAuthorSearchInput / runExpandAuthorSearch but
// with separate IDs (tl-earef-*) and separate in-memory caches so the
// two panes can coexist.
let _authorRefsSearchTimer = null;
let _lastAuthorRefsResults = [];

function onExpandAuthorRefsSearchInput(_event) {
  if (_authorRefsSearchTimer) clearTimeout(_authorRefsSearchTimer);
  _authorRefsSearchTimer = setTimeout(runExpandAuthorRefsSearch, 200);
}

async function runExpandAuthorRefsSearch() {
  const q = document.getElementById('tl-earef-q').value.trim();
  const box = document.getElementById('tl-earef-results');
  try {
    const url = '/api/catalog/authors?limit=15'
              + (q.length >= 2 ? '&q=' + encodeURIComponent(q) : '');
    const res = await fetch(url);
    const data = await res.json();
    const authors = data.authors || [];
    _lastAuthorRefsResults = authors;
    if (!authors.length) {
      box.style.display = 'block';
      box.innerHTML = '<div class="u-p-10 u-muted">No authors match.</div>';
      return;
    }
    const rows = authors.map((a, i) => {
      const safe = _escHtml(a.name);
      const orcidBit = a.orcid
        ? ` <span class="u-muted">orcid ${_escHtml(a.orcid)}</span>`
        : '';
      return `<div class="earef-row u-p-6-10 u-click u-border-b" data-idx="${i}"
        onmouseenter="this.style.background='var(--accent-light,#eef2ff)';"
        onmouseleave="this.style.background='';">
        <strong>${safe}</strong>${orcidBit}
        <div class="u-hint">
          ${a.n_papers} paper${a.n_papers === 1 ? '' : 's'} ·
          ${a.n_citations} citation${a.n_citations === 1 ? '' : 's'} in corpus
        </div></div>`;
    }).join('');
    box.innerHTML = rows;
    box.style.display = 'block';
    box.querySelectorAll('.earef-row').forEach(row => {
      row.addEventListener('click', () => {
        const idx = parseInt(row.dataset.idx, 10);
        const author = _lastAuthorRefsResults[idx];
        if (author) selectExpandAuthorRefs(author);
      });
    });
  } catch (exc) {
    box.style.display = 'block';
    box.innerHTML = `<div class="u-p-10 u-danger">Search failed: ${exc}</div>`;
  }
}

function selectExpandAuthorRefs(author) {
  window._selectedExpandAuthorRefsName = author.name;
  document.getElementById('tl-earef-q').value = author.name;
  document.getElementById('tl-earef-results').style.display = 'none';
  document.getElementById('tl-earef-selected').innerHTML =
    '<span style="color:var(--success,#059669);">Selected:</span> <strong>'
    + _escHtml(author.name) + '</strong>'
    + (author.orcid ? ' <code>' + _escHtml(author.orcid) + '</code>' : '')
    + ' — ' + author.n_papers + ' paper(s), ' + author.n_citations
    + ' citation(s) in this corpus';
}

// Phase 54.6.314 — run the references preview against the currently
// selected author (pre-fills the openExpandAuthorRefs picker so it
// doesn't prompt() a second time).
async function runExpandAuthorRefsPreview() {
  const name = (window._selectedExpandAuthorRefsName
                || document.getElementById('tl-earef-q').value
                || '').trim();
  if (!name) {
    alert('Pick an author first — type a name above and click a row.');
    return;
  }
  const minEl = document.getElementById('tl-earef-min');
  const minMentions = Math.max(1, parseInt((minEl && minEl.value) || '1', 10) || 1);
  _eapResetModal(
    '&#128218; Expand by author references &mdash; Preview',
    `Aggregating references cited by ${_escHtml(name)}…`,
    '(reads the local citations table — typically <1s)'
  );
  const fd = new FormData();
  fd.append('author', name);
  fd.append('min_mentions', String(minMentions));
  try {
    const res = await fetch('/api/corpus/expand-author-refs/preview', {
      method: 'POST', body: fd,
    });
    if (!res.ok) {
      _eapShowError('HTTP ' + res.status + ': ' + await res.text());
      return;
    }
    const data = await res.json();
    _eapCandidates = data.candidates || [];
    _eapSelected = new Set();
    _eapCandidates.forEach(c => { if (c.doi) _eapSelected.add(c.doi); });
    document.getElementById('eap-info').innerHTML =
      `Author <strong>${_escHtml(data.author || name)}</strong> · `
      + `<strong>${data.n_author_papers || 0}</strong> corpus paper(s) · `
      + `<strong>${data.n_unique_references || 0}</strong> unique reference(s) · `
      + `dropped <strong>${data.dropped_in_corpus || 0}</strong> already in corpus.`
      + (data.note ? `<br><em>${_escHtml(data.note)}</em>` : '');
    document.getElementById('eap-loading').style.display = 'none';
    document.getElementById('eap-content').style.display = 'block';
    if (typeof eapRender === 'function') {
      try { eapRender(); } catch (e) {}
    }
  } catch (e) {
    _eapShowError(String(e));
  }
}


// ── Phase 54.6.1 — Expand-by-Author preview modal ──────────────────────
// openExpandAuthorPreview() pulls the params from the panel, POSTs to
// /api/corpus/expand-author/preview, and renders a checkboxed table of
// candidates. User cherry-picks → eapDownloadSelected() ships the DOIs
// to /api/corpus/expand-author/download-selected (SSE-streamed CLI).
let _eapCandidates = [];   // all candidates from the last preview
let _eapSelected = new Set(); // doi set — source of truth for selection
// Phase 54.6.49 — disambiguation banner state. Persisted across
// re-queries so the banner remains visible when the user re-runs
// with author_ids (the server returns an empty candidate_authors
// list in that mode since no name-resolution happened).
let _eapCandAuthors = [];   // OpenAlex canonical authors matching the surname
let _eapAuthorSelection = new Set();  // short_ids of ticked rows

function _eapResetModal(title, loadingMsg, loadingSub) {
  _eapCandidates = [];
  _eapSelected = new Set();
  _eapCandAuthors = [];
  _eapAuthorSelection = new Set();
  document.getElementById('eap-title').innerHTML = title;
  document.getElementById('eap-loading-msg').textContent = loadingMsg || 'Loading…';
  document.getElementById('eap-loading-sub').textContent = loadingSub || '';
  document.getElementById('eap-loading-log').style.display = 'none';
  document.getElementById('eap-loading-log').textContent = '';
  document.getElementById('eap-loading').style.display = 'block';
  document.getElementById('eap-error').style.display = 'none';
  document.getElementById('eap-content').style.display = 'none';
  document.getElementById('eap-log').style.display = 'none';
  document.getElementById('eap-log').textContent = '';
  document.getElementById('eap-status').textContent = '';
  openModal('candidates-preview-modal');
}

// Phase 54.6.131 — small helper used by oeuvre + agentic preview
// flows to show an error inside the candidates modal without writing
// the same 4 lines per call site.
function _eapShowError(msg) {
  document.getElementById('eap-loading').style.display = 'none';
  const el = document.getElementById('eap-error');
  el.textContent = msg || 'Unknown error';
  el.style.display = 'block';
}

async function openExpandAuthorPreview() {
  const name = document.getElementById('tl-eauth-q').value.trim();
  const orcid = document.getElementById('tl-eauth-orcid').value.trim();
  if (!name && !orcid) {
    alert('Type an author name (or ORCID) first.');
    return;
  }
  _eapResetModal(
    '&#128269; Expand-by-Author &mdash; Preview Candidates',
    'Searching OpenAlex + Crossref…',
    '(typically 10-30s depending on author\'s paper count)'
  );

  // Gather params from the panel
  const fd = new FormData();
  fd.append('name', name);
  if (orcid) fd.append('orcid', orcid);
  const yFrom = parseInt(document.getElementById('tl-eauth-yfrom').value || '0', 10);
  const yTo = parseInt(document.getElementById('tl-eauth-yto').value || '0', 10);
  if (yFrom) fd.append('year_from', yFrom);
  if (yTo) fd.append('year_to', yTo);
  const limit = parseInt(document.getElementById('tl-eauth-limit').value || '0', 10);
  if (limit > 0) fd.append('limit', limit);
  fd.append('strict_author', document.getElementById('tl-eauth-strict').checked);
  fd.append('all_matches', document.getElementById('tl-eauth-all').checked);
  const relq = document.getElementById('tl-eauth-relq').value.trim();
  if (relq) fd.append('relevance_query', relq);

  try {
    const res = await fetch('/api/corpus/expand-author/preview', {
      method: 'POST',
      body: fd,
    });
    if (!res.ok) {
      const detail = await res.text();
      throw new Error('HTTP ' + res.status + ': ' + detail);
    }
    const data = await res.json();
    _eapCandidates = data.candidates || [];
    const info = data.info || {};
    // Pre-select everything above the default threshold; otherwise all.
    const threshold = info.relevance_threshold || 0.0;
    document.getElementById('eap-threshold').value = threshold.toFixed(2);
    _eapCandidates.forEach(c => {
      // Phase 54.6.52 — don't auto-select cached rows; pipeline would skip them.
      if (c.doi && !c.cached_status
          && (c.relevance_score == null || c.relevance_score >= threshold)) {
        _eapSelected.add(c.doi);
      }
    });
    // Render info line
    const pickedAuthors = (info.picked_authors || [])
      .map(a => a.display_name || a.name || '').filter(Boolean).slice(0, 3).join(', ');
    const pickedSuffix = pickedAuthors
      ? ` · matched author(s): ${_escHtml(pickedAuthors)}` : '';
    document.getElementById('eap-info').innerHTML =
      `Found <strong>${info.merged || 0}</strong> paper(s) `
      + `(<span title="from OpenAlex canonical author search">${info.openalex || 0} OA</span> + `
      + `<span title="from Crossref surname search — may include false positives for common names">${info.crossref_extra || 0} CR</span>), `
      + `dropped <strong>${info.dedup_dropped || 0}</strong> already in corpus`
      + (info.relevance_query_used
          ? ` · relevance anchor: <code>${_escHtml(info.relevance_query_used)}</code>`
          : ` · no relevance scoring`)
      + pickedSuffix + '.';

    // Phase 54.6.47 + 54.6.49 — ambiguity-disambiguation banner with
    // multi-select. OpenAlex surfaces multiple canonical authors for
    // common surnames (e.g. "zharkova" → A. Zharkova the materials
    // scientist, V. V. Zharkova the solar physicist, G. I. Zharkova
    // the chemist, etc.) AND frequently lists the SAME person under
    // slightly different name variants ("V. V. Zharkova" +
    // "Valentina V. Zharkova" + "V.V. Zharkova"). Render the banner
    // with checkboxes so the user can tick every row that's actually
    // their target, then click Re-query to merge works across all of
    // them. Auto-picked row is pre-checked.
    //
    // _eapCandAuthors persists across re-queries so the banner can
    // re-render after an explicit-ids query (which returns an empty
    // candidate_authors list).
    const newCandAuthors = (info.candidate_authors || []);
    if (newCandAuthors.length) {
      _eapCandAuthors = newCandAuthors.slice(0, 15);
      const newPickedIds = new Set((info.picked_authors || []).map(a => a.short_id || a.id));
      _eapAuthorSelection = new Set();
      for (const a of _eapCandAuthors) {
        const sid = a.short_id || a.id;
        if (newPickedIds.has(sid)) _eapAuthorSelection.add(sid);
      }
    }
    if (_eapCandAuthors && _eapCandAuthors.length > 1) {
      let rows = '';
      for (const a of _eapCandAuthors) {
        const sid = a.short_id || a.id || '';
        const isChecked = _eapAuthorSelection.has(sid);
        const affil = (a.affiliations || []).slice(0, 2).join(', ');
        const orcid = a.orcid
          ? `<a href="${_escHtml(a.orcid)}" target="_blank" onclick="event.stopPropagation();">ORCID</a>`
          : '<span class="u-muted">no ORCID</span>';
        rows += `<tr data-action="eap-toggle-author" data-sid="${_escHtml(sid)}" `
          + `style="cursor:pointer;${isChecked ? 'background:rgba(80,200,120,0.12);' : ''}">`
          + `<td class="u-pill-md">`
          + `<input type="checkbox" class="eap-author-cb" data-sid="${_escHtml(sid)}" ${isChecked ? 'checked' : ''} title="Pin this author ID. Re-query will scope the search to only the ticked authors — defeats OpenAlex name-collision disambiguation."></td>`
          + `<td class="u-pill"><strong>${_escHtml(a.display_name || '')}</strong></td>`
          + `<td class="u-pill-right">${a.works_count || 0}w</td>`
          + `<td class="u-pill u-xxs u-muted">${_escHtml(affil)}</td>`
          + `<td class="u-pill">${orcid}</td></tr>`;
      }
      const numSelected = _eapAuthorSelection.size;
      const banner = `<div style="margin-top:8px;padding:10px;background:rgba(255,200,80,0.12);border-left:3px solid var(--warning);border-radius:4px;font-size:12px;">`
        + `<strong>&#9888; ${_eapCandAuthors.length} canonical authors match this surname</strong>. `
        + `Tick the rows that are actually the person you want (OpenAlex often lists the same person under multiple name variants). `
        + `All ticked rows will be merged on re-query.`
        + `<table class="u-mt-6 u-bcollapse u-tiny u-w-full">`
        + `<thead><tr class="u-border-b">`
        + `<th class="u-pill-sq"></th>`
        + `<th class="u-pill">Name</th>`
        + `<th class="u-pill-right">Works</th>`
        + `<th class="u-pill">Affiliation</th>`
        + `<th class="u-pill">ID</th></tr></thead>`
        + `<tbody>${rows}</tbody></table>`
        + `<div class="u-mt-2 u-flex-raw u-gap-2 u-ai-center">`
        + `<button class="btn-primary u-tiny u-p-4-10" onclick="eapRequeryWithSelected()" title="Re-run the search pinned to the ticked author IDs only. Defeats OpenAlex name-collision disambiguation.">`
        + `&#128269; Re-query with <span id="eap-sel-count">${numSelected}</span> selected</button>`
        + `<span class="u-label-xs">`
        + `Click rows to toggle. Re-query runs preview scoped to the ticked author IDs only.`
        + `</span></div></div>`;
      document.getElementById('eap-info').innerHTML += banner;
    }
    document.getElementById('eap-loading').style.display = 'none';
    if (!_eapCandidates.length) {
      document.getElementById('eap-error').style.display = 'block';
      document.getElementById('eap-error').textContent =
        'No candidates returned. Everything may already be in your corpus, or the search found nothing.';
      return;
    }
    document.getElementById('eap-content').style.display = 'block';
    eapRender();
  } catch (exc) {
    document.getElementById('eap-loading').style.display = 'none';
    document.getElementById('eap-error').style.display = 'block';
    document.getElementById('eap-error').textContent = 'Preview failed: ' + exc.message;
  }
}

// Phase 54.6.49 — multi-select disambiguation helpers.
function eapToggleAuthor(shortId) {
  if (!shortId) return;
  if (_eapAuthorSelection.has(shortId)) {
    _eapAuthorSelection.delete(shortId);
  } else {
    _eapAuthorSelection.add(shortId);
  }
  // Reflect in the DOM: row highlight + checkbox state + counter
  const row = document.querySelector(`tr[data-action="eap-toggle-author"][data-sid="${shortId}"]`);
  if (row) {
    const checked = _eapAuthorSelection.has(shortId);
    const cb = row.querySelector('input.eap-author-cb');
    if (cb) cb.checked = checked;
    row.style.background = checked ? 'rgba(80,200,120,0.12)' : '';
  }
  const cntEl = document.getElementById('eap-sel-count');
  if (cntEl) cntEl.textContent = String(_eapAuthorSelection.size);
}

// Phase 54.6.309 — expand-by-author-references entry point.
// Loads the top corpus authors, prompts the user to pick one, then
// calls /api/corpus/expand-author-refs/preview and lands in the
// shared candidates modal — download routes through the existing
// /api/corpus/expand-author/download-selected endpoint.
async function openExpandAuthorRefs() {
  let author = '';
  try {
    const r = await fetch('/api/corpus/authors/top?limit=30');
    if (r.ok) {
      const data = await r.json();
      const authors = data.authors || [];
      if (authors.length) {
        const list = authors
          .map((a, i) => `${i + 1}. ${a.name}  (${a.n_papers} papers)`)
          .join('\n');
        const pick = prompt(
          'Expand by author references — aggregates every paper cited by one author\'s corpus works.\n\n' +
          'Top authors in the corpus:\n' + list + '\n\n' +
          'Type a row number, a surname, or paste a full name:',
          '1'
        );
        if (!pick) return;
        const trimmed = pick.trim();
        const asNum = parseInt(trimmed, 10);
        if (!isNaN(asNum) && asNum >= 1 && asNum <= authors.length) {
          author = authors[asNum - 1].name;
        } else {
          author = trimmed;
        }
      }
    }
  } catch (e) {}
  if (!author) {
    author = prompt('Author name (surname is fine):', '') || '';
    if (!author) return;
  }

  _eapResetModal(
    '&#128218; Expand by author references &mdash; Preview',
    `Aggregating references cited by ${_escHtml(author)}…`,
    '(reads the local citations table — typically <1s)'
  );
  const fd = new FormData();
  fd.append('author', author);
  fd.append('min_mentions', '1');
  try {
    const res = await fetch('/api/corpus/expand-author-refs/preview', {
      method: 'POST', body: fd,
    });
    if (!res.ok) {
      _eapShowError('HTTP ' + res.status + ': ' + await res.text());
      return;
    }
    const data = await res.json();
    _eapCandidates = data.candidates || [];
    _eapSelected = new Set();
    _eapCandidates.forEach(c => { if (c.doi) _eapSelected.add(c.doi); });
    document.getElementById('eap-info').innerHTML =
      `Author <strong>${_escHtml(data.author || author)}</strong> · `
      + `<strong>${data.n_author_papers || 0}</strong> corpus paper(s) · `
      + `<strong>${data.n_unique_references || 0}</strong> unique reference(s) · `
      + `dropped <strong>${data.dropped_in_corpus || 0}</strong> already in corpus.`
      + (data.note ? `<br><em>${_escHtml(data.note)}</em>` : '');
    document.getElementById('eap-loading').style.display = 'none';
    document.getElementById('eap-content').style.display = 'block';
    if (typeof eapRender === 'function') {
      try { eapRender(); } catch (e) {}
    }
  } catch (e) {
    _eapShowError(String(e));
  }
}

async function eapRequeryWithSelected() {
  if (!_eapAuthorSelection.size) {
    alert('Tick at least one author row first.');
    return;
  }
  const ids = Array.from(_eapAuthorSelection).join(',');
  // Re-open the loading state, keep the modal open, hit the preview
  // endpoint with explicit author_ids (skips name resolution).
  document.getElementById('eap-content').style.display = 'none';
  document.getElementById('eap-error').style.display = 'none';
  document.getElementById('eap-loading').style.display = 'block';
  document.getElementById('eap-loading-msg').textContent =
    `Re-querying with ${_eapAuthorSelection.size} selected author(s)…`;
  document.getElementById('eap-loading-sub').textContent =
    '(merging works from the ticked canonical authors)';

  const fd = new FormData();
  // Pass name too — it's still used for the Crossref surname post-filter
  // if not in strict mode, but author_ids alone is enough for OpenAlex.
  const name = document.getElementById('tl-eauth-q').value.trim();
  if (name) fd.append('name', name);
  fd.append('author_ids', ids);
  const yFrom = parseInt(document.getElementById('tl-eauth-yfrom').value || '0', 10);
  const yTo = parseInt(document.getElementById('tl-eauth-yto').value || '0', 10);
  if (yFrom) fd.append('year_from', yFrom);
  if (yTo) fd.append('year_to', yTo);
  const limit = parseInt(document.getElementById('tl-eauth-limit').value || '0', 10);
  if (limit > 0) fd.append('limit', limit);
  const relq = document.getElementById('tl-eauth-relq').value.trim();
  if (relq) fd.append('relevance_query', relq);

  try {
    const res = await fetch('/api/corpus/expand-author/preview', {
      method: 'POST', body: fd,
    });
    if (!res.ok) {
      const detail = await res.text();
      throw new Error('HTTP ' + res.status + ': ' + detail);
    }
    const data = await res.json();
    _eapCandidates = data.candidates || [];
    _eapSelected = new Set();
    const info = data.info || {};
    const threshold = info.relevance_threshold || 0.0;
    document.getElementById('eap-threshold').value = threshold.toFixed(2);
    _eapCandidates.forEach(c => {
      // Phase 54.6.52 — don't auto-select cached rows; pipeline would skip them.
      if (c.doi && !c.cached_status
          && (c.relevance_score == null || c.relevance_score >= threshold)) {
        _eapSelected.add(c.doi);
      }
    });
    // Render a compact info line for the explicit-ids case. Reuse the
    // stored _eapCandAuthors so the banner + table still render.
    const chosenNames = _eapCandAuthors
      .filter(a => _eapAuthorSelection.has(a.short_id || a.id))
      .map(a => a.display_name || '').filter(Boolean).slice(0, 5).join(', ');
    document.getElementById('eap-info').innerHTML =
      `Re-queried with <strong>${_eapAuthorSelection.size}</strong> `
      + `selected author(s): ${_escHtml(chosenNames)}. `
      + `Found <strong>${info.merged || 0}</strong> paper(s), `
      + `dropped <strong>${info.dedup_dropped || 0}</strong> already in corpus`
      + (info.relevance_query_used
          ? ` · relevance anchor: <code>${_escHtml(info.relevance_query_used)}</code>`
          : ` · no relevance scoring`) + '.';

    // Re-render the banner so the user can adjust the selection again.
    if (_eapCandAuthors && _eapCandAuthors.length > 1) {
      let rows = '';
      for (const a of _eapCandAuthors) {
        const sid = a.short_id || a.id || '';
        const isChecked = _eapAuthorSelection.has(sid);
        const affil = (a.affiliations || []).slice(0, 2).join(', ');
        const orcid = a.orcid
          ? `<a href="${_escHtml(a.orcid)}" target="_blank" onclick="event.stopPropagation();">ORCID</a>`
          : '<span class="u-muted">no ORCID</span>';
        rows += `<tr data-action="eap-toggle-author" data-sid="${_escHtml(sid)}" `
          + `style="cursor:pointer;${isChecked ? 'background:rgba(80,200,120,0.12);' : ''}">`
          + `<td class="u-pill-md">`
          + `<input type="checkbox" class="eap-author-cb" data-sid="${_escHtml(sid)}" ${isChecked ? 'checked' : ''} title="Pin this author ID. Re-query will scope the search to only the ticked authors — defeats OpenAlex name-collision disambiguation."></td>`
          + `<td class="u-pill"><strong>${_escHtml(a.display_name || '')}</strong></td>`
          + `<td class="u-pill-right">${a.works_count || 0}w</td>`
          + `<td class="u-pill u-xxs u-muted">${_escHtml(affil)}</td>`
          + `<td class="u-pill">${orcid}</td></tr>`;
      }
      const numSelected = _eapAuthorSelection.size;
      const banner = `<div style="margin-top:8px;padding:10px;background:rgba(255,200,80,0.12);border-left:3px solid var(--warning);border-radius:4px;font-size:12px;">`
        + `<strong>${_eapCandAuthors.length} canonical authors</strong>. `
        + `Adjust selection and re-query if needed.`
        + `<table class="u-mt-6 u-bcollapse u-tiny u-w-full">`
        + `<thead><tr class="u-border-b">`
        + `<th class="u-pill-sq"></th>`
        + `<th class="u-pill">Name</th>`
        + `<th class="u-pill-right">Works</th>`
        + `<th class="u-pill">Affiliation</th>`
        + `<th class="u-pill">ID</th></tr></thead>`
        + `<tbody>${rows}</tbody></table>`
        + `<div class="u-mt-2 u-flex-raw u-gap-2 u-ai-center">`
        + `<button class="btn-primary u-tiny u-p-4-10" onclick="eapRequeryWithSelected()" title="Re-run the search pinned to the ticked author IDs only. Defeats OpenAlex name-collision disambiguation.">`
        + `&#128269; Re-query with <span id="eap-sel-count">${numSelected}</span> selected</button>`
        + `</div></div>`;
      document.getElementById('eap-info').innerHTML += banner;
    }

    document.getElementById('eap-loading').style.display = 'none';
    if (!_eapCandidates.length) {
      document.getElementById('eap-error').style.display = 'block';
      document.getElementById('eap-error').textContent =
        'No candidates returned for this selection. These author(s) may have no DOI-bearing works not already in the corpus.';
      return;
    }
    document.getElementById('eap-content').style.display = 'block';
    eapRender();
  } catch (exc) {
    document.getElementById('eap-loading').style.display = 'none';
    document.getElementById('eap-error').style.display = 'block';
    document.getElementById('eap-error').textContent = 'Re-query failed: ' + exc.message;
  }
}

// ── Phase 54.6.3 — Expand-Citations preview ───────────────────────────
// The expand-citations pipeline is 30s-3min depending on corpus size
// (four ref-extraction sources + multi-signal RRF ranking), so unlike
// expand-author we stream progress from a subprocess and fetch the
// parsed shortlist TSV once the job completes.
async function openExpandCitesPreview() {
  _eapResetModal(
    '&#127760; Expand-Citations &mdash; Preview Candidates',
    'Running dry-run…',
    '(this can take 30s–3min: reference extraction + multi-signal RRF ranking)'
  );
  document.getElementById('eap-loading-log').style.display = 'block';
  const logEl = document.getElementById('eap-loading-log');

  // Build form from the existing expand-citations panel inputs.
  const fd = new FormData();
  const limit = parseInt(document.getElementById('tl-exp-limit').value || '0', 10);
  if (limit > 0) fd.append('limit', limit);
  const relthr = parseFloat(document.getElementById('tl-exp-relthr').value || '0');
  fd.append('relevance_threshold', relthr);
  fd.append('relevance', document.getElementById('tl-exp-relevance').checked);
  fd.append('resolve', document.getElementById('tl-exp-resolve').checked);
  const relq = document.getElementById('tl-exp-relq').value.trim();
  if (relq) fd.append('relevance_query', relq);

  let res;
  try {
    res = await fetch('/api/corpus/expand/preview', {method: 'POST', body: fd});
  } catch (exc) {
    document.getElementById('eap-loading').style.display = 'none';
    document.getElementById('eap-error').style.display = 'block';
    document.getElementById('eap-error').textContent = 'Request failed: ' + exc.message;
    return;
  }
  if (!res.ok) {
    document.getElementById('eap-loading').style.display = 'none';
    document.getElementById('eap-error').style.display = 'block';
    document.getElementById('eap-error').textContent = 'Preview start failed: HTTP ' + res.status;
    return;
  }
  const data = await res.json();
  const jobId = data.job_id;

  const source = new EventSource('/api/stream/' + jobId);
  source.onmessage = async function(e) {
    let evt;
    try { evt = JSON.parse(e.data); } catch (_) { return; }
    if (evt.type === 'log') {
      logEl.textContent += evt.text + '\n';
      logEl.scrollTop = logEl.scrollHeight;
    } else if (evt.type === 'progress') {
      if (evt.detail && evt.detail.startsWith('$ ')) {
        logEl.textContent += evt.detail + '\n';
      }
    } else if (evt.type === 'completed') {
      source.close();
      // Fetch parsed candidates.
      try {
        const r2 = await fetch('/api/corpus/expand/preview/' + jobId + '/candidates');
        if (!r2.ok) throw new Error('HTTP ' + r2.status);
        const cands = await r2.json();
        _eapCandidates = cands.candidates || [];
        const info = cands.info || {};
        // Pre-select only the KEEP rows (DROP rows fell below threshold /
        // had hard filters hit — user can still tick them if they want).
        _eapSelected = new Set();
        _eapCandidates.forEach(c => {
          if (c.doi && c.decision === 'KEEP') _eapSelected.add(c.doi);
        });
        // Info line
        const reasonSummary = Object.entries(info.drop_reasons || {})
          .map(([k, v]) => _escHtml(k) + ': ' + v).join(' · ');
        document.getElementById('eap-info').innerHTML =
          'Shortlist of <strong>' + (info.total || 0) + '</strong> candidate(s) '
          + '(<span class="u-success">' + (info.kept || 0)
          + ' KEEP</span> · <span class="u-muted">'
          + (info.dropped || 0) + ' DROP</span>). '
          + (reasonSummary ? 'Drop reasons: ' + reasonSummary + '.' : '')
          + ' Scores from the RRF-fused ranker (bge-m3 cosine shown below; '
          + 'full multi-signal breakdown in drop_reason / decision).';
        document.getElementById('eap-loading').style.display = 'none';
        if (!_eapCandidates.length) {
          document.getElementById('eap-error').style.display = 'block';
          document.getElementById('eap-error').textContent =
            'No candidates returned. The corpus may have no extractable references, or everything was already deduped.';
          return;
        }
        document.getElementById('eap-content').style.display = 'block';
        eapRender();
      } catch (exc) {
        document.getElementById('eap-loading').style.display = 'none';
        document.getElementById('eap-error').style.display = 'block';
        document.getElementById('eap-error').textContent = 'Candidates fetch failed: ' + exc.message;
      }
    } else if (evt.type === 'error') {
      source.close();
      document.getElementById('eap-loading').style.display = 'none';
      document.getElementById('eap-error').style.display = 'block';
      document.getElementById('eap-error').textContent =
        'Preview failed: ' + (evt.message || 'see log.');
    } else if (evt.type === 'done') {
      source.close();
    }
  };
}

// ── Phase 54.6.4 — three more preview flows (inbound / topic / coauth).
// All three use the same modal + eapRender as expand-author preview,
// just different backing endpoints and input fields.

async function _eapFetchPreview(url, fd, titleHtml, loadingMsg, loadingSub) {
  _eapResetModal(titleHtml, loadingMsg, loadingSub);
  try {
    const res = await fetch(url, {method: 'POST', body: fd});
    if (!res.ok) {
      const detail = await res.text();
      throw new Error('HTTP ' + res.status + ': ' + detail);
    }
    const data = await res.json();
    _eapCandidates = data.candidates || [];
    const info = data.info || {};
    // Default: pre-select all with score >= threshold (or all if no scores).
    const thr = 0.55;
    document.getElementById('eap-threshold').value = thr.toFixed(2);
    _eapSelected = new Set();
    _eapCandidates.forEach(c => {
      if (c.doi && !c.cached_status
          && (c.relevance_score == null || c.relevance_score >= thr)) {
        _eapSelected.add(c.doi);
      }
    });
    // Info banner — method-specific summary.
    const bits = [];
    if (info.raw != null) bits.push('<strong>' + info.raw + '</strong> raw');
    if (info.dedup_dropped != null)
      bits.push(info.dedup_dropped + ' already in corpus');
    if (info.seeds_resolved != null)
      bits.push(info.seeds_resolved + '/' + (info.seeds_requested || 0) + ' seeds resolved');
    if (info.seed_authors != null)
      bits.push(info.seed_authors + ' seed authors');
    if (info.relevance_query_used)
      bits.push('anchor: <code>' + _escHtml(info.relevance_query_used) + '</code>');
    document.getElementById('eap-info').innerHTML = bits.join(' · ');
    document.getElementById('eap-loading').style.display = 'none';
    if (!_eapCandidates.length) {
      document.getElementById('eap-error').style.display = 'block';
      document.getElementById('eap-error').textContent =
        'No candidates returned. '
        + 'Try a different query / lower threshold / wider seed cap.';
      return;
    }
    document.getElementById('eap-content').style.display = 'block';
    eapRender();
  } catch (exc) {
    document.getElementById('eap-loading').style.display = 'none';
    document.getElementById('eap-error').style.display = 'block';
    document.getElementById('eap-error').textContent = 'Preview failed: ' + exc.message;
  }
}

async function openExpandInboundPreview() {
  const fd = new FormData();
  fd.append('per_seed_cap', document.getElementById('tl-inb-seed').value || '30');
  fd.append('total_limit',  document.getElementById('tl-inb-total').value || '300');
  const relq = document.getElementById('tl-inb-relq').value.trim();
  if (relq) fd.append('relevance_query', relq);
  await _eapFetchPreview(
    '/api/corpus/expand-cites/preview', fd,
    '&#128258; Inbound-citation &mdash; Preview Candidates',
    'Querying OpenAlex for papers that cite your corpus…',
    '(30s–2 min: one API call per seed paper)'
  );
}

async function openExpandTopicPreview() {
  const q = document.getElementById('tl-top-q').value.trim();
  if (!q) {
    alert('Enter a topic query first.');
    return;
  }
  const fd = new FormData();
  fd.append('query', q);
  fd.append('limit', document.getElementById('tl-top-limit').value || '300');
  const relq = document.getElementById('tl-top-relq').value.trim();
  if (relq) fd.append('relevance_query', relq);
  await _eapFetchPreview(
    '/api/corpus/expand-topic/preview', fd,
    '&#128269; Topic search &mdash; Preview Candidates',
    'Searching OpenAlex for "' + _escHtml(q) + '"…',
    '(5–20s: paginated /works?search sorted by citation count)'
  );
}

async function openExpandCoauthPreview() {
  const fd = new FormData();
  fd.append('depth', document.getElementById('tl-coa-depth').value || '1');
  fd.append('per_author_cap', document.getElementById('tl-coa-per').value || '8');
  fd.append('total_limit',    document.getElementById('tl-coa-total').value || '300');
  const relq = document.getElementById('tl-coa-relq').value.trim();
  if (relq) fd.append('relevance_query', relq);
  await _eapFetchPreview(
    '/api/corpus/expand-coauthors/preview', fd,
    '&#128101; Coauthor snowball &mdash; Preview Candidates',
    'Enumerating corpus authors → fetching their works…',
    '(30s–3 min: scales with author count × per-author cap)'
  );
}

// Phase 54.6.5 — Gap-driven auto-expand. Called from the Dashboard gaps
// panel when the user clicks "Auto-expand from these gaps".
async function openAutoExpandPreview() {
  const fd = new FormData();
  fd.append('per_gap_limit', '80');
  await _eapFetchPreview(
    '/api/book/auto-expand/preview', fd,
    '&#128269; Auto-expand &mdash; Candidates for open gaps',
    'Running a topic search per open gap…',
    '(1–5 min: one OpenAlex query per topic/evidence gap, then merged + relevance-scored)'
  );
}

// Per-gap "Expand" button handler: opens the Corpus modal on the
// Topic-search subtab, pre-fills the query with the gap description,
// and kicks off the preview. Shorter path than running the Dashboard-
// level auto-expand if the user only cares about one specific gap.
// Phase 54.6.18 — now goes via openCorpusModal (Corpus is its own modal).
function expandSingleGap(gapDesc) {
  if (!gapDesc) return;
  openCorpusModal('corp-topic');
  // Give the modal a tick to mount, then prefill + fire.
  setTimeout(() => {
    const q = document.getElementById('tl-top-q');
    if (q) q.value = gapDesc;
    openExpandTopicPreview();
  }, 120);
}

// ── Phase 54.6.7 — Pending downloads modal ─────────────────────────────
// When expand flows can't find an OA PDF for a paper the user selected,
// the row goes into the pending_downloads table. This modal is the
// curator UI: retry / mark-done / abandon / export for papers stuck
// without a legal PDF.
let _pdlRows = [];
let _pdlSelected = new Set();
let _pdlRetryJob = null;

async function openPendingDownloadsModal() {
  openModal('pending-downloads-modal');
  await refreshPendingDownloads();
}

async function refreshPendingDownloads() {
  const status = document.getElementById('pdl-status').value || 'pending';
  const source = document.getElementById('pdl-source').value || '';
  const url = '/api/pending-downloads?status=' + encodeURIComponent(status)
            + (source ? '&source=' + encodeURIComponent(source) : '');
  const tbody = document.getElementById('pdl-tbody');
  tbody.innerHTML = '<tr><td class="u-p-20 u-text-center u-muted" colspan="8">Loading…</td></tr>';
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    _pdlRows = data.rows || [];
    _pdlSelected = new Set();
    document.getElementById('pdl-count').textContent = data.count + ' row(s)';
    _renderPendingTable();
  } catch (exc) {
    tbody.innerHTML = '<tr><td class="u-p-4 u-danger" colspan="8">Failed: '
      + _escHtml(exc.message) + '</td></tr>';
  }
}

function _renderPendingTable() {
  const tbody = document.getElementById('pdl-tbody');
  if (!_pdlRows.length) {
    tbody.innerHTML = '<tr><td class="u-p-20 u-text-center u-muted" colspan="8">'
      + 'No rows for the current filter.</td></tr>';
    _pdlUpdateHeaderCb();
    return;
  }
  const rows = _pdlRows.map(r => {
    const checked = _pdlSelected.has(r.doi) ? 'checked' : '';
    const authors = (r.authors || []).slice(0, 3).join(', ')
      + ((r.authors || []).length > 3 ? ` +${r.authors.length - 3}` : '');
    const doiUrl = r.doi
      ? `<a class="u-accent u-no-underline u-mono-sys u-xxs" href="https://doi.org/${_escHtml(r.doi)}" target="_blank" rel="noopener">${_escHtml(r.doi)}</a>`
      : '<span class="u-muted">(no DOI)</span>';
    const actions = `
      <button class="btn-secondary u-chip-xs"
              onclick="pendingRetrySingle('${_escHtml(r.doi)}')">&#8635;</button>
      <button class="btn-secondary u-chip-xs"
              onclick="pendingMarkStatus('${_escHtml(r.doi)}','manual_acquired')"
              title="Mark manually acquired">&#10003;</button>
      <button class="btn-secondary u-chip-xs"
              onclick="pendingMarkStatus('${_escHtml(r.doi)}','abandoned')"
              title="Abandon">&#215;</button>
    `;
    return `<tr class="u-border-t" data-doi="${_escHtml(r.doi)}">
      <td class="u-pill-lg"><input type="checkbox" class="pdl-row-cb" data-doi="${_escHtml(r.doi)}" ${checked} title="Select this row for bulk Retry / Mark-done / Abandon / Export."></td>
      <td class="u-pill-lg">
        <div class="u-fw-5">${_escHtml(r.title || '(untitled)')}</div>
        <div class="u-xxs u-mt-2px">${doiUrl}</div>
      </td>
      <td class="u-cell-muted">${_escHtml(authors)}</td>
      <td class="u-cell-muted">${r.year || '—'}</td>
      <td class="u-pill-lg u-muted u-xxs">${_escHtml(r.source_method || '')}</td>
      <td class="u-p-6-8 u-mono-sys">${r.attempt_count}</td>
      <td class="u-pill-lg u-muted u-tiny">${_escHtml(r.last_failure_reason || '')}</td>
      <td class="u-p-6-8 u-nowrap">${actions}</td>
    </tr>`;
  }).join('');
  tbody.innerHTML = rows;
  tbody.querySelectorAll('.pdl-row-cb').forEach(cb => {
    cb.addEventListener('change', () => {
      const doi = cb.dataset.doi;
      if (cb.checked) _pdlSelected.add(doi);
      else _pdlSelected.delete(doi);
      _pdlUpdateHeaderCb();
    });
  });
  _pdlUpdateHeaderCb();
}

function _pdlUpdateHeaderCb() {
  const hdr = document.getElementById('pdl-header-cb');
  if (!hdr) return;
  const sel = _pdlSelected.size;
  const tot = _pdlRows.length;
  hdr.checked = sel > 0 && sel === tot;
  hdr.indeterminate = sel > 0 && sel < tot;
}

function pendingSelectAll(on) {
  _pdlSelected = new Set();
  if (on) _pdlRows.forEach(r => { if (r.doi) _pdlSelected.add(r.doi); });
  _renderPendingTable();
}

async function pendingMarkStatus(doi, status) {
  if (!doi) return;
  if (status === 'abandoned' &&
      !confirm('Abandon "' + doi + '"? You can reopen it from the CLI later.')) return;
  try {
    const res = await fetch('/api/pending-downloads/update', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({doi: doi, status: status}),
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    await refreshPendingDownloads();
  } catch (exc) {
    alert('Update failed: ' + exc.message);
  }
}

async function pendingRetrySingle(doi) {
  if (!doi) return;
  await pendingRetryDois([{doi: doi}]);
}

async function pendingRetrySelected() {
  if (!_pdlSelected.size) {
    alert('Select at least one row (or use "Select all").');
    return;
  }
  const chosen = _pdlRows
    .filter(r => r.doi && _pdlSelected.has(r.doi))
    .map(r => ({doi: r.doi, title: r.title || '', year: r.year || null}));
  await pendingRetryDois(chosen);
}

async function pendingRetryDois(rowsOrDois) {
  const logEl = document.getElementById('pdl-retry-log');
  const status = document.getElementById('pdl-retry-status');
  logEl.style.display = 'block';
  logEl.textContent = '';
  status.textContent = 'Starting retry…';
  try {
    const res = await fetch('/api/pending-downloads/retry', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({dois: rowsOrDois, ingest: true, workers: 0}),
    });
    if (!res.ok) {
      const detail = await res.text();
      throw new Error('HTTP ' + res.status + ': ' + detail);
    }
    const data = await res.json();
    _pdlRetryJob = data.job_id;
    status.textContent = `Retrying ${data.n_retried} DOI(s)… (see log)`;
    const source = new EventSource('/api/stream/' + data.job_id);
    source.onmessage = function(e) {
      let evt;
      try { evt = JSON.parse(e.data); } catch (_) { return; }
      if (evt.type === 'log') {
        logEl.textContent += evt.text + '\n';
        logEl.scrollTop = logEl.scrollHeight;
      } else if (evt.type === 'progress') {
        if (evt.detail && evt.detail.startsWith('$ ')) {
          logEl.textContent += evt.detail + '\n';
        }
      } else if (evt.type === 'completed') {
        source.close(); _pdlRetryJob = null;
        status.textContent = 'Retry complete. Refreshing list…';
        setTimeout(refreshPendingDownloads, 600);
      } else if (evt.type === 'error') {
        source.close(); _pdlRetryJob = null;
        status.textContent = 'Retry failed: ' + (evt.message || 'see log');
      } else if (evt.type === 'done') {
        source.close(); _pdlRetryJob = null;
      }
    };
  } catch (exc) {
    status.textContent = 'Retry failed: ' + exc.message;
  }
}

function pendingExportCsv() {
  // Convert the currently-rendered _pdlRows to CSV client-side.
  if (!_pdlRows.length) {
    alert('No rows to export.');
    return;
  }
  const cols = ['doi','title','authors','year','source_method','source_query',
                'relevance_score','attempt_count','last_attempt_at',
                'last_failure_reason','status','notes'];
  const lines = [cols.join(',')];
  for (const r of _pdlRows) {
    const row = cols.map(k => {
      let v = r[k];
      if (Array.isArray(v)) v = v.join('; ');
      if (v == null) v = '';
      v = String(v);
      if (/[",\n]/.test(v)) v = '"' + v.replace(/"/g, '""') + '"';
      return v;
    });
    lines.push(row.join(','));
  }
  const blob = new Blob([lines.join('\n')], {type: 'text/csv'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'pending_downloads_' + (new Date().toISOString().slice(0,10)) + '.csv';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// Phase 54.6.115/120 — ±mark a candidate from the shortlist.
// 54.6.120 fix: moved from inline onclick with JSON.stringify(JSON.stringify(x))
// (which broke HTML attributes on any title containing '<' / '"' etc.) to
// data-* attributes + a delegated click handler attached after render.
async function eapFeedback(btn) {
  if (!btn) return;
  const kind = btn.dataset.kind;
  const doi = btn.dataset.doi || '';
  const arxiv_id = btn.dataset.arxivId || '';
  const title = btn.dataset.title || '';
  if (!kind || !(doi || arxiv_id || title)) return;
  try {
    const res = await fetch('/api/feedback', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        action: 'add', kind: kind,
        doi: doi, arxiv_id: arxiv_id, title: title,
      }),
    });
    if (!res.ok) return;
    // Visual ack — flash the button background and disable briefly.
    const prev = btn.style.background;
    btn.style.background = kind === 'positive'
      ? 'rgba(80,200,120,0.55)'
      : 'rgba(220,80,80,0.55)';
    btn.disabled = true;
    setTimeout(() => {
      btn.style.background = prev;
      btn.disabled = false;
    }, 600);
  } catch (_) {}
}

function eapRender() {
  const tbody = document.getElementById('eap-tbody');
  const sort = document.getElementById('eap-sort').value;
  // Phase 54.6.52 — cached-row filter. Count + filter before sort so
  // the "Hide cached (N)" label stays accurate.
  const hideCachedEl = document.getElementById('eap-hide-cached');
  const hideCached = hideCachedEl ? hideCachedEl.checked : true;
  const totalCached = _eapCandidates.filter(c => c.cached_status).length;
  const cntEl = document.getElementById('eap-cached-count');
  // 54.6.120 — always show the count so the user can see when there's
  // nothing to hide (previously empty string was indistinguishable from
  // the rendering being broken — the user reported the checkbox "didn't
  // seem to work" on a preview that had no cached rows to hide).
  if (cntEl) cntEl.textContent = `(${totalCached})`;
  const pool = hideCached
    ? _eapCandidates.filter(c => !c.cached_status)
    : _eapCandidates.slice();
  const sorted = pool.slice();
  if (sort === 'year') {
    sorted.sort((a, b) => (b.year || 0) - (a.year || 0));
  } else if (sort === 'title') {
    sorted.sort((a, b) => (a.title || '').localeCompare(b.title || ''));
  } else {
    sorted.sort((a, b) => (b.relevance_score || -1) - (a.relevance_score || -1));
  }
  // Push cached rows to the bottom within each sort (user can still
  // see them when "Hide cached" is unchecked, but they don't clutter
  // the top of the relevance-ranked list).
  sorted.sort((a, b) => (a.cached_status ? 1 : 0) - (b.cached_status ? 1 : 0));
  const rows = sorted.map(c => {
    const checked = _eapSelected.has(c.doi) ? 'checked' : '';
    const scoreText = (c.relevance_score == null)
      ? '<span class="u-muted">—</span>'
      : c.relevance_score.toFixed(3);
    const authors = (c.authors || []).slice(0, 3).join(', ')
      + ((c.authors || []).length > 3 ? ` +${c.authors.length - 3}` : '');
    const doi = c.doi
      ? `<a class="u-accent u-no-underline u-mono-sys u-xxs" href="https://doi.org/${_escHtml(c.doi)}" target="_blank" rel="noopener" onclick="event.stopPropagation();">${_escHtml(c.doi)}</a>`
      : '<span class="u-muted">(no DOI)</span>';
    // Phase 54.6.51 — alt-source badge: show "+N more" when the row
    // represents a title-dedup group covering multiple DOIs / arXiv IDs.
    const altCount = (c.alternate_dois || []).length + (c.alternate_arxiv_ids || []).length;
    const altBadge = altCount > 0
      ? ` <span title="Same paper also known under ${altCount} alternate identifier(s); will be used as download fallbacks" style="background:rgba(80,200,120,0.15);color:var(--success);padding:1px 5px;border-radius:3px;font-size:9px;margin-left:4px;">+${altCount} alt</span>`
      : '';
    // Phase 54.6.52 — cached-status badge + dimmed row.
    let cachedBadge = '';
    let rowStyle = 'border-top:1px solid var(--border);cursor:pointer;';
    if (c.cached_status === 'no_oa') {
      cachedBadge = ` <span title="Prior run confirmed no open-access PDF via all tried sources. Re-tick 'retry previously-failed' to re-probe." style="background:rgba(180,180,180,0.2);color:var(--fg-muted);padding:1px 5px;border-radius:3px;font-size:9px;margin-left:4px;">&#128164; cached: no OA</span>`;
      rowStyle += 'opacity:0.55;';
    } else if (c.cached_status === 'ingest_failed') {
      cachedBadge = ` <span title="Prior run downloaded the PDF but the converter (MinerU/Marker) couldn't parse it. Re-tick 'retry previously-failed' to re-try." style="background:rgba(220,160,80,0.18);color:var(--warning);padding:1px 5px;border-radius:3px;font-size:9px;margin-left:4px;">&#9888; cached: ingest fail</span>`;
      rowStyle += 'opacity:0.55;';
    }
    // Phase 54.6.131 — oeuvre + agentic preview source badge.
    // Shows "from: <author>" or "topic: <subtopic>" under the title
    // so the user knows which gap/author surfaced this row.
    let sourceBadge = '';
    if (c._oeuvre_author) {
      sourceBadge = `<div class="u-xxs u-mt-2px u-accent">&#128100; from oeuvre author: <strong>${_escHtml(c._oeuvre_author)}</strong></div>`;
    } else if (c._agentic_subtopic) {
      sourceBadge = `<div class="u-xxs u-mt-2px u-accent">&#129504; sub-topic: <strong>${_escHtml(c._agentic_subtopic)}</strong></div>`;
    }
    return `<tr style="${rowStyle}" data-doi="${_escHtml(c.doi || '')}">
      <td class="u-pill-lg"><input type="checkbox" class="eap-row-cb" ${checked}
           data-doi="${_escHtml(c.doi || '')}" onclick="event.stopPropagation();"></td>
      <td class="u-pill-lg">
        <div class="u-fw-5">${_escHtml(c.title || '(untitled)')}</div>
        <div class="u-xxs u-mt-2px">${doi}${altBadge}${cachedBadge}</div>
        ${sourceBadge}
      </td>
      <td class="u-cell-muted">${_escHtml(authors)}</td>
      <td class="u-cell-muted">${c.year || '—'}</td>
      <td class="u-p-6-8 u-mono-sys">${scoreText}</td>
      <td class="u-p-6-8 u-nowrap" onclick="event.stopPropagation();">
        <button class="eap-fb-btn" data-kind="positive"
                data-doi="${_escHtml(c.doi || '')}"
                data-arxiv-id="${_escHtml(c.arxiv_id || '')}"
                data-title="${_escHtml(c.title || '')}"
                title="Mark as positive — next expand round will favour similar papers"
                style="background:rgba(80,200,120,0.15);color:var(--success);border:1px solid var(--success);border-radius:3px;padding:1px 6px;font-size:11px;cursor:pointer;margin-right:2px;">+</button>
        <button class="eap-fb-btn" data-kind="negative"
                data-doi="${_escHtml(c.doi || '')}"
                data-arxiv-id="${_escHtml(c.arxiv_id || '')}"
                data-title="${_escHtml(c.title || '')}"
                title="Mark as negative — next expand round will penalize similar papers"
                style="background:rgba(220,80,80,0.15);color:var(--danger);border:1px solid var(--danger);border-radius:3px;padding:1px 6px;font-size:11px;cursor:pointer;">−</button>
      </td>
    </tr>`;
  }).join('');
  tbody.innerHTML = rows;
  // 54.6.120 — attach click handlers for the ± feedback buttons
  // (data-attr based; replaces the broken inline onclick in 54.6.115).
  tbody.querySelectorAll('.eap-fb-btn').forEach(btn => {
    btn.addEventListener('click', function(e) {
      e.stopPropagation();
      eapFeedback(this);
    });
  });
  // Row click toggles the row's checkbox (but not on link click — event.stopPropagation above).
  tbody.querySelectorAll('tr').forEach(row => {
    row.addEventListener('click', () => {
      const cb = row.querySelector('.eap-row-cb');
      if (!cb || !cb.dataset.doi) return;
      cb.checked = !cb.checked;
      if (cb.checked) _eapSelected.add(cb.dataset.doi);
      else _eapSelected.delete(cb.dataset.doi);
      eapUpdateCount();
    });
  });
  tbody.querySelectorAll('.eap-row-cb').forEach(cb => {
    cb.addEventListener('change', () => {
      if (!cb.dataset.doi) return;
      if (cb.checked) _eapSelected.add(cb.dataset.doi);
      else _eapSelected.delete(cb.dataset.doi);
      eapUpdateCount();
    });
  });
  eapUpdateCount();
}

function eapUpdateCount() {
  const tot = _eapCandidates.length;
  const sel = _eapSelected.size;
  document.getElementById('eap-selected-count').textContent =
    `${sel} of ${tot} selected`;
  const hdr = document.getElementById('eap-header-cb');
  if (hdr) {
    hdr.checked = sel > 0 && sel === tot;
    hdr.indeterminate = sel > 0 && sel < tot;
  }
}

function eapSelectAll(on) {
  _eapSelected = new Set();
  if (on) {
    _eapCandidates.forEach(c => { if (c.doi) _eapSelected.add(c.doi); });
  }
  eapRender();
}

function eapSelectByThreshold() {
  const thr = parseFloat(document.getElementById('eap-threshold').value || '0');
  _eapSelected = new Set();
  _eapCandidates.forEach(c => {
    if (!c.doi) return;
    if (c.relevance_score == null || c.relevance_score >= thr) {
      _eapSelected.add(c.doi);
    }
  });
  eapRender();
}

async function eapDownloadSelected() {
  if (!_eapSelected.size) {
    alert('Pick at least one paper (or use "Select all").');
    return;
  }
  const chosen = _eapCandidates.filter(c => c.doi && _eapSelected.has(c.doi));
  const payload = {
    candidates: chosen.map(c => ({
      doi: c.doi, title: c.title || '', year: c.year || null,
      // Phase 54.6.51 — forward alternate identifiers for title-merged
      // duplicates so the downloader can fall back (preprint → journal
      // → HAL → Zenodo → etc.) if the primary DOI has no OA mirror.
      alternate_dois: c.alternate_dois || [],
      alternate_arxiv_ids: c.alternate_arxiv_ids || [],
    })),
    workers: parseInt(document.getElementById('eap-workers').value || '0', 10),
    ingest: document.getElementById('eap-ingest').checked,
    // Phase 54.6.52 — retry cached failures if the checkbox is ticked.
    retry_failed: (document.getElementById('eap-retry-failed') || {}).checked || false,
  };
  const btn = document.getElementById('eap-download-btn');
  btn.disabled = true;
  btn.textContent = 'Starting…';
  document.getElementById('eap-status').textContent = '';
  document.getElementById('eap-log').style.display = 'block';
  document.getElementById('eap-log').textContent = '';
  try {
    const res = await fetch('/api/corpus/expand-author/download-selected', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const detail = await res.text();
      throw new Error('HTTP ' + res.status + ': ' + detail);
    }
    const data = await res.json();
    const jobId = data.job_id;
    document.getElementById('eap-status').textContent =
      `Job started (${data.n_selected} DOI(s)). Streaming log below…`;
    btn.textContent = 'Running…';
    startGlobalJob(jobId, {
      type: 'download-selected',
      taskDesc: `download-selected (${data.n_selected} papers)`,
    });
    const es = new EventSource('/api/stream/' + jobId);
    const logEl = document.getElementById('eap-log');
    es.onmessage = function(ev) {
      let evt;
      try { evt = JSON.parse(ev.data); } catch (_) { return; }
      if (evt.type === 'log') {
        logEl.textContent += evt.text + '\n';
        logEl.scrollTop = logEl.scrollHeight;
      } else if (evt.type === 'progress') {
        document.getElementById('eap-status').textContent = evt.detail || evt.stage || '';
      } else if (evt.type === 'completed') {
        es.close();
        document.getElementById('eap-status').textContent = 'Completed.';
        btn.innerHTML = '&#10003; Done';
        btn.disabled = false;
      } else if (evt.type === 'error') {
        es.close();
        document.getElementById('eap-status').textContent =
          'Failed: ' + (evt.message || 'see log.');
        btn.innerHTML = '&#128229; Download selected';
        btn.disabled = false;
      } else if (evt.type === 'done') {
        es.close();
      }
    };
    es.onerror = function() {
      es.close();
      btn.disabled = false;
      btn.innerHTML = '&#128229; Download selected';
    };
  } catch (exc) {
    document.getElementById('eap-status').textContent = 'Failed: ' + exc.message;
    btn.innerHTML = '&#128229; Download selected';
    btn.disabled = false;
  }
}


// Populate the "Anchor from topic" dropdown on the Expand-citations panel
// with the current corpus's ranked topic clusters.
async function loadCorpusTopicList() {
  const sel = document.getElementById('tl-exp-relq-topic');
  if (!sel || sel.dataset.loaded) return;
  try {
    const res = await fetch('/api/catalog/topics');
    const data = await res.json();
    const topics = (data.topics || []).slice(0, 40);
    topics.forEach(t => {
      const opt = document.createElement('option');
      opt.value = t.name;
      opt.textContent = t.name + '  (' + t.n + ' papers)';
      sel.appendChild(opt);
    });
    sel.dataset.loaded = '1';
  } catch (_) {}
}

// ── Search tab ────────────────────────────────────────────────────────
async function doToolSearch(mode) {
  const q = document.getElementById('tl-search-q').value.trim();
  if (!q) return;
  const status = document.getElementById('tl-search-status');
  const results = document.getElementById('tl-search-results');
  status.textContent = (mode === 'similar' ? 'Finding similar papers...' : 'Searching...');
  results.innerHTML = '';

  const fd = new FormData();
  if (mode === 'similar') {
    fd.append('identifier', q);
    fd.append('top_k', document.getElementById('tl-search-topk').value || '10');
  } else {
    fd.append('q', q);
    fd.append('top_k', document.getElementById('tl-search-topk').value || '10');
    const yf = document.getElementById('tl-search-yfrom').value;
    const yt = document.getElementById('tl-search-yto').value;
    const sec = document.getElementById('tl-search-section').value;
    const tc = document.getElementById('tl-search-topic').value.trim();
    const ex = document.getElementById('tl-search-expand').checked;
    if (yf) fd.append('year_from', yf);
    if (yt) fd.append('year_to', yt);
    if (sec) fd.append('section', sec);
    if (tc) fd.append('topic', tc);
    if (ex) fd.append('expand', 'true');
  }

  try {
    const url = (mode === 'similar') ? '/api/search/similar' : '/api/search/query';
    const res = await fetch(url, {method: 'POST', body: fd});
    const data = await res.json();
    if (data.error) {
      status.textContent = data.error;
      return;
    }
    const hits = data.results || [];
    if (hits.length === 0) {
      status.textContent = 'No results.';
      return;
    }
    status.textContent = (mode === 'similar'
      ? 'Similar to: ' + ((data.query || {}).title || q)
      : hits.length + ' result' + (hits.length === 1 ? '' : 's'));
    let html = '<ol class="tl-search-list u-pl-20">';
    hits.forEach(h => {
      const authors = (h.authors || []).slice(0, 3).map(a => (a.name || '').split(/\s+/).slice(-1)[0]).filter(Boolean).join(', ');
      const year = h.year ? ' (' + h.year + ')' : '';
      const sec = h.section_type ? '<span class="u-accent u-tiny">[' + h.section_type + ']</span> ' : '';
      const score = (typeof h.score === 'number') ? ' <span class="u-hint">score=' + h.score.toFixed(3) + '</span>' : '';
      html += '<li class="u-mb-m">';
      html += sec + '<strong>' + (h.title || '(untitled)').replace(/</g, '&lt;') + '</strong>' + year + score;
      if (authors) html += '<div class="u-hint">' + authors + '</div>';
      if (h.doi) html += '<div class="u-tiny"><a href="https://doi.org/' + h.doi + '" target="_blank" rel="noopener">doi:' + h.doi + '</a></div>';
      if (h.preview) html += '<div class="u-muted u-small u-mt-2px">' + h.preview.replace(/</g, '&lt;') + '</div>';
      html += '</li>';
    });
    html += '</ol>';
    results.innerHTML = html;
  } catch (exc) {
    status.textContent = 'Error: ' + exc;
  }
}

// ── Synthesize tab (SSE, mirrors doAsk) ──────────────────────────────
async function doToolSynthesize() {
  const topic = document.getElementById('tl-synth-topic').value.trim();
  if (!topic) return;
  const status = document.getElementById('tl-synth-status');
  const stream = document.getElementById('tl-synth-stream');
  const sources = document.getElementById('tl-synth-sources');
  status.textContent = 'Retrieving and synthesising...';
  stream.textContent = '';
  sources.style.display = 'none';
  sources.innerHTML = '';

  const stats = createStreamStats('tl-synth-stats', 'qwen3.5:27b');
  stats.start();
  setStreamCursor(stream, true);

  const fd = new FormData();
  fd.append('topic', topic);
  fd.append('context_k', document.getElementById('tl-synth-k').value || '12');
  const yf = document.getElementById('tl-synth-yfrom').value;
  const yt = document.getElementById('tl-synth-yto').value;
  const tf = document.getElementById('tl-synth-topicfilter').value.trim();
  if (yf) fd.append('year_from', yf);
  if (yt) fd.append('year_to', yt);
  if (tf) fd.append('topic_filter', tf);

  const res = await fetch('/api/ask/synthesize', {method: 'POST', body: fd});
  const data = await res.json();
  _toolsSynthJob = data.job_id;
  const source = new EventSource('/api/stream/' + data.job_id);
  let collected = null;

  source.onmessage = function(e) {
    const evt = JSON.parse(e.data);
    if (evt.type === 'token') {
      setStreamCursor(stream, false);
      stream.textContent += evt.text;
      setStreamCursor(stream, true);
      stream.scrollTop = stream.scrollHeight;
      stats.update(evt.text);
    } else if (evt.type === 'model_info') {
      stats.setModel(evt.writer_model);
    } else if (evt.type === 'progress') {
      status.textContent = evt.detail || evt.stage;
    } else if (evt.type === 'sources') {
      collected = evt.sources;
      status.textContent = 'Synthesising from ' + (evt.n || evt.sources.length) + ' passages...';
    } else if (evt.type === 'completed') {
      status.textContent = 'Done';
      stats.done('done');
      setStreamCursor(stream, false);
      if (collected && collected.length) {
        let html = '<div class="u-semibold u-mb-6">Sources (' + collected.length + ')</div>';
        collected.forEach(s => { html += '<div class="src-item">' + s + '</div>'; });
        sources.innerHTML = html;
        sources.style.display = 'block';
      }
      source.close(); _toolsSynthJob = null;
    } else if (evt.type === 'error') {
      status.textContent = 'Error: ' + evt.message;
      stats.done('error');
      setStreamCursor(stream, false);
      source.close(); _toolsSynthJob = null;
    } else if (evt.type === 'done') {
      stats.done('done');
      setStreamCursor(stream, false);
      source.close(); _toolsSynthJob = null;
    }
  };
}

// ── Topics tab ───────────────────────────────────────────────────────
async function loadToolTopics() {
  const list = document.getElementById('tl-topics-list');
  const papers = document.getElementById('tl-topics-papers');
  const dlist = document.getElementById('tl-domains-list');
  list.innerHTML = '<span class="u-hint-sm">Loading…</span>';
  papers.innerHTML = '';
  if (dlist) dlist.innerHTML = '<span class="u-hint">Loading…</span>';
  try {
    // Load topics + domains in parallel
    const [topicsRes, domainsRes] = await Promise.all([
      fetch('/api/catalog/topics'),
      fetch('/api/catalog/domains?limit=80'),
    ]);
    const topics = (await topicsRes.json()).topics || [];
    const domains = (await domainsRes.json()).domains || [];

    if (topics.length === 0) {
      list.innerHTML = '<span class="u-hint-sm">No topic clusters assigned yet. Run <code>sciknow catalog cluster</code> to build them.</span>';
    } else {
      list.innerHTML = '';
      topics.forEach(t => {
        const btn = document.createElement('button');
        btn.textContent = t.name + ' (' + t.n + ')';
        btn.title = t.name + ' — ' + t.n + ' papers';
        btn.style.cssText = 'background:var(--bg-alt,#f3f4f6);border:1px solid var(--border);border-radius:12px;padding:4px 10px;font-size:12px;cursor:pointer;';
        btn.onclick = () => loadToolTopicPapers(t.name);
        list.appendChild(btn);
      });
    }

    if (dlist) {
      if (!domains.length) {
        dlist.innerHTML = '<span class="u-hint">No domain tags on this corpus.</span>';
      } else {
        dlist.innerHTML = '';
        domains.forEach(d => {
          const el = document.createElement('span');
          el.textContent = d.name + ' (' + d.n + ')';
          el.title = d.name + ' — ' + d.n + ' papers';
          el.style.cssText = 'background:var(--accent-light,#eef2ff);border:1px solid var(--border);border-radius:10px;padding:2px 8px;font-size:11px;';
          dlist.appendChild(el);
        });
      }
    }
  } catch (exc) {
    list.innerHTML = '<span class="u-danger u-small">Error: ' + exc + '</span>';
  }
}

async function loadToolTopicPapers(name) {
  const papers = document.getElementById('tl-topics-papers');
  papers.innerHTML = '<div class="u-p-3 u-muted">Loading papers in "' + name + '"…</div>';
  try {
    const res = await fetch('/api/catalog/topics?name=' + encodeURIComponent(name));
    const data = await res.json();
    const list = data.papers || [];
    if (list.length === 0) {
      papers.innerHTML = '<div class="u-p-3 u-muted">No papers in this cluster.</div>';
      return;
    }
    let html = '<h4 style="margin:4px 0 8px;">' + name.replace(/</g, '&lt;') + ' &mdash; ' + list.length + ' papers</h4>';
    html += '<ol class="u-pl-20">';
    list.forEach(p => {
      const year = p.year ? ' (' + p.year + ')' : '';
      const authors = (p.authors || []).slice(0, 3).map(a => (a.name || '').split(/\s+/).slice(-1)[0]).filter(Boolean).join(', ');
      html += '<li class="u-mb-6"><strong>' + (p.title || '(untitled)').replace(/</g, '&lt;') + '</strong>' + year;
      if (authors) html += '<div class="u-hint">' + authors + '</div>';
      if (p.doi) html += '<div class="u-tiny"><a href="https://doi.org/' + p.doi + '" target="_blank" rel="noopener">doi:' + p.doi + '</a></div>';
      html += '</li>';
    });
    html += '</ol>';
    papers.innerHTML = html;
  } catch (exc) {
    papers.innerHTML = '<div class="u-danger u-p-3">Error: ' + exc + '</div>';
  }
}

// ── Corpus tab (enrich / expand, subprocess SSE) ─────────────────────
// Phase 54.6.111 — shell to the CLI via /api/cli-stream (allowlist-gated)
// and stream stdout into the Corpus modal's log. Used by Retraction
// Sweep; generalizable to any corpus-wide one-shot command.
async function runCorpusCliAction(argv, startMsg) {
  const status = document.getElementById('tl-corpus-status');
  const logEl = document.getElementById('tl-corpus-log');
  if (logEl) logEl.textContent = '';
  if (status) status.textContent = startMsg || 'Working…';
  try {
    const res = await fetch('/api/cli-stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({argv: argv}),
    });
    const d = await res.json();
    if (!res.ok || !d.job_id) {
      if (status) status.textContent = 'Failed: ' + (d.detail || res.status);
      return;
    }
    const src = new EventSource('/api/stream/' + d.job_id);
    src.onmessage = function(e) {
      const evt = JSON.parse(e.data);
      if (evt.type === 'log') {
        if (logEl) { logEl.textContent += evt.text + '\n'; logEl.scrollTop = logEl.scrollHeight; }
      } else if (evt.type === 'completed') {
        if (status) status.innerHTML = '<span class="u-success">\u2713 Done.</span>';
        src.close();
      } else if (evt.type === 'error') {
        if (status) status.innerHTML = '<span class="u-danger">\u2717 ' + (evt.message || 'error') + '</span>';
        src.close();
      } else if (evt.type === 'done') {
        src.close();
      }
    };
  } catch (exc) {
    if (status) status.textContent = 'Request failed: ' + exc.message;
  }
}

async function doToolCorpus(action) {
  const status = document.getElementById('tl-corpus-status');
  const logEl = document.getElementById('tl-corpus-log');
  const cancelBtn = document.getElementById('tl-corpus-cancel');
  logEl.textContent = '';
  status.textContent = 'Starting ' + action + '...';
  cancelBtn.style.display = 'inline-block';

  const fd = new FormData();
  if (action === 'cleanup') {
    // Phase 54.6.4 — cleanup-downloads; the `cleanup-downloads` endpoint
    // takes dry_run / delete_dupes / cross_project. Defaults: delete,
    // cross-project on. No-parameter GUI button = "just clean it".
    const res0 = await fetch('/api/corpus/cleanup-downloads', {method: 'POST', body: new FormData()});
    const data0 = await res0.json();
    if (!res0.ok) {
      status.textContent = 'Cleanup failed to start: ' + (data0.detail || res0.status);
      cancelBtn.style.display = 'none';
      return;
    }
    _toolsCorpusJob = data0.job_id;
  } else if (action === 'enrich') {
    fd.append('limit', document.getElementById('tl-enr-limit').value || '0');
    fd.append('threshold', document.getElementById('tl-enr-thresh').value || '0.85');
    fd.append('dry_run', document.getElementById('tl-enr-dry').checked ? 'true' : 'false');
  } else if (action === 'expand-author') {
    // Phase 46.E — expand-by-author
    const nm = (window._selectedExpandAuthorName
                 || document.getElementById('tl-eauth-q').value || '').trim();
    if (!nm) {
      status.textContent = 'Pick an author from the list (or type a name and press Enter).';
      cancelBtn.style.display = 'none';
      return;
    }
    fd.append('name', nm);
    const orcid = document.getElementById('tl-eauth-orcid').value.trim();
    if (orcid) fd.append('orcid', orcid);
    fd.append('year_from', document.getElementById('tl-eauth-yfrom').value || '0');
    fd.append('year_to',   document.getElementById('tl-eauth-yto').value   || '0');
    fd.append('limit',     document.getElementById('tl-eauth-limit').value || '0');
    fd.append('workers',   document.getElementById('tl-eauth-workers').value || '0');
    fd.append('relevance_threshold', document.getElementById('tl-eauth-relthr').value || '0.0');
    fd.append('strict_author', document.getElementById('tl-eauth-strict').checked ? 'true' : 'false');
    fd.append('all_matches',   document.getElementById('tl-eauth-all').checked ? 'true' : 'false');
    fd.append('relevance',     document.getElementById('tl-eauth-relevance').checked ? 'true' : 'false');
    fd.append('ingest',        document.getElementById('tl-eauth-ingest').checked ? 'true' : 'false');
    fd.append('dry_run',       document.getElementById('tl-eauth-dry').checked ? 'true' : 'false');
    const rq = document.getElementById('tl-eauth-relq').value.trim();
    if (rq) fd.append('relevance_query', rq);
  } else {
    fd.append('limit', document.getElementById('tl-exp-limit').value || '0');
    // Phase 54.6.113 — RRF pool size per round. Pass-through to the
    // expand endpoint which forwards to `db expand --budget N`.
    fd.append('budget', document.getElementById('tl-exp-budget').value || '50');
    fd.append('workers', document.getElementById('tl-exp-workers').value || '0');
    fd.append('relevance_threshold', document.getElementById('tl-exp-relthr').value || '0.0');
    fd.append('dry_run', document.getElementById('tl-exp-dry').checked ? 'true' : 'false');
    fd.append('resolve', document.getElementById('tl-exp-resolve').checked ? 'true' : 'false');
    fd.append('ingest', document.getElementById('tl-exp-ingest').checked ? 'true' : 'false');
    fd.append('relevance', document.getElementById('tl-exp-relevance').checked ? 'true' : 'false');
    const rq = document.getElementById('tl-exp-relq').value.trim();
    if (rq) fd.append('relevance_query', rq);
  }

  if (action !== 'cleanup') {
    try {
      const res = await fetch('/api/corpus/' + action, {method: 'POST', body: fd});
      const data = await res.json();
      if (!res.ok) {
        status.textContent = 'Failed to start: ' + (data.detail || res.status);
        cancelBtn.style.display = 'none';
        return;
      }
      _toolsCorpusJob = data.job_id;
    } catch (exc) {
      status.textContent = 'Failed to start: ' + exc;
      cancelBtn.style.display = 'none';
      return;
    }
  }

  const source = new EventSource('/api/stream/' + _toolsCorpusJob);
  source.onmessage = function(e) {
    const evt = JSON.parse(e.data);
    if (evt.type === 'log') {
      logEl.textContent += evt.text + '\n';
      logEl.scrollTop = logEl.scrollHeight;
    } else if (evt.type === 'progress') {
      status.textContent = evt.detail || evt.stage;
      if (evt.detail && evt.detail.startsWith('$ ')) {
        logEl.textContent += evt.detail + '\n';
      }
    } else if (evt.type === 'completed') {
      status.textContent = action + ' finished.';
      cancelBtn.style.display = 'none';
      source.close(); _toolsCorpusJob = null;
    } else if (evt.type === 'error') {
      status.textContent = 'Error: ' + evt.message;
      cancelBtn.style.display = 'none';
      source.close(); _toolsCorpusJob = null;
    } else if (evt.type === 'done') {
      cancelBtn.style.display = 'none';
      source.close(); _toolsCorpusJob = null;
    }
  };
}

async function cancelToolCorpus() {
  if (!_toolsCorpusJob) return;
  try {
    await fetch('/api/jobs/' + _toolsCorpusJob, {method: 'DELETE'});
  } catch (exc) {}
}

// ── Phase 14: Catalog Browser modal ───────────────────────────────────
let catalogPage = 1;

function openCatalogModal() {
  openModal('catalog-modal');
  loadCatalog(1);
}

async function loadCatalog(page) {
  catalogPage = page || 1;
  const params = new URLSearchParams({page: catalogPage, per_page: 25});
  const author = document.getElementById('cat-author').value.trim();
  const journal = document.getElementById('cat-journal').value.trim();
  const yf = document.getElementById('cat-year-from').value;
  const yt = document.getElementById('cat-year-to').value;
  if (author) params.set('author', author);
  if (journal) params.set('journal', journal);
  if (yf) params.set('year_from', yf);
  if (yt) params.set('year_to', yt);

  const results = document.getElementById('catalog-results');
  results.innerHTML = '<div class="u-empty">Loading...</div>';

  try {
    const res = await fetch('/api/catalog?' + params.toString());
    const data = await res.json();
    if (!data.papers || data.papers.length === 0) {
      results.innerHTML = '<div class="u-empty">No papers match.</div>';
      return;
    }

    let html = '<table class="catalog-table"><thead><tr><th>Title</th><th>Year</th><th>Journal</th><th>Authors</th></tr></thead><tbody>';
    data.papers.forEach(p => {
      const authorStr = (p.authors || []).slice(0, 2).map(a => (a.name || '').split(/\s+/).slice(-1)[0]).filter(Boolean).join(', ') + (p.authors && p.authors.length > 2 ? ' et al.' : '');
      // Phase 42 — data-paper-title carries the raw title; askAboutPaper
      // reads it from the dataset.  Browsers escape data-* values, so no
      // more JSON.stringify + quote juggling.
      html += '<tr data-action="ask-about-paper" data-paper-title="' + escapeHtml(p.title || '') + '">';
      html += '<td><div class="ct-title">' + (p.title || '').replace(/</g, '&lt;') + '</div>';
      if (p.abstract) html += '<div class="ct-meta">' + p.abstract.substring(0, 160).replace(/</g, '&lt;') + '...</div>';
      html += '</td>';
      html += '<td>' + (p.year || '—') + '</td>';
      html += '<td>' + (p.journal || '—').substring(0, 30) + '</td>';
      html += '<td>' + authorStr + '</td>';
      html += '</tr>';
    });
    html += '</tbody></table>';

    html += '<div class="catalog-pager">';
    html += '<button data-action="load-catalog" data-page="' + (catalogPage - 1) + '" ' + (catalogPage <= 1 ? 'disabled' : '') + ' title="Previous page of papers.">‹ Prev</button>';
    html += '<span>Page ' + data.page + ' of ' + data.n_pages + '  ·  ' + data.total + ' papers</span>';
    html += '<button data-action="load-catalog" data-page="' + (catalogPage + 1) + '" ' + (catalogPage >= data.n_pages ? 'disabled' : '') + ' title="Next page of papers.">Next ›</button>';
    html += '</div>';

    results.innerHTML = html;
  } catch (e) {
    results.innerHTML = '<div class="u-empty-danger">Error: ' + e.message + '</div>';
  }
}

function askAboutPaper(title) {
  closeModal('catalog-modal');
  openAskModal();
  document.getElementById('ask-input').value = 'In the paper "' + title + '", ';
  document.getElementById('ask-input').focus();
}

// ── Phase 14.3 + Phase 21: Plans modal (book / chapter / section) ────
//
// Phase 21 — context-aware. The Plan toolbar button auto-detects what's
// selected and routes to the right tab:
//   - section selected → Section tab focused on that section
//   - chapter selected → Chapter tab showing the chapter's section plans
//   - nothing selected → Book tab (the leitmotiv)
//
// Phase 54.6.66 — three tabs, always visible:
//   Book      — the leitmotiv (always loaded)
//   Chapters  — book-wide chapter manager (title / description / order)
//   Sections  — per-chapter section plan editor, with a chapter picker
// The legacy "Section" (single-section focused editor) tab is gone;
// section-level editing happens inside the Sections tab, filtered to
// the picked chapter.

let _planContext = { mode: 'book', chapterId: null, sectionSlug: null };

async function openPlanModal(context) {
  // Default context: derive from current selection state.
  if (!context) {
    if (currentChapterId) {
      // Section context still useful: auto-scroll to the section in
      // the Sections tab so the user lands on the relevant row.
      context = { mode: currentSectionType ? 'section' : 'chapter',
                   chapterId: currentChapterId,
                   sectionSlug: currentSectionType || null };
    } else {
      context = { mode: 'book', chapterId: null, sectionSlug: null };
    }
  }
  _planContext = context;
  // Phase 54.6.14 → 54.6.96: elicitation-method picker now lives in the
  // Outline tab. Populated lazily on first Outline-tab switch (see
  // switchPlanTab). Cached via _populateMethodSelect's own caching.

  // Phase 32.2 — reset per-chapter editing state every time the modal
  // opens so slug collisions between chapters (e.g. "introduction"
  // appearing in every chapter) don't leak overrides from one chapter
  // into another.
  _editingChapterPlans = {};
  _editingChapterTargetWords = {};
  _editingChapterCustomMode = {};
  _editingChapterTitles = {};

  openModal('plan-modal');
  document.getElementById('plan-status').textContent = 'Loading...';

  // All three tabs always visible post-54.6.66; no contextual hiding.

  // Always populate the Book tab fields (cheap fetch + the user might
  // switch tabs back to it).
  try {
    const res = await fetch('/api/book');
    const data = await res.json();
    document.getElementById('plan-title-input').value = data.title || '';
    document.getElementById('plan-desc-input').value = data.description || '';
    document.getElementById('plan-text-input').value = data.plan || '';
    const tcw = data.target_chapter_words;
    const dflt = data.default_target_chapter_words || 6000;
    window._chapterWordTarget = tcw || dflt;
    const tcwInput = document.getElementById('plan-target-words-input');
    const lstatus = document.getElementById('plan-length-status');
    if (tcwInput) tcwInput.value = tcw ? String(tcw) : '';
    if (lstatus) {
      lstatus.textContent = tcw
        ? ('current: ' + tcw + ' words/chapter')
        : ('using default: ' + dflt + ' words/chapter');
    }
    if (!data.plan) {
      document.getElementById('plan-status').innerHTML =
        '<span class="u-warning">No book plan set yet.</span> Click <strong>Regenerate with LLM</strong> to draft one.';
    } else {
      document.getElementById('plan-status').textContent =
        data.plan.split(/\s+/).filter(Boolean).length + ' words in book plan';
    }
    // Phase 54.6.x — wire debounced autosave on the Book tab inputs
    // so edits persist 1.2 s after the last keystroke without
    // requiring a Save click.
    _wireBookTabAutosave();
    _setAutosaveStatus('plan-book', 'idle');
  } catch (e) {
    document.getElementById('plan-status').textContent = 'Error loading book: ' + e.message;
  }

  // Populate the Chapters tab always (book-wide view). Populate the
  // Sections tab's chapter picker from the same in-memory chaptersData
  // cache and, if the user arrived with a chapter context, preselect
  // it so the section list is ready when they switch to that tab.
  populatePlanChaptersTab();
  populatePlanSectionsPicker(context.chapterId);
  if (context.chapterId) {
    const ch = chaptersData.find(c => c.id === context.chapterId);
    if (ch) populatePlanChapterTab(ch);
  }

  // Landing tab: Sections if the user came from a chapter/section
  // view, Book otherwise.
  if (context.mode === 'section' || context.mode === 'chapter') {
    switchPlanTab('plan-chapter');
    // Auto-scroll to the specific section on next tick so the DOM
    // from populatePlanChapterTab has settled.
    if (context.sectionSlug) {
      setTimeout(function() {
        const row = document.querySelector(
          '#plan-chapter-sections .sec-row[data-slug="' +
          (context.sectionSlug || '').replace(/"/g, '\\"') + '"]'
        );
        if (row && row.scrollIntoView) {
          row.scrollIntoView({block: 'center', behavior: 'smooth'});
          row.style.outline = '2px solid var(--accent, dodgerblue)';
          setTimeout(function() { row.style.outline = ''; }, 1200);
        }
      }, 100);
    }
  } else {
    switchPlanTab('plan-book');
  }
}

function switchPlanTab(name) {
  document.querySelectorAll('#plan-modal .tab').forEach(t => {
    t.classList.toggle('active', t.dataset.tab === name);
  });
  const show = {
    'plan-book':     'plan-book-pane',
    'plan-outline':  'plan-outline-pane',
    'plan-chapters': 'plan-chapters-pane',
    'plan-chapter':  'plan-chapter-pane',   // ← "Sections" tab keeps old id
  };
  ['plan-book-pane', 'plan-outline-pane', 'plan-chapters-pane', 'plan-chapter-pane'].forEach(function(id) {
    const el = document.getElementById(id);
    if (el) el.style.display = (show[name] === id) ? 'block' : 'none';
  });
  // The Regenerate button only makes sense for the book leitmotiv.
  const regenBtn = document.getElementById('plan-regen-btn');
  if (regenBtn) regenBtn.style.display = (name === 'plan-book') ? '' : 'none';
  // When entering the Outline tab, lazy-populate the method picker
  // from /api/methods (same catalogue the footer used to read).
  if (name === 'plan-outline') _populateOutlineMethodPicker();
}

// ── Phase 54.6.x — Plan modal autosave ─────────────────────────────────
// One debounced timer per "scope key" (plan-book, plan-chapters,
// plan-chapter-row:<cid>, …). Status indicator slot is `#autosave-<key>`
// so each tab can show its own state without colliding with the
// modal's footer status line. Save Now buttons just call
// `flushAutosave(key)` to skip the debounce.
const _planAutosaveTimers = {};
const _planAutosaveStatus = {};

function _setAutosaveStatus(key, state, detail) {
  _planAutosaveStatus[key] = state;
  const el = document.getElementById('autosave-' + key);
  if (!el) return;
  if (state === 'pending') {
    el.innerHTML = '<span class="u-muted">⏳ unsaved edits…</span>';
  } else if (state === 'saving') {
    el.innerHTML = '<span class="u-muted">… saving</span>';
  } else if (state === 'saved') {
    const stamp = new Date();
    const hh = String(stamp.getHours()).padStart(2, '0');
    const mm = String(stamp.getMinutes()).padStart(2, '0');
    const ss = String(stamp.getSeconds()).padStart(2, '0');
    el.innerHTML = '<span class="u-success">✓ saved at ' + hh + ':' + mm + ':' + ss + '</span>';
  } else if (state === 'error') {
    el.innerHTML = '<span class="u-danger">✗ ' + (detail || 'save failed — click Save now to retry') + '</span>';
  } else {
    el.innerHTML = '<span class="u-muted">○ Idle</span>';
  }
}

function scheduleAutosave(key, delayMs, fn) {
  if (_planAutosaveTimers[key]) clearTimeout(_planAutosaveTimers[key]);
  _setAutosaveStatus(key, 'pending');
  _planAutosaveTimers[key] = setTimeout(async () => {
    _planAutosaveTimers[key] = null;
    _setAutosaveStatus(key, 'saving');
    try {
      await fn();
      _setAutosaveStatus(key, 'saved');
    } catch (e) {
      _setAutosaveStatus(key, 'error', e && e.message);
    }
  }, delayMs);
}

function flushAutosave(key) {
  // Cancels the debounce — caller is expected to invoke the save fn
  // synchronously (e.g. via a Save Now button that calls savePlanBook(true)).
  if (_planAutosaveTimers[key]) {
    clearTimeout(_planAutosaveTimers[key]);
    _planAutosaveTimers[key] = null;
  }
}

// Wire input/change events for the Book tab inputs to the autosave
// helper. Idempotent: each input gets a `dataset.autosaveWired` flag
// so re-running on tab re-entry doesn't double-bind.
function _wireBookTabAutosave() {
  const ids = [
    'plan-title-input',
    'plan-desc-input',
    'plan-text-input',
    'plan-target-words-input',
  ];
  ids.forEach(function(id) {
    const el = document.getElementById(id);
    if (!el || el.dataset.autosaveWired === '1') return;
    el.dataset.autosaveWired = '1';
    const handler = function() {
      scheduleAutosave('plan-book', 1200, function() {
        return savePlanBook(false);
      });
    };
    el.addEventListener('input', handler);
    el.addEventListener('change', handler);
  });
}

// Wire input/change events for every chapter row in the Chapters tab.
// Saves are debounced per-row (key includes cid) so editing two
// chapters in succession doesn't lose the first one.
function _wireChaptersTabAutosave() {
  const rows = document.querySelectorAll('#plan-chapters-list .plan-ch-row');
  rows.forEach(function(row) {
    const cid = row.dataset.cid;
    if (!cid || row.dataset.autosaveWired === '1') return;
    row.dataset.autosaveWired = '1';
    const inputs = row.querySelectorAll(
      '.plan-ch-title, .plan-ch-desc, .plan-ch-tq, .plan-ch-target, .plan-ch-flex'
    );
    inputs.forEach(function(inp) {
      const handler = function() {
        // Reflect into the tab-level pill too so the user sees the
        // global state at the top, in addition to the row-level Save.
        scheduleAutosave('plan-chapters', 1000, async function() {
          await savePlanChapterRow(cid);
        });
      };
      inp.addEventListener('input', handler);
      inp.addEventListener('change', handler);
    });
  });
}

// Phase 54.6.x — wire autosave on the Book Settings modal. Two scopes:
// basics (title/description/target_chapter_words/book_type) → save
// runs `saveBookSettings('basics')` 1.2 s after the last keystroke.
// leitmotiv (the bs-plan textarea) → `saveBookSettings('leitmotiv')`
// 1.5 s after the last keystroke (longer debounce for the long-form
// plan textarea so the user's typing pauses don't churn save calls).
function _wireBookSettingsAutosave() {
  const basics = ['bs-title', 'bs-description', 'bs-target-chapter-words', 'bs-book-type'];
  basics.forEach(function(id) {
    const el = document.getElementById(id);
    if (!el || el.dataset.autosaveWired === '1') return;
    el.dataset.autosaveWired = '1';
    const handler = function() {
      scheduleAutosave('bs-basics', 1200, async function() {
        await saveBookSettings('basics');
      });
    };
    el.addEventListener('input', handler);
    el.addEventListener('change', handler);
  });
  const plan = document.getElementById('bs-plan');
  if (plan && plan.dataset.autosaveWired !== '1') {
    plan.dataset.autosaveWired = '1';
    plan.addEventListener('input', function() {
      scheduleAutosave('bs-leitmotiv', 1500, async function() {
        await saveBookSettings('leitmotiv');
      });
    });
  }
}

// Phase 54.6.x — autosave the wiki "My take" annotation.
function _wireWikiAnnotationAutosave() {
  const body = document.getElementById('wiki-annotation-body');
  if (!body || body.dataset.autosaveWired === '1') return;
  body.dataset.autosaveWired = '1';
  body.addEventListener('input', function() {
    scheduleAutosave('wiki-mytake', 1500, async function() {
      await saveWikiAnnotation();
    });
  });
}

// Phase 54.6.x — autosave the Plan modal's Sections tab (per-chapter
// section plans + per-section target_words). Existing flow already
// captures edits into _editingChapterPlans / _editingChapterTargetWords;
// debounce-call savePlanChapterSections() so those buffers flush
// without the user clicking Save.
function _wirePlanSectionsAutosave() {
  const pane = document.getElementById('plan-chapter-sections');
  if (!pane || pane.dataset.autosaveWired === '1') return;
  pane.dataset.autosaveWired = '1';
  // Delegated listener — the rows are re-rendered when chapter
  // changes, so per-row binding would need re-wiring on every
  // populatePlanChapterTab call. Capture all input/change events
  // bubbling out of the pane.
  const handler = function() {
    scheduleAutosave('plan-sections', 1200, async function() {
      await savePlanChapterSections();
    });
  };
  pane.addEventListener('input', handler);
  pane.addEventListener('change', handler);
}

// "Save all chapters now" button at the top of the Chapters tab.
async function savePlanAllChapters() {
  flushAutosave('plan-chapters');
  const rows = document.querySelectorAll('#plan-chapters-list .plan-ch-row');
  if (!rows.length) return;
  _setAutosaveStatus('plan-chapters', 'saving');
  try {
    for (const row of rows) {
      const cid = row.dataset.cid;
      if (cid) await savePlanChapterRow(cid);
    }
    _setAutosaveStatus('plan-chapters', 'saved');
  } catch (e) {
    _setAutosaveStatus('plan-chapters', 'error', e && e.message);
  }
}

// ── Phase 54.6.66 — Chapters tab: book-wide chapter manager ────────────

function populatePlanChaptersTab() {
  const list = document.getElementById('plan-chapters-list');
  if (!list) return;
  const chapters = (chaptersData || []).slice().sort(
    function(a, b) { return (a.num || 0) - (b.num || 0); }
  );
  if (chapters.length === 0) {
    list.innerHTML =
      '<div style="padding:16px;text-align:center;color:var(--fg-muted);'
      + 'border:1px dashed var(--border);border-radius:6px;">'
      + 'No chapters yet. Click <strong>Generate outline</strong> below '
      + 'to draft chapters from your corpus, or + Add chapter to create '
      + 'one manually.'
      + '</div>';
    return;
  }
  let html = '';
  chapters.forEach(function(ch, i) {
    const cid = ch.id;
    const isFirst = (i === 0);
    const isLast = (i === chapters.length - 1);
    html += '<div class="plan-ch-row u-card" data-cid="' + cid + '" '
         +  '>'
         +  '<div class="u-flex-raw u-ai-center u-gap-2 u-mb-6">'
         +  '<span style="font-weight:700;font-size:12px;color:var(--fg-muted);'
         +  'min-width:45px;">Ch.' + (ch.num || '?') + '</span>'
         +  '<input type="text" class="plan-ch-title u-flex-1 u-pill-md u-semibold" value="' + escapeHtml(ch.title || '')
         +  '" placeholder="Chapter title" '
         +  '>'
         +  '<button class="btn-secondary u-pill" title="Move up" '
         +  'onclick="movePlanChapter(\'' + cid + '\', -1)" '
         +  (isFirst ? 'disabled' : '') + '>&uarr;</button>'
         +  '<button class="btn-secondary u-pill" title="Move down" '
         +  'onclick="movePlanChapter(\'' + cid + '\', 1)" '
         +  (isLast ? 'disabled' : '') + '>&darr;</button>'
         +  '<button class="btn-secondary u-p-2-10" title="Save row" '
         +  'onclick="savePlanChapterRow(\'' + cid + '\')" '
         +  '>Save</button>'
         +  '<button class="btn-secondary u-pill u-danger" title="Delete chapter" '
         +  'onclick="deletePlanChapter(\'' + cid + '\')" '
         +  '>&times;</button>'
         +  '</div>'
         +  '<textarea class="plan-ch-desc u-w-full u-pill-md u-mb-6 u-small" rows="2" '
         +  'placeholder="Short description (1-2 sentences)" '
         +  '>'
         +  escapeHtml(ch.description || '') + '</textarea>'
         +  '<div class="u-flex-raw u-gap-6 u-ai-center u-mb-6">'
         +  '<label style="font-size:11px;color:var(--fg-muted);min-width:80px;">Topic query:</label>'
         +  '<input type="text" class="plan-ch-tq" value="' + escapeHtml(ch.topic_query || '')
         +  '" placeholder="3-6 word retrieval phrase" '
         +  'style="flex:1;padding:3px 8px;font-size:12px;">'
         +  '</div>'
         // Phase 54.6.x — per-chapter length controls. target_words
         // overrides the book-level chapter target; flexible_length
         // lets autowrite extend up to 2× when retrieval is rich.
         +  '<div class="u-flex-raw u-gap-6 u-ai-center u-wrap">'
         +  '<label style="font-size:11px;color:var(--fg-muted);min-width:80px;" '
         +  'title="Per-chapter length target (words). Empty = inherit the book-level target. Set to 0 to clear an existing override.">Target words:</label>'
         +  '<input type="number" class="plan-ch-target" min="0" step="500" '
         +  'value="' + (ch.target_words ? String(ch.target_words) : '') + '" '
         +  'placeholder="(book default)" '
         +  'style="width:120px;padding:3px 8px;font-size:12px;" '
         +  'title="Per-chapter length target. Empty = inherit book-level target. 0 clears the override.">'
         +  '<label class="u-chip" '
         +  'title="Phase 54.6.x — when checked, autowrite may extend up to 2× the target if the section\'s retrieval pool is rich (≥24 chunks). Only ever bigger, never smaller. The base target is still the scoring anchor.">'
         +  '<input type="checkbox" class="plan-ch-flex" '
         +  (ch.flexible_length ? 'checked ' : '')
         +  '> flexible (≤2× if corpus supports it)'
         +  '</label>'
         +  '</div>'
         +  '</div>';
  });
  list.innerHTML = html;
  // Phase 54.6.x — wire debounced autosave on every chapter row so
  // edits persist 1 s after the last keystroke without needing a
  // per-row Save click. Idempotent on re-render via row.dataset.autosaveWired.
  _wireChaptersTabAutosave();
  _setAutosaveStatus('plan-chapters', 'idle');
}

async function savePlanChapterRow(cid) {
  const row = document.querySelector('.plan-ch-row[data-cid="' + cid + '"]');
  if (!row) return;
  const title = row.querySelector('.plan-ch-title').value.trim();
  const desc = row.querySelector('.plan-ch-desc').value.trim();
  const tq = row.querySelector('.plan-ch-tq').value.trim();
  // Phase 54.6.x — per-chapter length controls.
  const targetEl = row.querySelector('.plan-ch-target');
  const flexEl = row.querySelector('.plan-ch-flex');
  const fd = new FormData();
  fd.append('title', title);
  fd.append('description', desc);
  fd.append('topic_query', tq);
  if (targetEl) {
    const tw = parseInt(targetEl.value || '0', 10);
    fd.append('target_words', isNaN(tw) ? 0 : tw);
  }
  if (flexEl) {
    fd.append('flexible_length', flexEl.checked ? 'true' : 'false');
  }
  const status = document.getElementById('plan-status');
  status.textContent = 'Saving chapter…';
  try {
    const res = await fetch('/api/chapters/' + cid, {method: 'PUT', body: fd});
    if (!res.ok) throw new Error('HTTP ' + res.status);
    status.textContent = '✓ Saved';
    // Update the in-memory cache so the Sections picker + landing
    // pages reflect the rename immediately.
    const ch = (chaptersData || []).find(function(c) { return c.id === cid; });
    if (ch) {
      ch.title = title;
      ch.description = desc;
      ch.topic_query = tq;
      if (targetEl) {
        const tw = parseInt(targetEl.value || '0', 10);
        ch.target_words = (!isNaN(tw) && tw > 0) ? tw : null;
      }
      if (flexEl) ch.flexible_length = !!flexEl.checked;
    }
    populatePlanSectionsPicker(
      document.getElementById('plan-sections-chapter-picker').value || null
    );
  } catch (e) {
    status.textContent = 'Save failed: ' + e.message;
  }
}

async function movePlanChapter(cid, delta) {
  const chapters = (chaptersData || []).slice().sort(
    function(a, b) { return (a.num || 0) - (b.num || 0); }
  );
  const idx = chapters.findIndex(function(c) { return c.id === cid; });
  const target = idx + delta;
  if (idx < 0 || target < 0 || target >= chapters.length) return;
  // Swap in-array, then POST the new order to the server.
  const tmp = chapters[idx];
  chapters[idx] = chapters[target];
  chapters[target] = tmp;
  const order = chapters.map(function(c) { return c.id; });
  const status = document.getElementById('plan-status');
  status.textContent = 'Reordering…';
  try {
    const res = await fetch('/api/chapters/reorder', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({chapter_ids: order}),
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    // Reflect new numbers client-side without a full refetch.
    chapters.forEach(function(c, i) { c.num = i + 1; });
    window.chaptersData = chapters;
    populatePlanChaptersTab();
    populatePlanSectionsPicker(
      document.getElementById('plan-sections-chapter-picker').value || null
    );
    status.textContent = '✓ Reordered';
  } catch (e) {
    status.textContent = 'Reorder failed: ' + e.message;
  }
}

async function deletePlanChapter(cid) {
  const ch = (chaptersData || []).find(function(c) { return c.id === cid; });
  const label = ch ? ('Ch.' + ch.num + ' "' + (ch.title || '') + '"') : 'this chapter';
  if (!confirm('Delete ' + label + '? Drafts will be unlinked but not deleted.')) return;
  const status = document.getElementById('plan-status');
  status.textContent = 'Deleting…';
  try {
    const res = await fetch('/api/chapters/' + cid, {method: 'DELETE'});
    if (!res.ok) throw new Error('HTTP ' + res.status);
    // Refresh local chaptersData cache
    const res2 = await fetch('/api/chapters');
    const data = await res2.json();
    window.chaptersData = data.chapters || [];
    populatePlanChaptersTab();
    populatePlanSectionsPicker(null);
    status.textContent = '✓ Deleted';
  } catch (e) {
    status.textContent = 'Delete failed: ' + e.message;
  }
}

async function addPlanChapter() {
  const title = prompt('Chapter title:');
  if (!title || !title.trim()) return;
  const fd = new FormData();
  fd.append('title', title.trim());
  const status = document.getElementById('plan-status');
  status.textContent = 'Adding chapter…';
  try {
    const res = await fetch('/api/chapters', {method: 'POST', body: fd});
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const res2 = await fetch('/api/chapters');
    const data = await res2.json();
    window.chaptersData = data.chapters || [];
    populatePlanChaptersTab();
    populatePlanSectionsPicker(null);
    status.textContent = '✓ Added';
  } catch (e) {
    status.textContent = 'Add failed: ' + e.message;
  }
}

// ── Sections tab — chapter picker wiring (Phase 54.6.66) ──────────────

function populatePlanSectionsPicker(preselectId) {
  const picker = document.getElementById('plan-sections-chapter-picker');
  if (!picker) return;
  const chapters = (chaptersData || []).slice().sort(
    function(a, b) { return (a.num || 0) - (b.num || 0); }
  );
  const current = preselectId || picker.value || (chapters[0] && chapters[0].id) || '';
  let html = '<option value="">(choose a chapter…)</option>';
  chapters.forEach(function(c) {
    const sel = (c.id === current) ? ' selected' : '';
    html += '<option value="' + c.id + '"' + sel + '>'
         +  'Ch.' + (c.num || '?') + ' — ' + escapeHtml(c.title || '')
         +  '</option>';
  });
  picker.innerHTML = html;
  if (current) {
    const ch = chapters.find(function(c) { return c.id === current; });
    if (ch) populatePlanChapterTab(ch);
  }
}

// Phase 54.6.163 — Plans modal auto-plan button. Uses the same
// /api/chapters/{id}/plan-sections endpoint as the Chapter modal's
// 54.6.155 button, just driven from the Plans modal's chapter picker.
async function planModalAutoPlanSections() {
  const picker = document.getElementById('plan-sections-chapter-picker');
  const status = document.getElementById('plan-auto-plan-status');
  const force  = document.getElementById('plan-auto-plan-force').checked;
  if (!picker || !picker.value) {
    status.innerHTML = '<span class="u-warning">Pick a chapter first.</span>';
    return;
  }
  const chId = picker.value;
  status.innerHTML = '<em>Generating plans… (~5-10s per empty section)</em>';
  try {
    const fd = new FormData();
    if (force) fd.append('force', 'true');
    const res = await fetch('/api/chapters/' + chId + '/plan-sections', {
      method: 'POST', body: fd,
    });
    if (!res.ok) {
      const t = await res.text();
      status.innerHTML = '<span class="u-danger">Failed: '
                        + _escHtml(t.slice(0, 200)) + '</span>';
      return;
    }
    const data = await res.json();
    status.innerHTML = '<span class="u-success">✓ planned '
                     + data.n_planned + ' · skipped '
                     + (data.n_skipped || 0) + '.</span>  '
                     + '<span class="u-muted">'
                     + 'Reloading sections…</span>';
    // Invalidate resolver cache so the Chapter modal (if reopened) shows
    // the new concept-density badges. Then reload this chapter in the
    // Plans modal Sections tab.
    if (window._resolvedTargetsByChapter) {
      delete window._resolvedTargetsByChapter[chId];
    }
    onPlanSectionsChapterChange(chId);
  } catch (e) {
    status.innerHTML = '<span class="u-danger">Error: '
                      + _escHtml(String(e).slice(0, 200)) + '</span>';
  }
}

function onPlanSectionsChapterChange(cid) {
  if (!cid) {
    document.getElementById('plan-chapter-header').innerHTML = '';
    document.getElementById('plan-chapter-sections').innerHTML = '';
    return;
  }
  const ch = (chaptersData || []).find(function(c) { return c.id === cid; });
  if (ch) {
    // Per-chapter editing state is per-slug; reset so we don't leak
    // overrides from one chapter into another's dropdown defaults.
    _editingChapterPlans = {};
    _editingChapterTargetWords = {};
    _editingChapterCustomMode = {};
    populatePlanChapterTab(ch);
    _planContext = {
      mode: 'chapter', chapterId: cid, sectionSlug: null,
    };
  }
}

// Phase 21 — render the chapter sections view inside the Plan modal.
// Each section gets a title input + plan textarea, just like the
// chapter modal's Sections tab — but read-only of the title (you
// rename via the chapter modal), editable for the plan.
// Phase 32.2 — every section now also gets a per-section length
// dropdown so the user can pick "Long" / "Custom" / etc directly
// inside the Plan modal without having to switch over to the
// chapter modal Sections tab. Save round-trips through the same
// PUT /api/chapters/{id}/sections endpoint, which already accepts
// target_words as part of each section dict.
function populatePlanChapterTab(ch) {
  const header = document.getElementById('plan-chapter-header');
  if (header) {
    header.innerHTML = '<strong>Ch.' + ch.num + ': ' + escapeHtml(ch.title) + '</strong>' +
      (ch.description ? '<br><span class="u-small">' + escapeHtml(ch.description.substring(0, 200)) + (ch.description.length > 200 ? '\u2026' : '') + '</span>' : '');
  }
  const list = document.getElementById('plan-chapter-sections');
  if (!list) return;
  const meta = Array.isArray(ch.sections_meta) ? ch.sections_meta : [];
  if (meta.length === 0) {
    list.innerHTML = '<div style="font-size:12px;color:var(--fg-muted);padding:12px;text-align:center;border:1px dashed var(--border);border-radius:4px;">This chapter has no sections defined. Open the chapter modal (\u2699 icon) and use the Sections tab to add some.</div>';
    return;
  }

  // Compute the per-section auto budget so the dropdown's "Auto"
  // option can show what the writer would aim for if no override
  // is set. Mirrors the logic in renderSectionEditor.
  const chapterTarget = (window._chapterWordTarget && window._chapterWordTarget > 0)
    ? window._chapterWordTarget
    : 6000;
  const nSec = Math.max(1, meta.length);
  const perSection = Math.max(400, Math.min(chapterTarget, Math.floor(chapterTarget / nSec)));

  // Seed the editing state with current target_words from meta so
  // the dropdown reflects whatever was previously saved.
  meta.forEach(s => {
    if (!(s.slug in _editingChapterTargetWords)) {
      _editingChapterTargetWords[s.slug] = (s.target_words && s.target_words > 0)
        ? s.target_words : null;
    }
  });

  let html = '<div class="u-tiny u-muted u-mb-2 u-p-6-10 u-bg-tb u-r-sm">';
  html += '<strong>' + meta.length + '</strong> section' + (meta.length === 1 ? '' : 's') +
          ' &middot; chapter target: <strong>' + chapterTarget + '</strong> words &middot; ' +
          'auto per section: <strong>~' + perSection + '</strong> words';
  html += '</div>';

  meta.forEach((s, i) => {
    const tw = _editingChapterTargetWords[s.slug];
    const isAuto = !tw || tw <= 0;
    const presets = [400, 800, 1500, 3000, 6000];
    const isCustomMode = !!_editingChapterCustomMode[s.slug];
    const presetMatch = !isAuto && presets.includes(tw);
    const isCustom = isCustomMode || (!isAuto && !presetMatch);
    let optsHtml = '<option value="">Auto (~' + perSection + 'w)</option>';
    presets.forEach(p => {
      const sel = (!isCustom && tw === p) ? ' selected' : '';
      const labelMap = {400: 'Very short', 800: 'Short', 1500: 'Medium', 3000: 'Long', 6000: 'Extra long'};
      optsHtml += '<option value="' + p + '"' + sel + '>' + labelMap[p] + ' (~' + p + 'w)</option>';
    });
    optsHtml += '<option value="custom"' + (isCustom ? ' selected' : '') + '>Custom\u2026</option>';
    const customStyle = isCustom ? '' : 'display:none;';
    const customVal = (isCustom && tw) ? String(tw) : '';
    const effectiveTw = (tw && tw > 0) ? tw : perSection;
    const badgeClass = (tw && tw > 0) ? 'sec-target-badge override' : 'sec-target-badge';
    const badgeTag = (tw && tw > 0)
      ? '<span class="badge-tag">override</span>'
      : '<span class="badge-tag muted">auto</span>';

    // Phase 54.6.67 — per-row toolbar (↑ ↓ Save ×) plus editable title.
    // Title edits feed _editingChapterTitles; slug stays stable so
    // drafts.section_type doesn't orphan on rename.
    const isFirst = (i === 0);
    const isLast = (i === meta.length - 1);
    const liveTitle = (s.slug in _editingChapterTitles)
      ? _editingChapterTitles[s.slug]
      : (s.title || _titleifyClient(s.slug));
    html += '<div class="sec-row" data-slug="' + s.slug + '">';
    html += '  <div class="sec-fields">';
    html += '    <div class="u-flex-raw u-ai-center u-gap-6 u-mb-6">';
    html += '      <span style="font-weight:700;font-size:12px;color:var(--fg-muted);min-width:24px;">' +
            (i + 1) + '.</span>';
    html += '      <input type="text" class="plan-sec-title u-flex-1 u-pill-md u-semibold u-md" ' +
            'data-title-slug="' + s.slug + '" ' +
            'value="' + escapeHtml(liveTitle) + '" placeholder="Section title" ' +
            'oninput="updatePlanChapterTitle(\'' + s.slug + '\', this.value)" ' +
            '>';
    html += '      <button class="btn-secondary u-pill" title="Move up" ' +
            'onclick="movePlanSection(\'' + s.slug + '\', -1)" ' +
            (isFirst ? 'disabled' : '') + '>&uarr;</button>';
    html += '      <button class="btn-secondary u-pill" title="Move down" ' +
            'onclick="movePlanSection(\'' + s.slug + '\', 1)" ' +
            (isLast ? 'disabled' : '') + '>&darr;</button>';
    html += '      <button class="btn-secondary u-p-2-10" title="Save this row" ' +
            'onclick="savePlanSectionRow(\'' + s.slug + '\')" ' +
            '>Save</button>';
    html += '      <button class="btn-secondary u-pill u-danger" title="Delete section" ' +
            'onclick="deletePlanSection(\'' + s.slug + '\')" ' +
            '>&times;</button>';
    html += '    </div>';
    // Phase 54.6.163 — live concept-count readout wired into the
    // Plans-modal textareas (previously only present in the
    // Chapter-modal via 54.6.152). Uses the slug-keyed variant of
    // the shared renderer.
    html += '    <textarea data-plan-slug="' + s.slug + '" placeholder="Section plan — what THIS section must cover (bullet one concept per line for concept-density sizing)" ' +
            'oninput="updatePlanChapterSection(\'' + s.slug + '\', this.value); updatePlanConceptReadoutBySlug(\'' + s.slug + '\', this);">' +
            escapeHtml(s.plan || '') + '</textarea>';
    html += '    <div id="plan-readout-slug-' + s.slug + '" class="plan-concept-readout u-indent-sm" '
         + '></div>';
    html += '    <div class="sec-size-row">';
    html += '      <label>Target:</label>';
    html += '      <select onchange="updatePlanChapterTargetWords(\'' + s.slug + '\', this.value)" title="Pick a preset word target for this section. Choose Custom to enter an exact number in the box on the right.">' + optsHtml + '</select>';
    html += '      <input type="number" class="sec-size-custom" placeholder="words" min="100" step="100" ';
    html += '             value="' + customVal + '" style="' + customStyle + '" ';
    html += '             oninput="updatePlanChapterTargetWordsCustom(\'' + s.slug + '\', this.value)" title="Custom target word count. Visible only when the preset dropdown is set to Custom.">';
    html += '      <span class="' + badgeClass + '">~' + effectiveTw + ' words ' + badgeTag + '</span>';
    html += '    </div>';
    html += '    <div class="sec-slug">slug: <code>' + s.slug + '</code></div>';
    html += '  </div>';
    html += '</div>';
  });
  // Phase 54.6.67 — + Add section button below the list.
  html += '<div class="u-mt-3">'
       +  '<button class="btn-secondary" onclick="addPlanSection()" title="Append a new empty section to the end of this chapter. You can rename and reorder after adding.">+ Add section</button>'
       +  '</div>';
  list.innerHTML = html;
  // Phase 54.6.163 — populate the concept-count readouts after render
  // so users see "3 concepts × 650 wpc = ~1,950 words" before typing.
  // Mirrors the pattern from 54.6.152 in renderSectionEditor.
  list.querySelectorAll('textarea[data-plan-slug]').forEach(ta => {
    updatePlanConceptReadoutBySlug(ta.dataset.planSlug, ta);
  });
  // Cache warm: if _swBookTypes or _currentBookType isn't loaded yet,
  // fetch and re-render so wpc midpoint is correct per project type.
  if (!window._currentBookType || !window._swBookTypes) {
    Promise.all([
      fetch('/api/book').then(r => r.json()).catch(() => ({})),
      (window._swBookTypes ? Promise.resolve(null) : swLoadBookTypes()),
    ]).then(([bookData]) => {
      if (bookData && bookData.book_type) {
        window._currentBookType = bookData.book_type;
      }
      list.querySelectorAll('textarea[data-plan-slug]').forEach(ta => {
        updatePlanConceptReadoutBySlug(ta.dataset.planSlug, ta);
      });
    }).catch(e => console.debug('plan-modal readout cache warm failed:', e));
  }
  // Phase 54.6.x — wire debounced autosave on every section editor
  // input. Idempotent (delegated listener; pane has dataset guard).
  _wirePlanSectionsAutosave();
}

// Track plan edits for the chapter tab so save can collect them.
let _editingChapterPlans = {};
// Phase 32.2 — parallel maps for per-section target_words overrides
// edited in the Plan modal's Chapter sections tab. Keyed by slug.
// Reset whenever a different chapter is loaded into the tab.
let _editingChapterTargetWords = {};
let _editingChapterCustomMode = {};
// Phase 54.6.67 — title edits tracked separately so re-renders don't
// drop in-flight changes. Keyed by slug (slug is stable across rename).
let _editingChapterTitles = {};

function _titleifyClient(slug) {
  // Mirror of core.book_ops._titleify_slug for the display-only
  // default when a section has no title set.
  return (slug || '').replace(/_/g, ' ').trim()
    .split(' ')
    .map(function(w) { return w ? (w[0].toUpperCase() + w.slice(1)) : ''; })
    .join(' ');
}

function updatePlanChapterSection(slug, value) {
  _editingChapterPlans[slug] = value;
}
function updatePlanChapterTitle(slug, value) {
  _editingChapterTitles[slug] = value;
}
function updatePlanChapterTargetWords(slug, value) {
  if (value === "" || value === "auto") {
    _editingChapterTargetWords[slug] = null;
    _editingChapterCustomMode[slug] = false;
  } else if (value === "custom") {
    _editingChapterCustomMode[slug] = true;
    if (!_editingChapterTargetWords[slug]) _editingChapterTargetWords[slug] = 1500;
  } else {
    const n = parseInt(value, 10);
    _editingChapterTargetWords[slug] = isNaN(n) ? null : n;
    _editingChapterCustomMode[slug] = false;
  }
  // Re-render the tab so the badge/custom-input visibility updates.
  const ch = chaptersData.find(c => c.id === _planContext.chapterId);
  if (ch) populatePlanChapterTab(ch);
}
function updatePlanChapterTargetWordsCustom(slug, value) {
  const n = parseInt(value, 10);
  _editingChapterTargetWords[slug] = (isNaN(n) || n <= 0) ? null : n;
  _editingChapterCustomMode[slug] = true;
  // Don't re-render — the user is actively typing in the custom input.
}

// ── Phase 54.6.67 — per-row section handlers (move/save/delete/add) ───
// All four go through PUT /api/chapters/{id}/sections, which is a
// full-replace endpoint. Each handler materializes the chapter's
// current sections (with pending edits folded in), applies the
// mutation, and re-submits the full list.

function _materializePlanSections(ch) {
  // Merge in-memory meta with the user's pending edits (title / plan
  // / target_words) so reorder/delete don't lose unsaved work.
  const meta = Array.isArray(ch.sections_meta) ? ch.sections_meta : [];
  return meta.map(function(s) {
    const tw = (s.slug in _editingChapterTargetWords)
      ? _editingChapterTargetWords[s.slug]
      : (s.target_words || null);
    return {
      slug: s.slug,
      title: (s.slug in _editingChapterTitles)
        ? _editingChapterTitles[s.slug]
        : (s.title || _titleifyClient(s.slug)),
      plan: (s.slug in _editingChapterPlans)
        ? _editingChapterPlans[s.slug]
        : (s.plan || ''),
      target_words: (tw && tw > 0) ? tw : null,
    };
  });
}

async function _putPlanSections(cid, sections, statusMsg) {
  const status = document.getElementById('plan-status');
  if (status) status.textContent = statusMsg || 'Saving sections…';
  const res = await fetch('/api/chapters/' + cid + '/sections', {
    method: 'PUT',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({sections: sections}),
  });
  if (!res.ok) throw new Error('HTTP ' + res.status);
  const data = await res.json();
  // Update the in-memory cache + re-render.
  const ch = chaptersData.find(function(c) { return c.id === cid; });
  if (ch && Array.isArray(data.sections)) {
    ch.sections_meta = data.sections;
    ch.sections_template = data.sections.map(function(s) { return s.slug; });
  }
  _editingChapterPlans = {};
  _editingChapterTargetWords = {};
  _editingChapterCustomMode = {};
  _editingChapterTitles = {};
  if (ch) populatePlanChapterTab(ch);
  // Also refresh the sidebar so renames / additions propagate.
  try {
    const sb = await fetch('/api/chapters');
    const sd = await sb.json();
    rebuildSidebar(sd.chapters || sd, currentDraftId);
  } catch (e) { /* non-fatal */ }
  if (status) status.innerHTML = '<span class="u-success">✓ Saved</span>';
}

async function savePlanSectionRow(slug) {
  const cid = _planContext.chapterId
    || document.getElementById('plan-sections-chapter-picker').value;
  if (!cid) return;
  const ch = chaptersData.find(function(c) { return c.id === cid; });
  if (!ch) return;
  try {
    await _putPlanSections(cid, _materializePlanSections(ch),
                           'Saving section…');
  } catch (e) {
    document.getElementById('plan-status').textContent =
      'Save failed: ' + e.message;
  }
}

async function movePlanSection(slug, delta) {
  const cid = _planContext.chapterId
    || document.getElementById('plan-sections-chapter-picker').value;
  if (!cid) return;
  const ch = chaptersData.find(function(c) { return c.id === cid; });
  if (!ch) return;
  const secs = _materializePlanSections(ch);
  const idx = secs.findIndex(function(s) { return s.slug === slug; });
  const target = idx + delta;
  if (idx < 0 || target < 0 || target >= secs.length) return;
  const tmp = secs[idx]; secs[idx] = secs[target]; secs[target] = tmp;
  try {
    await _putPlanSections(cid, secs, 'Reordering…');
  } catch (e) {
    document.getElementById('plan-status').textContent =
      'Reorder failed: ' + e.message;
  }
}

async function deletePlanSection(slug) {
  const cid = _planContext.chapterId
    || document.getElementById('plan-sections-chapter-picker').value;
  if (!cid) return;
  const ch = chaptersData.find(function(c) { return c.id === cid; });
  if (!ch) return;
  const sec = (ch.sections_meta || []).find(function(s) { return s.slug === slug; });
  const title = sec ? (sec.title || sec.slug) : slug;
  // The backend preserves drafts even when their section_type disappears
  // (they become orphans), so this is not destructive — but say so.
  if (!confirm('Delete section "' + title + '"? Any existing drafts in '
               + 'this section will become orphans (visible only via raw '
               + 'SQL) but are not deleted.')) return;
  const secs = _materializePlanSections(ch).filter(
    function(s) { return s.slug !== slug; }
  );
  try {
    await _putPlanSections(cid, secs, 'Deleting section…');
  } catch (e) {
    document.getElementById('plan-status').textContent =
      'Delete failed: ' + e.message;
  }
}

async function addPlanSection() {
  const cid = _planContext.chapterId
    || document.getElementById('plan-sections-chapter-picker').value;
  if (!cid) {
    alert('Pick a chapter first.');
    return;
  }
  const ch = chaptersData.find(function(c) { return c.id === cid; });
  if (!ch) return;
  const title = prompt('Section title:');
  if (!title || !title.trim()) return;
  const secs = _materializePlanSections(ch);
  // Slug auto-derives server-side; we can send just {title, plan}.
  secs.push({title: title.trim(), plan: '', target_words: null});
  try {
    await _putPlanSections(cid, secs, 'Adding section…');
  } catch (e) {
    document.getElementById('plan-status').textContent =
      'Add failed: ' + e.message;
  }
}

// Phase 21 — populate the single-section editor for the Section tab.
// Phase 32.2 — also populates the per-section length dropdown.
let _editingPlanSectionTargetWords = null;  // null = auto, number = override
function populatePlanSectionTab(chapterId, sectionSlug) {
  const ch = chaptersData.find(c => c.id === chapterId);
  if (!ch) return;
  const meta = Array.isArray(ch.sections_meta) ? ch.sections_meta : [];
  const sec = meta.find(s => s.slug === sectionSlug);
  const ctx = document.getElementById('plan-section-context');
  if (ctx) {
    ctx.innerHTML = '<strong>Ch.' + ch.num + ': ' + escapeHtml(ch.title) + '</strong> &middot; ' +
      'section <code>' + sectionSlug + '</code>';
  }
  document.getElementById('plan-section-title').value = sec ? sec.title : sectionSlug;
  document.getElementById('plan-section-text').value = sec ? (sec.plan || '') : '';

  // Phase 32.2 — seed the target dropdown from the section meta.
  const tw = (sec && sec.target_words && sec.target_words > 0) ? sec.target_words : null;
  _editingPlanSectionTargetWords = tw;
  const select = document.getElementById('plan-section-target-select');
  const customInput = document.getElementById('plan-section-target-custom');
  const presets = [400, 800, 1500, 3000, 6000];
  if (select) {
    if (!tw) {
      select.value = '';
    } else if (presets.includes(tw)) {
      select.value = String(tw);
    } else {
      select.value = 'custom';
    }
  }
  if (customInput) {
    if (tw && !presets.includes(tw)) {
      customInput.classList.remove('u-hidden');
      customInput.value = String(tw);
    } else {
      customInput.classList.add('u-hidden');
      customInput.value = '';
    }
  }
  _refreshPlanSectionTargetBadge(meta.length);

  // If the section is missing from the meta (orphan), warn the user.
  if (!sec) {
    const status = document.getElementById('plan-status');
    if (status) {
      status.innerHTML = '<span class="u-warning">This section\'s slug \'' +
        sectionSlug + '\' isn\'t in the chapter\'s sections list. Saving will add it.</span>';
    }
  }
}

// Phase 32.2 — refresh the target badge text + style based on the
// current editing state. Called from both the dropdown and custom-
// input change handlers.
function _refreshPlanSectionTargetBadge(numSections) {
  const badge = document.getElementById('plan-section-target-badge');
  if (!badge) return;
  const chapterTarget = (window._chapterWordTarget && window._chapterWordTarget > 0)
    ? window._chapterWordTarget : 6000;
  const n = Math.max(1, numSections || 1);
  const perSection = Math.max(400, Math.min(chapterTarget, Math.floor(chapterTarget / n)));
  const tw = _editingPlanSectionTargetWords;
  const effective = (tw && tw > 0) ? tw : perSection;
  const tag = (tw && tw > 0)
    ? '<span class="badge-tag">override</span>'
    : '<span class="badge-tag muted">auto</span>';
  badge.innerHTML = '~' + effective + ' words ' + tag;
  badge.className = (tw && tw > 0) ? 'sec-target-badge override' : 'sec-target-badge';
}

function updatePlanSectionTargetWords(value) {
  const customInput = document.getElementById('plan-section-target-custom');
  if (value === '' || value === 'auto') {
    _editingPlanSectionTargetWords = null;
    if (customInput) { customInput.classList.add('u-hidden'); customInput.value = ''; }
  } else if (value === 'custom') {
    if (!_editingPlanSectionTargetWords) _editingPlanSectionTargetWords = 1500;
    if (customInput) {
      customInput.classList.remove('u-hidden');
      customInput.value = String(_editingPlanSectionTargetWords);
      customInput.focus();
      customInput.select();
    }
  } else {
    const n = parseInt(value, 10);
    _editingPlanSectionTargetWords = isNaN(n) ? null : n;
    if (customInput) { customInput.classList.add('u-hidden'); customInput.value = ''; }
  }
  // Re-fetch num sections from the active chapter for the badge.
  const ch = chaptersData.find(c => c.id === _planContext.chapterId);
  const n = (ch && Array.isArray(ch.sections_meta)) ? ch.sections_meta.length : 1;
  _refreshPlanSectionTargetBadge(n);
}

function updatePlanSectionTargetWordsCustom(value) {
  const n = parseInt(value, 10);
  _editingPlanSectionTargetWords = (isNaN(n) || n <= 0) ? null : n;
  const ch = chaptersData.find(c => c.id === _planContext.chapterId);
  const num = (ch && Array.isArray(ch.sections_meta)) ? ch.sections_meta.length : 1;
  _refreshPlanSectionTargetBadge(num);
}

// Phase 17 — length preset buttons in the Plan modal. Setting a preset
// just fills the input; the user still has to click Save to persist.
function setLengthPreset(words) {
  const input = document.getElementById('plan-target-words-input');
  if (input) input.value = String(words);
  const lstatus = document.getElementById('plan-length-status');
  if (lstatus) lstatus.textContent = 'preset: ' + words + ' — click Save to apply';
}

async function savePlan() {
  // Phase 21 / 54.6.66 — savePlan dispatches on the active tab.
  // plan-book    → leitmotiv + target_chapter_words (book-wide record)
  // plan-chapters→ no bulk save; each chapter row has its own Save
  //                button (see savePlanChapterRow). We still respond
  //                with a status line so the global Save button feels
  //                responsive.
  // plan-chapter → per-chapter section plans (existing flow).
  const activeTabBtn = document.querySelector('#plan-modal .tab.active');
  const tab = activeTabBtn ? activeTabBtn.dataset.tab : 'plan-book';

  if (tab === 'plan-book') {
    return savePlanBook();
  } else if (tab === 'plan-chapters') {
    // Save every dirty row. Simple approach: PUT all rows (the PUT
    // is idempotent and cheap — text-only metadata update).
    const rows = document.querySelectorAll('#plan-chapters-list .plan-ch-row');
    if (!rows.length) return;
    document.getElementById('plan-status').textContent =
      'Saving ' + rows.length + ' chapter(s)…';
    for (const row of rows) {
      const cid = row.dataset.cid;
      if (cid) { await savePlanChapterRow(cid); }
    }
    document.getElementById('plan-status').innerHTML =
      '<span class="u-success">Saved all chapters.</span>';
    return;
  } else if (tab === 'plan-chapter') {
    return savePlanChapterSections();
  }
}

async function savePlanBook(viaSaveNow) {
  // Phase 54.6.x — `viaSaveNow=true` means the user clicked Save Now;
  // we cancel any pending autosave debounce and run synchronously
  // through the same code path. Both autosave + Save Now feed the
  // tab-level autosave status pill (#autosave-plan-book) so the
  // user has a single source of truth for save state.
  if (viaSaveNow) {
    flushAutosave('plan-book');
    _setAutosaveStatus('plan-book', 'saving');
  }
  const title = document.getElementById('plan-title-input').value.trim();
  const desc = document.getElementById('plan-desc-input').value.trim();
  const plan = document.getElementById('plan-text-input').value.trim();
  const tcwRaw = document.getElementById('plan-target-words-input').value.trim();
  if (!viaSaveNow) {
    // Autosave path — keep the footer status line quiet; the pill
    // above the tab is the canonical surface.
  } else {
    document.getElementById('plan-status').textContent = 'Saving...';
  }
  const fd = new FormData();
  fd.append('title', title);
  fd.append('description', desc);
  fd.append('plan', plan);
  if (tcwRaw !== '') {
    const n = parseInt(tcwRaw, 10);
    if (!isNaN(n)) fd.append('target_chapter_words', String(n));
  }
  try {
    const res = await fetch('/api/book', {method: 'PUT', body: fd});
    if (!res.ok) throw new Error('save failed');
    if (viaSaveNow) {
      document.getElementById('plan-status').innerHTML =
        '<span class="u-success">Saved.</span> ' +
        plan.split(/\s+/).filter(Boolean).length + ' words. The new plan will be injected into all future writes.';
      _setAutosaveStatus('plan-book', 'saved');
    }
    if (title) document.querySelector('.sidebar h2').textContent = title;
    if (tcwRaw !== '') {
      const n = parseInt(tcwRaw, 10);
      const lstatus = document.getElementById('plan-length-status');
      if (lstatus) {
        if (n > 0) lstatus.textContent = 'current: ' + n + ' words/chapter';
        else lstatus.textContent = 'cleared — using default';
      }
      window._chapterWordTarget = n > 0 ? n : 6000;
    }
  } catch (e) {
    if (viaSaveNow) {
      document.getElementById('plan-status').textContent = 'Save failed: ' + e.message;
    }
    throw e;  // bubble so scheduleAutosave can mark the pill as error
  }
}

// Phase 21 — save edits to all section plans for the active chapter.
// Sends the full sections list (with merged plan edits) via PUT
// /api/chapters/{id}/sections, which is a full-replace endpoint —
// existing titles + slugs are preserved, only plans get updated.
async function savePlanChapterSections() {
  const chId = _planContext.chapterId;
  if (!chId) return;
  const ch = chaptersData.find(c => c.id === chId);
  if (!ch) return;
  const meta = Array.isArray(ch.sections_meta) ? ch.sections_meta : [];

  // Apply pending plan + target_words + title edits.
  // Phase 32.2 — target_words: prefer the editing-state map (set by
  // the dropdown), fall back to whatever was on the section meta.
  // Phase 54.6.67 — title also editable; pending title edits live in
  // _editingChapterTitles and get folded in here too.
  const updated = meta.map(s => {
    const tw = (s.slug in _editingChapterTargetWords)
      ? _editingChapterTargetWords[s.slug]
      : (s.target_words || null);
    return {
      slug: s.slug,
      title: (s.slug in _editingChapterTitles)
        ? _editingChapterTitles[s.slug]
        : (s.title || s.slug),
      plan: (s.slug in _editingChapterPlans) ? _editingChapterPlans[s.slug] : (s.plan || ''),
      target_words: (tw && tw > 0) ? tw : null,
    };
  });

  document.getElementById('plan-status').textContent = 'Saving section plans...';
  try {
    const res = await fetch('/api/chapters/' + chId + '/sections', {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({sections: updated}),
    });
    if (!res.ok) throw new Error('save failed');
    const data = await res.json();
    if (data && Array.isArray(data.sections)) {
      ch.sections_meta = data.sections;
      ch.sections_template = data.sections.map(s => s.slug);
    }
    _editingChapterPlans = {};
    _editingChapterTargetWords = {};
    _editingChapterCustomMode = {};
    document.getElementById('plan-status').innerHTML =
      '<span class="u-success">Saved ' + updated.length + ' section plans.</span>';
    // Refresh sidebar so plan tooltips update
    const sidebarRes = await fetch('/api/chapters');
    const sd = await sidebarRes.json();
    rebuildSidebar(sd.chapters || sd, currentDraftId);
  } catch (e) {
    document.getElementById('plan-status').textContent = 'Save failed: ' + e.message;
  }
}

// Phase 21 — save the single-section plan editor (Section tab).
// Round-trips through the same /sections endpoint by patching the
// chapter's full sections list with this one section's new plan.
async function savePlanSection() {
  const chId = _planContext.chapterId;
  const slug = _planContext.sectionSlug;
  if (!chId || !slug) return;
  const ch = chaptersData.find(c => c.id === chId);
  if (!ch) return;
  const newPlan = document.getElementById('plan-section-text').value;
  const newTitle = document.getElementById('plan-section-title').value.trim();
  const meta = Array.isArray(ch.sections_meta) ? ch.sections_meta.slice() : [];

  // Phase 32.2 — also persist the per-section target_words override
  // edited via the dropdown.
  const newTw = (_editingPlanSectionTargetWords && _editingPlanSectionTargetWords > 0)
    ? _editingPlanSectionTargetWords : null;

  let found = false;
  const updated = meta.map(s => {
    if (s.slug === slug) {
      found = true;
      return {
        slug: s.slug,
        title: newTitle || s.title || s.slug,
        plan: newPlan,
        target_words: newTw,
      };
    }
    // Phase 32.2 — preserve target_words on every other section so
    // saving from the Section tab doesn't accidentally wipe overrides
    // set elsewhere in the chapter.
    return {
      slug: s.slug,
      title: s.title || s.slug,
      plan: s.plan || '',
      target_words: (s.target_words && s.target_words > 0) ? s.target_words : null,
    };
  });
  // Orphan section (slug not in meta): append it.
  if (!found) {
    updated.push({slug: slug, title: newTitle || slug, plan: newPlan, target_words: newTw});
  }

  document.getElementById('plan-status').textContent = 'Saving section plan...';
  try {
    const res = await fetch('/api/chapters/' + chId + '/sections', {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({sections: updated}),
    });
    if (!res.ok) throw new Error('save failed');
    const data = await res.json();
    if (data && Array.isArray(data.sections)) {
      ch.sections_meta = data.sections;
      ch.sections_template = data.sections.map(s => s.slug);
    }
    document.getElementById('plan-status').innerHTML =
      '<span class="u-success">Section plan saved.</span> Will be injected into the next write/autowrite of this section.';
    // Refresh sidebar so the tooltip updates
    const sidebarRes = await fetch('/api/chapters');
    const sd = await sidebarRes.json();
    rebuildSidebar(sd.chapters || sd, currentDraftId);
  } catch (e) {
    document.getElementById('plan-status').textContent = 'Save failed: ' + e.message;
  }
}

async function regeneratePlan() {
  const status = document.getElementById('plan-status');
  const ta = document.getElementById('plan-text-input');
  if (ta.value.trim() && !confirm('This will replace the current plan with an LLM-generated one. Continue?')) return;
  status.textContent = 'Starting generation...';
  ta.value = '';

  const stats = createStreamStats('plan-stream-stats', 'qwen3.5:27b');
  stats.start();

  const fd = new FormData();
  const res = await fetch('/api/book/plan/generate', {method: 'POST', body: fd});
  const data = await res.json();
  currentJobId = data.job_id;
  if (currentEventSource) currentEventSource.close();
  const source = new EventSource('/api/stream/' + data.job_id);
  currentEventSource = source;

  source.onmessage = function(e) {
    const evt = JSON.parse(e.data);
    if (evt.type === 'token') {
      ta.value += evt.text;
      ta.scrollTop = ta.scrollHeight;
      stats.update(evt.text);
    } else if (evt.type === 'model_info') {
      stats.setModel(evt.writer_model);
    } else if (evt.type === 'progress') {
      status.textContent = evt.detail || evt.stage;
    } else if (evt.type === 'completed') {
      status.innerHTML = '<span class="u-success">Generated and saved.</span> ' +
        (evt.chars || ta.value.length) + ' chars.';
      stats.done('done');
      source.close(); currentEventSource = null; currentJobId = null;
    } else if (evt.type === 'error') {
      status.textContent = 'Error: ' + evt.message;
      stats.done('error');
      source.close(); currentEventSource = null; currentJobId = null;
    } else if (evt.type === 'done') {
      stats.done('done');
      source.close(); currentEventSource = null; currentJobId = null;
    }
  };
}

// Phase 54.6.8 — Regenerate the book's chapter outline. Streams LLM
// tokens into a read-only buffer on the plan-status line; on completion,
// refreshes the sidebar so any newly-inserted chapters appear without
// a full page reload. Does NOT touch existing chapters (additive).
// Phase 54.6.96 — Outline tab handler (replaces old footer-button
// regenerateOutline). Reads method + model from the tab controls,
// streams tokens into #plan-outline-stream, surfaces inserted /
// skipped counts inline, and refreshes the sidebar on success.
let _outlineSource = null;

function _populateOutlineMethodPicker() {
  // Same elicitation catalogue the old footer picker used; we populate
  // once and rely on the helper's own caching to avoid refetching.
  _populateMethodSelect('plan-outline-method-select', 'elicitation');
}

async function runOutlineFromTab() {
  const status = document.getElementById('plan-outline-status');
  const stream = document.getElementById('plan-outline-stream');
  const result = document.getElementById('plan-outline-result');
  const btn = document.getElementById('plan-outline-run-btn');
  const cancelBtn = document.getElementById('plan-outline-cancel-btn');
  if (!confirm(
    'Generate a chapter outline from your corpus?\n\n'
    + 'The LLM reads your paper library and proposes chapter titles + '
    + 'descriptions + section slugs. ADDS new chapters only — your '
    + 'existing chapters and drafts are untouched. Run again to get a '
    + 'fresh suggestion if the first pass isn\'t great.\n\n'
    + 'Mirrors `sciknow book outline`.'
  )) return;

  btn.disabled = true;
  if (cancelBtn) cancelBtn.classList.remove('u-hidden');
  result.innerHTML = '';
  stream.style.display = 'block';
  stream.textContent = '';
  status.textContent = 'Starting outline generation…';

  const fd = new FormData();
  const methodEl = document.getElementById('plan-outline-method-select');
  if (methodEl && methodEl.value) fd.append('method', methodEl.value);
  const modelEl = document.getElementById('plan-outline-model-input');
  if (modelEl && modelEl.value.trim()) fd.append('model', modelEl.value.trim());

  let res;
  try {
    res = await fetch('/api/book/outline/generate', {method: 'POST', body: fd});
  } catch (exc) {
    status.textContent = 'Request failed: ' + exc.message;
    btn.disabled = false;
    if (cancelBtn) cancelBtn.classList.add('u-hidden');
    return;
  }
  const data = await res.json();
  if (_outlineSource) _outlineSource.close();
  const source = new EventSource('/api/stream/' + data.job_id);
  _outlineSource = source;
  let tokBuf = '';
  source.onmessage = async function(e) {
    let evt;
    try { evt = JSON.parse(e.data); } catch (_) { return; }
    if (evt.type === 'token') {
      tokBuf += evt.text;
      stream.textContent = tokBuf.slice(-3000);  // tail — avoid huge DOM
      stream.scrollTop = stream.scrollHeight;
      status.textContent = 'Drafting… ' + String(tokBuf.length) + ' chars';
    } else if (evt.type === 'progress') {
      status.textContent = evt.detail || evt.stage;
    } else if (evt.type === 'deep_plan_start') {
      window._outlineDeepN = evt.n_sections_total || 0;
      window._outlineDeepDone = 0; window._outlineDeepFailed = 0;
      status.textContent = 'Deep planning ' + window._outlineDeepN + ' section(s)…';
    } else if (evt.type === 'deep_plan_section_start') {
      status.textContent = 'Deep planning Ch.' + evt.chapter_index
        + '/' + evt.chapter_total + ' §' + evt.section_index
        + '/' + evt.section_total + ': ' + (evt.section_title || '');
    } else if (evt.type === 'deep_plan_section_done') {
      window._outlineDeepDone = (window._outlineDeepDone || 0) + 1;
      if (evt.error) window._outlineDeepFailed = (window._outlineDeepFailed || 0) + 1;
    } else if (evt.type === 'deep_plan_complete') {
      const _planned = evt.n_planned != null ? evt.n_planned : ((window._outlineDeepDone || 0) - (window._outlineDeepFailed || 0));
      const _failed = evt.n_failed != null ? evt.n_failed : (window._outlineDeepFailed || 0);
      status.textContent = 'Deep planning done — ' + _planned + ' section(s) planned, ' + _failed + ' failed.';
    } else if (evt.type === 'completed') {
      source.close(); _outlineSource = null;
      status.innerHTML = '<span class="u-success">\u2713 Outline generated — <strong>'
        + evt.n_inserted + '</strong> new chapter(s), <strong>'
        + evt.n_skipped + '</strong> skipped (already existed).</span>';
      btn.disabled = false;
      if (cancelBtn) cancelBtn.classList.add('u-hidden');
      // Pull fresh chapter list so both sidebar AND the inline result
      // preview reflect what just landed.
      try {
        const sidebarRes = await fetch('/api/chapters');
        const sd = await sidebarRes.json();
        const chapters = sd.chapters || sd;
        rebuildSidebar(chapters, currentDraftId);
        // Inline preview of the chapters now in the book.
        let html = '<h4 style="margin:12px 0 6px;">Current chapters</h4>';
        html += '<ol class="u-m-0 u-pl-20">';
        for (const ch of chapters) {
          const n = (ch.sections || []).length;
          html += '<li><strong>' + (ch.title || 'Untitled') + '</strong>'
            + ' <span class="u-muted">&mdash; ' + n + ' section' + (n === 1 ? '' : 's') + '</span></li>';
        }
        html += '</ol>';
        result.innerHTML = html;
      } catch (_) {}
    } else if (evt.type === 'error') {
      source.close(); _outlineSource = null;
      status.innerHTML = '<span class="u-danger">\u2717 ' + (evt.message || 'error') + '</span>';
      btn.disabled = false;
      if (cancelBtn) cancelBtn.classList.add('u-hidden');
    } else if (evt.type === 'done') {
      source.close(); _outlineSource = null;
      btn.disabled = false;
      if (cancelBtn) cancelBtn.classList.add('u-hidden');
    }
  };
}

function cancelOutline() {
  if (_outlineSource) { _outlineSource.close(); _outlineSource = null; }
  const btn = document.getElementById('plan-outline-run-btn');
  const cancelBtn = document.getElementById('plan-outline-cancel-btn');
  const status = document.getElementById('plan-outline-status');
  if (btn) btn.disabled = false;
  if (cancelBtn) cancelBtn.classList.add('u-hidden');
  if (status) status.textContent = 'Cancelled (stream disconnected — any chapters already committed stay).';
}

// Backwards-compat shim: some menus (and tests) still reference the old
// name. Keep it pointing at the new implementation until they're all
// migrated.
const regenerateOutline = runOutlineFromTab;

// Phase 54.6.x — Add an Introduction chapter at position 1, renumbering
// any existing chapters by +1. Used to retrofit the standard
// scientific-book front-matter shape onto a book that was outlined
// before the Introduction-required prompt landed.
async function insertIntroductionChapter() {
  const btn = document.getElementById('plan-insert-intro-btn');
  const status = document.getElementById('plan-autoplan-chapters-status');
  if (!confirm(
    'Insert an Introduction chapter at position 1?\n\n'
    + 'Every existing chapter will be renumbered +1 (Ch.1 → Ch.2, etc). '
    + 'Drafts are preserved — they move with their chapters. The new '
    + 'Introduction has the standard sections: Motivation & Stakes / '
    + 'Scope of the Argument / Key Terms / Roadmap of the Book.\n\n'
    + 'No-op if Ch.1 already looks like an introduction.'
  )) return;
  if (btn) btn.disabled = true;
  if (status) status.textContent = 'Inserting Introduction…';
  try {
    const res = await fetch('/api/book/insert-introduction', {method: 'POST'});
    const data = await res.json();
    if (!res.ok || !data.ok) {
      if (status) status.innerHTML = '<span class="u-danger">✗ ' + (data.error || res.status) + '</span>';
      if (btn) btn.disabled = false;
      return;
    }
    if (status) {
      const cls = data.noop ? 'u-muted' : 'u-success';
      const icon = data.noop ? '—' : '✓';
      status.innerHTML = '<span class="' + cls + '">' + icon + ' ' + data.message + '</span>';
    }
    // Refresh the sidebar + Chapters tab so the renumber is visible.
    try {
      const sidebarRes = await fetch('/api/chapters');
      const sd = await sidebarRes.json();
      const chapters = sd.chapters || sd;
      rebuildSidebar(chapters, currentDraftId);
      populatePlanChaptersTab();
    } catch (_) {}
  } catch (exc) {
    if (status) status.innerHTML = '<span class="u-danger">✗ ' + exc.message + '</span>';
  } finally {
    if (btn) btn.disabled = false;
  }
}

// Phase 54.6.x — Auto-plan chapters button on the Chapters tab.
// Same backend as runOutlineFromTab (the /api/book/outline/generate
// endpoint already produces a chapter list), but streams into the
// Chapters tab UI so users can stay in context. Additive: the
// endpoint never overwrites existing chapters.
let _autoplanChaptersSource = null;

async function autoPlanChapters() {
  const status = document.getElementById('plan-autoplan-chapters-status');
  const log = document.getElementById('plan-autoplan-chapters-log');
  const btn = document.getElementById('plan-autoplan-chapters-btn');
  if (!confirm(
    'Auto-plan chapters from your paper corpus?\n\n'
    + 'The LLM proposes chapter titles + descriptions + section slugs '
    + 'based on the book\'s leitmotiv and the papers in your library. '
    + 'ADDS new chapters only — existing chapters and drafts are '
    + 'untouched. Run again to re-roll.'
  )) return;

  if (btn) btn.disabled = true;
  if (log) { log.style.display = 'block'; log.textContent = ''; }
  if (status) status.textContent = 'Starting auto-plan…';

  let res;
  try {
    res = await fetch('/api/book/outline/generate', {method: 'POST', body: new FormData()});
  } catch (exc) {
    if (status) status.textContent = 'Request failed: ' + exc.message;
    if (btn) btn.disabled = false;
    return;
  }
  const data = await res.json();
  if (_autoplanChaptersSource) _autoplanChaptersSource.close();
  const source = new EventSource('/api/stream/' + data.job_id);
  _autoplanChaptersSource = source;
  let tokBuf = '';
  source.onmessage = async function(e) {
    let evt;
    try { evt = JSON.parse(e.data); } catch (_) { return; }
    if (evt.type === 'token') {
      tokBuf += evt.text;
      if (log) {
        log.textContent = tokBuf.slice(-3000);
        log.scrollTop = log.scrollHeight;
      }
      if (status) status.textContent = 'Drafting… ' + tokBuf.length + ' chars';
    } else if (evt.type === 'progress') {
      if (status) status.textContent = evt.detail || evt.stage;
    } else if (evt.type === 'deep_plan_section_start') {
      if (status) status.textContent = 'Deep planning Ch.' + evt.chapter_index
        + '/' + evt.chapter_total + ' §' + evt.section_index
        + '/' + evt.section_total + ': ' + (evt.section_title || '');
    } else if (evt.type === 'deep_plan_complete') {
      if (status) status.textContent = 'Deep planning done — '
        + (evt.n_planned || 0) + ' section(s) planned, '
        + (evt.n_failed || 0) + ' failed.';
    } else if (evt.type === 'completed') {
      source.close(); _autoplanChaptersSource = null;
      if (status) {
        status.innerHTML = '<span class="u-success">✓ Added <strong>'
          + evt.n_inserted + '</strong> new chapter(s), <strong>'
          + evt.n_skipped + '</strong> skipped.</span>';
      }
      if (btn) btn.disabled = false;
      try {
        const sidebarRes = await fetch('/api/chapters');
        const sd = await sidebarRes.json();
        const chapters = sd.chapters || sd;
        chaptersData = chapters;
        rebuildSidebar(chapters, currentDraftId);
        populatePlanChaptersTab();
      } catch (_) {}
    } else if (evt.type === 'error') {
      source.close(); _autoplanChaptersSource = null;
      if (status) status.innerHTML = '<span class="u-danger">✗ ' + (evt.message || 'error') + '</span>';
      if (btn) btn.disabled = false;
    } else if (evt.type === 'done') {
      source.close(); _autoplanChaptersSource = null;
      if (btn) btn.disabled = false;
    }
  };
}

// ── Phase 14.3: Chapter scope modal (description + topic_query) ─────
// Phase 18 — chapter modal carries an in-memory copy of the chapter's
// sections list while the modal is open. Saved on Save, discarded on
// Close. Each item is {slug, title, plan}; new rows have an empty slug
// and the server slugifies the title on save.
let _editingSections = [];

function openChapterModal(chId) {
  if (!chId) chId = currentChapterId;
  if (!chId) {
    showEmptyHint('Select a chapter from the sidebar first.');
    return;
  }
  // Look up the chapter from chaptersData (already in JS state)
  const ch = chaptersData.find(c => c.id === chId);
  if (!ch) {
    showEmptyHint('Chapter not found.');
    return;
  }
  document.getElementById('ch-title-input').value = ch.title || '';
  document.getElementById('ch-desc-input').value = ch.description || '';
  document.getElementById('ch-tq-input').value = ch.topic_query || '';
  document.getElementById('chapter-modal-status').textContent = 'Editing Ch.' + ch.num + ': ' + (ch.title || '');
  document.getElementById('chapter-modal').dataset.chId = chId;

  // Phase 21 — fetch the book's chapter word target so the section
  // editor can show "≈ N words per section". Cached on window so the
  // re-renders that happen on every keystroke don't refetch.
  if (!window._chapterWordTarget) {
    fetch('/api/book').then(r => r.json()).then(d => {
      window._chapterWordTarget = d.target_chapter_words || d.default_target_chapter_words || 6000;
      renderSectionEditor();
    }).catch(() => {
      window._chapterWordTarget = 6000;
    });
  }

  // Phase 18 — copy the sections meta into the editor's working state.
  // sections_meta is the rich [{slug, title, plan, target_words}, ...]
  // shape; falls back to deriving from sections_template (slugs only)
  // for legacy chapters that haven't been opened yet under the new schema.
  // Phase 32.1 — also copy target_words so a previously-saved per-section
  // override is restored when the modal reopens (was being silently
  // dropped, which is why the size dropdown always reset to "Auto").
  if (Array.isArray(ch.sections_meta) && ch.sections_meta.length > 0) {
    _editingSections = ch.sections_meta.map(s => ({
      slug: s.slug || '',
      title: s.title || '',
      plan: s.plan || '',
      target_words: (s.target_words && s.target_words > 0) ? s.target_words : null,
      // Phase 37 — per-section model override. Empty/null = use
      // caller model / global default.
      model: (s.model && typeof s.model === 'string') ? s.model : '',
    }));
  } else if (Array.isArray(ch.sections_template)) {
    _editingSections = ch.sections_template.map(slug => ({
      slug: slug, title: titleifyClient(slug), plan: '', model: ''
    }));
  } else {
    _editingSections = [];
  }
  renderSectionEditor();

  // Phase 54.6.149 — fetch the resolved per-section targets so the
  // sections editor can show "this would target 1,950 words via
  // concept-density from your plan". Fire-and-forget: we call
  // renderSectionEditor() again on completion so the extra info lands
  // without blocking the modal open.
  window._resolvedTargetsByChapter = window._resolvedTargetsByChapter || {};
  fetch('/api/chapters/' + chId + '/resolved-targets')
    .then(r => r.ok ? r.json() : null)
    .then(data => {
      if (!data) return;
      window._resolvedTargetsByChapter[chId] = data;
      // Only re-render if this modal is still on the same chapter
      const modal = document.getElementById('chapter-modal');
      if (modal && modal.dataset.chId === chId) renderSectionEditor();
    })
    .catch(e => console.debug('resolved-targets fetch failed:', e));

  // Phase 54.6.152 — ensure the live plan readout has the right wpc
  // per project type. Populate cache lazily on first modal open so
  // the user doesn't need to visit Book Settings first.
  if (!window._currentBookType || !window._swBookTypes) {
    Promise.all([
      fetch('/api/book').then(r => r.json()).catch(() => ({})),
      (window._swBookTypes ? Promise.resolve(null) : swLoadBookTypes()),
    ]).then(([bookData]) => {
      if (bookData && bookData.book_type) {
        window._currentBookType = bookData.book_type;
      }
      // Refresh the plan readouts with the now-correct wpc midpoint
      const tas = document.querySelectorAll('#ch-sections-list textarea[data-section-idx]');
      tas.forEach(ta => {
        const idx = parseInt(ta.dataset.sectionIdx, 10);
        if (!isNaN(idx)) updatePlanConceptReadout(idx, ta);
      });
    }).catch(e => console.debug('book-type cache warm failed:', e));
  }

  switchChapterTab('ch-scope');
  openModal('chapter-modal');
}

// Tiny client-side titleifier mirroring _titleify_slug. Used for legacy
// chapters that only have a flat slug list — the editor synthesizes a
// best-effort display title so the user can immediately see + edit.
function titleifyClient(slug) {
  return (slug || '').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function switchChapterTab(name) {
  document.querySelectorAll('#chapter-modal .tab').forEach(t => {
    t.classList.toggle('active', t.dataset.tab === name);
  });
  document.getElementById('ch-scope-pane').style.display = (name === 'ch-scope') ? 'block' : 'none';
  document.getElementById('ch-sections-pane').style.display = (name === 'ch-sections') ? 'block' : 'none';
}

// Phase 18 + Phase 21 — render the working sections list into the editor.
// Re-rendered on every change so reorder/add/delete reflect immediately.
// Phase 21 adds: live slug preview while typing the title, per-section
// word-count budget (chapter target / num sections), and a header showing
// the chapter-level target.
function renderSectionEditor() {
  const list = document.getElementById('ch-sections-list');
  if (!list) return;
  // Compute the per-section word budget from the book's chapter target.
  // This is identical to core.book_ops._section_target_words: floor 400,
  // ceiling = chapter_target. We read the target from the cached book
  // settings populated by openPlanModal().
  const chapterTarget = (window._chapterWordTarget && window._chapterWordTarget > 0)
    ? window._chapterWordTarget
    : 6000;
  const n = Math.max(1, _editingSections.length || 1);
  const perSection = Math.max(400, Math.min(chapterTarget, Math.floor(chapterTarget / n)));

  if (_editingSections.length === 0) {
    list.innerHTML = '<div style="font-size:12px;color:var(--fg-muted);padding:12px;text-align:center;border:1px dashed var(--border);border-radius:4px;">No sections yet. Click <strong>Add section</strong> to start.</div>';
    return;
  }

  // Header showing the budget split
  let html = '<div class="u-tiny u-muted u-mb-2 u-p-6-10 u-bg-tb u-r-sm">';
  html += '<strong>' + _editingSections.length + '</strong> section' + (_editingSections.length === 1 ? '' : 's') +
          ' &middot; chapter target: <strong>' + chapterTarget + '</strong> words &middot; ' +
          'per section: <strong>~' + perSection + '</strong> words';
  html += '</div>';

  _editingSections.forEach((s, i) => {
    // Live slug preview: derive from current slug or fall back to a
    // slugified title (matches core.book_ops._slugify_section_name).
    const liveSlug = (s.slug || s.title || '').trim().toLowerCase().replace(/\s+/g, '_');
    const slugDisplay = liveSlug || '<em>(slug auto-generated)</em>';
    // Phase 29 — per-section target_words override. The dropdown
    // offers presets + Custom (which reveals a number input). When
    // "Auto" is selected, target_words is null and the autowrite
    // resolution falls through to chapter target / num_sections.
    const tw = s.target_words;
    const isAuto = !tw || tw <= 0;
    const presets = [400, 800, 1500, 3000, 6000];
    // Phase 31 — bug fix: previously isCustom was derived from
    // `!presets.includes(tw)`, but the "Custom" branch initialized
    // tw=1500 (in presets) so on re-render isCustom became false and
    // the input stayed hidden. Fix: track an explicit _customMode flag
    // on the section dict, set when the user picks Custom, cleared
    // when they pick a preset or Auto.
    const presetMatch = !isAuto && presets.includes(tw);
    const isCustom = s._customMode || (!isAuto && !presetMatch);
    let optsHtml = '<option value="">Auto (~' + perSection + 'w)</option>';
    presets.forEach(p => {
      const sel = (!isCustom && tw === p) ? ' selected' : '';
      const labelMap = {400: 'Very short', 800: 'Short', 1500: 'Medium', 3000: 'Long', 6000: 'Extra long'};
      optsHtml += '<option value="' + p + '"' + sel + '>' + labelMap[p] + ' (~' + p + 'w)</option>';
    });
    optsHtml += '<option value="custom"' + (isCustom ? ' selected' : '') + '>Custom\u2026</option>';
    const customStyle = isCustom ? '' : 'display:none;';
    const customVal = (isCustom && tw) ? String(tw) : '';
    html += '<div class="sec-row" data-idx="' + i + '">';
    html += '  <div class="sec-handle">';
    // Phase 42 — data-action dispatch. delta is a signed int in dataset.
    html += '    <button class="u-dim-3 u-cursor-default" data-action="move-section" data-section-index="' + i + '" data-delta="-1" title="Move up"' + (i === 0 ? ' disabled' : '') + '>&uarr;</button>';
    html += '    <button class="u-dim-3 u-cursor-default" data-action="move-section" data-section-index="' + i + '" data-delta="1" title="Move down"' + (i === _editingSections.length - 1 ? ' disabled' : '') + '>&darr;</button>';
    html += '  </div>';
    html += '  <div class="sec-fields">';
    html += '    <input type="text" placeholder="Section title (e.g. The 11-Year Solar Cycle)" ';
    html += '           value="' + escapeHtml(s.title) + '" oninput="updateSectionTitle(' + i + ', this.value)">';
    // Phase 54.6.152 — live concept-count + target readout next to
    // the plan textarea. Updates as the user types so the concept-
    // density resolver's output is visible immediately, not just
    // after a modal-level re-fetch. Soft warning fires when bullet
    // count exceeds Cowan's 4-chunk ceiling (consistent with the
    // backend log-line in _get_section_concept_density_target).
    html += '    <textarea placeholder="Section plan — what THIS section must cover (a few sentences, or bullet one concept per line)" ';
    html += '              data-section-idx="' + i + '" ';
    html += '              oninput="updateSection(' + i + ', \'plan\', this.value); updatePlanConceptReadout(' + i + ', this);"';
    html += '              title="Write the plan as a bullet list (`- concept`, `* concept`, or `1. concept`). Bullet count drives the concept-density resolver: target_words = N × wpc_midpoint. Ceiling 4 per Cowan 2001 — above that, split the section.">' + escapeHtml(s.plan) + '</textarea>';
    // Live concept-count readout. Rendered below the textarea, populated
    // + refreshed on every keystroke via updatePlanConceptReadout().
    html += '    <div id="plan-readout-' + i + '" class="plan-concept-readout u-indent-sm" '
           + '></div>';
    // Phase 29 — size dropdown row, just below the plan textarea.
    // Phase 32.1 — show the effective target words inline next to the
    // dropdown so the user always sees what budget THIS section will
    // be written to (rather than burying it in the muted slug line).
    // Phase 54.6.149 — consult the resolver-explanation cache (populated
    // by openChapterModal) so the badge reflects the ACTUAL level that
    // would fire at autowrite time, not just "override vs auto split".
    // Four possible levels: explicit_section_override, concept_density
    // (plan counted), chapter_split (fallback), pending (cache not
    // loaded yet).
    const chIdForResolve = document.getElementById('chapter-modal').dataset.chId;
    const resolvedCache  = (window._resolvedTargetsByChapter || {})[chIdForResolve];
    const resolvedSec    = resolvedCache && resolvedCache.sections
      ? resolvedCache.sections.find(x => x.slug === (s.slug || ''))
      : null;
    let effectiveTw, badgeLabel, badgeTitle, badgeTag, targetBadgeClass;
    if (tw && tw > 0) {
      effectiveTw = tw;
      badgeTag = 'override';
      badgeTitle = 'Per-section override (set explicitly by you)';
      targetBadgeClass = 'sec-target-badge override';
    } else if (resolvedSec && resolvedSec.level === 'concept_density') {
      effectiveTw = resolvedSec.target;
      badgeTag = 'concept-density';
      badgeTitle = resolvedSec.explanation || 'Bottom-up from section plan';
      targetBadgeClass = 'sec-target-badge concept-density';
    } else if (resolvedSec) {
      effectiveTw = resolvedSec.target;
      badgeTag = 'chapter split';
      badgeTitle = resolvedSec.explanation || 'Chapter target / num sections';
      targetBadgeClass = 'sec-target-badge';
    } else {
      effectiveTw = perSection;
      badgeTag = 'auto';
      badgeTitle = 'Loading resolver…';
      targetBadgeClass = 'sec-target-badge';
    }
    html += '    <div class="sec-size-row">';
    html += '      <label>Target:</label>';
    html += '      <select onchange="updateSectionTargetWords(' + i + ', this.value)" title="Pick a preset word target for this section. Choose Custom to enter an exact number in the box on the right. Auto = the resolver picks: concept-density (plan × wpc) when a plan exists, else chapter target / num sections.">' + optsHtml + '</select>';
    html += '      <input type="number" class="sec-size-custom" placeholder="words" min="100" step="100" ';
    html += '             value="' + customVal + '" style="' + customStyle + '" ';
    html += '             oninput="updateSectionTargetWordsCustom(' + i + ', this.value)" title="Custom target word count. Visible only when the preset dropdown is set to Custom.">';
    html += '      <span class="' + targetBadgeClass + '" title="' + escapeHtml(badgeTitle) + '">';
    html += '~' + effectiveTw.toLocaleString() + ' words';
    html += ' <span class="badge-tag' + (badgeTag === 'auto' || badgeTag === 'chapter split' ? ' muted' : '') + '">' + escapeHtml(badgeTag) + '</span>';
    html += '</span>';
    html += '    </div>';
    // Phase 37 — per-section model override. Free-text input with a
    // shared datalist of common Ollama tags. Empty = use the caller's
    // model (CLI --model / API form) or fall through to settings.
    // llm_model. Pairs with the Phase 35 compute counter: dial
    // expensive models up only on sections that need them.
    const modelVal = (s.model || '').trim();
    const modelBadgeClass = modelVal ? 'sec-target-badge override' : 'sec-target-badge';
    const modelBadgeTitle = modelVal
      ? 'Per-section model override (this section only)'
      : 'Uses the caller-provided model or settings.llm_model default';
    html += '    <div class="sec-size-row">';
    html += '      <label>Model:</label>';
    html += '      <input type="text" list="sec-model-suggestions" class="sec-size-custom" ';
    html += '             style="width:160px;" placeholder="(default)" ';
    html += '             value="' + escapeHtml(modelVal) + '" ';
    html += '             oninput="updateSectionModel(' + i + ', this.value)">';
    html += '      <span class="' + modelBadgeClass + '" title="' + modelBadgeTitle + '">';
    html += (modelVal ? escapeHtml(modelVal) + ' <span class="badge-tag">override</span>'
                      : '— <span class="badge-tag muted">default</span>');
    html += '</span>';
    html += '    </div>';
    html += '    <div class="sec-slug">slug: <code>' + slugDisplay + '</code></div>';
    html += '  </div>';
    html += '  <button class="sec-delete" data-action="remove-section" data-section-index="' + i + '" title="Delete this section">&times;</button>';
    html += '</div>';
  });
  // Phase 37 — one shared datalist for the per-section model inputs.
  // The list is hints, not a whitelist — Ollama accepts any tag.
  html += '<datalist id="sec-model-suggestions">';
  ['qwen3:32b', 'qwen3:14b', 'qwen3:8b', 'qwen2.5:32b', 'qwen2.5:14b',
   'qwen2.5:7b', 'llama3.1:70b', 'llama3.1:8b', 'mistral-nemo:12b',
   'gemma2:27b', 'gemma2:9b', 'phi3.5:3.8b']
    .forEach(m => { html += '<option value="' + m + '">'; });
  html += '</datalist>';
  list.innerHTML = html;
  // Phase 54.6.152 — populate the live readouts on initial render so
  // users see "3 concepts → ~1,950 words" without having to type
  // first. Also runs after every re-render so edits to existing
  // sections stay in sync.
  const _planTAs = list.querySelectorAll('textarea[data-section-idx]');
  _planTAs.forEach(ta => {
    const idx = parseInt(ta.dataset.sectionIdx, 10);
    if (!isNaN(idx)) updatePlanConceptReadout(idx, ta);
  });
}

// Phase 29/31 — handle the size dropdown selection. "" → Auto
// (clear override + clear custom mode), a numeric preset → set
// (and clear custom mode), "custom" → enter custom mode (reveal
// the number input).
function updateSectionTargetWords(idx, value) {
  if (idx < 0 || idx >= _editingSections.length) return;
  const sec = _editingSections[idx];
  if (value === "" || value === "auto") {
    sec.target_words = null;
    sec._customMode = false;
  } else if (value === "custom") {
    // Phase 31 — set the explicit _customMode flag so the next
    // re-render keeps the input visible regardless of whether
    // target_words happens to coincide with a preset value.
    sec._customMode = true;
    // Default to a starting value if there isn't one yet
    if (!sec.target_words) sec.target_words = 1500;
  } else {
    const n = parseInt(value, 10);
    sec.target_words = isNaN(n) ? null : n;
    sec._customMode = false;
  }
  renderSectionEditor();
  // After re-render, focus the custom input if we just entered
  // custom mode so the user can type immediately.
  if (value === "custom") {
    setTimeout(() => {
      const rows = document.querySelectorAll('#ch-sections-list .sec-row');
      if (rows[idx]) {
        const input = rows[idx].querySelector('.sec-size-custom');
        if (input) {
          input.focus();
          input.select();
        }
      }
    }, 0);
  }
}

function updateSectionTargetWordsCustom(idx, value) {
  if (idx < 0 || idx >= _editingSections.length) return;
  const sec = _editingSections[idx];
  const n = parseInt(value, 10);
  if (isNaN(n) || n <= 0) {
    sec.target_words = null;
  } else {
    sec.target_words = n;
  }
  // Stay in custom mode while the user types — only the dropdown
  // change handler clears it.
  sec._customMode = true;
  // Don't re-render here — the user is actively typing in the input.
}

// Phase 21 — title-input handler that ALSO updates the slug live so the
// slug preview shows immediately instead of waiting for save. We only
// auto-derive slug from title when slug is empty (untouched), so users
// who manually entered a slug aren't surprised when it overwrites.
function updateSectionTitle(idx, value) {
  if (idx < 0 || idx >= _editingSections.length) return;
  const sec = _editingSections[idx];
  sec.title = value;
  // Live slug derivation only when slug is empty or matches the
  // previously-derived slug (so manual slug edits stick).
  if (!sec.slug) {
    sec.slug = (value || '').trim().toLowerCase().replace(/\s+/g, '_');
  }
  // Re-render to refresh the live slug preview at the bottom of the row.
  // We preserve focus by re-finding the input and restoring the cursor
  // position after the innerHTML rebuild.
  const activeIdx = document.activeElement && document.activeElement.closest('.sec-row');
  const cursorPos = document.activeElement && document.activeElement.selectionStart;
  renderSectionEditor();
  if (activeIdx) {
    const newRows = document.querySelectorAll('#ch-sections-list .sec-row');
    if (newRows[idx]) {
      const newInput = newRows[idx].querySelector('input');
      if (newInput) {
        newInput.focus();
        if (typeof cursorPos === 'number') {
          try { newInput.setSelectionRange(cursorPos, cursorPos); } catch (e) {}
        }
      }
    }
  }
}

function escapeHtml(s) {
  return (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;')
                  .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// Phase 23 — collapse/expand chapter sections. Per-chapter chevron
// toggles a .collapsed class; state persists in localStorage so
// refreshes don't reset the user's view. The collapse-all button at
// the top of the sidebar flips every chapter at once.

const _COLLAPSED_KEY = 'sciknow.collapsedChapters';

function _getCollapsedChapterIds() {
  try {
    return JSON.parse(localStorage.getItem(_COLLAPSED_KEY) || '[]');
  } catch (e) {
    return [];
  }
}

function _setCollapsedChapterIds(ids) {
  try {
    localStorage.setItem(_COLLAPSED_KEY, JSON.stringify(ids));
  } catch (e) { /* localStorage full or disabled — ignore */ }
}

function toggleChapter(group) {
  if (!group) return;
  const chId = group.dataset.chId;
  group.classList.toggle('collapsed');
  const collapsed = _getCollapsedChapterIds();
  const idx = collapsed.indexOf(chId);
  if (group.classList.contains('collapsed')) {
    if (idx === -1) collapsed.push(chId);
  } else {
    if (idx !== -1) collapsed.splice(idx, 1);
  }
  _setCollapsedChapterIds(collapsed);
  _refreshToggleAllButton();
}

function toggleAllChapters() {
  const groups = document.querySelectorAll('#sidebar-sections .ch-group');
  // If ANY chapter is currently expanded, collapse all. Otherwise expand all.
  let anyExpanded = false;
  groups.forEach(g => { if (!g.classList.contains('collapsed')) anyExpanded = true; });
  const newCollapsed = [];
  groups.forEach(g => {
    if (anyExpanded) {
      g.classList.add('collapsed');
      newCollapsed.push(g.dataset.chId);
    } else {
      g.classList.remove('collapsed');
    }
  });
  _setCollapsedChapterIds(newCollapsed);
  _refreshToggleAllButton();
}

function _refreshToggleAllButton() {
  const groups = document.querySelectorAll('#sidebar-sections .ch-group');
  if (groups.length === 0) return;
  let anyExpanded = false;
  groups.forEach(g => { if (!g.classList.contains('collapsed')) anyExpanded = true; });
  const icon = document.getElementById('toggle-all-icon');
  const label = document.getElementById('toggle-all-label');
  if (icon) icon.textContent = anyExpanded ? '\u25bd' : '\u25b7';
  if (label) label.textContent = anyExpanded ? 'Collapse all' : 'Expand all';
}

// Restore the persisted collapsed state on page load + after every
// rebuildSidebar. Idempotent — safe to call multiple times.
function restoreCollapsedChapters() {
  const collapsed = _getCollapsedChapterIds();
  collapsed.forEach(chId => {
    const group = document.querySelector(
      '#sidebar-sections .ch-group[data-ch-id="' + chId + '"]'
    );
    if (group) group.classList.add('collapsed');
  });
  _refreshToggleAllButton();
}

// Run on initial page load (after the static sidebar HTML is parsed).
document.addEventListener('DOMContentLoaded', restoreCollapsedChapters);

// Phase 54.6.194 — sidebar rail mode. Toggleable compact sidebar
// (~64 px) showing chapter numbers + status dots only. Persists in
// localStorage; applied to <body> so CSS rules under body.sidebar-rail
// pick it up. Independent of the Hide column state (54.6.164).
const _SIDEBAR_RAIL_KEY = 'sciknow.ui.sidebarRail';
function toggleSidebarRail() {
  const on = document.body.classList.toggle('sidebar-rail');
  try { localStorage.setItem(_SIDEBAR_RAIL_KEY, on ? '1' : '0'); } catch (e) {}
}
function _restoreSidebarRail() {
  try {
    if (localStorage.getItem(_SIDEBAR_RAIL_KEY) === '1') {
      document.body.classList.add('sidebar-rail');
    }
  } catch (e) {}
}
document.addEventListener('DOMContentLoaded', _restoreSidebarRail);

// Phase 54.6.164 — collapsible left / right columns. Four localStorage
// keys: two current-state ("…Hidden") + two auto-hide prefs
// ("autohide…"). On load: if auto-hide is on, start hidden;
// otherwise restore the last toggled state. Toggling during the
// session persists the new state only when auto-hide is off, so
// auto-hide wins at reload without losing session toggles mid-use.
const _COL_KEYS = {
  sidebar: { hidden: 'sciknow.ui.sidebarHidden',
             auto:   'sciknow.ui.autohideSidebar',
             cls:    'sidebar-hidden' },
  panel:   { hidden: 'sciknow.ui.panelHidden',
             auto:   'sciknow.ui.autohidePanel',
             cls:    'panel-hidden' },
};

function _colPref(which, key) {
  const cfg = _COL_KEYS[which];
  if (!cfg) return false;
  try { return localStorage.getItem(cfg[key]) === '1'; }
  catch (e) { return false; }
}

function _colSetPref(which, key, val) {
  const cfg = _COL_KEYS[which];
  if (!cfg) return;
  try { localStorage.setItem(cfg[key], val ? '1' : '0'); }
  catch (e) { /* localStorage unavailable — ignore */ }
}

function toggleColumn(which) {
  const cfg = _COL_KEYS[which];
  if (!cfg) return;
  const hidden = document.body.classList.toggle(cfg.cls);
  // Only persist the current-state when auto-hide is OFF — auto-hide
  // always wins at page load, so persisting the toggle would
  // leak session-level choices into the reload baseline.
  if (!_colPref(which, 'auto')) _colSetPref(which, 'hidden', hidden);
}

function _applyInitialColumnState() {
  Object.keys(_COL_KEYS).forEach(which => {
    const cfg = _COL_KEYS[which];
    const start = _colPref(which, 'auto') || _colPref(which, 'hidden');
    document.body.classList.toggle(cfg.cls, start);
  });
}

// Apply before DOMContentLoaded would also work, but the body must
// exist first — inline script after <body> runs on parse, this is
// safe since the sidebar/panel elements are rendered below us.
document.addEventListener('DOMContentLoaded', _applyInitialColumnState);

function bsLoadViewPrefs() {
  const s = document.getElementById('bs-autohide-sidebar');
  const p = document.getElementById('bs-autohide-panel');
  if (s) s.checked = _colPref('sidebar', 'auto');
  if (p) p.checked = _colPref('panel', 'auto');
}

function bsSaveViewPrefs() {
  const s = document.getElementById('bs-autohide-sidebar');
  const p = document.getElementById('bs-autohide-panel');
  if (s) {
    _colSetPref('sidebar', 'auto', s.checked);
    // Immediate feedback: turning auto-hide ON hides the column now.
    // Turning it OFF leaves the current visible state alone.
    if (s.checked) document.body.classList.add('sidebar-hidden');
  }
  if (p) {
    _colSetPref('panel', 'auto', p.checked);
    if (p.checked) document.body.classList.add('panel-hidden');
  }
}

// Phase 22 — word target progress bar in the subtitle. Shows
// "actual/target" plus a coloured bar (warning under 70%, accent
// 70-100%, success over). Hidden when no target is set.
function updateWordTargetBar(actual, target) {
  const wrap = document.getElementById('word-target');
  const fill = document.getElementById('word-target-fill');
  const txt  = document.getElementById('word-target-text');
  if (!wrap || !fill || !txt) return;
  if (!target || target <= 0) {
    wrap.style.display = 'none';
    return;
  }
  wrap.style.display = 'inline-flex';
  const pct = Math.min(150, Math.round((actual / target) * 100));
  fill.style.width = Math.min(100, pct) + '%';
  fill.classList.remove('over', 'under');
  if (pct >= 100) fill.classList.add('over');
  else if (pct < 70) fill.classList.add('under');
  txt.textContent = actual + ' / ' + target + 'w';
}

// Phase 22 — delete an orphan draft from the sidebar. Confirms first
// because deletion is permanent.
async function deleteOrphanDraft(draftId) {
  if (!draftId) return;
  if (!confirm('Permanently delete this orphan draft? This cannot be undone.')) return;
  try {
    const res = await fetch('/api/draft/' + draftId, {method: 'DELETE'});
    if (!res.ok) throw new Error('delete failed (' + res.status + ')');
    // If the deleted draft was the active one, fall back to the
    // dashboard so we don't show stale content.
    if (currentDraftId === draftId) {
      currentDraftId = '';
      showDashboard();
    }
    // Refresh sidebar so the orphan disappears
    const sidebarRes = await fetch('/api/chapters');
    const sd = await sidebarRes.json();
    rebuildSidebar(sd.chapters || sd, currentDraftId);
  } catch (e) {
    alert('Delete failed: ' + e.message);
  }
}

// ── Phase 33: chapter drag-and-drop reordering ──────────────────────────
//
// The POST /api/chapters/reorder endpoint has existed since Phase 14
// but was never wired to a GUI affordance. Phase 33 adds the drag
// handlers, mirroring Phase 26's section drag-drop. The user drags
// a chapter title bar above or below another chapter title bar; on
// drop, the full chapter_ids order is POSTed and the sidebar rebuilt.
//
// Unlike section drag-drop, there's no within-chapter constraint —
// chapters can be reordered freely across the whole book.

let _chDragId = null;
function chDragStart(e, chId) {
  _chDragId = chId;
  e.dataTransfer.effectAllowed = 'move';
  e.dataTransfer.setData('text/plain', chId);
  const group = e.target.closest('.ch-group');
  if (group) setTimeout(() => group.classList.add('dragging'), 0);
}
function chDragOver(e) {
  if (!_chDragId) return;
  e.preventDefault();
  e.dataTransfer.dropEffect = 'move';
  const title = e.target.closest('.ch-title');
  // Visual indicator: top/bottom border based on cursor position.
  document.querySelectorAll('.ch-title').forEach(t => {
    t.classList.remove('ch-drag-over-top', 'ch-drag-over-bottom');
  });
  if (title) {
    const rect = title.getBoundingClientRect();
    const mid = rect.top + rect.height / 2;
    if (e.clientY < mid) {
      title.classList.add('ch-drag-over-top');
    } else {
      title.classList.add('ch-drag-over-bottom');
    }
  }
}
function chDragEnd(e) {
  _chDragId = null;
  document.querySelectorAll('.ch-group').forEach(g => g.classList.remove('dragging'));
  document.querySelectorAll('.ch-title').forEach(t => {
    t.classList.remove('ch-drag-over-top', 'ch-drag-over-bottom');
  });
}
async function chDrop(e, targetChId) {
  e.preventDefault();
  if (!_chDragId || _chDragId === targetChId) {
    chDragEnd(e);
    return;
  }
  // Compute the new order by removing the dragged chapter from
  // its current position and inserting it before or after the target.
  const ids = chaptersData.map(c => c.id);
  const fromIdx = ids.indexOf(_chDragId);
  if (fromIdx < 0) { chDragEnd(e); return; }
  ids.splice(fromIdx, 1);
  const toIdx = ids.indexOf(targetChId);
  if (toIdx < 0) { chDragEnd(e); return; }
  // Insert above or below based on cursor position in the title bar.
  const title = e.target.closest('.ch-title');
  let insertIdx = toIdx;
  if (title) {
    const rect = title.getBoundingClientRect();
    insertIdx = e.clientY > rect.top + rect.height / 2 ? toIdx + 1 : toIdx;
  }
  ids.splice(insertIdx, 0, _chDragId);
  chDragEnd(e);
  // POST the new order
  try {
    const res = await fetch('/api/chapters/reorder', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({chapter_ids: ids}),
    });
    if (!res.ok) throw new Error('reorder failed (' + res.status + ')');
    // Refresh the sidebar with the new order.
    const sidebarRes = await fetch('/api/chapters');
    const sd = await sidebarRes.json();
    if (Array.isArray(sd.chapters)) chaptersData = sd.chapters;
    rebuildSidebar(sd.chapters || sd, currentDraftId);
  } catch (err) {
    alert('Chapter reorder failed: ' + err.message);
  }
}


// ── Phase 26: drag-and-drop section reordering ────────────────────────
//
// Sections in the sidebar are HTML5-draggable. The user clicks and
// holds a section row, drags it above or below another section in
// the SAME chapter, and drops it. The handler computes the new
// order, sends the full sections list to PUT /api/chapters/{id}/sections
// (the existing full-replace endpoint, also used by the chapter
// modal's Sections tab), and refreshes the sidebar.
//
// Cross-chapter drags are not supported — they would need to also
// update drafts.chapter_id, which is a more dangerous operation
// that deserves its own confirmation flow. The handler silently
// rejects drops onto a different .ch-group.
//
// Click vs drag: the browser distinguishes a plain click (no
// movement) from a drag (movement during mousedown). The existing
// onclick handlers for navigation (navTo, writeForCell) keep
// firing on plain clicks even though the row is draggable.

let _draggedSection = null;

function _findDraggableSection(target) {
  return target && target.closest && target.closest('.sec-link[draggable="true"]');
}

function handleSectionDragStart(e) {
  const link = _findDraggableSection(e.target);
  if (!link) return;
  const group = link.closest('.ch-group');
  if (!group) return;
  _draggedSection = {
    chapterId: group.dataset.chId,
    slug: link.dataset.sectionSlug,
  };
  link.classList.add('dragging');
  if (e.dataTransfer) {
    e.dataTransfer.effectAllowed = 'move';
    // Required by Firefox to actually fire dragstart properly.
    e.dataTransfer.setData('text/plain', _draggedSection.slug);
  }
}

function handleSectionDragOver(e) {
  if (!_draggedSection) return;
  const link = _findDraggableSection(e.target);
  if (!link) return;
  const group = link.closest('.ch-group');
  if (!group) return;
  // Don't show a drop indicator on the dragged row itself.
  if (link.dataset.sectionSlug === _draggedSection.slug
      && group.dataset.chId === _draggedSection.chapterId) return;
  e.preventDefault();
  if (e.dataTransfer) e.dataTransfer.dropEffect = 'move';
  // Compute drop position based on cursor Y vs row midpoint.
  const rect = link.getBoundingClientRect();
  const isAbove = e.clientY < (rect.top + rect.height / 2);
  // Clear previous indicator on any other row, set new one.
  document.querySelectorAll('.sec-link.drag-over-top, .sec-link.drag-over-bottom')
    .forEach(el => el.classList.remove('drag-over-top', 'drag-over-bottom'));
  link.classList.add(isAbove ? 'drag-over-top' : 'drag-over-bottom');
}

function handleSectionDrop(e) {
  if (!_draggedSection) { _cleanupDrag(); return; }
  const link = _findDraggableSection(e.target);
  if (!link) { _cleanupDrag(); return; }
  const group = link.closest('.ch-group');
  if (!group) { _cleanupDrag(); return; }
  e.preventDefault();

  const targetChId = group.dataset.chId;
  const targetSlug = link.dataset.sectionSlug;
  const sourceChId = _draggedSection.chapterId;
  const sourceSlug = _draggedSection.slug;

  if (targetSlug === sourceSlug && targetChId === sourceChId) {
    _cleanupDrag();
    return;
  }

  const rect = link.getBoundingClientRect();
  const position = e.clientY < (rect.top + rect.height / 2) ? 'before' : 'after';

  if (targetChId === sourceChId) {
    // Within-chapter reorder (Phase 26, unchanged)
    reorderSections(sourceChId, sourceSlug, targetSlug, position);
  } else {
    // Phase 33 — cross-chapter move. Requires a confirm because it
    // updates drafts.chapter_id, which changes where the draft lives
    // in the book's chapter structure.
    const srcCh = chaptersData.find(c => c.id === sourceChId);
    const tgtCh = chaptersData.find(c => c.id === targetChId);
    const srcName = srcCh ? 'Ch.' + srcCh.num : 'source chapter';
    const tgtName = tgtCh ? 'Ch.' + tgtCh.num : 'target chapter';
    if (confirm('Move section "' + sourceSlug + '" from ' + srcName + ' to ' + tgtName + '?')) {
      moveSectionCrossChapter(sourceChId, sourceSlug, targetChId, targetSlug, position);
    }
  }
  _cleanupDrag();
}

function handleSectionDragEnd() {
  _cleanupDrag();
}

function _cleanupDrag() {
  _draggedSection = null;
  document.querySelectorAll('.sec-link.dragging')
    .forEach(el => el.classList.remove('dragging'));
  document.querySelectorAll('.sec-link.drag-over-top, .sec-link.drag-over-bottom')
    .forEach(el => el.classList.remove('drag-over-top', 'drag-over-bottom'));
}

// Reorder a section by sending the full updated sections list to the
// existing PUT /api/chapters/{id}/sections endpoint. Idempotent and
// safe — the endpoint is full-replace, not patch.
async function reorderSections(chapterId, draggedSlug, targetSlug, position) {
  const ch = chaptersData.find(c => c.id === chapterId);
  if (!ch || !Array.isArray(ch.sections_meta)) return;
  const sections = ch.sections_meta.slice();
  const draggedIdx = sections.findIndex(s => s.slug === draggedSlug);
  if (draggedIdx === -1) return;
  const dragged = sections.splice(draggedIdx, 1)[0];
  // After removing the dragged row, the target index may have shifted.
  const targetIdx = sections.findIndex(s => s.slug === targetSlug);
  if (targetIdx === -1) {
    sections.push(dragged);
  } else {
    const insertAt = position === 'before' ? targetIdx : targetIdx + 1;
    sections.splice(insertAt, 0, dragged);
  }
  try {
    const res = await fetch('/api/chapters/' + chapterId + '/sections', {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({sections: sections}),
    });
    if (!res.ok) throw new Error('reorder failed (' + res.status + ')');
    const data = await res.json();
    // Patch local cache so the next render uses the new order
    // immediately (the /api/chapters refetch below is just to be sure).
    if (data && Array.isArray(data.sections)) {
      ch.sections_meta = data.sections;
      ch.sections_template = data.sections.map(s => s.slug);
    }
    // Full sidebar refresh — preserves collapsed state via Phase 23
    // restoreCollapsedChapters() at the end of rebuildSidebar.
    const sidebarRes = await fetch('/api/chapters');
    const sd = await sidebarRes.json();
    rebuildSidebar(sd.chapters || sd, currentDraftId);
  } catch (e) {
    alert('Reorder failed: ' + e.message);
  }
}

// Phase 33 — cross-chapter section move. Four API calls:
// 1. Find the draft for this slug in the source chapter
// 2. PUT /api/draft/{id}/chapter — move the draft to the target chapter
// 3. Remove the slug from source chapter's sections
// 4. Add the slug to target chapter's sections at the right position
// Then refresh the sidebar.
async function moveSectionCrossChapter(srcChId, slug, tgtChId, targetSlug, position) {
  const srcCh = chaptersData.find(c => c.id === srcChId);
  const tgtCh = chaptersData.find(c => c.id === tgtChId);
  if (!srcCh || !tgtCh) return;

  // 1) Find the draft id for this section in the source chapter.
  // Draft could be in the sections list returned from /api/chapters.
  const draft = (srcCh.sections || []).find(s =>
    (s.type || '').toLowerCase() === slug.toLowerCase() && s.id
  );
  const draftId = draft ? draft.id : null;

  try {
    // 2) Move the draft if it exists
    if (draftId) {
      const fd = new FormData();
      fd.append('chapter_id', tgtChId);
      const r = await fetch('/api/draft/' + draftId + '/chapter', {method: 'PUT', body: fd});
      if (!r.ok) throw new Error('draft move failed (' + r.status + ')');
    }

    // 3) Remove slug from source chapter's sections
    const srcMeta = (srcCh.sections_meta || []).filter(s => s.slug !== slug);
    const srcSections = srcMeta.map(s => ({
      slug: s.slug, title: s.title || s.slug, plan: s.plan || '',
      target_words: (s.target_words && s.target_words > 0) ? s.target_words : null,
    }));
    await fetch('/api/chapters/' + srcChId + '/sections', {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({sections: srcSections}),
    });

    // 4) Add slug to target chapter's sections at the right position
    const movedEntry = (srcCh.sections_meta || []).find(s => s.slug === slug) || {slug: slug, title: slug, plan: ''};
    const tgtMeta = (tgtCh.sections_meta || []).map(s => ({
      slug: s.slug, title: s.title || s.slug, plan: s.plan || '',
      target_words: (s.target_words && s.target_words > 0) ? s.target_words : null,
    }));
    const insertIdx = tgtMeta.findIndex(s => s.slug === targetSlug);
    const newEntry = {
      slug: movedEntry.slug, title: movedEntry.title || movedEntry.slug,
      plan: movedEntry.plan || '',
      target_words: (movedEntry.target_words && movedEntry.target_words > 0)
        ? movedEntry.target_words : null,
    };
    if (insertIdx >= 0) {
      tgtMeta.splice(position === 'before' ? insertIdx : insertIdx + 1, 0, newEntry);
    } else {
      tgtMeta.push(newEntry);
    }
    await fetch('/api/chapters/' + tgtChId + '/sections', {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({sections: tgtMeta}),
    });

    // 5) Refresh sidebar
    const sidebarRes = await fetch('/api/chapters');
    const sd = await sidebarRes.json();
    if (Array.isArray(sd.chapters)) chaptersData = sd.chapters;
    rebuildSidebar(sd.chapters || sd, currentDraftId);
  } catch (e) {
    alert('Cross-chapter move failed: ' + e.message);
  }
}


// Wire drag-and-drop handlers via event delegation on the sidebar
// container. Idempotent — re-running attaches a NEW listener but the
// old ones are still there. We only call this once on DOMContentLoaded
// (the container element is stable; only its children get replaced
// by rebuildSidebar, but event delegation handles that automatically).
let _sectionDndWired = false;
function setupSectionDragDrop() {
  if (_sectionDndWired) return;
  const container = document.getElementById('sidebar-sections');
  if (!container) return;
  container.addEventListener('dragstart', handleSectionDragStart);
  container.addEventListener('dragover', handleSectionDragOver);
  container.addEventListener('drop', handleSectionDrop);
  container.addEventListener('dragend', handleSectionDragEnd);
  // Also clear the drop indicator if the user drags outside the container
  container.addEventListener('dragleave', e => {
    // Only clear when leaving the whole sidebar, not jumping between rows
    if (!container.contains(e.relatedTarget)) {
      document.querySelectorAll('.sec-link.drag-over-top, .sec-link.drag-over-bottom')
        .forEach(el => el.classList.remove('drag-over-top', 'drag-over-bottom'));
    }
  });
  _sectionDndWired = true;
}
document.addEventListener('DOMContentLoaded', setupSectionDragDrop);

// Phase 25 — adopt an orphan draft's slug into the chapter's sections
// list. Idempotent: if the slug already exists, the server returns
// added=false and we just refresh the sidebar (which is a no-op
// visually). On success the orphan re-classifies as drafted.
async function adoptOrphanSection(chapterId, slug) {
  if (!chapterId || !slug) return;
  try {
    const res = await fetch('/api/chapters/' + chapterId + '/sections/adopt', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({slug: slug}),
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error('adopt failed: ' + text);
    }
    const data = await res.json();
    // Update the in-memory chapter cache so the next sidebar render
    // reflects the new section without a full /api/chapters refetch.
    const ch = chaptersData.find(c => c.id === chapterId);
    if (ch && data.sections) {
      ch.sections_meta = data.sections;
      ch.sections_template = data.sections.map(s => s.slug);
    }
    // Refresh sidebar so the row re-renders as a drafted section.
    const sidebarRes = await fetch('/api/chapters');
    const sd = await sidebarRes.json();
    rebuildSidebar(sd.chapters || sd, currentDraftId);
  } catch (e) {
    alert('Adopt failed: ' + e.message);
  }
}

function addSection() {
  _editingSections.push({slug: '', title: '', plan: ''});
  renderSectionEditor();
  // Focus the new row's title input.
  setTimeout(() => {
    const rows = document.querySelectorAll('#ch-sections-list .sec-row');
    if (rows.length > 0) {
      const last = rows[rows.length - 1];
      const input = last.querySelector('input');
      if (input) input.focus();
    }
  }, 50);
}

// Phase 54.6.155 — LLM-auto-plan every section in the current chapter.
// Wraps `sciknow book plan-sections --chapter N` (54.6.154) as a single
// POST + JSON response; no SSE because per-section cost is small (~5-10s
// on LLM_FAST_MODEL) and total chapter latency typically stays under
// 30s. Refreshes the sections list on success so the new plans light
// up the live concept-count readout + resolver badge immediately.
async function autoPlanChapterSections() {
  const chId = document.getElementById('chapter-modal').dataset.chId;
  if (!chId) return;
  const force = document.getElementById('ch-auto-plan-force').checked;
  const status = document.getElementById('ch-auto-plan-status');
  // Commit any pending edits first — otherwise a user who renamed
  // sections in this modal would find the server writing plans keyed
  // by the old slugs.
  try {
    status.innerHTML = '<em>saving pending edits first…</em>';
    await saveChapterInfo();
  } catch (e) {
    status.innerHTML = '<span class="u-danger">save failed: '
                     + _escHtml(String(e).slice(0, 120)) + '</span>';
    return;
  }
  status.innerHTML = '<em>generating plans… (~5-10s per empty section)</em>';
  try {
    const fd = new FormData();
    if (force) fd.append('force', 'true');
    const res = await fetch('/api/chapters/' + chId + '/plan-sections', {
      method: 'POST', body: fd,
    });
    if (!res.ok) {
      const t = await res.text();
      status.innerHTML = '<span class="u-danger">failed: '
                        + _escHtml(t.slice(0, 200)) + '</span>';
      return;
    }
    const data = await res.json();
    status.innerHTML = '<span class="u-success">✓ planned '
                     + data.n_planned + ' section(s), '
                     + (data.n_skipped || 0) + ' skipped.</span> '
                     + '<span class="u-muted">'
                     + '(force overwrite is ' + (force ? 'on' : 'off') + ')</span>';
    // Re-open the modal against the same chapter so the newly-written
    // plans show up in the sections editor. openChapterModal re-loads
    // from the DB + resets _resolvedTargetsByChapter[chId] via a fresh
    // fetch so the resolver badges also update to concept_density.
    if (window._resolvedTargetsByChapter) {
      delete window._resolvedTargetsByChapter[chId];
    }
    openChapterModal(chId);
    // openChapterModal switches to the scope tab by default; flip back
    // to sections so the user sees the new plans.
    setTimeout(() => switchChapterTab('ch-sections'), 100);
  } catch (e) {
    status.innerHTML = '<span class="u-danger">error: '
                     + _escHtml(String(e).slice(0, 200)) + '</span>';
  }
}

function removeSection(idx) {
  if (idx < 0 || idx >= _editingSections.length) return;
  const s = _editingSections[idx];
  // Confirm if the section has content (non-empty title).
  if (s.title || s.plan) {
    if (!confirm('Delete section "' + (s.title || s.slug) + '"? This will not delete any existing drafts.')) return;
  }
  _editingSections.splice(idx, 1);
  renderSectionEditor();
}

function moveSection(idx, delta) {
  const j = idx + delta;
  if (j < 0 || j >= _editingSections.length) return;
  const tmp = _editingSections[idx];
  _editingSections[idx] = _editingSections[j];
  _editingSections[j] = tmp;
  renderSectionEditor();
}

function updateSection(idx, field, value) {
  if (idx < 0 || idx >= _editingSections.length) return;
  _editingSections[idx][field] = value;
}

// Phase 54.6.152 — live concept-count + target readout for the plan
// textarea. Client-side mirror of core.book_ops._count_plan_concepts
// + _get_section_concept_density_target so the user sees the resolver's
// output as they type. Soft warning when bullet count exceeds 4
// (Cowan 2001 novel-chunk capacity; consistent with the backend log
// in _get_section_concept_density_target).
const _PLAN_BULLET_RE = /^\s*(?:[-*•‣]|\d+\s*[.\)])\s+.{3,}/gm;

function _countPlanConceptsJS(text) {
  if (!text || !text.trim()) return 0;
  const m = text.match(_PLAN_BULLET_RE) || [];
  if (m.length > 0) return m.length;
  // Prose fallback: substantial lines >= 20 chars, capped at 6
  const lines = text.split('\n').filter(ln => ln.trim().length >= 20);
  return Math.min(lines.length, 6);
}

// Phase 54.6.163 — renderer that writes the concept-count readout into
// a given element. Used by both the Chapter-modal (54.6.152,
// idx-based) and the Plans-modal (slug-based) entry points.
function _renderPlanConceptReadout(el, text) {
  if (!el) return;
  const n = _countPlanConceptsJS(text || '');
  if (n <= 0) {
    el.innerHTML = '<em class="u-muted">No concepts detected yet — use bullet lines (<code>- concept</code>) or a few substantial sentences to activate concept-density sizing.</em>';
    return;
  }
  let wpcMid = 650;
  const bookType = (window._currentBookType || 'scientific_book');
  if (window._swBookTypes) {
    const t = window._swBookTypes.find(x => x.slug === bookType);
    if (t && t.words_per_concept_range) {
      const [lo, hi] = t.words_per_concept_range;
      wpcMid = Math.floor((lo + hi) / 2);
    }
  }
  const target = Math.max(400, n * wpcMid);
  const maxConcepts = 4;
  let warn = '';
  if (n > maxConcepts) {
    warn = ' <span class="u-warning">⚠ ' + n +
           ' concepts exceeds Cowan 2001 cap of ' + maxConcepts +
           ' — consider splitting.</span>';
  }
  el.innerHTML = '<strong>' + n + '</strong> concept' + (n === 1 ? '' : 's') +
                 ' × ' + wpcMid + ' wpc = <strong>~' + target.toLocaleString() + '</strong> words' +
                 warn;
}

// Phase 54.6.163 — wrapper used by the Plans modal (slug-keyed readouts)
function updatePlanConceptReadoutBySlug(slug, textarea) {
  const el = document.getElementById('plan-readout-slug-' + slug);
  _renderPlanConceptReadout(el, (textarea && textarea.value) || '');
}

function updatePlanConceptReadout(idx, textarea) {
  const el = document.getElementById('plan-readout-' + idx);
  if (!el) return;
  const text = (textarea && textarea.value) || '';
  const n = _countPlanConceptsJS(text);
  if (n <= 0) {
    el.innerHTML = '<em class="u-muted">No concepts detected yet — use bullet lines (<code>- concept</code>) or a few substantial sentences to activate concept-density sizing.</em>';
    return;
  }
  // Pull wpc from the cached project-types registry (window._swBookTypes,
  // populated by swLoadBookTypes on wizard open / book-settings open).
  // Fall back to a sensible midpoint if the cache isn't warm yet.
  let wpcMid = 650;
  const bookType = (window._currentBookType || 'scientific_book');
  if (window._swBookTypes) {
    const t = window._swBookTypes.find(x => x.slug === bookType);
    if (t && t.words_per_concept_range) {
      const [lo, hi] = t.words_per_concept_range;
      wpcMid = Math.floor((lo + hi) / 2);
    }
  }
  const target = Math.max(400, n * wpcMid);
  // Ceiling check — Cowan's 3-4 novel-chunk cap. Expert-type carve-outs
  // (academic_monograph goes to 5) are server-side only; the client
  // warns at 4 uniformly to match the universal default. Over-ceilings
  // stay non-blocking — matches the backend's soft-warning policy.
  const maxConcepts = 4;
  let warn = '';
  if (n > maxConcepts) {
    warn = ' <span class="u-warning">⚠ ' + n +
           ' concepts exceeds Cowan 2001 cap of ' + maxConcepts +
           ' — consider splitting.</span>';
  }
  el.innerHTML = '<strong>' + n + '</strong> concept' + (n === 1 ? '' : 's') +
                 ' × ' + wpcMid + ' wpc = <strong>~' + target.toLocaleString() + '</strong> words' +
                 warn;
}

// Phase 37 — per-section model override input handler.  Blank string
// clears the override so the section falls back to the caller-provided
// model / global default. Trim, store; do NOT re-render (user is
// typing; re-render would eat focus).
function updateSectionModel(idx, value) {
  if (idx < 0 || idx >= _editingSections.length) return;
  _editingSections[idx].model = (value || '').trim();
}

async function saveChapterInfo() {
  const chId = document.getElementById('chapter-modal').dataset.chId;
  if (!chId) return;
  const title = document.getElementById('ch-title-input').value.trim();
  const desc = document.getElementById('ch-desc-input').value;
  const tq = document.getElementById('ch-tq-input').value.trim();
  const status = document.getElementById('chapter-modal-status');
  status.textContent = 'Saving...';

  const fd = new FormData();
  if (title) fd.append('title', title);
  fd.append('description', desc);
  fd.append('topic_query', tq);

  try {
    // 1) Save scope (existing endpoint)
    const res = await fetch('/api/chapters/' + chId, {method: 'PUT', body: fd});
    if (!res.ok) throw new Error('save failed');

    // 2) Save sections (Phase 18) — only sections with a non-empty
    // title are persisted. The server slugifies for us.
    // Phase 29 — also include target_words per section.
    const sectionsToSave = _editingSections
      .filter(s => (s.title || '').trim() || (s.slug || '').trim())
      .map(s => ({
        slug: (s.slug || '').trim() || (s.title || '').trim(),
        title: (s.title || '').trim() || (s.slug || ''),
        plan: (s.plan || '').trim(),
        target_words: (s.target_words && s.target_words > 0) ? s.target_words : null,
        // Phase 37 — per-section model override. Empty string is
        // persisted as null by _normalize_chapter_sections so the
        // section falls through to the caller/global default.
        model: (s.model || '').trim() || null,
      }));
    const secRes = await fetch('/api/chapters/' + chId + '/sections', {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({sections: sectionsToSave})
    });
    if (!secRes.ok) throw new Error('sections save failed');
    const secData = await secRes.json();

    // Update the in-memory chapter cache
    const ch = chaptersData.find(c => c.id === chId);
    if (ch) {
      if (title) ch.title = title;
      ch.description = desc;
      ch.topic_query = tq;
      if (secData && Array.isArray(secData.sections)) {
        ch.sections_meta = secData.sections;
        ch.sections_template = secData.sections.map(s => s.slug);
      }
    }
    status.innerHTML = '<span class="u-success">Saved.</span>';
    // Refresh the sidebar so renamed chapters + new sections show up
    const sidebarRes = await fetch('/api/chapters');
    const sd = await sidebarRes.json();
    rebuildSidebar(sd.chapters, currentDraftId);
    setTimeout(() => closeModal('chapter-modal'), 800);
  } catch (e) {
    status.textContent = 'Save failed: ' + e.message;
  }
}

// ── Corkboard View ────────────────────────────────────────────────────
async function showCorkboard() {
  const res = await fetch('/api/corkboard');
  const data = await res.json();

  let html = '<div class="u-sys">';
  html += '<h2>Corkboard</h2>';
  html += '<p class="u-small u-dim u-mb-3">Click a card to navigate. Color = status.</p>';
  html += '<div class="corkboard">';

  data.cards.forEach(c => {
    const statusCls = c.status || 'to_do';
    // Phase 54.6.x — switched to data-action dispatch so section_type
    // values with apostrophes / colons stop breaking the attribute.
    const cardAttrs = c.draft_id
      ? ('data-action="cork-open-card" data-draft-id="' + c.draft_id + '"')
      : ('data-action="cork-open-card" data-chapter-id="' + c.chapter_id +
         '" data-sec-type="' + escapeHtml(c.section_type || '') + '"');
    html += '<div class="cork-card" ' + cardAttrs + '>';
    html += '<div class="cc-head"><span class="cc-ch">Ch.' + c.chapter_num + '</span>';
    html += '<span class="cc-status ' + statusCls + '">' + statusCls.replace('_', ' ') + '</span></div>';
    html += '<div class="cc-type">' + (c.section_title || (c.section_type.charAt(0).toUpperCase() + c.section_type.slice(1))) + '</div>';
    if (c.summary) {
      html += '<div class="cc-summary">' + c.summary + '</div>';
    } else {
      html += '<div class="cc-summary u-dim-3">' + (c.draft_id ? 'No summary' : 'Not started') + '</div>';
    }
    html += '<div class="cc-footer">';
    if (c.draft_id) html += 'v' + c.version + ' \u00b7 ' + c.words + 'w';
    html += '</div></div>';
  });

  html += '</div></div>';

  document.getElementById('dashboard-view').innerHTML = html;
  document.getElementById('dashboard-view').style.display = 'block';
  document.getElementById('read-view').style.display = 'none';
  document.getElementById('edit-view').style.display = 'none';
  document.getElementById('draft-title').textContent = 'Corkboard';
  document.getElementById('draft-subtitle').style.display = 'none';
  document.getElementById('stream-panel').style.display = 'none';
  document.getElementById('version-panel').style.display = 'none';
  document.querySelectorAll('.sec-link').forEach(l => l.classList.remove('active'));
  history.pushState({corkboard: true}, '', '/');
}

// ── Chapter Reader (continuous scroll) ────────────────────────────────
async function showChapterReader() {
  if (!currentChapterId) { showEmptyHint("No chapter selected &mdash; click any chapter title in the sidebar to select it, then try again."); return; }

  // Phase 31 — context-aware Read button. If the user has a section
  // selected, fetch only that section in the reader layout (same h2
  // styling, sources panel, citation popovers — just one section).
  // If only the chapter is selected, show the whole chapter.
  const url = currentSectionType
    ? ('/api/chapter-reader/' + currentChapterId + '?only_section=' + encodeURIComponent(currentSectionType))
    : ('/api/chapter-reader/' + currentChapterId);
  const res = await fetch(url);
  if (!res.ok) { alert('Chapter not found.'); return; }
  const data = await res.json();

  const isSectionOnly = !!currentSectionType;
  let html = '<div class="reader-view">';
  html += '<h1>Chapter ' + data.chapter_num + ': ' + data.chapter_title + '</h1>';
  html += '<p class="u-md u-dim u-mb-2">' +
    data.total_words + ' words \u00b7 ' + data.section_count + ' section' +
    (data.section_count === 1 ? '' : 's') +
    (isSectionOnly ? ' &middot; <strong>showing only ' + escapeHtml(currentSectionType) + '</strong>' : '') +
    '</p>';
  // Phase 18 — outline / table-of-contents at the top of the chapter
  // view so the user can see the section structure at a glance and
  // jump straight to any section.
  if (data.outline && data.outline.length > 0) {
    html += '<div style="margin-bottom:24px;padding:12px 16px;background:var(--toolbar-bg);border-radius:6px;font-size:13px;">';
    html += '<div class="u-semibold u-mb-6 u-muted u-tiny u-upper u-ls-sm">Sections</div>';
    data.outline.forEach((o, i) => {
      html += '<div class="u-my-1"><a class="u-accent u-no-underline" href="#reader-section-' + o.slug + '">' + (i + 1) + '. ' + escapeHtml(o.title) + '</a> <span class="u-hint">' + o.words + 'w</span></div>';
    });
    html += '</div>';
  }
  html += data.html;
  html += '</div>';

  document.getElementById('dashboard-view').innerHTML = html;
  document.getElementById('dashboard-view').style.display = 'block';
  document.getElementById('read-view').style.display = 'none';
  document.getElementById('edit-view').style.display = 'none';
  document.getElementById('draft-title').textContent = 'Ch.' + data.chapter_num + ': ' + data.chapter_title;
  document.getElementById('draft-subtitle').style.display = 'none';

  // Phase 18 — populate the right-hand sources panel with the chapter's
  // global (renumbered) source list so [N] click-to-source works in
  // the chapter reader view. Without this, panel-sources still has
  // whatever section was last loaded — and the global citation
  // numbers would point at the wrong papers.
  if (data.sources_html) {
    document.getElementById('panel-sources').innerHTML = data.sources_html;
  }

  history.pushState({reader: true}, '', '/');

  // Build popovers for citations in reader view
  setTimeout(buildPopovers, 100);
}

// ── Phase 39: consolidated Book Settings modal ───────────────────────
// Brings title/description/plan/target_chapter_words/style_fingerprint
// into one editor. All fields round-trip through existing endpoints:
// GET /api/book for reads, PUT /api/book for writes, and a new
// POST /api/book/style-fingerprint/refresh for the style tab.
async function openBookSettings() {
  openModal('book-settings-modal');
  switchBookSettingsTab('bs-basics');
  // Phase 54.6.148 — reuse the wizard's book-types loader for the
  // Basics-tab dropdown + info panel. swLoadBookTypes caches in
  // window._swBookTypes so the subsequent bsUpdateTypeInfo() reads
  // without a second fetch.
  await swLoadBookTypes();
  await populateBookSettingsTypeDropdown();
  await loadBookSettings();
  // Phase 54.6.x — wire debounced autosave on the Book Settings
  // modal's Basics + Leitmotiv inputs.
  _wireBookSettingsAutosave();
}

// Phase 54.6.148 — wire the Basics-tab book-type dropdown. Same
// registry that powers the wizard (window._swBookTypes), just pointed
// at a different `<select>` id. Kept separate from swLoadBookTypes so
// the two dropdowns don't fight over the `<option>` list.
async function populateBookSettingsTypeDropdown() {
  const sel = document.getElementById('bs-book-type');
  if (!sel || !window._swBookTypes) return;
  sel.innerHTML = '';
  for (const t of window._swBookTypes) {
    const opt = document.createElement('option');
    opt.value = t.slug;
    opt.textContent = `${t.display_name} (${t.default_target_chapter_words.toLocaleString()} words/chap)`;
    sel.appendChild(opt);
  }
}

function bsUpdateTypeInfo() {
  const panel = document.getElementById('bs-book-type-info');
  const sel = document.getElementById('bs-book-type');
  if (!panel || !sel || !window._swBookTypes) return;
  const t = window._swBookTypes.find(x => x.slug === sel.value);
  if (!t) { panel.innerHTML = ''; return; }
  const [clo, chi]   = t.concepts_per_section_range;
  const [wlo, whi]   = t.words_per_concept_range;
  const [slo, shi]   = t.section_at_midpoint_range;
  const chap         = t.default_target_chapter_words;
  const nchap        = t.default_chapter_count;
  const totalLo      = (chap * nchap * 0.7).toLocaleString();
  const totalHi      = (chap * nchap * 1.3).toLocaleString();
  const tcwInput     = document.getElementById('bs-target-chapter-words');
  if (tcwInput && !tcwInput.value) {
    tcwInput.placeholder = `(type default: ${chap.toLocaleString()})`;
  }
  panel.innerHTML = `
    <div class="u-grid-kv">
      <span class="u-muted">Description:</span>
      <span>${_escHtml(t.description)}</span>
      <span class="u-muted">Default chapters:</span>
      <span>${nchap} &middot; ${chap.toLocaleString()} words each
            ${t.is_flat ? '<em>(flat IMRaD — one chapter)</em>' : ''}</span>
      <span class="u-muted">Concepts / section:</span>
      <span>${clo}–${chi} novel chunks (Cowan 2001)</span>
      <span class="u-muted">Words / concept:</span>
      <span>${wlo}–${whi} (midpoint ${Math.floor((wlo + whi) / 2)})</span>
      <span class="u-muted">Section at midpoint:</span>
      <span>${slo.toLocaleString()}–${shi.toLocaleString()} words</span>
      <span class="u-muted">Typical book total:</span>
      <span>~${totalLo}–${totalHi} words</span>
    </div>
  `;
}

// Phase 54.6.162 — populate the Book Settings "Projected length report"
// panel. Wraps GET /api/book/length-report (which delegates to the
// 54.6.153 walk_book_lengths helper). Renders chapter-by-chapter
// collapsed view with per-section rows on expand.
async function loadBookLengthReportPanel() {
  const host = document.getElementById('bs-length-report-panel');
  if (!host) return;
  host.innerHTML = '<em class="u-muted">Loading…</em>';
  try {
    const r = await fetch('/api/book/length-report');
    if (!r.ok) {
      host.innerHTML = '<span class="u-danger">Failed: '
                      + r.status + '</span>';
      return;
    }
    const d = await r.json();
    const levelColour = {
      explicit_section_override: 'var(--success)',
      concept_density:           'var(--accent)',
      chapter_split:             'var(--fg-muted)',
    };
    // Aggregate header
    const hist = {};
    for (const c of (d.chapters || [])) {
      for (const s of (c.sections || [])) {
        hist[s.level] = (hist[s.level] || 0) + 1;
      }
    }
    let html = '<div class="u-mb-6">'
             + '<strong>' + (d.total_words || 0).toLocaleString() + '</strong> projected words'
             + '  ·  ' + (d.n_chapters || 0) + ' chapter(s)'
             + '  ·  ' + (d.n_sections || 0) + ' section(s)'
             + '  ·  book type <code>' + _escHtml(d.book_type || '') + '</code>'
             + '</div>';
    const histBits = Object.entries(hist).map(
      ([lvl, n]) => '<span style="color:' + (levelColour[lvl] || 'var(--fg)') + ';">'
                    + n + ' ' + lvl + '</span>'
    );
    if (histBits.length) {
      html += '<div class="u-mb-2 u-muted">Levels: '
            + histBits.join('  ·  ') + '</div>';
    }
    for (const c of (d.chapters || [])) {
      html += '<details class="u-mb-1">'
            + '<summary style="cursor:pointer;padding:3px 0;">'
            + '<strong>Ch.' + c.number + '</strong> '
            + _escHtml(c.title || '')
            + '  ·  <span class="u-accent">' + (c.total_words || 0).toLocaleString() + '</span> words'
            + '  <span class="u-muted">(' + (c.sections ? c.sections.length : 0) + ' sections · target ' + (c.chapter_target || 0).toLocaleString() + ' ' + (c.chapter_level || '') + ')</span>'
            + '</summary>';
      html += '<table style="width:100%;border-collapse:collapse;margin-left:12px;margin-top:4px;">';
      for (const s of (c.sections || [])) {
        const colour = levelColour[s.level] || 'var(--fg)';
        html += '<tr>'
              + '<td class="u-p-2-6 u-muted u-mono-sys u-xxs">' + _escHtml((s.slug || '').slice(0, 28)) + '</td>'
              + '<td class="u-p-2-6">' + _escHtml((s.title || '').slice(0, 36)) + '</td>'
              + '<td class="u-p-2-6 u-text-right">' + (s.target || 0).toLocaleString() + '</td>'
              + '<td style="padding:2px 6px;color:' + colour + ';">' + _escHtml(s.level || '') + '</td>'
              + '<td class="u-p-2-6 u-muted">' + _escHtml((s.explanation || '').slice(0, 60)) + '</td>'
              + '</tr>';
      }
      html += '</table></details>';
    }
    host.innerHTML = html;
  } catch (e) {
    host.innerHTML = '<span class="u-danger">Error: '
                    + _escHtml(String(e).slice(0, 200)) + '</span>';
  }
}

// Phase 54.6.159 — populate the Book Settings "Corpus section-length
// distribution" panel. Wraps GET /api/bench/section-lengths, which
// delegates to the 54.6.157 bench function. Alignment tags are
// colour-coded: aligned=green, shorter/longer-skewed=warning,
// below/above-range=danger.
async function loadSectionLengthPanel() {
  const host = document.getElementById('bs-section-length-panel');
  if (!host) return;
  host.innerHTML = '<em class="u-muted">Loading…</em>';
  try {
    const r = await fetch('/api/bench/section-lengths');
    if (!r.ok) {
      host.innerHTML = '<span class="u-danger">Failed: '
                      + r.status + '</span>';
      return;
    }
    const d = await r.json();
    if (!d.sections || d.sections.length === 0) {
      host.innerHTML = '<em class="u-muted">'
                      + 'No section data yet — ingest some papers first.</em>';
      return;
    }
    const colourFor = (tag) => {
      if (!tag) return 'var(--fg-muted)';
      if (tag === 'aligned') return 'var(--success)';
      if (tag === 'shorter-skewed' || tag === 'longer-skewed') return 'var(--warning)';
      return 'var(--danger)';
    };
    let html = '<table class="u-w-full u-bcollapse u-tiny">';
    html += '<thead><tr class="u-border-b">'
         + '<th class="u-cell-sm">Section</th>'
         + '<th class="u-cell-right-sm">n</th>'
         + '<th class="u-cell-right-sm">Median</th>'
         + '<th class="u-cell-right-sm">IQR</th>'
         + '<th class="u-cell-sm">§24 Ref</th>'
         + '<th class="u-cell-sm">Alignment</th>'
         + '</tr></thead><tbody>';
    for (const s of d.sections) {
      const colour = colourFor(s.alignment);
      html += '<tr class="u-border-b">'
           + '<td class="u-p-4-6 u-semibold">' + _escHtml(s.section_type) + '</td>'
           + '<td class="u-p-4-6 u-text-right u-muted">' + (s.n || '—') + '</td>'
           + '<td class="u-p-4-6 u-text-right">' + ((s.median || 0).toLocaleString()) + 'w</td>'
           + '<td class="u-p-4-6 u-text-right u-mono-sys">' + _escHtml(s.iqr || '—') + '</td>'
           + '<td class="u-p-4-6 u-muted">' + _escHtml(s.ref_iqr || '—') + '</td>'
           + '<td style="padding:4px 6px;color:' + colour + ';">' + _escHtml(s.alignment || '—') + '</td>'
           + '</tr>';
    }
    html += '</tbody></table>';
    html += '<p class="u-mt-6 u-muted">'
         + 'IQR = interquartile range. Reference IQRs are PubMed N=61,517 per '
         + '<code>RESEARCH.md §24</code>. <em>aligned</em> = corpus median '
         + 'sits inside the reference IQR; <em>shorter/longer-skewed</em> = '
         + 'overlap but median is outside.</p>';
    host.innerHTML = html;
  } catch (e) {
    host.innerHTML = '<span class="u-danger">Error: '
                    + _escHtml(String(e).slice(0, 200)) + '</span>';
  }
}

// Phase 54.6.156 — book-wide auto-plan wrapper. Streams
// `sciknow book plan-sections <book_id>` via /api/cli-stream and
// pipes log + status into the Basics tab's dedicated elements
// (runCorpusCliAction targets different DOM ids). Reuses the
// allowlisted CLI so the backend stays identical to the Chapter
// modal's per-chapter button (54.6.155) — same generator, same
// LLM prompt, same safety checks.
async function autoPlanEntireBook() {
  const status = document.getElementById('bs-plan-book-status');
  const logEl  = document.getElementById('bs-plan-book-log');
  const force  = document.getElementById('bs-plan-book-force').checked;

  // Resolve the current book's id so the CLI subprocess targets the
  // right project even when `book serve` is running.
  let bookId = '';
  try {
    const r = await fetch('/api/book');
    const d = await r.json();
    bookId = d.id || '';
  } catch (_) {}
  if (!bookId) {
    status.innerHTML = '<span class="u-danger">Failed to resolve current book.</span>';
    return;
  }

  // CLI accepts a UUID prefix via its ILIKE match.
  const argv = ['book', 'plan-sections', bookId];
  if (force) argv.push('--force');

  status.innerHTML = '<em>Starting book-wide auto-plan… (typical book ~4-8 min)</em>';
  logEl.style.display = 'block';
  logEl.textContent = '';

  try {
    const res = await fetch('/api/cli-stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({argv: argv}),
    });
    const d = await res.json();
    if (!res.ok || !d.job_id) {
      status.innerHTML = '<span class="u-danger">Failed: '
                       + _escHtml(d.detail || ('status ' + res.status)) + '</span>';
      return;
    }
    const src = new EventSource('/api/stream/' + d.job_id);
    src.onmessage = function(e) {
      const evt = JSON.parse(e.data);
      if (evt.type === 'log') {
        logEl.textContent += evt.text + '\n';
        logEl.scrollTop = logEl.scrollHeight;
      } else if (evt.type === 'completed') {
        status.innerHTML = '<span class="u-success">✓ Done.</span> '
                         + '<span class="u-muted">'
                         + 'Open a chapter to see the new plans + concept-density badges.</span>';
        src.close();
      } else if (evt.type === 'error') {
        status.innerHTML = '<span class="u-danger">✗ '
                         + _escHtml(evt.message || 'error') + '</span>';
        src.close();
      } else if (evt.type === 'done') {
        src.close();
      }
    };
    src.onerror = function() {
      status.innerHTML = '<span class="u-danger">Connection lost.</span>';
      src.close();
    };
  } catch (e) {
    status.innerHTML = '<span class="u-danger">Error: '
                     + _escHtml(String(e).slice(0, 200)) + '</span>';
  }
}

// Phase 54.6.170 — Command palette. Cmd/Ctrl+K opens a modal input
// and filters a registry of actions. Arrow keys move the highlight;
// Enter runs; Esc closes. Each command resolves its action by
// invoking a global function by name — so the palette naturally
// tracks refactors and never calls a stale function reference that
// cached at definition time.
const _CMDK_COMMANDS = [
  // Write
  { id: 'edit',           label: 'Edit (toggle markdown editor)',   fn: 'toggleEdit',     group: 'write' },
  { id: 'autowrite',      label: 'AI Autowrite (write > review > revise loop)', fn: 'doAutowrite', group: 'write' },
  { id: 'write',          label: 'AI Write (single-pass draft)',    fn: 'doWrite',        group: 'write' },
  { id: 'review',         label: 'AI Review (critic pass)',         fn: 'doReview',       group: 'write' },
  { id: 'revise',         label: 'AI Revise (apply review feedback)', fn: 'doRevise',     group: 'write' },
  // Verify
  { id: 'verify',         label: 'Verify citations',                fn: 'doVerify',       group: 'verify' },
  { id: 'verify-draft',   label: 'Verify draft (claim atomization)', fn: 'doVerifyDraft', group: 'verify' },
  { id: 'finalize-draft', label: 'Finalize draft (L3 VLM)',         fn: 'doFinalizeDraft', group: 'verify' },
  { id: 'align-cites',    label: 'Align citations',                 fn: 'doAlignCitations', group: 'verify' },
  { id: 'insert-cites',   label: 'Insert citations',                fn: 'doInsertCitations', group: 'verify' },
  { id: 'scores',         label: 'Convergence scores panel',        fn: 'showScoresPanel', group: 'verify' },
  // Critique
  { id: 'argue',          label: 'Argue (map claim)',               fn: 'promptArgue',    group: 'critique' },
  { id: 'gaps',           label: 'Find gaps',                       fn: 'doGaps',         group: 'critique' },
  { id: 'adversarial',    label: 'Adversarial review',              fn: 'doAdversarialReview', group: 'critique' },
  { id: 'edge-cases',     label: 'Edge cases',                      fn: 'doEdgeCases',    group: 'critique' },
  { id: 'ensemble',       label: 'Ensemble review',                 fn: 'doEnsembleReview', group: 'critique' },
  // Extras
  { id: 'bundles',        label: 'Bundles / snapshots',             fn: 'openBundleSnapshots', group: 'extras' },
  { id: 'chapter-reader', label: 'Chapter reader',                  fn: 'showChapterReader', group: 'extras' },
  // Navigate
  { id: 'plan',           label: 'Open Plan',                       fn: 'openPlanModal',  group: 'navigate' },
  { id: 'dashboard',      label: 'Open Dashboard',                  fn: 'showDashboard',  group: 'navigate' },
  { id: 'wiki',           label: 'Open Compiled Knowledge Wiki',    fn: 'openWikiModal',  group: 'navigate' },
  { id: 'ask',            label: 'Ask the Corpus (RAG)',            fn: 'openAskModal',   group: 'navigate' },
  // Settings
  { id: 'book-settings',  label: 'Book Settings',                   fn: 'openBookSettings',   group: 'settings' },
  { id: 'projects',       label: 'Projects',                        fn: 'openProjectsModal',  group: 'settings' },
  { id: 'tools',          label: 'Tools · CLI parity',              fn: 'openToolsModal',     group: 'settings' },
  { id: 'setup',          label: 'Setup Wizard',                    fn: 'openSetupWizard',    group: 'settings' },
  { id: 'backups',        label: 'Backups',                         fn: 'openBackupsModal',   group: 'settings' },
  { id: 'export',         label: 'Export book',                     fn: 'openExportModal',    group: 'settings' },
  { id: 'catalog',        label: 'Browse Papers',                   fn: 'openCatalogModal',   group: 'navigate' },
  { id: 'visuals',        label: 'Visuals (Tables/Figs/Eqs)',       fn: 'openVisualsModal',   group: 'navigate' },
  { id: 'corkboard',      label: 'Corkboard',                       fn: 'showCorkboard',      group: 'navigate' },
  { id: 'monitor',        label: 'System Monitor (live pipeline + GPU + models)', fn: 'openMonitorModal', group: 'navigate' },
  { id: 'help',           label: 'AI actions help',                 fn: 'openAIActionsHelp',  group: 'settings' },
];
let _cmdkSelected = 0;
let _cmdkFiltered = [];

function openCmdK() {
  const cmdk = document.getElementById('cmdk');
  if (!cmdk) return;
  cmdk.style.display = 'block';
  const input = document.getElementById('cmdk-input');
  input.value = '';
  _cmdkRenderList('');
  setTimeout(function () { input.focus(); }, 10);
}
function closeCmdK() {
  const cmdk = document.getElementById('cmdk');
  if (cmdk) cmdk.style.display = 'none';
}
function _cmdkMatch(cmd, q) {
  if (!q) return true;
  const hay = (cmd.label + ' ' + cmd.group).toLowerCase();
  // Subsequence match — all query chars appear in order. Cheap &
  // forgiving: "aw" matches "AI Autowrite" via a-w.
  let i = 0;
  for (const c of hay) {
    if (c === q[i]) i++;
    if (i === q.length) return true;
  }
  return false;
}
function _cmdkRenderList(q) {
  q = (q || '').trim().toLowerCase();
  _cmdkFiltered = _CMDK_COMMANDS.filter(c => _cmdkMatch(c, q));
  _cmdkSelected = 0;
  const list = document.getElementById('cmdk-list');
  if (!list) return;
  if (_cmdkFiltered.length === 0) {
    list.innerHTML = '<div class="cmdk-empty">No commands match.</div>';
    return;
  }
  list.innerHTML = _cmdkFiltered.map(function (c, i) {
    return '<div class="cmdk-item' + (i === 0 ? ' is-selected' : '')
      + '" data-idx="' + i + '" onclick="_cmdkRun(' + i + ')">'
      + '<span class="cmdk-label">' + _escHtml(c.label) + '</span>'
      + '<span class="cmdk-desc">' + _escHtml(c.group) + '</span>'
      + '</div>';
  }).join('');
}
function _cmdkMove(delta) {
  if (_cmdkFiltered.length === 0) return;
  _cmdkSelected = (_cmdkSelected + delta + _cmdkFiltered.length) % _cmdkFiltered.length;
  document.querySelectorAll('.cmdk-item').forEach(function (el, i) {
    el.classList.toggle('is-selected', i === _cmdkSelected);
    if (i === _cmdkSelected) el.scrollIntoView({ block: 'nearest' });
  });
}
function _cmdkRun(idx) {
  const cmd = _cmdkFiltered[typeof idx === 'number' ? idx : _cmdkSelected];
  if (!cmd) return;
  closeCmdK();
  // Resolve the function by name at call time — so the palette
  // never holds a stale reference across SPA navigation.
  setTimeout(function () {
    try {
      const f = window[cmd.fn];
      if (typeof f === 'function') f();
      else console.warn('[cmdk] function not found:', cmd.fn);
    } catch (e) { console.error('[cmdk] action error:', e); }
  }, 10);
}
document.addEventListener('keydown', function (e) {
  const open = document.getElementById('cmdk') && document.getElementById('cmdk').style.display === 'block';
  // Open anywhere with Cmd/Ctrl+K — don't swallow if a text input
  // is focused UNLESS the palette itself is already open.
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
    e.preventDefault();
    if (open) closeCmdK(); else openCmdK();
    return;
  }
  if (!open) return;
  if (e.key === 'Escape')    { e.preventDefault(); closeCmdK(); return; }
  if (e.key === 'ArrowDown') { e.preventDefault(); _cmdkMove(1);  return; }
  if (e.key === 'ArrowUp')   { e.preventDefault(); _cmdkMove(-1); return; }
  if (e.key === 'Enter')     { e.preventDefault(); _cmdkRun();    return; }
});
document.addEventListener('input', function (e) {
  if (e.target && e.target.id === 'cmdk-input') _cmdkRenderList(e.target.value);
});

// Phase 54.6.167 — context rail tab switcher. One of sources / review
// / comments is visible at a time; last choice is remembered in
// localStorage so switching drafts keeps the user's rail preference.
function switchContextTab(name) {
  const panes = ['sources', 'review', 'comments', 'visuals'];
  if (!panes.includes(name)) name = 'sources';
  panes.forEach(n => {
    const btn = document.querySelector('.panel-seg-btn[data-ctx="' + n + '"]');
    const pane = document.getElementById('panel-pane-' + n);
    const on = (n === name);
    if (btn) {
      btn.classList.toggle('is-active', on);
      btn.setAttribute('aria-selected', on ? 'true' : 'false');
    }
    if (pane) pane.classList.toggle('is-active', on);
  });
  try { localStorage.setItem('sciknow.ui.contextTab', name); } catch (e) {}
  // Phase 54.6.312 — visuals pane: NEVER auto-rank on tab open. Load the
  // cached ranking (if any) for the current draft and let the user press
  // the Rank / Re-rank button explicitly. The ranker costs ~1–2s and the
  // result is persisted per-draft so subsequent opens are instant.
  if (name === 'visuals') {
    const pane = document.getElementById('panel-visuals');
    const did = (typeof currentDraftId !== 'undefined' ? currentDraftId : '') || '';
    if (pane && pane.dataset.loadedFor !== did) {
      loadCachedVisualSuggestions();
    }
  }
}

// Phase 54.6.312 — visual suggestions panel. Three entry points:
//
//   loadCachedVisualSuggestions()  — on tab-open, GET without compute=1.
//                                    Renders the saved ranking (if any);
//                                    otherwise shows a "click Rank" hint.
//   rankVisualSuggestions()        — the Rank / Re-rank button: POST to
//                                    /api/visuals/suggestions, which
//                                    computes + persists to the draft's
//                                    custom_metadata. Button text flips
//                                    to "Re-rank" once a ranking exists.
//   clearVisualSuggestions()       — DELETE the saved ranking.
//
// The pane supports gallery + list view modes and a click-to-enlarge
// lightbox (#visuals-lightbox) so the small thumbnails can be inspected.
let _visualsPaneView = 'gallery';
try {
  _visualsPaneView = localStorage.getItem('sciknow.ui.visualsPaneView') || 'gallery';
} catch (e) {}

function setVisualsPaneView(mode) {
  _visualsPaneView = (mode === 'list') ? 'list' : 'gallery';
  try { localStorage.setItem('sciknow.ui.visualsPaneView', _visualsPaneView); } catch (e) {}
  document.getElementById('vis-pane-view-gallery')?.classList.toggle('is-active', _visualsPaneView === 'gallery');
  document.getElementById('vis-pane-view-list')?.classList.toggle('is-active', _visualsPaneView === 'list');
  const pane = document.getElementById('panel-visuals');
  if (pane) {
    pane.classList.toggle('visuals-mode--gallery', _visualsPaneView === 'gallery');
    pane.classList.toggle('visuals-mode--list', _visualsPaneView === 'list');
  }
}

function _visualsPaneButtons(hasRanking) {
  const rankBtn = document.getElementById('visuals-rank-btn');
  const clearBtn = document.getElementById('visuals-clear-btn');
  if (rankBtn) rankBtn.textContent = hasRanking ? 'Re-rank' : 'Rank';
  if (clearBtn) clearBtn.classList.toggle('u-hidden', !hasRanking);
}

function _renderVisualsPane(data, opts) {
  const pane = document.getElementById('panel-visuals');
  const hint = document.getElementById('panel-visuals-hint');
  if (!pane) return;
  opts = opts || {};
  pane.classList.add(_visualsPaneView === 'list' ? 'visuals-mode--list' : 'visuals-mode--gallery');
  pane.classList.remove(_visualsPaneView === 'list' ? 'visuals-mode--gallery' : 'visuals-mode--list');
  if (!data || !data.hits || data.hits.length === 0) {
    const msg = (data && data.note)
      ? data.note
      : (opts.placeholder || 'No ranking saved for this draft. Click Rank to compute.');
    pane.innerHTML = '<em>' + escapeHtml(msg) + '</em>';
    return;
  }
  const stale = opts.stale
    ? ' <span class="visual-tag" title="The draft content has changed since this ranking was computed. Re-rank for fresh results.">stale</span>'
    : '';
  const stamp = opts.ranked_at
    ? '<div class="panel-visuals-stamp" title="When this ranking was computed.">ranked ' + escapeHtml(opts.ranked_at.replace('T', ' ').replace('Z', ' UTC')) + stale + '</div>'
    : '';
  let html = stamp + '<ul class="visual-suggestions">';
  data.hits.forEach((v) => {
    const cap = escapeHtml(v.caption || '(no caption)');
    const paperTitle = escapeHtml(v.paper_title || '');
    const kind = escapeHtml(v.kind || '');
    const figNum = escapeHtml(v.figure_num || '');
    const score = (v.composite_score || 0).toFixed(2);
    const samePaper = v.same_paper
      ? '<span class="visual-tag visual-tag--same" title="Comes from a paper already cited by this draft.">cited</span>'
      : '';
    const imgUrl = v.image_url || '';
    const img = imgUrl
      ? '<img class="visual-thumb" loading="lazy" src="' + escapeHtml(imgUrl) +
        '" data-full-src="' + escapeHtml(imgUrl) +
        '" data-caption="' + cap +
        '" onclick="openVisualLightbox(this)" alt="' + cap + '" title="Click to enlarge — ' + cap + '">'
      : '<div class="visual-thumb visual-thumb--text" title="No rendered image for this kind (' + kind + ').">[' + kind + ']</div>';
    html += '<li class="visual-row" data-visual-id="' + escapeHtml(v.visual_id) + '">' +
      img +
      '<div class="visual-meta">' +
        '<div class="visual-caption">' + cap + '</div>' +
        '<div class="visual-sub"><span class="visual-src" title="' + paperTitle + '">' + paperTitle + '</span> ' +
        '<span class="visual-tag">' + kind + (figNum ? ' ' + figNum : '') + '</span> ' +
        samePaper +
        '<span class="visual-score" title="Composite ranker score (caption + mention + same-paper + section prior).">score ' + score + '</span></div>' +
        '<button class="visual-insert btn-sm" onclick="insertVisualIntoDraft(\'' + v.visual_id + '\')" title="Append an image reference + caption to the end of this draft. The image is served from /api/visuals/image/{id}.">Insert</button>' +
      '</div>' +
    '</li>';
  });
  html += '</ul>';
  pane.innerHTML = html;
}

async function loadCachedVisualSuggestions() {
  const pane = document.getElementById('panel-visuals');
  const did = (typeof currentDraftId !== 'undefined' ? currentDraftId : '') || '';
  if (!pane) return;
  setVisualsPaneView(_visualsPaneView);
  if (!did) {
    pane.innerHTML = '<em>Open a drafted section first.</em>';
    _visualsPaneButtons(false);
    return;
  }
  try {
    const res = await fetch('/api/visuals/suggestions?draft_id=' + encodeURIComponent(did) + '&limit=12');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    pane.dataset.loadedFor = did;
    const hasRanking = !!(data.cached && data.hits && data.hits.length);
    _visualsPaneButtons(hasRanking);
    _renderVisualsPane(data, { stale: data.stale, ranked_at: data.ranked_at });
  } catch (e) {
    pane.innerHTML = '<em>Failed to load saved ranking: ' + escapeHtml(String(e)) + '</em>';
    _visualsPaneButtons(false);
  }
}

async function rankVisualSuggestions() {
  const pane = document.getElementById('panel-visuals');
  const btn = document.getElementById('visuals-rank-btn');
  const did = (typeof currentDraftId !== 'undefined' ? currentDraftId : '') || '';
  if (!pane) return;
  if (!did) {
    pane.innerHTML = '<em>Open a drafted section first.</em>';
    return;
  }
  if (btn) { btn.disabled = true; btn.textContent = 'Ranking…'; }
  pane.innerHTML = '<em>Ranking visuals against this draft…</em>';
  try {
    const res = await fetch('/api/visuals/suggestions?draft_id=' + encodeURIComponent(did) + '&limit=12', {
      method: 'POST',
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    pane.dataset.loadedFor = did;
    _renderVisualsPane(data, { stale: false, ranked_at: data.ranked_at });
    _visualsPaneButtons(!!(data.hits && data.hits.length));
  } catch (e) {
    pane.innerHTML = '<em>Failed to rank: ' + escapeHtml(String(e)) + '</em>';
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function clearVisualSuggestions() {
  const did = (typeof currentDraftId !== 'undefined' ? currentDraftId : '') || '';
  if (!did) return;
  if (!confirm('Delete the saved ranking for this draft? The next Rank press will recompute from scratch.')) return;
  try {
    await fetch('/api/visuals/suggestions?draft_id=' + encodeURIComponent(did), { method: 'DELETE' });
    await loadCachedVisualSuggestions();
  } catch (e) {
    alert('Could not clear: ' + e);
  }
}

// Click-to-enlarge lightbox. The <img> in the panel carries
// data-full-src + data-caption so the modal can read them without a
// second round-trip.
function openVisualLightbox(imgEl) {
  const src = imgEl?.dataset?.fullSrc || imgEl?.src;
  const cap = imgEl?.dataset?.caption || imgEl?.alt || '';
  const modal = document.getElementById('visuals-lightbox');
  if (!modal) return;
  const mImg = document.getElementById('visuals-lightbox-img');
  const mCap = document.getElementById('visuals-lightbox-cap');
  if (mImg) mImg.src = src;
  if (mCap) mCap.innerHTML = escapeHtml(cap);
  modal.classList.add('open');
}
function closeVisualLightbox() {
  const modal = document.getElementById('visuals-lightbox');
  if (modal) modal.classList.remove('open');
  const mImg = document.getElementById('visuals-lightbox-img');
  if (mImg) mImg.src = '';
}

// ── Phase 54.6.312 — Citation preview popup ──────────────────────────
// Click on an inline [N] in a draft → fetch the full publication
// metadata and render a rich card (title, authors, year, journal,
// abstract, open-access link). Shift/Cmd/Ctrl+click keeps the legacy
// scroll-to-source behaviour so the two UX don't collide.
async function openCitationPreview(n) {
  const modal = document.getElementById('citation-preview');
  const card = document.getElementById('citation-preview-card');
  if (!modal || !card) return;
  card.innerHTML = '<em>Loading citation [' + escapeHtml(String(n)) + ']…</em>';
  modal.classList.add('open');
  try {
    const res = await fetch('/api/bibliography/citation/' + encodeURIComponent(n));
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    const meta = data.metadata;
    const sourceLine = escapeHtml(data.source_line || '');
    if (!meta) {
      card.innerHTML = '<div class="cpv-head">Citation [' + escapeHtml(String(n)) + ']</div>'
        + '<div class="cpv-line">' + sourceLine + '</div>'
        + '<div class="cpv-note">Bibliographic record not found in paper_metadata — the source string above is what the draft stores.</div>';
      return;
    }
    const authors = Array.isArray(meta.authors) ? meta.authors.join(', ') : (meta.authors || '');
    const yr = meta.year ? ' (' + escapeHtml(String(meta.year)) + ')' : '';
    const jr = meta.journal ? '<div class="cpv-journal">' + escapeHtml(meta.journal) + '</div>' : '';
    const doi = meta.doi
      ? '<a class="cpv-link" href="https://doi.org/' + encodeURIComponent(meta.doi) + '" target="_blank" rel="noopener">doi:' + escapeHtml(meta.doi) + '</a>'
      : '';
    const oa = meta.open_access_url
      ? '<a class="cpv-link" href="' + escapeHtml(meta.open_access_url) + '" target="_blank" rel="noopener">Open access link</a>'
      : '';
    const abstract = meta.abstract ? '<div class="cpv-abstract">' + escapeHtml(meta.abstract) + '</div>' : '';
    card.innerHTML = '<div class="cpv-head">Citation [' + escapeHtml(String(n)) + ']</div>'
      + '<h3 class="cpv-title">' + escapeHtml(meta.title || '(untitled)') + yr + '</h3>'
      + '<div class="cpv-authors">' + escapeHtml(authors) + '</div>'
      + jr
      + abstract
      + '<div class="cpv-links">' + [doi, oa].filter(Boolean).join(' · ') + '</div>';
  } catch (e) {
    card.innerHTML = '<em>Failed to load citation: ' + escapeHtml(String(e)) + '</em>';
  }
}

function closeCitationPreview() {
  document.getElementById('citation-preview')?.classList.remove('open');
}

// ── Phase 54.6.312 — Bibliography audit + sort helpers ──────────────
async function runBibliographyAudit() {
  const out = document.getElementById('bib-tools-output');
  if (!out) return;
  out.textContent = 'Running sanity check…';
  try {
    const res = await fetch('/api/bibliography/audit');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    const totals = data.totals || {};
    const rows = data.rows || [];
    let msg = `Drafts checked: ${totals.drafts_checked || 0}\n`
      + `Broken citation refs: ${totals.broken || 0}\n`
      + `Orphan sources: ${totals.orphans || 0}\n`
      + `Duplicate source groups: ${totals.dupes || 0}\n\n`;
    if (!rows.length) {
      msg += 'No issues found. The bibliography is healthy.';
    } else {
      msg += `Drafts with issues (${rows.length}):\n`;
      rows.forEach(r => {
        const chLabel = r.chapter_num ? ('Ch.' + r.chapter_num + ': ' + (r.chapter_title || '')) : '(orphan)';
        msg += `\n• ${chLabel} — ${r.section_type || '?'} — ${r.title || ''}\n`;
        if (r.broken_refs && r.broken_refs.length) msg += `    broken refs: [${r.broken_refs.join(', ')}]\n`;
        if (r.orphan_sources && r.orphan_sources.length) msg += `    orphan sources: [${r.orphan_sources.join(', ')}]\n`;
        if (r.duplicate_groups && r.duplicate_groups.length) {
          msg += `    duplicates: `;
          msg += r.duplicate_groups.map(g => '[' + g.join(',') + ']').join(' ');
          msg += '\n';
        }
      });
      msg += '\nFix by pressing "Sort & renumber bibliography" — it re-aligns draft markdown with the global numbering.';
    }
    out.textContent = msg;
  } catch (e) {
    out.textContent = 'Sanity check failed: ' + e;
  }
}

async function runBibliographySort() {
  const out = document.getElementById('bib-tools-output');
  if (!out) return;
  if (!confirm('Rewrite every draft\'s [N] citations to match the global bibliography order? This flattens local→global numbers so the stored markdown equals what the reader shows. Idempotent, but backup via Snapshot first if you are nervous.')) return;
  out.textContent = 'Renumbering drafts…';
  try {
    const res = await fetch('/api/bibliography/sort', {method: 'POST'});
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    out.textContent = `Updated ${data.drafts_updated || 0} draft(s). `
      + `The bibliography now has ${data.total_global_sources || 0} unique sources. `
      + `Reload the reader to see the new numbers.`;
  } catch (e) {
    out.textContent = 'Sort failed: ' + e;
  }
}

function openBibliographyTools() { openModal('bib-tools-modal'); }

// Append an image reference for ``visualId`` to the draft body via the
// existing edit-in-place flow. If the draft is currently in edit mode
// we insert at the end of the textarea (caret would be preferable but
// the textarea is uncontrolled); otherwise we PATCH the content.
async function insertVisualIntoDraft(visualId) {
  const did = (typeof currentDraftId !== 'undefined' ? currentDraftId : '') || '';
  if (!did || !visualId) return;
  // Grab the suggestion row so we can pull its caption for the alt text.
  const row = document.querySelector('.visual-row[data-visual-id="' + visualId + '"] .visual-caption');
  const caption = (row ? row.textContent : '').trim() || ('Visual ' + visualId.slice(0, 8));
  const safeCap = caption.replace(/\|/g, '\\|').slice(0, 200);
  const snippet = '\n\n![' + safeCap + '](/api/visuals/image/' + visualId + ')\n\n*' + safeCap + '*\n';
  // If the editor is visible, append to the textarea and save.
  const ta = document.getElementById('edit-area');
  const editor = document.getElementById('edit-view');
  if (editor && editor.style.display !== 'none' && ta) {
    ta.value = (ta.value || '') + snippet;
    if (typeof edPreview === 'function') { try { edPreview(); } catch (e) {} }
    if (typeof edSave === 'function') {
      try { await edSave(); } catch (e) {}
    }
    return;
  }
  // Otherwise PATCH the draft's raw content via the edit endpoint.
  try {
    const curRes = await fetch('/api/section/' + encodeURIComponent(did));
    if (!curRes.ok) throw new Error('could not read current draft');
    const cur = await curRes.json();
    const next = (cur.content_raw || '') + snippet;
    const fd = new FormData();
    fd.append('content', next);
    const r = await fetch('/edit/' + encodeURIComponent(did), {method: 'POST', body: fd});
    if (!r.ok) throw new Error('edit failed (HTTP ' + r.status + ')');
    // Re-render.
    if (typeof loadSection === 'function') loadSection(did);
  } catch (e) {
    alert('Could not insert visual: ' + e);
  }
}

document.addEventListener('DOMContentLoaded', function () {
  let last = 'sources';
  try { last = localStorage.getItem('sciknow.ui.contextTab') || 'sources'; } catch (e) {}
  // Only switch if the rail is present (some pages — Setup, Projects —
  // render without the context rail).
  if (document.querySelector('.panel-seg')) switchContextTab(last);
});

function switchBookSettingsTab(name) {
  document.querySelectorAll('#book-settings-modal .tab').forEach(t => {
    t.classList.toggle('active', t.dataset.tab === name);
  });
  ['bs-basics', 'bs-leitmotiv', 'bs-style', 'bs-models', 'bs-view'].forEach(n => {
    const pane = document.getElementById(n + '-pane');
    if (pane) pane.style.display = (n === name) ? 'block' : 'none';
  });
  if (name === 'bs-models') loadBookSettingsModels();
  if (name === 'bs-view') bsLoadViewPrefs();
}

async function loadBookSettingsModels() {
  const tbody = document.getElementById('bs-models-table');
  if (!tbody || tbody.dataset.loaded === '1') return;
  try {
    const r = await fetch('/api/settings/models');
    const data = await r.json();
    // Phase 54.6.203 — fourth column names the per-role fallback so
    // "(unset)" isn't misleading. VISUALS_CAPTION_MODEL and
    // MINERU_VLM_MODEL don't fall back to LLM_MODEL; they have their
    // own code-level defaults (qwen2.5vl:32b + the MinerU pipeline
    // VLM). BOOK_REVIEW / AUTOWRITE_SCORER / BOOK_WRITE DO fall back
    // to LLM_MODEL. Phase 54.6.244 — added BOOK_WRITE_MODEL row and
    // moved "book write · autowrite writer" out of the LLM_MODEL
    // description (where it was before the 54.6.243 split).
    const rows = [
      ['LLM_MODEL',              data.llm_model,              'ask · wiki compile · extract-kg · everything without a per-role override', null],
      ['LLM_FAST_MODEL',         data.llm_fast_model,         'classify-papers · paraphrase-equations · RAPTOR · metadata fallback', 'LLM_MODEL'],
      ['BOOK_OUTLINE_MODEL',     data.book_outline_model,     'book outline (3-candidate tournament + density-driven section growth)', 'LLM_MODEL'],
      ['BOOK_WRITE_MODEL',       data.book_write_model,       'book write · autowrite writer/scorer/verify/cove (54.6.243 split)', 'LLM_MODEL'],
      ['BOOK_REVIEW_MODEL',      data.book_review_model,      'book review (5-dim critic)', 'LLM_MODEL'],
      ['AUTOWRITE_SCORER_MODEL', data.autowrite_scorer_model, 'autowrite score + rescore (not verify/cove)', 'LLM_MODEL'],
      ['VISUALS_CAPTION_MODEL',  data.visuals_caption_model,  'db caption-visuals (figures + charts) — 54.6.74 VLM-sweep winner', 'qwen2.5vl:32b (code default)'],
      ['MINERU_VLM_MODEL',       data.mineru_vlm_model,       'PDF parse only when PDF_CONVERTER_BACKEND=mineru-vlm-pro', 'opendatalab/MinerU2.5-Pro-2604-1.2B (code default)'],
      ['EMBEDDING_MODEL',        data.embedding_model,        'chunk embedding (dense + sparse)', null],
      ['RERANKER_MODEL',         data.reranker_model,         'hybrid search rerank step', null],
    ];
    tbody.innerHTML = rows.map(r => {
      const fallback = r[3]
        ? '<em class="u-muted">(unset — falls back to ' + _escHtml(r[3]) + ')</em>'
        : '<em class="u-muted">(required)</em>';
      return '<tr class="u-border-b">'
        + '<td class="u-pill-lg u-mono u-tiny">' + _escHtml(r[0]) + '</td>'
        + '<td class="u-pill-lg u-mono u-tiny">' + (r[1] ? _escHtml(r[1]) : fallback) + '</td>'
        + '<td class="u-pill-lg u-muted u-tiny">' + _escHtml(r[2]) + '</td>'
        + '</tr>';
    }).join('');
    tbody.dataset.loaded = '1';
  } catch (e) {
    tbody.innerHTML = '<tr><td class="u-p-2 u-danger" colspan="3">Failed: ' + e.message + '</td></tr>';
  }
}

async function loadBookSettings() {
  try {
    const res = await fetch('/api/book');
    const data = await res.json();
    document.getElementById('bs-title').value = data.title || '';
    document.getElementById('bs-description').value = data.description || '';
    document.getElementById('bs-target-chapter-words').value = (data.target_chapter_words != null) ? String(data.target_chapter_words) : '';
    document.getElementById('bs-plan').value = data.plan || '';
    // Phase 54.6.148 — restore the current book_type + refresh info panel
    const typeSel = document.getElementById('bs-book-type');
    if (typeSel && data.book_type) {
      typeSel.value = data.book_type;
      bsUpdateTypeInfo();
    }
    // Phase 54.6.152 — cache the book_type so the chapter-modal plan
    // textarea's live concept-density readout knows the correct wpc
    // range to use (per project type).
    if (data.book_type) window._currentBookType = data.book_type;
    // Basics meta — chapter / draft / gaps counts come through the
    // same endpoint, so surface them as a read-only summary.
    const meta = document.getElementById('bs-basics-meta');
    const defaultTcw = data.default_target_chapter_words || 6000;
    const effectiveTcw = data.target_chapter_words || defaultTcw;
    meta.innerHTML = '<strong>' + (data.chapters || 0) + '</strong> chapter' + ((data.chapters || 0) === 1 ? '' : 's')
                   + ' · <strong>' + (data.drafts || 0) + '</strong> draft' + ((data.drafts || 0) === 1 ? '' : 's')
                   + ' · Status: <strong>' + (data.status || 'draft') + '</strong>'
                   + ' · Effective target: <strong>~' + effectiveTcw + '</strong> words/chapter'
                   + (data.target_chapter_words ? '' : ' <em>(default)</em>');
    renderStyleFingerprint(data.style_fingerprint);
  } catch (exc) {
    document.getElementById('bs-basics-status').textContent = 'Load failed: ' + exc;
  }
}

function renderStyleFingerprint(fp) {
  const el = document.getElementById('bs-style-fingerprint');
  if (!fp || !fp.n_drafts_sampled) {
    el.innerHTML = '<div class="u-muted u-md">No fingerprint yet &mdash; mark some drafts as <em>final</em> / <em>reviewed</em> / <em>revised</em> and click <strong>Recompute</strong>.</div>';
    return;
  }
  const rows = [
    ['Drafts sampled', fp.n_drafts_sampled],
    ['Average words per draft', fp.avg_words_per_draft],
    ['Median sentence length', (fp.median_sentence_length || 0) + ' words'],
    ['Median paragraph length', (fp.median_paragraph_words || 0) + ' words'],
    ['Citations per 100 words', (fp.citations_per_100_words != null) ? fp.citations_per_100_words.toFixed(2) : '—'],
    ['Hedging rate', (fp.hedging_rate != null) ? (fp.hedging_rate * 100).toFixed(1) + '%' : '—'],
  ];
  let html = '<div style="display:grid;grid-template-columns:max-content 1fr;gap:6px 16px;font-size:13px;">';
  rows.forEach(([k, v]) => {
    html += '<div class="u-muted">' + k + '</div>';
    html += '<div><strong>' + (v != null ? v : '—') + '</strong></div>';
  });
  html += '</div>';
  const trans = fp.top_transitions || [];
  if (trans.length) {
    html += '<div class="u-mt-10 u-small u-muted">Top sentence-initial transitions: ';
    html += trans.slice(0, 8).map(t => '<span style="display:inline-block;padding:2px 8px;margin:2px;border:1px solid var(--border);border-radius:10px;background:var(--bg);font-family:ui-monospace,monospace;font-size:11px;">' + (t[0] || t).replace(/</g, '&lt;') + '</span>').join('');
    html += '</div>';
  }
  if (fp.computed_at) {
    html += '<div class="u-mt-10 u-tiny u-muted">Computed at: ' + fp.computed_at + '</div>';
  }
  el.innerHTML = html;
}

async function saveBookSettings(tab) {
  const statusId = 'bs-' + tab + '-status';
  const status = document.getElementById(statusId);
  status.textContent = 'Saving…';
  const fd = new FormData();
  if (tab === 'basics') {
    fd.append('title', document.getElementById('bs-title').value);
    fd.append('description', document.getElementById('bs-description').value);
    const tcw = document.getElementById('bs-target-chapter-words').value;
    // Blank leaves unchanged; 0 clears back to default; positive sets.
    if (tcw !== '') fd.append('target_chapter_words', tcw);
    // Phase 54.6.148 — send book_type too so type changes round-trip.
    const typeSel = document.getElementById('bs-book-type');
    if (typeSel && typeSel.value) fd.append('book_type', typeSel.value);
  } else if (tab === 'leitmotiv') {
    fd.append('plan', document.getElementById('bs-plan').value);
  }
  try {
    const res = await fetch('/api/book', {method: 'PUT', body: fd});
    const data = await res.json();
    if (!res.ok || !data.ok) {
      status.textContent = 'Save failed: ' + (data.detail || 'unknown');
      return;
    }
    status.innerHTML = '<span class="u-success">Saved.</span>';
    // Rehydrate meta line — chapter counts may shift if the target changed
    if (tab === 'basics') await loadBookSettings();
  } catch (exc) {
    status.textContent = 'Save failed: ' + exc;
  }
}

async function refreshStyleFingerprint() {
  const status = document.getElementById('bs-style-status');
  status.textContent = 'Computing from approved drafts…';
  try {
    const res = await fetch('/api/book/style-fingerprint/refresh', {method: 'POST'});
    const data = await res.json();
    if (!res.ok || !data.ok) {
      status.textContent = 'Refresh failed: ' + (data.error || data.detail || 'unknown');
      return;
    }
    renderStyleFingerprint(data.fingerprint);
    const sampled = (data.fingerprint && data.fingerprint.n_drafts_sampled) || 0;
    if (sampled === 0) {
      status.textContent = 'No approved drafts yet — mark some as final/reviewed/revised first.';
    } else {
      status.innerHTML = '<span class="u-success">Updated from ' + sampled + ' draft' + (sampled === 1 ? '' : 's') + '.</span>';
    }
  } catch (exc) {
    status.textContent = 'Refresh failed: ' + exc;
  }
}

// ── Phase 38: scoped snapshot bundles (chapter + book) ───────────────
// Safety net for autowrite-all. Chapter bundle stores every section's
// current content; restore creates NEW draft versions per section
// (non-destructive) so the old versions stay around as an undo path.
function openBundleSnapshots() {
  openModal('bundle-modal');
  switchBundleTab('sb-chapter');
}

function switchBundleTab(name) {
  document.querySelectorAll('#bundle-modal .tab').forEach(t => {
    t.classList.toggle('active', t.dataset.tab === name);
  });
  document.getElementById('sb-chapter-pane').style.display = (name === 'sb-chapter') ? 'block' : 'none';
  document.getElementById('sb-book-pane').style.display = (name === 'sb-book') ? 'block' : 'none';
  if (name === 'sb-chapter') loadBundleList('chapter');
  else loadBundleList('book');
}

async function doBundleSnapshot(scope) {
  const nameEl = document.getElementById('sb-' + scope + '-name');
  const status = document.getElementById('sb-' + scope + '-status');
  let url;
  if (scope === 'chapter') {
    if (!currentChapterId) {
      status.textContent = 'No current chapter — select a section first.';
      return;
    }
    url = '/api/snapshot/chapter/' + currentChapterId;
  } else {
    url = '/api/snapshot/book/' + window.SCIKNOW_BOOTSTRAP.bookId;
  }
  status.textContent = 'Saving…';
  const fd = new FormData();
  fd.append('name', nameEl.value || '');
  try {
    const res = await fetch(url, {method: 'POST', body: fd});
    const data = await res.json();
    if (!res.ok || !data.ok) {
      status.textContent = 'Error: ' + (data.detail || data.error || 'failed');
      return;
    }
    const extra = scope === 'chapter'
      ? ` (${data.drafts_included} section${data.drafts_included === 1 ? '' : 's'}, ${data.total_words} words)`
      : ` (${data.chapters_included} chapter${data.chapters_included === 1 ? '' : 's'}, ${data.total_words} words)`;
    status.innerHTML = '<span class="u-success">Saved &quot;' + (data.name || '').replace(/</g, '&lt;') + '&quot;</span>' + extra;
    nameEl.value = '';
    loadBundleList(scope);
  } catch (exc) {
    status.textContent = 'Error: ' + exc;
  }
}

async function loadBundleList(scope) {
  const list = document.getElementById('sb-' + scope + '-list');
  const target = (scope === 'chapter') ? currentChapterId : window.SCIKNOW_BOOTSTRAP.bookId;
  if (!target) {
    list.innerHTML = '<div class="u-hint-sm">Open any section first so a chapter is active.</div>';
    return;
  }
  list.innerHTML = '<div class="u-hint-sm">Loading…</div>';
  try {
    const res = await fetch('/api/snapshots/' + scope + '/' + target);
    const data = await res.json();
    const snaps = data.snapshots || [];
    if (snaps.length === 0) {
      list.innerHTML = '<div class="u-hint-sm">No ' + scope + ' snapshots yet.</div>';
      return;
    }
    let html = '<table class="u-w-full u-bcollapse u-md">';
    html += '<thead><tr class="u-muted u-border-b">';
    html += '<th class="u-pad-sm">Label</th><th class="u-pad-sm">Words</th><th class="u-pad-sm">Saved</th><th></th></tr></thead><tbody>';
    snaps.forEach(s => {
      const created = (s.created_at || '').split('.')[0].replace('T', ' ');
      html += '<tr class="u-border-b">';
      html += '<td class="u-pad-sm">' + (s.name || '').replace(/</g, '&lt;') + '</td>';
      html += '<td class="u-pad-sm u-muted">' + (s.word_count || 0).toLocaleString() + '</td>';
      html += '<td class="u-pad-sm u-muted u-tiny">' + created + '</td>';
      html += '<td class="u-pad-sm u-text-right">';
      html += '<button data-action="restore-bundle" data-snapshot-id="' + s.id + '" data-scope="' + scope + '" style="font-size:12px;padding:3px 10px;" title="Restore every draft in this bundle as NEW draft versions. Existing drafts are kept as an undo path.">Restore</button>';
      html += '</td></tr>';
    });
    html += '</tbody></table>';
    list.innerHTML = html;
  } catch (exc) {
    list.innerHTML = '<div class="u-danger u-small">Error: ' + exc + '</div>';
  }
}

async function restoreBundle(snapId, scope) {
  const confirmMsg = scope === 'chapter'
    ? 'Restore this chapter snapshot? Each section will get a NEW draft version. Existing drafts stay untouched.'
    : 'Restore this BOOK snapshot? Every chapter will get new draft versions for every section. Existing drafts stay untouched.';
  if (!confirm(confirmMsg)) return;
  const status = document.getElementById('sb-' + scope + '-status');
  status.textContent = 'Restoring…';
  try {
    const res = await fetch('/api/snapshot/restore-bundle/' + snapId, {method: 'POST'});
    const data = await res.json();
    if (!res.ok || !data.ok) {
      status.textContent = 'Error: ' + (data.detail || data.error || 'failed');
      return;
    }
    status.innerHTML = '<span class="u-success">Restored ' + data.drafts_created + ' draft' + (data.drafts_created === 1 ? '' : 's') + ' across ' + data.chapters_restored + ' chapter' + (data.chapters_restored === 1 ? '' : 's') + '. Reload to see them.</span>';
  } catch (exc) {
    status.textContent = 'Error: ' + exc;
  }
}

// ── Snapshots ─────────────────────────────────────────────────────────
async function takeSnapshot() {
  if (!currentDraftId) { showEmptyHint("No draft selected &mdash; click a section in the sidebar, or click <strong>Start writing</strong> under any chapter to create a first draft."); return; }
  const name = prompt('Snapshot name (leave empty for timestamp):');
  if (name === null) return;

  const fd = new FormData();
  fd.append('name', name);
  const res = await fetch('/api/snapshot/' + currentDraftId, {method: 'POST', body: fd});
  const data = await res.json();
  if (data.ok) {
    alert('Snapshot saved: ' + data.name);
  }
}

async function showSnapshots() {
  if (!currentDraftId) return;
  const res = await fetch('/api/snapshots/' + currentDraftId);
  const data = await res.json();
  if (!data.snapshots || data.snapshots.length === 0) return;

  // Show in the version panel
  const panel = document.getElementById('version-panel');
  const timeline = document.getElementById('version-timeline');
  const diffView = document.getElementById('diff-view');
  panel.style.display = 'block';

  let html = '<div class="snap-list">';
  html += '<div class="u-semibold u-mb-6">Snapshots</div>';
  data.snapshots.forEach(s => {
    html += '<div class="snap-item">';
    html += '<span>' + s.name + ' (' + s.word_count + 'w)</span>';
    html += '<div>';
    html += '<button data-action="diff-snapshot" data-snapshot-id="' + s.id + '" title="Show a line-diff between this snapshot and the current draft.">Diff</button> ';
    html += '<button data-action="restore-snapshot" data-snapshot-id="' + s.id + '" title="Restore this snapshot as a new draft version. Current draft is preserved.">Restore</button>';
    html += '</div></div>';
  });
  html += '</div>';
  timeline.innerHTML = html;
  diffView.innerHTML = '<p class="u-dim">Click "Diff" to compare a snapshot with current content.</p>';
}

async function diffSnapshot(snapId) {
  // Get snapshot content and current content, do client-side word diff
  const snapRes = await fetch('/api/snapshot-content/' + snapId);
  const snapData = await snapRes.json();
  const secRes = await fetch('/api/section/' + currentDraftId);
  const secData = await secRes.json();

  // Simple word diff
  const oldWords = snapData.content.split(/\s+/);
  const newWords = secData.content_raw.split(/\s+/);

  // Use a basic LCS-based diff
  let html = '';
  let i = 0, j = 0;
  // Simplified: just show both for now, use the server diff endpoint
  const diffRes = await fetch('/api/diff/' + 'snapshot' + '/' + currentDraftId);
  // Fallback: show snapshot content with note
  html = '<div class="u-mb-2 u-bold">Snapshot content:</div>';
  html += '<div class="u-dim-7 u-pre">' + snapData.content.substring(0, 5000).replace(/</g, '&lt;') + '</div>';
  document.getElementById('diff-view').innerHTML = html;
}

async function restoreSnapshot(snapId) {
  if (!confirm('Restore this snapshot? Current content will be overwritten.')) return;
  const snapRes = await fetch('/api/snapshot-content/' + snapId);
  const snapData = await snapRes.json();

  const fd = new FormData();
  fd.append('content', snapData.content);
  await fetch('/edit/' + currentDraftId, {method: 'POST', body: fd});
  loadSection(currentDraftId);
}

// ── Status selector ───────────────────────────────────────────────────
async function updateStatus(status) {
  if (!currentDraftId) return;
  const fd = new FormData();
  fd.append('status', status);
  await fetch('/api/draft/' + currentDraftId + '/status', {method: 'PUT', body: fd});
}


// ── Phase 46.F — Setup Wizard (end-to-end from empty to book) ─────────
//
// Walks a new user through: project choice → corpus ingest → index
// builds → expand → book creation. Each step reads live state via
// /api/setup/status so the trail shows real progress, not just UI
// position. Subprocess-backed steps (ingest, cluster, raptor, wiki)
// stream to a per-step <pre> log.

let _swCurrentStep = 'project';
let _swCurrentJob  = null;

function openSetupWizard() {
  openModal('setup-wizard-modal');
  swGoto('project');
  swRefreshStatus();
  swLoadProjectsForWizard();
  swLoadBookTypes();  // Phase 54.6.147 — populate book-type dropdown + info panel
}

function swGoto(step) {
  _swCurrentStep = step;
  document.querySelectorAll('#setup-wizard-modal .sw-step-pane').forEach(p => {
    p.style.display = 'none';
  });
  const pane = document.getElementById('sw-step-' + step);
  if (pane) pane.style.display = 'block';
  document.querySelectorAll('#sw-trail .sw-step').forEach(s => {
    s.classList.toggle('active', s.dataset.swStep === step);
  });
  // Auto-refresh status on entering steps 2 + 3 so counts are fresh
  if (step === 'corpus' || step === 'indices') swRefreshStatus();
  if (step === 'book') swUpdateTypeInfo();   // Phase 54.6.147
}

// Phase 54.6.147 — load project types (with concept-density ranges)
// from /api/book-types and populate the wizard dropdown + live info
// panel. Cached in window._swBookTypes so the `onchange` handler
// doesn't re-fetch on every dropdown change.
window._swBookTypes = null;
async function swLoadBookTypes() {
  try {
    const res = await fetch('/api/book-types');
    const data = await res.json();
    window._swBookTypes = data.types || [];
    const sel = document.getElementById('sw-book-type');
    if (!sel) return;
    // Preserve current value if user already picked something
    const prior = sel.value;
    sel.innerHTML = '';
    for (const t of window._swBookTypes) {
      const opt = document.createElement('option');
      opt.value = t.slug;
      opt.textContent = `${t.display_name} (${t.default_target_chapter_words.toLocaleString()} words/chap)`;
      if (t.slug === prior) opt.selected = true;
      sel.appendChild(opt);
    }
    if (!prior) sel.value = 'scientific_book';
    swUpdateTypeInfo();
  } catch (e) {
    console.warn('swLoadBookTypes failed:', e);
  }
}

function swUpdateTypeInfo() {
  const panel = document.getElementById('sw-book-type-info');
  const sel = document.getElementById('sw-book-type');
  if (!panel || !sel || !window._swBookTypes) return;
  const slug = sel.value;
  const t = window._swBookTypes.find(x => x.slug === slug);
  if (!t) { panel.innerHTML = ''; return; }
  const [clo, chi]   = t.concepts_per_section_range;
  const [wlo, whi]   = t.words_per_concept_range;
  const [slo, shi]   = t.section_at_midpoint_range;
  const chap         = t.default_target_chapter_words;
  const nchap        = t.default_chapter_count;
  const totalLo      = (chap * nchap * 0.7).toLocaleString();   // -30% lower envelope
  const totalHi      = (chap * nchap * 1.3).toLocaleString();   // +30% upper envelope
  const targetInput  = document.getElementById('sw-book-target');
  if (targetInput && !targetInput.value) {
    targetInput.placeholder = `(type default: ${chap.toLocaleString()})`;
  }
  panel.innerHTML = `
    <div class="u-grid-kv">
      <span class="u-muted">Description:</span>
      <span>${_escHtml(t.description)}</span>
      <span class="u-muted">Default chapters:</span>
      <span>${nchap} &middot; ${chap.toLocaleString()} words each
            ${t.is_flat ? '<em>(flat IMRaD — one chapter)</em>' : ''}</span>
      <span class="u-muted">Concepts / section:</span>
      <span>${clo}–${chi} novel chunks (Cowan 2001)</span>
      <span class="u-muted">Words / concept:</span>
      <span>${wlo}–${whi} (midpoint ${Math.floor((wlo + whi) / 2)})</span>
      <span class="u-muted">Section at midpoint:</span>
      <span>${slo.toLocaleString()}–${shi.toLocaleString()} words</span>
      <span class="u-muted">Typical book total:</span>
      <span>~${totalLo}–${totalHi} words</span>
    </div>
    <div class="u-mt-6 u-muted u-tiny">
      Sections with a bullet plan auto-size bottom-up (concept count × ${Math.floor((wlo + whi) / 2)} wpc).
      Sections without a plan fall back to the chapter-level target.
    </div>
  `;
}

async function swRefreshStatus() {
  try {
    const res = await fetch('/api/setup/status');
    const d = await res.json();
    const proj = d.project || {slug: 'unknown'};
    // Corpus step
    const cstat = document.getElementById('sw-corpus-status');
    if (cstat) {
      cstat.innerHTML =
        '<strong>Active project:</strong> <code>' + proj.slug + '</code>'
        + (proj.is_default ? ' <em>(legacy default)</em>' : '')
        + '<br><strong>Documents:</strong> ' + (d.n_documents || 0).toLocaleString()
        + ' &middot; <strong>Complete:</strong> ' + (d.n_complete || 0).toLocaleString()
        + ' &middot; <strong>Chunks:</strong> ' + (d.n_chunks || 0).toLocaleString();
    }
    // Indices step
    const istat = document.getElementById('sw-indices-status');
    if (istat) {
      const rapt = d.raptor_levels || {};
      const raptStr = Object.keys(rapt).length
        ? Object.keys(rapt).sort().map(k => k + '=' + rapt[k]).join(', ')
        : '(not built)';
      istat.innerHTML =
        '<strong>Topic clusters:</strong> ' + (d.n_with_topic || 0)
        + ' papers tagged &middot; <strong>RAPTOR:</strong> ' + raptStr
        + ' &middot; <strong>Wiki pages:</strong> ' + (d.n_wiki_pages || 0);
    }
  } catch (_) {}
}

async function swLoadProjectsForWizard() {
  const list = document.getElementById('sw-project-list');
  list.innerHTML = 'Loading…';
  try {
    const res = await fetch('/api/projects');
    const d = await res.json();
    const active = d.active_slug;
    const running = d.running_slug;
    if (!d.projects || !d.projects.length) {
      list.innerHTML = '<div class="u-p-10 u-muted">No projects yet. Create one on the right.</div>';
      return;
    }
    list.innerHTML = d.projects.map(p => {
      const mark = p.slug === active ? '●' : '○';
      const running_mark = p.slug === running
        ? ' <span class="u-accent u-xxs">(running here)</span>'
        : '';
      const useBtn = p.slug === active ? '' :
        `<button onclick="swUseProject('${p.slug}')" title="Set this project as active (.active-project file). Requires restarting \`sciknow book serve\` to take effect.">Use</button>`;
      return `<div class="u-p-6-10 u-border-b u-flex-raw u-ai-center u-gap-2">
        <span class="u-accent">${mark}</span>
        <strong class="u-flex-1">${p.slug}</strong>${running_mark}
        ${useBtn}</div>`;
    }).join('');
  } catch (exc) {
    list.innerHTML = '<div class="u-p-10 u-danger">Failed: ' + exc + '</div>';
  }
}

async function swUseProject(slug) {
  try {
    const res = await fetch('/api/projects/use', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({slug: slug}),
    });
    const d = await res.json();
    document.getElementById('sw-project-status').textContent =
      d.message || ('Active project: ' + slug);
    swLoadProjectsForWizard();
    swRefreshStatus();
  } catch (exc) {
    document.getElementById('sw-project-status').textContent = 'Failed: ' + exc;
  }
}

async function swCreateProject() {
  const slug = (document.getElementById('sw-new-slug').value || '').trim();
  if (!slug) return;
  if (!/^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$/.test(slug)) {
    document.getElementById('sw-project-status').textContent =
      'Slug must be lowercase alphanumerics + hyphens.';
    return;
  }
  document.getElementById('sw-project-status').textContent =
    'Creating ' + slug + '…';
  try {
    const res = await fetch('/api/projects/init', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({slug: slug}),
    });
    const d = await res.json();
    if (!res.ok) {
      document.getElementById('sw-project-status').textContent =
        'Failed: ' + (d.detail || res.status);
      return;
    }
    document.getElementById('sw-project-status').textContent =
      '✓ Created ' + slug + '. Use it below to activate.';
    swLoadProjectsForWizard();
  } catch (exc) {
    document.getElementById('sw-project-status').textContent = 'Failed: ' + exc;
  }
}

// ── Corpus step ─────────────────────────────────────────────────────
async function swUploadPDFs() {
  const input = document.getElementById('sw-upload-files');
  if (!input.files || !input.files.length) return;
  const fd = new FormData();
  for (const f of input.files) fd.append('files', f);
  fd.append('start_ingest',
    document.getElementById('sw-upload-start-ingest').checked ? 'true' : 'false');
  const log = document.getElementById('sw-ingest-log');
  log.textContent = 'Uploading ' + input.files.length + ' file(s)…\n';
  try {
    const res = await fetch('/api/corpus/upload', {method: 'POST', body: fd});
    const d = await res.json();
    if (!res.ok) {
      log.textContent += 'Upload failed: ' + (d.detail || res.status) + '\n';
      return;
    }
    log.textContent += '✓ Staged ' + d.n_files + ' file(s) to ' + d.staging_dir + '\n';
    if (d.job_id) swAttachLogStream(d.job_id, 'sw-ingest-log');
  } catch (exc) {
    log.textContent += 'Upload failed: ' + exc + '\n';
  }
}

async function swIngestDirectory() {
  const path = (document.getElementById('sw-ingest-path').value || '').trim();
  if (!path) return;
  const fd = new FormData();
  fd.append('path', path);
  fd.append('recursive',
    document.getElementById('sw-ingest-recursive').checked ? 'true' : 'false');
  fd.append('force',
    document.getElementById('sw-ingest-force').checked ? 'true' : 'false');
  const log = document.getElementById('sw-ingest-log');
  log.textContent = 'Starting ingest of ' + path + '…\n';
  try {
    const res = await fetch('/api/corpus/ingest-directory',
      {method: 'POST', body: fd});
    const d = await res.json();
    if (!res.ok) {
      log.textContent += 'Failed: ' + (d.detail || res.status) + '\n';
      return;
    }
    swAttachLogStream(d.job_id, 'sw-ingest-log');
  } catch (exc) {
    log.textContent += 'Failed: ' + exc + '\n';
  }
}

// ── Indices step ────────────────────────────────────────────────────
async function swRunIndex(kind) {
  const fd = new FormData();
  let url = '';
  if (kind === 'cluster') {
    url = '/api/catalog/cluster';
    fd.append('rebuild',
      document.getElementById('sw-cluster-rebuild').checked ? 'true' : 'false');
  } else if (kind === 'raptor') {
    url = '/api/catalog/raptor/build';
  } else if (kind === 'wiki') {
    url = '/api/wiki/compile';
    fd.append('rebuild',
      document.getElementById('sw-wiki-rebuild').checked ? 'true' : 'false');
    fd.append('rewrite_stale',
      document.getElementById('sw-wiki-stale').checked ? 'true' : 'false');
  }
  const log = document.getElementById('sw-indices-log');
  log.textContent = 'Starting ' + kind + '…\n';
  try {
    const res = await fetch(url, {method: 'POST', body: fd});
    const d = await res.json();
    if (!res.ok) {
      log.textContent += 'Failed: ' + (d.detail || res.status) + '\n';
      return;
    }
    swAttachLogStream(d.job_id, 'sw-indices-log');
  } catch (exc) {
    log.textContent += 'Failed: ' + exc + '\n';
  }
}

// ── Book step ───────────────────────────────────────────────────────
async function swCreateBook() {
  const title = (document.getElementById('sw-book-title').value || '').trim();
  if (!title) {
    document.getElementById('sw-book-status').textContent = 'Title is required.';
    return;
  }
  const type = document.getElementById('sw-book-type').value;
  const desc = document.getElementById('sw-book-desc').value.trim();
  const target = document.getElementById('sw-book-target').value;
  const payload = {
    title: title, type: type, description: desc, bootstrap: true,
  };
  if (target) payload.target_chapter_words = parseInt(target, 10);
  const stat = document.getElementById('sw-book-status');
  stat.textContent = 'Creating…';
  try {
    const res = await fetch('/api/book/create', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    const d = await res.json();
    if (!res.ok) {
      stat.innerHTML = '<span class="u-danger">Failed: '
        + (d.detail || res.status) + '</span>';
      return;
    }
    const flatNote = d.is_flat
      ? ' &middot; Auto-created chapter 1 with sections: '
        + (d.default_sections || []).join(', ')
      : '';
    stat.innerHTML = '<span class="u-success">✓ Created '
      + d.display_name + ' "' + d.title + '"</span>'
      + '<br><code>' + d.book_id.slice(0, 8) + '</code>' + flatNote
      + '<br>Next: restart <code>sciknow book serve "' + d.title
      + '"</code> to open this book in the reader.';
  } catch (exc) {
    stat.innerHTML = '<span class="u-danger">Failed: ' + exc + '</span>';
  }
}

// Shared SSE log attacher — renders `log` events line-by-line, and
// updates status counts when the job ends.
function swAttachLogStream(jobId, logElId) {
  _swCurrentJob = jobId;
  const logEl = document.getElementById(logElId);
  const source = new EventSource('/api/stream/' + jobId);
  source.onmessage = function(e) {
    const evt = JSON.parse(e.data);
    if (evt.type === 'log') {
      logEl.textContent += evt.text + '\n';
      logEl.scrollTop = logEl.scrollHeight;
    } else if (evt.type === 'progress') {
      if (evt.detail && evt.detail.startsWith('$ ')) {
        logEl.textContent += evt.detail + '\n';
      }
    } else if (evt.type === 'error') {
      logEl.textContent += 'ERROR: ' + evt.message + '\n';
      source.close(); _swCurrentJob = null;
      swRefreshStatus();
    } else if (evt.type === 'completed' || evt.type === 'done') {
      logEl.textContent += '— done —\n';
      source.close(); _swCurrentJob = null;
      swRefreshStatus();
    }
  };
}


// ── Phase 43h — Project management modal ──────────────────────────────
// Mirrors `sciknow project` from the CLI. Switching the active project
// only writes .active-project; the running web reader keeps serving its
// original book until the user restarts `sciknow book serve`.

// ── Phase 21.c — Visuals browser ─────────────────────────────────────

function openVisualsModal() {
  document.querySelectorAll('.nav-dropdown.open').forEach(d => d.classList.remove('open'));
  openModal('visuals-modal');
  // 54.6.100 — invalidate the stats cache on open so a reopened modal
  // reflects any newly-ingested papers.
  _visStatsCache = null;
  loadVisuals();
  // Load stats into the footer-level summary (separate from the per-
  // query counter that loadVisuals writes into #vis-stats).
  fetch('/api/visuals/stats').then(r => r.json()).then(d => {
    _visStatsCache = d;
    const s = d.stats || {};
    const parts = Object.entries(s).map(([k,v]) => k + ': ' + v).join(', ');
    const el = document.getElementById('vis-stats');
    if (el && !el.textContent) el.textContent = parts ? ('Total: ' + d.total + ' (' + parts + ')') : 'No visuals extracted yet. Run: sciknow corpus extract-visuals';
  }).catch(() => {});
}

// Phase 54.6.99 — redesigned Visual Elements browser. Two modes:
//   * gallery — image-kind-only (figure+chart), CSS grid of thumbnails
//                with captions overlay, click-to-enlarge
//   * list    — every kind, list layout, figures/charts get real <img>
//                thumbnails instead of italic placeholder text
// Both modes render tables as HTML (MinerU emits HTML table_body),
// equations via KaTeX ($$…$$), code in <pre>.
//
// Phase 54.6.100 — pagination so users can see all N visuals not just
// the first 40. We track totals from /api/visuals/stats, page size
// 60, and a "Load more" button at the bottom appends the next page
// without redrawing. Fresh searches / mode toggles reset pagination.
let _visPage = {items: [], offset: 0, pageSize: 60, totals: null, exhausted: false};
let _visStatsCache = null;

async function _visFetchPage(kinds, query, limit, offset, orderBy) {
  // Interleave figures + charts by offsetting each proportionally to
  // their totals so we don't return 40 figures first then 40 charts.
  const totals = _visStatsCache && _visStatsCache.stats ? _visStatsCache.stats : {};
  const sumTotal = kinds.reduce((s, k) => s + (totals[k] || 0), 0) || 1;
  const perKind = kinds.map(k => ({
    kind: k,
    n: Math.max(1, Math.round(limit * ((totals[k] || 0) / sumTotal))),
    off: Math.round(offset * ((totals[k] || 0) / sumTotal)),
  }));
  const fetches = perKind.map(({kind, n, off}) => {
    const p = new URLSearchParams();
    p.set('kind', kind);
    if (query) p.set('query', query);
    p.set('limit', String(n));
    p.set('offset', String(off));
    if (orderBy) p.set('order_by', orderBy);
    return fetch('/api/visuals?' + p.toString()).then(r => r.json())
      .then(arr => Array.isArray(arr) ? arr : []);
  });
  const arrays = await Promise.all(fetches);
  // Round-robin merge so the final order is kind-interleaved.
  const merged = [];
  const maxLen = Math.max.apply(null, arrays.map(a => a.length).concat([0]));
  for (let i = 0; i < maxLen; i++) {
    for (const arr of arrays) { if (i < arr.length) merged.push(arr[i]); }
  }
  return merged;
}

async function loadVisuals(append) {
  const mode = document.getElementById('vis-mode').value || 'gallery';
  const kindSel = document.getElementById('vis-kind-filter');
  let kind = kindSel.value;
  const orderBy = (document.getElementById('vis-order') || {value: 'importance'}).value || 'importance';
  const query = document.getElementById('vis-search').value.trim();
  const results = document.getElementById('vis-results');
  const statsEl = document.getElementById('vis-stats');

  // Make sure we know corpus totals before building pagination state.
  if (!_visStatsCache) {
    try {
      _visStatsCache = await fetch('/api/visuals/stats').then(r => r.json());
    } catch (_) { _visStatsCache = {stats: {}, total: 0}; }
  }

  // Reset pagination state on a fresh load; keep it on append.
  if (!append) {
    _visPage = {items: [], offset: 0, pageSize: 60, totals: null, exhausted: false};
    results.innerHTML = '<em>Loading...</em>';
  }

  // In gallery mode, we force kind to image types only. If the user
  // picked a non-image kind in the dropdown while in gallery mode,
  // transparently show them "All image kinds" rather than an empty
  // result. If they pick a specific image kind (figure or chart) the
  // filter still honours it.
  const imageKinds = ['figure', 'chart'];
  let activeKinds;
  if (mode === 'gallery') {
    activeKinds = (!kind || !imageKinds.includes(kind)) ? imageKinds : [kind];
  } else {
    activeKinds = kind ? [kind] : ['figure', 'chart', 'table', 'equation', 'code'];
  }

  // Compute the universe total for the active filter so the stats
  // line can say "shown N of M".
  const allStats = (_visStatsCache && _visStatsCache.stats) || {};
  const universe = activeKinds.reduce((s, k) => s + (allStats[k] || 0), 0);
  _visPage.totals = universe;

  try {
    let newItems = [];
    if (mode === 'gallery') {
      // Merge proportional-offset pages across all active image kinds.
      newItems = await _visFetchPage(activeKinds, query, _visPage.pageSize, _visPage.offset, orderBy);
    } else {
      // List mode: either one kind (use offset directly) or all kinds
      // (proportional like gallery).
      if (kind) {
        const p = new URLSearchParams();
        p.set('kind', kind);
        if (query) p.set('query', query);
        p.set('limit', String(_visPage.pageSize));
        p.set('offset', String(_visPage.offset));
        if (orderBy) p.set('order_by', orderBy);
        const res = await fetch('/api/visuals?' + p.toString());
        const arr = await res.json();
        newItems = Array.isArray(arr) ? arr : [];
      } else {
        newItems = await _visFetchPage(activeKinds, query, _visPage.pageSize, _visPage.offset, orderBy);
      }
    }

    _visPage.offset += _visPage.pageSize;
    if (!newItems.length || newItems.length < _visPage.pageSize) {
      _visPage.exhausted = true;
    }
    _visPage.items = _visPage.items.concat(newItems);
    const items = _visPage.items;

    if (!items.length) {
      results.innerHTML = '<em class="u-muted">No visuals found.</em>';
      if (statsEl) statsEl.textContent = universe ? '0 of ' + universe : '';
      return;
    }
    if (statsEl) {
      const parts = activeKinds.map(k => (allStats[k] || 0) + ' ' + k).join(' + ');
      statsEl.textContent = items.length + ' of ' + universe + ' (' + parts + ')';
    }

    // Renderer shared by both modes for one visual, returns HTML.
    const renderPreview = (v) => {
      if ((v.kind === 'figure' || v.kind === 'chart') && v.id) {
        const imgUrl = '/api/visuals/image/' + encodeURIComponent(v.id);
        return '<a class="u-block u-bg u-r-sm u-ov-hidden" href="' + imgUrl + '" target="_blank">'
          + '<img src="' + imgUrl + '" loading="lazy" '
          + 'style="width:100%;height:180px;object-fit:contain;display:block;" '
          + 'onerror="this.parentElement.innerHTML=\'<em style=padding:8px;color:var(--fg-muted);font-size:11px;>image unavailable</em>\'"></a>';
      }
      if (v.kind === 'equation') {
        // 54.6.105 — dropped auto-render. The previous approach built
        // HTML strings like `$$<body>$$` and hoped KaTeX auto-render
        // would find the delimiters at a text-node boundary. It
        // worked in isolated JSDOM tests but kept failing in the real
        // app across four iterations (54.6.101-103). Root cause:
        // auto-render walks the live DOM and matches delimiters
        // against text nodes; any adjacent HTML structure or escaped
        // entity that changed the text-node layout could drop the
        // match and leave raw `$$…$$` visible.
        //
        // Fix: emit a container with the LaTeX body stashed in a
        // data-latex attribute (exempt from HTML parsing since it's
        // an attribute value) and call katex.render() directly in a
        // post-pass. No delimiter matching needed — we know exactly
        // which elements carry equations and what their LaTeX source
        // is, so we drive the render programmatically.
        const raw = String(v.content || '').trim();
        let body = raw;
        body = body.replace(/^\s*\$\$\s*/, '').replace(/\s*\$\$\s*$/, '');
        body = body.replace(/^\s*\$\s*/, '').replace(/\s*\$\s*$/, '');
        body = body.replace(/^\s*\\\[\s*/, '').replace(/\s*\\\]\s*$/, '');
        body = body.replace(/^\s*\\\(\s*/, '').replace(/\s*\\\)\s*$/, '');
        body = body.trim();
        // 54.6.108 — THE BUG. The 600-char truncation introduced in
        // 54.6.101 broke every equation longer than 600 chars by
        // cutting mid-command (\begin + array blocks, multi-line
        // sums etc.). Server already caps content at 2000 chars in
        // the API. Render the full body — KaTeX display mode with
        // overflow-x:auto handles wide formulas via horizontal
        // scroll in the card. On this corpus: ~25% of equations on
        // the first page are >600 chars (arrays + systems); the
        // user's "half don't render" lines up with this range.
        const truncated = body;
        // Escape attribute value: &, <, >, ", ' so `data-latex="..."` is well-formed.
        const attrSafe = _escHtml(truncated);
        return '<div class="vis-eq" style="padding:14px 16px;background:#fff;color:#111;border-radius:6px;border:1px solid var(--border);margin:2px 0;">'
          + '<div class="eq-target" data-latex="' + attrSafe + '" '
          +      'style="font-size:17px;line-height:1.55;text-align:center;overflow-x:auto;min-height:30px;">'
          +      _escHtml(truncated) /* fallback text if KaTeX unavailable */
          + '</div>'
          + '<details class="u-mt-2">'
          +   '<summary style="font-size:10px;color:var(--fg-muted);cursor:pointer;user-select:none;">LaTeX source</summary>'
          +   '<pre style="font-family:var(--font-mono);font-size:11px;margin:6px 0 0;padding:6px 8px;background:var(--bg);border-radius:4px;white-space:pre-wrap;word-break:break-word;">'
          +   _escHtml(raw)
          +   '</pre>'
          + '</details>'
          + '</div>';
      }
      if (v.kind === 'table') {
        // 54.6.106 — if the structured-parse pass ran (db parse-tables),
        // surface the title + summary + column headers above the raw
        // HTML. Otherwise fall back to just the styled HTML table.
        let parsed = '';
        if (v.table_title || v.table_summary || (v.table_headers && v.table_headers.length)) {
          const title = v.table_title ? '<div class="u-semibold u-md u-fg-ink u-mb-1">' + _escHtml(v.table_title) + '</div>' : '';
          const shape = (v.table_n_rows || v.table_n_cols)
            ? '<span class="u-muted u-xxs">[' + (v.table_n_rows || '?') + ' rows × ' + (v.table_n_cols || '?') + ' cols]</span> '
            : '';
          const summary = v.table_summary
            ? '<div style="font-size:11px;color:#333;line-height:1.45;margin-bottom:6px;">' + shape + _escHtml(v.table_summary) + '</div>'
            : '';
          const hdrs = Array.isArray(v.table_headers) && v.table_headers.length
            ? '<div class="u-xxs u-muted u-mb-6"><strong>Columns:</strong> '
                + v.table_headers.map(h => _escHtml(String(h))).join(' · ') + '</div>'
            : '';
          parsed = '<div style="padding:8px;background:var(--bg-alt, var(--bg-elevated));border-radius:4px;border:1px solid var(--border);margin-bottom:8px;">'
            + title + summary + hdrs + '</div>';
        }
        return parsed
          + '<div class="vis-table-wrap" style="max-height:260px;overflow:auto;border:1px solid var(--border);border-radius:6px;padding:8px;background:#fff;color:#111;">'
          + (v.content || '<em>empty table</em>') + '</div>';
      }
      if (v.kind === 'code') {
        return '<pre style="max-height:120px;overflow:auto;font-size:11px;padding:6px;background:var(--bg);border-radius:4px;margin:0;">'
          + _escHtml((v.content || '').substring(0, 800)) + '</pre>';
      }
      return '<em class="u-hint">' + _escHtml((v.content || v.caption || '').substring(0, 200)) + '</em>';
    };

    const kindIcon = (k) => k === 'table' ? '\uD83D\uDCCA'
      : k === 'equation' ? '\u2211'
      : k === 'figure' ? '\uD83D\uDDBC'
      : k === 'chart' ? '\uD83D\uDCC8'
      : '\uD83D\uDCBB';

    let html = '';
    if (mode === 'gallery') {
      // CSS grid of cards. Image-kind cards stay narrow (~220px) so
      // more thumbnails fit per row; equation/table/code cards need
      // more width to be legible, so bump the minmax when the active
      // filter isn't image-kind.
      const onlyImages = activeKinds.every(k => imageKinds.includes(k));
      const minCol = onlyImages ? '220px' : '340px';
      html += '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(' + minCol + ',1fr));gap:12px;">';
      for (const v of items) {
        const cap = v.ai_caption || v.caption || '';
        const label = (v.figure_num ? v.figure_num : kindIcon(v.kind) + ' ' + (v.kind || ''));
        const paperInfo = (v.paper_title || '').substring(0, 48) + (v.year ? ' (' + v.year + ')' : '');
        html += '<div class="u-border u-r-md u-p-2 u-bg-alt-raw">'
          + renderPreview(v)
          + '<div class="u-flex-raw u-jc-between u-ai-center u-mt-6 u-gap-1">'
          +   '<strong class="u-tiny">' + _escHtml(label.substring(0, 40)) + '</strong>'
          +   '<button class="btn-secondary u-xxs u-p-1-6" '
          +     'onclick="insertVisualAtCursor(' + JSON.stringify(JSON.stringify(v)) + ')">Insert</button>'
          + '</div>'
          + '<div class="u-xxs u-muted u-mt-2px">' + _escHtml(paperInfo) + '</div>'
          + (cap ? '<div style="font-size:11px;line-height:1.35;margin-top:4px;max-height:3.4em;overflow:hidden;" title="' + _escHtml(cap) + '">' + _escHtml(cap.substring(0, 140)) + '</div>' : '')
          + '</div>';
      }
      html += '</div>';
    } else {
      // List mode — unchanged layout but with real image thumbnails
      // for figures/charts.
      for (const v of items) {
        const label = (v.figure_num || v.kind)
          + (v.caption ? ': ' + String(v.caption).substring(0, 80) : '');
        const paperInfo = (v.paper_title || '').substring(0, 60) + (v.year ? ' (' + v.year + ')' : '');
        html += '<div class="u-border u-r-md u-p-2 u-mb-2">'
          + '<div class="u-flex-raw u-jc-between u-ai-center u-mb-1 u-gap-2">'
          +   '<strong>' + kindIcon(v.kind) + ' ' + _escHtml(label.substring(0, 100)) + '</strong>'
          +   '<button class="btn-secondary" style="font-size:11px;padding:2px 8px;flex-shrink:0;" '
          +     'onclick="insertVisualAtCursor(' + JSON.stringify(JSON.stringify(v)) + ')">Insert</button>'
          + '</div>'
          + '<div class="u-note-mb-1">' + _escHtml(paperInfo) + '</div>'
          + renderPreview(v)
          + '</div>';
      }
    }
    // Append "Load more" footer when there might be more pages AND the
    // UI already shows at least one item. Fresh loads get a full
    // replacement; append calls splice into the existing grid/list.
    const hasMore = !_visPage.exhausted && items.length < universe;
    const loadMoreHtml =
      '<div id="vis-loadmore-wrap" style="display:flex;justify-content:center;padding:10px 0;">'
      + (hasMore
          ? '<button class="btn-secondary u-small" onclick="loadVisuals(true)" title="Fetch the next page of visuals matching the current filters.">Load more (' + items.length + ' / ' + universe + ')</button>'
          : '<span class="u-hint">End of list (' + items.length + ' / ' + universe + ')</span>')
      + '</div>';

    results.innerHTML = html + loadMoreHtml;

    // 54.6.109 — macros map for commands KaTeX doesn't natively ship
    // but that appear legitimately in our corpus. All either pure
    // display hints (safe to strip) or old LaTeX 2.09 aliases
    // (mapped to KaTeX equivalents). Recovers ~20 of the ~50 failing
    // equations in the 4,925-row corpus; remainder are MinerU
    // extraction errors (malformed \end, double-subscript, etc.)
    // that need a re-ingest, not a render fix.
    const _KATEX_MACROS = {
      '\\displaylimits': '',     // 14× — display-mode already stacks limits
      '\\mit':           '',     // 2×  — LaTeX 2.09 italic, drop
      '\\sc':            '',     // 1×  — small-caps, drop
      '\\mathbfcal':     '\\mathcal',      // 1×  — bold cal → cal
      '\\textless':      '<',    // 1×
      '\\textgreater':   '>',    // 0×  — pre-emptive
      '\\hdots':         '\\ldots',        // 1×
    };
    // 54.6.105/107 — drive KaTeX programmatically on every .eq-target
    // we just inserted. 54.6.107 adds diagnostics: count ok/failed,
    // mark failures with a red border + post-it note so the user can
    // see WHICH equations fail and what they contained. Console logs
    // the LaTeX that killed KaTeX so we have something to debug with.
    const renderAllEquations = () => {
      const targets = results.querySelectorAll('.eq-target[data-latex]');
      if (!targets.length) return true;
      if (typeof window.katex === 'undefined') return false;
      let ok = 0, fail = 0;
      targets.forEach(el => {
        if (el.dataset.rendered === '1') { ok++; return; }
        const tex = el.dataset.latex || '';
        try {
          // displayMode+throwOnError:true lets us catch KaTeX parse
          // errors explicitly so we can badge the card. With
          // throwOnError:false KaTeX would embed a red error token
          // silently — the user reports "half don't render" because
          // the inline red error LOOKS like raw LaTeX next to the
          // parts that rendered. Turning throwOnError back on and
          // handling the exception ourselves gives clean classification.
          window.katex.render(tex, el, {
            displayMode: true,
            throwOnError: true,
            output: 'html',
            strict: false,   // tolerate LaTeX-incompatible constructs
            macros: _KATEX_MACROS,
          });
          el.dataset.rendered = '1';
          ok++;
        } catch (e) {
          // Fall back to throwOnError:false so the user sees
          // partial rendering (KaTeX highlights the offending token
          // in red but renders what it can), and badge the container
          // so it's obvious which ones failed.
          try {
            window.katex.render(tex, el, {
              displayMode: true, throwOnError: false, output: 'html',
              strict: 'ignore', macros: _KATEX_MACROS,
            });
          } catch (_) {
            el.textContent = tex;
          }
          el.dataset.rendered = '0';
          el.style.border = '2px dashed #e57373';
          el.title = 'KaTeX: ' + (e.message || '').slice(0, 200);
          console.warn('[vis-eq] parse failed:', (e.message || '').split('\n')[0],
                       '\n  latex:', tex.slice(0, 180));
          fail++;
        }
      });
      // Post a small stats footer so the user can see the tally
      // without opening devtools.
      const statsEl = document.getElementById('vis-stats');
      if (statsEl && (ok + fail) > 0) {
        const prev = statsEl.textContent || '';
        if (!prev.includes('· eq')) {
          statsEl.textContent = prev + '  · eq ' + ok + '/' + (ok + fail) + ' rendered';
        } else {
          statsEl.textContent = prev.replace(/· eq \d+\/\d+ rendered/, '· eq ' + ok + '/' + (ok + fail) + ' rendered');
        }
      }
      return true;
    };
    if (!renderAllEquations()) {
      // KaTeX still loading; retry a few times at increasing delay.
      let attempts = 0;
      const tick = setInterval(() => {
        attempts++;
        if (renderAllEquations() || attempts > 10) clearInterval(tick);
      }, 200);
    }
  } catch (exc) {
    results.innerHTML = '<em class="u-danger">Failed: ' + exc + '</em>';
  }
}

function insertVisualAtCursor(vJson) {
  const v = JSON.parse(vJson);
  let md = '';
  // Phase 54.6.87 — figure + chart inserts now emit markdown image
  // syntax referencing the server's /api/visuals/image/<id> endpoint
  // so the editor preview + the saved draft both render a real
  // thumbnail. Equations keep $$...$$ which KaTeX auto-render picks up.
  if (v.kind === 'figure' || v.kind === 'chart') {
    const fig = v.figure_num || (v.kind === 'chart' ? 'Chart' : 'Figure');
    const cap = (v.ai_caption || v.caption || 'No caption').replace(/[\[\]()]/g, '');
    md = '\n\n![' + fig + ': ' + cap.substring(0, 180) + '](/api/visuals/image/'
         + encodeURIComponent(v.id) + ')\n\n';
  } else if (v.kind === 'equation') {
    md = '\n\n$$' + (v.content || '') + '$$\n\n';
  } else if (v.kind === 'table') {
    // Tables stay inline; MinerU HTML renders in the read-view via
    // innerHTML (the content is sanitized on the server).
    md = '\n\n' + (v.figure_num || 'Table') + ': '
         + (v.caption || '') + '\n\n'
         + (v.content || '').substring(0, 2000) + '\n\n';
  } else {
    md = '\n\n```\n' + (v.content || '').substring(0, 1000) + '\n```\n\n';
  }
  // Target our real editor (#edit-area). Fall back to the old
  // selectors for unknown hosts; then clipboard.
  const editor = document.getElementById('edit-area')
              || document.querySelector('.editor-area textarea, #section-editor');
  if (editor) {
    const start = editor.selectionStart || editor.value.length;
    editor.value = editor.value.slice(0, start) + md + editor.value.slice(start);
    editor.focus();
    // Re-render the live preview so the just-inserted thumbnail /
    // equation shows immediately instead of needing a keystroke first.
    if (typeof edPreview === 'function') edPreview();
    closeModal('visuals-modal');
  } else {
    navigator.clipboard.writeText(md).then(() => {
      alert('Copied to clipboard — paste into the editor.');
      closeModal('visuals-modal');
    });
  }
}

// ── Phase 54.6.24 — Backups ──────────────────────────────────────────
let _backupScheduleActive = false;

function openBackupsModal() {
  document.querySelectorAll('.nav-dropdown.open').forEach(d => d.classList.remove('open'));
  openModal('backups-modal');
  refreshBackupsList();
}

async function refreshBackupsList() {
  try {
    const res = await fetch('/api/backups');
    const d = await res.json();
    const status = document.getElementById('backup-status');
    const location = document.getElementById('backup-location');
    const list = document.getElementById('backup-list');
    const unschedBtn = document.getElementById('backup-unschedule-btn');

    // Status
    const backups = d.backups || [];
    const sched = d.schedule;
    _backupScheduleActive = !!sched;
    let statusHtml = '';
    if (backups.length) {
      const last = backups[backups.length - 1];
      const mb = (last.total_bytes || 0) / 1024 / 1024;
      statusHtml += '<strong>Last backup:</strong> ' + last.timestamp
        + ' (' + mb.toFixed(1) + ' MB, '
        + (last.projects || []).join(', ') + ')';
    } else {
      statusHtml += '<span class="u-warning">No backups yet.</span>';
    }
    statusHtml += '<br>';
    if (sched) {
      const human = sched.human || sched.cron_expression || '?';
      statusHtml += '<strong>Schedule:</strong> ' + human
        + ' <span class="u-hint">(cron: ' + sched.cron_expression + ')</span>';
      if (unschedBtn) unschedBtn.classList.remove('u-hidden');
      // Populate the schedule form with current values
      const freq = sched.frequency || 'daily';
      const freqEl = document.getElementById('backup-sched-freq');
      if (freqEl) freqEl.value = freq;
      const parts = (sched.cron_expression || '0 3 * * *').split(' ');
      if (parts.length >= 5) {
        const minEl = document.getElementById('backup-sched-minute');
        if (minEl && /^\d+$/.test(parts[0])) minEl.value = parts[0];
        const hourEl = document.getElementById('backup-sched-hour');
        if (hourEl && /^\d+$/.test(parts[1])) hourEl.value = parts[1];
        const wdEl = document.getElementById('backup-sched-weekday');
        if (wdEl && /^\d+$/.test(parts[4])) wdEl.value = parts[4];
      }
      _updateScheduleFormVisibility();
    } else {
      statusHtml += '<span class="u-muted">No schedule active.</span>';
      if (unschedBtn) unschedBtn.classList.add('u-hidden');
    }
    status.innerHTML = statusHtml;

    // Location
    if (location) {
      location.innerHTML = '<strong>Location:</strong> ' + (d.backup_dir || '?');
    }

    // Badge on Manage button
    const badge = document.getElementById('backup-badge');
    if (badge) {
      if (!backups.length) {
        badge.style.background = 'var(--danger, #e74c3c)';
      } else {
        const lastTs = backups[backups.length - 1].timestamp;
        const ageSec = (Date.now() - new Date(lastTs).getTime()) / 1000;
        badge.style.background = ageSec < 90000 ? 'var(--success, #27ae60)'
          : ageSec < 172800 ? 'var(--warning, #f39c12)'
          : 'var(--danger, #e74c3c)';
      }
    }

    // List with delete + restore per row
    if (!backups.length) {
      list.innerHTML = '<em class="u-muted">No backups. Click "Run Backup Now".</em>';
      return;
    }
    let html = '<table class="u-table-full-sm">';
    html += '<tr class="u-border-b"><th class="u-pad-xs">Date</th><th>Projects</th><th class="u-text-right">Size</th><th>Sys</th><th>Files</th><th>Actions</th></tr>';
    for (let i = backups.length - 1; i >= 0; i--) {
      const b = backups[i];
      const mb = (b.total_bytes || 0) / 1024 / 1024;
      const files = Object.keys(b.files || {});
      const dlLinks = files.map(f =>
        '<a href="/api/backups/download/' + encodeURIComponent(b.dir) + '/' + encodeURIComponent(f) + '" download style="color:var(--link);text-decoration:underline;font-size:11px;">' + f + '</a>'
      ).join('<br>');
      const safeTs = b.dir.replace(/"/g, '');
      const actions =
        '<button class="btn-secondary u-tiny u-p-2-6" onclick="restoreBackup(\''
          + safeTs + '\')" title="Restore this backup">\u21BB</button> '
        + '<button class="btn-secondary" style="font-size:11px;padding:2px 6px;color:var(--danger);border-color:var(--danger);" '
          + 'onclick="deleteBackup(\'' + safeTs + '\')" title="Delete this backup">\u2715</button>';
      html += '<tr class="u-border-b">'
        + '<td class="u-pad-xs">' + b.timestamp + '</td>'
        + '<td>' + (b.projects || []).join(', ') + '</td>'
        + '<td class="u-text-right">' + mb.toFixed(1) + ' MB</td>'
        + '<td class="u-text-center">' + (b.system_bundle ? '\u2713' : '\u2014') + '</td>'
        + '<td>' + dlLinks + '</td>'
        + '<td>' + actions + '</td>'
        + '</tr>';
    }
    html += '</table>';
    const totalMb = backups.reduce((s, b) => s + (b.total_bytes || 0), 0) / 1024 / 1024;
    html += '<div class="u-note-mt-6">'
      + backups.length + ' backup(s), ' + totalMb.toFixed(1) + ' MB total.</div>';
    list.innerHTML = html;
  } catch (exc) {
    document.getElementById('backup-list').innerHTML = '<span class="u-danger">Failed to load: ' + exc + '</span>';
  }
}

function _updateScheduleFormVisibility() {
  const freq = (document.getElementById('backup-sched-freq') || {}).value || 'daily';
  const wdLabel = document.getElementById('backup-sched-weekday-label');
  const hourLabel = document.getElementById('backup-sched-hour-label');
  if (wdLabel) wdLabel.style.display = (freq === 'weekly') ? '' : 'none';
  if (hourLabel) hourLabel.style.display = (freq === 'hourly') ? 'none' : '';
}
document.addEventListener('DOMContentLoaded', function() {
  const sel = document.getElementById('backup-sched-freq');
  if (sel) sel.addEventListener('change', _updateScheduleFormVisibility);
});

async function runBackupNow() {
  const log = document.getElementById('backup-log');
  log.style.display = 'block';
  log.textContent = 'Starting backup...\n';
  try {
    const res = await fetch('/api/backups/run', {method: 'POST'});
    const d = await res.json();
    if (!res.ok) {
      log.textContent += 'Failed: ' + (d.detail || res.status) + '\n';
      return;
    }
    const source = new EventSource('/api/stream/' + d.job_id);
    source.onmessage = function(e) {
      const evt = JSON.parse(e.data);
      if (evt.type === 'log') {
        log.textContent += evt.text + '\n';
        log.scrollTop = log.scrollHeight;
      } else if (evt.type === 'completed') {
        log.textContent += '\n\u2713 Backup complete.\n';
        source.close();
        refreshBackupsList();
      } else if (evt.type === 'error') {
        log.textContent += '\u2717 ' + (evt.message || 'error') + '\n';
        source.close();
      }
    };
    source.onerror = function() { source.close(); };
  } catch (exc) {
    log.textContent += 'Failed: ' + exc + '\n';
  }
}

async function restoreBackup(ts) {
  const timestamp = ts || 'latest';
  if (!confirm('Restore from backup "' + timestamp + '"?\n\nThis will destroy existing projects with matching slugs and replace them with the backup version. Qdrant vectors will need rebuilding after restore.\n\nContinue?')) return;
  const log = document.getElementById('backup-log');
  log.style.display = 'block';
  log.textContent = 'Starting restore from ' + timestamp + '...\n';
  try {
    const res = await fetch('/api/backups/restore', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({timestamp: timestamp, force: true}),
    });
    const d = await res.json();
    if (!res.ok) {
      log.textContent += 'Failed: ' + (d.detail || res.status) + '\n';
      return;
    }
    const source = new EventSource('/api/stream/' + d.job_id);
    source.onmessage = function(e) {
      const evt = JSON.parse(e.data);
      if (evt.type === 'log') {
        log.textContent += evt.text + '\n';
        log.scrollTop = log.scrollHeight;
      } else if (evt.type === 'completed') {
        log.textContent += '\n\u2713 Restore complete. Restart the server to pick up the restored project.\n';
        source.close();
        refreshBackupsList();
      } else if (evt.type === 'error') {
        log.textContent += '\u2717 ' + (evt.message || 'error') + '\n';
        source.close();
      }
    };
    source.onerror = function() { source.close(); };
  } catch (exc) {
    log.textContent += 'Failed: ' + exc + '\n';
  }
}

function _streamJobToBackupLog(jobId, startMsg) {
  const log = document.getElementById('backup-log');
  log.style.display = 'block';
  log.textContent = startMsg + '\n';
  const source = new EventSource('/api/stream/' + jobId);
  source.onmessage = function(e) {
    const evt = JSON.parse(e.data);
    if (evt.type === 'log') {
      log.textContent += evt.text + '\n';
      log.scrollTop = log.scrollHeight;
    } else if (evt.type === 'completed' || evt.type === 'error') {
      source.close();
      refreshBackupsList();
    }
  };
  source.onerror = function() { source.close(); };
}

async function enableBackupSchedule() {
  const freq = (document.getElementById('backup-sched-freq').value || 'daily').trim();
  const hour = parseInt(document.getElementById('backup-sched-hour').value || '3', 10);
  const minute = parseInt(document.getElementById('backup-sched-minute').value || '0', 10);
  const weekday = parseInt(document.getElementById('backup-sched-weekday').value || '0', 10);
  try {
    const res = await fetch('/api/backups/schedule', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        action: 'enable',
        frequency: freq, hour: hour, minute: minute, weekday: weekday,
      }),
    });
    const d = await res.json();
    if (d.job_id) _streamJobToBackupLog(d.job_id, 'Installing ' + freq + ' schedule...');
  } catch (exc) { alert('Schedule failed: ' + exc); }
}

async function disableBackupSchedule() {
  if (!confirm('Remove auto-backup schedule?')) return;
  try {
    const res = await fetch('/api/backups/schedule', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action: 'disable'}),
    });
    const d = await res.json();
    if (d.job_id) _streamJobToBackupLog(d.job_id, 'Removing crontab entry...');
  } catch (exc) { alert('Disable failed: ' + exc); }
}

async function deleteBackup(ts) {
  if (!confirm('Delete backup "' + ts + '"?\n\nThis removes the backup files on disk and cannot be undone.')) return;
  try {
    const res = await fetch('/api/backups/delete', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({timestamp: ts}),
    });
    const d = await res.json();
    if (d.job_id) _streamJobToBackupLog(d.job_id, 'Deleting ' + ts + '...');
  } catch (exc) { alert('Delete failed: ' + exc); }
}

async function purgeBackups() {
  const all = document.getElementById('backup-purge-all-check').checked;
  const days = parseInt(document.getElementById('backup-purge-days').value || '0', 10);
  let body, confirmMsg;
  if (all) {
    body = {all: true};
    confirmMsg = 'Delete ALL backups?\n\nThis wipes every backup in archives/backups/ and cannot be undone.';
  } else if (days > 0) {
    body = {older_than_days: days};
    confirmMsg = 'Delete backups older than ' + days + ' days?\n\nCannot be undone.';
  } else {
    alert('Check "Delete ALL backups" or enter a positive number of days.');
    return;
  }
  if (!confirm(confirmMsg)) return;
  try {
    const res = await fetch('/api/backups/purge', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    const d = await res.json();
    if (d.job_id) _streamJobToBackupLog(d.job_id, 'Purging...');
  } catch (exc) { alert('Purge failed: ' + exc); }
}

// Load backup badge on page load
setTimeout(function() {
  fetch('/api/backups').then(r => r.json()).then(d => {
    const badge = document.getElementById('backup-badge');
    if (!badge) return;
    const backups = d.backups || [];
    if (!backups.length) {
      badge.style.background = 'var(--danger, #e74c3c)';
    } else {
      const lastTs = backups[backups.length - 1].timestamp;
      const ageSec = (Date.now() - new Date(lastTs).getTime()) / 1000;
      badge.style.background = ageSec < 90000 ? 'var(--success, #27ae60)'
        : ageSec < 172800 ? 'var(--warning, #f39c12)'
        : 'var(--danger, #e74c3c)';
    }
  }).catch(() => {});
}, 2000);

// ── Projects ─────────────────────────────────────────────────────────

function openProjectsModal() {
  openModal('projects-modal');
  refreshProjectsList();
  // Phase 54.6.112 — load the venue config for the active project.
  refreshVenueConfig();
}

// Phase 54.6.112 — venue block/allow UI
async function refreshVenueConfig() {
  try {
    const res = await fetch('/api/projects/active/venues');
    if (!res.ok) return;
    const d = await res.json();
    const blockList = document.getElementById('proj-ven-block-list');
    const allowList = document.getElementById('proj-ven-allow-list');
    const blockCount = document.getElementById('proj-ven-block-count');
    const allowCount = document.getElementById('proj-ven-allow-count');
    const render = (items, ulEl, kind) => {
      if (!ulEl) return;
      if (!items.length) {
        ulEl.innerHTML = '<li class="u-muted u-py-1">(empty)</li>';
        return;
      }
      ulEl.innerHTML = items.map(p => {
        const enc = _escHtml(p);
        return '<li style="display:flex;justify-content:space-between;padding:2px 0;border-bottom:1px dotted var(--border);">'
          + '<code class="u-mono-xs">' + enc + '</code>'
          + '<button onclick="removeVenuePattern(\'' + kind + '\', ' + JSON.stringify(p) + ')" '
            + 'style="background:none;border:none;color:var(--fg-muted);cursor:pointer;font-size:13px;padding:0 4px;" '
            + 'title="remove">\u00d7</button>'
          + '</li>';
      }).join('');
    };
    render(d.blocklist || [], blockList, 'block');
    render(d.allowlist || [], allowList, 'allow');
    if (blockCount) blockCount.textContent = (d.blocklist || []).length + ' pattern(s)';
    if (allowCount) allowCount.textContent = (d.allowlist || []).length + ' pattern(s)';
  } catch (_) {}
}

async function addVenuePattern(kind) {
  const inp = document.getElementById('proj-ven-' + kind + '-in');
  const pat = (inp.value || '').trim();
  if (!pat) return;
  try {
    const res = await fetch('/api/projects/active/venues', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action: 'add', kind: kind, pattern: pat}),
    });
    if (res.ok) { inp.value = ''; refreshVenueConfig(); }
  } catch (_) {}
}

async function removeVenuePattern(kind, pattern) {
  try {
    const res = await fetch('/api/projects/active/venues', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action: 'remove', kind: kind, pattern: pattern}),
    });
    if (res.ok) refreshVenueConfig();
  } catch (_) {}
}

function _projMsg(text, kind) {
  const el = document.getElementById('proj-msg');
  if (!el) return;
  el.textContent = text || '';
  el.style.color = kind === 'error' ? 'var(--danger)'
                 : kind === 'ok'    ? 'var(--success)'
                 : 'var(--fg-muted)';
}

async function refreshProjectsList() {
  _projMsg('Loading…');
  const wrap = document.getElementById('projects-list');
  try {
    const resp = await fetch('/api/projects');
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const data = await resp.json();
    document.getElementById('proj-running').textContent = data.running_slug || '(unknown)';
    if (!data.projects || data.projects.length === 0) {
      wrap.innerHTML = '<em class="u-muted">No projects yet. Create one below.</em>';
      _projMsg('');
      return;
    }
    const rows = data.projects.map(p => {
      const activeMark = p.active ? '<span class="u-accent u-semibold">●</span>' : '<span class="u-faint">○</span>';
      const statusBadge = p.status === 'ok'
        ? '<span class="u-success">ok</span>'
        : '<span class="u-warning">incomplete</span>';
      const isRunning = p.slug === data.running_slug;
      const useBtn    = p.active ? ''
        : `<button onclick="useProject('${p.slug}')" title="Set .active-project to ${p.slug}">Use</button>`;
      const destroyBtn = (p.is_default || isRunning) ? ''
        : `<button class="u-danger" onclick="destroyProject('${p.slug}')" title="Drop DB + collections + data dir">Destroy</button>`;
      const showBtn = `<button onclick="showProjectDetail('${p.slug}')" title="Show this project's stats (paper count, chunk count, embedding model) and migration + venue-config state.">Details</button>`;
      return `<tr>
        <td style="text-align:center;width:30px;">${activeMark}</td>
        <td class="u-semibold">${p.slug}${isRunning ? ' <span class="u-xxs u-accent">(running)</span>' : ''}</td>
        <td class="u-muted u-mono u-tiny">${p.pg_database}</td>
        <td class="u-muted u-mono u-tiny">${p.papers_collection}</td>
        <td>${statusBadge}</td>
        <td class="u-nowrap">${showBtn} ${useBtn} ${destroyBtn}</td>
      </tr>`;
    }).join('');
    wrap.innerHTML = `<table class="u-table-full-sm">
      <thead><tr class="u-border-b u-muted">
        <th></th><th>Slug</th><th>PG DB</th><th>Papers coll.</th><th>Status</th><th></th>
      </tr></thead>
      <tbody>${rows}</tbody></table>`;
    _projMsg(data.projects.length + ' project' + (data.projects.length === 1 ? '' : 's') + '.', 'ok');
  } catch (exc) {
    wrap.innerHTML = '';
    _projMsg('Failed to list projects: ' + exc, 'error');
  }
}

async function showProjectDetail(slug) {
  const dest = document.getElementById('proj-detail');
  dest.innerHTML = 'Loading details for <code>' + slug + '</code>…';
  try {
    const resp = await fetch('/api/projects/' + encodeURIComponent(slug));
    if (!resp.ok) {
      const msg = await resp.text();
      throw new Error('HTTP ' + resp.status + ': ' + msg);
    }
    const d = await resp.json();
    const counts = (d.n_documents !== undefined)
      ? `<ul style="margin:6px 0 0 18px;font-size:12px;">
           <li>Documents: <strong>${(d.n_documents||0).toLocaleString()}</strong></li>
           <li>Chunks: <strong>${(d.n_chunks||0).toLocaleString()}</strong></li>
           <li>Books: <strong>${d.n_books||0}</strong></li>
           <li>Drafts: <strong>${d.n_drafts||0}</strong></li>
         </ul>`
      : (d.counts_error ? `<div class="u-warning u-tiny">Counts unavailable: ${d.counts_error}</div>` : '');
    dest.innerHTML = `<div style="border:1px solid var(--border);border-radius:var(--r-md);padding:10px;background:var(--toolbar-bg);">
      <div class="u-row-between-mb">
        <strong>${d.slug}${d.is_default ? ' <span class="u-hint">(legacy default)</span>' : ''}</strong>
        <button onclick="document.getElementById('proj-detail').innerHTML=''" title="Close the project details panel.">&times;</button>
      </div>
      <dl style="display:grid;grid-template-columns:140px 1fr;gap:4px 12px;font-size:12px;margin:0;">
        <dt class="u-muted">Root</dt><dd class="u-mono-xs">${d.root}</dd>
        <dt class="u-muted">Data dir</dt><dd class="u-mono-xs">${d.data_dir}${d.data_dir_exists ? '' : ' <span class="u-warning">(missing)</span>'}</dd>
        <dt class="u-muted">PG database</dt><dd class="u-mono-xs">${d.pg_database}${d.pg_database_exists ? '' : ' <span class="u-warning">(missing)</span>'}</dd>
        <dt class="u-muted">Qdrant prefix</dt><dd class="u-mono-xs">${d.qdrant_prefix || '(none)'}</dd>
        <dt class="u-muted">Collections</dt><dd class="u-mono-xs">${d.papers_collection}, ${d.abstracts_collection}, ${d.wiki_collection}</dd>
        <dt class="u-muted">Env overlay</dt><dd class="u-mono-xs">${d.env_overlay_path}${d.env_overlay_exists ? '' : ' <span class="u-faint">(not present)</span>'}</dd>
      </dl>
      ${counts}
    </div>`;
  } catch (exc) {
    dest.innerHTML = '<div class="u-danger u-small">Failed: ' + exc + '</div>';
  }
}

async function useProject(slug) {
  _projMsg('Switching active project to ' + slug + '…');
  try {
    const resp = await fetch('/api/projects/use', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({slug: slug}),
    });
    const data = await resp.json();
    if (!resp.ok) {
      _projMsg('Use failed: ' + (data.detail || resp.status), 'error');
      return;
    }
    _projMsg(data.message || ('Active project: ' + slug), 'ok');
    if (data.restart_required) {
      renderProjectSwitchBanner(slug, data.running_slug);
    }
    refreshProjectsList();
  } catch (exc) {
    _projMsg('Use failed: ' + exc, 'error');
  }
}

function renderProjectSwitchBanner(newSlug, runningSlug) {
  // Phase 54.6.2 — after a successful /api/projects/use, replace the
  // old "Ctrl-C your terminal" alert with an inline banner that offers
  // a one-click graceful shutdown. The user still needs to rerun
  // `sciknow book serve` themselves (no supervisor to auto-restart)
  // but at least they don't have to leave the browser.
  const dest = document.getElementById('proj-detail');
  if (!dest) return;
  const safeNew = _escHtml(newSlug);
  const safeRun = _escHtml(runningSlug || '?');
  const cmd = 'sciknow --project ' + safeNew + ' book serve "<book title>"';
  dest.innerHTML =
    '<div style="margin-top:14px;padding:14px;border:2px solid var(--accent);'
    + 'border-radius:8px;background:var(--bg-elevated);">'
    + '<div class="u-bold u-mb-6">&#9888;&#65039; Restart required</div>'
    + '<div class="u-note-md">'
    + 'The <code>.active-project</code> file now points at <strong>' + safeNew
    + '</strong>, but this server is still bound to <strong>' + safeRun
    + '</strong>. DB / Qdrant clients can&rsquo;t hot-swap, so you need to '
    + 'restart the server to work on the new project.'
    + '</div>'
    + '<div class="u-row-wrap">'
    + '<button class="btn-primary" onclick="shutdownServer()" '
    + 'title="Graceful shutdown — your terminal will return to $, ready for the re-run command below">'
    + '&#9211; Stop this server</button>'
    + '<code class="u-flex-1 u-pill-md u-bg-tb u-r-sm u-small u-click"'
    + ' onclick="navigator.clipboard.writeText(this.textContent);_projMsg(&quot;Command copied.&quot;,&quot;ok&quot;);" '
    + 'title="Click to copy">' + _escHtml(cmd) + '</code>'
    + '</div>'
    + '<div class="u-mt-2 u-tiny u-muted">'
    + 'After stopping, paste the command into your terminal and edit '
    + '<code>"&lt;book title&gt;"</code> to a book that exists in <strong>'
    + safeNew + '</strong>.</div>'
    + '</div>';
}

// Phase 54.6.x — top-level "Close server / exit session" flow.
// Distinct from the Projects-modal restart flow (shutdownServer below)
// because it lives outside the project-switching context and offers
// the option to also stop the llama-server substrate. Reuses the
// /api/server/shutdown endpoint with body.stop_substrate = true|false.
function openShutdownModal() {
  const cb = document.getElementById('shutdown-stop-substrate');
  if (cb) cb.checked = false;
  const btn = document.getElementById('shutdown-confirm-btn');
  if (btn) btn.disabled = false;
  // Close the Book menu dropdown if it's open.
  document.querySelectorAll('.nav-dropdown.open').forEach(d => d.classList.remove('open'));
  openModal('shutdown-modal');
}

async function confirmShutdown() {
  const cb = document.getElementById('shutdown-stop-substrate');
  const stopSubstrate = !!(cb && cb.checked);
  const btn = document.getElementById('shutdown-confirm-btn');
  if (btn) {
    btn.disabled = true;
    btn.textContent = '⏻ Stopping…';
  }
  let payload = {stop_substrate: stopSubstrate};
  let resp;
  try {
    const r = await fetch('/api/server/shutdown', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    resp = await r.json().catch(() => ({}));
  } catch (_) {
    // Connection drop is the expected outcome — uvicorn shuts down
    // ~200 ms after the response flushes, so the fetch may race the
    // close. Treat as success and render the goodbye screen anyway.
    resp = {ok: true, stopped_substrate_roles: []};
  }
  closeModal('shutdown-modal');
  const stoppedRoles = (resp && resp.stopped_substrate_roles) || [];
  document.body.innerHTML =
    '<div style="padding:48px;max-width:640px;margin:64px auto;'
    + 'font-family:-apple-system,Inter Tight,sans-serif;'
    + 'border:1px solid var(--border,#ccc);border-radius:10px;'
    + 'background:var(--bg-elevated,#fff);color:var(--fg,#111);'
    + 'box-shadow:0 4px 16px rgba(0,0,0,0.08);">'
    + '<h2 style="margin:0 0 12px;">SciKnow server stopped</h2>'
    + '<p style="font-size:14px;line-height:1.55;color:var(--fg-muted,#555);">'
    + 'Your terminal is back at <code>$</code>. '
    + (stoppedRoles.length
        ? 'Also stopped llama-server roles: <code>' + stoppedRoles.join(', ') + '</code>. '
        : 'The llama-server substrate is still running (use <code>sciknow infer down</code> to stop it).')
    + '</p>'
    + '<p style="font-size:14px;line-height:1.55;">To restart the reader, run:</p>'
    + '<pre style="padding:12px 14px;background:var(--bg,#f5f4ef);border:1px solid var(--border,#e7e5e0);'
    + 'border-radius:6px;font-size:13px;font-family:JetBrains Mono,ui-monospace,monospace;'
    + 'color:var(--fg,#111);overflow-x:auto;">'
    + 'sciknow book serve "&lt;book title&gt;"</pre>'
    + '<p style="font-size:12px;color:var(--fg-faint,#999);margin:14px 0 0;">'
    + 'Then reload this browser tab.</p>'
    + '</div>';
}

async function shutdownServer() {
  if (!confirm(
    'Stop the running sciknow server?\n\n'
    + 'This will return your terminal to the shell prompt. Any in-flight LLM job '
    + 'will be killed. You will need to rerun `sciknow book serve ...` manually '
    + 'to open the reader again.'
  )) return;
  _projMsg('Shutting down…');
  try {
    await fetch('/api/server/shutdown', {method: 'POST'});
    document.body.innerHTML =
      '<div style="padding:40px;max-width:620px;margin:60px auto;font-family:-apple-system,sans-serif;'
      + 'border:1px solid var(--border);border-radius:8px;background:var(--bg-elevated);'
      + 'color:var(--fg);">'
      + '<h2 style="margin:0 0 12px;">Server stopped</h2>'
      + '<p>Your terminal is back at <code>$</code>. To pick up the new active project, run:</p>'
      + '<pre style="padding:10px;background:var(--bg);border:1px solid var(--border);'
      + 'border-radius:4px;color:var(--fg);">'
      + 'sciknow book serve "&lt;book title&gt;"</pre>'
      + '<p style="font-size:12px;color:var(--fg-muted);margin:12px 0 0;">Then reload this browser tab.</p>'
      + '</div>';
  } catch (exc) {
    _projMsg('Shutdown request failed: ' + exc, 'error');
  }
}

async function createProject() {
  const slug = (document.getElementById('proj-new-slug').value || '').trim();
  if (!slug) {
    _projMsg('Enter a slug first.', 'error');
    return;
  }
  if (!/^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$/.test(slug)) {
    _projMsg('Slug must be lowercase alphanumerics + hyphens (e.g. "global-cooling").', 'error');
    return;
  }
  if (!confirm('Create empty project "' + slug + '"? This runs migrations + initialises Qdrant collections (takes a few seconds).')) return;
  _projMsg('Creating ' + slug + '…');
  try {
    const resp = await fetch('/api/projects/init', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({slug: slug}),
    });
    const data = await resp.json();
    if (!resp.ok) {
      _projMsg('Create failed: ' + (data.detail || resp.status), 'error');
      return;
    }
    _projMsg('Created ' + slug + ' (DB: ' + data.pg_database + ').', 'ok');
    document.getElementById('proj-new-slug').value = '';
    refreshProjectsList();
  } catch (exc) {
    _projMsg('Create failed: ' + exc, 'error');
  }
}

async function destroyProject(slug) {
  const confirmSlug = prompt(
    'DESTROY project "' + slug + '"?\n\n'
    + 'This drops the PostgreSQL database, the Qdrant collections, and the data directory. '
    + 'Run `sciknow project archive ' + slug + '` from the CLI first if you might want it back.\n\n'
    + 'Type the slug to confirm:');
  if (confirmSlug === null) return;
  if (confirmSlug !== slug) {
    _projMsg('Slug did not match — destroy cancelled.', 'error');
    return;
  }
  _projMsg('Destroying ' + slug + '…');
  try {
    const resp = await fetch('/api/projects/destroy', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({slug: slug, confirm: slug}),
    });
    const data = await resp.json();
    if (!resp.ok) {
      _projMsg('Destroy failed: ' + (data.detail || resp.status), 'error');
      return;
    }
    const errs = (data.errors && data.errors.length) ? (' (errors: ' + data.errors.join('; ') + ')') : '';
    _projMsg('Destroyed ' + slug + errs, errs ? 'error' : 'ok');
    refreshProjectsList();
  } catch (exc) {
    _projMsg('Destroy failed: ' + exc, 'error');
  }
}



// ── Phase 54.6.328 (snapshot-versioning Phase 5) — Timeline modal ──────────
//
// Unified replacement for the old History modal + Snapshots modal. Three
// scopes (Section / Chapter / Book) accessed via tabs; each row carries
// the diff brief from the meta column. Compare-any-two via row checkboxes
// + the existing /api/diff endpoint.

let _tlScope = 'section';        // 'section' | 'chapter' | 'book'
let _tlSelected = [];            // up to 2 row ids for compare

function _tlBriefHtml(meta) {
  if (!meta || typeof meta !== 'object') return '<span class="u-muted">—</span>';
  const totals = meta.totals || meta;  // bundle vs prose shape
  const wa = totals.words_added || 0;
  const wr = totals.words_removed || 0;
  const pa = totals.paragraphs_added || 0;
  const pr = totals.paragraphs_removed || 0;
  const ca = totals.citations_added || 0;
  const cr = totals.citations_removed || 0;
  const bits = [];
  if (wa || wr) bits.push('+' + wa.toLocaleString() + '/-' + wr.toLocaleString() + 'w');
  if (pa || pr) bits.push('+' + pa + '/-' + pr + '¶');
  if (ca || cr) bits.push('+' + ca + '/-' + cr + 'cite');
  if (meta.totals && (meta.totals.sections_total != null)) {
    bits.push((meta.totals.sections_changed || 0) + '/' + meta.totals.sections_total + '§');
  }
  const struct = meta.structural;
  if (struct) {
    const sa = (struct.added_chapters || []).length;
    const sr = (struct.removed_chapters || []).length;
    const sn = (struct.renamed_chapters || []).length;
    if (sa) bits.push('+' + sa + 'ch');
    if (sr) bits.push('-' + sr + 'ch');
    if (sn) bits.push('~' + sn + 'ch');
  }
  return bits.length
    ? '<span class="u-muted">' + bits.join(' · ') + '</span>'
    : '<span class="u-muted">—</span>';
}

function _tlSetActiveTab(scope) {
  document.querySelectorAll('#tl-tabs button[data-tl-scope]').forEach(b => {
    if (b.dataset.tlScope === scope) {
      b.classList.add('active');
      b.setAttribute('aria-selected', 'true');
    } else {
      b.classList.remove('active');
      b.setAttribute('aria-selected', 'false');
    }
  });
}

async function tlSwitchScope(scope) {
  _tlScope = scope;
  _tlSelected = [];
  _tlSetActiveTab(scope);
  document.getElementById('tl-compare').classList.add('u-hidden');
  document.getElementById('tl-body').innerHTML = '<p class="u-muted">Loading…</p>';
  await tlRender();
}

async function tlRender() {
  const body = document.getElementById('tl-body');
  const status = document.getElementById('tl-status');
  if (_tlScope === 'section') {
    if (!currentDraftId) {
      body.innerHTML = '<p class="u-muted">Open a section first to see its history.</p>';
      status.textContent = '';
      return;
    }
    const r = await fetch('/api/timeline/section/' + currentDraftId);
    if (!r.ok) { body.innerHTML = '<p class="u-muted">No data.</p>'; return; }
    const data = await r.json();
    const entries = data.entries || [];
    status.textContent = entries.length + ' entries';
    body.innerHTML = _tlSectionHtml(entries);
  } else if (_tlScope === 'chapter') {
    if (!currentChapterId) {
      body.innerHTML = '<p class="u-muted">Open a chapter or one of its sections first.</p>';
      status.textContent = '';
      return;
    }
    const r = await fetch('/api/timeline/chapter/' + currentChapterId);
    if (!r.ok) { body.innerHTML = '<p class="u-muted">No data.</p>'; return; }
    const data = await r.json();
    status.textContent =
      (data.sections || []).length + ' sections · ' +
      (data.snapshots || []).length + ' snapshots';
    body.innerHTML = _tlChapterHtml(data);
  } else if (_tlScope === 'book') {
    const bid = (window.SCIKNOW_BOOTSTRAP && window.SCIKNOW_BOOTSTRAP.bookId) || '';
    if (!bid) {
      body.innerHTML = '<p class="u-muted">No active book.</p>';
      status.textContent = '';
      return;
    }
    const r = await fetch('/api/timeline/book/' + bid);
    if (!r.ok) { body.innerHTML = '<p class="u-muted">No data.</p>'; return; }
    const data = await r.json();
    const snaps = data.snapshots || [];
    status.textContent = snaps.length + ' snapshots';
    body.innerHTML = _tlBookHtml(snaps);
  }
}

function _tlSectionHtml(entries) {
  if (!entries.length) return '<p class="u-muted">No history yet for this section.</p>';
  let html = '<table class="stats-table" style="width:100%;font-size:13px;">'
    + '<tr><th style="width:32px;">·</th>'
    + '<th>Version / snapshot</th><th>Date</th>'
    + '<th style="text-align:right;">Words</th><th>Δ</th><th>Actions</th></tr>';
  for (const e of entries) {
    const checked = _tlSelected.includes(e.id) ? 'checked' : '';
    const activeBadge = e.is_active
      ? ' <span class="u-faint" style="color:var(--accent);">✓ active</span>'
      : '';
    const score = (e.extra && typeof e.extra.final_overall === 'number')
      ? ' <span class="u-muted">score=' + e.extra.final_overall.toFixed(2) + '</span>'
      : '';
    html += '<tr><td><input type="checkbox" data-tl-id="' + _escHTML(e.id) + '" '
      + checked + ' onchange="tlOnSelect(this)"></td>'
      + '<td>'
        + '<code>' + _escHTML(e.id.slice(0, 8)) + '</code> '
        + '<span class="u-muted">' + _escHTML(e.kind) + '</span> '
        + '<strong>' + _escHTML(e.label) + '</strong>'
        + activeBadge + score
      + '</td>'
      + '<td class="u-muted">' + _escHTML(e.created_at.slice(0, 19)) + '</td>'
      + '<td style="text-align:right;">' + (e.word_count || 0).toLocaleString() + '</td>'
      + '<td>' + _tlBriefHtml(e.meta) + '</td>'
      + '<td>'
        + (e.kind === 'draft' && !e.is_active
            ? '<button class="btn btn--sm" onclick="tlActivateDraft(\'' + _escHTML(e.id) + '\')" '
              + 'title="Mark this draft version active so the reader + bibliography pick it up.">Activate</button> '
            : '')
        + (e.kind === 'snapshot'
            ? '<button class="btn btn--sm" onclick="tlRestoreSnapshot(\'' + _escHTML(e.id) + '\')" '
              + 'title="Insert this snapshot as a NEW draft version (non-destructive).">Restore</button>'
            : '')
      + '</td></tr>';
  }
  html += '</table>';
  return html;
}

function _tlChapterHtml(data) {
  let html = '';
  const secs = data.sections || [];
  if (secs.length) {
    html += '<h4>Latest per section</h4>'
      + '<table class="stats-table" style="width:100%;font-size:13px;">'
      + '<tr><th>Section</th><th>v</th><th>Date</th>'
      + '<th style="text-align:right;">Words</th><th>·</th></tr>';
    for (const s of secs) {
      html += '<tr><td>'
        + (s.is_active ? '<span style="color:var(--accent);">✓</span> ' : '')
        + _escHTML(s.section_type) + '</td>'
        + '<td>' + (s.version || 0) + '</td>'
        + '<td class="u-muted">' + _escHTML(s.created_at.slice(0, 19)) + '</td>'
        + '<td style="text-align:right;">' + (s.word_count || 0).toLocaleString() + '</td>'
        + '<td><button class="btn btn--sm" onclick="loadSection(\'' + _escHTML(s.id) + '\');closeModal(\'timeline-modal\');" '
          + 'title="Open this section in the reader.">Open</button></td>'
        + '</tr>';
    }
    html += '</table>';
  }
  const snaps = data.snapshots || [];
  if (snaps.length) {
    html += '<h4 style="margin-top:18px;">Chapter snapshots</h4>';
    html += _tlSnapshotsTable(snaps, 'chapter');
  }
  return html || '<p class="u-muted">No history yet for this chapter.</p>';
}

function _tlBookHtml(snaps) {
  if (!snaps.length) return '<p class="u-muted">No snapshots yet for this book.</p>';
  return _tlSnapshotsTable(snaps, 'book');
}

function _tlSnapshotsTable(snaps, kind) {
  let html = '<table class="stats-table" style="width:100%;font-size:13px;">'
    + '<tr><th style="width:32px;">·</th>'
    + '<th>Snapshot</th><th>Scope</th><th>Date</th>'
    + '<th style="text-align:right;">Words</th><th>Δ</th><th>Actions</th></tr>';
  for (const s of snaps) {
    const checked = _tlSelected.includes(s.id) ? 'checked' : '';
    const scopeTag = (s.scope === 'book')
      ? '<span class="u-muted">book</span>'
      : ('<span class="u-muted">chapter Ch.' + (s.chapter_number || '?') + '</span>');
    html += '<tr><td><input type="checkbox" data-tl-id="' + _escHTML(s.id) + '" '
      + checked + ' onchange="tlOnSelect(this)"></td>'
      + '<td><code>' + _escHTML(s.id.slice(0, 8)) + '</code> '
      + _escHTML(s.name || '') + '</td>'
      + '<td>' + scopeTag + '</td>'
      + '<td class="u-muted">' + _escHTML((s.created_at || '').slice(0, 19)) + '</td>'
      + '<td style="text-align:right;">' + (s.word_count || 0).toLocaleString() + '</td>'
      + '<td>' + _tlBriefHtml(s.meta) + '</td>'
      + '<td><button class="btn btn--sm" onclick="tlRestoreSnapshot(\'' + _escHTML(s.id) + '\')" '
        + 'title="Insert this snapshot as a NEW draft bundle (non-destructive).">Restore</button></td>'
      + '</tr>';
  }
  html += '</table>';
  return html;
}

function tlOnSelect(checkbox) {
  const id = checkbox.dataset.tlId;
  if (checkbox.checked) {
    if (_tlSelected.length >= 2) {
      const ev = _tlSelected.shift();
      const old = document.querySelector(
        '#tl-body input[data-tl-id="' + ev + '"]'
      );
      if (old) old.checked = false;
    }
    _tlSelected.push(id);
  } else {
    _tlSelected = _tlSelected.filter(x => x !== id);
  }
  if (_tlSelected.length === 2) {
    tlShowCompare(_tlSelected[0], _tlSelected[1]);
  } else {
    document.getElementById('tl-compare').classList.add('u-hidden');
  }
}

async function tlShowCompare(a, b) {
  const pane = document.getElementById('tl-compare');
  pane.classList.remove('u-hidden');
  pane.innerHTML = '<p class="u-muted">Loading diff…</p>';
  try {
    const r = await fetch('/api/diff/' + encodeURIComponent(a) + '/' + encodeURIComponent(b));
    if (!r.ok) {
      pane.innerHTML = '<p class="u-muted">Diff endpoint requires both refs to be drafts. '
        + 'Use `sciknow book diff &lt;a&gt; &lt;b&gt;` from the CLI for snapshot-vs-snapshot diffs.</p>';
      return;
    }
    const data = await r.json();
    pane.innerHTML = '<h4>Compare</h4>'
      + '<div style="line-height:1.5;font-family:var(--font-serif);">'
      + (data.diff_html || '<em class="u-muted">No textual differences.</em>')
      + '</div>';
  } catch (exc) {
    pane.innerHTML = '<p class="u-muted">Diff failed: ' + _escHTML(String(exc)) + '</p>';
  }
}

async function tlActivateDraft(id) {
  if (!id) return;
  const r = await fetch('/api/draft/' + id + '/activate', {method: 'POST'});
  if (!r.ok) {
    alert('Activate failed (' + r.status + ')');
    return;
  }
  await tlRender();
  if (currentDraftId) loadSection(currentDraftId);
}

async function tlRestoreSnapshot(id) {
  if (!id) return;
  if (!confirm('Restore this snapshot? Bundles insert NEW draft versions; section snapshots overwrite the active draft.')) return;
  // Try the bundle restore endpoint (works for chapter + book scope).
  let r = await fetch('/api/snapshot/restore-bundle/' + id, {method: 'POST'});
  if (!r.ok && r.status === 400) {
    // Probably scope='draft' — fall back to overwriting active draft.
    const cr = await fetch('/api/snapshot-content/' + id);
    if (!cr.ok) { alert('Snapshot not found'); return; }
    const sd = await cr.json();
    if (currentDraftId) {
      const fd = new FormData();
      fd.append('content', sd.content || '');
      await fetch('/edit/' + currentDraftId, {method: 'POST', body: fd});
      loadSection(currentDraftId);
    }
  }
  if (r.ok) {
    await refreshAfterJob(null);
    await tlRender();
  }
}

function openTimelineModal(scope) {
  _tlSelected = [];
  openModal('timeline-modal');
  // Default scope: section if a draft is open, else chapter, else book.
  const initial = scope || (
    currentDraftId ? 'section' : (currentChapterId ? 'chapter' : 'book')
  );
  tlSwitchScope(initial);
}
