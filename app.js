import { PROBLEMS, SPRINTS, TRACK_META } from './data/problems.js';
import { mountVisualizer } from './components/visualizers.js';
import { mountLlmSim } from './modules/llm-sim.js';

const STORAGE_KEY = 'leetgpu_dual_track_v2';
const LEGACY_KEY = 'leetgpu_triton_1day_v1';

const DEFAULT_STATE = {
  module: 'leetgpu',      // 'leetgpu' | 'llm'
  view: 'roadmap',        // 'roadmap' | 'studio' | 'cheatsheet'
  track: 'triton',        // 'triton' | 'cuda'
  theme: 'light',         // 'light' | 'dark'（默认清爽浅色主题）
  activeProblemId: 1,
  activeCodeTab: 'triton', // 'triton' | 'cuda' | 'pytorch'
  sortMode: 'sprint',     // 'sprint' | 'official'
  solvedByTrack: { triton: [], cuda: [] },
  drillsByTrack: { triton: [], cuda: [] }
};

let state = loadState();
let selectedAnnotationLine = null;
let llmMounted = false;

// 冲刺番茄钟状态
const TIMER_MODES = [15, 25, 45];
let timerModeIdx = 1;
let timerSeconds = TIMER_MODES[timerModeIdx] * 60;
let timerHandle = null;

function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      return {
        ...DEFAULT_STATE,
        ...parsed,
        track: parsed.track === 'cuda' ? 'cuda' : 'triton',
        solvedByTrack: {
          triton: Array.isArray(parsed.solvedByTrack?.triton) ? parsed.solvedByTrack.triton : [],
          cuda: Array.isArray(parsed.solvedByTrack?.cuda) ? parsed.solvedByTrack.cuda : []
        },
        drillsByTrack: {
          triton: Array.isArray(parsed.drillsByTrack?.triton) ? parsed.drillsByTrack.triton : [],
          cuda: Array.isArray(parsed.drillsByTrack?.cuda) ? parsed.drillsByTrack.cuda : []
        }
      };
    }
    // 兼容迁移 v1 纯 Triton 进度
    const legacy = localStorage.getItem(LEGACY_KEY);
    if (legacy) {
      const old = JSON.parse(legacy);
      return {
        ...DEFAULT_STATE,
        activeProblemId: old.activeProblemId || 1,
        solvedByTrack: {
          triton: Array.isArray(old.solvedIds) ? old.solvedIds : [],
          cuda: []
        },
        drillsByTrack: {
          triton: Array.isArray(old.drillsPassedIds) ? old.drillsPassedIds : [],
          cuda: []
        }
      };
    }
  } catch {
    // ignore storage errors
  }
  return structuredClone(DEFAULT_STATE);
}

function saveState() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // ignore storage quota errors
  }
}

function currentSolved() {
  return state.solvedByTrack[state.track];
}

function currentDrills() {
  return state.drillsByTrack[state.track];
}

function getOrderedProblems() {
  const copy = [...PROBLEMS];
  if (state.sortMode === 'sprint') {
    return copy.sort((a, b) => a.sprintOrder - b.sprintOrder);
  }
  return copy.sort((a, b) => a.id - b.id);
}

function getProblemById(id) {
  return PROBLEMS.find((p) => p.id === id) || PROBLEMS[0];
}

/* ---------- 轻量级安全语法高亮 ---------- */
function escapeHtml(str) {
  return str
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}

function highlightLine(rawLine, lang) {
  let comment = '';
  let code = rawLine;

  const commentMarker = lang === 'cuda' ? '//' : '#';
  const cIdx = rawLine.indexOf(commentMarker);
  if (cIdx !== -1) {
    code = rawLine.slice(0, cIdx);
    comment = rawLine.slice(cIdx);
  }

  let html = escapeHtml(code);
  html = html.replace(/(&quot;[^&]*&quot;|"[^"]*"|'[^']*')/g, '<span class="tok-str">$1</span>');

  if (lang === 'cuda') {
    html = html.replace(/(#include|#define|#pragma)/g, '<span class="tok-dec">$1</span>');
    html = html.replace(/\b(__global__|__device__|__shared__|__forceinline__|__restrict__|extern|void|int|size_t|float|float4|uint4|uint32_t|unsigned|char|const|for|if|return|static_cast|reinterpret_cast|dim3)\b/g, '<span class="tok-kw">$1</span>');
    html = html.replace(/\b(__syncthreads|__ldg|fmaxf|fmaf|make_float4|max)\b/g, '<span class="tok-fn">$1</span>');
  } else {
    html = html.replace(/(@triton\.jit)/g, '<span class="tok-dec">$1</span>');
    html = html.replace(/\b(import|from|as|def|for|in|range|if|return|and|or|not)\b/g, '<span class="tok-kw">$1</span>');
    html = html.replace(/\b(tl\.(?:program_id|arange|load|store|maximum|where|trans|dot|zeros|full|static_range|pointer_type|constexpr|float32|uint32)|triton\.cdiv|torch\.(?:Tensor|add|relu|matmul|flip|full_like|int64))\b/g, '<span class="tok-fn">$1</span>');
  }

  html = html.replace(/(^|[\s(,=+\-*\/^&<>:])(0x[0-9a-fA-F]+u?|\d+(?:\.\d+)?[fu]?)\b/g, '$1<span class="tok-num">$2</span>');

  if (comment) {
    html += `<span class="tok-com">${escapeHtml(comment)}</span>`;
  }
  return html || ' ';
}

/* ---------- Toast ---------- */
let toastTimer = null;
function showToast(msg) {
  const el = document.getElementById('toast');
  if (!el) return;
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 2200);
}

/* ---------- 平台一级模块切换（LeetGPU ⇄ LLM 4D 推演台） ---------- */
function updateLlmHeaderMetrics(metrics) {
  const pEl = document.getElementById('llm-hdr-params');
  const cEl = document.getElementById('llm-hdr-compute');
  const vEl = document.getElementById('llm-hdr-vram');
  const pill = document.getElementById('llm-hdr-vram-pill');
  if (pEl) pEl.textContent = metrics.paramsText;
  if (cEl) cEl.textContent = metrics.computeText;
  if (vEl) vEl.textContent = metrics.vramText;
  if (pill) {
    pill.classList.toggle('is-oom', Boolean(metrics.isOOM));
    pill.classList.toggle('is-ok', !metrics.isOOM);
  }
}

function setModule(modName, { syncHash = true, silent = false } = {}) {
  const next = modName === 'llm' ? 'llm' : 'leetgpu';
  state.module = next;
  saveState();

  const shell = document.getElementById('app-shell');
  shell.dataset.module = next;

  document.querySelectorAll('[data-module-btn]').forEach((btn) => {
    const active = btn.dataset.moduleBtn === next;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-selected', String(active));
  });

  const wsLeet = document.getElementById('workspace-leetgpu');
  const wsLlm = document.getElementById('workspace-llm');
  wsLeet.hidden = next !== 'leetgpu';
  wsLlm.hidden = next !== 'llm';

  if (next === 'llm') {
    document.getElementById('brand-logo').textContent = '🧠';
    document.title = 'LLM 4D 显存推演台 · AI Infra 极简实战台 | SuperSimpleTutor';
    if (!llmMounted) {
      mountLlmSim(wsLlm, updateLlmHeaderMetrics);
      llmMounted = true;
    }
  } else {
    document.getElementById('brand-logo').textContent = TRACK_META[state.track].icon;
    document.title = 'LeetGPU 双轨速通 · AI Infra 极简实战台 | SuperSimpleTutor';
  }

  if (syncHash) {
    const targetHash = next === 'llm' ? '#llm' : '#leetgpu';
    if (window.location.hash !== targetHash) {
      history.replaceState(null, '', targetHash);
    }
  }

  window.scrollTo({ top: 0, behavior: 'instant' });
  if (!silent) {
    showToast(next === 'llm' ? '已切换至 🧠 LLM 显存与 4D 并行推演台' : '已切换至 ⚡ LeetGPU 一天速通');
  }
}

/* ---------- 主题切换（默认浅色） ---------- */
function applyTheme(themeName, silent = false) {
  const next = themeName === 'dark' ? 'dark' : 'light';
  state.theme = next;
  document.documentElement.dataset.theme = next;
  const btn = document.getElementById('theme-toggle-btn');
  if (btn) {
    btn.textContent = next === 'light' ? '🌙' : '☀️';
    btn.title = next === 'light' ? '切换到深色主题' : '切换到浅色主题';
  }
  saveState();
  if (!silent) {
    showToast(next === 'light' ? '已切换至 ☀️ 浅色主题' : '已切换至 🌙 深色主题');
  }
}

/* ---------- 主轨切换（Triton ⇄ CUDA） ---------- */
function setTrack(trackName, silent = false) {
  if (trackName !== 'triton' && trackName !== 'cuda') return;
  state.track = trackName;
  state.activeCodeTab = trackName;
  selectedAnnotationLine = null;
  saveState();

  const meta = TRACK_META[trackName];
  document.getElementById('app-shell').dataset.track = trackName;
  if (state.module === 'leetgpu') {
    document.getElementById('brand-logo').textContent = meta.icon;
  }
  document.querySelectorAll('[data-track-btn]').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.trackBtn === trackName);
  });

  renderStats();
  renderSidebar();
  if (state.view === 'roadmap') renderRoadmapView();
  if (state.view === 'studio') renderStudioView();
  if (state.view === 'cheatsheet') renderCheatsheetView();

  if (!silent) {
    showToast(`已切换至 ${meta.icon} ${meta.name}`);
  }
}

/* ---------- 顶部统计与侧边栏 ---------- */
function renderStats() {
  const meta = TRACK_META[state.track];
  const solved = currentSolved();
  const solvedCount = solved.length;
  const totalPoints = PROBLEMS.reduce((sum, p) => sum + p.points, 0);
  const earnedPoints = PROBLEMS
    .filter((p) => solved.includes(p.id))
    .reduce((sum, p) => sum + p.points, 0);

  const timeFactor = state.track === 'cuda' ? 1.25 : 1.0;
  const remMinutes = Math.round(
    PROBLEMS
      .filter((p) => !solved.includes(p.id))
      .reduce((sum, p) => sum + p.estMinutes * timeFactor, 0)
  );

  document.getElementById('stat-track-label').textContent = meta.shortName;
  document.getElementById('solved-count').textContent = `${solvedCount}/10`;
  document.getElementById('points-count').textContent = `${earnedPoints} / ${totalPoints}`;
  document.getElementById('sidebar-track-heading').textContent = `${meta.shortName} 10 题路线`;

  const pct = Math.round((solvedCount / PROBLEMS.length) * 100);
  const fill = document.getElementById('sidebar-progress-fill');
  const bar = document.getElementById('sidebar-progressbar');
  fill.style.width = `${pct}%`;
  bar.setAttribute('aria-valuenow', String(solvedCount));

  const hrs = Math.floor(remMinutes / 60);
  const mins = remMinutes % 60;
  document.getElementById('time-remaining').textContent =
    remMinutes === 0 ? `🎉 ${meta.shortName} 轨已全通关！` : `剩余预计约 ${hrs > 0 ? `${hrs}h ` : ''}${mins}m`;

  const dualEl = document.getElementById('dual-progress-mini');
  if (dualEl) {
    dualEl.innerHTML = `
      <span>⚡ Triton: <strong>${state.solvedByTrack.triton.length}/10</strong></span>
      <span>🟢 CUDA: <strong>${state.solvedByTrack.cuda.length}/10</strong></span>
    `;
  }
}

function renderSidebar() {
  const listEl = document.getElementById('problem-list');
  const ordered = getOrderedProblems();
  const solved = currentSolved();
  const isCuda = state.track === 'cuda';
  const trackBadgeCls = isCuda ? 'badge-cuda' : 'badge-triton';
  const trackTag = isCuda ? 'CUDA' : 'Triton';

  document.getElementById('sort-toggle-btn').textContent =
    state.sortMode === 'sprint' ? '冲刺顺序 ⇅' : '官方 #1-#10 ⇅';

  let html = '';
  let lastSprint = null;

  ordered.forEach((p) => {
    if (state.sortMode === 'sprint' && p.sprint !== lastSprint) {
      lastSprint = p.sprint;
      const sp = SPRINTS.find((s) => s.id === p.sprint);
      html += `<div class="sprint-group-label"><span>Sprint ${sp.id}</span><span>${sp.problemIds.length} 题</span></div>`;
    }
    const isSolved = solved.includes(p.id);
    const isActive = state.view === 'studio' && state.activeProblemId === p.id;
    const est = isCuda ? Math.round(p.estMinutes * 1.25) : p.estMinutes;
    html += `
      <button class="problem-item ${isActive ? 'active' : ''} ${isSolved ? 'solved' : ''}" data-pid="${p.id}" role="listitem">
        <span class="problem-check">${isSolved ? '✓' : `#${p.id}`}</span>
        <span class="problem-item-body">
          <div class="problem-item-title">${p.title}</div>
          <div class="problem-item-sub">
            <span class="badge ${trackBadgeCls}" style="padding:1px 6px;font-size:10px">${trackTag}</span>
            <span>${est}m</span>
            <span>·</span>
            <span>${p.points} 分</span>
          </div>
        </span>
      </button>`;
  });

  listEl.innerHTML = html;
  listEl.querySelectorAll('.problem-item').forEach((btn) => {
    btn.addEventListener('click', () => {
      openProblem(Number(btn.dataset.pid));
      closeMobileSidebar();
    });
  });
}

/* ---------- 视图切换 ---------- */
function setView(viewName) {
  state.view = viewName;
  saveState();

  document.querySelectorAll('.nav-tab').forEach((tab) => {
    const active = tab.dataset.view === viewName;
    tab.classList.toggle('active', active);
    tab.setAttribute('aria-selected', String(active));
  });

  document.querySelectorAll('.view-section').forEach((sec) => {
    sec.classList.toggle('active', sec.id === `view-${viewName}`);
  });

  if (viewName === 'roadmap') renderRoadmapView();
  if (viewName === 'studio') renderStudioView();
  if (viewName === 'cheatsheet') renderCheatsheetView();
  renderSidebar();
}

function openProblem(problemId) {
  state.activeProblemId = problemId;
  state.activeCodeTab = state.track;
  selectedAnnotationLine = null;
  setView('studio');
  document.getElementById('main-content').scrollTop = 0;
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* ---------- 视图 1：一天路线图（双轨联动） ---------- */
function renderRoadmapView() {
  const container = document.getElementById('view-roadmap');
  const meta = TRACK_META[state.track];
  const otherTrack = state.track === 'triton' ? 'cuda' : 'triton';
  const otherMeta = TRACK_META[otherTrack];
  const solved = currentSolved();
  const nextUnsolved = getOrderedProblems().find((p) => !solved.includes(p.id)) || PROBLEMS[0];
  const badgeCls = state.track === 'cuda' ? 'badge-cuda' : 'badge-triton';

  const primitivesHtml = meta.primitives.map((pr) => `
    <div class="card-subtle">
      <div class="badge ${badgeCls}" style="margin-bottom:8px">${pr.tag}</div>
      <div style="font-family:var(--font-mono);font-size:12.5px;color:var(--cyan);word-break:break-word">${escapeHtml(pr.code)}</div>
      <p style="margin:6px 0 0;font-size:13px;color:var(--text-secondary)">${pr.desc}</p>
    </div>`).join('');

  const sprintsHtml = SPRINTS.map((sp) => {
    const chips = sp.problemIds.map((pid) => {
      const p = getProblemById(pid);
      const isS = solved.includes(pid);
      const est = state.track === 'cuda' ? Math.round(p.estMinutes * 1.25) : p.estMinutes;
      return `<button class="sprint-chip ${isS ? 'solved' : ''}" data-open-pid="${pid}">
        <span>${isS ? '✓' : `#${p.id}`}</span>
        <span>${p.title}</span>
        <span style="font-size:11px;color:var(--text-muted)">${est}m</span>
      </button>`;
    }).join('');

    return `
      <div class="card sprint-card">
        <div class="sprint-header">
          <div>
            <h3 style="margin:0;font-size:16px">${sp.subtitle}</h3>
            <p style="margin:4px 0 0;font-size:13.5px;color:var(--text-secondary)">${meta.sprintGoals[sp.id]}</p>
          </div>
          <span class="sprint-time">${sp.name}</span>
        </div>
        <div class="sprint-chips">${chips}</div>
      </div>`;
  }).join('');

  const ironRulesHtml = meta.ironRules.map((r, i) => `${i + 1}) ${r}`).join('<br/>');

  container.innerHTML = `
    <div class="stack-lg">
      <div class="hero-banner">
        <div class="hero-eyebrow">${meta.eyebrow}</div>
        <h1 class="hero-title" id="roadmap-heading">${meta.heroTitle}</h1>
        <p class="hero-desc">${meta.heroDesc}</p>
        <div class="hero-actions">
          <button class="btn btn-primary" id="start-next-btn">
            🚀 ${solved.length === 0 ? `开始 ${meta.shortName} 第 1 题冲刺` : `继续下一题：#${nextUnsolved.id} ${nextUnsolved.title}`}
          </button>
          <button class="btn btn-cyan" id="switch-other-track-btn">
            ${otherMeta.icon} 切换到 ${otherMeta.name}（快捷键 T）
          </button>
          <button class="btn" id="goto-cheatsheet-btn">📖 查看双轨速查表</button>
        </div>
        <div class="hero-metrics">
          <div class="metric-box">
            <div class="metric-value accent-emerald">10 题</div>
            <div class="metric-label">覆盖 LeetGPU 官方前 10 题</div>
          </div>
          <div class="metric-box">
            <div class="metric-value accent-cyan">${meta.totalTimeLabel}</div>
            <div class="metric-label">${meta.shortName} 轨一天总实战时长</div>
          </div>
          <div class="metric-box">
            <div class="metric-value accent-purple">6 大原语</div>
            <div class="metric-label">吃透即可写通全部 10 题</div>
          </div>
          <div class="metric-box">
            <div class="metric-value accent-amber">双轨独立</div>
            <div class="metric-label">Triton ${state.solvedByTrack.triton.length}/10 · CUDA ${state.solvedByTrack.cuda.length}/10</div>
          </div>
        </div>
      </div>

      <!-- 双轨定位对比 -->
      <div class="grid-2">
        <div class="card lang-card ${state.track === 'triton' ? 'recommended' : ''}">
          <h3>
            <span>⚡ Triton 极速轨</span>
            <span class="badge badge-triton">一天速通首选</span>
          </h3>
          <p>最简单的真正 GPU Kernel 语言，纯 Python 块级抽象，用最少精力拿高分。</p>
          <ul class="lang-bullets">
            <li>10~30 行 Python 完成全部 10 题，无需写指针类型和同步栅栏</li>
            <li><code>tl.trans</code> / <code>tl.dot</code> 自动生成共享内存与 Tensor Core 指令</li>
            <li>适合想 1 天内把前 10 题全部刷绿并拿合理高分的新手</li>
          </ul>
        </div>
        <div class="card lang-card ${state.track === 'cuda' ? 'recommended' : ''}">
          <h3>
            <span>🟢 CUDA C++ 硬核轨</span>
            <span class="badge badge-cuda">底层提分进阶</span>
          </h3>
          <p>掌控真实 GPU 硬件细节，掌握工业界和面试最高频的 CUDA 优化套路。</p>
          <ul class="lang-bullets">
            <li>用 <code>float4 / uint4</code> 128-bit 向量化打满显存带宽</li>
            <li>手写 <code>__shared__ smem[32][33]</code>（零 Bank Conflict）与 16×16 Tiled GEMM</li>
            <li>适合学完 Triton 想再冲击更高百分位或备战 GPU/AI Infra 面试</li>
          </ul>
        </div>
      </div>

      <!-- 6 大核心原语 -->
      <div>
        <h2 class="section-heading">${meta.primitivesHeading}</h2>
        <div class="grid-3">${primitivesHtml}</div>
      </div>

      <!-- 一天 3 个 Sprint 日程 -->
      <div>
        <h2 class="section-heading">🗓️ 一天冲刺日程表（当前：${meta.name}）</h2>
        <div class="stack-md">${sprintsHtml}</div>
      </div>

      <!-- 避坑三铁律 -->
      <div class="callout callout-amber">
        <span class="callout-icon">⚠️</span>
        <div>
          <strong>${meta.ironRulesTitle}</strong>
          <p>${ironRulesHtml}</p>
        </div>
      </div>
    </div>
  `;

  container.querySelector('#start-next-btn').addEventListener('click', () => openProblem(nextUnsolved.id));
  container.querySelector('#switch-other-track-btn').addEventListener('click', () => setTrack(otherTrack));
  container.querySelector('#goto-cheatsheet-btn').addEventListener('click', () => setView('cheatsheet'));
  container.querySelectorAll('[data-open-pid]').forEach((btn) => {
    btn.addEventListener('click', () => openProblem(Number(btn.dataset.openPid)));
  });
}

/* ---------- 视图 2：10 题交互精讲工作室（双轨联动） ---------- */
function renderStudioView() {
  const container = document.getElementById('view-studio');
  const p = getProblemById(state.activeProblemId);
  const track = state.track;
  const meta = TRACK_META[track];
  const tInfo = p.trackInfo[track];
  const isCuda = track === 'cuda';

  const ordered = getOrderedProblems();
  const idxInOrder = ordered.findIndex((item) => item.id === p.id);
  const prevP = idxInOrder > 0 ? ordered[idxInOrder - 1] : null;
  const nextP = idxInOrder < ordered.length - 1 ? ordered[idxInOrder + 1] : null;

  const isSolved = currentSolved().includes(p.id);
  const isDrillPassed = currentDrills().includes(p.id);
  const lang = state.activeCodeTab;
  const codeObj = p.code[lang];
  const lines = codeObj.source.split('\n');
  const annByLine = new Map(codeObj.annotations.map((a) => [a.line, a]));

  if (selectedAnnotationLine === null && codeObj.annotations.length > 0) {
    selectedAnnotationLine = codeObj.annotations[0].line;
  }

  const codeLinesHtml = lines.map((lineText, idx) => {
    const lineNum = idx + 1;
    const hasAnn = annByLine.has(lineNum);
    const isSel = selectedAnnotationLine === lineNum;
    return `
      <div class="code-line ${hasAnn ? 'annotated' : ''} ${isSel ? 'selected' : ''}" data-line="${lineNum}">
        <span class="line-num">${lineNum}</span>
        <span class="line-text">${highlightLine(lineText, lang)}</span>
        ${hasAnn ? '<span class="line-dot" title="点击查看这一行的 GPU 原理解析"></span>' : ''}
      </div>`;
  }).join('');

  const annotationsHtml = codeObj.annotations.map((a) => `
    <button type="button" class="annotation-item ${selectedAnnotationLine === a.line ? 'selected' : ''}" data-ann-line="${a.line}">
      <div class="annotation-line-tag">第 ${a.line} 行 · ${a.label}</div>
      <p>${a.text}</p>
    </button>`).join('');

  // 构建当前主轨的填空速练 HTML
  let drillHtml = escapeHtml(tInfo.drill.template);
  tInfo.drill.blanks.forEach((b) => {
    const inputTag = `<input type="text" class="drill-input" data-blank-id="${b.id}" placeholder="${escapeHtml(b.placeholder)}" autocomplete="off" spellcheck="false" />`;
    drillHtml = drillHtml.replace(`{{${b.id}}}`, inputTag);
  });

  const tipsHtml = tInfo.pointsTips.map((t) => `<li>${t}</li>`).join('');
  const badgeCls = isCuda ? 'badge-cuda' : 'badge-triton';
  const primitivesHtml = tInfo.corePrimitives.map((pr) => `<span class="badge ${badgeCls}">${pr}</span>`).join('');
  const est = isCuda ? Math.round(p.estMinutes * 1.25) : p.estMinutes;

  // 把当前主轨的代码 Tab 排在第一位并标为「主线」
  const tabOrder = isCuda
    ? [
        { id: 'cuda', label: 'CUDA C++ 主线', isMain: true },
        { id: 'triton', label: 'Triton 对照', isMain: false },
        { id: 'pytorch', label: 'PyTorch 对照', isMain: false }
      ]
    : [
        { id: 'triton', label: 'Triton 主线', isMain: true },
        { id: 'cuda', label: 'CUDA C++ 对照', isMain: false },
        { id: 'pytorch', label: 'PyTorch 对照', isMain: false }
      ];

  const tabsHtml = tabOrder.map((t) => `
    <button class="code-tab ${lang === t.id ? 'active' : ''}" data-lang-tab="${t.id}" role="tab">
      ${t.label} ${t.isMain ? '<span class="rec-pill">当前轨</span>' : ''}
    </button>`).join('');

  container.innerHTML = `
    <div class="stack-lg">
      <!-- 题目头部 -->
      <div class="studio-header">
        <div class="studio-title-group">
          <div class="studio-badges">
            <span class="badge badge-easy">LeetGPU #${p.id} · ${p.difficulty}</span>
            <span class="badge ${badgeCls}">${meta.icon} ${meta.name}</span>
            <span class="badge badge-pts">${p.points} 积分</span>
            <span class="badge badge-pt">⏱ 预计 ${est} 分钟</span>
            <a href="https://leetgpu.com/challenges/${p.slug}" target="_blank" rel="noopener noreferrer" style="font-size:12.5px;font-weight:600">
              前往 LeetGPU 原题 ↗
            </a>
          </div>
          <h1 class="studio-title" id="studio-heading">${p.title} <span style="font-size:16px;font-weight:500;color:var(--text-secondary)">— ${p.zhSubtitle}</span></h1>
          <p class="studio-summary">${p.summary}</p>
          <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:2px">${primitivesHtml}</div>
        </div>
        <div class="studio-actions">
          <button id="toggle-solved-btn" class="btn ${isSolved ? 'btn-solved' : 'btn-primary'}">
            ${isSolved ? `✓ ${meta.shortName} 轨已通关` : `标记 ${meta.shortName} 通关`}
          </button>
        </div>
      </div>

      <!-- 主轨核心思路卡片 -->
      <div class="callout ${isCuda ? 'callout-emerald' : 'callout-purple'}">
        <span class="callout-icon">🧠</span>
        <div>
          <strong>${meta.shortName} 轨破题思路（为什么这样写又稳又快）</strong>
          <p>${tInfo.angle}</p>
        </div>
      </div>

      <!-- 代码精讲 + 逐行批注面板 -->
      <div class="code-studio">
        <div class="code-toolbar">
          <div class="code-lang-tabs" role="tablist">${tabsHtml}</div>
          <div style="display:flex;gap:8px;align-items:center">
            <span style="font-size:11.5px;color:var(--text-muted)">点击带蓝点代码行查看批注</span>
            <button id="copy-code-btn" class="mini-btn primary-mini" style="flex:0 0 auto">📋 复制代码</button>
          </div>
        </div>
        <div class="code-body-split">
          <div class="code-pane" aria-label="代码块">${codeLinesHtml}</div>
          <div class="annotation-pane">
            <h4>逐行 GPU 原理解析（${lang.toUpperCase()}）</h4>
            <div class="annotation-list">${annotationsHtml}</div>
          </div>
        </div>
      </div>

      <!-- LeetGPU 避坑提醒 -->
      <div class="callout callout-amber">
        <span class="callout-icon">⚠️</span>
        <div>
          <strong>${meta.shortName} 轨实操避坑指南</strong>
          <p>${tInfo.gotcha.replaceAll('\n', '<br/>')}</p>
        </div>
      </div>

      <!-- 交互式 GPU 可视化实验室（随主轨切换视角） -->
      <div class="card viz-card" id="viz-mount"></div>

      <!-- 提分秘籍 + 60秒速练双栏 -->
      <div class="grid-2">
        <div class="card stack-md" style="gap:10px">
          <h3 style="margin:0;font-size:16px">🏆 ${meta.shortName} 轨怎么拿合理高分？</h3>
          <ul class="lang-bullets" style="margin-top:4px">${tipsHtml}</ul>
          <div class="card-subtle" style="margin-top:auto">
            <div style="font-size:11.5px;color:var(--text-muted);margin-bottom:4px">LeetGPU 精确入口签名（${meta.shortName}）</div>
            <code style="font-family:var(--font-mono);font-size:12px;color:var(--cyan);word-break:break-all">${escapeHtml(tInfo.signature)}</code>
          </div>
        </div>

        <div class="card drill-card">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:8px">
            <h3 style="margin:0;font-size:16px">⚡ ${meta.shortName} 60 秒填空速练</h3>
            ${isDrillPassed ? '<span class="badge badge-easy">✓ 速练已通过</span>' : '<span class="badge badge-pts">动手防忘</span>'}
          </div>
          <p class="drill-prompt">${tInfo.drill.prompt}</p>
          <div class="drill-code-box">${drillHtml}</div>
          <div class="drill-actions">
            <button id="drill-check-btn" class="btn btn-primary" style="padding:7px 14px;font-size:13px">验证答案</button>
            <button id="drill-hint-btn" class="btn" style="padding:7px 12px;font-size:13px">看提示</button>
            <button id="drill-fill-btn" class="text-btn" style="margin-left:auto">显示标准答案</button>
          </div>
          <div id="drill-feedback" class="drill-feedback"></div>
        </div>
      </div>

      <!-- 底部上下题导航 -->
      <div class="studio-footer">
        <div>
          ${prevP ? `<button class="btn" id="prev-problem-btn">← 上一题：#${prevP.id} ${prevP.title}</button>` : '<span></span>'}
        </div>
        <div style="font-size:12px;color:var(--text-muted)">
          快捷键：<span class="kbd">T</span> 切双轨 · <span class="kbd">←</span>/<span class="kbd">→</span> 切题 · <span class="kbd">1</span><span class="kbd">2</span><span class="kbd">3</span> 切代码页
        </div>
        <div>
          ${nextP
            ? `<button class="btn btn-primary" id="next-problem-btn">下一题：#${nextP.id} ${nextP.title} →</button>`
            : `<button class="btn btn-primary" id="finish-all-btn">🎉 查看双轨速查表 →</button>`}
        </div>
      </div>
    </div>
  `;

  // 挂载可视化实验室（传入当前主轨）
  mountVisualizer(container.querySelector('#viz-mount'), p, track);

  // 绑定事件
  container.querySelector('#toggle-solved-btn').addEventListener('click', () => toggleSolved(p.id));

  container.querySelectorAll('[data-lang-tab]').forEach((tab) => {
    tab.addEventListener('click', () => {
      state.activeCodeTab = tab.dataset.langTab;
      selectedAnnotationLine = null;
      saveState();
      renderStudioView();
    });
  });

  container.querySelector('#copy-code-btn').addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(codeObj.source);
      showToast(`已复制 #${p.id} ${p.title} (${lang.toUpperCase()}) 代码！`);
    } catch {
      showToast('复制失败，请手动选中代码复制');
    }
  });

  const selectLine = (ln) => {
    if (!annByLine.has(ln)) return;
    selectedAnnotationLine = ln;
    container.querySelectorAll('.code-line').forEach((el) => {
      el.classList.toggle('selected', Number(el.dataset.line) === ln);
    });
    container.querySelectorAll('.annotation-item').forEach((el) => {
      el.classList.toggle('selected', Number(el.dataset.annLine) === ln);
    });
  };

  container.querySelectorAll('.code-line.annotated').forEach((el) => {
    el.addEventListener('click', () => selectLine(Number(el.dataset.line)));
  });
  container.querySelectorAll('.annotation-item').forEach((el) => {
    el.addEventListener('click', () => selectLine(Number(el.dataset.annLine)));
  });

  // 填空速练逻辑
  const fbEl = container.querySelector('#drill-feedback');
  const checkDrill = () => {
    const inputs = [...container.querySelectorAll('.drill-input')];
    let allCorrect = true;
    inputs.forEach((inp) => {
      const b = tInfo.drill.blanks.find((item) => item.id === inp.dataset.blankId);
      const val = inp.value.trim().replace(/\s+/g, ' ');
      const ok = b.answers.some((ans) => ans.replace(/\s+/g, ' ').toLowerCase() === val.toLowerCase());
      inp.style.borderColor = ok ? 'var(--emerald)' : 'var(--rose)';
      if (!ok) allCorrect = false;
    });

    fbEl.classList.add('show');
    if (allCorrect) {
      fbEl.className = 'drill-feedback show pass';
      fbEl.innerHTML = `✅ 全部答对！你已经掌握本题 ${meta.shortName} 核心写法，可以直接去 LeetGPU 提交拿分！`;
      if (!currentDrills().includes(p.id)) {
        currentDrills().push(p.id);
      }
      if (!currentSolved().includes(p.id)) {
        currentSolved().push(p.id);
        showToast(`🎉 ${meta.shortName} 速练通关！解锁 #${p.id} +${p.points} 分`);
      }
      saveState();
      renderStats();
      renderSidebar();
    } else {
      fbEl.className = 'drill-feedback show fail';
      fbEl.innerHTML = '❌ 还有空没填对（红框处），可以点「看提示」或对照上方代码再试一次！';
    }
  };

  container.querySelector('#drill-check-btn').addEventListener('click', checkDrill);
  container.querySelectorAll('.drill-input').forEach((inp) => {
    inp.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') checkDrill();
    });
  });
  container.querySelector('#drill-hint-btn').addEventListener('click', () => {
    fbEl.className = 'drill-feedback show hint';
    fbEl.textContent = `💡 提示：${tInfo.drill.hint}`;
  });
  container.querySelector('#drill-fill-btn').addEventListener('click', () => {
    container.querySelectorAll('.drill-input').forEach((inp) => {
      const b = tInfo.drill.blanks.find((item) => item.id === inp.dataset.blankId);
      inp.value = b.answers[0];
      inp.style.borderColor = 'var(--cyan)';
    });
    fbEl.className = 'drill-feedback show hint';
    fbEl.textContent = '已填入标准答案，建议默念一遍后点击「验证答案」。';
  });

  if (prevP) {
    container.querySelector('#prev-problem-btn').addEventListener('click', () => openProblem(prevP.id));
  }
  if (nextP) {
    container.querySelector('#next-problem-btn').addEventListener('click', () => openProblem(nextP.id));
  } else {
    container.querySelector('#finish-all-btn')?.addEventListener('click', () => setView('cheatsheet'));
  }
}

/* ---------- 视图 3：双轨速查表 ---------- */
function renderCheatsheetView() {
  const container = document.getElementById('view-cheatsheet');
  const meta = TRACK_META[state.track];

  container.innerHTML = `
    <div class="stack-lg">
      <div>
        <h1 class="studio-title" id="cheatsheet-heading">📖 Triton ⇄ CUDA 双轨速查表 &amp; 提分手册</h1>
        <p class="studio-summary" style="margin-top:6px">把这份罗塞塔石碑开在 LeetGPU 旁边，无论是 Triton 速通还是 CUDA 底层提分，前 10 题零卡壳一口气刷完。</p>
      </div>

      <div class="card" style="padding:0;overflow:hidden">
        <div style="padding:16px 20px;border-bottom:1px solid var(--border-subtle)">
          <h2 style="margin:0;font-size:16px">🗺️ Triton ↔ CUDA C++ ↔ PyTorch 语法对照（罗塞塔石碑）</h2>
        </div>
        <div style="overflow-x:auto">
          <table class="data-table">
            <thead>
              <tr>
                <th>GPU 编程概念</th>
                <th>⚡ Triton 写法</th>
                <th>🟢 CUDA C++ 写法</th>
                <th>PyTorch 对照</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>声明 GPU 内核</td>
                <td><code>@triton.jit</code></td>
                <td><code>__global__ void fn(...)</code></td>
                <td>内置算子</td>
              </tr>
              <tr>
                <td>LeetGPU 入口签名</td>
                <td><code>def solve(a, b, c, N):</code></td>
                <td><code>extern "C" void solve(...)</code></td>
                <td><code>def solve(...):</code></td>
              </tr>
              <tr>
                <td>获取块 / 全局索引</td>
                <td><code>pid = tl.program_id(0)</code><br/><code>offs = pid * BS + tl.arange(0, BS)</code></td>
                <td><code>int idx = blockIdx.x * blockDim.x + threadIdx.x;</code></td>
                <td>自动向量化</td>
              </tr>
              <tr>
                <td>128-bit 向量化访存</td>
                <td>自动生成（<code>BLOCK_SIZE=1024</code>）</td>
                <td><code>reinterpret_cast&lt;const float4*&gt;(A)[idx]</code></td>
                <td>底层 ATen 处理</td>
              </tr>
              <tr>
                <td>越界保护读写</td>
                <td><code>tl.load(ptr+offs, mask=offs&lt;N, other=0.0)</code></td>
                <td><code>if (idx &lt; N) val = ptr[idx];</code></td>
                <td>框架管理边界</td>
              </tr>
              <tr>
                <td>无分支条件选择</td>
                <td><code>tl.where(x &gt; 0, x, 0.01 * x)</code></td>
                <td><code>x &gt; 0.0f ? x : 0.01f * x</code> / <code>fmaxf(0.0f, x)</code></td>
                <td><code>torch.where(...)</code></td>
              </tr>
              <tr>
                <td>片上共享内存转置</td>
                <td><code>tl.trans(tile)</code>（自动零冲突）</td>
                <td><code>__shared__ float s[32][33];</code> + <code>__syncthreads();</code></td>
                <td><code>x.T.contiguous()</code></td>
              </tr>
              <tr>
                <td>矩阵块乘累加</td>
                <td><code>acc += tl.dot(a, b, input_precision="ieee")</code></td>
                <td><code>As[16][16], Bs[16][16]</code> + <code>fmaf</code></td>
                <td><code>torch.matmul(a, b, out=c)</code></td>
              </tr>
              <tr>
                <td>只读缓存 &amp; 循环展开</td>
                <td><code>tl.static_range(0, 32, 8)</code></td>
                <td><code>__ldg(&amp;ptr[i])</code> + <code>#pragma unroll</code></td>
                <td>不可手动控制</td>
              </tr>
              <tr>
                <td>向上取整启动网格</td>
                <td><code>grid = (triton.cdiv(N, BS),)</code></td>
                <td><code>int blocks = (N + threads - 1) / threads;</code></td>
                <td>自动调度</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div class="grid-2">
        <div class="card stack-md" style="gap:10px">
          <h3 style="margin:0;font-size:16px">${meta.troubleshootTitle}</h3>
          <ul class="lang-bullets">${meta.troubleshoot.map((item) => `<li>${item}</li>`).join('')}</ul>
        </div>

        <div class="card stack-md" style="gap:10px">
          <h3 style="margin:0;font-size:16px">${meta.leversTitle}</h3>
          <ul class="lang-bullets">${meta.levers.map((item) => `<li>${item}</li>`).join('')}</ul>
        </div>
      </div>
    </div>
  `;
}

/* ---------- 进度与交互操作 ---------- */
function toggleSolved(problemId) {
  const solved = currentSolved();
  const meta = TRACK_META[state.track];
  const idx = solved.indexOf(problemId);
  if (idx === -1) {
    solved.push(problemId);
    const p = getProblemById(problemId);
    showToast(`✓ 已标记 ${meta.shortName} 轨 #${p.id} ${p.title} 通关 (+${p.points} 分)`);
  } else {
    solved.splice(idx, 1);
    showToast(`已取消 ${meta.shortName} 轨该题完成标记`);
  }
  saveState();
  renderStats();
  renderSidebar();
  if (state.view === 'studio') renderStudioView();
  if (state.view === 'roadmap') renderRoadmapView();
}

function closeMobileSidebar() {
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('mobile-menu-btn').setAttribute('aria-expanded', 'false');
}

/* ---------- 冲刺番茄钟 ---------- */
function renderTimer() {
  const m = String(Math.floor(timerSeconds / 60)).padStart(2, '0');
  const s = String(timerSeconds % 60).padStart(2, '0');
  document.getElementById('timer-display').textContent = `${m}:${s}`;
  document.getElementById('timer-toggle').textContent = timerHandle ? '暂停' : '开始';
  document.getElementById('timer-mode').textContent = `${TIMER_MODES[timerModeIdx]}m`;
}

function setupTimer() {
  document.getElementById('timer-toggle').addEventListener('click', () => {
    if (timerHandle) {
      clearInterval(timerHandle);
      timerHandle = null;
    } else {
      timerHandle = setInterval(() => {
        if (timerSeconds > 0) {
          timerSeconds--;
          renderTimer();
        } else {
          clearInterval(timerHandle);
          timerHandle = null;
          renderTimer();
          showToast('⏰ 本轮冲刺计时结束！喝口水休息 5 分钟再战！');
        }
      }, 1000);
    }
    renderTimer();
  });

  document.getElementById('timer-reset').addEventListener('click', () => {
    clearInterval(timerHandle);
    timerHandle = null;
    timerSeconds = TIMER_MODES[timerModeIdx] * 60;
    renderTimer();
  });

  document.getElementById('timer-mode').addEventListener('click', () => {
    clearInterval(timerHandle);
    timerHandle = null;
    timerModeIdx = (timerModeIdx + 1) % TIMER_MODES.length;
    timerSeconds = TIMER_MODES[timerModeIdx] * 60;
    renderTimer();
  });

  renderTimer();
}

/* ---------- 全局绑定与初始化 ---------- */
function init() {
  document.querySelectorAll('.nav-tab').forEach((tab) => {
    tab.addEventListener('click', () => setView(tab.dataset.view));
  });

  const brandHome = document.getElementById('brand-home');
  brandHome.addEventListener('click', () => setView('roadmap'));
  brandHome.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') setView('roadmap');
  });

  // 一级模块切换按钮
  document.querySelectorAll('[data-module-btn]').forEach((btn) => {
    btn.addEventListener('click', () => setModule(btn.dataset.moduleBtn));
  });

  // 监听 URL Hash 变化（支持 #llm / #leetgpu 直达）
  window.addEventListener('hashchange', () => {
    if (window.location.hash === '#llm') setModule('llm', { syncHash: false, silent: true });
    else if (window.location.hash === '#leetgpu') setModule('leetgpu', { syncHash: false, silent: true });
  });

  // 双轨主切换按钮
  document.querySelectorAll('[data-track-btn]').forEach((btn) => {
    btn.addEventListener('click', () => setTrack(btn.dataset.trackBtn));
  });

  // 主题切换按钮
  document.getElementById('theme-toggle-btn')?.addEventListener('click', () => {
    applyTheme(state.theme === 'light' ? 'dark' : 'light');
  });

  // 侧边栏排序切换
  document.getElementById('sort-toggle-btn').addEventListener('click', () => {
    state.sortMode = state.sortMode === 'sprint' ? 'official' : 'sprint';
    saveState();
    renderSidebar();
    if (state.view === 'studio') renderStudioView();
  });

  // 重置当前主轨进度
  document.getElementById('reset-progress-btn').addEventListener('click', () => {
    const meta = TRACK_META[state.track];
    state.solvedByTrack[state.track] = [];
    state.drillsByTrack[state.track] = [];
    saveState();
    renderStats();
    renderSidebar();
    if (state.view === 'roadmap') renderRoadmapView();
    if (state.view === 'studio') renderStudioView();
    showToast(`已重置 ${meta.shortName} 轨本地进度`);
  });

  // 移动端侧边栏抽屉
  document.getElementById('mobile-menu-btn').addEventListener('click', () => {
    const sb = document.getElementById('sidebar');
    const isOpen = sb.classList.toggle('open');
    document.getElementById('mobile-menu-btn').setAttribute('aria-expanded', String(isOpen));
  });

  // 键盘快捷键（非输入框焦点下，仅在 LeetGPU 模块生效）
  document.addEventListener('keydown', (e) => {
    if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) return;
    if (state.module !== 'leetgpu') return;
    if (e.key === 't' || e.key === 'T') {
      setTrack(state.track === 'triton' ? 'cuda' : 'triton');
      return;
    }
    if (state.view !== 'studio') return;
    const ordered = getOrderedProblems();
    const idx = ordered.findIndex((p) => p.id === state.activeProblemId);
    if (e.key === 'ArrowRight' && idx < ordered.length - 1) {
      openProblem(ordered[idx + 1].id);
    } else if (e.key === 'ArrowLeft' && idx > 0) {
      openProblem(ordered[idx - 1].id);
    } else if (e.key === '1') {
      state.activeCodeTab = state.track;
      selectedAnnotationLine = null;
      renderStudioView();
    } else if (e.key === '2') {
      state.activeCodeTab = state.track === 'triton' ? 'cuda' : 'triton';
      selectedAnnotationLine = null;
      renderStudioView();
    } else if (e.key === '3') {
      state.activeCodeTab = 'pytorch';
      selectedAnnotationLine = null;
      renderStudioView();
    }
  });

  setupTimer();
  applyTheme(state.theme || 'light', true);
  setTrack(state.track, true);
  setView(state.view);

  // 根据 URL Hash 或上次记忆决定初始模块
  const hash = window.location.hash;
  const initialModule = hash === '#llm' ? 'llm' : hash === '#leetgpu' ? 'leetgpu' : (state.module || 'leetgpu');
  setModule(initialModule, { syncHash: Boolean(hash), silent: true });
}

init();
