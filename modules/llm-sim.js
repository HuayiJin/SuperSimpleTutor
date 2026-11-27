// LLM 分布式训练显存与 4D 并行推演台 — 独立 ES Module
// 完整保留原版精确瓶颈 Stage 建模、TP/SP/PP/CP(Ring/Ulysses)/ZeRO/重计算与 DualPipe 推演引擎

const MODEL_PRESETS = [
  {
    id: 'toy',
    name: '极简 Transformer (0.5B)',
    tag: '单卡入门验证',
    desc: '适合手算核对 kP 与 B·S·H 比例',
    H: 1024, L: 24, H_ff: 2816, n_h: 16, n_kv: 8, V: 32000, tieEmbeddings: true,
    defaultGpus: 1, defaultVram: 24, defaultMbs: 2, defaultS: 2048, defaultGas: 8,
    defaultTp: 1, defaultPp: 1, defaultCp: 1, defaultZero: 0
  },
  {
    id: 'qwen3_8b',
    name: 'Qwen3-8B Dense 风格',
    tag: 'GQA + SwiGLU',
    desc: '典型 8B 稠密架构，观察单卡 OOM 到多维并行破局',
    H: 4096, L: 36, H_ff: 12288, n_h: 32, n_kv: 8, V: 151936, tieEmbeddings: false,
    defaultGpus: 4, defaultVram: 80, defaultMbs: 1, defaultS: 4096, defaultGas: 16,
    defaultTp: 1, defaultPp: 1, defaultCp: 1, defaultZero: 1
  },
  {
    id: 'qwen3_32b_long',
    name: 'Qwen3-32B 长文本 64K',
    tag: 'TP + CP + 重计算',
    desc: '64K 超长序列激活爆炸，体验 CP 切分 S 与重计算组合',
    H: 5120, L: 64, H_ff: 25600, n_h: 40, n_kv: 8, V: 151936, tieEmbeddings: false,
    defaultGpus: 32, defaultVram: 80, defaultMbs: 1, defaultS: 65536, defaultGas: 8,
    defaultTp: 4, defaultPp: 1, defaultCp: 4, defaultZero: 1
  },
  {
    id: 'scale_70b',
    name: '70B 级超大规模',
    tag: '4D 并行 + 调度博弈',
    desc: '64卡集群推演 TP + SP + PP + CP + DualPipe 延后 dW',
    H: 8192, L: 80, H_ff: 28672, n_h: 64, n_kv: 8, V: 151936, tieEmbeddings: false,
    defaultGpus: 64, defaultVram: 80, defaultMbs: 1, defaultS: 8192, defaultGas: 16,
    defaultTp: 8, defaultPp: 4, defaultCp: 2, defaultZero: 1
  }
];

const VRAM_OPTIONS = [
  { label: '24 GiB (RTX 4090)', value: 24 },
  { label: '40 GiB (A100-40G)', value: 40 },
  { label: '80 GiB (H100 / A100)', value: 80 },
  { label: '141 GiB (H200)', value: 141 }
];

const NH_OPTIONS = [8, 16, 32, 40, 64];
const NKV_OPTIONS = [1, 2, 4, 8, 16, 32, 64];
const V_OPTIONS = [32000, 65536, 128256, 151936];
const S_OPTIONS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];
const GPU_OPTIONS = [1, 2, 4, 8, 16, 32, 64, 128, 256];

const ICONS = {
  alert: '<svg class="icon icon-md" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>',
  alertXs: '<svg class="icon icon-xs" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>',
  checkXs: '<svg class="icon icon-xs" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>',
  sparklesXs: '<svg class="icon icon-xs" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/><path d="M5 3v4"/><path d="M19 17v4"/><path d="M3 5h4"/><path d="M17 19h4"/></svg>',
  scissorsLg: '<svg class="icon icon-lg tc-cyan-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="6" r="3"/><path d="M8.12 8.12 12 12"/><path d="M20 4 8.12 15.88"/><circle cx="6" cy="18" r="3"/><path d="M14.8 14.8 20 20"/></svg>',
  refreshLg: '<svg class="icon icon-lg tc-emerald-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/></svg>',
  gaugeLg: '<svg class="icon icon-lg tc-amber-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/></svg>',
  activityXs: '<svg class="icon icon-xs" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/></svg>',
  calcXs: '<svg class="icon icon-xs" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="16" height="20" x="4" y="2" rx="2"/><line x1="8" x2="16" y1="6" y2="6"/><line x1="16" x2="16" y1="14" y2="18"/><path d="M16 10h.01"/><path d="M12 10h.01"/><path d="M8 10h.01"/><path d="M12 14h.01"/><path d="M8 14h.01"/><path d="M12 18h.01"/><path d="M8 18h.01"/></svg>',
  layersXs: '<svg class="icon icon-xs" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/></svg>',
  infoMd: '<svg class="icon icon-md" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>'
};

function formatGiB(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0.0 MiB';
  const gib = bytes / (1024 ** 3);
  if (gib >= 1) return `${gib.toFixed(2)} GiB`;
  const mib = bytes / (1024 ** 2);
  return `${mib.toFixed(1)} MiB`;
}

function formatParams(count) {
  if (count >= 1e9) return `${(count / 1e9).toFixed(2)} B`;
  if (count >= 1e6) return `${(count / 1e6).toFixed(1)} M`;
  return Math.round(count).toLocaleString();
}

export function mountLlmSim(container, onMetricsUpdate) {
  if (!container) return;

  const state = {
    presetId: 'qwen3_8b',
    H: 4096,
    L: 36,
    H_ff: 12288,
    n_h: 32,
    n_kv: 8,
    V: 151936,
    tieEmbeddings: false,
    MBS: 1,
    S: 4096,
    GAS: 16,
    numGpus: 4,
    vramCap: 80,
    fp32GradBuffer: true,
    includeSysOverhead: true,
    TP: 1,
    SP: false,
    PP: 1,
    CP: 1,
    cpMode: 'ring',
    zeroStage: 1,
    zeroAcrossCp: true,
    flashAttn: true,
    chunkedCe: false,
    ckptMode: 'none',
    selectiveFlags: {
      coreAttn: true,
      norm: true,
      swiglu: true,
      mlpFull: false,
      attnProj: false
    },
    recompLayers: 36,
    recompBlockSize: 1,
    scheduleMode: '1f1b',
    delayedDwWindow: 2,
    activeTab: 'lifecycle'
  };

  container.innerHTML = `
    <div class="llm-sim">
      <div class="llm-intro-bar">
        <div class="llm-intro-left">
          <div class="llm-intro-icon">
            <svg class="icon icon-lg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/></svg>
          </div>
          <div>
            <h1 class="llm-intro-title">LLM 分布式训练显存与 4D 并行推演台</h1>
            <p class="llm-intro-sub">精确瓶颈 Stage 建模 · 上下文并行 (Ring / Ulysses) · 细粒度与分层重计算算力权衡</p>
          </div>
        </div>
        <span class="llm-pill">TP · SP · PP · CP · ZeRO · 重计算 · DualPipe</span>
      </div>

      <section id="llm-presets-bar" class="presets-grid"></section>

      <div class="workspace-grid">
        <div class="col-stack">
          <!-- 卡片 1：模型架构参数 -->
          <div class="llm-card">
            <div class="card-header">
              <div class="card-title-group">
                <svg class="icon icon-md tc-sky-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/></svg>
                <h2 class="card-title">1. 模型架构参数（GQA + SwiGLU）</h2>
              </div>
              <span id="llm-lbl-dhead" class="font-mono text-muted" style="font-size:12px;"></span>
            </div>

            <div class="llm-grid-2">
              <div>
                <div class="ctrl-label-row">
                  <span class="text-muted">隐藏维度 H</span>
                  <span id="llm-lbl-H" class="font-mono text-strong"></span>
                </div>
                <input id="llm-inp-H" type="range" min="512" max="8192" step="512" class="accent-sky" />
              </div>

              <div>
                <div class="ctrl-label-row">
                  <span class="text-muted">Transformer 层数 L</span>
                  <span id="llm-lbl-L" class="font-mono text-strong"></span>
                </div>
                <input id="llm-inp-L" type="range" min="1" max="96" step="1" class="accent-sky" />
              </div>

              <div>
                <div class="ctrl-label-row">
                  <span class="text-muted">MLP 维度 H_ff</span>
                  <span id="llm-lbl-Hff" class="font-mono text-strong"></span>
                </div>
                <input id="llm-inp-Hff" type="range" min="1024" max="32768" step="512" class="accent-sky" />
              </div>

              <div>
                <div class="ctrl-label-row">
                  <span class="text-muted">注意力头 n_h / n_kv</span>
                  <span id="llm-lbl-heads" class="font-mono text-strong"></span>
                </div>
                <div style="display:flex;gap:6px;">
                  <select id="llm-sel-nh" class="ctrl-select font-mono w-half"></select>
                  <select id="llm-sel-nkv" class="ctrl-select font-mono w-half"></select>
                </div>
              </div>
            </div>

            <div class="ctrl-divider">
              <div style="display:flex;align-items:center;gap:8px;">
                <span class="text-muted">词表 V:</span>
                <select id="llm-sel-V" class="ctrl-select font-mono"></select>
              </div>
              <label class="check-label">
                <input id="llm-chk-tie" type="checkbox" class="accent-sky" />
                <span>共享词表 (Tie Emb)</span>
              </label>
            </div>
          </div>

          <!-- 卡片 2：批次与集群硬件 -->
          <div class="llm-card">
            <div class="card-header">
              <div class="card-title-group">
                <svg class="icon icon-md tc-emerald-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" x2="4" y1="21" y2="14"/><line x1="4" x2="4" y1="10" y2="3"/><line x1="12" x2="12" y1="21" y2="12"/><line x1="12" x2="12" y1="8" y2="3"/><line x1="20" x2="20" y1="21" y2="16"/><line x1="20" x2="20" y1="12" y2="3"/><line x1="2" x2="6" y1="14" y2="14"/><line x1="10" x2="14" y1="8" y2="8"/><line x1="18" x2="22" y1="16" y2="16"/></svg>
                <h2 class="card-title">2. 批次、序列与硬件配置</h2>
              </div>
              <span id="llm-lbl-gbs" class="font-mono tc-emerald" style="font-size:12px;font-weight:500;"></span>
            </div>

            <div class="llm-grid-2">
              <div>
                <div class="ctrl-label-row">
                  <span class="text-muted">Micro-Batch (B)</span>
                  <span id="llm-lbl-mbs" class="font-mono tc-emerald" style="font-weight:600;"></span>
                </div>
                <input id="llm-inp-mbs" type="range" min="1" max="16" step="1" class="accent-emerald" />
              </div>

              <div>
                <div class="ctrl-label-row">
                  <span class="text-muted">总序列长度 S</span>
                  <span id="llm-lbl-S" class="font-mono tc-emerald" style="font-weight:600;"></span>
                </div>
                <select id="llm-sel-S" class="ctrl-select font-mono w-full"></select>
              </div>

              <div>
                <div class="ctrl-label-row">
                  <span class="text-muted">梯度累积 GAS</span>
                  <span id="llm-lbl-gas" class="font-mono text-strong"></span>
                </div>
                <input id="llm-inp-gas" type="range" min="1" max="64" step="1" class="accent-emerald" />
              </div>

              <div>
                <div class="ctrl-label-row">
                  <span class="text-muted">总 GPU 卡数</span>
                  <span id="llm-lbl-gpus" class="font-mono text-strong"></span>
                </div>
                <select id="llm-sel-gpus" class="ctrl-select font-mono w-full"></select>
              </div>
            </div>

            <div class="ctrl-divider-col">
              <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;">
                <span class="text-muted">单卡显存上限:</span>
                <select id="llm-sel-vram" class="ctrl-select font-mono"></select>
              </div>
              <div style="display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:8px;">
                <label class="check-label">
                  <input id="llm-chk-fp32grad" type="checkbox" class="accent-sky" />
                  <span>FP32 梯度缓冲 (4P)</span>
                </label>
                <label class="check-label">
                  <input id="llm-chk-sysoverhead" type="checkbox" class="accent-sky" />
                  <span>含 CUDA/碎片预留</span>
                </label>
              </div>
            </div>
          </div>

          <!-- 卡片 3：4D 并行拓扑 -->
          <div class="llm-card">
            <div class="card-header">
              <div class="card-title-group">
                <svg class="icon icon-md tc-violet-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="6" x2="6" y1="3" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/></svg>
                <h2 class="card-title">3. 4D 并行切分 (TP · PP · CP · ZeRO)</h2>
              </div>
              <span id="llm-lbl-topo" class="llm-badge badge-violet font-mono"></span>
            </div>

            <div class="llm-grid-3">
              <div>
                <div class="text-sub" style="font-size:12px;font-weight:500;margin-bottom:6px;">张量并行 TP</div>
                <div id="llm-grp-tp" class="llm-grid-2 gap-xs"></div>
              </div>
              <div>
                <div class="text-sub" style="font-size:12px;font-weight:500;margin-bottom:6px;">流水线并行 PP</div>
                <div id="llm-grp-pp" class="llm-grid-2 gap-xs"></div>
              </div>
              <div>
                <div class="tc-cyan" style="font-size:12px;font-weight:600;margin-bottom:6px;display:flex;align-items:center;gap:4px;">
                  <svg class="icon icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="6" r="3"/><path d="M8.12 8.12 12 12"/><path d="M20 4 8.12 15.88"/><circle cx="6" cy="18" r="3"/><path d="M14.8 14.8 20 20"/></svg>
                  <span>上下文 CP</span>
                </div>
                <div id="llm-grp-cp" class="llm-grid-2 gap-xs"></div>
              </div>
            </div>

            <div id="llm-panel-cp" class="sub-panel cyan hidden">
              <div style="display:flex;align-items:center;justify-content:space-between;font-size:12px;gap:8px;">
                <span id="llm-lbl-cp-local" style="font-weight:600;"></span>
                <span id="llm-lbl-cp-buf" class="font-mono tc-cyan" style="font-size:11px;font-weight:500;"></span>
              </div>
              <div class="llm-grid-2 gap-sm">
                <button id="llm-btn-cp-ring" class="opt-btn" style="font-size:11px;padding:6px 8px;">Ring Attention (环传 KV)</button>
                <button id="llm-btn-cp-ulysses" class="opt-btn" style="font-size:11px;padding:6px 8px;">Ulysses (头维 All-to-All)</button>
              </div>
              <label style="display:flex;align-items:center;justify-content:space-between;font-size:11px;cursor:pointer;padding-top:2px;gap:8px;">
                <span id="llm-lbl-zero-cp"></span>
                <input id="llm-chk-zero-cp" type="checkbox" class="accent-cyan" style="cursor:pointer;" />
              </label>
            </div>

            <div>
              <div class="ctrl-label-row" style="margin-bottom:6px;">
                <span id="llm-lbl-zero-domain" class="text-sub" style="font-weight:500;"></span>
                <span id="llm-lbl-zero-desc" class="font-mono tc-violet" style="font-size:11px;font-weight:500;"></span>
              </div>
              <div id="llm-grp-zero" class="llm-grid-4"></div>
            </div>

            <div class="llm-grid-3" style="gap:8px;padding-top:2px;">
              <button id="llm-btn-sp" class="toggle-card">
                <div class="toggle-card-top">
                  <span>SP 序列并行</span>
                  <span id="llm-lbl-sp-state" class="font-mono" style="font-size:10px;"></span>
                </div>
                <div class="toggle-card-hint">Norm/残差再÷TP</div>
              </button>
              <button id="llm-btn-fa" class="toggle-card">
                <div class="toggle-card-top">
                  <span>FlashAttn</span>
                  <span id="llm-lbl-fa-state" class="font-mono" style="font-size:10px;"></span>
                </div>
                <div class="toggle-card-hint">省 S² 存 FP32 LSE</div>
              </button>
              <button id="llm-btn-ce" class="toggle-card">
                <div class="toggle-card-top">
                  <span>分块 Loss</span>
                  <span id="llm-lbl-ce-state" class="font-mono" style="font-size:10px;"></span>
                </div>
                <div class="toggle-card-hint">压低 Logits(V维)</div>
              </button>
            </div>
          </div>

          <!-- 卡片 4：重计算引擎 + DualPipe 调度 -->
          <div class="llm-card">
            <div class="card-header">
              <div class="card-title-group">
                <svg class="icon icon-md tc-emerald-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/></svg>
                <h2 class="card-title">4. 激活重计算引擎与反向调度</h2>
              </div>
              <div style="display:flex;align-items:center;gap:6px;" class="font-mono">
                <span id="llm-badge-act-red" class="llm-badge badge-emerald"></span>
                <span id="llm-badge-comp-ovh" class="llm-badge badge-amber"></span>
              </div>
            </div>

            <div id="llm-grp-ckpt" class="llm-grid-3 gap-sm"></div>

            <div id="llm-panel-selective" class="sub-panel emerald hidden">
              <div style="display:flex;align-items:center;justify-content:space-between;font-size:11px;font-weight:600;gap:8px;">
                <span>选择要丢弃并在反向重算的子模块（Megatron-Core 风格）</span>
                <span id="llm-lbl-sel-ws" class="font-mono tc-emerald" style="flex-shrink:0;"></span>
              </div>
              <div id="llm-grp-selective" class="llm-grid-2" style="gap:8px;"></div>
            </div>

            <div id="llm-panel-block" class="sub-panel emerald hidden" style="gap:12px;">
              <div>
                <div class="ctrl-label-row">
                  <span id="llm-lbl-recomp-k-title" style="font-weight:500;"></span>
                  <span id="llm-lbl-recomp-k-val" class="font-mono tc-emerald" style="font-weight:700;"></span>
                </div>
                <input id="llm-inp-recomp-k" type="range" min="0" max="36" step="1" class="accent-emerald" />
              </div>

              <div>
                <div class="ctrl-label-row">
                  <span class="text-sub">Checkpoint 块粒度 G（每 G 层存 1 份输入）</span>
                  <span id="llm-lbl-recomp-g-val" class="font-mono tc-amber" style="font-weight:600;"></span>
                </div>
                <div id="llm-grp-block-g" class="llm-grid-4"></div>
              </div>
            </div>

            <div class="sub-panel">
              <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;">
                <div>
                  <div class="tc-amber" style="font-size:12px;font-weight:700;display:flex;align-items:center;gap:6px;">
                    <svg class="icon icon-xs" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                    <span>反向 dX / dW 拆分调度 (DualPipe)</span>
                  </div>
                  <p class="text-muted" style="font-size:11px;margin-top:2px;">需偶数 PP ≥ 2；延后算 dW 填充流水线气泡</p>
                </div>
                <div class="llm-seg-group">
                  <button id="llm-btn-sched-1f1b" class="llm-seg-btn">1F1B</button>
                  <button id="llm-btn-sched-dual" class="llm-seg-btn">DualPipe</button>
                </div>
              </div>

              <div id="llm-panel-dw-window" class="hidden" style="padding-top:8px;border-top:1px solid var(--llm-border);">
                <div class="ctrl-label-row">
                  <span class="text-sub">延后 dW 暂存窗口</span>
                  <span id="llm-lbl-dw-val" class="font-mono tc-amber" style="font-weight:700;"></span>
                </div>
                <input id="llm-inp-dw" type="range" min="1" max="8" step="1" class="accent-amber" />
              </div>
            </div>
          </div>
        </div>

        <div id="llm-right-panel" class="col-stack"></div>
      </div>
    </div>
  `;

  const $ = (id) => container.querySelector(`#llm-${id}`);

  function computeTopology() {
    const validTP = Math.min(state.TP, state.numGpus);
    const validPP = Math.min(state.PP, Math.max(1, Math.floor(state.numGpus / validTP)));
    const validCP = Math.min(state.CP, Math.max(1, Math.floor(state.numGpus / (validTP * validPP))));
    const parallelUnit = validTP * validPP * validCP;
    const isDivisible = state.numGpus % parallelUnit === 0;
    const DP = isDivisible ? Math.max(1, state.numGpus / parallelUnit) : 1;
    const zeroDomain = state.zeroAcrossCp ? DP * validCP : DP;
    const S_local = state.S / validCP;
    const GBS = state.MBS * DP * state.GAS;
    const layersPerStage = Math.ceil(state.L / validPP);
    const isDualPipeValid = validPP >= 2 && validPP % 2 === 0;

    return {
      TP: validTP, PP: validPP, CP: validCP, DP,
      zeroDomain, parallelUnit, isDivisible, S_local, GBS, layersPerStage, isDualPipeValid
    };
  }

  function computeMemStats(topology) {
    const { TP: tp, PP: pp, CP: cp, zeroDomain, S_local, layersPerStage, isDualPipeValid } = topology;
    const {
      H, L, H_ff, n_h, n_kv, V, tieEmbeddings, MBS, S, GAS, numGpus, vramCap,
      fp32GradBuffer, includeSysOverhead, SP, cpMode, zeroStage, zeroAcrossCp,
      flashAttn, chunkedCe, ckptMode, selectiveFlags, recompLayers, recompBlockSize,
      scheduleMode, delayedDwWindow
    } = state;

    const warnings = [];
    if (!topology.isDivisible) {
      warnings.push(`总卡数 (${numGpus}) 不能被 TP(${tp})×PP(${pp})×CP(${cp}) = ${topology.parallelUnit} 整除，DP 已回退为 1。`);
    }
    if (S % cp !== 0) {
      warnings.push(`序列长度 S (${S}) 不能被 CP (${cp}) 整除，请调整 S 或 CP。`);
    }
    if (L % pp !== 0) {
      warnings.push(`层数 L (${L}) 不能被 PP (${pp}) 整除，按最重 Stage = ${layersPerStage} 层保守估算。`);
    }
    if (H % n_h !== 0) {
      warnings.push(`隐藏维度 H (${H}) 不能被 Q 头数 n_h (${n_h}) 整除，head_dim 已取整。`);
    }
    if (n_kv > n_h || n_h % n_kv !== 0) {
      warnings.push(`GQA 配置异常：n_kv (${n_kv}) 应小于等于且整除 n_h (${n_h})。`);
    }
    if (tp > n_kv) {
      warnings.push(`TP (${tp}) > n_kv (${n_kv})：KV 头无法继续按 TP 切分，将在 ${tp / n_kv} 张卡间复制。`);
    }
    if (cp > 1 && cpMode === 'ulysses' && n_h / tp < cp) {
      warnings.push(`Ulysses CP 要求每卡 Q 头数 (n_h/TP = ${(n_h / tp).toFixed(1)}) ≥ CP (${cp})，否则无法按头做 All-to-All，建议改用 Ring Attention。`);
    }
    if (scheduleMode === 'dualpipe' && !isDualPipeValid) {
      warnings.push(`DualPipe 双向流水线要求 PP ≥ 2 且为偶数（当前 PP=${pp}），已按常规 1F1B 处理。`);
    }
    if (zeroStage > 0 && zeroDomain === 1) {
      warnings.push(`当前 ZeRO 切分域为 1（${zeroAcrossCp ? 'DP×CP=1' : 'DP=1'}），ZeRO-${zeroStage} 不会减少静态显存。`);
    }

    const d_head = Math.floor(H / Math.max(1, n_h));
    const q_dim = n_h * d_head;
    const kv_dim = n_kv * d_head;
    const kvTpDiv = Math.min(tp, Math.max(1, n_kv));

    const embedParams = V * H;
    const lmHeadParams = tieEmbeddings ? 0 : V * H;
    const attnParamsPerLayer = H * q_dim + 2 * H * kv_dim + q_dim * H;
    const mlpParamsPerLayer = 3 * H * H_ff;
    const perLayerParams = attnParamsPerLayer + mlpParamsPerLayer + 2 * H;
    const totalParams = embedParams + L * perLayerParams + H + lmHeadParams;

    const activeDualPipe = scheduleMode === 'dualpipe' && isDualPipeValid;
    let stageParams = totalParams;
    if (pp > 1) {
      if (activeDualPipe) {
        stageParams = layersPerStage * perLayerParams + embedParams + lmHeadParams + H;
      } else {
        const edgeExtra = Math.max(embedParams, (tieEmbeddings ? embedParams : lmHeadParams) + H);
        stageParams = layersPerStage * perLayerParams + edgeExtra;
      }
    }

    const zeroWeightDiv = zeroStage >= 3 ? zeroDomain : 1;
    const zeroGradDiv = zeroStage >= 2 ? zeroDomain : 1;
    const zeroOptDiv = zeroStage >= 1 ? zeroDomain : 1;

    const kvRepRatio = tp / kvTpDiv;
    const replicatedKvParamsPerLayer = 2 * H * kv_dim * (kvRepRatio - 1);
    const effectiveStageParamsAfterTp = stageParams / tp + (layersPerStage * replicatedKvParamsPerLayer) / tp;

    const weightBytes = (2 * effectiveStageParamsAfterTp) / zeroWeightDiv;
    const gradCoeff = fp32GradBuffer ? 4 : 2;
    const gradBytes = (gradCoeff * effectiveStageParamsAfterTp) / zeroGradDiv;
    const masterWeightBytes = (4 * effectiveStageParamsAfterTp) / zeroOptDiv;
    const adamMomentsBytes = (8 * effectiveStageParamsAfterTp) / zeroOptDiv;
    const optimizerTotalBytes = masterWeightBytes + adamMomentsBytes;

    const zero3TempBytes =
      zeroStage >= 3 && zeroDomain > 1
        ? (2 * perLayerParams * 2) / tp + (perLayerParams * gradCoeff) / tp
        : 0;
    const staticTotalBytes = weightBytes + gradBytes + optimizerTotalBytes + zero3TempBytes;

    const bshLocalBytes = MBS * S_local * H * 2;
    const qBshLocalBytes = MBS * S_local * q_dim * 2;
    const kvBshLocalBytes = MBS * S_local * kv_dim * 2;
    const ffBshLocalBytes = MBS * S_local * H_ff * 2;
    const spDiv = SP && tp > 1 ? tp : 1;

    const norm1PreAct = bshLocalBytes / spDiv;
    const norm1PostAct = bshLocalBytes / spDiv;
    const attnProjAct = qBshLocalBytes / tp + (2 * kvBshLocalBytes) / kvTpDiv + qBshLocalBytes / tp;
    const coreAttnAct = flashAttn
      ? (MBS * n_h * S_local * 4) / tp
      : (MBS * n_h * S_local * S * 7) / tp;
    const norm2PreAct = bshLocalBytes / spDiv;
    const norm2PostAct = bshLocalBytes / spDiv;
    const mlpLinearAct = (2 * ffBshLocalBytes) / tp;
    const mlpSiluAct = ffBshLocalBytes / tp;

    const fullLayerActBytes =
      norm1PreAct + norm1PostAct + attnProjAct + coreAttnAct + norm2PreAct + norm2PostAct + mlpLinearAct + mlpSiluAct;

    const attnProjFlops = 2 * MBS * S * (2 * H * q_dim + 2 * H * kv_dim);
    const coreAttnFlops = 2 * MBS * S * S * q_dim;
    const mlpFlops = 2 * MBS * S * (3 * H * H_ff);
    const normActFlops = 0.03 * (attnProjFlops + mlpFlops);
    const totalLayerFwdFlops = attnProjFlops + coreAttnFlops + mlpFlops + normActFlops;

    let perStageLayersActBytes = fullLayerActBytes * layersPerStage;
    let avgLayerActBytes = fullLayerActBytes;
    let recompWorkspaceBytes = 0;
    let recompFwdFraction = 0;
    let effectiveRecompLayers = 0;
    let effectiveBlockSize = 1;

    if (ckptMode === 'selective') {
      let savedPerLayer = 0;
      let extraFlopsPerLayer = 0;
      if (selectiveFlags.coreAttn && !flashAttn) {
        savedPerLayer += coreAttnAct;
        extraFlopsPerLayer += coreAttnFlops;
      }
      if (selectiveFlags.norm) {
        savedPerLayer += norm1PostAct + norm2PostAct;
        extraFlopsPerLayer += 0.5 * normActFlops;
      }
      if (selectiveFlags.mlpFull) {
        savedPerLayer += mlpLinearAct + mlpSiluAct + (selectiveFlags.norm ? 0 : norm2PostAct);
        extraFlopsPerLayer += mlpFlops + 0.5 * normActFlops;
      } else if (selectiveFlags.swiglu) {
        savedPerLayer += mlpSiluAct;
        extraFlopsPerLayer += 0.5 * normActFlops;
      }
      if (selectiveFlags.attnProj) {
        savedPerLayer += attnProjAct;
        extraFlopsPerLayer += attnProjFlops;
      }
      avgLayerActBytes = Math.max(bshLocalBytes / spDiv, fullLayerActBytes - savedPerLayer);
      perStageLayersActBytes = avgLayerActBytes * layersPerStage;
      recompWorkspaceBytes = savedPerLayer;
      recompFwdFraction = extraFlopsPerLayer / Math.max(1, totalLayerFwdFlops);
    } else if (ckptMode === 'block') {
      effectiveRecompLayers = Math.min(Math.max(0, recompLayers), layersPerStage);
      effectiveBlockSize = Math.min(Math.max(1, recompBlockSize), Math.max(1, effectiveRecompLayers));
      const keptLayers = layersPerStage - effectiveRecompLayers;
      const ckptBoundaryCount = effectiveRecompLayers > 0 ? Math.ceil(effectiveRecompLayers / effectiveBlockSize) : 0;
      const recompStoredBytes = ckptBoundaryCount * (bshLocalBytes / spDiv);
      const keptStoredBytes = keptLayers * fullLayerActBytes;
      perStageLayersActBytes = recompStoredBytes + keptStoredBytes;
      avgLayerActBytes = perStageLayersActBytes / Math.max(1, layersPerStage);
      recompWorkspaceBytes =
        effectiveRecompLayers > 0
          ? Math.max(0, effectiveBlockSize * fullLayerActBytes - bshLocalBytes / spDiv)
          : 0;
      recompFwdFraction = effectiveRecompLayers / Math.max(1, layersPerStage);
    }

    const ceExtraOverheadPct = chunkedCe ? 0.5 : 0;
    const computeOverheadPct = (recompFwdFraction / 3) * 100 + ceExtraOverheadPct;

    const ceChunkTokens = Math.min(S_local, 1024);
    const logitsActBytes = chunkedCe
      ? (MBS * ceChunkTokens * V * 6) / tp + bshLocalBytes / spDiv
      : (MBS * S_local * V * 6) / tp;

    let cpCommBufferBytes = 0;
    if (cp > 1) {
      cpCommBufferBytes = cpMode === 'ring' ? (4 * kvBshLocalBytes) / kvTpDiv : (2 * bshLocalBytes) / tp;
    }

    let inFlightMicroBatches = 1;
    let forwardPeakActBytes = 0;
    let bottleneckStageLabel = '单 Stage (无 PP)';

    if (pp === 1) {
      inFlightMicroBatches = 1;
      forwardPeakActBytes = perStageLayersActBytes + logitsActBytes;
    } else if (activeDualPipe) {
      inFlightMicroBatches = Math.min(pp + 1, GAS);
      forwardPeakActBytes = perStageLayersActBytes * inFlightMicroBatches + logitsActBytes;
      bottleneckStageLabel = `DualPipe Rank 0（双向 ${inFlightMicroBatches} 份在飞 + LM Head）`;
    } else {
      const stage0InFlight = Math.min(pp, GAS);
      const stage0Peak = perStageLayersActBytes * stage0InFlight;
      const lastStagePeak = perStageLayersActBytes * 1 + logitsActBytes;
      if (stage0Peak >= lastStagePeak) {
        inFlightMicroBatches = stage0InFlight;
        forwardPeakActBytes = stage0Peak;
        bottleneckStageLabel = `PP Stage 0（${stage0InFlight} 份在飞激活主导）`;
      } else {
        inFlightMicroBatches = 1;
        forwardPeakActBytes = lastStagePeak;
        bottleneckStageLabel = `PP 末 Stage（1 份在飞 + Logits 词表峰值主导）`;
      }
    }

    const bwdTempGradBytes = (2 * bshLocalBytes + ffBshLocalBytes) / tp;
    const effectiveWindow = activeDualPipe ? Math.min(delayedDwWindow, GAS) : 0;
    if (activeDualPipe && ckptMode === 'block' && effectiveRecompLayers > 0) {
      warnings.push(`机制提示：DualPipe 延后 dW 期间需暂存重算后的 (X, dY)，会抵消延后窗口内约 ${effectiveWindow} 步的部分重计算显存收益。`);
    }

    const perLayerDelayedDwRetain = activeDualPipe
      ? 4 * (bshLocalBytes / spDiv) + 2 * (ffBshLocalBytes / tp)
      : 0;
    const delayedDwExtraBytes = perLayerDelayedDwRetain * layersPerStage * effectiveWindow;
    const backwardPeakExtraBytes = recompWorkspaceBytes + bwdTempGradBytes + delayedDwExtraBytes + cpCommBufferBytes;
    const rawPeakBytes = staticTotalBytes + forwardPeakActBytes + backwardPeakExtraBytes;
    const sysOverheadBytes = includeSysOverhead ? 2.0 * (1024 ** 3) + rawPeakBytes * 0.05 : 0;
    const totalPeakBytes = rawPeakBytes + sysOverheadBytes;
    const vramBytes = vramCap * (1024 ** 3);
    const isOOM = totalPeakBytes > vramBytes;

    const baselineNoRecompAct = fullLayerActBytes * layersPerStage;
    const actReductionPct =
      baselineNoRecompAct > 0 ? Math.max(0, (1 - perStageLayersActBytes / baselineNoRecompAct) * 100) : 0;

    return {
      warnings, d_head, kvTpDiv, totalParams, stageParams, perLayerParams, layersPerStage,
      inFlightMicroBatches, bottleneckStageLabel, activeDualPipe, bshLocalBytes, ffBshLocalBytes, kvBshLocalBytes,
      weightBytes, gradBytes, gradCoeff, optimizerTotalBytes, zero3TempBytes, staticTotalBytes,
      norm1PreAct, norm1PostAct, attnProjAct, coreAttnAct, norm2PreAct, norm2PostAct, mlpLinearAct, mlpSiluAct,
      fullLayerActBytes, avgLayerActBytes, perStageLayersActBytes, logitsActBytes, forwardPeakActBytes,
      recompWorkspaceBytes, effectiveRecompLayers, effectiveBlockSize, computeOverheadPct, actReductionPct,
      bwdTempGradBytes, delayedDwExtraBytes, effectiveWindow, cpCommBufferBytes, backwardPeakExtraBytes,
      sysOverheadBytes, totalPeakBytes, vramBytes, isOOM
    };
  }

  function applyPreset(p) {
    state.presetId = p.id;
    state.H = p.H;
    state.L = p.L;
    state.H_ff = p.H_ff;
    state.n_h = p.n_h;
    state.n_kv = p.n_kv;
    state.V = p.V;
    state.tieEmbeddings = p.tieEmbeddings;
    state.numGpus = p.defaultGpus;
    state.vramCap = p.defaultVram;
    state.MBS = p.defaultMbs;
    state.S = p.defaultS;
    state.GAS = p.defaultGas;
    state.TP = p.defaultTp;
    state.PP = p.defaultPp;
    state.CP = p.defaultCp;
    state.zeroStage = p.defaultZero;
    state.SP = p.defaultTp > 1;
    state.recompLayers = Math.ceil(p.L / p.defaultPp);
    state.recompBlockSize = 1;
    if (p.defaultPp < 2 || p.defaultPp % 2 !== 0) state.scheduleMode = '1f1b';
    if (p.id === 'qwen3_32b_long') {
      state.ckptMode = 'selective';
      state.chunkedCe = true;
    } else {
      state.ckptMode = 'none';
      state.chunkedCe = false;
    }
    render();
  }

  function handleNumGpusChange(nextGpus) {
    state.numGpus = nextGpus;
    const nextTP = Math.min(state.TP, nextGpus);
    const nextPP = Math.min(state.PP, Math.max(1, Math.floor(nextGpus / nextTP)));
    const nextCP = Math.min(state.CP, Math.max(1, Math.floor(nextGpus / (nextTP * nextPP))));
    state.TP = nextTP;
    state.PP = nextPP;
    state.CP = nextCP;
    if (nextTP === 1) state.SP = false;
    if (nextPP < 2 || nextPP % 2 !== 0) state.scheduleMode = '1f1b';
    state.recompLayers = Math.min(state.recompLayers, Math.ceil(state.L / nextPP));
    render();
  }

  function handleGasChange(nextGas) {
    state.GAS = nextGas;
    state.delayedDwWindow = Math.min(state.delayedDwWindow, Math.max(1, Math.min(8, nextGas)));
    render();
  }

  function applyRecommendedRecipe() {
    const topology = computeTopology();
    const memStats = computeMemStats(topology);
    if (!state.flashAttn) { state.flashAttn = true; render(); return; }
    if (!state.chunkedCe && memStats.logitsActBytes > 1 * (1024 ** 3)) { state.chunkedCe = true; render(); return; }
    if (topology.TP > 1 && !state.SP) { state.SP = true; render(); return; }
    if (state.zeroStage < 1 && topology.zeroDomain > 1) { state.zeroStage = 1; render(); return; }
    if (state.S >= 16384 && topology.CP === 1 && topology.DP >= 2) { state.CP = 2; render(); return; }
    if (state.ckptMode === 'none') { state.ckptMode = 'selective'; render(); return; }
    if (state.ckptMode === 'selective') {
      state.ckptMode = 'block';
      state.recompLayers = topology.layersPerStage;
      state.recompBlockSize = 1;
      render();
      return;
    }
    if (state.ckptMode === 'block' && state.recompLayers < topology.layersPerStage) {
      state.recompLayers = topology.layersPerStage;
      render();
      return;
    }
    if (state.zeroStage < 2 && topology.zeroDomain > 1) { state.zeroStage = 2; render(); return; }
    if (state.zeroStage < 3 && topology.zeroDomain > 1) { state.zeroStage = 3; render(); }
  }

  function initStaticControls() {
    $('sel-nh').innerHTML = NH_OPTIONS.map((v) => `<option value="${v}">Q:${v}</option>`).join('');
    $('sel-V').innerHTML = V_OPTIONS.map((v) => `<option value="${v}">${v.toLocaleString()}</option>`).join('');
    $('sel-S').innerHTML = S_OPTIONS.map((v) => `<option value="${v}">${v.toLocaleString()} tokens</option>`).join('');
    $('sel-gpus').innerHTML = GPU_OPTIONS.map((n) => `<option value="${n}">${n} × GPU</option>`).join('');
    $('sel-vram').innerHTML = VRAM_OPTIONS.map((o) => `<option value="${o.value}">${o.label}</option>`).join('');

    $('presets-bar').innerHTML = MODEL_PRESETS.map((p) => `
      <button class="preset-card" data-preset="${p.id}">
        <div class="preset-head">
          <span class="preset-name">${p.name}</span>
          <span class="preset-tag">${p.tag}</span>
        </div>
        <p class="preset-desc line-clamp-1">${p.desc}</p>
        <div class="preset-meta font-mono">
          <span>H=${p.H}</span><span>·</span>
          <span>L=${p.L}</span><span>·</span>
          <span>S=${p.defaultS >= 1024 ? `${p.defaultS / 1024}K` : p.defaultS}</span><span>·</span>
          <span>${p.defaultGpus}卡</span>
        </div>
      </button>
    `).join('');

    $('presets-bar').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-preset]');
      if (!btn) return;
      const preset = MODEL_PRESETS.find((x) => x.id === btn.dataset.preset);
      if (preset) applyPreset(preset);
    });

    $('grp-tp').innerHTML = [1, 2, 4, 8].map((v) => `<button class="opt-btn font-mono" data-tp="${v}">${v}</button>`).join('');
    $('grp-pp').innerHTML = [1, 2, 4, 8].map((v) => `<button class="opt-btn font-mono" data-pp="${v}">${v}</button>`).join('');
    $('grp-cp').innerHTML = [1, 2, 4, 8].map((v) => `<button class="opt-btn font-mono" data-cp="${v}">${v}</button>`).join('');
    $('grp-zero').innerHTML = [
      { s: 0, label: 'DDP' }, { s: 1, label: 'ZeRO-1' }, { s: 2, label: 'ZeRO-2' }, { s: 3, label: 'ZeRO-3' }
    ].map((item) => `<button class="opt-btn" data-zero="${item.s}">${item.label}</button>`).join('');
    $('grp-ckpt').innerHTML = [
      { id: 'none', label: '关闭 (全留)' },
      { id: 'selective', label: '细粒度选择性' },
      { id: 'block', label: '分层 / 全层块重算' }
    ].map((m) => `<button class="opt-btn" data-ckpt="${m.id}">${m.label}</button>`).join('');
    $('grp-block-g').innerHTML = [1, 2, 4, 8].map((g) => `<button class="opt-btn font-mono" data-g="${g}">每 ${g} 层</button>`).join('');

    $('inp-H').addEventListener('input', (e) => { state.H = Number(e.target.value); state.presetId = 'custom'; render(); });
    $('inp-L').addEventListener('input', (e) => {
      const nextL = Number(e.target.value);
      state.L = nextL;
      const topo = computeTopology();
      state.recompLayers = Math.min(state.recompLayers, Math.ceil(nextL / topo.PP));
      state.presetId = 'custom';
      render();
    });
    $('inp-Hff').addEventListener('input', (e) => { state.H_ff = Number(e.target.value); state.presetId = 'custom'; render(); });
    $('sel-nh').addEventListener('change', (e) => {
      const nextNh = Number(e.target.value);
      state.n_h = nextNh;
      if (state.n_kv > nextNh) state.n_kv = nextNh;
      state.presetId = 'custom';
      render();
    });
    $('sel-nkv').addEventListener('change', (e) => { state.n_kv = Number(e.target.value); state.presetId = 'custom'; render(); });
    $('sel-V').addEventListener('change', (e) => { state.V = Number(e.target.value); state.presetId = 'custom'; render(); });
    $('chk-tie').addEventListener('change', (e) => { state.tieEmbeddings = e.target.checked; state.presetId = 'custom'; render(); });

    $('inp-mbs').addEventListener('input', (e) => { state.MBS = Number(e.target.value); render(); });
    $('sel-S').addEventListener('change', (e) => { state.S = Number(e.target.value); render(); });
    $('inp-gas').addEventListener('input', (e) => handleGasChange(Number(e.target.value)));
    $('sel-gpus').addEventListener('change', (e) => handleNumGpusChange(Number(e.target.value)));
    $('sel-vram').addEventListener('change', (e) => { state.vramCap = Number(e.target.value); render(); });
    $('chk-fp32grad').addEventListener('change', (e) => { state.fp32GradBuffer = e.target.checked; render(); });
    $('chk-sysoverhead').addEventListener('change', (e) => { state.includeSysOverhead = e.target.checked; render(); });

    $('grp-tp').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-tp]');
      if (!btn || btn.disabled) return;
      const t = Number(btn.dataset.tp);
      state.TP = t;
      if (t === 1) state.SP = false;
      const remAfterTp = Math.max(1, Math.floor(state.numGpus / t));
      const nextPp = Math.min(state.PP, remAfterTp);
      const nextCp = Math.min(state.CP, Math.max(1, Math.floor(remAfterTp / nextPp)));
      state.PP = nextPp;
      state.CP = nextCp;
      render();
    });
    $('grp-pp').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-pp]');
      if (!btn || btn.disabled) return;
      const p = Number(btn.dataset.pp);
      const topo = computeTopology();
      state.PP = p;
      const remAfterPp = Math.max(1, Math.floor(state.numGpus / (topo.TP * p)));
      state.CP = Math.min(state.CP, remAfterPp);
      state.recompLayers = Math.min(state.recompLayers, Math.ceil(state.L / p));
      if (p < 2 || p % 2 !== 0) state.scheduleMode = '1f1b';
      render();
    });
    $('grp-cp').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-cp]');
      if (!btn || btn.disabled) return;
      state.CP = Number(btn.dataset.cp);
      render();
    });
    $('btn-cp-ring').addEventListener('click', () => { state.cpMode = 'ring'; render(); });
    $('btn-cp-ulysses').addEventListener('click', () => { state.cpMode = 'ulysses'; render(); });
    $('chk-zero-cp').addEventListener('change', (e) => { state.zeroAcrossCp = e.target.checked; render(); });
    $('grp-zero').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-zero]');
      if (!btn) return;
      state.zeroStage = Number(btn.dataset.zero);
      render();
    });
    $('btn-sp').addEventListener('click', () => {
      if (computeTopology().TP === 1) return;
      state.SP = !state.SP;
      render();
    });
    $('btn-fa').addEventListener('click', () => { state.flashAttn = !state.flashAttn; render(); });
    $('btn-ce').addEventListener('click', () => { state.chunkedCe = !state.chunkedCe; render(); });

    $('grp-ckpt').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-ckpt]');
      if (!btn) return;
      state.ckptMode = btn.dataset.ckpt;
      render();
    });
    $('grp-selective').addEventListener('change', (e) => {
      const inp = e.target.closest('[data-sel]');
      if (!inp) return;
      const k = inp.dataset.sel;
      state.selectiveFlags[k] = !state.selectiveFlags[k];
      render();
    });
    $('inp-recomp-k').addEventListener('input', (e) => { state.recompLayers = Number(e.target.value); render(); });
    $('grp-block-g').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-g]');
      if (!btn || btn.disabled) return;
      state.recompBlockSize = Number(btn.dataset.g);
      render();
    });
    $('btn-sched-1f1b').addEventListener('click', () => { state.scheduleMode = '1f1b'; render(); });
    $('btn-sched-dual').addEventListener('click', () => {
      if (!computeTopology().isDualPipeValid) return;
      state.scheduleMode = 'dualpipe';
      render();
    });
    $('inp-dw').addEventListener('input', (e) => { state.delayedDwWindow = Number(e.target.value); render(); });

    $('right-panel').addEventListener('click', (e) => {
      if (e.target.closest('#llm-btn-recipe')) {
        applyRecommendedRecipe();
        return;
      }
      const tabBtn = e.target.closest('[data-tab]');
      if (tabBtn) {
        state.activeTab = tabBtn.dataset.tab;
        render();
      }
    });
  }

  function render() {
    const topology = computeTopology();
    const memStats = computeMemStats(topology);

    if (typeof onMetricsUpdate === 'function') {
      onMetricsUpdate({
        paramsText: formatParams(memStats.totalParams),
        computeText: `+${memStats.computeOverheadPct.toFixed(1)}%`,
        vramText: `${formatGiB(memStats.totalPeakBytes)} / ${state.vramCap} GiB`,
        isOOM: memStats.isOOM
      });
    }

    container.querySelectorAll('#llm-presets-bar [data-preset]').forEach((el) => {
      el.classList.toggle('active', el.dataset.preset === state.presetId);
    });

    $('lbl-dhead').textContent = `d_head=${memStats.d_head}`;
    $('lbl-H').textContent = state.H;
    $('inp-H').value = state.H;
    $('lbl-L').textContent = state.L;
    $('inp-L').value = state.L;
    $('lbl-Hff').textContent = state.H_ff;
    $('inp-Hff').value = state.H_ff;
    $('lbl-heads').textContent = `${state.n_h} / ${state.n_kv}`;
    $('sel-nh').value = String(state.n_h);
    const validNkv = NKV_OPTIONS.filter((v) => v <= state.n_h);
    const selNkv = $('sel-nkv');
    const expectedNkvKey = validNkv.join(',');
    if (selNkv.dataset.opts !== expectedNkvKey) {
      selNkv.innerHTML = validNkv.map((v) => `<option value="${v}">KV:${v}</option>`).join('');
      selNkv.dataset.opts = expectedNkvKey;
    }
    selNkv.value = String(state.n_kv);
    $('sel-V').value = String(state.V);
    $('chk-tie').checked = state.tieEmbeddings;

    $('lbl-gbs').textContent = `GBS = ${state.MBS}×${topology.DP}×${state.GAS} = ${topology.GBS}`;
    $('lbl-mbs').textContent = state.MBS;
    $('inp-mbs').value = state.MBS;
    $('lbl-S').textContent = state.S >= 1024 ? `${state.S / 1024}K` : state.S;
    $('sel-S').value = String(state.S);
    $('lbl-gas').textContent = state.GAS;
    $('inp-gas').value = state.GAS;
    $('lbl-gpus').textContent = `${state.numGpus} 卡`;
    $('sel-gpus').value = String(state.numGpus);
    $('sel-vram').value = String(state.vramCap);
    $('chk-fp32grad').checked = state.fp32GradBuffer;
    $('chk-sysoverhead').checked = state.includeSysOverhead;

    $('lbl-topo').textContent = `TP=${topology.TP}·PP=${topology.PP}·CP=${topology.CP}·DP=${topology.DP}`;
    container.querySelectorAll('#llm-grp-tp [data-tp]').forEach((btn) => {
      const v = Number(btn.dataset.tp);
      btn.disabled = v > state.numGpus;
      btn.classList.toggle('active-sky', topology.TP === v);
    });
    container.querySelectorAll('#llm-grp-pp [data-pp]').forEach((btn) => {
      const v = Number(btn.dataset.pp);
      btn.disabled = v * topology.TP > state.numGpus;
      btn.classList.toggle('active-sky', topology.PP === v);
    });
    container.querySelectorAll('#llm-grp-cp [data-cp]').forEach((btn) => {
      const v = Number(btn.dataset.cp);
      btn.disabled = v * topology.TP * topology.PP > state.numGpus;
      btn.classList.toggle('active-cyan', topology.CP === v);
    });

    const panelCp = $('panel-cp');
    panelCp.classList.toggle('hidden', topology.CP <= 1);
    if (topology.CP > 1) {
      $('lbl-cp-local').textContent = `CP=${topology.CP} 生效：单卡序列 S_local = ${Math.round(topology.S_local).toLocaleString()}`;
      $('lbl-cp-buf').textContent = `通信缓冲 ${formatGiB(memStats.cpCommBufferBytes)}`;
      $('btn-cp-ring').classList.toggle('active-cyan', state.cpMode === 'ring');
      $('btn-cp-ulysses').classList.toggle('active-cyan', state.cpMode === 'ulysses');
      $('lbl-zero-cp').textContent = `ZeRO 沿 DP×CP (${topology.DP * topology.CP}) 联合切分状态 (Megatron-Core)`;
      $('chk-zero-cp').checked = state.zeroAcrossCp;
    }

    $('lbl-zero-domain').textContent = `ZeRO 切分（切分域 = ${topology.zeroDomain}）`;
    const zeroDescMap = {
      0: '关闭 (DDP)',
      1: '切 12P 优化器',
      2: '+切梯度缓冲',
      3: '+切 2P 权重(含收集桶)'
    };
    $('lbl-zero-desc').textContent = zeroDescMap[state.zeroStage];
    container.querySelectorAll('#llm-grp-zero [data-zero]').forEach((btn) => {
      btn.classList.toggle('active-violet', Number(btn.dataset.zero) === state.zeroStage);
    });

    const spActive = state.SP && topology.TP > 1;
    $('btn-sp').disabled = topology.TP === 1;
    $('btn-sp').classList.toggle('on', spActive);
    $('lbl-sp-state').textContent = spActive ? 'ON' : 'OFF';
    $('btn-fa').classList.toggle('on', state.flashAttn);
    $('lbl-fa-state').textContent = state.flashAttn ? 'ON' : 'OFF';
    $('btn-ce').classList.toggle('on', state.chunkedCe);
    $('lbl-ce-state').textContent = state.chunkedCe ? 'ON' : 'OFF';

    $('badge-act-red').textContent = `激活 -${memStats.actReductionPct.toFixed(0)}%`;
    $('badge-comp-ovh').textContent = `算力 +${memStats.computeOverheadPct.toFixed(1)}%`;
    container.querySelectorAll('#llm-grp-ckpt [data-ckpt]').forEach((btn) => {
      btn.classList.toggle('active-emerald', btn.dataset.ckpt === state.ckptMode);
    });

    $('panel-selective').classList.toggle('hidden', state.ckptMode !== 'selective');
    if (state.ckptMode === 'selective') {
      $('lbl-sel-ws').textContent = `工作区 ${formatGiB(memStats.recompWorkspaceBytes)}`;
      const selOpts = [
        { key: 'norm', label: 'RMSNorm 输出', hint: '低开销 (~0.5% FLOPs)' },
        { key: 'swiglu', label: 'SwiGLU 激活函数', hint: state.selectiveFlags.mlpFull ? '已含在 MLP 整块' : '省 1×H_ff (~0.5%)' },
        { key: 'coreAttn', label: 'Core Attention', hint: state.flashAttn ? 'FlashAttn 已内置' : '省 S² 矩阵' },
        { key: 'mlpFull', label: 'MLP 整块重算', hint: '省 3×H_ff (高算力代价)' },
        { key: 'attnProj', label: 'Attn QKV/O 投影', hint: '省 QKVO (中算力代价)' }
      ];
      $('grp-selective').innerHTML = selOpts.map((opt) => {
        const checked = !!state.selectiveFlags[opt.key];
        return `
          <label class="sel-card ${checked ? 'checked' : ''}">
            <input type="checkbox" data-sel="${opt.key}" class="accent-emerald" ${checked ? 'checked' : ''} />
            <div>
              <div class="sel-title">${opt.label}</div>
              <div class="sel-hint">${opt.hint}</div>
            </div>
          </label>
        `;
      }).join('');
    }

    $('panel-block').classList.toggle('hidden', state.ckptMode !== 'block');
    if (state.ckptMode === 'block') {
      const pct = Math.round((memStats.effectiveRecompLayers / Math.max(1, topology.layersPerStage)) * 100);
      $('lbl-recomp-k-title').textContent = `本卡重计算层数 K（共 ${topology.layersPerStage} 层）`;
      $('lbl-recomp-k-val').textContent = `${memStats.effectiveRecompLayers} / ${topology.layersPerStage} 层 (${pct}%)`;
      const inpK = $('inp-recomp-k');
      inpK.max = String(topology.layersPerStage);
      inpK.value = String(memStats.effectiveRecompLayers);

      $('lbl-recomp-g-val').textContent = `G=${memStats.effectiveBlockSize}（工作区 ${formatGiB(memStats.recompWorkspaceBytes)}）`;
      container.querySelectorAll('#llm-grp-block-g [data-g]').forEach((btn) => {
        const g = Number(btn.dataset.g);
        btn.disabled = g > Math.max(1, memStats.effectiveRecompLayers);
        btn.classList.toggle('active-emerald', memStats.effectiveBlockSize === g);
      });
    }

    $('btn-sched-1f1b').classList.toggle('active-dark', state.scheduleMode === '1f1b');
    const btnDual = $('btn-sched-dual');
    btnDual.disabled = !topology.isDualPipeValid;
    btnDual.classList.toggle('active-amber', state.scheduleMode === 'dualpipe' && topology.isDualPipeValid);

    $('panel-dw-window').classList.toggle('hidden', !memStats.activeDualPipe);
    if (memStats.activeDualPipe) {
      $('lbl-dw-val').textContent = `${memStats.effectiveWindow} 步 (+${formatGiB(memStats.delayedDwExtraBytes)})`;
      const inpDw = $('inp-dw');
      inpDw.max = String(Math.max(1, Math.min(8, state.GAS)));
      inpDw.value = String(memStats.effectiveWindow);
    }

    renderRightPanel(topology, memStats);
  }

  function renderRightPanel(topology, memStats) {
    const barScaleMax = Math.max(memStats.totalPeakBytes, memStats.vramBytes);
    const vramLinePct = Math.min(100, (memStats.vramBytes / barScaleMax) * 100);

    const segments = [
      {
        id: 'weights',
        label: '模型权重 W (2P)',
        bytes: memStats.weightBytes,
        bg: 'bg-sky',
        tc: 'tc-sky',
        cardCls: 'sky',
        desc: `瓶颈 Stage 参数经 TP(${topology.TP})${state.zeroStage >= 3 ? `×ZeRO-3(${topology.zeroDomain})` : ''} 切分`
      },
      {
        id: 'grads',
        label: `梯度 dW (${memStats.gradCoeff}P)`,
        bytes: memStats.gradBytes,
        bg: 'bg-indigo',
        tc: 'tc-indigo',
        cardCls: 'indigo',
        desc: `${state.fp32GradBuffer ? 'FP32' : 'BF16'} 累加桶，GAS=${state.GAS} 原地累加不翻倍`
      },
      {
        id: 'opt',
        label: state.zeroStage >= 3 ? '优化器 (12P) + Z3 缓冲' : '优化器状态 (12P)',
        bytes: memStats.optimizerTotalBytes + memStats.zero3TempBytes,
        bg: 'bg-violet',
        tc: 'tc-violet',
        cardCls: 'violet',
        desc: `4P Master + 8P m/v，经 ZeRO 域 (${state.zeroStage >= 1 ? topology.zeroDomain : 1}) 切分${
          memStats.zero3TempBytes > 0 ? `，含 Z3 通信桶 ${formatGiB(memStats.zero3TempBytes)}` : ''
        }`
      },
      {
        id: 'fwd_act',
        label: '前向峰值激活 (B·S/CP·H)',
        bytes: memStats.forwardPeakActBytes,
        bg: 'bg-emerald',
        tc: 'tc-emerald',
        cardCls: 'emerald',
        desc: `${memStats.bottleneckStageLabel}，单卡序列 S_local=${Math.round(topology.S_local)}`
      },
      {
        id: 'bwd_extra',
        label: memStats.activeDualPipe ? '反向工作区 + 延后 dW + CP 缓冲' : '反向工作区 / 重算 / CP 缓冲',
        bytes: memStats.backwardPeakExtraBytes,
        bg: 'bg-amber',
        tc: 'tc-amber',
        cardCls: 'amber',
        desc: `重算区 ${formatGiB(memStats.recompWorkspaceBytes)} + 梯度临时区 + CP 缓冲 ${formatGiB(memStats.cpCommBufferBytes)}`
      }
    ];

    if (state.includeSysOverhead) {
      segments.push({
        id: 'sys',
        label: 'CUDA / NCCL / 碎片预留',
        bytes: memStats.sysOverheadBytes,
        bg: 'bg-slate',
        tc: 'tc-slate',
        cardCls: 'slate',
        desc: '约 2 GiB 基础上下文 + 5% 显存分配器碎片安全余量'
      });
    }

    const warningsHtml = memStats.warnings.length > 0
      ? `<div class="warn-box">${memStats.warnings.map((w) => `
          <div class="warn-row">
            <span class="tc-amber-icon" style="margin-top:1px;">${ICONS.alert}</span>
            <span>${w}</span>
          </div>`).join('')}</div>`
      : '';

    const statusBadge = memStats.isOOM
      ? `<span class="llm-badge llm-badge-pill badge-rose">${ICONS.alertXs}<span>显存溢出 (OOM +${formatGiB(memStats.totalPeakBytes - memStats.vramBytes)})</span></span>`
      : `<span class="llm-badge llm-badge-pill badge-ok">${ICONS.checkXs}<span>可运行 (余量 ${formatGiB(memStats.vramBytes - memStats.totalPeakBytes)})</span></span>`;

    const recipeBtn = memStats.isOOM
      ? `<button id="llm-btn-recipe" class="btn-recipe">${ICONS.sparklesXs}<span>逐步叠加优化配方</span></button>`
      : '';

    const barSegsHtml = segments.map((seg) => {
      const widthPct = (seg.bytes / barScaleMax) * 100;
      if (widthPct <= 0.2) return '';
      return `
        <div class="bar-seg ${seg.bg}" style="width:${widthPct}%" title="${seg.label}: ${formatGiB(seg.bytes)}">
          ${widthPct > 8 ? `<span class="bar-seg-label font-mono truncate">${formatGiB(seg.bytes)}</span>` : ''}
        </div>`;
    }).join('');

    const segCardsHtml = segments.map((seg) => {
      const pctOfTotal = memStats.totalPeakBytes > 0 ? (seg.bytes / memStats.totalPeakBytes) * 100 : 0;
      return `
        <div class="seg-card ${seg.cardCls}">
          <div class="seg-card-top">
            <div class="seg-title-wrap">
              <span class="seg-dot ${seg.bg}"></span>
              <span class="seg-title truncate">${seg.label}</span>
            </div>
            <span class="seg-pct font-mono">${pctOfTotal.toFixed(1)}%</span>
          </div>
          <div class="seg-val font-mono tabular-nums ${seg.tc}">${formatGiB(seg.bytes)}</div>
          <p class="seg-desc">${seg.desc}</p>
        </div>`;
    }).join('');

    const tabs = [
      { id: 'lifecycle', label: '训练阶段显存水位图', icon: ICONS.activityXs },
      { id: 'formulas', label: '4D 并行与重计算公式核对', icon: ICONS.calcXs },
      { id: 'layer_detail', label: '单层张量与 CP/重计算账本', icon: ICONS.layersXs }
    ];

    let tabContentHtml = '';
    if (state.activeTab === 'lifecycle') {
      const stages = [
        {
          step: '阶段 1：静态基线',
          title: '权重 + 优化器状态',
          bytes: memStats.weightBytes + memStats.optimizerTotalBytes + memStats.sysOverheadBytes,
          note: `瓶颈 Stage 参数 ${formatParams(memStats.stageParams)}`,
          bg: 'bg-sky'
        },
        {
          step: '阶段 2：前向峰值',
          title: '+ 压缩后激活 + 梯度桶',
          bytes: memStats.staticTotalBytes + memStats.forwardPeakActBytes + memStats.sysOverheadBytes,
          note: memStats.bottleneckStageLabel,
          bg: 'bg-emerald'
        },
        {
          step: '阶段 3：反向峰值',
          title: '+ 重算工作区 & CP/dW',
          bytes: memStats.totalPeakBytes,
          note: `重算区 ${formatGiB(memStats.recompWorkspaceBytes)} · CP 缓冲 ${formatGiB(memStats.cpCommBufferBytes)}`,
          bg: memStats.isOOM ? 'bg-rose' : 'bg-amber'
        },
        {
          step: '阶段 4：Optimizer Step',
          title: '释放激活，更新权重',
          bytes: memStats.staticTotalBytes + memStats.sysOverheadBytes,
          note: `走完 GAS=${state.GAS} 后用 FP32 Master 更新`,
          bg: 'bg-violet'
        }
      ];

      tabContentHtml = `
        <div class="stage-grid">
          ${stages.map((st) => {
            const hPct = Math.min(100, Math.max(12, (st.bytes / barScaleMax) * 100));
            return `
              <div class="stage-card">
                <div>
                  <div class="stage-step font-mono">${st.step}</div>
                  <div class="stage-title">${st.title}</div>
                </div>
                <div class="stage-well">
                  <div class="stage-fill ${st.bg}" style="height:${hPct}%">
                    <span class="bar-seg-label font-mono">${formatGiB(st.bytes)}</span>
                  </div>
                </div>
                <p class="stage-note">${st.note}</p>
              </div>`;
          }).join('')}
        </div>
        <div class="llm-callout">
          <div class="callout-title">${ICONS.infoMd}<span>为什么长上下文必须用 CP + 重计算组合？</span></div>
          <p class="callout-body">
            当序列长度达到 32K~128K 时，即便开启全层重计算，单层边界输入与反向重算工作区仍正比于
            <code class="code-chip cyan font-mono">S</code>；引入
            <code class="code-chip cyan font-mono">CP=${topology.CP}</code> 将单卡序列切分为
            <code class="code-chip cyan font-mono">S_local = ${Math.round(topology.S_local)}</code>，同时线性压低每层保存激活、重算工作区与词表 Logits，且配合
            <code class="code-chip violet font-mono">ZeRO 沿 DP×CP 切分</code>，不会浪费静态显存切分度。
          </p>
        </div>`;
    } else if (state.activeTab === 'formulas') {
      tabContentHtml = `
        <div style="display:flex;flex-direction:column;gap:16px;">
          <div class="formula-block">
            <div class="formula-heading tc-sky">一、 静态账本（瓶颈 Stage 参数 P_stage = ${formatParams(memStats.stageParams)}）</div>
            <div class="formula-code font-mono">
              <div>• 权重 W (2P_stage) ÷ (TP=${topology.TP} × Z3=${state.zeroStage >= 3 ? topology.zeroDomain : 1}) = <strong class="tc-sky">${formatGiB(memStats.weightBytes)}</strong></div>
              <div>• 梯度 dW (${memStats.gradCoeff}P_stage) ÷ (TP=${topology.TP} × Z2=${state.zeroStage >= 2 ? topology.zeroDomain : 1}) = <strong class="tc-indigo">${formatGiB(memStats.gradBytes)}</strong></div>
              <div>• 优化器 (12P_stage) ÷ (TP=${topology.TP} × Z1=${state.zeroStage >= 1 ? topology.zeroDomain : 1}) = <strong class="tc-violet">${formatGiB(memStats.optimizerTotalBytes)}</strong></div>
              ${memStats.zero3TempBytes > 0 ? `<div>• ZeRO-3 瞬时 AllGather / ReduceScatter 桶 = <strong class="tc-violet">${formatGiB(memStats.zero3TempBytes)}</strong></div>` : ''}
            </div>
          </div>
          <div class="formula-block">
            <div class="formula-heading tc-emerald">二、 动态激活与重计算账本（B=${state.MBS}, S_local=${Math.round(topology.S_local)}, H=${state.H}）</div>
            <div class="formula-code font-mono">
              <div>• 单张 (B, S/CP, H) BF16 = ${state.MBS} × ${Math.round(topology.S_local)} × ${state.H} × 2B = <strong class="tc-emerald">${formatGiB(memStats.bshLocalBytes)}</strong></div>
              <div>• 本卡 ${memStats.layersPerStage} 层保存激活 = <strong class="tc-emerald">${formatGiB(memStats.perStageLayersActBytes)}</strong>（无重算基线 ${formatGiB(memStats.fullLayerActBytes * memStats.layersPerStage)}）</div>
              <div>• 输出 Logits &amp; CE（${state.chunkedCe ? '分块融合 1024-tok' : '完整 FP32+BF16 6B'}）= <strong class="tc-emerald">${formatGiB(memStats.logitsActBytes)}</strong></div>
              <div>• 反向重算临时工作区 = <strong class="tc-amber">${formatGiB(memStats.recompWorkspaceBytes)}</strong> | 算力开销 = <strong class="tc-amber">+${memStats.computeOverheadPct.toFixed(1)}%</strong></div>
            </div>
          </div>
        </div>`;
    } else {
      const spOn = state.SP && topology.TP > 1;
      const spSeq = Math.round(topology.S_local / (spOn ? topology.TP : 1));
      const { ckptMode, selectiveFlags, flashAttn, MBS, H, H_ff, n_h, S } = state;

      tabContentHtml = `
        <p class="text-sub" style="font-size:12px;">
          单层各子张量在当前 4D 并行（TP=${topology.TP}, SP=${spOn ? 'ON' : 'OFF'}, CP=${topology.CP}）与重计算模式下的单卡实占字节数：
        </p>
        <div class="table-wrap">
          <table class="detail-table">
            <thead>
              <tr>
                <th>张量 / 子模块</th>
                <th>单卡形状</th>
                <th>原始单层占用</th>
                <th>当前重计算状态</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td class="font-mono tc-emerald" style="font-weight:500;">RMSNorm 1/2 输入 (X)</td>
                <td class="font-mono">2×(${MBS}, ${spSeq}, ${H})</td>
                <td class="font-mono">${formatGiB(memStats.norm1PreAct + memStats.norm2PreAct)}</td>
                <td>${ckptMode === 'block' && memStats.effectiveRecompLayers > 0 ? `每 ${memStats.effectiveBlockSize} 层仅留 1 份边界 X` : '保留（用于 Norm 反向）'}</td>
              </tr>
              <tr>
                <td class="font-mono tc-emerald" style="font-weight:500;">RMSNorm 1/2 归一化输出</td>
                <td class="font-mono">2×(${MBS}, ${spSeq}, ${H})</td>
                <td class="font-mono">${formatGiB(memStats.norm1PostAct + memStats.norm2PostAct)}</td>
                <td>${ckptMode === 'block' || (ckptMode === 'selective' && selectiveFlags.norm) ? '已丢弃，反向瞬时重算' : '保留（供 QKV/MLP 求 dW）'}</td>
              </tr>
              <tr>
                <td class="font-mono tc-sky" style="font-weight:500;">Attn 投影 (Q, K, V, O)</td>
                <td class="font-mono">S_local=${Math.round(topology.S_local)}, KV÷${memStats.kvTpDiv}</td>
                <td class="font-mono">${formatGiB(memStats.attnProjAct)}</td>
                <td>${ckptMode === 'block' || (ckptMode === 'selective' && selectiveFlags.attnProj) ? '已丢弃，反向重算' : '保留'}</td>
              </tr>
              <tr>
                <td class="font-mono tc-cyan" style="font-weight:500;">Core Attention (${flashAttn ? 'FP32 LSE' : 'Scores+Probs+Mask'})</td>
                <td class="font-mono">${
                  flashAttn
                    ? `(${MBS}, ${Math.ceil(n_h / topology.TP)}, ${Math.round(topology.S_local)})`
                    : `(${MBS}, ${Math.ceil(n_h / topology.TP)}, ${Math.round(topology.S_local)}, ${S})`
                }</td>
                <td class="font-mono">${formatGiB(memStats.coreAttnAct)}</td>
                <td>${
                  flashAttn
                    ? 'FlashAttn SRAM 内置重算'
                    : ckptMode === 'block' || (ckptMode === 'selective' && selectiveFlags.coreAttn)
                    ? '已重计算丢弃 S²'
                    : '完整保留 (高危 OOM)'
                }</td>
              </tr>
              <tr>
                <td class="font-mono tc-emerald" style="font-weight:500;">SwiGLU MLP (gate/up + SiLU)</td>
                <td class="font-mono">3×(${MBS}, ${Math.round(topology.S_local)}, ${Math.round(H_ff / topology.TP)})</td>
                <td class="font-mono">${formatGiB(memStats.mlpLinearAct + memStats.mlpSiluAct)}</td>
                <td>${
                  ckptMode === 'block' || (ckptMode === 'selective' && selectiveFlags.mlpFull)
                    ? '整块 MLP 丢弃并重算'
                    : ckptMode === 'selective' && selectiveFlags.swiglu
                    ? '丢弃 SiLU 乘积 (省 1/3)'
                    : '完整保留 3 份'
                }</td>
              </tr>
            </tbody>
          </table>
        </div>`;
    }

    $('right-panel').innerHTML = `
      ${warningsHtml}

      <div class="llm-card ${memStats.isOOM ? 'oom-card' : ''}">
        <div style="display:flex;flex-wrap:wrap;align-items:flex-start;justify-content:space-between;gap:12px;">
          <div>
            <div style="display:flex;flex-wrap:wrap;align-items:center;gap:8px;">
              <h2 class="card-title-lg">单卡显存峰值全景拆解（瓶颈 Stage）</h2>
              ${statusBadge}
            </div>
            <p class="text-muted" style="font-size:12px;margin-top:4px;">
              静态 ${formatGiB(memStats.staticTotalBytes)} + 前向峰值 ${formatGiB(memStats.forwardPeakActBytes)} + 反向/重算/CP 缓冲 ${formatGiB(memStats.backwardPeakExtraBytes)}${state.includeSysOverhead ? ` + 系统预留 ${formatGiB(memStats.sysOverheadBytes)}` : ''}
            </p>
          </div>
          ${recipeBtn}
        </div>

        <div class="bar-section">
          <div class="vram-line" style="left:${vramLinePct}%">
            <span class="vram-line-tag font-mono ${vramLinePct > 75 ? 'flip' : ''}">单卡上限 ${state.vramCap} GiB</span>
          </div>
          <div class="bar-track">${barSegsHtml}</div>
        </div>

        <div class="seg-grid">${segCardsHtml}</div>
      </div>

      <div class="strip-grid">
        <div class="strip-card">
          ${ICONS.scissorsLg}
          <div>
            <div class="strip-label">单卡本地序列 S_local</div>
            <div class="strip-val font-mono tc-cyan">${Math.round(topology.S_local).toLocaleString()} tokens (CP=${topology.CP})</div>
          </div>
        </div>
        <div class="strip-card">
          ${ICONS.refreshLg}
          <div>
            <div class="strip-label">单卡层激活压缩</div>
            <div class="strip-val font-mono tc-emerald">-${memStats.actReductionPct.toFixed(1)}% (${formatGiB(memStats.avgLayerActBytes)}/层)</div>
          </div>
        </div>
        <div class="strip-card">
          ${ICONS.gaugeLg}
          <div>
            <div class="strip-label">重计算算力代价 (FLOPs)</div>
            <div class="strip-val font-mono tc-amber">+${memStats.computeOverheadPct.toFixed(1)}% step compute</div>
          </div>
        </div>
      </div>

      <div class="tabs-card">
        <div class="tabs-bar">
          ${tabs.map((t) => `
            <button class="tab-btn ${state.activeTab === t.id ? 'active' : ''}" data-tab="${t.id}">
              ${t.icon}<span>${t.label}</span>
            </button>`).join('')}
        </div>
        <div class="tabs-body">${tabContentHtml}</div>
      </div>
    `;
  }

  initStaticControls();
  render();
}
