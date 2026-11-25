// 6 套交互式 GPU 原理可视化实验室（支持 Triton / CUDA 双轨视角）

function fmtNum(v) {
  if (Number.isInteger(v)) return String(v);
  return Number(v.toFixed(2)).toString();
}

export function mountVisualizer(container, problem, track = 'triton') {
  if (!container) return;
  const type = problem.vizType;
  if (type === 'grid1d') return renderGrid1DViz(container, problem, track);
  if (type === 'matmul') return renderMatmulViz(container, problem, track);
  if (type === 'transpose') return renderTransposeViz(container, problem, track);
  if (type === 'rgba') return renderRgbaViz(container, problem, track);
  if (type === 'conv1d') return renderConv1DViz(container, problem, track);
  if (type === 'rainbow') return renderRainbowViz(container, problem, track);
}

/* =========================================================
   1. 1D Block / Thread / Mask 模拟器（双轨联动）
   ========================================================= */
function renderGrid1DViz(container, problem, track) {
  const cfg = problem.vizConfig;
  const isCuda = track === 'cuda';
  let N = cfg.defaultN;
  let blockSize = cfg.defaultBlock;
  let executed = true;

  const opNames = {
    add: 'C[i] = A[i] + B[i]',
    copy: 'B[i] = A[i] (1D 展平拷贝)',
    relu: 'out[i] = max(0, x[i])',
    leaky_relu: 'out[i] = x > 0 ? x : 0.01 * x',
    reverse: '原地交换 input[i] ↔ input[N-1-i]（仅启前 N/2）'
  };

  function computeResult(idx) {
    const a = cfg.a[idx % cfg.a.length];
    if (cfg.op === 'add') {
      const b = cfg.b[idx % cfg.b.length];
      return { inText: `${fmtNum(a)}+${fmtNum(b)}`, outText: fmtNum(a + b) };
    }
    if (cfg.op === 'copy') {
      return { inText: `A[${idx}]=${fmtNum(a)}`, outText: fmtNum(a) };
    }
    if (cfg.op === 'relu') {
      return { inText: fmtNum(a), outText: fmtNum(Math.max(0, a)) };
    }
    if (cfg.op === 'leaky_relu') {
      const res = a > 0 ? a : a * (cfg.alpha ?? 0.01);
      return { inText: fmtNum(a), outText: fmtNum(res) };
    }
    if (cfg.op === 'reverse') {
      const partner = N - 1 - idx;
      const b = cfg.a[partner % cfg.a.length];
      return { inText: `[${idx}]↔[${partner}]`, outText: `${fmtNum(b)}↔${fmtNum(a)}` };
    }
    return { inText: '', outText: '' };
  }

  function draw() {
    const activeLimit = cfg.op === 'reverse' ? Math.floor(N / 2) : N;
    const numBlocks = Math.max(1, Math.ceil(activeLimit / blockSize));

    let blocksHtml = '';
    for (let pid = 0; pid < numBlocks; pid++) {
      let cellsHtml = '';
      for (let t = 0; t < blockSize; t++) {
        const offset = pid * blockSize + t;
        const valid = offset < activeLimit;
        const idxLabel = isCuda ? `tid=${t}→${offset}` : `off=${offset}`;
        if (!valid) {
          cellsHtml += `
            <div class="t-cell masked" title="${isCuda ? `idx=${offset} >= ${activeLimit}，if 分支跳过` : 'mask = False，不访问显存'}">
              <span class="t-cell-idx">${idxLabel}</span>
              <span class="t-cell-val">${isCuda ? 'SKIP' : 'MASK'}</span>
            </div>`;
        } else {
          const r = computeResult(offset);
          cellsHtml += `
            <div class="t-cell ${executed ? 'valid' : 'highlight'}" title="${isCuda ? `blockIdx.x=${pid}, threadIdx.x=${t}, idx=${offset}` : `pid=${pid}, offset=${offset}, mask=True`}">
              <span class="t-cell-idx">${idxLabel}</span>
              <span class="t-cell-val">${executed ? r.outText : r.inText}</span>
            </div>`;
        }
      }
      const blockHeader = isCuda
        ? `blockIdx.x = ${pid} · blockDim.x = ${blockSize}`
        : `pid = ${pid} · offsets [${pid * blockSize}..${pid * blockSize + blockSize - 1}]`;
      blocksHtml += `
        <div class="block-box">
          <div class="block-title" style="${isCuda ? 'color:var(--emerald)' : ''}">${blockHeader}</div>
          <div class="thread-cells" style="grid-template-columns: repeat(${Math.min(blockSize, 4)}, minmax(0, 1fr))">
            ${cellsHtml}
          </div>
        </div>`;
    }

    const extraReverseRow = cfg.op === 'reverse'
      ? `<div class="viz-formula">原数组 (N=${N}): [${Array.from({ length: N }, (_, i) => cfg.a[i % cfg.a.length]).join(', ')}] → 反转后: [${Array.from({ length: N }, (_, i) => cfg.a[(N - 1 - i) % cfg.a.length]).join(', ')}]${N % 2 === 1 ? `（中点下标 ${Math.floor(N / 2)} 自动保持不动）` : ''}</div>`
      : '';

    const launchFormula = isCuda
      ? `kernel&lt;&lt;&lt;<strong class="accent-emerald">${numBlocks}</strong>, <strong class="accent-cyan">${blockSize}</strong>&gt;&gt;&gt;（idx = blockIdx.x * ${blockSize} + threadIdx.x）`
      : `grid = (cdiv(${activeLimit}, ${blockSize}),) = <strong class="accent-emerald">(${numBlocks},)</strong>`;

    const legendText = isCuda
      ? `绿色线程：满足 <code>if (idx &lt; ${activeLimit})</code> 正常执行 ｜ 红色 SKIP：越界线程跳过访存（实战中配合 <code>float4</code> 每线程一次吃 4 个元素！）`
      : `绿色块：<code>mask = (offsets &lt; ${activeLimit})</code> 为 True（合法读写显存） ｜ 红色 MASK：越界保护，硬件自动屏蔽不访存！`;

    container.innerHTML = `
      <div class="viz-header">
        <h3>🔬 交互实验室：${isCuda ? 'CUDA 1D Grid / Thread 映射模拟器' : 'Triton 1D Block & Mask 映射模拟器'}</h3>
        <span class="badge ${isCuda ? 'badge-cuda' : 'badge-triton'}">${opNames[cfg.op]}</span>
      </div>
      <div class="viz-controls">
        <div class="viz-control-group">
          <label for="viz-n">数组长度 N:</label>
          <input id="viz-n" type="range" min="3" max="12" value="${N}" />
          <strong class="accent-cyan" style="font-family:var(--font-mono)">${N}</strong>
        </div>
        <div class="viz-control-group">
          <label for="viz-bs">${isCuda ? 'blockDim.x:' : 'BLOCK_SIZE:'}</label>
          <select id="viz-bs" class="viz-input">
            <option value="2" ${blockSize === 2 ? 'selected' : ''}>2</option>
            <option value="4" ${blockSize === 4 ? 'selected' : ''}>4</option>
            <option value="8" ${blockSize === 8 ? 'selected' : ''}>8</option>
          </select>
        </div>
        <button id="viz-toggle-exec" class="mini-btn primary-mini" style="flex:0 0 auto">
          ${executed ? '切换看输入值' : '执行 Kernel 看输出'}
        </button>
        <span style="font-size:12px;color:var(--text-secondary)">${launchFormula}</span>
      </div>
      <div class="viz-stage">
        <div class="thread-grid">${blocksHtml}</div>
        ${extraReverseRow}
        <div class="viz-formula">${legendText}</div>
      </div>
    `;

    container.querySelector('#viz-n').addEventListener('input', (e) => {
      N = Number(e.target.value);
      draw();
    });
    container.querySelector('#viz-bs').addEventListener('change', (e) => {
      blockSize = Number(e.target.value);
      draw();
    });
    container.querySelector('#viz-toggle-exec').addEventListener('click', () => {
      executed = !executed;
      draw();
    });
  }

  draw();
}

/* =========================================================
   2. Matrix Multiplication 分块与维度模拟器
   ========================================================= */
function renderMatmulViz(container, problem, track) {
  const isCuda = track === 'cuda';
  const { A, B } = problem.vizConfig;
  const M = A.length;
  const N = A[0].length;
  const K = B[0].length;
  let selR = 0;
  let selC = 1;

  const C = Array.from({ length: M }, (_, r) =>
    Array.from({ length: K }, (_, c) => {
      let s = 0;
      for (let i = 0; i < N; i++) s += A[r][i] * B[i][c];
      return s;
    })
  );

  function draw() {
    const terms = [];
    for (let i = 0; i < N; i++) {
      terms.push(`${A[selR][i]}×${B[i][selC]}`);
    }

    const renderMat = (mat, kind) => {
      const cols = mat[0].length;
      let html = `<div class="matrix-grid" style="grid-template-columns: repeat(${cols}, 44px)">`;
      mat.forEach((row, r) => {
        row.forEach((val, c) => {
          let cls = 'm-cell';
          if (kind === 'A' && r === selR) cls += ' active-row';
          if (kind === 'B' && c === selC) cls += ' active-col';
          if (kind === 'C' && r === selR && c === selC) cls += ' active-out';
          const dataAttr = kind === 'C' ? `data-r="${r}" data-c="${c}"` : '';
          html += `<button type="button" class="${cls}" ${dataAttr}>${val}</button>`;
        });
      });
      html += '</div>';
      return html;
    };

    const noteLine = isCuda
      ? `CUDA 分块提速：把 A 和 B 按 16×16 协同搬进 <code>__shared__ As/Bs</code>，两次 <code>__syncthreads()</code> 配合片上复用，全局显存流量直降 16 倍！`
      : `Triton 分块提速：一个 Block 一次加载 (BLOCK_M, BLOCK_N) 和 (BLOCK_N, BLOCK_K) 两整块，用一句 <code>acc += tl.dot(a, b, input_precision="ieee")</code> 算出整块！`;

    container.innerHTML = `
      <div class="viz-header">
        <h3>🔬 交互实验室：MatMul 维度映射（点击右侧矩阵 C 的任意元素）</h3>
        <span class="badge ${isCuda ? 'badge-cuda' : 'badge-pt'}">LeetGPU 约定：A(M×N) @ B(N×K) = C(M×K)</span>
      </div>
      <div class="viz-stage">
        <div class="matrix-wrap">
          <div class="matrix-panel">
            <div class="matrix-label">A (${M}×${N}) · 第 ${selR} 行高亮</div>
            ${renderMat(A, 'A')}
          </div>
          <div style="font-size:20px;color:var(--text-muted);font-weight:700">×</div>
          <div class="matrix-panel">
            <div class="matrix-label">B (${N}×${K}) · 第 ${selC} 列高亮</div>
            ${renderMat(B, 'B')}
          </div>
          <div style="font-size:20px;color:var(--text-muted);font-weight:700">=</div>
          <div class="matrix-panel">
            <div class="matrix-label">C (${M}×${K}) · 点击切换目标格</div>
            ${renderMat(C, 'C')}
          </div>
        </div>
        <div class="viz-formula">
          C[${selR}][${selC}] = 沿公共维 N=${N} 做点积 = ${terms.join(' + ')} = <strong class="accent-emerald">${C[selR][selC]}</strong>
          <br/>${noteLine}
        </div>
      </div>
    `;

    container.querySelectorAll('.m-cell[data-r]').forEach((btn) => {
      btn.addEventListener('click', () => {
        selR = Number(btn.dataset.r);
        selC = Number(btn.dataset.c);
        draw();
      });
    });
  }

  draw();
}

/* =========================================================
   3. Matrix Transpose 合并访存与坐标跳变模拟器
   ========================================================= */
function renderTransposeViz(container, problem, track) {
  const isCuda = track === 'cuda';
  const { rows, cols, matrix } = problem.vizConfig;
  let selR = 1;
  let selC = 2;

  const transposed = Array.from({ length: cols }, (_, c) =>
    Array.from({ length: rows }, (_, r) => matrix[r][c])
  );

  function draw() {
    const inFlat = selR * cols + selC;
    const outFlat = selC * rows + selR;

    const makeGrid = (mat, isOut) => {
      const cCount = mat[0].length;
      let html = `<div class="matrix-grid" style="grid-template-columns: repeat(${cCount}, 46px)">`;
      mat.forEach((row, r) => {
        row.forEach((val, c) => {
          const isMatch = isOut ? (r === selC && c === selR) : (r === selR && c === selC);
          const isSameRow = !isOut && r === selR;
          const isSameColOut = isOut && c === selR;
          let cls = 'm-cell';
          if (isMatch) cls += ' active-out';
          else if (isSameRow) cls += ' active-row';
          else if (isSameColOut) cls += ' active-col';
          html += `<button type="button" class="${cls}" data-kind="${isOut ? 'out' : 'in'}" data-r="${r}" data-c="${c}">${val}</button>`;
        });
      });
      html += '</div>';
      return html;
    };

    const tipLine = isCuda
      ? `CUDA 解法：先按行连续读进 <code>__shared__ float smem[32][33]</code>（+1 列消灭 Bank Conflict），<code>__syncthreads()</code> 后再按行连续写出！`
      : `Triton 解法：用 <code>tl.trans(tile)</code> 在片上翻转整块 32×32 Tile，编译器自动管好共享内存与 Bank Conflict，输出按行连续写回！`;

    container.innerHTML = `
      <div class="viz-header">
        <h3>🔬 交互实验室：Matrix Transpose 显存地址跳变与 2D 分块</h3>
        <span class="badge ${isCuda ? 'badge-cuda' : 'badge-triton'}">${isCuda ? 'smem[32][33] 消除 Bank Conflict' : 'tl.trans(tile) 消除跨步写'}</span>
      </div>
      <div class="viz-stage">
        <div class="matrix-wrap">
          <div class="matrix-panel">
            <div class="matrix-label">Input (${rows}×${cols}) · 黄色为连续同一行</div>
            ${makeGrid(matrix, false)}
          </div>
          <div style="font-size:20px;color:var(--cyan);font-weight:700">→ T →</div>
          <div class="matrix-panel">
            <div class="matrix-label">Output (${cols}×${rows}) · 紫色变成跨步同一列</div>
            ${makeGrid(transposed, true)}
          </div>
        </div>
        <div class="viz-formula">
          选中值 <strong>${matrix[selR][selC]}</strong>：
          输入 (${selR}, ${selC}) → 线性地址 <code>${selR}*${cols} + ${selC} = ${inFlat}</code>
          ｜ 输出 (${selC}, ${selR}) → 线性地址 <code>${selC}*${rows} + ${selR} = ${outFlat}</code>
          <br/>${tipLine}
        </div>
      </div>
    `;

    container.querySelectorAll('.m-cell').forEach((btn) => {
      btn.addEventListener('click', () => {
        const r = Number(btn.dataset.r);
        const c = Number(btn.dataset.c);
        if (btn.dataset.kind === 'in') {
          selR = r;
          selC = c;
        } else {
          selR = c;
          selC = r;
        }
        draw();
      });
    });
  }

  draw();
}

/* =========================================================
   4. RGBA Color Inversion 像素与位异或实验室
   ========================================================= */
function renderRgbaViz(container, problem, track) {
  const isCuda = track === 'cuda';
  const basePixels = problem.vizConfig.pixels;
  let inverted = false;
  let selectedIdx = 0;

  const toHex2 = (n) => n.toString(16).toUpperCase().padStart(2, '0');

  function draw() {
    const p = basePixels[selectedIdx];
    const [r, g, b, a] = p;
    const inv = [255 - r, 255 - g, 255 - b, a];
    const cur = inverted ? inv : p;

    const u32Orig = `0x${toHex2(a)}${toHex2(b)}${toHex2(g)}${toHex2(r)}`;
    const u32Inv = `0x${toHex2(a)}${toHex2(inv[2])}${toHex2(inv[1])}${toHex2(inv[0])}`;

    const swatches = basePixels.map((px, idx) => {
      const disp = inverted ? [255 - px[0], 255 - px[1], 255 - px[2], px[3]] : px;
      const rgbaCss = `rgba(${disp[0]}, ${disp[1]}, ${disp[2]}, ${(disp[3] / 255).toFixed(2)})`;
      return `<button type="button" class="pixel-swatch ${idx === selectedIdx ? 'selected' : ''}" data-idx="${idx}" style="background:${rgbaCss}" aria-label="像素 ${idx}"></button>`;
    }).join('');

    container.innerHTML = `
      <div class="viz-header">
        <h3>🔬 交互实验室：RGBA 小端序 ${isCuda ? 'uint4 / uint32' : 'uint32'} 异或反色</h3>
        <button id="rgba-invert-btn" class="mini-btn primary-mini" style="flex:0 0 auto">
          ${inverted ? '还原原图' : '⚡ 执行 pixels ^ 0x00FFFFFF'}
        </button>
      </div>
      <div class="viz-stage">
        <div class="rgba-layout">
          <div>
            <div class="matrix-label" style="margin-bottom:8px">4×4 RGBA 画布（点击像素）</div>
            <div class="pixel-grid-4">${swatches}</div>
          </div>
          <div class="stack-md" style="gap:10px">
            <div class="matrix-label">像素 #${selectedIdx} 内存字节序（低地址 → 高地址）：</div>
            <div class="byte-strip">
              <div class="byte-chip">R: <strong>${cur[0]}</strong> (0x${toHex2(cur[0])})</div>
              <div class="byte-chip">G: <strong>${cur[1]}</strong> (0x${toHex2(cur[1])})</div>
              <div class="byte-chip">B: <strong>${cur[2]}</strong> (0x${toHex2(cur[2])})</div>
              <div class="byte-chip alpha">A: <strong>${cur[3]}</strong> (0x${toHex2(cur[3])} · 锁定不变)</div>
            </div>
            <div class="viz-formula" style="margin-top:0">
              小端序打包 uint32 (0xAABBGGRR)：<br/>
              原值 <code>${u32Orig}</code>  ^  掩码 <code>0x00FFFFFF</code>  =  <strong class="accent-emerald">${u32Inv}</strong><br/>
              高 8 位 Alpha (<code>0x${toHex2(a)}</code> ^ <code>0x00</code>) 纹丝不动，RGB 全部变成 <code>255 - val</code>！${isCuda ? '（CUDA 用 uint4 一次反 4 个像素！）' : ''}
            </div>
          </div>
        </div>
      </div>
    `;

    container.querySelector('#rgba-invert-btn').addEventListener('click', () => {
      inverted = !inverted;
      draw();
    });
    container.querySelectorAll('.pixel-swatch').forEach((el) => {
      el.addEventListener('click', () => {
        selectedIdx = Number(el.dataset.idx);
        draw();
      });
    });
  }

  draw();
}

/* =========================================================
   5. 1D Convolution 滑动窗口步进器
   ========================================================= */
function renderConv1DViz(container, problem, track) {
  const isCuda = track === 'cuda';
  const { input, kernel } = problem.vizConfig;
  const outSize = input.length - kernel.length + 1;
  let step = 0;

  const outputs = Array.from({ length: outSize }, (_, i) => {
    let s = 0;
    for (let k = 0; k < kernel.length; k++) s += input[i + k] * kernel[k];
    return s;
  });

  function draw() {
    const terms = kernel.map((kw, k) => `${input[step + k]}×(${kw})`).join(' + ');

    const inCells = input.map((v, idx) => {
      const inWin = idx >= step && idx < step + kernel.length;
      return `<div class="t-cell ${inWin ? 'highlight' : ''}" style="min-width:46px;background:${inWin ? '' : 'var(--bg-surface)'};border-color:${inWin ? '' : 'var(--border-strong)'}">
        <span class="t-cell-idx">in[${idx}]</span>
        <span class="t-cell-val">${v}</span>
      </div>`;
    }).join('');

    const kCells = kernel.map((kw, k) => `
      <div class="t-cell" style="min-width:52px;background:var(--purple-soft);border-color:var(--purple-border);color:var(--badge-triton-text)">
        <span class="t-cell-idx">k[${k}]</span>
        <span class="t-cell-val">${kw}</span>
      </div>`).join('');

    const outCells = outputs.map((ov, i) => `
      <button type="button" class="t-cell ${i === step ? 'valid' : ''}" data-step="${i}" style="min-width:50px;cursor:pointer;background:${i === step ? '' : 'var(--bg-surface)'};border-color:${i === step ? '' : 'var(--border-strong)'}">
        <span class="t-cell-idx">out[${i}]</span>
        <span class="t-cell-val">${ov}</span>
      </button>`).join('');

    container.innerHTML = `
      <div class="viz-header">
        <h3>🔬 交互实验室：1D Valid 卷积滑动窗口步进</h3>
        <span class="badge ${isCuda ? 'badge-cuda' : 'badge-pts'}">output_size = ${input.length} - ${kernel.length} + 1 = ${outSize}</span>
      </div>
      <div class="viz-controls">
        <button id="conv-prev" class="mini-btn" ${step === 0 ? 'disabled' : ''}>← 上一格</button>
        <span style="font-family:var(--font-mono);font-size:13px">当前输出位 i = <strong class="accent-emerald">${step}</strong> / ${outSize - 1}</span>
        <button id="conv-next" class="mini-btn" ${step === outSize - 1 ? 'disabled' : ''}>下一格 →</button>
      </div>
      <div class="viz-stage stack-md" style="gap:12px">
        <div>
          <div class="matrix-label" style="margin-bottom:6px">Input 数组（蓝色为当前对齐窗口 in[${step}..${step + kernel.length - 1}]）</div>
          <div style="display:flex;gap:6px;flex-wrap:wrap">${inCells}</div>
        </div>
        <div>
          <div class="matrix-label" style="margin-bottom:6px">Kernel 卷积核（不翻转，按 k=0..${kernel.length - 1} ${isCuda ? '走 __ldg 只读缓存' : '广播加载'}）</div>
          <div style="display:flex;gap:6px;flex-wrap:wrap">${kCells}</div>
        </div>
        <div>
          <div class="matrix-label" style="margin-bottom:6px">Output 输出数组（点击任意格跳转）</div>
          <div style="display:flex;gap:6px;flex-wrap:wrap">${outCells}</div>
        </div>
        <div class="viz-formula" style="margin-top:0">
          out[${step}] = ${terms} = <strong class="accent-emerald">${outputs[step]}</strong>
        </div>
      </div>
    `;

    container.querySelector('#conv-prev').addEventListener('click', () => {
      if (step > 0) { step--; draw(); }
    });
    container.querySelector('#conv-next').addEventListener('click', () => {
      if (step < outSize - 1) { step++; draw(); }
    });
    container.querySelectorAll('[data-step]').forEach((btn) => {
      btn.addEventListener('click', () => {
        step = Number(btn.dataset.step);
        draw();
      });
    });
  }

  draw();
}

/* =========================================================
   6. Rainbow Table (FNV-1a 32-bit) 实时多轮哈希推演器
   ========================================================= */
function fnv1aOnce(val) {
  let h = 2166136261 >>> 0;
  const v = val >>> 0;
  for (let shift = 0; shift < 32; shift += 8) {
    const byte = (v >>> shift) & 0xFF;
    h = (h ^ byte) >>> 0;
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}

function renderRainbowViz(container, problem, track) {
  const isCuda = track === 'cuda';
  const trackLabel = isCuda ? 'CUDA' : 'Triton';
  let inputs = [...problem.vizConfig.inputs];
  let rounds = problem.vizConfig.defaultRounds;

  function buildRowsHtml() {
    return inputs.map((startVal, idx) => {
      let cur = startVal >>> 0;
      const trace = [];
      for (let r = 1; r <= rounds; r++) {
        cur = fnv1aOnce(cur);
        if (r <= 3 || r === rounds) {
          trace.push(`R${r}: 0x${cur.toString(16).toUpperCase().padStart(8, '0')}`);
        } else if (r === 4) {
          trace.push('...');
        }
      }
      return `
        <tr>
          <td><code>input[${idx}] = ${startVal}</code></td>
          <td style="font-family:var(--font-mono);font-size:12px;color:var(--text-secondary)">${trace.join(' → ')}</td>
          <td><strong class="accent-emerald" style="font-family:var(--font-mono)">${cur}</strong> <span style="font-size:11px;color:var(--text-muted)">(0x${cur.toString(16).toUpperCase().padStart(8, '0')})</span></td>
        </tr>`;
    }).join('');
  }

  function updateTableOnly() {
    const tbody = container.querySelector('#rb-tbody');
    const roundsLabel = container.querySelector('#rb-rounds-val');
    const colHead = container.querySelector('#rb-final-head');
    const compare = container.querySelector('#rb-compare');
    if (tbody) tbody.innerHTML = buildRowsHtml();
    if (roundsLabel) roundsLabel.textContent = `${rounds} 轮`;
    if (colHead) colHead.textContent = `第 ${rounds} 轮最终 uint32 输出`;
    if (compare) compare.innerHTML = `${trackLabel} 启动 <strong>1</strong> 次 Kernel vs PyTorch 约 <strong>${rounds * 16}</strong> 次 Kernel！`;
  }

  container.innerHTML = `
    <div class="viz-header">
      <h3>🔬 交互实验室：FNV-1a 多轮哈希寄存器融合推演</h3>
      <span class="badge ${isCuda ? 'badge-cuda' : 'badge-triton'}">32-bit FNV-1a 真实运行</span>
    </div>
    <div class="viz-controls">
      <div class="viz-control-group">
        <label for="rb-rounds">哈希轮数 R:</label>
        <input id="rb-rounds" type="range" min="1" max="20" value="${rounds}" />
        <strong id="rb-rounds-val" class="accent-cyan" style="font-family:var(--font-mono)">${rounds} 轮</strong>
      </div>
      <div class="viz-control-group">
        <label for="rb-custom">测试自定义 input[0]:</label>
        <input id="rb-custom" class="viz-input" type="number" value="${inputs[0]}" style="width:110px" />
      </div>
      <span id="rb-compare" style="font-size:12px;color:var(--emerald)">
        ${trackLabel} 启动 <strong>1</strong> 次 Kernel vs PyTorch 约 <strong>${rounds * 16}</strong> 次 Kernel！
      </span>
    </div>
    <div class="viz-stage" style="padding:0;overflow:hidden">
      <table class="data-table">
        <thead>
          <tr>
            <th>输入值</th>
            <th>逐轮片上寄存器演变（零显存回写）</th>
            <th id="rb-final-head">第 ${rounds} 轮最终 uint32 输出</th>
          </tr>
        </thead>
        <tbody id="rb-tbody">${buildRowsHtml()}</tbody>
      </table>
    </div>
  `;

  container.querySelector('#rb-rounds').addEventListener('input', (e) => {
    rounds = Number(e.target.value);
    updateTableOnly();
  });
  container.querySelector('#rb-custom').addEventListener('input', (e) => {
    const parsed = Number(e.target.value);
    inputs[0] = Number.isFinite(parsed) ? Math.trunc(parsed) : 0;
    updateTableOnly();
  });
}
