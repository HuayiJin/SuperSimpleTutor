// LeetGPU 官方前 10 题（Triton 极速轨 ⇄ CUDA C++ 硬核轨 双轨合一完整课程数据）

export const TRACK_META = {
  triton: {
    id: 'triton',
    name: 'Triton 极速轨',
    shortName: 'Triton',
    icon: '⚡',
    eyebrow: '⚡ Triton 极速轨 · 零 CUDA C++ 负担，一天速通前 10 题',
    heroTitle: '一天通关 LeetGPU 前 10 题：<br/>用最简单的 GPU Kernel 语言拿下合理高分',
    heroDesc:
      '为什么先学 <strong>Triton</strong>？纯 Python 语法、按「数据块」思考，不用手写 Shared Memory、不用管 <code>__syncthreads()</code> 和 Bank Conflict，就能跑出接近硬件极限的合理高分！刷完后再切右上角 <strong>CUDA 轨</strong>，无缝进阶底层提分。',
    totalTimeLabel: '~3.5h',
    primitivesHeading: '🔑 上手前 5 分钟：只需记住这 6 个 Triton 核心原语',
    primitives: [
      { tag: '1. 定位当前块', code: 'pid = tl.program_id(0)', desc: '每个程序实例负责一整块连续数据（如 1024 个元素），不再操心单线程 ID。' },
      { tag: '2. 生成块内偏移', code: 'offs = pid * BS + tl.arange(0, BS)', desc: '一次算出整块全局下标；2D 题只需用 [:, None] 和 [None, :] 广播。' },
      { tag: '3. 安全读写显存', code: 'tl.load(ptr + offs, mask=offs < N)', desc: '加上 mask 防尾块越界，累加/点积时记得加 other=0.0。' },
      { tag: '4. 无分支条件选择', code: 'tl.where(x > 0, x, 0.01 * x)', desc: '绝不对张量写 Python 的 if/else；用 tl.where / tl.maximum 编译为单指令。' },
      { tag: '5. 2D 矩阵神器', code: 'tl.trans(tile) / tl.dot(a, b)', desc: '一句替代 80 行手写 Shared Memory，自动处理 Bank Conflict 与 Tensor Core。' },
      { tag: '6. 向上取整启网格', code: 'grid = (triton.cdiv(N, BS),)', desc: '元组尾逗号不能省！BLOCK_SIZE 必须是 2 的幂并标为 tl.constexpr。' }
    ],
    ironRulesTitle: 'LeetGPU Triton 轨必背「三铁律」（90% 新手都踩过）',
    ironRules: [
      '<strong>原地写入输出张量</strong>：LeetGPU 评测直接检查传入的 <code>c</code> / <code>output</code> 显存，Kernel 必须用 <code>tl.store(output_ptr + ..., ...)</code> 写回，不要试图 return 新张量。',
      '<strong>第 2 题 MatMul 维度陷阱</strong>：LeetGPU 约定 <code>A=(M, N)</code>、<code>B=(N, K)</code>、<code>C=(M, K)</code>，公共维是 <strong>N</strong> 不是 K！且 <code>tl.dot</code> 加 <code>input_precision="ieee"</code> 防 TF32 误差。',
      '<strong>BLOCK_SIZE 选 2 的幂次</strong>：1D 内存题默认 <code>1024</code>、2D 转置用 <code>32</code>、MatMul 用 <code>64/32/64</code>，即可拿到极具竞争力的分数。'
    ],
    sprintGoals: {
      1: '用同一套 10 行 Triton 1D 模板吃透 pid / arange / mask / load / store，90 分钟先稳拿 40 分！',
      2: '掌握双端反转、2D 分块 tl.trans 与 uint32 指针强转，解决合并访存与带宽瓶颈。',
      3: '攻克 1D 卷积滑窗、分块 MatMul（tl.dot）以及多轮 FNV-1a 寄存器融合。'
    },
    troubleshootTitle: '🚨 Triton 轨常见报错 5 秒排查',
    troubleshoot: [
      '<strong>输出全是 0</strong>：忘记 <code>tl.store</code> 写回输出指针，或者在 Python 里把 <code>c = ...</code> 重新赋值了。',
      '<strong><code>ValueError: range must be power of 2</code></strong>：<code>BLOCK_SIZE</code> 必须是 16/32/64/256/512/1024 且标注 <code>tl.constexpr</code>。',
      '<strong>MatMul 精度误差略大（~1e-3）</strong>：Ampere+ GPU 默认用 TF32，给 <code>tl.dot</code> 加上 <code>input_precision="ieee"</code> 立刻全绿。',
      '<strong>MatMul 非方阵越界 / 结果错</strong>：LeetGPU 的形状是 <code>A(M,N) @ B(N,K) = C(M,K)</code>，把公共维 <code>N</code> 和列数 <code>K</code> 搞反了！',
      '<strong>Color Inversion 颜色全黑或透明</strong>：小端序 Alpha 在最高字节，掩码是 <code>0x00FFFFFF</code> 不是 <code>0xFFFFFF00</code>。'
    ],
    leversTitle: '📈 Triton 轨不卷极限也能拿「合理高分」的 4 个杠杆',
    levers: [
      '<strong>杠杆 1：大块连续读写</strong> — 1D 算子一律 <code>BLOCK_SIZE = 1024</code>（Rainbow Table 等高寄存器算子用 512）。',
      '<strong>杠杆 2：能展平就 1D</strong> — 如 Matrix Copy 直接按 <code>N*N</code> 一维搬运，减少索引运算。',
      '<strong>杠杆 3：向量化打包与融合</strong> — Color Inversion 用 <code>uint32</code> 一次处理 4 字节；Rainbow Table 把 R 轮全融合在片上寄存器。',
      '<strong>杠杆 4：2D 题必须分块</strong> — Transpose 用 <code>32×32 + tl.trans</code>，MatMul 用 <code>64×32×64 + tl.dot</code>，直接领先朴素实现一个数量级！'
    ]
  },
  cuda: {
    id: 'cuda',
    name: 'CUDA C++ 硬核轨',
    shortName: 'CUDA',
    icon: '🟢',
    eyebrow: '🟢 CUDA C++ 硬核轨 · 掌控底层硬件，冲刺更高百分位积分',
    heroTitle: '一天攻克 LeetGPU 前 10 题（CUDA 版）：<br/>从 1D 线程映射到 float4 与 Shared Memory 分块',
    heroDesc:
      '想彻底吃透 GPU 底层或在排行榜再冲一截？<strong>CUDA C++ 轨</strong> 把前 10 题拆成「能直接跑出高分」的实战套路：先用 <code>float4</code> 128-bit 向量化打满带宽，再用 <code>__shared__</code> 分块与 <code>+1</code> Padding 攻克转置与矩阵乘！',
    totalTimeLabel: '~4.5h',
    primitivesHeading: '🔑 上手前 5 分钟：只需掌握这 6 个 CUDA 核心套路',
    primitives: [
      { tag: '1. 全局线程索引', code: 'int idx = blockIdx.x * blockDim.x + threadIdx.x;', desc: 'CUDA 按单线程视角编程，2D 网格则同时算出 row(y) 和 col(x)。' },
      { tag: '2. 向上取整网格', code: 'int blocks = (N + threads - 1) / threads;', desc: '启动配置 kernel<<<blocks, threads>>>(...)，threads 通常取 256。' },
      { tag: '3. 128-bit 向量化', code: 'reinterpret_cast<const float4*>(A)[idx]', desc: '单线程一次搬 4 个 float（16 字节），大幅减少内存事务并拉满带宽！' },
      { tag: '4. 片上共享内存', code: '__shared__ float smem[32][33];', desc: '延迟比全局显存低 30 倍；转置加 +1 列 Padding 可彻底消灭 Bank Conflict。' },
      { tag: '5. 块内同步栅栏', code: '__syncthreads();', desc: '读写 Shared Memory 前后必须同步！且绝不能放在只有部分线程进入的 if 里。' },
      { tag: '6. LeetGPU 导出签名', code: 'extern "C" void solve(...)', desc: 'LeetGPU 评测框架通过 C ABI 动态链接 solve，漏写 extern "C" 会直接链接失败！' }
    ],
    ironRulesTitle: 'LeetGPU CUDA 轨必背「三铁律」（少踩 90% 段错误与链接坑）',
    ironRules: [
      '<strong>必须写 <code>extern "C" void solve(...)</code></strong>：C++ 默认会做 Name Mangling，去掉 <code>extern "C"</code> 评测 harness 找不到入口符号！',
      '<strong><code>__syncthreads()</code> 绝不能放在越界 <code>if (row &lt; M)</code> 内部</strong>：边界 Block 里部分线程提前 return 会导致其余线程永久死锁（Timeout）！必须所有线程先共同执行搬运与同步，最后写回时再判界。',
      '<strong><code>float4</code> 向量化一定要处理尾部余数</strong>：当 <code>N % 4 != 0</code> 时，用 <code>int vecN = N / 4</code> 处理前段，再补 <code>tail</code> 标量循环，绝不越界！'
    ],
    sprintGoals: {
      1: '掌握 CUDA 1D 启动模板 + float4 128-bit 向量化访存与尾部处理，轻松连斩 4 题拿高带宽分！',
      2: '掌握双端反转、uint32 打包位异或，以及 __shared__[32][33] 零 Bank Conflict 分块转置！',
      3: '用 __ldg 只读缓存攻克 1D 卷积，手写 16×16 Shared Memory Tiled GEMM 与 #pragma unroll 哈希融合！'
    },
    troubleshootTitle: '🚨 CUDA 轨常见报错 5 秒排查',
    troubleshoot: [
      '<strong><code>undefined symbol: solve</code></strong>：忘记在 <code>void solve(...)</code> 前加 <code>extern "C"</code>！',
      '<strong>提交超时（Timeout / 死锁）</strong>：把 <code>__syncthreads()</code> 写进了 <code>if (row &lt; M &amp;&amp; col &lt; K)</code> 分支里，导致边界线程没走到同步点。',
      '<strong><code>an illegal memory access was encountered</code></strong>：1) 漏了 <code>if (idx &lt; N)</code>；2) Matrix Copy 传了 <code>N</code> 而不是 <code>N * N</code>；3) MatMul 写反了 <code>N</code> 和 <code>K</code>。',
      '<strong><code>float4</code> 尾部几个元素错</strong>：只处理了 <code>N / 4</code> 个向量，忘了在最后补 <code>N % 4</code> 个标量尾部元素！',
      '<strong>Transpose 速度很慢</strong>：直接 <code>out[x * rows + y] = in[y * cols + x]</code> 触发非合并写，换成 <code>__shared__ smem[32][33]</code> 立提 5~10 倍！'
    ],
    leversTitle: '📈 CUDA 轨不写汇编也能拿「合理高分」的 4 个杠杆',
    levers: [
      '<strong>杠杆 1：<code>float4</code> 向量化访存</strong> — 逐元素题单线程吞 4 个 float（16B），指令数降为 1/4，显存带宽吃满！',
      '<strong>杠杆 2：<code>__shared__</code> 分块复用</strong> — MatMul 用 16×16 分块把全局显存读次数砍掉 16 倍；Transpose 用 <code>[32][33]</code> 实现读写全合并且零 Bank Conflict。',
      '<strong>杠杆 3：<code>__ldg()</code> 与 <code>#pragma unroll</code></strong> — 1D 卷积用 <code>__ldg</code> 走只读纹理缓存，Rainbow Table 展开 4 字节循环全放寄存器。',
      '<strong>杠杆 4：避免分支发散</strong> — ReLU 用 <code>fmaxf(0.0f, x)</code>，Color Inversion 用 <code>uint32_t ^ 0x00FFFFFFu</code> 彻底消灭条件跳转。'
    ]
  }
};

export const SPRINTS = [
  {
    id: 1,
    name: 'Sprint 1 · 上午 09:00–10:30',
    subtitle: '万能 1D 模板四连击（同一套骨架连刷 4 题）',
    problemIds: [1, 10, 7, 8]
  },
  {
    id: 2,
    name: 'Sprint 2 · 上午 10:45–12:30',
    subtitle: '索引变换、位运算与 2D 分块转置',
    problemIds: [6, 4, 3]
  },
  {
    id: 3,
    name: 'Sprint 3 · 下午 14:00–16:30',
    subtitle: '滑窗卷积、分块矩阵乘与寄存器融合',
    problemIds: [5, 2, 9]
  }
];

export const PROBLEMS = [
  /* ===================== #1 Vector Addition ===================== */
  {
    id: 1,
    sprint: 1,
    sprintOrder: 1,
    slug: 'vector-addition',
    title: 'Vector Addition',
    zhSubtitle: '1D 向量加法 · 万能第一课',
    difficulty: 'Easy',
    points: 10,
    estMinutes: 15,
    summary: '给定两个长度为 N 的 float32 GPU 向量 A 和 B，计算逐元素加法 C[i] = A[i] + B[i]。这是整套教程的「母模板」——后面第 7、8、10 题只需改中间 1~2 行就能直接通关！',
    vizType: 'grid1d',
    vizConfig: {
      op: 'add',
      defaultN: 10,
      defaultBlock: 4,
      a: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
      b: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    },
    trackInfo: {
      triton: {
        corePrimitives: ['tl.program_id', 'tl.arange', 'tl.load / tl.store', 'mask'],
        signature: 'def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):',
        angle: 'Triton 按「Block（数据块）」思考：每个程序实例（pid）一次搬运 BLOCK_SIZE=1024 个连续 float32，编译器自动帮你做 128-bit 向量化合并访存，10 行就能跑出接近显存带宽极限的速度。',
        gotcha: 'LeetGPU 的 solve() 是原地写入输出张量 c（不靠 return 返回结果）。不要写成 c = a + b，那只会改局部变量引用，评测读到的仍是全 0 的原始显存！',
        pointsTips: [
          '把 BLOCK_SIZE 设为 1024（2 的幂次）：满足编译要求且充分隐藏显存延迟。',
          '用 triton.cdiv(N, BLOCK_SIZE) 向上取整算 Grid 大小，绝不漏掉尾部余数元素。',
          '连续地址 + 合法 mask 即可吃满显存带宽，拿下高分段性能。'
        ],
        drill: {
          prompt: '在完整代码中补全两处核心逻辑：空 1（防越界掩码）、空 2（向上取整网格块数）。',
          blanks: [
            { id: 'BLANK_1', match: 'offsets < N', placeholder: '空 1', answers: ['offsets < N', 'offsets < n_elements', 'offsets<N'] },
            { id: 'BLANK_2', match: 'triton.cdiv(N, BLOCK_SIZE)', placeholder: '空 2', answers: ['triton.cdiv(N, BLOCK_SIZE)', 'triton.cdiv(N,BLOCK_SIZE)', '(N + BLOCK_SIZE - 1) // BLOCK_SIZE'] }
          ],
          checks: [
            { label: '包含 @triton.jit 内核与 solve 入口', regex: '@triton\\.jit[\\s\\S]*def\\s+solve' },
            { label: '计算块偏移 (program_id + arange)', regex: 'tl\\.program_id[\\s\\S]*tl\\.arange' },
            { label: '越界保护 mask (offsets < N)', regex: 'offsets\\s*<\\s*(N|n_elements)' },
            { label: '原地写出结果 (tl.store)', regex: 'tl\\.store\\(' }
          ],
          hint: '空 1 检查 offsets < N；空 2 用向上取整函数 triton.cdiv(N, BLOCK_SIZE)。'
        }
      },
      cuda: {
        corePrimitives: ['blockIdx/threadIdx', 'float4 向量化', '尾部余数处理', 'extern "C"'],
        signature: 'extern "C" void solve(const float* A, const float* B, float* C, int N)',
        angle: '想在 CUDA 轨拿到比朴素版本高一截的带宽分？用 float4 一次读写 4 个 float（128-bit LDG.E.128 指令），让单线程工作量翻 4 倍、指令发射数砍到 1/4，最后对 N % 4 的余数做标量收尾！',
        gotcha: '1) 入口必须带 extern "C"；2) 如果用 float4 向量化，一定要处理末尾 base + i（即 N 不是 4 的倍数时的剩余 1~3 个元素），否则非 4 倍数测试用例必挂！',
        pointsTips: [
          'float4 向量化让相邻线程组一次发起对齐的 128-bit 合并事务，带宽通常比标量版提升 20%~40%。',
          '线程块大小 threads = 256（8 个 Warp），兼顾高占用率与低寄存器压力。',
          '因为前段处理 vecN = N / 4，Grid 大小按 vecN 计算，末尾由最后一个合法线程顺手扫尾即可。'
        ],
        drill: {
          prompt: '在完整 CUDA 代码中补全两处核心：空 1（1D 全局线程号）、空 2（按 vecN 向上取整的网格块数）。',
          blanks: [
            { id: 'BLANK_1', match: 'blockIdx.x * blockDim.x + threadIdx.x', placeholder: '空 1', answers: ['blockIdx.x * blockDim.x + threadIdx.x', 'blockDim.x * blockIdx.x + threadIdx.x'] },
            { id: 'BLANK_2', match: '(vecN + threads - 1) / threads', placeholder: '空 2', answers: ['(vecN + threads - 1) / threads', '(vecN+threads-1)/threads'] }
          ],
          checks: [
            { label: '包含 extern "C" void solve 入口', regex: 'extern\\s+"C"\\s+void\\s+solve' },
            { label: '计算全局线程号 blockIdx.x * blockDim.x + threadIdx.x', regex: 'blockIdx\\.x[\\s\\S]*threadIdx\\.x' },
            { label: '使用 <<<blocks, threads>>> 启动 Kernel', regex: '<<<[\\s\\S]+>>>' },
            { label: '包含越界保护 if (idx < ...)', regex: 'if\\s*\\(\\s*idx\\s*<' }
          ],
          hint: '空 1 为 blockIdx.x * blockDim.x + threadIdx.x；空 2 为 (vecN + threads - 1) / threads。'
        }
      }
    },
    code: {
      triton: {
        source: `import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)

# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE=BLOCK_SIZE)`,
        annotations: [
          { line: 6, label: 'tl.constexpr 编译期常量', text: 'BLOCK_SIZE 必须标为 tl.constexpr 且是 2 的幂（如 256/512/1024），让编译器生成静态向量指令。' },
          { line: 7, label: '获取当前块编号 pid', text: '等价于 CUDA 的 blockIdx.x。每个 pid 负责连续的一整块 BLOCK_SIZE 个元素。' },
          { line: 8, label: '生成向量偏移 offsets', text: 'tl.arange(0, BLOCK_SIZE) 生成 [0..1023]，加上块起点得到本块全部全局索引。' },
          { line: 9, label: '越界保护 mask', text: '当 N 不是 1024 的整数倍时，mask=False 的位置不会访问显存，杜绝越界。' },
          { line: 18, label: '1D Grid 元组配置', text: 'grid 必须是元组 (num_blocks,)，尾逗号不能漏！triton.cdiv 自动向上取整。' }
        ]
      },
      cuda: {
        source: `#include <cuda_runtime.h>

__global__ void vector_add_vec4(const float* A, const float* B, float* C, int N) {
    int vecN = N / 4;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. 主路径：每次加载/写出 1 个 float4（4 个 float = 128-bit 合并访存）
    if (idx < vecN) {
        float4 a = reinterpret_cast<const float4*>(A)[idx];
        float4 b = reinterpret_cast<const float4*>(B)[idx];
        float4 c = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
        reinterpret_cast<float4*>(C)[idx] = c;
    }
    // 2. 尾部收尾：处理 N % 4 剩余的 0~3 个标量元素
    if (idx == 0) {
        for (int i = vecN * 4; i < N; ++i) {
            C[i] = A[i] + B[i];
        }
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    if (N <= 0) return;
    int vecN = N / 4;
    int threads = 256;
    int blocks = max(1, (vecN + threads - 1) / threads);
    vector_add_vec4<<<blocks, threads>>>(A, B, C, N);
}`,
        annotations: [
          { line: 5, label: '标准 1D 全局线程号', text: '每个线程拿到唯一编号 idx，负责数组里的第 idx 个 float4（即 4 个连续 float）。' },
          { line: 9, label: 'float4 128-bit 向量加载', text: '编译成单条 LDG.E.128 指令，减少内存请求发射数，是 CUDA 内存带宽题的王牌提速招！' },
          { line: 15, label: '零额外 Kernel 的尾部收尾', text: '最多只有 3 个尾巴元素，直接由 idx==0 线程循环处理，省去再启一个尾巴 Kernel 的开销。' },
          { line: 22, label: '必须加 extern "C"', text: 'LeetGPU 通过 C ABI 查找 solve 符号，漏写 extern "C" 会报链接找不到符号！' },
          { line: 26, label: 'max(1, ...) 防止 N < 4 时启动 0 块', text: '当 N=1,2,3 时 vecN=0，用 max(1, ...) 保证至少启 1 个块去执行尾部循环。' }
        ]
      },
      pytorch: {
        source: `import torch

# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    torch.add(a, b, out=c)`,
        annotations: [
          { line: 5, label: '用 out=c 避免临时显存分配', text: 'torch.add(..., out=c) 直接原地写出，避免 c.copy_(a + b) 造成的二次显存读写。' }
        ]
      }
    }
  },

  /* ===================== #10 Matrix Copy ===================== */
  {
    id: 10,
    sprint: 1,
    sprintOrder: 2,
    slug: 'matrix-copy',
    title: 'Matrix Copy',
    zhSubtitle: '矩阵拷贝 · 2D 降维成 1D 秒杀',
    difficulty: 'Easy',
    points: 10,
    estMinutes: 10,
    summary: '把一个 N × N 的 float32 矩阵 A 完整拷贝到矩阵 B。核心思维：行优先（Row-major）存储的 2D 连续矩阵在显存里就是一条长度为 N*N 的 1D 直线！',
    vizType: 'grid1d',
    vizConfig: {
      op: 'copy',
      defaultN: 9,
      defaultBlock: 4,
      a: [1.5, 2.0, -3.5, 4.0, 5.2, 6.1, 7.0, 8.8, 9.9, 0, 0, 0]
    },
    trackInfo: {
      triton: {
        corePrimitives: ['1D 展平思维', 'tl.load / tl.store', '显存带宽极限'],
        signature: 'def solve(a: torch.Tensor, b: torch.Tensor, N: int):',
        angle: '千万别写 2D Grid！把 N×N 矩阵当成 n_elements = N * N 的 1D 数组，直接复用第 1 题模板删掉加法，5 分钟通关且带宽利用率拉满。',
        gotcha: '入参 N 是矩阵边长，总元素数是 N * N，不是 N！如果直接传 N 进 mask，只会拷贝第一行。',
        pointsTips: [
          '1D 展平 + BLOCK_SIZE = 1024，让显存控制器跑满连续 burst 读写。',
          '纯带宽受限算子，去掉任何多余的 2D 乘法索引就是提速。'
        ],
        drill: {
          prompt: '在完整代码中补全两处：空 1（写回目标张量 b）、空 2（N×N 展平后的总元素数）。',
          blanks: [
            { id: 'BLANK_1', match: 'tl.store(b_ptr + offsets, val, mask=mask)', placeholder: '空 1', answers: ['tl.store(b_ptr + offsets, val, mask=mask)', 'tl.store(b_ptr+offsets, val, mask=mask)'] },
            { id: 'BLANK_2', match: 'N * N', placeholder: '空 2', answers: ['N * N', 'N*N', 'a.numel()'] }
          ],
          checks: [
            { label: '包含 @triton.jit 与 solve 入口', regex: '@triton\\.jit[\\s\\S]*def\\s+solve' },
            { label: '正确展平总元素数 N * N (或 numel)', regex: '(N\\s*\\*\\s*N|numel\\(\\))' },
            { label: '包含 tl.load 与 tl.store 带 mask', regex: 'tl\\.load[\\s\\S]*tl\\.store' }
          ],
          hint: '空 1 用 tl.store(b_ptr + offsets, val, mask=mask)；空 2 是 N * N（矩阵是 N×N）。'
        }
      },
      cuda: {
        corePrimitives: ['1D 展平 total = N*N', 'float4 16B 搬运', '纯显存拷贝'],
        signature: 'extern "C" void solve(const float* A, float* B, int N)',
        angle: '同样直接 1D 展平为 total = N * N！用 float4 一次拷贝 16 字节（或直接用 cudaMemcpyDeviceToDevice），既能吃满 HBM/GDDR 带宽，又比 2D dim3 写法简洁一半。',
        gotcha: '一定要算 int total = N * N（大矩阵可用 size_t 防止 N > 46340 时 int 溢出），且处理 total % 4 尾部元素！',
        pointsTips: [
          'float4 拷贝每线程搬运 16 字节，吞吐量几乎贴齐理论显存带宽上限。',
          '在 CUDA 里如果 N 很大（如 N=16384），N*N 约 2.68 亿在 int 范围内，用 size_t 更稳妥。'
        ],
        drill: {
          prompt: '在完整代码中补全两处：空 1（float4 源矩阵读取）、空 2（N×N 展平后的总元素数）。',
          blanks: [
            { id: 'BLANK_1', match: 'reinterpret_cast<const float4*>(A)[idx]', placeholder: '空 1', answers: ['reinterpret_cast<const float4*>(A)[idx]'] },
            { id: 'BLANK_2', match: '(size_t)N * N', placeholder: '空 2', answers: ['(size_t)N * N', 'N * N', 'N*N', 'static_cast<size_t>(N) * N'] }
          ],
          checks: [
            { label: '包含 extern "C" void solve 入口', regex: 'extern\\s+"C"\\s+void\\s+solve' },
            { label: '正确计算 N * N 总元素数', regex: 'N\\s*\\*\\s*N' },
            { label: '完成 A → B 的显存拷贝（Kernel 或 cudaMemcpy）', regex: '(B\\[|cudaMemcpy|float4)' }
          ],
          hint: '空 1 读源矩阵用 reinterpret_cast<const float4*>(A)[idx]；空 2 总元素数为 (size_t)N * N。'
        }
      }
    },
    code: {
      triton: {
        source: `import torch
import triton
import triton.language as tl

@triton.jit
def matrix_copy_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    val = tl.load(a_ptr + offsets, mask=mask)
    tl.store(b_ptr + offsets, val, mask=mask)

# a, b are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, N: int):
    n_elements = N * N
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    matrix_copy_kernel[grid](a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE)`,
        annotations: [
          { line: 16, label: '2D 展平为 1D', text: '行优先连续显存中，(N, N) 矩阵与长度为 N*N 的 1D 向量完全同构，展平后访存完全连续。' },
          { line: 11, label: '单次搬运 4KB 连续显存', text: '每个 Triton program 搬运 1024 个 float32，编译器自动展开成 128-bit 向量加载指令。' }
        ]
      },
      cuda: {
        source: `#include <cuda_runtime.h>

__global__ void matrix_copy_vec4(const float* A, float* B, size_t total) {
    size_t vecN = total / 4;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < vecN) {
        reinterpret_cast<float4*>(B)[idx] = reinterpret_cast<const float4*>(A)[idx];
    }
    if (idx == 0) {
        for (size_t i = vecN * 4; i < total; ++i) {
            B[i] = A[i];
        }
    }
}

extern "C" void solve(const float* A, float* B, int N) {
    size_t total = (size_t)N * N;
    if (total == 0) return;
    size_t vecN = total / 4;
    int threads = 256;
    int blocks = max(1, (int)((vecN + threads - 1) / threads));
    matrix_copy_vec4<<<blocks, threads>>>(A, B, total);
}`,
        annotations: [
          { line: 8, label: '一行完成 16 字节拷贝', text: 'float4 结构体赋值直接编译为 128-bit 读写，带宽比标量逐个拷贝显著更高。' },
          { line: 18, label: '使用 size_t 防大矩阵溢出', text: '当 N 很大时，(size_t)N * N 确保乘法不发生 32 位整数溢出。' }
        ]
      },
      pytorch: {
        source: `import torch

def solve(a: torch.Tensor, b: torch.Tensor, N: int):
    b.copy_(a)`,
        annotations: [
          { line: 4, label: '底层调用 cudaMemcpyAsync', text: 'b.copy_(a) 是原地拷贝，直接走设备内高速拷贝路径。' }
        ]
      }
    }
  },

  /* ===================== #7 ReLU Activation ===================== */
  {
    id: 7,
    sprint: 1,
    sprintOrder: 3,
    slug: 'relu',
    title: 'ReLU Activation',
    zhSubtitle: 'ReLU 激活函数 · 无分支硬件 FMAX',
    difficulty: 'Easy',
    points: 10,
    estMinutes: 10,
    summary: '对长度为 N 的 float32 向量逐元素计算 ReLU：output[i] = max(0, input[i])。继续复用 Sprint 1 模板，只换 1 行核心算子！',
    vizType: 'grid1d',
    vizConfig: {
      op: 'relu',
      defaultN: 10,
      defaultBlock: 4,
      a: [-3.2, 1.5, 0.0, -0.4, 4.8, -7.1, 2.3, -1.0, 5.5, -2.2, 1.0, -4.0]
    },
    trackInfo: {
      triton: {
        corePrimitives: ['tl.maximum', '无分支计算', '1D 模板复用'],
        signature: 'def solve(input: torch.Tensor, output: torch.Tensor, N: int):',
        angle: '在 Triton 里直接用 tl.maximum(x, 0.0)，它会编译成 GPU 单周期指令 FMAX（无 Warp 分支发散），既简洁又快。',
        gotcha: '一定写 0.0（浮点字面量）而不是整数 0，防止部分 Triton 版本触发隐式类型提升。',
        pointsTips: [
          'tl.maximum(x, 0.0) 直接映射到单条 PTX max.f32 指令。',
          'BLOCK_SIZE = 1024 足以打满显存带宽。'
        ],
        drill: {
          prompt: '在完整代码中补全 ReLU 无分支核心计算表达式（空 1）。',
          blanks: [
            { id: 'BLANK_1', match: 'tl.maximum(x, 0.0)', placeholder: '空 1', answers: ['tl.maximum(x, 0.0)', 'tl.maximum(x, 0)', 'tl.where(x > 0, x, 0.0)', 'tl.where(x > 0.0, x, 0.0)'] }
          ],
          checks: [
            { label: '包含 @triton.jit 与 solve 入口', regex: '@triton\\.jit[\\s\\S]*def\\s+solve' },
            { label: '使用无分支 tl.maximum 或 tl.where', regex: '(tl\\.maximum|tl\\.where)' },
            { label: '包含 mask 保护与 tl.store', regex: 'mask[\\s\\S]*tl\\.store' }
          ],
          hint: '推荐单周期硬件指令写法：tl.maximum(x, 0.0)（或 tl.where(x > 0, x, 0.0)）。'
        }
      },
      cuda: {
        corePrimitives: ['fmaxf(0.0f, x)', 'float4 向量化', '零分支发散'],
        signature: 'extern "C" void solve(const float* input, float* output, int N)',
        angle: 'CUDA 里用内置单精度函数 fmaxf(0.0f, x) 配合 float4 一次处理 4 个元素，4 条 FMAX 指令 + 1 次 128-bit 读写，带宽直接拉满！',
        gotcha: '别用 std::max(0.0f, x)（需额外头文件且有时不够直接）或 fmax（双精度 double 会触发隐式转型拖慢速度！）。单精度一定要带 f 后缀：fmaxf(0.0f, x)！',
        pointsTips: [
          'fmaxf 是单精度浮点硬件内建指令，0.0f 的 f 后缀也不能漏（0.0 是 double！）。',
          '叠加 float4 向量化后，ReLU 可以稳定跑在显存带宽上限附近。'
        ],
        drill: {
          prompt: '在完整代码中补全单精度无分支 ReLU 函数调用（空 1，注意防止 double 降速）。',
          blanks: [
            { id: 'BLANK_1', match: 'fmaxf(0.0f, v.x)', placeholder: '空 1', answers: ['fmaxf(0.0f, v.x)', 'fmaxf(v.x, 0.0f)'] }
          ],
          checks: [
            { label: '包含 extern "C" void solve 入口', regex: 'extern\\s+"C"\\s+void\\s+solve' },
            { label: '使用单精度 fmaxf 或三目 > 0.0f', regex: '(fmaxf|>\\s*0\\.0f)' },
            { label: '包含边界判断与 Kernel 启动', regex: '<<<[\\s\\S]+>>>' }
          ],
          hint: '单精度函数是 fmaxf(0.0f, v.x)。'
        }
      }
    },
    code: {
      triton: {
        source: `import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(input_ptr + offsets, mask=mask)
    out = tl.maximum(x, 0.0)
    tl.store(output_ptr + offsets, out, mask=mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    relu_kernel[grid](input, output, N, BLOCK_SIZE=BLOCK_SIZE)`,
        annotations: [
          { line: 12, label: '单周期 FMAX 指令', text: 'tl.maximum(x, 0.0) 在向量块上逐元素取最大值，完全避免分支发散。' },
          { line: 17, label: '与第 1 题一致的启动代码', text: '复用 1D 模板即可零负担拿下本题。' }
        ]
      },
      cuda: {
        source: `#include <cuda_runtime.h>

__global__ void relu_vec4(const float* input, float* output, int N) {
    int vecN = N / 4;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < vecN) {
        float4 v = reinterpret_cast<const float4*>(input)[idx];
        v.x = fmaxf(0.0f, v.x);
        v.y = fmaxf(0.0f, v.y);
        v.z = fmaxf(0.0f, v.z);
        v.w = fmaxf(0.0f, v.w);
        reinterpret_cast<float4*>(output)[idx] = v;
    }
    if (idx == 0) {
        for (int i = vecN * 4; i < N; ++i) {
            output[i] = fmaxf(0.0f, input[i]);
        }
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    if (N <= 0) return;
    int vecN = N / 4;
    int threads = 256;
    int blocks = max(1, (vecN + threads - 1) / threads);
    relu_vec4<<<blocks, threads>>>(input, output, N);
}`,
        annotations: [
          { line: 9, label: 'fmaxf 单精度硬件指令', text: '必须用 fmaxf 和 0.0f（而不是 fmax 和 0.0），避免隐式转成 FP64 double 导致算力暴跌！' },
          { line: 8, label: 'float4 原地修改并写回', text: '一次加载 4 个 float 到寄存器，做完 4 次 fmaxf 后单条 STG.E.128 写回。' }
        ]
      },
      pytorch: {
        source: `import torch

def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    torch.relu(input, out=output)`,
        annotations: [
          { line: 4, label: '原地写出版本', text: 'torch.relu(input, out=output) 直接写进输出显存。' }
        ]
      }
    }
  },

  /* ===================== #8 Leaky ReLU ===================== */
  {
    id: 8,
    sprint: 1,
    sprintOrder: 4,
    slug: 'leaky-relu',
    title: 'Leaky ReLU',
    zhSubtitle: '带斜率激活 · 谓词选择无分支',
    difficulty: 'Easy',
    points: 10,
    estMinutes: 10,
    summary: '实现 Leaky ReLU：当 x > 0 时输出 x，否则输出 0.01 * x。这一题学会无分支条件选择，完成后 Sprint 1 四连击直接拿下 40 分！',
    vizType: 'grid1d',
    vizConfig: {
      op: 'leaky_relu',
      alpha: 0.01,
      defaultN: 10,
      defaultBlock: 4,
      a: [-10.0, 2.5, 0.0, -5.0, 3.0, -20.0, 1.2, -1.0, 4.0, -8.0, 2.0, -3.0]
    },
    trackInfo: {
      triton: {
        corePrimitives: ['tl.where', '条件掩码选择', '标量广播'],
        signature: 'def solve(input: torch.Tensor, output: torch.Tensor, N: int):',
        angle: 'Triton 不能对向量写 Python 的 if x > 0，标准姿势是用 tl.where(x > 0, x, 0.01 * x) 做全向量谓词选择（SELP 指令），零分支发散！',
        gotcha: 'LeetGPU 的 Leaky ReLU 固定负斜率为 0.01；绝不能对 Triton 块写 Python if/else！',
        pointsTips: [
          'tl.where 编译为无分支 SELP 指令，所有线程同步推进。',
          'Sprint 1 四连击完成，已稳拿 40 分！'
        ],
        drill: {
          prompt: '在完整代码中补全 Leaky ReLU 的无分支向量条件选择（空 1，负斜率为 0.01）。',
          blanks: [
            { id: 'BLANK_1', match: 'tl.where(x > 0, x, 0.01 * x)', placeholder: '空 1', answers: ['tl.where(x > 0, x, 0.01 * x)', 'tl.where(x > 0.0, x, 0.01 * x)', 'tl.where(x > 0, x, x * 0.01)', 'tl.where(x >= 0, x, 0.01 * x)'] }
          ],
          checks: [
            { label: '包含 @triton.jit 与 solve 入口', regex: '@triton\\.jit[\\s\\S]*def\\s+solve' },
            { label: '使用 tl.where 做向量谓词选择', regex: 'tl\\.where\\(' },
            { label: '使用负斜率 0.01 并写回输出', regex: '0\\.01[\\s\\S]*tl\\.store' }
          ],
          hint: '用向量条件原语 tl.where(x > 0, x, 0.01 * x)。'
        }
      },
      cuda: {
        corePrimitives: ['三目 SELP 谓词指令', '0.01f 单精度字面量', 'float4 向量化'],
        signature: 'extern "C" void solve(const float* input, float* output, int N)',
        angle: '在 CUDA 中写短小的三目表达式 `x > 0.0f ? x : 0.01f * x`（配合 inline device helper 与 float4），NVCC 会自动编译为无分支的 SELP 谓词选择指令！',
        gotcha: '常数一定要写 0.01f（带 f）！写成 0.01 会触发 FP32→FP64→FP32 双精度转换，性能会直接掉档！',
        pointsTips: [
          '把单元素公式封装成 `__device__ __forceinline__ float lrelu(float x)`，代码整洁且零调用开销。',
          '配合 float4 向量化，与 ReLU 一样可以跑满显存带宽！'
        ],
        drill: {
          prompt: '在完整代码中补全单精度无分支 Leaky ReLU 返回表达式（空 1）。',
          blanks: [
            { id: 'BLANK_1', match: 'x > 0.0f ? x : 0.01f * x', placeholder: '空 1', answers: ['x > 0.0f ? x : 0.01f * x', 'x > 0.0f ? x : x * 0.01f', 'x >= 0.0f ? x : 0.01f * x', 'x > 0 ? x : 0.01f * x'] }
          ],
          checks: [
            { label: '包含 extern "C" void solve 入口', regex: 'extern\\s+"C"\\s+void\\s+solve' },
            { label: '包含单精度负斜率 0.01f', regex: '0\\.01f' },
            { label: '包含 Kernel 启动 <<<...>>>', regex: '<<<[\\s\\S]+>>>' }
          ],
          hint: '单精度三目运算符：x > 0.0f ? x : 0.01f * x。'
        }
      }
    },
    code: {
      triton: {
        source: `import torch
import triton
import triton.language as tl

@triton.jit
def leaky_relu_kernel(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(input_ptr + offsets, mask=mask)
    out = tl.where(x > 0, x, 0.01 * x)
    tl.store(output_ptr + offsets, out, mask=mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    leaky_relu_kernel[grid](input, output, N, BLOCK_SIZE=BLOCK_SIZE)`,
        annotations: [
          { line: 12, label: '向量三目运算 tl.where', text: 'x > 0 生成布尔掩码向量，tl.where 逐位挑选，编译成无分支 SELP 指令。' }
        ]
      },
      cuda: {
        source: `#include <cuda_runtime.h>

__device__ __forceinline__ float lrelu(float x) {
    return x > 0.0f ? x : 0.01f * x;
}

__global__ void leaky_relu_vec4(const float* input, float* output, int N) {
    int vecN = N / 4;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < vecN) {
        float4 v = reinterpret_cast<const float4*>(input)[idx];
        v.x = lrelu(v.x);
        v.y = lrelu(v.y);
        v.z = lrelu(v.z);
        v.w = lrelu(v.w);
        reinterpret_cast<float4*>(output)[idx] = v;
    }
    if (idx == 0) {
        for (int i = vecN * 4; i < N; ++i) output[i] = lrelu(input[i]);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    if (N <= 0) return;
    int vecN = N / 4;
    int threads = 256;
    int blocks = max(1, (vecN + threads - 1) / threads);
    leaky_relu_vec4<<<blocks, threads>>>(input, output, N);
}`,
        annotations: [
          { line: 4, label: '0.01f 避免双精度惩罚', text: '0.01f 保证全程在 FP32 单精度管线中运行，且三目表达式会被 NVCC 编译成无分支 SELP 指令。' },
          { line: 3, label: '__forceinline__ 零函数调用开销', text: '强制内联进 Kernel，让代码像 Python 一样清爽又不损一丝性能。' }
        ]
      },
      pytorch: {
        source: `import torch
import torch.nn.functional as F

def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    output.copy_(F.leaky_relu(input, negative_slope=0.01))`,
        annotations: [
          { line: 5, label: 'F.leaky_relu 无 out= 参数', text: 'PyTorch 多一次临时张量拷贝，这题 Triton 和 CUDA 都比 PyTorch 快！' }
        ]
      }
    }
  },

  /* ===================== #6 Reverse Array ===================== */
  {
    id: 6,
    sprint: 2,
    sprintOrder: 5,
    slug: 'reverse-array',
    title: 'Reverse Array',
    zhSubtitle: '原地反转数组 · 只启一半线程防「反转两次」',
    difficulty: 'Easy',
    points: 10,
    estMinutes: 15,
    summary: '将长度为 N 的 float32 数组在 GPU 上原地反转（input[i] 与 input[N - 1 - i] 互换）。核心在于避免并行写冲突以及避免交换两次又变回原样！',
    vizType: 'grid1d',
    vizConfig: {
      op: 'reverse',
      defaultN: 9,
      defaultBlock: 4,
      a: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    },
    trackInfo: {
      triton: {
        corePrimitives: ['双端对称索引', 'N // 2 网格裁剪', '原地无竞争交换'],
        signature: 'def solve(input: torch.Tensor, N: int):',
        angle: '只启动覆盖前半段 half = N // 2 的线程块！每个块同时读 left = offsets 与 right = N - 1 - offsets 再交叉写回。左右严格不重叠，天然零数据竞争！',
        gotcha: '1) 千万别用全长 N 做 Grid，否则线程 i 和 N-1-i 并发读写同一对元素会触发竞争错乱！\n2) N <= 1 时 half = 0，提前 return 防止启动空 Grid 报错。',
        pointsTips: [
          '只启动 N // 2 个元素，工作量减半且零数据竞争！',
          'N 为奇数时正中间元素 N // 2 自然被跳过，奇偶长度统一正确。'
        ],
        drill: {
          prompt: '在完整代码中补全两处核心：空 1（避免反转两次的半长上限）、空 2（对称右端下标）。',
          blanks: [
            { id: 'BLANK_1', match: 'N // 2', placeholder: '空 1', answers: ['N // 2', 'N//2'] },
            { id: 'BLANK_2', match: 'N - 1 - offsets', placeholder: '空 2', answers: ['N - 1 - offsets', 'N-1-offsets'] }
          ],
          checks: [
            { label: '只启动前一半 N // 2 防止并发冲突', regex: 'N\\s*//\\s*2' },
            { label: '计算对称右端下标 N - 1 - offsets', regex: 'N\\s*-\\s*1\\s*-\\s*offsets' },
            { label: '交叉写回左右两端', regex: 'tl\\.store[\\s\\S]*tl\\.store' }
          ],
          hint: '空 1 只处理前半段 N // 2；空 2 与 offsets 对称的右端下标是 N - 1 - offsets。'
        }
      },
      cuda: {
        corePrimitives: ['half = N / 2', '原地寄存器交换', '奇偶长度通用'],
        signature: 'extern "C" void solve(float* input, int N)',
        angle: 'CUDA 原地反转极简高效：只启动 half = N / 2 个线程，每个线程把 input[idx] 与 input[N - 1 - idx] 读进两个寄存器再互换写回。相邻线程访问的两端都是连续缓存行，硬件 L1/L2 自动合并！',
        gotcha: '不要写 `if (idx < N)`！必须写 `if (idx < half)` 且 Grid 只按 `half = N / 2` 启动。若 N <= 1，直接 return 避免启动 0 个 Block！',
        pointsTips: [
          '每个线程同时发出 left 和 right 两个加载请求，编译器可做指令级并行（ILP）掩盖延迟。',
          '若想更进一步：当 N 为 4 的倍数时可用 float4 两端各读 4 个并在寄存器内逆序重排 (.w, .z, .y, .x) 写回！'
        ],
        drill: {
          prompt: '在完整代码中补全两处核心：空 1（防并发冲突的一半长度）、空 2（镜像右端下标）。',
          blanks: [
            { id: 'BLANK_1', match: 'N / 2', placeholder: '空 1', answers: ['N / 2', 'N/2'] },
            { id: 'BLANK_2', match: 'N - 1 - idx', placeholder: '空 2', answers: ['N - 1 - idx', 'N-1-idx'] }
          ],
          checks: [
            { label: '包含 extern "C" void solve 入口', regex: 'extern\\s+"C"\\s+void\\s+solve' },
            { label: '仅遍历前一半 N / 2 避免重复交换', regex: 'N\\s*/\\s*2' },
            { label: '使用对称下标 N - 1 - idx 原地互换', regex: 'N\\s*-\\s*1\\s*-\\s*idx' }
          ],
          hint: '空 1 为整数除法 N / 2；空 2 镜像下标为 N - 1 - idx。'
        }
      }
    },
    code: {
      triton: {
        source: `import torch
import triton
import triton.language as tl

@triton.jit
def reverse_kernel(input_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    half = N // 2
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < half

    right_offsets = N - 1 - offsets
    left_vals = tl.load(input_ptr + offsets, mask=mask)
    right_vals = tl.load(input_ptr + right_offsets, mask=mask)

    tl.store(input_ptr + offsets, right_vals, mask=mask)
    tl.store(input_ptr + right_offsets, left_vals, mask=mask)

# input is a tensor on the GPU
def solve(input: torch.Tensor, N: int):
    half = N // 2
    if half == 0:
        return
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(half, BLOCK_SIZE),)
    reverse_kernel[grid](input, N, BLOCK_SIZE=BLOCK_SIZE)`,
        annotations: [
          { line: 8, label: '只处理前一半', text: '左右两段严格不重叠，天然零数据竞争！' },
          { line: 12, label: '镜像下标 N - 1 - offsets', text: '递减连续地址依然会被 L1/L2 Cache 合并。' },
          { line: 22, label: '防御 N <= 1', text: '避免启动 0-block Grid 报错。' }
        ]
      },
      cuda: {
        source: `#include <cuda_runtime.h>

__global__ void reverse_kernel(float* input, int N) {
    int half = N / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < half) {
        int mirror = N - 1 - idx;
        float left = input[idx];
        float right = input[mirror];
        input[idx] = right;
        input[mirror] = left;
    }
}

extern "C" void solve(float* input, int N) {
    int half = N / 2;
    if (half <= 0) return;
    int threads = 256;
    int blocks = (half + threads - 1) / threads;
    reverse_kernel<<<blocks, threads>>>(input, N);
}`,
        annotations: [
          { line: 6, label: '零竞争边界 half = N / 2', text: '左侧 [0, N/2) 与右侧 [N - N/2, N) 永不重叠，无需任何锁或同步。' },
          { line: 8, label: '同时发射两次读取', text: '先分别读入 left/right 两个寄存器再写出，提高内存级并行度（MLP）。' },
          { line: 17, label: '空数组/单元素保护', text: 'N <= 1 时 half == 0，直接返回避免 CUDA 启动 0 个 Block 报错。' }
        ]
      },
      pytorch: {
        source: `import torch

def solve(input: torch.Tensor, N: int):
    input.copy_(torch.flip(input, dims=[0]))`,
        annotations: [
          { line: 4, label: '额外显存分配开销', text: 'torch.flip 非原地，读写流量比 Triton/CUDA 原地互换翻倍。' }
        ]
      }
    }
  },

  /* ===================== #4 Color Inversion ===================== */
  {
    id: 4,
    sprint: 2,
    sprintOrder: 6,
    slug: 'color-inversion',
    title: 'Color Inversion',
    zhSubtitle: '图像反色 · uint32 指针强转 + XOR 降维打击',
    difficulty: 'Easy',
    points: 10,
    estMinutes: 20,
    summary: '给定一张 width × height 的 RGBA 图像（展平为长度为 width*height*4 的 uint8 一维数组），原地将每个像素的 R、G、B 反色为 255 - val，而 Alpha 通道保持不变！',
    vizType: 'rgba',
    vizConfig: {
      pixels: [
        [255, 60, 80, 255], [16, 185, 129, 255], [56, 189, 248, 220], [245, 158, 11, 255],
        [167, 139, 250, 200], [30, 41, 59, 255], [244, 63, 94, 180], [20, 184, 166, 255],
        [99, 102, 241, 255], [234, 179, 8, 150], [236, 72, 153, 255], [148, 163, 184, 255],
        [0, 0, 0, 255], [255, 255, 255, 128], [34, 197, 94, 255], [14, 165, 233, 255]
      ]
    },
    trackInfo: {
      triton: {
        corePrimitives: ['tl.pointer_type(tl.uint32)', '位异或 ^ 0x00FFFFFF', '4 字节向量化'],
        signature: 'def solve(image: torch.Tensor, width: int, height: int):',
        angle: '把 uint8* 指针直接强转成 uint32*——1 次加载就是 1 个完整 RGBA 像素（4 字节），小端序下是 0xAABBGGRR，直接位异或 `pixels ^ 0x00FFFFFF`，RGB 全部等价于 255-x，Alpha 异或 00 原封不动！',
        gotcha: 'NVIDIA GPU 全部是小端序（Little-Endian）：低地址 [R, G, B, A] 读成 uint32 后 A 在最高 8 位，所以保留 Alpha 的掩码是 0x00FFFFFF，绝不是 0xFFFFFF00！',
        pointsTips: [
          '用 image.to(tl.pointer_type(tl.uint32)) 吞吐量直接翻 4 倍，1 条 XOR 替代取模+比较+减法+选择！',
          'Grid 传的是 num_pixels = width * height（不是 * 4）。'
        ],
        drill: {
          prompt: '在完整代码中补全两处核心：空 1（4 字节打包指针类型）、空 2（小端序保留 Alpha 的异或掩码）。',
          blanks: [
            { id: 'BLANK_1', match: 'tl.pointer_type(tl.uint32)', placeholder: '空 1', answers: ['tl.pointer_type(tl.uint32)'] },
            { id: 'BLANK_2', match: 'pixels ^ 0x00FFFFFF', placeholder: '空 2', answers: ['pixels ^ 0x00FFFFFF', 'pixels ^ 0x00ffffff', 'pixels^0x00FFFFFF', 'pixels ^ 0xFFFFFF'] }
          ],
          checks: [
            { label: '包含 @triton.jit 与 solve(image, width, height)', regex: '@triton\\.jit[\\s\\S]*def\\s+solve' },
            { label: '强转为 uint32 指针或按像素数 width * height 调度', regex: '(pointer_type\\(tl\\.uint32\\)|width\\s*\\*\\s*height)' },
            { label: '反色 RGB 同时保留 Alpha（^ 0x00FFFFFF 或 255 -）', regex: '(0x00?[fF]{6}|255\\s*-)' }
          ],
          hint: '空 1 为 tl.pointer_type(tl.uint32)；空 2 为 pixels ^ 0x00FFFFFF（小端序高 8 位是 Alpha）。'
        }
      },
      cuda: {
        corePrimitives: ['reinterpret_cast<uint32_t*>', 'uint4 一次反 4 像素', '^= 0x00FFFFFFu'],
        signature: 'extern "C" void solve(unsigned char* image, int width, int height)',
        angle: 'CUDA 直接用 `reinterpret_cast<uint4*>(image)`——更狠！一个 uint4 是 16 字节 = 整整 4 个 RGBA 像素，4 条 `^= 0x00FFFFFFu` 指令一次反转 4 个像素，128-bit 满带宽秒杀逐字节写法！',
        gotcha: '1) 同样注意小端序掩码是 `0x00FFFFFFu`；2) 按 uint4 处理时记得用尾部循环把剩余 `num_pixels % 4` 个像素按 `uint32_t` 扫尾！',
        pointsTips: [
          'uint4（16B = 4 像素）比单 uint32（4B）吞吐再上一个台阶，且依然只多 4 行扫尾代码！',
          '零分支、零取模、零减法，纯位运算直接贴近 GPU 理论内存带宽天花板。'
        ],
        drill: {
          prompt: '在完整代码中补全两处核心：空 1（小端序保留 Alpha 的位掩码）、空 2（一次读取 4 个像素的指针强转）。',
          blanks: [
            { id: 'BLANK_1', match: '0x00FFFFFFu', placeholder: '空 1', answers: ['0x00FFFFFFu', '0x00FFFFFF', '0x00ffffffu', '0x00ffffff'] },
            { id: 'BLANK_2', match: 'reinterpret_cast<uint4*>(image)[idx]', placeholder: '空 2', answers: ['reinterpret_cast<uint4*>(image)[idx]'] }
          ],
          checks: [
            { label: '包含 extern "C" void solve 入口', regex: 'extern\\s+"C"\\s+void\\s+solve' },
            { label: '反色 RGB 且保持 Alpha 不变', regex: '(0x00?[fF]{6}u?|255\\s*-)' },
            { label: '使用 width * height 计算像素规模', regex: 'width\\s*\\*\\s*height' }
          ],
          hint: '空 1 异或掩码为 0x00FFFFFFu；空 2 为 reinterpret_cast<uint4*>(image)[idx]。'
        }
      }
    },
    code: {
      triton: {
        source: `import torch
import triton
import triton.language as tl

@triton.jit
def invert_kernel(image_ptr, num_pixels, BLOCK_SIZE: tl.constexpr):
    # 将 uint8* 强转为 uint32*，每个元素正好是 1 个 RGBA 像素（4 字节）
    pixels_ptr = image_ptr.to(tl.pointer_type(tl.uint32))
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_pixels

    pixels = tl.load(pixels_ptr + offsets, mask=mask)
    # 小端序中 RGBA = 0xAABBGGRR，异或 0x00FFFFFF 反色 RGB，A 保持不变
    inverted = pixels ^ 0x00FFFFFF
    tl.store(pixels_ptr + offsets, inverted, mask=mask)

# image is a 1D uint8 tensor on the GPU of length width * height * 4
def solve(image: torch.Tensor, width: int, height: int):
    num_pixels = width * height
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(num_pixels, BLOCK_SIZE),)
    invert_kernel[grid](image, num_pixels, BLOCK_SIZE=BLOCK_SIZE)`,
        annotations: [
          { line: 8, label: '4 字节打包强转', text: '把 uint8* 视作 uint32* 后，Grid 元素数缩小 4 倍，吞吐翻倍！' },
          { line: 15, label: '神技：x ^ 0xFF === 255 - x', text: '对 8 位无符号整数按位取反严格等于 255 - x；最高字节 Alpha 异或 0x00 保持不变！' },
          { line: 20, label: '传像素数而非字节数', text: 'num_pixels = width * height（每个 uint32 已经包含 4 字节）。' }
        ]
      },
      cuda: {
        source: `#include <cuda_runtime.h>
#include <stdint.h>

__global__ void invert_vec4_kernel(unsigned char* image, int num_pixels) {
    int vecN = num_pixels / 4; // 每个 uint4 = 4 个 RGBA 像素 = 16 字节
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t MASK = 0x00FFFFFFu;

    if (idx < vecN) {
        uint4 p = reinterpret_cast<uint4*>(image)[idx];
        p.x ^= MASK;
        p.y ^= MASK;
        p.z ^= MASK;
        p.w ^= MASK;
        reinterpret_cast<uint4*>(image)[idx] = p;
    }
    if (idx == 0) {
        uint32_t* p32 = reinterpret_cast<uint32_t*>(image);
        for (int i = vecN * 4; i < num_pixels; ++i) p32[i] ^= MASK;
    }
}

extern "C" void solve(unsigned char* image, int width, int height) {
    int num_pixels = width * height;
    if (num_pixels <= 0) return;
    int vecN = num_pixels / 4;
    int threads = 256;
    int blocks = max(1, (vecN + threads - 1) / threads);
    invert_vec4_kernel<<<blocks, threads>>>(image, num_pixels);
}`,
        annotations: [
          { line: 5, label: 'uint4 一次处理 4 个像素', text: '1 个 uint4 = 4 个 uint32 = 16 字节，完美契合 128-bit 内存总线宽度！' },
          { line: 7, label: '小端序掩码 0x00FFFFFFu', text: '内存 [R, G, B, A] 读成 uint32 是 0xAABBGGRR，异或 0x00FFFFFFu 翻转 RGB 锁定 Alpha。' },
          { line: 17, label: '像素尾部扫尾', text: '处理 num_pixels % 4 剩余的 0~3 个像素，任意宽高图片全部正确！' }
        ]
      },
      pytorch: {
        source: `import torch

def solve(image: torch.Tensor, width: int, height: int):
    pixels = image.view(-1, 4)
    pixels[:, :3].bitwise_not_()`,
        annotations: [
          { line: 5, label: '非连续切片较慢', text: 'pixels[:, :3] 非连续，性能远不如 Triton/CUDA 的 uint32 XOR。' }
        ]
      }
    }
  },

  /* ===================== #3 Matrix Transpose ===================== */
  {
    id: 3,
    sprint: 2,
    sprintOrder: 7,
    slug: 'matrix-transpose',
    title: 'Matrix Transpose',
    zhSubtitle: '2D 矩阵转置 · 片上分块消除跨步写',
    difficulty: 'Easy',
    points: 10,
    estMinutes: 25,
    summary: '将 rows × cols 的行优先矩阵转置为 cols × rows 矩阵（output[c][r] = input[r][c]）。这是从 1D 进阶到 2D 分块编程（Tiling）的第一座里程碑！',
    vizType: 'transpose',
    vizConfig: {
      rows: 3,
      cols: 4,
      matrix: [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
      ]
    },
    trackInfo: {
      triton: {
        corePrimitives: ['2D Grid', '[:, None] 广播构造 2D 块', 'tl.trans', '合并访存'],
        signature: 'def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):',
        angle: '直接 1D 写会导致输出按列大跨步（Stride = rows），带宽暴跌 10 倍！用 Triton 一次读入 32×32 Tile，片上调用 `tl.trans(tile)` 翻好，再按行连续写回——编译器自动搞定 Shared Memory 与 Bank Conflict！',
        gotcha: '2D 偏移的广播维度别写反：输入是 `offs_r[:, None] * cols + offs_c[None, :]`，写回输出是 `offs_c[:, None] * rows + offs_r[None, :]`！',
        pointsTips: [
          '用 2D Tile + tl.trans(tile) 让读写双方都是连续合并访存。',
          'BLOCK 选 32 或 64，刚好贴合 128B 缓存行。'
        ],
        drill: {
          prompt: '在完整代码中补全 2D 分块转置两处关键：空 1（片上块转置）、空 2（转置后的输出 2D 偏移）。',
          blanks: [
            { id: 'BLANK_1', match: 'tl.trans(tile)', placeholder: '空 1', answers: ['tl.trans(tile)'] },
            { id: 'BLANK_2', match: 'offs_c[:, None] * rows + offs_r[None, :]', placeholder: '空 2', answers: ['offs_c[:, None] * rows + offs_r[None, :]', 'offs_c[:, None]*rows + offs_r[None, :]'] }
          ],
          checks: [
            { label: '使用 2D program_id (axis=0 和 axis=1)', regex: 'program_id\\((axis=)?0\\)[\\s\\S]*program_id\\((axis=)?1\\)' },
            { label: '调用片上转置 tl.trans', regex: 'tl\\.trans\\(' },
            { label: '正确写出转置后地址 (* rows +)', regex: '\\*\\s*rows\\s*\\+' }
          ],
          hint: '空 1 片上转置用 tl.trans(tile)；空 2 转置后每行长为 rows，偏移为 offs_c[:, None] * rows + offs_r[None, :]。'
        }
      },
      cuda: {
        corePrimitives: ['__shared__ smem[32][33]', '消除 Bank Conflict', '__syncthreads()', '2D dim3'],
        signature: 'extern "C" void solve(const float* input, float* output, int rows, int cols)',
        angle: 'CUDA 面试与竞赛最经典的 Shared Memory 范例！把 32×32 块连续读进 `__shared__ float smem[32][33]`（+1 列 Padding 错开 32 个 Bank，彻底消除 Bank Conflict！），`__syncthreads()` 后，把线程的 x/y 角色对调按行连续写出到 `output[oy * rows + ox]`！',
        gotcha: '1) `__syncthreads()` 必须位于所有边界 `if` 之外，保证 32×32=1024 个线程全部同步；2) 写回时块坐标要翻转：`ox = blockIdx.y * TILE + threadIdx.x`、`oy = blockIdx.x * TILE + threadIdx.y`，但 `threadIdx.x` 依然沿行连续移动，所以写全局显存 100% 合并！',
        pointsTips: [
          '关键提速细节就是 `[TILE][TILE + 1]` 那多出来的 `+ 1`：让同一列的 32 个元素落入 32 个不同 Shared Memory Bank，零冲突！',
          '读全局显存按 `threadIdx.x` 连续，写全局显存也按 `threadIdx.x` 连续，带宽直冲朴素版的 6~10 倍！'
        ],
        drill: {
          prompt: '在完整代码中补全两处核心：空 1（消除 Bank Conflict 的共享内存第二维）、空 2（块内线程同步）。',
          blanks: [
            { id: 'BLANK_1', match: 'TILE + 1', placeholder: '空 1', answers: ['TILE + 1', 'TILE+1', '33'] },
            { id: 'BLANK_2', match: '__syncthreads()', placeholder: '空 2', answers: ['__syncthreads()', '__syncthreads'] }
          ],
          checks: [
            { label: '包含 extern "C" void solve 入口', regex: 'extern\\s+"C"\\s+void\\s+solve' },
            { label: '声明 __shared__ 分块并加 +1 列 Padding', regex: '__shared__[\\s\\S]*(\\+\\s*1|33)' },
            { label: '调用 __syncthreads() 同步', regex: '__syncthreads\\s*\\(\\s*\\)' }
          ],
          hint: '空 1 第二维开成 TILE + 1（即 33）消除 Bank Conflict；空 2 同步原语为 __syncthreads()。'
        }
      }
    },
    code: {
      triton: {
        source: `import torch
import triton
import triton.language as tl

@triton.jit
def transpose_kernel(input_ptr, output_ptr, rows, cols, BLOCK: tl.constexpr):
    pid_r = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)

    offs_r = pid_r * BLOCK + tl.arange(0, BLOCK)
    offs_c = pid_c * BLOCK + tl.arange(0, BLOCK)

    in_ptrs = input_ptr + offs_r[:, None] * cols + offs_c[None, :]
    in_mask = (offs_r[:, None] < rows) & (offs_c[None, :] < cols)
    tile = tl.load(in_ptrs, mask=in_mask)

    tile_t = tl.trans(tile)

    out_ptrs = output_ptr + offs_c[:, None] * rows + offs_r[None, :]
    out_mask = (offs_c[:, None] < cols) & (offs_r[None, :] < rows)
    tl.store(out_ptrs, tile_t, mask=out_mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    BLOCK = 32
    grid = (triton.cdiv(rows, BLOCK), triton.cdiv(cols, BLOCK))
    transpose_kernel[grid](input, output, rows, cols, BLOCK=BLOCK)`,
        annotations: [
          { line: 7, label: '2D Grid 双轴编号', text: 'axis=0 管行块，axis=1 管列块。' },
          { line: 13, label: 'NumPy 风格广播构造 2D 指针', text: 'offs_r[:, None] 和 offs_c[None, :] 广播后一次得到 32×32 子块地址。' },
          { line: 17, label: '一行 tl.trans 省掉 20 行 CUDA', text: 'Triton 编译器自动申请 Shared Memory、自动加 Padding 躲 Bank Conflict 并同步！' },
          { line: 19, label: '输出矩阵步长为 rows', text: '输出形状为 (cols, rows)，行号 offs_c 乘以 rows 再加列号 offs_r，写入完全合并。' }
        ]
      },
      cuda: {
        source: `#include <cuda_runtime.h>
#define TILE 32

__global__ void transpose_shared_kernel(const float* in, float* out, int rows, int cols) {
    // +1 列 Padding：让同一列元素分散到 32 个不同 Bank，彻底消灭 Bank Conflict！
    __shared__ float smem[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (y < rows && x < cols) {
        smem[threadIdx.y][threadIdx.x] = in[y * cols + x];
    }
    __syncthreads(); // 必须在 if 外面，确保全部线程同步

    // 翻转块坐标，但保持 threadIdx.x 沿输出行连续写出（100% 合并写）
    int ox = blockIdx.y * TILE + threadIdx.x;
    int oy = blockIdx.x * TILE + threadIdx.y;
    if (oy < cols && ox < rows) {
        out[oy * rows + ox] = smem[threadIdx.x][threadIdx.y];
    }
}

extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 block(TILE, TILE);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
    transpose_shared_kernel<<<grid, block>>>(input, output, rows, cols);
}`,
        annotations: [
          { line: 6, label: '灵魂代码：TILE + 1', text: 'Shared Memory 有 32 个 Bank；若开 [32][32]，读同一列 smem[threadIdx.x][threadIdx.y] 会引发 32-way Bank Conflict！加 +1 后冲突归零！' },
          { line: 13, label: '无条件 __syncthreads()', text: '等块内 1024 个线程把输入全搬进 smem 后，再开始转置读取，绝不能放进 if 里。' },
          { line: 16, label: '合并写回技巧', text: '同一 Warp 里 threadIdx.x 连续变化，所以写入 out[oy * rows + ox] 地址完全连续，吃满显存写带宽！' }
        ]
      },
      pytorch: {
        source: `import torch

def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    output.copy_(input.view(rows, cols).T)`,
        annotations: [
          { line: 4, label: 'copy_ 触发转置内核', text: '.T 仅改 stride，copy_ 触发底层分块转置。' }
        ]
      }
    }
  },

  /* ===================== #5 1D Convolution ===================== */
  {
    id: 5,
    sprint: 3,
    sprintOrder: 8,
    slug: '1d-convolution',
    title: '1D Convolution',
    zhSubtitle: '一维有效卷积 · 累加器与只读缓存复用',
    difficulty: 'Easy',
    points: 15,
    estMinutes: 25,
    summary: '给定长度为 input_size 的 1D 输入和长度为 kernel_size 的卷积核，计算 Valid（无填充）1D 卷积：output[i] = Σ input[i + k] * kernel[k]，输出长度为 input_size - kernel_size + 1。',
    vizType: 'conv1d',
    vizConfig: {
      input: [1, 3, 2, 5, 4, 1, 6, 2],
      kernel: [1, 0, -1]
    },
    trackInfo: {
      triton: {
        corePrimitives: ['tl.zeros 累加器', 'for k in range(kernel_size)', 'Valid 边界公式'],
        signature: 'def solve(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor, input_size: int, kernel_size: int):',
        angle: '用 `acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)` 在寄存器建累加器，沿 `k` 循环加载连续块 `input[offsets + k]` 乘以标量 `kernel[k]`，权重走常量缓存、输入高度命中 L1 Cache！',
        gotcha: '1) 深度学习语境的卷积是互相关 `input[i + k] * kernel[k]`，不需要翻转 kernel；2) 输出长度一定是 `input_size - kernel_size + 1`！',
        pointsTips: [
          '标量 `tl.load(kernel_ptr + k)` 会被广播并命中 GPU 只读缓存，极快！',
          'BLOCK_SIZE = 1024 让相邻输出共享绝大部分 input 缓存行。'
        ],
        drill: {
          prompt: '在完整代码中补全两处：空 1（滑窗内层乘加项）、空 2（Valid 卷积输出长度）。',
          blanks: [
            { id: 'BLANK_1', match: 'x_val * k_val', placeholder: '空 1', answers: ['x_val * k_val', 'k_val * x_val'] },
            { id: 'BLANK_2', match: 'input_size - kernel_size + 1', placeholder: '空 2', answers: ['input_size - kernel_size + 1', 'input_size-kernel_size+1'] }
          ],
          checks: [
            { label: '包含寄存器累加器 tl.zeros', regex: 'tl\\.zeros\\(' },
            { label: '沿 kernel_size 循环累加 offsets + k', regex: 'range\\(kernel_size\\)[\\s\\S]*offsets\\s*\\+\\s*k' },
            { label: '正确计算 Valid 输出长度 input_size - kernel_size + 1', regex: 'input_size\\s*-\\s*kernel_size\\s*\\+\\s*1' }
          ],
          hint: '空 1 每步累加 x_val * k_val；空 2 Valid 输出长度为 input_size - kernel_size + 1。'
        }
      },
      cuda: {
        corePrimitives: ['__ldg() 只读纹理缓存', 'fmaf 融合乘加', '#pragma unroll 4'],
        signature: 'extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size)',
        angle: '因为 LeetGPU 的 kernel_size 可能很大（静态 Shared Memory 容易超容），最稳且高分的 CUDA 写法是用 `__ldg()` 显式走 GPU 只读缓存（Read-Only Data Cache / Texture Cache）+ `#pragma unroll 4` 循环展开 + `fmaf` 融合乘加！',
        gotcha: '1) 记得先算 `int output_size = input_size - kernel_size + 1;` 并判 `if (output_size <= 0) return;`；2) `input` 和 `kernel` 加上 `__restrict__` 告诉编译器指针无别名，允许激进缓存优化！',
        pointsTips: [
          '加 `const float* __restrict__` + `__ldg()` 让滑窗重复读取稳定命中只读缓存，不占写缓存带宽。',
          '用 `fmaf(a, b, acc)` 或 `acc += a * b` 编译为单周期 FFMA 指令，精度高且吞吐翻倍。'
        ],
        drill: {
          prompt: '在完整代码中补全两处：空 1（只读缓存融合乘加语句）、空 2（Valid 卷积输出长度）。',
          blanks: [
            { id: 'BLANK_1', match: 'fmaf(__ldg(&input[idx + k]), __ldg(&kernel[k]), acc)', placeholder: '空 1', answers: ['fmaf(__ldg(&input[idx + k]), __ldg(&kernel[k]), acc)', 'acc + __ldg(&input[idx + k]) * __ldg(&kernel[k])', 'acc + input[idx + k] * kernel[k]'] },
            { id: 'BLANK_2', match: 'input_size - kernel_size + 1', placeholder: '空 2', answers: ['input_size - kernel_size + 1', 'input_size-kernel_size+1'] }
          ],
          checks: [
            { label: '包含 extern "C" void solve 入口', regex: 'extern\\s+"C"\\s+void\\s+solve' },
            { label: '计算 Valid 输出长度 input_size - kernel_size + 1', regex: 'input_size\\s*-\\s*kernel_size\\s*\\+\\s*1' },
            { label: '循环 kernel_size 累加 input[idx + k] * kernel[k]', regex: 'input\\[idx\\s*\\+\\s*k\\]' }
          ],
          hint: '空 1 为 fmaf(__ldg(&input[idx + k]), __ldg(&kernel[k]), acc)（或 acc + input[idx + k] * kernel[k]）；空 2 为 input_size - kernel_size + 1。'
        }
      }
    },
    code: {
      triton: {
        source: `import torch
import triton
import triton.language as tl

@triton.jit
def conv1d_kernel(
    input_ptr, kernel_ptr, output_ptr,
    output_size, kernel_size,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_size

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for k in range(kernel_size):
        x_val = tl.load(input_ptr + offsets + k, mask=mask, other=0.0)
        k_val = tl.load(kernel_ptr + k)
        acc += x_val * k_val

    tl.store(output_ptr + offsets, acc, mask=mask)

# input, kernel, output are tensors on the GPU
def solve(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor, input_size: int, kernel_size: int):
    output_size = input_size - kernel_size + 1
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(output_size, BLOCK_SIZE),)
    conv1d_kernel[grid](input, kernel, output, output_size, kernel_size, BLOCK_SIZE=BLOCK_SIZE)`,
        annotations: [
          { line: 15, label: '寄存器累加器 tl.zeros', text: '把累加过程完全留在片上寄存器中，滑完整个卷积核再写回一次显存。' },
          { line: 17, label: '为什么 mask=offsets<output_size 就安全？', text: '当 offsets <= output_size - 1 且 k <= kernel_size - 1 时，offsets + k <= input_size - 1，绝不越界！' },
          { line: 18, label: '标量权重广播', text: 'tl.load(kernel_ptr + k) 自动广播乘到整块 1024 元素，命中常量缓存。' }
        ]
      },
      cuda: {
        source: `#include <cuda_runtime.h>

__global__ void conv1d_ldg_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int output_size,
    int kernel_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_size) return;

    float acc = 0.0f;
    #pragma unroll 4
    for (int k = 0; k < kernel_size; ++k) {
        acc = fmaf(__ldg(&input[idx + k]), __ldg(&kernel[k]), acc);
    }
    output[idx] = acc;
}

extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    if (output_size <= 0) return;
    int threads = 256;
    int blocks = (output_size + threads - 1) / threads;
    conv1d_ldg_kernel<<<blocks, threads>>>(input, kernel, output, output_size, kernel_size);
}`,
        annotations: [
          { line: 4, label: '__restrict__ 消除指针别名', text: '向编译器承诺 input/kernel/output 内存互不重叠，解锁只读缓存与指令重排优化。' },
          { line: 14, label: '#pragma unroll 4 展开循环', text: '减少循环分支指令并让多个 __ldg 内存加载重叠发射，隐藏访存延迟。' },
          { line: 16, label: '__ldg + fmaf 融合乘加', text: '__ldg 显式走只读缓存（对任意 kernel_size 都安全不爆共享内存），fmaf 编译成单周期 FFMA 指令！' }
        ]
      },
      pytorch: {
        source: `import torch
import torch.nn.functional as F

def solve(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor, input_size: int, kernel_size: int):
    res = F.conv1d(input.view(1, 1, -1), kernel.view(1, 1, -1))
    output.copy_(res.view(-1))`,
        annotations: [
          { line: 5, label: '需先 view 成 3D (N, C, L)', text: 'PyTorch conv1d 强制 3D 输入，算出后再拍平写回 output。' }
        ]
      }
    }
  },

  /* ===================== #2 Matrix Multiplication ===================== */
  {
    id: 2,
    sprint: 3,
    sprintOrder: 9,
    slug: 'matrix-multiplication',
    title: 'Matrix Multiplication',
    zhSubtitle: '矩阵乘法 GEMM · 分块复用决胜题',
    difficulty: 'Easy',
    points: 20,
    estMinutes: 35,
    summary: '计算矩阵乘积 C = A × B。这是前 10 题含金量最高、最能拉开排行榜差距的压轴题！无论 Triton 还是 CUDA，核心都是「分块复用（Tiling）」。',
    vizType: 'matmul',
    vizConfig: {
      M: 3,
      N: 3,
      K: 3,
      A: [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
      ],
      B: [
        [2, 0, 1],
        [1, 3, 2],
        [0, 1, 4]
      ]
    },
    trackInfo: {
      triton: {
        corePrimitives: ['2D 分块循环', 'tl.dot(a, b)', '维度陷阱 M×N @ N×K'],
        signature: 'def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):',
        angle: '沿公共维 N 每次切一块 BLOCK_N=32，用一句 `acc += tl.dot(a, b, input_precision="ieee")` 累加——30 行 Python 自动生成 Shared Memory 预取与硬件矩阵乘指令，直接吊打朴素 CUDA 十几倍！',
        gotcha: '🚨 超级大坑：LeetGPU 的矩阵乘维度定义是 A=(M × N)、B=(N × K)、C=(M × K)！公共缩并维是 N（不是教科书里的 K）！另外 tl.dot 务必加 `input_precision="ieee"` 防 TF32 精度误差！',
        pointsTips: [
          '分块大小选 BLOCK_M=64, BLOCK_K=64, BLOCK_N=32（全 >= 16 满足 tl.dot 硬件要求）。',
          '加载时越界位置必须填 other=0.0，否则垃圾值会污染整个输出块！'
        ],
        drill: {
          prompt: '避开 LeetGPU 维度大坑（A: M×N, B: N×K, C: M×K），补全两处核心：空 1（沿公共维分块循环）、空 2（带精度保护的块乘）。',
          blanks: [
            { id: 'BLANK_1', match: 'range(0, N, BLOCK_N)', placeholder: '空 1', answers: ['range(0, N, BLOCK_N)', 'range(0,N,BLOCK_N)'] },
            { id: 'BLANK_2', match: 'tl.dot(a, b, input_precision="ieee")', placeholder: '空 2', answers: ['tl.dot(a, b, input_precision="ieee")', "tl.dot(a, b, input_precision='ieee')", 'tl.dot(a, b)'] }
          ],
          checks: [
            { label: '沿公共维 N 分块循环', regex: 'range\\(\\s*0\\s*,\\s*N' },
            { label: '加载越界补零 other=0.0', regex: 'other\\s*=\\s*0(\\.0)?' },
            { label: '使用 tl.dot 做块矩阵乘', regex: 'tl\\.dot\\(' }
          ],
          hint: '空 1 公共维是 N，写 range(0, N, BLOCK_N)；空 2 用 tl.dot(a, b, input_precision="ieee")。'
        }
      },
      cuda: {
        corePrimitives: ['16×16 Shared Memory Tiling', '双重 __syncthreads()', '维度陷阱 M×N @ N×K'],
        signature: 'extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K)',
        angle: '朴素三重循环 CUDA 每次乘加都要读全局显存，慢 10 倍以上！升级为标准 **16×16 Shared Memory Tiled GEMM**：整个 Block 协同把 A 的 16×16 块和 B 的 16×16 块搬进片上 `As`/`Bs`，在片上做完 16 次复用再滑向下一块，全局显存流量瞬间砍成 1/16！',
        gotcha: '🚨 两大死穴：\n1) LeetGPU 形状是 A(M×N) @ B(N×K) = C(M×K)，公共维是 N，所以外层 tile 循环上界是 `(N + TILE - 1) / TILE`，A 的行步长是 N，B 和 C 的行步长都是 K！\n2) 两次 `__syncthreads()` 都不能放在 `if (row < M && col < K)` 里！越界线程必须把 `As/Bs` 填成 `0.0f` 并照常参与同步，最后写回 C 时再判界！',
        pointsTips: [
          '每块 16×16（或 32×32）Shared Memory 分块即可在 LeetGPU 上直接超越绝大多数朴素 CUDA 提交！',
          '内层 16 次点积加 `#pragma unroll` 全部留在寄存器里跑 FFMA，算术强度（Arithmetic Intensity）暴涨 16 倍！'
        ],
        drill: {
          prompt: '在完整 16×16 Shared Memory GEMM 中补全两处核心：空 1（公共维分块数）、空 2（协同加载后的块内同步）。',
          blanks: [
            { id: 'BLANK_1', match: '(N + TILE - 1) / TILE', placeholder: '空 1', answers: ['(N + TILE - 1) / TILE', '(N+TILE-1)/TILE'] },
            { id: 'BLANK_2', match: '__syncthreads()', placeholder: '空 2', answers: ['__syncthreads()', '__syncthreads'] }
          ],
          checks: [
            { label: '包含 extern "C" void solve 入口', regex: 'extern\\s+"C"\\s+void\\s+solve' },
            { label: '声明 __shared__ 子块 As / Bs', regex: '__shared__' },
            { label: '沿公共维 N 分块并调用 __syncthreads()', regex: '__syncthreads\\s*\\(\\s*\\)' },
            { label: '按行步长 K 写回 C[row * K + col]', regex: 'row\\s*\\*\\s*K\\s*\\+\\s*col' }
          ],
          hint: '空 1 为 (N + TILE - 1) / TILE（公共维是 N）；空 2 同步原语为 __syncthreads()。'
        }
      }
    },
    code: {
      triton: {
        source: `import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # 注意 LeetGPU 维度约定：A=(M, N), B=(N, K), C=(M, K)
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        a_ptrs = a_ptr + offs_m[:, None] * N + offs_n[None, :]
        b_ptrs = b_ptr + offs_n[:, None] * K + offs_k[None, :]

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")

    c_ptrs = c_ptr + offs_m[:, None] * K + offs_k[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(c_ptrs, acc, mask=c_mask)

# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 32, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
    matmul_kernel[grid](a, b, c, M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)`,
        annotations: [
          { line: 11, label: '⚠️ LeetGPU 专属维度陷阱', text: 'A=(M, N)、B=(N, K)、C=(M, K)！A 行步长是 N，B 和 C 行步长都是 K。' },
          { line: 19, label: '分块缩并循环', text: '沿公共维 N 每次切 32 列，把子块留在片上复用 64 次！' },
          { line: 24, label: 'other=0.0 不可或缺', text: '非整除边界必须补 0.0，防止脏数据污染整行整列点积。' },
          { line: 26, label: 'tl.dot + ieee 精度保护', text: '加 input_precision="ieee" 避免 TF32 截断导致精度测试不过。' }
        ]
      },
      cuda: {
        source: `#include <cuda_runtime.h>
#define TILE 16

__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // LeetGPU 维度约定：A=(M, N), B=(N, K), C=(M, K)
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    int numTiles = (N + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        // 1. 协同加载当前 16x16 子块进片上 Shared Memory（越界补 0.0f）
        As[threadIdx.y][threadIdx.x] = (row < M && aCol < N) ? A[row * N + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < N && col < K) ? B[bRow * K + col] : 0.0f;
        __syncthreads();

        // 2. 在超高速片上共享内存做 16 次乘加复用
        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            sum = fmaf(As[threadIdx.y][i], Bs[i][threadIdx.x], sum);
        }
        __syncthreads(); // 等全部线程算完再进入下一轮覆盖 As/Bs
    }

    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(TILE, TILE);
    dim3 grid((K + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_tiled_kernel<<<grid, block>>>(A, B, C, M, N, K);
}`,
        annotations: [
          { line: 6, label: '片上双缓存 As / Bs', text: '把 A 和 B 的 16×16 子块搬到极低延迟的 Shared Memory，每个加载进来的元素被复用 16 次！' },
          { line: 19, label: '三目补零避免死锁', text: '越界线程不提前 return，而是写入 0.0f 后共同到达 __syncthreads()，完美支持任意奇葩矩阵尺寸！' },
          { line: 28, label: '为什么必须有第二次 __syncthreads()？', text: '防止跑得快的 Warp 已经进入下一轮循环把 As/Bs 覆写了，而慢的 Warp 还没读完当前轮数据！' },
          { line: 38, label: 'Grid 的 x 对应列 K，y 对应行 M', text: '这样同一 Warp 的 threadIdx.x 沿列连续变化，读 A/B 和写 C 全部实现合并访存！' }
        ]
      },
      pytorch: {
        source: `import torch

def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    torch.matmul(a, b, out=c)`,
        annotations: [
          { line: 4, label: '直通 cuBLAS 汇编库', text: 'PyTorch 一行直接调 NVIDIA 官方 cuBLAS，适合对照验证精度。' }
        ]
      }
    }
  },

  /* ===================== #9 Rainbow Table ===================== */
  {
    id: 9,
    sprint: 3,
    sprintOrder: 10,
    slug: 'rainbow-table',
    title: 'Rainbow Table',
    zhSubtitle: 'FNV-1a 多轮哈希 · 寄存器融合收官之战',
    difficulty: 'Easy',
    points: 15,
    estMinutes: 30,
    summary: '给定 N 个 32 位整数，在 GPU 上对每个数连续执行 R 轮 32-bit FNV-1a 哈希（offset=2166136261, prime=16777619，逐 4 个字节异或再乘质数），结果写入 output。',
    vizType: 'rainbow',
    vizConfig: {
      inputs: [0, 1, 42, 1337, 2026],
      defaultRounds: 3
    },
    trackInfo: {
      triton: {
        corePrimitives: ['Kernel Fusion（算子融合）', 'tl.uint32 溢出算术', 'tl.static_range 展开'],
        signature: 'def solve(input: torch.Tensor, output: torch.Tensor, N: int, R: int):',
        angle: '把整整 R 轮循环全部融合进**单个 Kernel 的片上寄存器**里——只读 1 次显存、只写 1 次显存，比 PyTorch 每轮起十几个 Kernel 快 30~80 倍！',
        gotcha: '1) 必须用无符号 `tl.uint32` 做乘法与右移（防符号位扩展出错）；2) 每轮开始前 `h` 都要重置为 `2166136261`，依次混入 `val` 的 4 个字节，算完后 `val = h` 进入下一轮。',
        pointsTips: [
          '用 `tl.static_range(0, 32, 8)` 在编译期彻底展开 4 字节循环。',
          '算力密集型算子，BLOCK_SIZE=512 寄存器占用率更优。'
        ],
        drill: {
          prompt: '在完整代码中补全两处核心：空 1（4 字节编译期展开循环）、空 2（FNV-1a 先异或再乘质数）。',
          blanks: [
            { id: 'BLANK_1', match: 'tl.static_range(0, 32, 8)', placeholder: '空 1', answers: ['tl.static_range(0, 32, 8)', 'range(0, 32, 8)'] },
            { id: 'BLANK_2', match: '(h ^ byte) * prime', placeholder: '空 2', answers: ['(h ^ byte) * prime', '(h^byte)*prime', '(h ^ byte) * 16777619'] }
          ],
          checks: [
            { label: '包含无符号 uint32 类型转换', regex: 'tl\\.uint32' },
            { label: '包含 FNV-1a 偏置 2166136261 与质数 16777619', regex: '2166136261[\\s\\S]*16777619' },
            { label: '在单 Kernel 内循环 R 轮融合计算', regex: 'range\\(R\\)' }
          ],
          hint: '空 1 为 tl.static_range(0, 32, 8)；空 2 为 (h ^ byte) * prime（先异或再相乘）。'
        }
      },
      cuda: {
        corePrimitives: ['__device__ __forceinline__', 'unsigned int 无符号溢位', '#pragma unroll'],
        signature: 'extern "C" void solve(const int* input, unsigned int* output, int N, int R)',
        angle: 'CUDA 的主场！用 `unsigned int` 天然支持 32-bit 模 2^32 无符号乘法溢出和逻辑右移，把 4 字节循环用 `#pragma unroll` 展开，整整 R 轮全程呆在 1 个 32 位寄存器 `val` 里疯狂迭代，零中间显存访问！',
        gotcha: '1) `input` 是 `const int*`，读出来一定要立刻转成 `unsigned int`（`static_cast<unsigned int>(input[idx])`），否则负数右移会触发算术右移补 1 导致哈希全错！\n2) 常量末尾加 `u`（`2166136261u`、`16777619u`）保证全程无符号运算。',
        pointsTips: [
          '纯算力密集型（Compute-bound）Kernel：#pragma unroll 消除内层 4 字节循环开销，几乎打满 GPU INT32 ALU 吞吐！',
          'threads = 256 即可让每个 SM 塞满活跃 Warp，轻松拿下高分。'
        ],
        drill: {
          prompt: '在完整代码中补全 FNV-1a 32-bit 两个标准常量：空 1（初始偏置）、空 2（FNV 质数）。',
          blanks: [
            { id: 'BLANK_1', match: '2166136261u', placeholder: '空 1', answers: ['2166136261u', '2166136261', '0x811c9dc5u', '0x811c9dc5'] },
            { id: 'BLANK_2', match: '16777619u', placeholder: '空 2', answers: ['16777619u', '16777619', '0x01000193u', '0x01000193'] }
          ],
          checks: [
            { label: '包含 extern "C" void solve 入口', regex: 'extern\\s+"C"\\s+void\\s+solve' },
            { label: '使用 unsigned int 做逻辑右移与无符号溢出乘法', regex: 'unsigned\\s+int' },
            { label: '包含 FNV-1a 常数与 R 轮循环', regex: '(2166136261|0x811c9dc5)[\\s\\S]*(16777619|0x01000193)' }
          ],
          hint: '空 1 初始偏置为 2166136261u（0x811c9dc5u）；空 2 质数为 16777619u（0x01000193u）。'
        }
      }
    },
    code: {
      triton: {
        source: `import torch
import triton
import triton.language as tl

@triton.jit
def fnv1a_hash(val):
    h = tl.full(val.shape, 2166136261, dtype=tl.uint32)
    prime = tl.full(val.shape, 16777619, dtype=tl.uint32)
    for shift in tl.static_range(0, 32, 8):
        byte = (val >> shift) & 0xFF
        h = (h ^ byte) * prime
    return h

@triton.jit
def rainbow_kernel(input_ptr, output_ptr, N, R, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    val = tl.load(input_ptr + offsets, mask=mask).to(tl.uint32)
    for _ in range(R):
        val = fnv1a_hash(val)
    tl.store(output_ptr + offsets, val.to(output_ptr.dtype.element_ty), mask=mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, R: int):
    BLOCK_SIZE = 512
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    rainbow_kernel[grid](input, output, N, R, BLOCK_SIZE=BLOCK_SIZE)`,
        annotations: [
          { line: 6, label: '@triton.jit 设备辅助函数', text: '编译时自动内联，零调用开销！' },
          { line: 9, label: 'tl.static_range 编译期展开', text: '将 4 字节循环展开为 4 组连续指令，消除循环分支。' },
          { line: 21, label: 'R 轮全在寄存器融合', text: '只读 1 次、只写 1 次显存，算子融合的教科书级收益！' },
          { line: 23, label: '安全回写目标数据类型', text: '自动适配 int32 或 uint32 输出张量。' }
        ]
      },
      cuda: {
        source: `#include <cuda_runtime.h>

__device__ __forceinline__ unsigned int fnv1a_hash(unsigned int val) {
    unsigned int h = 2166136261u;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        h ^= (val >> (i * 8)) & 0xFFu;
        h *= 16777619u;
    }
    return h;
}

__global__ void rainbow_kernel(const int* __restrict__ input, unsigned int* __restrict__ output, int N, int R) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        unsigned int val = static_cast<unsigned int>(input[idx]);
        for (int r = 0; r < R; ++r) {
            val = fnv1a_hash(val);
        }
        output[idx] = val;
    }
}

extern "C" void solve(const int* input, unsigned int* output, int N, int R) {
    if (N <= 0) return;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    rainbow_kernel<<<blocks, threads>>>(input, output, N, R);
}`,
        annotations: [
          { line: 5, label: '#pragma unroll 彻底展开 4 字节', text: '消除内层循环计数器，让 4 次 XOR + IMUL 连续流水线执行！' },
          { line: 16, label: '必须转为 unsigned int', text: '确保右移 >> 为逻辑右移（高位补 0），且乘法自动按无符号 32 位自然回绕（mod 2^32）。' },
          { line: 17, label: '单寄存器迭代 R 轮', text: '全程 0 次中间显存访问，INT32 算力吃满！' }
        ]
      },
      pytorch: {
        source: `import torch

def solve(input: torch.Tensor, output: torch.Tensor, N: int, R: int):
    val = input.to(torch.int64) & 0xFFFFFFFF
    for _ in range(R):
        h = torch.full_like(val, 2166136261)
        for shift in (0, 8, 16, 24):
            h = ((h ^ ((val >> shift) & 0xFF)) * 16777619) & 0xFFFFFFFF
        val = h
    output.copy_(val.to(output.dtype))`,
        annotations: [
          { line: 5, label: '为什么这题必须写 Kernel', text: 'PyTorch 每轮启 16+ 个 Kernel 来回刷显存，比 Triton/CUDA 慢几十倍！' }
        ]
      }
    }
  }
];
