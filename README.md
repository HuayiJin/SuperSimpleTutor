# SuperSimpleTutor — LeetGPU 一天速通（Triton ⇄ CUDA 双轨版）

一个零依赖、开箱即用的交互式 Web 教程，带你 **一天攻克 LeetGPU 官方前 10 题**，同时支持：

- **⚡ Triton 极速轨**：最简单的真正 GPU Kernel 语言，6 个核心原语 + 10~30 行 Python，约 3.5 小时稳拿合理高分。
- **🟢 CUDA C++ 硬核轨**：128-bit `float4` / `uint4` 向量化、`__shared__ smem[32][33]` 零 Bank Conflict 转置、16×16 Shared Memory Tiled GEMM、`__ldg()` 只读缓存与 `#pragma unroll` 寄存器融合。
- **PyTorch 对照视角**：每题均可随时切换查看 PyTorch 写法与性能差异。

## 覆盖 LeetGPU 官方前 10 题

| Sprint | 题号 | 题目 | 核心技能（Triton / CUDA） |
| --- | --- | --- | --- |
| Sprint 1 | `#1` | Vector Addition | 1D Block/Mask 模板 / `float4` 向量化 |
| Sprint 1 | `#10` | Matrix Copy | 2D→1D 展平思维 / 16B 连续拷贝 |
| Sprint 1 | `#7` | ReLU Activation | `tl.maximum` / `fmaxf(0.0f, x)` 无分支 |
| Sprint 1 | `#8` | Leaky ReLU | `tl.where` / 三目 SELP 谓词选择 |
| Sprint 2 | `#6` | Reverse Array | `N // 2` 网格裁剪，原地零竞争交换 |
| Sprint 2 | `#4` | Color Inversion | `uint32` / `uint4` 强转 + `^ 0x00FFFFFF` 锁定 Alpha |
| Sprint 2 | `#3` | Matrix Transpose | 2D Tile `tl.trans` / `__shared__ smem[32][33]` |
| Sprint 3 | `#5` | 1D Convolution | 滑窗累加器 / `__ldg()` + `fmaf` |
| Sprint 3 | `#2` | Matrix Multiplication | `tl.dot(ieee)` / 16×16 Shared Memory Tiled GEMM |
| Sprint 3 | `#9` | Rainbow Table | FNV-1a 多轮哈希片上寄存器融合 |

## 核心功能

1. **顶部双轨一键切换（快捷键 `T`）**：切换 Triton / CUDA 后，路线图、核心原语、破题思路、避坑提示、逐行批注、填空速练与可视化实验室全部同步切换，双轨进度独立持久化。
2. **6 套交互式 GPU 原理可视化实验室**：1D Grid/Mask 模拟器、MatMul 维度点积、Matrix Transpose 地址跳变、RGBA 位异或画布、1D 卷积滑窗步进、FNV-1a 多轮哈希实时演算。
3. **每题 60 秒填空速练 + 冲刺番茄钟 + 双轨罗塞塔速查表**。

## 本地运行

纯静态 ES Module，无需 `npm install` 或构建步骤：

```bash
python3 -m http.server 8765
```

然后在浏览器打开 `http://localhost:8765/` 即可开始学习（也可直接开启 GitHub Pages 部署）。
