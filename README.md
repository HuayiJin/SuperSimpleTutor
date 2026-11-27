# SuperSimpleTutor — AI Infra 极简互动学习平台

一个零依赖、纯静态、默认清爽浅色模式（支持深色一键切换）的 **AI Infra 互动学习平台**，涵盖从**单卡 GPU Kernel 编程**到**多卡大模型分布式训练显存推演**两大核心模块：

- 在线访问：https://huayijin.github.io/SuperSimpleTutor/
  - 直达 LeetGPU 前 10 题：`https://huayijin.github.io/SuperSimpleTutor/#leetgpu`
  - 直达 LLM 4D 并行显存推演台：`https://huayijin.github.io/SuperSimpleTutor/#llm`

---

## 模块一：⚡ LeetGPU 一天速通（Triton ⇄ CUDA 双轨版）

- **Triton 极速轨**：6 个核心原语 + 10~30 行 Python，约 3.5 小时攻克 LeetGPU 官方前 10 题。
- **CUDA C++ 硬核轨**：128-bit `float4`/`uint4` 向量化、`__shared__ smem[32][33]` 零 Bank Conflict 转置、16×16 Shared Memory Tiled GEMM、`__ldg()` 只读缓存与 `#pragma unroll` 寄存器融合。
- **6 套交互式 GPU 可视化实验室** + **每题 60 秒填空速练** + **双轨独立进度打卡** + **罗塞塔速查表**。

## 模块二：🧠 LLM 分布式训练显存与 4D 并行推演台

- **4 大工业级预设**：0.5B Toy、Qwen3-8B Dense、Qwen3-32B 64K 长文本、70B 64 卡集群。
- **全维度建模**：GQA + SwiGLU 架构、TP / SP / PP / CP（Ring Attention & Ulysses）、ZeRO-1/2/3、FlashAttention、分块 Cross-Entropy Loss、细粒度/分层块级重计算与 DualPipe 延后 dW 调度。
- **全景显存拆解**：实时瓶颈 Stage 显存条、一键 OOM 逐步优化配方、4 阶段显存水位图、4D 并行公式核对与单层张量账本。

## 本地运行

纯静态 ES Module，无需构建：

```bash
python3 -m http.server 8765
```
