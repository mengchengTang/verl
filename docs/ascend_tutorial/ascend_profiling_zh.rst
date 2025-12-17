Performance data collection based on FSDP or MindSpeed(Megatron) on Ascend devices(zh)
====================================

在昇腾设备上基于FSDP或MindSpeed(Megatron)后端进行性能数据采集

Last updated: 12/20/2025.

这是一份在昇腾设备上基于FSDP或MindSpeed(Megatron)后端，使用GRPO或DAPO算法进行数据采集的教程。

配置
----

使用两级profile设置来控制数据采集

- 全局采集控制：使用verl/trainer/config/ppo_trainer.yaml(FSDP)或verl/trainer/config/ppo_megatron_trainer.yaml(MindSpeed)中的配置项控制采集的模式和步数，
- 角色profile控制：通过每个角色中的配置项控制等参数。

全局采集控制
~~~~~~~~~~~~

通过 ppo_trainer.yaml 中的参数控制采集步数和模式：

-  global_profiler: 控制采集的rank和模式

   -  tool: 使用的采集工具，选项有 nsys、npu、torch、torch_memory。
   -  steps: 此参数可以设置为包含采集步数的列表，例如 [2, 4]，表示将采集第2步和第4步。如果设置为 null，则不进行采集。
   -  save_path: 保存采集数据的路径。默认值为 "outputs/profile"。

角色profiler控制
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在每个角色的 ``profiler`` 字段中，您可以控制该角色的采集模式。

-  enable: 是否为此角色启用性能分析。
-  all_ranks: 是否从所有rank收集数据。
-  ranks: 要收集数据的rank列表。如果为空，则不收集数据。
-  tool_config: 此角色使用的性能分析工具的配置。

通过每个角色的 ``profiler.tool_config.npu`` 中的参数控制具体采集行为：

-  level: 采集级别—选项有 level_none、level0、level1 和 level2

   -  level_none: 禁用所有基于级别的数据采集（关闭 profiler_level）。
   -  level0: 采集高级应用数据、底层NPU数据和NPU上的算子执行详情。在权衡数据量和分析能力后，level0是推荐的默认配置。
   -  level1: 在level0基础上增加CANN层AscendCL数据和NPU上的AI Core性能指标。
   -  level2: 在level1基础上增加CANN层Runtime数据和AI CPU指标。

-  contents: 控制采集内容的选项列表，例如
   npu、cpu、memory、shapes、module、stack。
   
   -  npu: 是否采集设备端性能数据。
   -  cpu: 是否采集主机端性能数据。
   -  memory: 是否启用内存分析。
   -  shapes: 是否记录张量形状。
   -  module: 是否记录框架层Python调用栈信息。相较于stack，更推荐使用module记录调用栈信息，因其产生的性能膨胀更低。
   -  stack: 是否记录算子调用栈信息。

-  analysis: 启用自动数据解析。
-  discrete: 使用离散模式。

示例
----

禁用采集
~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

      global_profiler:
         steps: null # disable profile

端到端采集
~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

      global_profiler:
         steps: [1, 2, 5]
         save_path: ./outputs/profile
      actor_rollout_ref:
         actor:  # 设置actor role的profiler采集配置参数
            profiler:
               enable: True
               all_ranks: True
               tool_config:
                  npu:
                     discrete: False
                     contents: [npu, cpu]

        # rollout & ref follow actor settings


离散模式采集
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

      global_profiler:
         steps: [1, 2, 5]
      actor_rollout_ref:
         actor:
            profiler:
               enable: True
               all_ranks: True
               tool_config:
                  npu:
                     discrete: True
                     contents: [npu, cpu]
        # rollout & ref follow actor settings


可视化
------

采集后的数据存放在用户设置的save_path下，可通过 `MindStudio Insight <https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html>`_ 工具进行可视化。

另外在Linux环境下，MindStudio Insight工具提供了 `JupyterLab插件 <https://www.hiascend.com/document/detail/zh/mindstudio/82RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0130.html>`_ 形态，提供更直观和交互式强的操作界面。JupyterLab插件优势如下：

- 无缝集成：支持在Jupyter环境中直接运行MindStudio Insight工具，无需切换平台，无需拷贝服务器上的数据，实现数据即采即用。
- 快速启动：通过JupyterLab的命令行或图形界面，可快速启动MindStudio Insight工具。
- 运行流畅：在Linux环境下，通过JupyterLab环境启动MindStudio Insight，相较于整包通信，有效解决了运行卡顿问题，操作体验显著提升。
- 远程访问：支持远程启动MindStudio Insight，可通过本地浏览器远程连接服务直接进行可视化分析，缓解了大模型训练或推理数据上传和下载的困难。

如果analysis参数设置为False，采集之后需要进行离线解析：

.. code:: python

    import torch_npu
    # profiler_path请设置为"localhost.localdomain_<PID>_<timestamp>_ascend_pt"目录的上一级目录
    torch_npu.profiler.profiler.analyse(profiler_path=profiler_path)


--------

VeRL 性能数据精细化采集指导
---------------------------

背景与挑战
~~~~~~~~~~

上文的“采集指导”可以帮助大家快速上手 Ascend 场景下的性能分析。然而，在长序列（Long Context）或大全局批量（Large Global Batch Size）的训练场景中，在一个完整的训练步（Step）内，模型计算呈现出高频次、重复性的特征：

1. **Rollout 阶段**：序列生成（Generate Sequence）是一个自回归过程，涉及成千上万次 Decoder 模型的前向计算（每次生成一个 Token）。

2. **Training 阶段**：为了控制显存峰值，VeRL 通常采用 Micro-Batch 策略，将庞大的数据流切分为多个微批次进行流水线或累积式计算。

  - **compute_log_prob (Actor/Ref)**：涉及多轮纯前向传播（Forward）。

  - **update_policy (Actor/Critic)**：涉及多轮前向与反向传播（Backward）。

这种多轮次、高频次的执行特性，会导致全量 Profiling 产生海量且重复的算子记录。如下图所示：

.. image:: https://raw.githubusercontent.com/mengchengTang/verl-data/master/verl_ascend_profiler.png

即使我们使用了 ``discrete`` 模式将不同阶段（如 rollout、update）的数据分开，单个阶段的 Profiling 数据文件仍然可能非常巨大（可能达到数 TB）。这会导致以下问题：

1. **解析困难**：数据解析耗时极长，甚至因内存溢出导致解析失败。

2. **可视化卡顿**：可视化工具难以流畅加载和渲染如此庞大的时间轴数据。

解决方案
~~~~~~~~

针对上述重复计算流程，我们可以采用**关键路径采样**策略：仅采集具有代表性的数据片段（如特定 Decode Step 或首个 Micro-Batch），即可有效分析性能瓶颈，同时大幅降低 Profiling 数据规模。

    **重要提示**

    本章节介绍的方法涉及直接修改 Python 源码。建议在修改前备份相关文件，或在调试完成后恢复代码，以免影响后续正常训练流程。
    另外，使用手动代码插桩进行采集时，建议在 ``verl/trainer/config/ppo_trainer.yaml`` 中禁用全局采集（设置 ``global_profiler: steps: null``），以避免 Profiler 冲突。


`torch_npu.profiler <https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/devaids/auxiliarydevtool/atlasprofiling_16_0038.html>`_ 提供 Ascend 全流程性能数据采集 API，开发者可自定义去 Python 代码中加 API，进行自定义阶段的采集。这样，用户可以针对自身感兴趣的性能数据片段进行采集。

1. Rollout 阶段性能数据精细化采集
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

当前 VeRL 采用服务化在线调用方式，如果想采集特定 Decode Step 的性能数据，可以在推理引擎中加入 ``torch_npu.profiler`` 采集代码。

**vLLM 引擎**

修改文件：``path/to/vllm-ascend/vllm/worker/worker_v1.py``

.. code-block:: diff

      class NPUWorker(WorkerBase):
  
          def __init__(self, *args, **kwargs):
              # ... existing code ...
  
  +           # Initialize profiler
  +           import torch_npu
  +           experimental_config = torch_npu.profiler._ExperimentalConfig(
  +               profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
  +               export_type=torch_npu.profiler.ExportType.Db,
  +           )
  +           self.profiler_npu = torch_npu.profiler.profile(
  +               activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
  +               with_module=False,
  +               profile_memory=False,
  +               experimental_config=experimental_config,
  +               # 跳过第一步，warmup一步，采集3步，重复1次。如果想采集第30~70 decode step，可以设置为schedule=torch_npu.profiler.schedule(wait=29, warmup=1, active=30, repeat=1)
  +               schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
  +               on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./outputs/vllm_profile", analyse_flag=True)
  +           )
  +           self.profiler_npu.start()
  
              # ... existing code ...
  
          def execute_model(self, scheduler_output=None, intermediate_tensors=None, **kwargs):
              # ... existing code ...
              output = self.model_runner.execute_model(scheduler_output,
                                                  intermediate_tensors)
              
  +           # 驱动 schedule
  +           self.profiler_npu.step()
              
              # ... existing code ...

**SGLang 引擎**

修改文件：``path/to/sglang/python/sglang/srt/model_executor/model_runner.py``

.. code-block:: diff

      # ... existing imports ...
  +   import torch_npu
  
      class ModelRunner:
  
          def __init__(self, *args, **kwargs):
              # ... existing init code ...
              
  +           # Initialize profiler (配置同上，略)
  +           self.profiler_npu = torch_npu.profiler.profile(
  +               # ...
  +               # 跳过第一步，warmup一步，采集3步，重复1次。
  +               schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
  +               on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./outputs/sglang_profile", analyse_flag=True)
  +           )
  +           self.profiler_npu.start()
  
          def forward(self, input_ids, positions, forward_batch=None, **kwargs):
              # ... existing code ...
              logits = self.model.forward(input_ids, positions, forward_batch)
              
  +           self.profiler_npu.step()
              
              return logits


2. compute_log_prob (Actor & Ref) 阶段精细化采集
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

该阶段计算新旧策略的概率分布，通常被切分为多个 Micro-Batch。Ref 模型与 Actor 模型共用代码。

修改文件：``verl/workers/actor/dp_actor.py``

.. code-block:: diff

      # ... 引入依赖 ...
  +   import torch_npu
  
      class DataParallelPPOActor(BasePPOActor):
  
          def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
          
  +           # 准备 profiler (配置同上，略)
  +           prof = torch_npu.profiler.profile(
  +               # ...
  +               # wait=0, warmup=0, active=1: 直接采集第一个 micro-batch
  +               schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
  +               on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./outputs/actor_compute_log_prob", analyse_flag=True)
  +           )
      
              is_ref = self.actor_optimizer is None
              
  +           # 仅采集 Ref 模型
  +           if is_ref:
  +               prof.start()
  
              for micro_batch in micro_batches:
  
                  # ... 原始计算逻辑 ...
                  with torch.no_grad():
                      entropy, log_probs = self._forward_micro_batch(...)
                      
  +                   # 驱动 schedule
  +                   if is_ref:
  +                       prof.step()
                  
                  # ...


3. update阶段性能数据精细化采集
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Update 阶段包含前向和反向传播。

修改文件：``verl/workers/actor/dp_actor.py``

.. code-block:: diff

      # ... 引入依赖 ...
  +   import torch_npu
  
      class DataParallelPPOActor(BasePPOActor):
  
          def update_policy(self, data: DataProto):
              
  +           # 准备 profiler (配置同上，略)
  +           prof = torch_npu.profiler.profile(
  +               # ...
  +               # 仅采集第一个 Mini Batch（包含所有 Micro-Batch 的计算和一次优化器更新）
  +               schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
  +               on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./outputs/actor_update_profile", analyse_flag=True)
  +           )
  +           prof.start()
              
              # ... PPO Epochs 循环 ...
              for _ in range(self.config.ppo_epochs):
                  # ... Mini Batch 循环 ...
                  for batch_idx, mini_batch in enumerate(mini_batches):
                      # ... mini_batches 切分 ...
  
                      for i, micro_batch in enumerate(micro_batches):
                          # ... 原始 Forward & Backward 逻辑 ...
                          # ... loss.backward() ...
                          pass
      
                      grad_norm = self._optimizer_step()
                      
  +                   # 驱动 schedule
  +                   prof.step()


Megatron 场景说明
~~~~~~~~~~~~~~~~~

Megatron 后端的 Micro-Batch 调度由 Megatron 内部管理（Pipeline Parallelism），因此无法像 FSDP 那样在 Micro-Batch 级别进行精细控制。可以通过修改 ``update_policy`` 方法进行精细化采集。

修改文件：``verl/workers/actor/megatron_actor.py``

**update_policy 方法**

在 ``MegatronPPOActor.update_policy`` 中，修改为：

.. code-block:: diff

      class MegatronPPOActor(BasePPOActor):
          
          def update_policy(self, dataloader: Iterable[DataProto]) -> dict:
              # ...
  +           # 准备 profiler (配置同上，略)
  +           prof = torch_npu.profiler.profile(
  +               # ...
  +               # 仅采集第一个 Mini Batch 的计算（含所有 Micro-Batch）和一次优化器更新
  +               schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
  +               on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./outputs/megatron_update_profile", analyse_flag=True)
  +           )
  +           prof.start()
              
              for data in dataloader:
                  # ... 内部会调用 self.forward_backward_batch 进行计算 ...
                  # ... metric_micro_batch = self.forward_backward_batch(...)
                  
                  # ... self.actor_optimizer.step() ...
                  
  +               # 驱动 schedule
  +               prof.step()
