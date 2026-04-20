# TAPSN 脚本命令行小文档

这份文档整理了仓库里几个关键脚本的常用命令，重点包括：

- 拓扑 `walkable cell` 的生成
- `tools/run_scenario_batch.py` 的批量生成与执行
- 运行前需要设置的 `PYTHONPATH` 和 LLM API 参数
- 基于已经生成好的 `scenario.toml` 只跑单个 case

建议所有命令都在仓库根目录执行：

```bash
cd /home/zhoumohan/codes/TAPSN
```

## 1. 环境准备

### 1.1 PYTHONPATH

运行前建议先设置仓库路径和 CARLA Python API：

```bash
export REPO_ROOT=/home/zhoumohan/codes/TAPSN
export CARLA_ROOT=/home/zhoumohan/carla/CARLA_0.9.15
export PYTHONPATH="$REPO_ROOT:$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla${PYTHONPATH:+:$PYTHONPATH}"
```

说明：

- `run_simulation.py` 需要仓库根目录在 `PYTHONPATH` 中。
- `utils/generate_walkable_cells.py` 和 `run_simulation.py` 都需要 CARLA Python API。
- 如果已经用 `conda activate carla` 并且环境里带了部分路径，仍然建议显式导出，避免路径不完整。

### 1.2 LLM API 参数

如果 `tools/run_scenario_batch.py` 使用 `--planner llm`，还需要准备 API Key。

代码支持以下环境变量，优先级依次为：

1. `CASE_BUILDER_API_KEY`
2. `DASHSCOPE_API_KEY`
3. `OPENAI_API_KEY`

推荐直接设置：

```bash
export CASE_BUILDER_API_KEY="sk-88a182b441b34b01af4a0875c79d3fb0"
```

如果走 DashScope 或其他 OpenAI-compatible 服务，也可以：

```bash
export DASHSCOPE_API_KEY="<your-api-key>"
# 或
export OPENAI_API_KEY="<your-api-key>"
```

补充：

- `--api-key` 命令行参数会覆盖环境变量。
- `--base-url` 用来指定 OpenAI-compatible 接口地址。
- 当前 `tools/run_scenario_batch.py` 默认 `--base-url` 是：

```bash
https://dashscope.aliyuncs.com/compatible-mode/v1
```

## 2. 拓扑 Cell 怎么生成

核心脚本是 [generate_walkable_cells.py](/home/zhoumohan/codes/TAPSN/utils/generate_walkable_cells.py)。

### 2.1 最常用命令

```bash
conda run -n carla python utils/generate_walkable_cells.py \
  --map-name Town10HD_Opt \
  --timeout 240 \
  --output-dir topology/Town10HD_Opt
```

如果不写 `--output-dir`，默认输出到：

```bash
artifacts/<map_name>/topology/
```

典型输出：

- `walkable_cells.json`
- `walkable_cells.svg`

### 2.2 常用参数

- `--map-name`：要分析的地图名，例如 `Town10HD_Opt`
- `--host`：CARLA host，默认 `127.0.0.1`
- `--port`：CARLA 端口，默认 `2000`
- `--timeout`：CARLA RPC 超时秒数，默认 `180`
- `--output-dir`：输出目录
- `--waypoint-distance`：构建行人拓扑图时的 waypoint 间距
- `--approach-length`：crossing cell 向连接 sidewalk 延伸的长度
- `--sidewalk-chunk-length`：普通 sidewalk cell 的目标长度
- `--sidewalk-tail-merge-length`：合并末尾短 sidewalk 片段的阈值
- `--sidewalk-min-longitudinal-length`：过短 sidewalk chunk 的最小纵向长度阈值
- `--route-portal-merge-radius`：crossing 边界 portal 合并半径
- `--route-portal-sidewalk-merge-length`：按 sidewalk 连通性合并 portal 的最大局部路径长度
- `--large-map-safe`：大地图安全模式，加载更保守
- `--skip-static-obstacles`：跳过静态障碍物提取，适合先快速导出 catalog

### 2.3 大地图建议

如果地图较大，优先尝试：

```bash
conda run -n carla python utils/generate_walkable_cells.py \
  --map-name Town10HD_Opt \
  --timeout 240 \
  --output-dir topology/Town10HD_Opt \
  --large-map-safe
```

这个模式会走更保守的地图加载路径，并跳过静态障碍物枚举。

### 2.4 批量生成多个地图

仓库里已有一个简单脚本 [generate_maps_topology.sh](/home/zhoumohan/codes/TAPSN/generate_maps_topology.sh)，改 `MAPS` 后可以直接跑：

```bash
bash generate_maps_topology.sh
```

## 3. `tools/run_scenario_batch.py` 的命令和参数

核心脚本是 [run_scenario_batch.py](/home/zhoumohan/codes/TAPSN/tools/run_scenario_batch.py)。

它的作用是：

- 从一个或多个 `walkable_cells.json` 里选可用 scene
- 用 heuristic 或 llm planner 构造 case
- 按 coverage 状态持续生成
- 可选地立刻调用 `run_simulation.py` 执行
- 达到目标 `accepted case` 数量后停止，或没有可选任务时停止

### 3.1 最常用命令

```bash
conda run -n carla python tools/run_scenario_batch.py \
  --output-root data \
  --target-count 20 \
  --catalog topology/Town10HD_Opt/walkable_cells.json \
  --catalog topology/Town05_Opt/walkable_cells.json \
  --planner llm \
  --model qwen-plus \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --execute \
  --seed 2026 \
  --wall-time-limit 300 \
  --scenario-prefix case \
  --carla-host 127.0.0.1 \
  --carla-port 2000
```

如果希望脚本在运行过程中能托管 CARLA 重启，再补上：

```bash
--carla-root "$CARLA_ROOT"
```

如果只想生成 case，不立即执行，把 `--execute` 去掉即可。

### 3.2 参数总览

#### 必填参数

- `--output-root`
  批量生成的根输出目录。最终 case 一般会写到：

  ```bash
  <output-root>/<map_name>/<case_id>/
  ```

  例如：

  ```bash
  data/Town10HD_Opt/001/
  ```

- `--target-count`
  目标接受 case 数量。这里的“接受”指：
  - 如果没有 `--execute`，build 成功就算接受
  - 如果有 `--execute`，执行返回码为 0 才算接受

- `--catalog`
  输入的 `walkable_cells.json` 路径。这个参数可以重复传多次。

  示例：

  ```bash
  --catalog topology/Town10HD_Opt/walkable_cells.json \
  --catalog topology/Town05_Opt/walkable_cells.json
  ```

#### 规划与生成参数

- `--planner {heuristic,llm}`
  规划器模式。
  - `heuristic`：不依赖 LLM
  - `llm`：走 OpenAI-compatible 接口

- `--model`
  仅在 `--planner llm` 时有意义。默认是 `qwen-plus`。

- `--base-url`
  LLM 接口地址。默认：

  ```bash
  https://dashscope.aliyuncs.com/compatible-mode/v1
  ```

- `--api-key`
  直接在命令行传 API Key。若填写，会覆盖环境变量。

- `--llm-strict`
  如果 LLM 不可用、返回非法结果或多次失败，则当前 case 直接失败，不回退到 heuristic。

- `--llm-retries`
  LLM 总尝试次数。默认 `2`。

- `--llm-timeout`
  单次 LLM HTTP 超时秒数。默认 `90.0`。

- `--scenario-prefix`
  生成的 `scenario_name` 前缀。默认 `case`。

- `--seed`
  批量生成的总随机种子。默认 `2026`。

  这里需要特别注意：

  - 它不是“所有 case 都完全同一个 seed”。
  - 它是 `ScenarioDirector` 的批量级随机种子，用来驱动任务选择顺序、每个 case 的派生 seed、重启间隔抽样等。
  - 同一组输入 catalog、同一个 `output-root/state-path`、同一个初始状态下，固定 `--seed` 更容易复现相同流程。
  - 但如果你用了 `--resume`，继续跑时会基于已有 state 接着走，不是重新从空状态复现。

- `--dry-run`
  只选择一个下一步任务并打印，不真正 build，也不 execute。适合先检查 selection 逻辑。

#### 执行参数

- `--execute`
  build 完 case 后，立刻调用 [run_simulation.py](/home/zhoumohan/codes/TAPSN/run_simulation.py) 执行。

- `--no-sampling`
  关闭 sampling 和 ego 声明。适合只想快速生成/验证仿真，不需要采样产物时使用。

- `--conda-env`
  执行 `run_simulation.py` 时使用的 conda 环境。默认 `carla`。

- `--carla-host`
  CARLA host，默认 `127.0.0.1`。

- `--carla-port`
  CARLA 端口，默认 `2000`。

- `--wall-time-limit`
  每个 case 的 wall-clock 超时秒数，不是整个 batch 的总时长。

  这个参数的实际行为是：

  - 只在 `--execute` 时有意义
  - 超时后会终止当前 `run_simulation.py`
  - 当前 case 记为失败
  - 返回码通常表现为 `124`
  - batch 不会整体退出，而是继续尝试后续 case

  示例：

  ```bash
  --wall-time-limit 300
  ```

  表示单个 case 最多跑 300 秒墙钟时间。

#### 状态恢复参数

- `--resume`
  表示从已有 state 继续跑。

  实际行为：

  - 只有当 state 文件存在时，才会真的加载已有状态
  - 如果 state 文件不存在，即使写了 `--resume`，也会从空状态开始初始化

- `--state-path`
  手动指定 director state 文件路径。

  默认值不是写死在参数里，而是运行时按下面规则生成：

  ```bash
  <output-root>/scenario_director_state.json
  ```

  例如：

  ```bash
  data/scenario_director_state.json
  ```

  建议理解成：

  - `--output-root` 决定 case 产物目录
  - `--state-path` 决定批量进度状态文件放哪里
  - 如果你想把同一批任务的状态和产物分开管理，可以手动指定 `--state-path`

  示例：

  ```bash
  --state-path tmp/batch_state/town10_llm_state.json
  ```

  如果你写了 `--state-path`，通常也应配合 `--resume` 使用，否则只是把新的 state 写到这个位置。

#### CARLA 托管重启参数

- `--carla-root`
  CARLA 安装目录。只有提供了它，脚本才有能力主动重启 CARLA。

  如果没提供 `--carla-root`：

  - 不会做主动重启
  - 即使设置了 `--restart-min-cases`、`--restart-max-cases`、`--restart-wait`，也不会真正生效

- `--restart-min-cases`
  两次主动重启之间，最少间隔多少个成功尝试的 case。默认 `5`。

- `--restart-max-cases`
  两次主动重启之间，最多间隔多少个成功尝试的 case。默认 `8`。

  这两个参数的实际行为要注意：

  - 代码会从 `[restart_min_cases, restart_max_cases]` 之间随机抽一个整数，作为下一次主动重启阈值
  - 只有 `--execute` 且设置了 `--carla-root` 时，这个阈值才有意义
  - 如果 `--restart-min-cases <= 0`，内部会自动修正成 `1`
  - 如果 `--restart-max-cases < --restart-min-cases`，内部会自动把 `max` 提升到 `min`
  - 也就是说，这两个参数通常不会因为大小关系报运行时错误，但会被内部纠正

  例如：

  ```bash
  --restart-min-cases 5 --restart-max-cases 8
  ```

  表示每成功跑完若干个 case 后，在 5 到 8 之间随机抽一个阈值，到达后主动重启 CARLA。

  如果你之前“报错 restart min/max cases”，优先检查：

  - 是否把值写成了非整数
  - 是否把参数名拼错了
  - 是否误以为它们在没传 `--carla-root` 的情况下也会生效

- `--restart-wait`
  CARLA 重启后，检测到 TCP 端口已经可连接后，额外再等待多少秒再继续跑。默认 `12.0`。

  这个参数不是“重启前等待”，而是“重启完成并探测到端口 ready 之后，再额外 sleep 一段时间”。

  对一些地图加载慢、端口虽然打开但世界还不完全稳定的情况，这个参数很有用。

- `--carla-ready-timeout`
  重启 CARLA 后，最多等待多久让它恢复可连接。默认 `90.0` 秒。

  超过这个时间仍不可连接，会直接抛出运行时错误。

### 3.3 常见组合

#### 只 build，不执行

```bash
conda run -n carla python tools/run_scenario_batch.py \
  --output-root data \
  --target-count 10 \
  --catalog topology/Town10HD_Opt/walkable_cells.json \
  --planner heuristic
```

#### build 后立刻执行

```bash
conda run -n carla python tools/run_scenario_batch.py \
  --output-root data \
  --target-count 10 \
  --catalog topology/Town10HD_Opt/walkable_cells.json \
  --planner llm \
  --execute \
  --wall-time-limit 300
```

#### 带 state 恢复继续跑

```bash
conda run -n carla python tools/run_scenario_batch.py \
  --output-root data \
  --target-count 20 \
  --catalog topology/Town10HD_Opt/walkable_cells.json \
  --planner llm \
  --execute \
  --state-path tmp/state/town10_batch.json \
  --resume
```

#### 带托管重启

```bash
conda run -n carla python tools/run_scenario_batch.py \
  --output-root data \
  --target-count 20 \
  --catalog topology/Town10HD_Opt/walkable_cells.json \
  --planner llm \
  --execute \
  --carla-root "$CARLA_ROOT" \
  --restart-min-cases 5 \
  --restart-max-cases 8 \
  --restart-wait 15 \
  --carla-ready-timeout 120
```

### 3.4 输出结构

`run_scenario_batch.py` 常见输出结构如下：

```text
data/
  scenario_director_state.json
  Town10HD_Opt/
    001/
      config/
        scenario.toml
        sampling_policy.toml
        scene_description.txt
        case.json
      logs/
        run.log
        run_result.json
      simulation/
      sampling/
      visualization/
      manifest.json
```

补充：

- `scenario.toml` 通常在 `data/<map_name>/<case_id>/config/scenario.toml`
- 默认 state 文件在 `data/scenario_director_state.json`
- 如果 sampling 启用，采样产物会落到该 map 根目录下按 `sequence_id` 管理的结构中

## 4. 基于已经生成的 TOML，只跑单个 case

如果你已经有一个生成好的配置文件，比如：

```bash
data/Town10HD_Opt/001/config/scenario.toml
```

那就不需要再跑 `run_scenario_batch.py`，直接调用 [run_simulation.py](/home/zhoumohan/codes/TAPSN/run_simulation.py)。

### 4.1 最简单命令

```bash
conda run -n carla python run_simulation.py \
  --scenario-config data/Town10HD_Opt/001/config/scenario.toml \
  --carla-host 127.0.0.1 \
  --carla-port 2000 \
  --debug
```

### 4.2 常用调试命令

```bash
conda run -n carla python run_simulation.py \
  --scenario-config data/Town10HD_Opt/001/config/scenario.toml \
  --carla-host 127.0.0.1 \
  --carla-port 2000 \
  --duration 60 \
  --output tmp/single_case_debug \
  --csv \
  --debug
```

### 4.3 `run_simulation.py` 常用参数

- `--scenario-config`：要跑的场景配置文件
- `--sfm-config`：SFM 配置文件，默认 `config/sfm_config.toml`
- `--carla-host` / `--carla-port`：CARLA 地址
- `--duration`：最大仿真时长，单位秒
- `--output`：输出目录，默认 `output`
- `--csv`：导出模拟结果 CSV
- `--debug`：打开调试日志

### 4.4 注意事项

- 最好始终在仓库根目录执行。
- 如果 `scenario.toml` 里启用了 `[sampling]`，它还会继续读取采样相关配置。
- 对 batch 生成出来的 case，仓库里的配置通常会引用默认 sensor rig 和当前 case 自己的 `sampling_policy.toml`。
- 在仓库根目录运行时，默认的 `config/sensors/default_multimodal.toml` 才能稳定解析到。
- 如果该 case 之前已经跑过一次，并且 `sampling.output_root + sequence_id` 指向的目录已经存在，重跑时默认会新建一个带时间戳后缀的目录，而不是覆盖旧目录。

## 5. 推荐工作流

如果你是从零开始，一般建议按这个顺序：

1. 启动 CARLA
2. 设置 `PYTHONPATH`
3. 如果用 `llm` planner，设置 `CASE_BUILDER_API_KEY`
4. 用 `utils/generate_walkable_cells.py` 生成目标地图的 `walkable_cells.json`
5. 先拿一个 `scenario.toml` 用 `run_simulation.py` 跑单 case 验证
6. 再用 `tools/run_scenario_batch.py --execute` 批量生成

## 6. 最短速查

生成拓扑：

```bash
conda run -n carla python utils/generate_walkable_cells.py \
  --map-name Town10HD_Opt \
  --output-dir topology/Town10HD_Opt
```

批量生成并执行：

```bash
conda run -n carla python tools/run_scenario_batch.py \
  --output-root data \
  --target-count 20 \
  --catalog topology/Town10HD_Opt/walkable_cells.json \
  --planner llm \
  --execute
```

只跑一个已有 case：

```bash
conda run -n carla python run_simulation.py \
  --scenario-config data/Town10HD_Opt/001/config/scenario.toml
```
