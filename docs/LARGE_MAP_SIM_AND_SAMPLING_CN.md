# 大地图仿真与采集链路简要整理

本文简要整理 Town12/Town13/Town15 这类大地图下，拓扑生成、case 运行、采集与避障相关问题的处理思路与当前状态。

## 1. 问题背景

原始链路在大地图下主要有几个问题：

- 整图 `PedPathPlanner` 在大地图上不稳定，容易导致构图失败或段错误。
- 运行时虽然使用了 tile 级 `walkable_cells.json`，但 world 仍然按整张地图加载，没有把 tile streaming 真正接入 case 执行链路。
- pedestrian ego 的传感器直接 `attach_to walker` 时，在大地图中会出现世界位置更新异常，表现为第一帧正常、后续帧 `x/y` 漂移。
- 静态障碍物提取原先使用绝对 `z` 过滤，在高程地图中会把大量地面障碍误判为悬空物。

## 2. 方案演进

整体方案从“整图处理”改成“tile-first”：

- 先基于 `.xodr` 离线提取有 sidewalk 的区域。
- 只对这些区域切 tile，并生成每个 tile 的 `walkable_cells.json/svg`。
- case 构建时直接使用 tile 级 topology。
- 运行时引入 tile streaming anchor，让 CARLA 只稳定加载当前 tile 周边区域。
- 采集阶段不再依赖 walker-attached sensor，而是按 ego 每帧姿态手动更新传感器世界位姿。

这条链路已经通过：

- tile 内 walkable semantics 可视化验证
- tile 级 walkable cell 生成
- tile case 构建与运行
- tile case 下 ego 采集链路验证

## 3. 关键代码改造

### 3.1 tile 级拓扑生成

- `utils/generate_walkable_cells_large_map.py`

改成以 `map -> all tiles -> per-tile cells` 的方式运行，输出目录为：

- `topology/<map>/tile_catalog.json`
- `topology/<map>/<tile_id>/walkable_cells.json`
- `topology/<map>/<tile_id>/walkable_cells.svg`

### 3.2 运行时 tile streaming

- `carla_simulation.py`

在检测到 tile catalog case 后：

- 自动读取 tile 元数据
- 设置 `tile_stream_distance`
- 设置 `actor_active_distance`
- 设置 `spectator_as_ego=True`
- 用 spectator 作为 tile streaming 锚点
- 在 tile 中心上方 warmup 若干帧后再进入后续初始化

### 3.3 运行时 PedPathPlanner 按需化

- `pedestrian_spawner.py`

对 scene-route case：

- 不再无条件初始化整图 `PedPathPlanner`
- 只有在旧路径确实需要 `generate_route` 或 random pedestrian 时，才懒加载 planner

这样避免了运行时重新触发整图 planner 的大地图问题。

### 3.4 pedestrian ego 传感器同步

- `sampling/sensors/base.py`
- `sampling/sensors/registry.py`
- `sampling/collector.py`
- `run_simulation.py`

当前 pedestrian ego 的采集链路改成：

- walker ego 下，传感器不再直接 `attach_to walker`
- 传感器按 ego 当前世界姿态生成初始世界位姿
- 每个 CARLA tick 前，基于：
  - ego 当前世界位姿
  - rig 中定义的相对偏移
  - gait motion 带来的每帧抖动
  重新计算并设置每个传感器的世界位姿

这样规避了大地图中 walker-attached sensor 的世界位置漂移问题。

### 3.5 静态障碍物高度过滤

- `obstacles.py`

静态障碍物过滤从“绝对世界 `z` 阈值”改成“相对最近地面的 clearance 阈值”：

- 先获取 tile 已加载区域内的 `EnvironmentObject`
- 直接使用 world-space bounding box 计算 `bottom_z`
- 查找最近 `Sidewalk -> Driving -> Any waypoint` 的 `ground_z`
- 计算 `clearance = bottom_z - ground_z`
- 使用 `clearance` 判断是否为悬空物

其中 `max_obstacle_z_pos` 这个配置键现在等价于“最大允许离地间隙阈值”。

## 4. 当前状态

目前大地图链路已经形成闭环：

- 可以离线生成 tile 级 walkable cells
- 可以基于 tile 级 topology 构建并运行 case
- 运行时可以围绕 tile 稳定加载大地图区域
- pedestrian ego 的采集链路可以在大地图下保持传感器世界位姿正确
- 静态障碍物提取逻辑已经切换到基于地面 clearance 的判定方式

后续如果继续扩展，重点应放在：

- obstacle 提取质量的进一步验证
- 不同大地图 tile 下的稳定性回归
- 采集结果与障碍物避让效果的系统性检查
