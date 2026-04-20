# 基于 CARLA SFM 原始项目的场景拓展与编排改造说明

本文档用于沉淀当前阶段在原始 `carla-social-force-model` 项目上的场景相关改造，目标是把“地图中的可行走拓扑单元”逐步升级为“可被语义编排和可复现执行的场景目录”。

文档覆盖：
- 改造目标
- 当前已实现内容
- 场景相关目录与数据结构
- 各阶段核心方法
- 基于当前实现的工作流
- 配套脚本的使用方式
- 当前边界与下一步建议

## 1. 改造目标

原始项目的能力重点是：
- 基于 `path_planner.py` 从 CARLA 地图构建行人可行走图
- 基于 `scenario_config.toml` 手工配置行人生成点、目标点、waypoints
- 运行 Social Force Model 行人仿真

本次改造的目标，是在这个基础上增加一层“场景目录”和“基于场景的编排入口”。

目标链路如下：

1. 从 CARLA 地图中提取行人可运动拓扑。
2. 将地图划分为一组尽量互斥、可复用的候选场景单元 `scene cell`。
3. 在每个可用场景单元上预生成一批 canonical routes。
4. 为 route 绑定允许的 behavior profiles。
5. 在仿真配置层新增：
   - `map.scene_id`
   - 行人 `behavior`
6. 运行时按 `scene_id + behavior` 自动选 route，生成实际可执行的 `spawn_location + waypoints + crossing_road_bools`。

最终目标不是替换原始手工配置方式，而是在其之上增加一条更高层、可自动编排的配置通路。

## 2. 当前阶段已实现内容

截至当前节点，已经实现了以下能力。

### 2.1 候选场景目录导出

已实现脚本：
- [utils/generate_walkable_cells.py](/home/zmh/carla-social-force-model/utils/generate_walkable_cells.py)

脚本能够：
- 连接 CARLA，读取 `PedPathPlanner` 的行人拓扑图
- 提取以下三类行人相关 edge type：
  - `SIDEWALK`
  - `CROSSWALK`
  - `JAYWALKING_JUNCTION`
- 忽略：
  - `JAYWALKING`
  - `SIDEWALK_TO_ROAD`
- 将地图划分为：
  - `crossing_*`
  - `sidewalk_*`

当前划分策略是：
- 先用 `CROSSWALK + JAYWALKING_JUNCTION` 作为 crossing seed，吸收局部连接的 sidewalk，形成 `crossing cell`
- 剩余纯 sidewalk 再按 band 合并与长度切分，形成 `sidewalk cell`
- sidewalk cell 会计算 `longitudinal_length_m`
- 若 `< 30m`，则标为 `usable = false`

导出结果：
- `walkable_cells.json`
- `walkable_cells.svg`

### 2.2 Route catalog 导出

在 `walkable_cells.json` 中，每个 `usable cell` 已经带：
- `portals`
- `routes`
- `route_stats`
- `route_support`

当前 route type 已定义为：
- `sidewalk_through`
- `junction_through`
- `crosswalk_cross`
- `unmarked_junction_cross`
- `junction_turn`
- `junction_u_turn`

当前 behavior profiles 已定义为：
- `steady_walk`
- `slow_walk`
- `fast_pass`
- `cautious_cross`
- `normal_cross`
- `assertive_cross`
- `ignore_gap_cross`

route 与 behavior 是一对多关系，不是绑定关系。

### 2.3 Scene-based 仿真配置入口

已实现文件：
- [scene_catalog.py](/home/zmh/carla-social-force-model/scene_catalog.py)
- [pedestrian_spawner.py](/home/zmh/carla-social-force-model/pedestrian_spawner.py)
- [carla_simulation.py](/home/zmh/carla-social-force-model/carla_simulation.py)

新增能力：
- `[map]` 下支持 `scene_id`
- `[map]` 下支持 `scene_catalog_path`
- `[map]` 下支持 `carla_timeout`
- `[walker]` 下支持 `default_speed`
- `[[walker.ped_spawner]]` 支持 `behavior`
- `[[walker.ped_spawner]]` 可选支持 `route_type`
- `[[walker.ped_spawner]]` 可选支持 `route_id`
- `[[walker.ped_spawner]]` 可选支持 `scene_id`

运行时逻辑：
- 读取 scene catalog
- 根据 `scene_id` 找到 cell
- 从 cell 的 `routes` 中筛选出支持该 `behavior` 的 routes
- 若额外指定了 `route_type` 或 `route_id`，继续过滤
- 随机选一条 route
- 将 route 转换为原始运行时所需的：
  - `spawn_location`
  - `config_waypoints`
  - `config_crossing_road_bools`

因此，原始仿真运行时接口没有被破坏，只是在 `pedestrian_spawner` 上层新增了一个“scene 编译层”。

## 3. 场景相关目录结构

当前与场景改造直接相关的文件如下：

```text
carla-social-force-model/
├── SCENARIO.md
├── SCENE_TOPOLOGY_PIPELINE.md
├── scene_catalog.py
├── pedestrian_spawner.py
├── carla_simulation.py
├── config/
│   └── scenarios/
│       └── scene_id_scenario_config.toml
├── utils/
│   ├── generate_walkable_cells.py
│   └── render_walkable_cell_svg.py
└── artifacts/
    └── Town03_Opt/
        └── topology/
            ├── walkable_cells.json
            ├── walkable_cells.svg
            ├── crossing_002_routes.svg
            └── crossing_024_routes.svg
```

各文件职责：

- [SCENARIO.md](/home/zmh/carla-social-force-model/SCENARIO.md)
  - 语义场景的数学化定义
- [utils/generate_walkable_cells.py](/home/zmh/carla-social-force-model/utils/generate_walkable_cells.py)
  - 从地图拓扑导出候选场景目录
- [utils/render_walkable_cell_svg.py](/home/zmh/carla-social-force-model/utils/render_walkable_cell_svg.py)
  - 可视化单个 cell 的 2D 可行走范围与 routes
- [scene_catalog.py](/home/zmh/carla-social-force-model/scene_catalog.py)
  - 运行时对 scene catalog 的读取、筛选与验证
- [pedestrian_spawner.py](/home/zmh/carla-social-force-model/pedestrian_spawner.py)
  - 将 `scene_id + behavior` 编译为实际行人生成参数
- [config/scenarios/scene_id_scenario_config.toml](/home/zmh/carla-social-force-model/config/scenarios/scene_id_scenario_config.toml)
  - 新能力的示例配置
- [artifacts/Town03_Opt/topology/walkable_cells.json](/home/zmh/carla-social-force-model/artifacts/Town03_Opt/topology/walkable_cells.json)
  - Town03_Opt 的当前候选场景目录

## 4. 核心数据结构

### 4.1 path planner 底层图

底层仍然基于原始 [path_planner.py](/home/zmh/carla-social-force-model/path_planner.py)。

在本次导出脚本中，planner edge 被重新包装为 `EdgeRecord`：
- 定义位置：[generate_walkable_cells.py](/home/zmh/carla-social-force-model/utils/generate_walkable_cells.py#L105)

核心字段：
- `edge_id`
- `n1`, `n2`
- `edge_type`
- `length`
- `road_id`, `section_id`, `lane_id`
- `in_junction`
- `entry_lane_type`, `exit_lane_type`
- `x1`, `y1`, `x2`, `y2`
- `width_hint`

语义说明：
- `n1/n2` 是图节点 id
- `edge_type` 是路线段语义，而不是 waypoint 语义
- `SW/CW/JWJ` 对应的是 edge type，不是点属性

### 4.2 候选场景 catalog 顶层结构

`walkable_cells.json` 顶层字段：

```json
{
  "map_name": "Town03_Opt",
  "catalog_type": "candidate_scene_catalog",
  "cells_are_candidate_scenes": true,
  "rules": {...},
  "route_type_definitions": {...},
  "behavior_profiles": {...},
  "stats": {...},
  "sidewalk_longitudinal_stats": {...},
  "candidate_scene_stats": {...},
  "route_stats": {...},
  "cells": [...]
}
```

语义说明：
- `rules`：说明当前导出使用的划分规则
- `route_type_definitions`：route 类型定义
- `behavior_profiles`：行为 profile 参数
- `stats`：整体划分统计
- `cells`：候选场景实体列表

### 4.3 Cell 结构

单个 cell 结构示意：

```json
{
  "id": "crossing_002",
  "kind": "crossing",
  "color": "#xxxxxx",
  "edge_count": 123,
  "total_length_m": 321.0,
  "longitudinal_length_m": null,
  "centroid": [x, y],
  "bounds": {...},
  "road_ids": [...],
  "junction_road_ids": [...],
  "lane_keys": [...],
  "attributes": {
    "usable": true,
    "usable_reason": "usable",
    "has_junction_context": true,
    "lane_count": 0,
    "sidewalk_min_longitudinal_length_m": null
  },
  "edge_type_counts": {...},
  "edges": [...],
  "route_support": {...},
  "portals": [...],
  "routes": [...],
  "route_stats": {...}
}
```

说明：
- `kind` 当前只有 `crossing` 和 `sidewalk`
- `usable` 用于后续场景编排过滤
- `longitudinal_length_m` 只对 sidewalk cell 有意义

### 4.4 Route 结构

单个 route 结构示意：

```json
{
  "route_id": "crossing_002/route_01",
  "route_type": "crosswalk_cross",
  "route_length_m": 56.832,
  "start_portal": "crossing_002_portal_03",
  "end_portal": "crossing_002_portal_07",
  "path_source": "shortest_path",
  "node_ids": [...],
  "edge_ids": [...],
  "path_waypoints": [...],
  "path_crossing_road_bools": [...],
  "spawn_location": [...],
  "destination": [...],
  "config_waypoints": [...],
  "config_crossing_road_bools": [...],
  "edge_type_counts": {...},
  "attributes": {
    "uses_crosswalk": true,
    "uses_unmarked_junction_cross": false,
    "crossing_required": true,
    "supported_behaviors": [
      "cautious_cross",
      "normal_cross",
      "assertive_cross",
      "ignore_gap_cross"
    ]
  }
}
```

重点：
- `path_waypoints` 是完整路径
- `config_waypoints` 是直接喂给运行时配置的 waypoint 列表
- `spawn_location` 是行人起点
- `config_crossing_road_bools` 与原始项目运行时格式对齐

### 4.5 Portal 结构

当前 crossing 路由生成使用的是“叶子节点 portal”：

```json
{
  "portal_id": "crossing_002_portal_01",
  "node_ids": [123],
  "centroid": [x, y, z],
  "kind": "leaf_node"
}
```

当前语义：
- `portal` 是 route-support 子图上的边界出入口
- 当前版本不再使用几何聚类 portal，而是更直接地使用叶子节点作为入口出口

## 5. 当前 route type 与 behavior 的对应关系

定义位置：
- [generate_walkable_cells.py](/home/zmh/carla-social-force-model/utils/generate_walkable_cells.py#L18)
- [generate_walkable_cells.py](/home/zmh/carla-social-force-model/utils/generate_walkable_cells.py#L45)
- [generate_walkable_cells.py](/home/zmh/carla-social-force-model/utils/generate_walkable_cells.py#L90)

对应关系如下：

- `sidewalk_through`
  - `steady_walk`
  - `slow_walk`
  - `fast_pass`
- `junction_through`
  - `steady_walk`
  - `slow_walk`
  - `fast_pass`
- `junction_turn`
  - `steady_walk`
  - `slow_walk`
  - `fast_pass`
- `junction_u_turn`
  - `steady_walk`
  - `slow_walk`
  - `fast_pass`
- `crosswalk_cross`
  - `cautious_cross`
  - `normal_cross`
  - `assertive_cross`
  - `ignore_gap_cross`
- `unmarked_junction_cross`
  - `cautious_cross`
  - `normal_cross`
  - `assertive_cross`
  - `ignore_gap_cross`

## 6. 各阶段核心方法

### 6.1 地图与 planner 图读取

核心入口：
- [load_world()](/home/zmh/carla-social-force-model/utils/generate_walkable_cells.py#L222)
- [build_edge_records()](/home/zmh/carla-social-force-model/utils/generate_walkable_cells.py#L238)

作用：
- 连接 CARLA world
- 构建 `PedPathPlanner`
- 将原始 planner topology 转成 `EdgeRecord`
- 建立 adjacency

### 6.2 Cell 划分

核心入口：
- [partition_walkable_cells()](/home/zmh/carla-social-force-model/utils/generate_walkable_cells.py#L1399)

内部关键逻辑：
- crossing 优先保留
- sidewalk 再切块
- 纯 sidewalk lane 先按 band 合并，再切段
- 短段尝试与相邻区段合并

关键方法：
- `build_sidewalk_assignments(...)`
- [merge_short_sidewalk_chunks()](/home/zmh/carla-social-force-model/utils/generate_walkable_cells.py#L685)
- [build_cell_record()](/home/zmh/carla-social-force-model/utils/generate_walkable_cells.py#L731)

规则：
- crossing 单元优先，防止与 sidewalk 冲突
- sidewalk 使用 `longitudinal_length_m` 作为后续场景可用性的阈值依据

### 6.3 Route 分类

核心方法：
- [route_type_for_path()](/home/zmh/carla-social-force-model/utils/generate_walkable_cells.py#L1019)

分类逻辑：
- 若路径经过 `CROSSWALK` -> `crosswalk_cross`
- 若路径经过 `JAYWALKING_JUNCTION` -> `unmarked_junction_cross`
- 若全部为 `SIDEWALK`：
  - 偏转角小 -> `junction_through`
  - 偏转角大且满足纵向回退、横向偏移小 -> `junction_u_turn`
  - 否则 -> `junction_turn`

几何辅助方法：
- `path_deflection_degrees(...)`
- `path_endpoint_frame(...)`

当前 `u_turn` 的补充判据：
- 不能只看方向反转
- 还要看“横向位移是否足够小”
- 避免大型 U 形绕行被误判为 `u_turn`

### 6.4 Route 生成

核心方法：
- [build_route_support_for_cell()](/home/zmh/carla-social-force-model/utils/generate_walkable_cells.py#L1102)
- [build_leaf_portals()](/home/zmh/carla-social-force-model/utils/generate_walkable_cells.py#L1126)
- `build_cell_routes(...)`
- [attach_routes_to_cells()](/home/zmh/carla-social-force-model/utils/generate_walkable_cells.py#L1339)

当前 crossing route 生成逻辑：

1. 使用 cell 自身作为 route-support 子图。
2. 在该子图中找 `degree == 1` 的叶子节点，作为 portal。
3. 对 portal 两两求最短路。
4. 根据路径经过的 edge type 和几何形状进行分类。
5. 写入 route catalog。

当前实现特征：
- route 完全来自 `path_planner` 图
- 不手造路线
- 但 crossing cell 上会出现较多 leaf-node pair 组合，route 数量偏大

### 6.5 单 cell 路线可视化

核心方法：
- [render_cell_svg()](/home/zmh/carla-social-force-model/utils/render_walkable_cell_svg.py#L121)

脚本：
- [utils/render_walkable_cell_svg.py](/home/zmh/carla-social-force-model/utils/render_walkable_cell_svg.py)

当前图中语义：
- `SW` = `SIDEWALK`
- `CW` = `CROSSWALK`
- `JWJ` = `JAYWALKING_JUNCTION`
- route 用不同颜色和虚线样式区分
- portal 用蓝点标记

### 6.6 Scene-based 运行时选择

核心方法：
- [SceneCatalog.from_map_config()](/home/zmh/carla-social-force-model/scene_catalog.py#L29)
- [SceneCatalog.get_scene()](/home/zmh/carla-social-force-model/scene_catalog.py#L50)
- [SceneCatalog.get_routes()](/home/zmh/carla-social-force-model/scene_catalog.py#L62)
- [PedSpawnManager._load_scene_catalog_if_needed()](/home/zmh/carla-social-force-model/pedestrian_spawner.py#L124)
- [PedSpawnManager._spawn_point_uses_scene_route()](/home/zmh/carla-social-force-model/pedestrian_spawner.py#L139)
- [PedSpawnManager._generate_scene_ped_spawners()](/home/zmh/carla-social-force-model/pedestrian_spawner.py#L188)

运行时行为：

1. 读取 `[map]` 中的 `scene_id` 和 `scene_catalog_path`
2. 加载 `walkable_cells.json`
3. 对每个声明了 `behavior` 的 `ped_spawner`
4. 在 scene 内筛选支持该 behavior 的 routes
5. 随机选一条
6. 编译成原始 `PedSpawner`

这个设计的关键点是：
- 新能力没有改原始 SFM 运行内核
- 只是改了“如何构造 ped_spawner”

## 7. 当前工作流

### 7.1 第一步：导出候选场景目录

启动 CARLA 后执行：

```bash
export PYTHONPATH=$PYTHONPATH:/home/zmh/carla/PythonAPI/carla
conda run -n carla python utils/generate_walkable_cells.py \
  --map-name Town03_Opt \
  --timeout 240 \
  --approach-length 20 \
  --sidewalk-chunk-length 50 \
  --sidewalk-tail-merge-length 15 \
  --sidewalk-min-longitudinal-length 30
```

默认输出：
- `artifacts/Town03_Opt/topology/walkable_cells.json`
- `artifacts/Town03_Opt/topology/walkable_cells.svg`

这个步骤的产出就是“场景目录源数据”。

### 7.2 第二步：查看整体划分结果

直接查看：
- [walkable_cells.svg](/home/zmh/carla-social-force-model/artifacts/Town03_Opt/topology/walkable_cells.svg)

它用于确认：
- 划分是否互斥
- crossing / sidewalk 是否分布合理
- 是否有明显的重复或重叠块

### 7.3 第三步：查看单个 cell 的 route 可视化

示例：

```bash
export PYTHONPATH=$PYTHONPATH:/home/zmh/carla/PythonAPI/carla
conda run -n carla python utils/render_walkable_cell_svg.py \
  --map-name Town03_Opt \
  --cell-id crossing_002 \
  --timeout 240
```

默认输出：
- `artifacts/Town03_Opt/topology/crossing_002_routes.svg`

这个步骤用于：
- 检查某个 crossing / sidewalk 单元的 routes 是否合理
- 检查 turn / through / cross 的分类

### 7.4 第四步：编写 scene-based scenario config

当前示例配置在：
- [scene_id_scenario_config.toml](/home/zmh/carla-social-force-model/config/scenarios/scene_id_scenario_config.toml)

示例结构：

```toml
[map]
map_name = 'Town03_Opt'
map_path = 'Carla/Maps/'
carla_timeout = 30.0
scene_id = 'crossing_002'
scene_catalog_path = 'artifacts/Town03_Opt/topology/walkable_cells.json'

[walker]
pedestrian_seed = 2026
default_speed = 1.2

[[walker.ped_spawner]]
behavior = 'normal_cross'
route_type = 'crosswalk_cross'

[[walker.ped_spawner]]
behavior = 'cautious_cross'

[[walker.ped_spawner]]
behavior = 'steady_walk'
```

规则：
- 如果写了 `behavior`，表示该 spawner 启用 scene-based 模式
- 不能再同时写：
  - `spawn_location`
  - `destination`
  - `generate_route`
  - `waypoints`

### 7.5 第五步：运行仿真

```bash
export PYTHONPATH=$PYTHONPATH:/home/zmh/carla-social-force-model:/home/zmh/carla/PythonAPI/carla
conda run -n carla python run_simulation.py \
  --scenario-config config/scenarios/scene_id_scenario_config.toml
```

运行时行为：
- 根据 `scene_id` 取 scene
- 根据 `behavior` 取 route 候选
- 随机选 route
- 自动生成底层行人路线配置
- 进入原始 SFM 仿真主循环

## 8. 已知边界

### 8.1 当前 route 数量偏多

原因：
- crossing route 现在使用“叶子节点两两最短路”
- 对复杂路口会产生较多组合路径

后续可能需要：
- representative route 压缩
- family 级 route 去重
- route rank / priority

### 8.2 当前 scene cell 仍是几何拓扑单元，不是最终语义场景

当前 `scene_id` 代表的是：
- 一个候选拓扑单元

它还不是：
- 用户文本语义场景的最终编译结果

后续还需要在这个基础上再加一层：
- semantic scenario compiler

### 8.3 CARLA 服务稳定性影响最终 smoke 验证

当前代码层已经完成：
- scene catalog 导出
- scene-based route selection
- 示例配置接线

但 end-to-end 仿真是否能直接跑通，还受 CARLA server 是否稳定可响应影响。

## 9. 当前节点的结论

到当前阶段，本项目已经从“手工 waypoint 配置”前进到：

- 地图 walkable topology 自动切分为 candidate scenes
- 每个 scene 自动带 route catalog
- route 与 behavior profile 已解耦
- 仿真配置层已支持 `scene_id + behavior`
- 运行时已能随机挑 route 并编译成原始项目可执行格式

也就是说，场景编排链路已经具备了一个可用的中间层：

`地图拓扑 -> candidate scenes -> routes -> behaviors -> 运行时 ped spawner`

下一步更合理的工作，将是继续往上做：
- 文本语义到 `scene_id + actors + behaviors` 的编译
- route family 压缩与更稳定的场景代表性
- scene 与车辆/ego 任务的统一编排
