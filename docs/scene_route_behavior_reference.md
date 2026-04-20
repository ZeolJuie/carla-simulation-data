# Scene Route 与 Behavior 说明

这份文档描述当前仓库里 scene-based pedestrian 配置链路的三层语义：

1. `route_type` 有哪些
2. 每种 `route_type` 支持哪些 `behavior`
3. 每种 `behavior` 在运行时是通过哪些参数表达出来的

适用范围：

- `map.scene_id` / `walker.ped_spawner.scene_id`
- `walker.ped_spawner.behavior`
- `walker.ped_spawner.route_type`
- `walker.ped_spawner.route_id`

也就是当前 scene catalog 驱动的行人配置方式。

## 1. 当前 Route Type

当前 route type 定义来自：

- `utils/generate_walkable_cells.py` 中的 `ROUTE_TYPE_DEFINITIONS`
- 导出的 `artifacts/<map>/topology/walkable_cells.json` 中的 `route_type_definitions`

当前一共有 6 类：

| route_type | crossing_required | 含义 |
| --- | --- | --- |
| `sidewalk_through` | `false` | 在一个 sidewalk cell 内沿连续人行道通行 |
| `junction_through` | `false` | 在 junction cell 内沿 sidewalk 几何直行穿过 |
| `crosswalk_cross` | `true` | 经过至少一段有标记的 `CROSSWALK` 完成过街 |
| `unmarked_junction_cross` | `true` | 不走标记斑马线，而通过 `JAYWALKING_JUNCTION` 几何过街 |
| `junction_turn` | `false` | 在 junction cell 内沿 sidewalk 几何完成转向 |
| `junction_u_turn` | `false` | 在 junction 支撑图内掉头，仍然只走 sidewalk 几何 |

说明：

- `crossing_required = true` 表示这条 route 在语义上包含真正的过街段。
- `crossing_required = false` 表示这条 route 只是在 sidewalk / junction 内部移动，不要求进入 crossing 逻辑。

## 2. 每种 Route 支持哪些 Behavior

route 和 behavior 的兼容关系由 `ROUTE_TYPE_SUPPORTED_BEHAVIORS` 定义，并在导出 catalog 时写进每条 route 的：

- `route.attributes.supported_behaviors`

当前映射如下：

| route_type | supported_behaviors |
| --- | --- |
| `sidewalk_through` | `steady_walk`, `slow_walk`, `fast_pass` |
| `junction_through` | `steady_walk`, `slow_walk`, `fast_pass` |
| `junction_turn` | `steady_walk`, `slow_walk`, `fast_pass` |
| `junction_u_turn` | `steady_walk`, `slow_walk`, `fast_pass` |
| `crosswalk_cross` | `cautious_cross`, `normal_cross`, `assertive_cross`, `ignore_gap_cross` |
| `unmarked_junction_cross` | `cautious_cross`, `normal_cross`, `assertive_cross`, `ignore_gap_cross` |

可以把它理解成两大类：

- `walk` 类：纯 sidewalk / junction 内移动
- `cross` 类：真正过街

当前代码里不会把 `steady_walk` 这种 sidewalk 行为分配给 `crosswalk_cross`，也不会把 `normal_cross` 分配给 `sidewalk_through`。

## 3. 当前 Behavior Profile

当前 behavior profile 定义来自：

- `utils/generate_walkable_cells.py` 中的 `BEHAVIOR_PROFILES`
- 导出的 `artifacts/<map>/topology/walkable_cells.json` 中的 `behavior_profiles`

当前一共有 7 种：

| behavior | description | speed_multiplier | crossing_speed_factor | crossing_safety_margin |
| --- | --- | ---: | ---: | ---: |
| `steady_walk` | 标准步行 | `1.0` | `1.5` | `1.5` |
| `slow_walk` | 慢速、保守步行 | `0.8` | `1.3` | `1.8` |
| `fast_pass` | 快速通过 | `1.25` | `1.7` | `1.2` |
| `cautious_cross` | 保守过街 | `0.95` | `1.2` | `2.0` |
| `normal_cross` | 默认过街 | `1.0` | `1.5` | `1.5` |
| `assertive_cross` | 更激进地抢小 gap 过街 | `1.05` | `1.7` | `0.8` |
| `ignore_gap_cross` | 基本忽略 gap 接受判断 | `1.1` | `1.9` | `-0.5` |

这些 profile 不是不同的控制器，也不是不同的 SFM 模型分支。  
当前实现里，它们本质上只是给行人提供一组默认参数。

## 4. Behavior 在运行时如何生效

### 4.1 选路阶段

scene-based spawner 的选路流程是：

1. 从 `walker.ped_spawner.scene_id` 取 scene；若未设置，则退回 `map.scene_id`
2. 读取 scene catalog
3. 根据 `behavior`
4. 可选再加上 `route_type`
5. 可选再加上 `route_id`
6. 从满足条件的 route 中随机选一条

当前过滤条件是：

- `route.attributes.supported_behaviors` 必须包含该 `behavior`
- 如果配置了 `route_type`，route 的 `route_type` 必须匹配
- 如果配置了 `route_id`，route 的 `route_id` 必须精确匹配

所以：

- `behavior` 是必选的语义入口
- `route_type` 是候选子集过滤器
- `route_id` 是精确钉死某条 route 的方式

## 4.2 速度参数如何确定

当前 scene-based 模式里，行人的基础目标速度按下面的优先级确定：

1. 如果 `walker.ped_spawner.speed` 显式配置，则直接使用这个值
2. 否则取 `base_speed`
3. 如果 `base_speed` 没配，则取 `walker.default_speed`
4. 然后乘上 `behavior_profiles[behavior].speed_multiplier`

也就是：

```text
target_speed =
    speed                                  if speed is set
    else base_speed * speed_multiplier

base_speed =
    walker.ped_spawner.base_speed          if set
    else walker.default_speed
```

例子：

- `walker.default_speed = 1.2`
- `behavior = 'slow_walk'`

则默认：

```text
target_speed = 1.2 * 0.8 = 0.96 m/s
```

如果显式写：

```toml
speed = 1.5
```

那么 `speed_multiplier` 就不再参与。

## 4.3 过街速度如何确定

过街时的目标速度不是单独再配置一个绝对值，而是：

```text
crossing_speed = crossing_speed_factor * target_speed
```

它的取值优先级是：

1. `walker.ped_spawner.crossing_speed_factor`
2. 否则用 `behavior profile` 默认值

所以：

- `normal_cross` 默认是 `1.5 * target_speed`
- `assertive_cross` 默认是 `1.7 * target_speed`
- `ignore_gap_cross` 默认是 `1.9 * target_speed`

## 4.4 过街 gap 接受如何确定

`crossing_safety_margin` 影响的是过街前的 gap acceptance 判断。

它的优先级是：

1. `walker.ped_spawner.crossing_safety_margin`
2. 否则用 `behavior profile` 默认值

运行时逻辑在 `check_traffic.py`：

- 行人根据当前位置和 `next_waypoint` 形成一条计划 crossing 轨迹
- 车辆用当前速度线性外推
- 如果行人与车辆会在时间窗内冲突，则需要等待

这个判断里：

```text
更大的 crossing_safety_margin -> 更保守
更小的 crossing_safety_margin -> 更激进
负的 crossing_safety_margin -> 直接视为可过
```

所以：

- `cautious_cross = 2.0`，最保守
- `normal_cross = 1.5`，默认
- `assertive_cross = 0.8`，更愿意抢 gap
- `ignore_gap_cross = -0.5`，当前实现里等价于“不做 gap 检查”

## 4.5 行人模式机如何体现这些参数

当前行人的模式机是：

- `WALKING_SIDEWALK`
- `CHECKING_TRAFFIC`
- `CROSSING_ROAD`
- `ROAD_TO_SIDEWALK`

行为参数进入模式机后的表现是：

- `target_speed`
  - 在 `WALKING_SIDEWALK` 下使用基础目标速度
- `crossing_speed_factor`
  - 切到 `CROSSING_ROAD` 后，速度变成 `crossing_speed_factor * target_speed`
- `crossing_safety_margin`
  - 在 `CHECKING_TRAFFIC` 阶段被 `check_traffic()` 使用

也就是说，当前所谓的 “行为模式” 主要通过这三类参数表达：

- sidewalk 行走速度
- crossing 行走速度
- crossing 前 gap 接受阈值

而不是通过完全不同的路径规划器或完全不同的 SFM 力模型表达。

## 5. 当前用户应该怎么配

### 5.1 用 route_type 选一类路线

```toml
[map]
scene_id = 'crossing_002'
scene_catalog_path = 'artifacts/Town03_Opt/topology/walkable_cells.json'

[walker]
default_speed = 1.2

[[walker.ped_spawner]]
behavior = 'normal_cross'
route_type = 'crosswalk_cross'
quantity = 1
spawn_time = 0.0
```

含义：

- 在 `crossing_002` 这个 scene 内
- 只从 `crosswalk_cross` 里选
- 且 route 必须支持 `normal_cross`

### 5.2 用 route_id 精确指定一条路线

```toml
[[walker.ped_spawner]]
behavior = 'steady_walk'
route_id = 'crossing_002/route_59'
quantity = 1
spawn_time = 3.0
```

含义：

- 不再随机选 route
- 直接使用这条 route
- 但该 route 仍然必须支持 `steady_walk`

### 5.3 覆盖 behavior 默认参数

```toml
[[walker.ped_spawner]]
behavior = 'normal_cross'
route_type = 'crosswalk_cross'
base_speed = 1.2
crossing_speed_factor = 1.8
crossing_safety_margin = 0.7
```

含义：

- route 语义仍然是 `normal_cross`
- 但运行时把这名行人的过街速度和 gap 接受阈值改掉

## 6. 当前实现的边界

这套 behavior 目前表达的是“参数化行为”，不是“策略级行为”。

也就是说，当前不同 `behavior` 之间主要差在：

- 走得快还是慢
- 过街时走得快还是慢
- 愿不愿意等大 gap

当前没有按 `behavior` 区分这些更高层差异：

- 不同的 route completion 策略
- 不同的避障 / 让行风格
- 不同的 side preference
- 不同的 force parameter set

所以如果后面你想把 “行为模式” 做得更强，可以继续扩展到：

- 行为专属的 force 参数
- 行为专属的 scene route 跟踪策略
- 行为专属的 stuck recovery / replanning

## 7. 结论

当前 scene-based 行人配置可以概括为：

- `route_type` 定义路线语义类别
- `supported_behaviors` 定义每条路线允许哪些行为
- `behavior` 提供默认速度与过街 gap 参数
- `walker.ped_spawner` 上的显式参数可以覆盖这些默认值

当前代码下：

- route type 有 `6` 种
- behavior profile 有 `7` 种
- 行为差异主要通过 `speed_multiplier`、`crossing_speed_factor`、`crossing_safety_margin` 来表达

