# 场景语义 Registry

这份文档用于约束一个“场景语义描述 agent”直接生成当前仓库可运行的最终 `scenario_config.toml`。

目标不是引入新的编译器或模板系统，而是把 agent 的自然语言生成收敛成一组受控语义槽位，使它可以：

- 自动化生成场景描述
- 批量覆盖不同场景组合
- 直接落到当前仓库支持的最终 TOML 配置
- 避免写出当前实现不支持的组合

适用范围：

- `run_simulation.py`
- scene-based pedestrian route selection
- 固定小组 group
- background vehicle population
- weather generator
- sampling / sequence collection

相关参考：

- [README.md](../README.md)
- [SCENE_TOPOLOGY_PIPELINE.md](../SCENE_TOPOLOGY_PIPELINE.md)
- [docs/scene_route_behavior_reference.md](./scene_route_behavior_reference.md)

---

## 1. 核心原则

这个 agent 不需要使用模板，也不需要中间编译器，但必须遵守下面 4 条原则：

1. 自然语言描述可以自由修饰，但必须能归一化为固定语义槽位。
2. 所有关键语义都必须映射到当前仓库已有字段，不发明新能力。
3. 优先使用 `scene_id + route_id + behavior`，不优先写自由坐标。
4. 最终输出是当前仓库可直接运行的最终 TOML，而不是中间格式。

一句话概括：

> 自然语言只是表层，底层必须是有限可选项组合。

---

## 2. 标准语义槽位

agent 在描述一个场景时，至少要覆盖下面这些语义槽位。

| 槽位 | 含义 | 最终主要映射 |
| --- | --- | --- |
| `weather` | 天气、时间段、云量、雨雾 | `[weather]` |
| `map_type` | 城市 / 城镇 / 开阔郊区等地图类型 | `map.map_name` |
| `topology` | 人行道 / 路口区域 | `map.scene_id` + `route_type` |
| `ped_action` | 普通行人整体怎么走 | `walker.ped_spawner.behavior` |
| `group` | 是否有结伴行人以及结伴形式 | `quantity` + `formation` + shared `route_id` |
| `ped_density` | 场景整体行人稀疏还是拥挤 | pedestrian actor 数量 |
| `vehicle_density` | 背景车流强度 | `[vehicle.background_population]` |
| `ego_action` | 主体行人的行动模式 | ego 那个 ped spawner 的 `behavior` |
| `sampling` | 采集 rig 和 policy | `[sampling]` |

agent 可以把这些槽位揉成一段更自然的场景描述，但不能漏掉其语义。

---

## 3. 自然语言自由度

允许：

- 自由修饰场景，例如“阴天、略带潮湿的城市人行道”“一个相对空闲但仍有背景车流的路口”
- 调整句序
- 增加少量合理修辞
- 用“结伴同行”“并排行走”“尾随通行”“正常过街”“快速通过”这类自然短语表达行为

不允许：

- 描述当前代码不支持的复杂群体语义
- 描述需要中途新增 actor 才能成立的车流逻辑
- 说“群体一定不会被穿插”这类当前实现不保证的结果
- 说“严格同步协同过街”这类当前没有实现的 group 决策

推荐把最终描述理解成下面这个归一化元组：

```text
(
  weather,
  map_type,
  topology,
  route_type,
  ped_action,
  group,
  ped_density,
  vehicle_density,
  ego_action
)
```

---

## 4. 槽位白名单

### 4.1 地图类型 `map_type`

`map_type` 是语义别名，不是 TOML 原生字段。  
agent 必须先选语义地图类型，再从该类型允许的 `map_name` 中选一个实际地图。

建议的初始分组如下：

| map_type | 含义 | 推荐 `map_name` |
| --- | --- | --- |
| `urban_dense` | 稠密城市、道路组织复杂、适合过街和背景车流 | `Town10HD_Opt`, `Town05` |
| `town_grid` | 中小城镇、规则街区、适合基础 sidewalk / junction 场景 | `Town01`, `Town02`, `Town03` |
| `suburban_open` | 较开阔、住宅/郊区感更强 | `Town04`, `Town06`, `Town07` |

约束：

- agent 最终必须写具体 `map.map_name`
- `map_type` 只是选择依据，不直接写入 config
- 如果没有覆盖要求，默认优先 `Town10HD_Opt`

### 4.2 拓扑类型 `topology`

支持 2 类语义一级拓扑：

| topology | 含义 | 约束 |
| --- | --- | --- |
| `sidewalk` | 纯人行道走行 | 选 `kind=sidewalk` 的 scene |
| `junction` | 路口区域，包含路口过街、路口内直行、转向或掉头 | scene 中至少要有 `crosswalk_cross`, `unmarked_junction_cross` 或 `junction_*` route |

约束：

- 不再使用 `topology=crossing`
- 有无标记过街语义由 `route_type` 表达，而不是由一级 `topology` 表达
- 自然语言里的“过街”“穿过路口”“走斑马线”都应归一化为 `topology=junction`，再选择对应 `route_type`

### 4.3 Route Type

当前仓库支持的 `route_type` 只有 6 个：

| route_type | 语义 |
| --- | --- |
| `sidewalk_through` | 人行道连续通行 |
| `junction_through` | 路口内沿 sidewalk 几何直行 |
| `crosswalk_cross` | 走有标记斑马线过街 |
| `unmarked_junction_cross` | 走无标记路口过街 |
| `junction_turn` | 路口内转向 |
| `junction_u_turn` | 路口内掉头 |

### 4.4 行人行为 `behavior`

当前支持的 `behavior` 只有 7 个：

| behavior | 语义 |
| --- | --- |
| `steady_walk` | 标准步行 |
| `slow_walk` | 慢速保守步行 |
| `fast_pass` | 快速通过 |
| `cautious_cross` | 保守过街 |
| `normal_cross` | 默认过街 |
| `assertive_cross` | 较激进过街 |
| `ignore_gap_cross` | 基本忽略 gap 接受判断 |

### 4.5 群体 `group`

当前 group 只支持固定小组：

| group 语义 | 配置映射 |
| --- | --- |
| `none` | 不配置 formation group |
| `pair_follow` | `quantity=2`, `formation='follow'` |
| `pair_abreast` | `quantity=2`, `formation='abreast'` |
| `triple_follow` | `quantity=3`, `formation='follow'` |
| `triple_abreast` | `quantity=3`, `formation='abreast'` |

group 硬约束：

- 只支持 `2` 或 `3` 人
- 必须共享同一个 `route_id`
- 必须共享同一个 `behavior`
- `formation` 只支持 `follow` 和 `abreast`

### 4.6 行人密度 `ped_density`

这不是 TOML 原生枚举，而是 agent 选 actor 数量时的语义层。

建议映射：

| ped_density | 推荐总人数范围 | 说明 |
| --- | --- | --- |
| `sparse` | `1 - 3` | 以 ego 为主，少量配角 |
| `medium` | `4 - 8` | 典型中等行人流 |
| `crowded` | `9 - 16` | 明显拥挤，但仍应保持可运行 |

约束：

- `crowded` 不是越多越好，优先保证 scene 可完成
- 如果已经有 group，人数统计按组内总人数计

### 4.7 背景车流 `vehicle_density`

背景车辆只通过 `[vehicle.background_population]` 表达。

建议映射：

| vehicle_density | 配置建议 |
| --- | --- |
| `none` | 不写 `[vehicle.background_population]` 或 `enabled=false` |
| `low` | `local_count=4`, `global_count=8` |
| `medium` | `local_count=8`, `global_count=16` |
| `high` | `local_count=12`, `global_count=24` |

固定规则：

- 使用固定 `seed`
- 车辆全部在仿真开始时一次性生成
- 采集后不再新增 vehicle actor

### 4.8 天气 `weather`

天气不使用 `weather_preset`，只使用当前 weather generator。

建议把天气再拆成 4 个子槽位：

| 子槽位 | 可选值 |
| --- | --- |
| `time_of_day` | `day`, `low_sun`, `twilight`, `night` |
| `cloudiness_level` | `clear`, `mid`, `overcast` |
| `rain_regime` | `none`, `drizzle`, `rain`, `heavy` |
| `fog_level` | `none`, `light` |

### 4.9 采集配置 `sampling`

当前推荐固定用已经存在的 rig / policy：

| 语义 | 推荐值 |
| --- | --- |
| sensor rig | `config/sensors/default_multimodal.toml` |
| sampling policy | `config/sampling/default.toml` |

如果不是专门做采集，可以省略 `[sampling]`。  
如果是数据集生成，默认应启用。

### 4.10 Actor Blueprint

当前仓库已经支持为 pedestrian 和 vehicle 显式指定 CARLA blueprint。

#### Pedestrian blueprint

- 配置字段：`walker.ped_spawner.blueprint`
- 生效范围：普通 ped spawner 和 scene-based ped spawner 都支持
- 不写时：从当前 CARLA world 的 walker blueprint library 中随机选择

约束：

- 当前 group / formation 的 `blueprint` 是按 **spawner 级别** 配置的，不是按成员单独配置
- 也就是说，一个 `quantity=2/3` 的 formation group 会共享同一个 ped blueprint

命名规则：

- pedestrian blueprint 使用 CARLA 原生 id：`walker.pedestrian.XXXX`

当前 live CARLA (`Carla/Maps/Town10HD_Opt`) 中已实际导出的 pedestrian blueprint 共 `51` 个：

- `walker.pedestrian.0001`
- `walker.pedestrian.0002`
- `walker.pedestrian.0003`
- `walker.pedestrian.0004`
- `walker.pedestrian.0005`
- `walker.pedestrian.0006`
- `walker.pedestrian.0007`
- `walker.pedestrian.0008`
- `walker.pedestrian.0009`
- `walker.pedestrian.0010`
- `walker.pedestrian.0011`
- `walker.pedestrian.0012`
- `walker.pedestrian.0013`
- `walker.pedestrian.0014`
- `walker.pedestrian.0015`
- `walker.pedestrian.0016`
- `walker.pedestrian.0017`
- `walker.pedestrian.0018`
- `walker.pedestrian.0019`
- `walker.pedestrian.0020`
- `walker.pedestrian.0021`
- `walker.pedestrian.0022`
- `walker.pedestrian.0023`
- `walker.pedestrian.0024`
- `walker.pedestrian.0025`
- `walker.pedestrian.0026`
- `walker.pedestrian.0027`
- `walker.pedestrian.0028`
- `walker.pedestrian.0029`
- `walker.pedestrian.0030`
- `walker.pedestrian.0031`
- `walker.pedestrian.0032`
- `walker.pedestrian.0033`
- `walker.pedestrian.0034`
- `walker.pedestrian.0035`
- `walker.pedestrian.0036`
- `walker.pedestrian.0037`
- `walker.pedestrian.0038`
- `walker.pedestrian.0039`
- `walker.pedestrian.0040`
- `walker.pedestrian.0041`
- `walker.pedestrian.0042`
- `walker.pedestrian.0043`
- `walker.pedestrian.0044`
- `walker.pedestrian.0045`
- `walker.pedestrian.0046`
- `walker.pedestrian.0047`
- `walker.pedestrian.0048`
- `walker.pedestrian.0049`
- `walker.pedestrian.0050`
- `walker.pedestrian.0051`

所有 pedestrian blueprint 的 attribute schema 一致，字段如下：

| attribute | type | modifiable | 说明 |
| --- | --- | --- | --- |
| `generation` | `Int` | `false` | 代际标识 |
| `gender` | `String` | `false` | 性别，`male` / `female` |
| `age` | `String` | `false` | 年龄段，`child` / `teenager` / `adult` |
| `is_invincible` | `Bool` | `true` | 当前导出里推荐值为 `true` |
| `ros_name` | `String` | `true` | 默认与 blueprint id 一致 |
| `speed` | `Float` | `true` | 推荐速度档位，因 actor 类型不同而不同 |
| `role_name` | `String` | `true` | 默认值 `pedestrian` |

按 `gender / age / speed` 分组后的摘要如下：

| 组别 | blueprint 数量 | speed 推荐值 | blueprint 范围 |
| --- | --- | --- | --- |
| `female adult` | `6` | `0.0, 1.8, 4.0` | `0001, 0005, 0006, 0007, 0008, 0015` |
| `female adult` | `15` | `0.0, 1.7, 4.0` | `0019-0023, 0031-0032, 0035-0037, 0040-0044` |
| `male adult` | `20` | `0.0, 1.7, 4.0` | `0002-0004, 0016-0018, 0024-0030, 0033-0034, 0038-0039, 0045-0047` |
| `female child` | `5` | `0.0, 1.1, 2.0` | `0009-0011, 0048-0049` |
| `male child` | `3` | `0.0, 1.1, 2.0` | `0012-0014` |
| `female teenager` | `1` | `0.0, 1.1, 2.0` | `0050` |
| `male teenager` | `1` | `0.0, 0.85, 2.0` | `0051` |

说明：

- 当前仓库代码支持任意当前 world blueprint library 中存在的 `walker.pedestrian.*`
- 如果你想做语义上更稳定的数据集配置，建议优先按上面的 `gender / age / speed` 分组来选 blueprint，而不是只随机写一个 id

#### Vehicle blueprint

- 配置字段：`vehicle.vehicle_spawner.blueprint`
- 生效范围：手动 vehicle spawner
- 不写时：从当前 CARLA world 的 vehicle blueprint library 中随机选择

背景车流额外支持：

- `vehicle.background_population.blueprint = "vehicle.xxx"`
- 或 `vehicle.background_population.blueprints = ["vehicle.audi.a2", "vehicle.tesla.model3", ...]`

当前本地 CARLA 安装中可用、且已从本地安装内容核实的 vehicle blueprint 候选如下：

四轮车辆：

- `vehicle.audi.a2`
- `vehicle.audi.etron`
- `vehicle.audi.tt`
- `vehicle.bmw.grandtourer`
- `vehicle.carlamotors.carlacola`
- `vehicle.chevrolet.impala`
- `vehicle.citroen.c3`
- `vehicle.dodge.charger_police`
- `vehicle.ford.mustang`
- `vehicle.jeep.wrangler_rubicon`
- `vehicle.lincoln.mkz_2017`
- `vehicle.mercedes.coupe`
- `vehicle.micro.microlino`
- `vehicle.mini.cooper_s`
- `vehicle.nissan.micra`
- `vehicle.nissan.patrol`
- `vehicle.seat.leon`
- `vehicle.tesla.cybertruck`
- `vehicle.tesla.model3`
- `vehicle.toyota.prius`
- `vehicle.volkswagen.t2`

两轮车辆 / 自行车：

- `vehicle.bh.crossbike`
- `vehicle.diamondback.century`
- `vehicle.gazelle.omafiets`
- `vehicle.harley-davidson.low_rider`
- `vehicle.kawasaki.ninja`
- `vehicle.yamaha.yzf`

约束：

- 如果 `vehicle.no_bikes = true`，则随机 vehicle blueprint 池会自动过滤成四轮车辆
- 如果想固定某几类背景车，优先在 `background_population.blueprints` 中显式给出白名单

---

## 5. 槽位兼容性规则

### 5.1 topology 和 route_type

| topology | 允许 route_type |
| --- | --- |
| `sidewalk` | `sidewalk_through` |
| `junction` | `crosswalk_cross`, `unmarked_junction_cross`, `junction_through`, `junction_turn`, `junction_u_turn` |

### 5.2 route_type 和 behavior

这是当前最关键的兼容规则：

| route_type | allowed behavior |
| --- | --- |
| `sidewalk_through` | `steady_walk`, `slow_walk`, `fast_pass` |
| `junction_through` | `steady_walk`, `slow_walk`, `fast_pass` |
| `junction_turn` | `steady_walk`, `slow_walk`, `fast_pass` |
| `junction_u_turn` | `steady_walk`, `slow_walk`, `fast_pass` |
| `crosswalk_cross` | `cautious_cross`, `normal_cross`, `assertive_cross`, `ignore_gap_cross` |
| `unmarked_junction_cross` | `cautious_cross`, `normal_cross`, `assertive_cross`, `ignore_gap_cross` |

禁止：

- `sidewalk_through + normal_cross`
- `crosswalk_cross + steady_walk`
- `junction_turn + normal_cross`
- `topology=crossing`

### 5.3 group 和 route

如果 `group != none`，则：

- 必须显式写 `route_id`
- 不能只写 `route_type`
- 全组用同一个 `route_id`
- 全组用同一个 `behavior`

### 5.4 ego 和 group

如果 ego 在 group 内：

- `ego = true`
- `ego_member_index` 必须显式写
- 全文件只能有一个 ego

### 5.5 车辆和采集

背景车规则：

- 只允许 `[vehicle.background_population]`
- 不推荐和大量 `[[vehicle.vehicle_spawner]]` 混用
- 采集阶段不新增 actor

### 5.6 采集时序和 actor 生命周期

如果目标是生成最终用于数据集采集的场景，而不是普通演示场景，agent 必须额外遵守下面这些时序规则。

采集开始前：

- ego 应尽量最后一个入场
- ego 的 `spawn_time` 应控制在场景开始后 `10s` 内
- 非 ego actor 可以更早入场，用来形成更自然的背景状态
- 不要求所有 actor 都在 `t=0` 生成
- 允许普通行人和背景车在 ego 入场前已经在场景中运动

采集时长：

- 目标采集窗口应尽量长，推荐 `>= 20s`
- agent 应优先选择更长的 ego route，避免 ego 路径过短
- 如果一个 `scene_id` 下可选 route 较短，则不应用于正式采集配置
- 对 ego 场景，优先选 `sidewalk_through` 或较长的 `crosswalk_cross / unmarked_junction_cross / junction_*` route

采集结束前 actor 销毁：

- 当前运行时不强制“任意 actor 销毁即停止采集”
- 允许 actor 在采集期间自然完成并从 world 中消失
- 这类消失事件应被视为后处理阶段要检查的对象级有效性问题，而不是配置阶段必须完全消除的问题
- 因此，agent 不需要为了避免销毁而把所有 actor 都堆到 `spawn_time = 0.0`

后处理假设：

- `labels.json` 中某个 actor 在后续帧消失，不自动视为配置错误
- 后处理可以根据最后一次出现帧、视野覆盖、lidar 点云命中和遮挡判断是否需要：
  - 截断该序列
  - 或为后续帧补充 `valid` / `invalid` 标记
- 因此，agent 的目标是降低“明显不合理的视野内突然消失”概率，而不是在配置层强行完全避免

---

## 6. 语义到最终配置的映射规则

这一节定义 agent 在写最终 TOML 时如何把自然语言槽位落到具体字段。

### 6.1 地图和拓扑

自然语言：

- “城市的人行道”
- “城镇路口”
- “郊区过街”

最终映射：

1. 先确定 `map_type`
2. 再选一个具体 `map.map_name`
3. 从该地图的 `walkable_cells.json` 里选一个兼容 `topology` 的 `scene_id`
4. 再从该 scene 内选一个兼容 `route_type` 的 `route_id`

建议：

- 不让 agent 发明坐标
- 优先用 `scene_id + route_id`

### 6.2 普通行人怎么走

自然语言短语到 `behavior` 的建议归一化：

| 自然语言短语 | 归一化 behavior |
| --- | --- |
| 正常步行、平稳走、普通走行 | `steady_walk` |
| 缓慢走、谨慎走、保守步行 | `slow_walk` |
| 快速通过、快步经过 | `fast_pass` |
| 正常过街 | `normal_cross` |
| 保守过街、先观察再过 | `cautious_cross` |
| 较激进过街、抢小 gap | `assertive_cross` |
| 几乎不看车直接过 | `ignore_gap_cross` |

### 6.3 group

自然语言短语到 group 的建议归一化：

| 自然语言短语 | 归一化 group |
| --- | --- |
| 无结伴、单人 | `none` |
| 两人结伴、前后同行 | `pair_follow` |
| 两人并排同行 | `pair_abreast` |
| 三人前后同行 | `triple_follow` |
| 三人并排行走 | `triple_abreast` |

### 6.4 人流密度

自然语言到人数规模：

| 自然语言短语 | 归一化 density |
| --- | --- |
| 空闲、稀疏、少量行人 | `sparse` |
| 一般、适中、中等人流 | `medium` |
| 拥挤、明显人多、人流密集 | `crowded` |

建议生成策略：

- `sparse`：ego + 少量配角
- `medium`：ego + 若干普通 pedestrians + 可选 group
- `crowded`：ego + 多个普通 pedestrians + 1 个 group

### 6.5 车流密度

自然语言到 background population：

| 自然语言短语 | 归一化 density |
| --- | --- |
| 无车流、无背景车 | `none` |
| 零星车流、少量车辆 | `low` |
| 正常城市背景车流 | `medium` |
| 明显较多车流 | `high` |

### 6.6 ego 行动

主体行人 `ego_action` 原则上也是从 `behavior` 里选。  
推荐规则：

- `sidewalk` 场景：ego 默认 `steady_walk`
- `junction` 且 `route_type` 为 `crosswalk_cross` 或 `unmarked_junction_cross`：ego 默认 `normal_cross`
- `junction` 且 `route_type` 为 `junction_through`, `junction_turn`, `junction_u_turn`：ego 默认 `steady_walk`

如果用户特别强调更快、更慢、更保守或更激进，再偏到其他行为。

### 6.7 采集时序

如果场景目标是正式数据采集，agent 在写 `spawn_time` 和 `duration` 时应遵守下面这些归一化策略。

ego 入场：

- ego 应晚于大多数背景 actor 入场
- ego 推荐作为最后一个或最后一批入场的行人
- ego 推荐 `spawn_time` 范围：`2.0s - 10.0s`
- 除非用户明确要求更晚入场，否则不建议把 ego 放到 `10s` 之后

非 ego 行人：

- 普通配角行人和 group 可以在 ego 之前入场
- 为了让采集开始后场景更自然，推荐它们的 `spawn_time` 早于 ego
- 如果某些配角只是为了填充密度，优先分散到 ego 前 `0s - 6s` 的区间内

背景车：

- 背景车优先用 `[vehicle.background_population]`
- 背景车在仿真启动时一次性生成是允许的
- agent 不需要让背景车和 ego 一起入场

采集时长：

- 采集场景默认应保证 `duration >= 20s`
- 如果难以满足 `20s`，则不应作为正式采集样本
- agent 应优先通过选择更长的 ego route 来满足时长，而不是单纯把 ego 放得更晚
- 如果 `ego spawn_time` 太晚，会压缩有效采集窗口，不推荐把“延后入场”当成主要手段

推荐理解：

- 让背景先形成
- 让 ego 在前 `10s` 内进入
- 让 ego 在进入后仍有 `20s+` 的可采集运动时间

---

## 7. 天气表达规则

weather 不要求 agent 逐字段自由发挥，而是按语义子槽位写配置。

### 7.1 时间段

| time_of_day | 建议写法 |
| --- | --- |
| `day` | `weather.sun_altitude.weights = { day = 1.0, low_sun = 0.0, twilight = 0.0, night = 0.0 }` |
| `low_sun` | `weather.sun_altitude.weights = { day = 0.0, low_sun = 1.0, twilight = 0.0, night = 0.0 }` |
| `twilight` | `weather.sun_altitude.weights = { day = 0.0, low_sun = 0.0, twilight = 1.0, night = 0.0 }` |
| `night` | `weather.sun_altitude.weights = { day = 0.0, low_sun = 0.0, twilight = 0.0, night = 1.0 }` |

### 7.2 云量

| cloudiness_level | 建议写法 |
| --- | --- |
| `clear` | `weather.cloudiness.weights = { clear = 1.0, mid = 0.0, overcast = 0.0 }` |
| `mid` | `weather.cloudiness.weights = { clear = 0.0, mid = 1.0, overcast = 0.0 }` |
| `overcast` | `weather.cloudiness.weights = { clear = 0.0, mid = 0.0, overcast = 1.0 }` |

### 7.3 雨量

推荐规则：

- `none`：把所有 precipitation bucket 的概率设成 `none = 1.0`
- `drizzle`：把目标 bucket 设成 `drizzle = 1.0`
- `rain`：把目标 bucket 设成 `rain = 1.0`
- `heavy`：把目标 bucket 设成 `heavy = 1.0`

实际写法不要求每次都把全部默认值展开，但必须保证语义单一明确。

### 7.4 雾

| fog_level | 建议写法 |
| --- | --- |
| `none` | `weather.fog.day_probability = 0.0` 且 `weather.fog.night_or_twilight_probability = 0.0` |
| `light` | 使用默认 `fog` 配置即可，或只保留轻雾概率 |

---

## 8. 直接生成最终 TOML 时的硬约束

agent 直接写最终 config 时，必须满足下面这些规则。

### 8.1 map

- 必须写 `map.map_name`
- 如果是 scene-based 配置，必须写 `map.scene_id`
- 建议写 `map.scene_catalog_path`

### 8.2 walker

- scene-based pedestrian 必须写 `behavior`
- 如果写 `formation`，必须同时写 `route_id`
- 如果 `quantity > 1` 且有 ego，必须写 `ego_member_index`
- 全文件只能有一个 ego

### 8.3 vehicle

- 背景车优先只用 `[vehicle.background_population]`
- 不写补流逻辑
- 不要求采集中新增 vehicle actor

### 8.4 sampling

如果是采集场景：

- `[sampling].enabled = true`
- 必须写 `sensor_rig`
- 必须写 `policy`
- 必须写 `sequence_id`
- 如果目标是正式数据集采集，推荐 `stop_on_ego_end = false`

推荐最终目录规则：

```text
data/<map_name>/<map_name>_<NNN>/
```

推荐 `sequence_id` 与目录名一致。

### 8.5 采集场景的额外硬约束

如果 agent 生成的是正式采集用 config，而不是普通演示 config，则还必须满足：

- ego 必须显式声明：`ego = true`
- ego 应在场景开始 `10s` 内入场
- `duration` 应尽量保证 `>= 20s`
- 不应把极短 ego route 配成正式采集场景
- 不应依赖“采集开始后再新增 actor”来制造动态感
- 背景车优先通过 `[vehicle.background_population]` 提供
- 允许 actor 在采集中自然消失，但不应主动设计会在 ego 前方近距离快速结束的短命 actor

---

## 9. 面向批量生成的覆盖规则

如果目标是数据集覆盖，建议把每个场景归一化到下面这些轴：

| 轴 | 可选值 |
| --- | --- |
| `map_type` | `urban_dense`, `town_grid`, `suburban_open` |
| `topology` | `sidewalk`, `junction` |
| `route_type` | 当前 6 类之一 |
| `ego_action` | 当前 7 类之一 |
| `group` | `none`, `pair_*`, `triple_*` |
| `ped_density` | `sparse`, `medium`, `crowded` |
| `vehicle_density` | `none`, `low`, `medium`, `high` |
| `time_of_day` | `day`, `low_sun`, `twilight`, `night` |
| `cloudiness_level` | `clear`, `mid`, `overcast` |
| `rain_regime` | `none`, `drizzle`, `rain`, `heavy` |
| `fog_level` | `none`, `light` |

批量生成时，优先补足未覆盖组合，而不是重复生成语言表面不同、但槽位组合相同的场景。

如果目标是正式采集，还应同时满足下面两个时序覆盖要求：

- ego 入场时间覆盖：
  - 早入场：`0s - 3s`
  - 中等入场：`3s - 6s`
  - 较晚入场：`6s - 10s`
- 采集时长覆盖：
  - 基础：`20s - 30s`
  - 中等：`30s - 45s`
  - 长时：`45s+`

但约束优先级是：

1. 先保证配置合法
2. 再保证 ego route 足够长
3. 再做入场时间分布覆盖
4. 不要为了覆盖“较晚入场”而牺牲有效采集时长

建议使用下面这个去重键：

```text
(
  map_type,
  topology,
  route_type,
  ego_action,
  group,
  ped_density,
  vehicle_density,
  time_of_day,
  cloudiness_level,
  rain_regime,
  fog_level
)
```

---

## 10. 推荐生成流程

agent 每次生成一个场景时，按下面顺序思考最稳：

1. 选 `map_type`
2. 选具体 `map_name`
3. 选 `topology`
4. 选兼容 `scene_id`
5. 选兼容 `route_type`
6. 选兼容 `route_id`
7. 选 `ego_action`
8. 决定是否有 `group`
9. 决定 `ped_density`
10. 决定 `vehicle_density`
11. 决定 `weather`
12. 决定 ego 入场时间和总 `duration`
13. 写最终 TOML

不建议的顺序：

- 先想复杂故事，再强行塞进 config
- 先写自由坐标，再回头找 scene
- 先写 group，再临时拼 route

---

## 11. 推荐禁止项

下面这些语义不要让 agent 写进描述或 config：

- “群体绝对不会被穿插”
- “组员严格同步协同决策”
- “采集开始后再补背景车”
- “复杂社交语义对抗”
- “任意地图任意坐标拼场景”
- “一辆车在 ROI 内循环出现很多次以制造车流”
- “ego 在很晚时刻才进入，导致有效采集窗口不足 20s”
- “为了避免 actor 消失，把所有 actor 都强制在 t=0 一起生成”

这些要么当前未实现，要么与当前数据采集约束冲突。

---

## 12. 最终使用方式

给 agent 的要求可以很简单：

1. 先用自然语言写一个更丰富的场景描述
2. 再把它归一化到本 registry 的语义槽位
3. 最后直接写当前仓库可运行的最终 TOML

重要的是：

- 自然语言可以自由
- 槽位选择必须受控
- 最终 config 必须只使用当前仓库支持的字段和组合

这样就能在不写新代码、不使用模板系统的前提下，稳定批量生成最终配置文件。
