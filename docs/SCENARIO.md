# 场景生成策略的数学结构化表达

在构建社会导航（Social Navigation）的全栈闭环评测基准时，需要将复杂的动态物理世界与人类社会行为进行抽象与解耦。为了实现场景的程序化生成（Procedural Generation）并保证实验的控制变量原则，我们将一个完整的测试场景片段（Episode）形式化为一个位于高维参数空间的采样实例。

定义一个完整的社会导航测试场景 $\mathcal{S}$ 为一个五元组（5-Tuple）：

$$\mathcal{S} = \langle \mathcal{M_T}, \mathcal{E}, \mathcal{V}_{init}, \mathcal{B}, \mathcal{I} \rangle$$

以下是对该五元组中各个子空间的严格数学定义及实例推演。

---

## 1. 拓扑几何子空间 $\mathcal{M_T}$ (Map Topology)

**定义：** $\mathcal{M_T}$ 定义了场景发生的静态局部几何特征与语义拓扑。它是由全局地图 $\Omega$ 中切割出的特定结构实体。

**数学表达：**
$$\mathcal{M_T} \in \{ \text{Sidewalk}(W, L), \text{Crosswalk}(\vec{P}_{start}, \vec{P}_{end}), \text{Junction}(N_{arms}), \dots \}$$

**实例说明：**
假设我们要在一个长方形斑马线上进行测试。该拓扑实体可以参数化表示为：
$$\mathcal{M_T}_{1} = \text{Crosswalk}(W=4.0m, L=15.0m, \vec{C}=(100.0, 200.0))$$
其中 $W$ 为宽度，$L$ 为长度，$\vec{C}$ 为该斑马线在世界坐标系中的几何中心坐标。

---

## 2. 环境物理子空间 $\mathcal{E}$ (Environment)

**定义：**
$\mathcal{E}$ 定义了场景的全局物理环境参数，主要用于模拟不同传感器（如视觉、激光雷达）在不同信噪比下的输入条件。

**数学表达：**
$$\mathcal{E} = (W_{rain}, W_{fog}, I_{illumination})$$

**实例说明：**
在一个“黄昏且伴有微雨”的场景中，环境子空间可赋值为：
$$\mathcal{E}_{1} = (W_{rain}=0.3, W_{fog}=0.1, I_{illumination}=0.4)$$
固定该向量可以确保同一测试集下的视觉/物理渲染条件完全一致。

---

## 3. 背景交通子空间 $\mathcal{V}_{init}$ (Background Traffic - Closed Loop)

**定义：**
$\mathcal{V}_{init}$ 定义了场景中的机械式背景交通流。在真实的**闭环博弈（Closed-loop Interaction）**中，背景车辆不能是写死轨迹的“幽灵车”，它们必须能够对主智能体（Ego）和行人（$\mathcal{B}$）的不同行为做出符合物理与交通规则的响应（如让行、急刹）。因此，我们不固定其时空轨迹，而是固定其**初始状态、导航目标与驾驶策略（Policy）**。

**数学表达：**
$$\mathcal{V}_{init} = \{ (\vec{X}_{k, start}, \vec{X}_{k, goal}, \pi_k) \}_{k=1}^{K}$$
其中 $K$ 为背景车辆总数，$\vec{X}_{k, start}$ 和 $\vec{X}_{k, goal}$ 分别是第 $k$ 辆车的初始位姿与导航目的地，$\pi_k$ 是该车的驾驶策略模型（如目标速度、安全跟车距离、激进程度等）。

**实例说明：**
假设斑马线前有一辆车正准备驶过：
$$\mathcal{V}_{init\_1} = \{ (\vec{X}_{start}=(100, 180), \vec{X}_{goal}=(100, 300), \pi_{conservative}) \}$$
在同一组 $\mathcal{V}_{init\_1}$ 下，如果我们在测试变体中配置行人突然冲出（$\mathcal{B}_{hurry}$），由 $\pi_{conservative}$ 驱动的背景车会紧急刹车；如果是正常过马路，背景车会平稳让行。其最终的时空轨迹 $\vec{\tau}_k(t)$ 是闭环仿真的**涌现结果（Emergent Property）**，而非预设输入。

---

## 4. 社会行为配置 $\mathcal{B}$ (Social Behavior Configuration)

**定义：**
$\mathcal{B}$ 是社会导航场景的核心，定义了场景中所有行人的内在状态（意图、运动学属性）以及他们之间的隐性社会拓扑关系（社会群体连接）。

**数学表达：**
社会行为配置由**个体状态集合 $\mathcal{P}$** 与 **群体拓扑集合 $\mathcal{G}$** 的并集构成：
$$\mathcal{B} = \mathcal{P} \cup \mathcal{G}$$

1.  **个体状态 $\mathcal{P}$：**
    $$\mathcal{P} = \{ p_i \mid p_i \in \text{StateSpace}, i \in [1, N] \}$$
    其中 $N$ 为行人总数，$\text{StateSpace}$ 是行人行为属性的正交组合（如：运动速度 $\times$ 注意力状态 $\times$ 避让意愿）。
2.  **群体拓扑 $\mathcal{G}$：**
    $$\mathcal{G} = \{ g_j \mid g_j \subseteq \{1, 2, \dots, N\}, |g_j| \ge 2 \}$$
    $\mathcal{G}$ 是一系列子集的集合，每个子集代表一个结伴而行的社会群体。若 $\mathcal{G} = \emptyset$，则所有人均为独立的散客。

**实例说明：**
假设场景中有 3 名行人（$N=3$）。
* **定义个体状态：**
    $p_1 = (\text{Speed}_{normal}, \text{Attention}_{high}, \text{Yield}_{active})$
    $p_2 = (\text{Speed}_{normal}, \text{Attention}_{high}, \text{Yield}_{active})$
    $p_3 = (\text{Speed}_{fast}, \text{Attention}_{distracted}, \text{Yield}_{oblivious})$
* **定义社会拓扑：**
    假设 $p_1$ 和 $p_2$ 是一起逛街的情侣，而 $p_3$ 是一个正在看手机狂奔的独立外卖员。则群体关系可表示为：
    $$\mathcal{G} = \{ \{p_1, p_2\} \}$$
* **最终行为配置：**
    $$\mathcal{B}_{1} = \{ p_1, p_2, p_3 \} \cup \{ \{p_1, p_2\} \}$$
在这种配置下，导航算法不仅需要避开具有高动态不确定性的 $p_3$，还需要识别出 $p_1$ 与 $p_2$ 之间的社会力场连接，避免从二人中间穿插（破坏 $\mathcal{G}$ 的完整性）。

---

## 5. 交互任务定义 $\mathcal{I}$ (Interaction Task)

**定义：**
$\mathcal{I}$ 定义了自车/主智能体（Ego Agent，如助盲设备使用者）在当前场景下的导航物理目标与约束条件。

**数学表达：**
$$\mathcal{I} = (\vec{X}_{start}, \vec{X}_{goal}, \mathcal{C})$$
其中 $\vec{X}_{start}$ 为初始状态，$\vec{X}_{goal}$ 为目标状态集，$\mathcal{C}$ 为任务执行时的几何与运动学约束（如最大速度、最大角加速度等）。

**实例说明：**
对于上述斑马线拓扑 $\mathcal{M_T}_{1}$，定义主智能体需要从斑马线一端走到另一端：
$$\mathcal{I}_{1} = (\vec{X}_{start}=(100, 190), \vec{X}_{goal}=(100, 210), v_{max} \le 1.2 m/s)$$

---

## 6. 全局评测数据集的组合生成策略 (Dataset Formulation)

为了系统性地评估社会导航算法在不同干预条件下的鲁棒性，我们需要通过上述子空间的**笛卡尔积（Cartesian Product）**来生成结构化的闭环评测数据集 $\mathcal{D}$。在这个过程中，底层的物理交互逻辑（仿真器 $f_{sim}$）将基于初始条件演算出最终的轨迹结果。

**生成方程：**
$$\mathcal{D} = \bigcup_{\mathcal{M_T} \in \Omega} \bigcup_{\mathcal{E} \in E} \bigcup_{\mathcal{V}_{init} \in V} \bigcup_{\mathcal{B} \in B} f_{sim}(\mathcal{M_T}, \mathcal{E}, \mathcal{V}_{init}, \mathcal{B}, \mathcal{I})$$

*(注：此处使用连集符号 $\bigcup$ 表示将每次仿真运行产生的数据轨迹收集到最终的数据集中)*

**方程推演与实例意义：**
1.  **最内层循环**：这是社会性评测的核心。在**同一个路口** ($\mathcal{M_T}$)、**同样的天气** ($\mathcal{E}$)、**同一批具有相同初始状态和驾驶策略的车流** ($\mathcal{V}_{init}$) 以及**相同的起点终点** ($\mathcal{I}$) 下，系统仅变换行人的社会行为 $\mathcal{B}$。
    * *实例：* 第一次生成 $\mathcal{B}_{solo}$（3个互不认识的散客）；第二次重置场景，生成 $\mathcal{B}_{group}$（3个人变为一组群体）。由于车流 $\mathcal{V}_{init}$ 具有闭环响应能力，这两次仿真中车辆的时空轨迹可能会因行人的不同而产生合理分化，这正是真实世界复杂博弈的体现。
2.  **向外扩展**：逐步改变环境 $\mathcal{E}$（测试传感器的鲁棒性），最终遍历地图中所有的拓扑实体 $\mathcal{M_T}$（如所有人行道、所有斑马线），从而在数学和逻辑上实现覆盖各种边缘情况（Corner Cases）的全栈评测基准。