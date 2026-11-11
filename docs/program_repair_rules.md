# 程序修复规则（rule-based program repair）

本文整理我们的布局 workflow（`workflow/simple.py`，task mode）里程序修复所用的全部规则。每条规则都标了出处的代码位置，所有阈值都是代码里实际使用的值。

**基本原则**

- LLM 只负责提出摆放方案。一个方案能不能被采用，只由程序判断。
- 任何一步都只移动或旋转物体，**从不改变物体尺寸**。
- 必须家具（required）**从不删除**。

## 0. 总流程

```
① plan     LLM：每个物体放在哪里（地面 / 墙 / 哪件家具上）、功能组、相对关系
② sizes    FastFill 给每个物体的尺寸，之后固定不变
③ place    LLM 一次摆放整个房间（并行 2 份草稿，选硬违规最少的一份）
④ check    程序检查 → 有硬违规就进入修复循环（Loop），只修硬违规
④b give way 扩充家具给必须家具让位：先移动，移动不了才删除
④c required first  必须家具仍有问题时：只拿必须家具重新求解，再把扩充家具逐件放回
⑤ review   LLM 审一次全屋，只处理 severe 问题；每个修改必须先过程序门槛，再在双盲成对比较中两次都胜出
④d left out  扩充家具仍有物理违规（碰撞、出界、没有支撑、挡门、挡窗）：放在家具上的先挪到台面空位，放不下才删除
④e wall fronts  挂墙物体前方地面不到 55% 是空的（SceneEval ACC 的那条地面）：沿墙挪到最近一个至少 65% 空的位置；
             不能产生新违规、不能破坏关系，地上的东西不动，也不删除任何物体
⑦ fallback 必须家具仍不合法时：只摆必须家具（纯程序），结果更好才采用
```

论文实验用的参数（也是 `python -m workflow run` 的默认值）：机器人宽 0.5 m，规划时再加 0.1 m 余量，手臂 0.4 m；开启 hard zones 和 relocate；修复最多 3 轮（`--repair-rounds 3`），每轮程序搜索最多 180 s（`--repair-seconds 180`）；草稿 2 份（`--drafts 2`）；每行有 `task` 时开启任务模式。

## 1. 违规的定义（critic，`critic.py` / `geometry.py` / `robot.py`）

每个违规都有一个连续的幅度（magnitude，单位 m）。比如把 20 cm 的重叠压到 1 cm，也算进步。

### 1.1 硬违规：合法性（legality）

| 代码 | 规则 | 阈值 |
|---|---|---|
| `boundary` | 物体的平面投影必须完全在地面多边形内，且离墙至少 1 mm（对齐 SceneEval OOB：贴墙物体伸出一点点就会被判出界） | 内缩 0.001 m |
| `vertical_bounds` | 底面不能低于地面，顶面不能高于天花板 | 1e-4 m |
| `floor_support` | 放在地面的物体，底面要贴地 | 1e-4 m |
| `wall_support` | 挂墙的物体，背面要贴墙、正面朝向房间（对齐 SceneEval Support：背面 5 mm 内要碰到墙） | 背面离墙 ≤ 0.004 m，朝向偏差 ≤ 10°；摆放时离墙 0.002 m |
| `object_support` | 放在家具上的物体：底面要在家具台面的高度上，投影中心要落在台面内（允许悬出；对齐 SceneEval Support 的重心-接触凸包检查） | 高度 1e-4 m；中心至少在台面内 0.01 m |
| `collision` | 两个物体的平面投影有重叠，且高度区间也有重叠；薄物体（地毯等）同样计入 | 重叠面积 > 1e-5 m² |
| `keepout` | 物体进入输入里规定的禁区多边形 | > 1e-5 m² |
| `door_clearance` | 落地障碍物占用门前 0.8 m 深的区域（地毯、垫子不算；家具上和墙上的物体不算） | 底面离地 ≤ 0.15 m、高 ≥ 0.1 m、占用面积 > 0.01 m² |
| `relation_*` | 输入里标为 hard 的约束（on、near、faces、between、against_wall…）不成立 | 按各约束自身的距离或角度 |

窗户只算软违规（`hard_windows=False`）。

### 1.2 硬违规：机器人（robot，`robot.py`）

所有检查都按 2D 平面计算。机器人宽 `width_m` = 0.5 m，规划时按 0.6 m。

| 代码 | 规则 |
|---|---|
| `robot_door` | 每扇门都能和其他门连通。可通行地面 = 地面向内收缩半个车宽，再减去每个落地障碍物外扩半个车宽。房间只有一扇门时，这扇门要连到房间内部；没有门时，用最大的空地代替入口 |
| `robot_reach` | 需要机器人去操作的家具，从门出发要能到达：车的中心能走到它某个可用侧面的 `半车宽 + 0.4 m` 以内。需要到达的家具包括：任务物体（物体放在家具上时，以下面的落地家具为准）、上面放着物体的家具、可开合的家具 |
| `robot_openable` | 可开合家具（柜、衣柜、冰箱、烤箱、洗衣机、床头柜……以及任务要打开的家具）门前的开门扇区要在房间内，且被落地障碍物盖住的部分 ≤ 10%。扇区和正面一样宽，深度按家具自身深度取，限制在 0.35–0.6 m 之间 |
| `robot_arm` | 放在家具上的物体（以及每个任务物体），车的中心能到达的某个点离它 ≤ 0.85 m（同 EmbodiedGen 的 `check_reachable`） |

违规记录里写的 id 依次是：受影响的物体，然后是"单独挪开就能解决问题"的挡路物体，修复时就移动这些物体。

### 1.3 软代价（只用来排序，不需要强行修到零）

每一项的代价 = 权重 × min(幅度 / 尺度, 1)：

| 项 | 内容 | 权重 |
|---|---|---|
| `relation_user` | 输入里的软约束 | 1 |
| `relation_plan` | 步骤 ① 给出的软关系、功能组 | 0.5 |
| `template_near` / `template_faces` | 从训练房间统计的"A 靠近 B / 朝向 B"模板 | 0.5 |
| `access` | 物体可用的一侧前方 0.5 m 内，空闲不足一半 | 0.5 |
| `wall_back` | 训练房间里 ≥ 80% 背靠墙的类别：背面离墙 > 0.15 m，或朝向偏离墙面法线 > 10° | 0.5 |
| `reach` | 训练房间里 ≥ 90% 能走到的类别，宽 0.4 m 的人从门走不到 | 1 |
| `window` | 窗前被挡 | 0.25 |

## 2. 接受规则（commit rule，`Critic.better`）

候选布局要和当前布局比较，按三层依次判断，前一层分出胜负就不再看后一层：

1. **合法性层**（除机器人外的所有硬违规）
2. **机器人层**（`robot_*`）
3. **软代价总和**

每一层硬违规的判断（`_tier`）：

- 当前这一层已经没有违规时，候选在这一层**不能出现任何新违规**。
- 新出现的违规中，只要有一个幅度 > **0.05 m**（`PROTECT_M`），就判为更差。
- 否则，违规条数更少即更好；条数相同时，总幅度要降低超过 **0.005 m**（`EPS_H`）才算更好。
- 软代价至少降低 **0.01**（`EPS_S`）才算进步。

因此，修好合法性的修改不会因为机器人检查被拒绝，软代价上的改进也换不来新的机器人违规。

## 3. 修复循环（`loop.py` 的 `Loop.run`）

**选目标**：每次取一个还没尝试过的违规，排序依次为：合法性先于机器人 → 涉及必须家具的先处理 → 幅度大的先处理。每个违规（同一代码 + 同一组物体）每轮只尝试一次；修改后新出现的违规也会被处理。

**修复范围**：每个目标从小到大依次尝试三层范围（scope）：

| 层级 | 可以移动的物体 |
|---|---|
| 0 | 违规涉及的物体 + 与它们碰撞的物体 + 与它们有约束关系的物体（一跳）+ 放在这些物体上的东西 |
| 1 | 再加上它们所在的功能组 |
| 2 | 房间里所有物体（只用于硬违规） |

范围外的物体固定不动，当作障碍物；固定物体（fixed objects）永远不动。每个候选都在整个房间上检查。

**生成候选**（`solver.beam_search`，mode="local"，尺寸固定，不允许缩小）：

- 以当前位置为中心，在 0.05 / 0.10 / 0.20 / 0.40 m 四个半径上各取 8 个方向，共 32 个偏移，加上原位置。
- 朝向保留原朝向；如果草稿给了第二个备选朝向，也一并尝试，但会加罚分。
- 离原位置越远罚分越高：硬违规每米 2.0（`DRAFT_WEIGHT`），软违规每米 0.5。
- 物体移动时，放在它上面的东西一起刚性移动。
- 束宽 24，每个物体最多 48 个候选；取最后一层束里最多 8 个完整状态交给 critic，选其中最好的一个，再用第 2 节的规则决定接不接受。
- 模板类违规会临时作为一个软约束加入搜索，例如"放到 B 旁边 0.25 m 内"、"朝向 B，误差 40° 以内"。

**整体搬家（relocate，只针对 `robot_reach`、`robot_openable`、`robot_door`）**：局部候选都失败时，把违规里点名的挡路物体（最多 3 个）依次搬到房间任意位置，最后再试被挡住的家具本身。只搬落地物体，上面放的东西一起搬。搬动距离几乎不罚分（每米 0.1）。候选位置来自：
- 沿墙：离墙 0.04 m 或 0.35 m，朝向背靠墙。
- 网格：步长 ≥ 0.45 m，最多 11 × 11 个格点，每个格点试 4 个朝向。

**结束条件**：所有违规都尝试过一次；如果这一轮有修改被接受、而且还剩硬违规，就再来一轮，最多 3 轮。尝试次数上限为 24 + 3 × 初始违规数。程序搜索累计超过 180 s 也会停止，结果中记为 "time limit"，不会悄悄截断。

## 4. 扩充家具让位（task mode 步骤 ④b，`simple.py`）

只针对**必须家具身上的硬违规**，即 `on_required`：涉及必须家具的合法性违规、以必须家具为对象的机器人违规，或者门被隔断。

1. **选哪件扩充家具**（`give_way_choice`）：每个违规给它涉及的每件扩充家具记分，分值 = min(幅度, 1)，违规涉及必须家具时再加 1。总分最高的扩充家具被选中；同分时选占地面积小的，再按 id。上面放着必须家具的扩充家具不能被选。
2. **先移动**：把被选中的扩充家具（连同上面的东西）搬到房间任意位置。只有同时满足以下三条才接受：这件家具自身没有硬违规、必须家具的违规数没有增加、critic 判定更好。
3. **移动不了才删除**：连同上面放的东西一起删，每次删除都记录原因。
4. **机器人违规里没点名任何扩充家具时**（几件家具一起挡住了路，单独删哪一件都看不出效果）：依次尝试必须家具 2.5 m 以内、离它最近的落地扩充家具，最多 5 件。先移动；移动无效再删除；删除后必须家具的违规数没有下降，就把它放回去。
5. **有删除时**，在腾出空间的房间上再跑一次修复循环。
6. **扩充家具自身的违规**：只尝试移动，**永远不因为自身问题被删除**，修不好的保留并报告（`expansion_issues`）。
7. **之后必须家具若仍有违规**，再执行一次第 1–3 步。

## 5. 必须家具优先（步骤 ④c）

如果必须家具仍有硬违规，就这样做：

1. 把扩充家具全部拿走，只对必须家具跑修复循环（允许整体搬家，至少 3 轮）。如果还不行，就从零开始全局搜索（束宽 ≥ 64，每个物体候选 ≥ 96），再修复一次，取违规更少的结果。
2. 把扩充家具**逐件放回**，包括 ④b 删掉的。顺序：先放落地的，再放放在其他物体上的；同一层里先放大的。
   - 先试原位置。如果它原来放在某件必须家具上，就跟着那件家具的新位置刚性移动（`carry`）。
   - 原位置会让必须家具的违规数超过"只有必须家具时"的数量，就在全屋找一个不超过这个数量的位置。
   - 找不到位置就删除，原因记为"no place that keeps the required objects legal and reachable"。
3. 再执行一次第 4 节的让位、修复，以及扩充家具的移动。
4. **只有必须家具的违规数比这一步之前更少，才采用新结果**；否则恢复成 ④c 之前的布局。

## 6. 兜底（步骤 ⑦，`required_only`）

做完前面所有步骤，如果仍有必须家具不合法或机器人到不了：

1. 只摆必须家具，完全由程序完成，不调用 LLM。候选起点有两类：它们在完整布局里的位置，以及三种全局搜索（global、wall、spacious）各自最好的 2 个结果。每个起点都跑一次修复循环。
2. 如果约束作为硬约束时找不到合法结果，就把约束改成软约束再试一遍（`on` 和 `keepout` 仍保持硬约束）。无法满足的约束会报告出来。
3. 只有当必须家具的违规数比完整布局时**更少**，才用兜底结果替换（status 记为 `fallback_required_only`，完整布局保存在 diagnostics 里）。test150doors 的 150 个房间里，这一步被用到 1 次（房间 110）。

## 7. LLM 审查的程序门槛（步骤 ⑤，`semantic.py`）

LLM 审一次全屋，只处理 severe 问题，最多 3 个。每处理一个问题，流程是：LLM 给出修改方案 → 程序只修复这个方案带来的硬违规 → 通过下面的门槛（`admissible`）→ 双盲成对比较：交换两个布局的先后顺序各问一次，两次都偏好新布局才接受。

门槛：
- 硬违规（合法性和机器人两层分别看）条数不增加，总幅度不增加。原来某一层没有违规的，修改后这一层也不能有违规。
- `reach`、`access`、`door` 三项代价都不能变差（容差 0.01）。
- 软代价总和最多上升 0.1。

## 8. 阈值汇总

| 常量 | 值 | 位置 |
|---|---|---|
| 机器人宽 / 规划余量 | 0.5 m / +0.1 m | `cli.py`（`--robot-width`、`--robot-margin`） |
| 能使用某个侧面的距离（车中心到侧面） | 半车宽 + 0.4 m | `robot.Robot.arm_m` |
| 手臂够到物体的距离（车中心到物体） | 0.85 m | `robot.Robot.arm_reach_m` |
| 开门扇区深度 / 可被遮挡比例 | 0.35–0.6 m / 10% | `robot.py` |
| 门前区域深度 / 允许占用面积 | 0.8 m / 0.01 m² | `geometry.py` |
| 新违规保护阈值 `PROTECT_M` | 0.05 m | `critic.py` |
| 硬 / 软进步阈值 `EPS_H` / `EPS_S` | 0.005 m / 0.01 | `critic.py` |
| 局部偏移半径 | 0.05 / 0.10 / 0.20 / 0.40 m × 8 个方向 | `solver.LOCAL_RINGS_M` |
| 离原位置的罚分（硬 / 软 / 整体搬家） | 2.0 / 0.5 / 0.1 每米 | `loop.py` |
| 束宽 / 每个物体候选数 / 交给 critic 的完整状态数 | 24 / 48 / 8 | `simple.py`, `loop.py` |
| 每个违规最多搬几个挡路物体 | 3 | `loop.RELOCATE_MOVERS` |
| 尝试次数上限 / 轮数 / 搜索时限 | 24 + 3 × 违规数 / 3 / 180 s | `loop.py`, `cli.py` |
| 让位时找附近挡路物体的范围 / 最多试几件 | 2.5 m / 5 | `simple.NEAR_BLOCKER_*` |
| 审查时软代价最多上升 | 0.1 | `semantic.SOFT_TOLERANCE` |

## 9. 和早先实验的区别

2026-11-02 之前的版本在搜索找不到合法位置时会把物体逐级缩小（0.95 / 0.90 / 0.85 / 0.80），现在的代码已经没有这一步：任何物体都不改尺寸，上面写的就是当前的规则。

## 附：论文用的算法框（英文）

```
Algorithm 1  Rule-based layout repair (sizes fixed; required objects never removed)
Input: draft layout L from the LLM, fixed sizes, required set R, robot of width w
 1: V ← Critic(L)                       ▷ hard: legality ∪ robot; soft: weighted, capped costs
 2: while some hard violation v ∈ V not yet tried in this pass do
 3:     pick v: legality before robot, involving R first, largest magnitude first
 4:     for scope in [v's objects + colliding + related (+children), + their groups, all objects] do
 5:         C ← local beam search over scope (rings 5–40 cm × 8 directions, current yaw; children move rigidly)
 6:         L' ← best of C by Critic;  if Better(L', L) then L ← L'; break
 7:     if not accepted and v is a robot check then
 8:         for o in v's blockers (≤3), then v's object: relocate o anywhere (wall / grid candidates); accept if Better
 9: Give way (task mode): for each hard violation on R, pick the expansion object scoring most;
10:     move it anywhere if that clears it without hurting R, else remove it (with what stands on it)
11: Required first: if R still violated, solve R alone (repair, else global search), then put each expansion
12:     object back where it was, else anywhere keeping R's violations ≤ alone, else drop it; keep only if R improves
13: Leave out: an expansion object still in a physical violation goes to a free spot on its parent's top, else
14:     is removed (with what stands on it)
15: Wall fronts: a wall object whose front floor strip (SceneEval ACC's) is < 55% free slides along the walls to
16:     the nearest place >= 65% free, with what stands on it, if that adds no violation and breaks no relation
17: Fallback: if R still violated, lay out R alone by the program; keep only if R improves
Better(a, b): compare tiers legality → robot → soft; in a tier, no new violation on a clean tier, no new one > 5 cm,
              fewer violations, or as many with total magnitude lower by > 5 mm; soft must drop by > 0.01
```
