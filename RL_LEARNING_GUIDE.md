# MuJoCo 机械臂强化学习入门教学指南

本指南面向想在 **本仓库** 中入门强化学习（Reinforcement Learning, RL）的同学。仓库提供了 3 个由浅入深的 Panda 机械臂 RL 任务，配套详细的中文注释代码，是一套完整的「机械臂 RL 实战教程」。

---

## 一、前置知识准备

在动手之前，建议先具备以下基础（不必精通，有概念即可）：

| 领域 | 需要掌握 | 推荐资源 |
|------|---------|----------|
| Python | NumPy 数组操作、类与继承、装饰器 | 官方 Tutorial |
| 机器人学 | 关节空间/笛卡尔空间、正运动学(FK)、关节限位 | 仓库内 `control_joint_pos.py`、`get_body_pos.py` |
| MuJoCo | MJCF 模型文件、`MjModel`/`MjData`、`mj_step` | 仓库内 `mujoco_get_all_body.py`、`control_joint_pos.py` |
| 深度学习 | MLP、激活函数、梯度下降、PyTorch 基础 | PyTorch 官方 60 分钟入门 |
| 强化学习 | MDP、策略、价值函数、PPO 算法思想 | OpenAI Spinning Up、Stable-Baselines3 文档 |

> 💡 如果 RL 完全零基础，强烈建议先读一遍 [OpenAI Spinning Up](https://spinningup.openai.com/) 的「Part 1: Key Concepts」与「Part 3: Intro to PPO」章节，建立直觉后再回到本仓库实战。

---

## 二、环境搭建

按仓库 [README.md](file:///home/playeee/projects/mujoco-learning/README.md) 的 Installation 章节安装即可。核心依赖：

- `mujoco`：物理仿真引擎
- `gymnasium`（或 `gym`）：标准 RL 环境接口
- `stable-baselines3`：PPO 等算法的成熟实现
- `torch`：神经网络后端
- `tensorboard`：训练曲线可视化

安装完成后，运行以下命令验证：

```bash
python -c "import mujoco, gymnasium, stable_baselines3, torch; print('OK')"
```

---

## 三、强化学习核心概念速览（结合本仓库）

理解下面 6 个概念，就能读懂仓库里所有 RL 代码：

### 1. 环境（Environment）
智能体交互的世界。本仓库中是 MuJoCo 里的 Panda 机械臂场景。自定义环境继承 `gym.Env`，必须实现 `reset()` 与 `step(action)` 两个方法。

### 2. 观测（Observation, `obs`）
智能体能感知到的状态。本仓库的观测一般是「关节角 + 关节速度 + 任务相关位置」拼接成的向量。例如 [rl_panda_reach_target_high_profile.py](file:///home/playeee/projects/mujoco-learning/rl_panda_reach_target_high_profile.py) 中是 10 维：7 关节角 + 3 目标位置。

### 3. 动作（Action）
智能体的输出。本仓库采用 **增量位置控制**：动作是 7 维归一化向量 `[-1,1]`，表示 7 个臂关节的角度增量方向与幅度。夹爪开合在 pickup 任务中由脚本化逻辑控制（不交给策略）。

### 4. 奖励（Reward）
每一步环境返回的标量反馈，是智能体学习的唯一信号。本仓库的奖励设计是核心难点，详见第五节。

### 5. 回合（Episode）与终止（Termination/Truncation）
- **terminated**：回合因「失败/成功」等条件结束（如方块掉落）
- **truncated**：回合因「超时」被截断（如超过最大步数）
- 两者都为 `False` 时，回合继续。

### 6. 策略（Policy, π）
从观测到动作的映射。PPO 训练出的就是一个神经网络策略 `π(a|s)`。测试时用 `model.predict(obs, deterministic=True)` 调用。

---

## 四、学习路径（三个任务，由浅入深）

仓库的 3 个 RL 文件构成一条清晰的学习曲线，**务必按顺序学习**：

```
rl_panda_reach_target_high_profile.py   ← 入门：先学这个
        ↓
rl_panda_obstacle_high_profile.py       ← 进阶：加入障碍物
        ↓
rl_panda_pickup_cube.py                 ← 高阶：多阶段抓取放置
```

### 📘 阶段 1：reach_target（到达目标点）

**文件**：[rl_panda_reach_target_high_profile.py](file:///home/playeee/projects/mujoco-learning/rl_panda_reach_target_high_profile.py)
**配套视频**：[B站 BV1DAskzmEPZ](https://www.bilibili.com/video/BV1DAskzmEPZ/)
**预训练模型**：[assets/model/rl_reach_target_checkpoint/](file:///home/playeee/projects/mujoco-learning/assets/model/rl_reach_target_checkpoint/)

**任务**：控制机械臂末端从 home 位姿运动到工作空间内随机生成的目标点。

**学习重点**（按阅读顺序）：
1. **环境类结构**：`__init__` 中如何加载 MJCF、定义 `observation_space`/`action_space`、获取 body id
2. **`reset()`**：每回合如何随机化目标点、复位机械臂、返回初始观测
3. **`step(action)`**：动作如何转换为关节角度增量 → 写入 `data.ctrl` → `mj_step` 推进仿真
4. **`_calc_reward()`**：稠密奖励设计——距离目标的负距离 + 姿态惩罚 + 动作平滑惩罚
5. **`train_ppo()`**：`make_vec_env` + `SubprocVecEnv` 多进程并行、PPO 超参数、`model.learn`
6. **`test_ppo()`**：加载模型、`deterministic=True` 贪心评估

**实践建议**：
- 先**直接运行测试**看效果：将 `if __name__ == "__main__"` 中 `TRAIN_MODE = False`，指定 `MODEL_SAVE_PATH` 为预训练模型路径，运行 `python rl_panda_reach_target_high_profile.py`
- 再**短时间训练**（如 `total_timesteps=200_000`）观察曲线
- 用 TensorBoard 观察训练：`tensorboard --logdir ./tensorboard/`

### 📗 阶段 2：obstacle_avoidance（避障）

**文件**：[rl_panda_obstacle_high_profile.py](file:///home/playeee/projects/mujoco-learning/rl_panda_obstacle_high_profile.py)
**配套视频**：[B站 BV1bd6sBpEvT](https://www.bilibili.com/video/BV1bd6sBpEvT/)
**预训练模型**：[assets/model/rl_obstacle_avoidance_checkpoint/](file:///home/playeee/projects/mujoco-learning/assets/model/rl_obstacle_avoidance_checkpoint/)

**任务**：在存在球形障碍物的工作空间中运动到目标点，需学会避障。

**相比阶段 1 的新增点**：
- 观测空间**扩展**：增加障碍物位置（带噪声）与尺寸
- 奖励中**碰撞惩罚**权重大幅提升，碰撞即终止回合
- 策略网络**加深**：`[512, 256, 128]`
- 引入**学习率线性衰减**，后期收敛更稳

**学习重点**：
- 碰撞检测：如何通过 `data.ncon` 与 `contact` 判断是否碰撞
- 障碍物随机化：每回合随机化障碍物 Y 坐标，提升泛化性
- 观测噪声：模拟真实感知，提升鲁棒性

**实践建议**：对比阶段 1，重点理解「为什么观测要加障碍物信息」「为什么网络要加深」。

### 📕 阶段 3：pickup_cube（抓取并放置方块）

**文件**：[rl_panda_pickup_cube.py](file:///home/playeee/projects/mujoco-learning/rl_panda_pickup_cube.py)
**配套视频**：[B站 BV1YKTG6GEqj](https://www.bilibili.com/video/BV1YKTG6GEqj/)
**预训练模型**：[assets/model/rl_pickup_checkpoint/](file:///home/playeee/projects/mujoco-learning/assets/model/rl_pickup_checkpoint/)

**任务**：抓取桌面方块 → 抬升 → 搬运 → 放置到目标位置（长时序、多阶段任务）。

**这是本仓库的精华，也是难点所在**。相比前两个任务，它引入了 4 项关键设计：

| 设计 | 说明 | 阅读位置 |
|------|------|----------|
| **阶梯式奖励** | 4 个 stage 各有 base 奖励，阶段越高 base 越大，避免阶段间奖励尺度冲突 | `_calc_reward()` |
| **脚本化夹爪** | 夹爪开合由状态机自动控制，不交给策略，降低学习难度 | `_scripted_gripper()` |
| **助力机制(assist)** | 抬升阶段混合 home 动作与策略动作，帮助可靠抬起方块 | `step()` 中 `_assist_on` |
| **接触式抓取判定** | 通过左右手指与方块的物理接触判断是否真正抓稳 | `_is_grasped_by_contact()` |

**4 个阶段（stage）**：
```
stage0 接近 → stage1 闭合 → stage2 抬升 → stage3 放置
   base=0      base=1       base=1        base=3
```

**学习重点**（建议精读）：
1. `_calc_reward()`：逐段理解每个 stage 的稠密奖励如何引导智能体推进进度
2. `_scripted_gripper()`：理解状态机 `phase 0→1→2` 的切换条件
3. `_is_grasped_by_contact()`：理解为什么基于接触比基于夹爪指令更可靠
4. 抓取标志位**消抖**逻辑：连续 4 步脱接触才取消 grasped，避免状态频繁跳变

**实践建议**：
- 先用预训练模型测试，观察 4 个 stage 的实际行为
- 重点调试：如果学不会抓取，检查 `_is_grasped_by_contact()` 的阈值；如果学不会抬升，检查 assist 机制是否生效

---

## 五、奖励函数设计要点（核心难点）

奖励设计是机械臂 RL 的**灵魂**，也是最容易踩坑的地方。本仓库的设计经验：

### 1. 稠密奖励优于稀疏奖励
- ❌ 只有「到达目标 +1」→ 智能体几乎收不到信号，学不会
- ✅ 每步给「负距离」或 `1 - tanh(k*距离)` → 持续引导

### 2. 多阶段任务用阶梯式 base
避免后期阶段（如放置）的奖励被前期阶段（如接近）淹没。stage 越高 base 越大，保证进度推进有回报。

### 3. 惩罚项要克制
- 惩罚权重过大会让智能体「什么都不做」以避免惩罚
- 本仓库的做法：奖励主体大（几元），惩罚项小（零点几元）

### 4. 平滑性惩罚
`-0.01 * ||action - last_action||` 抑制抖动，让动作更平滑，也更利于真实硬件执行。

### 5. 时间惩罚
每步 `-0.02`，鼓励智能体尽快完成任务，避免「磨蹭」。

---

## 六、训练与调试技巧

### 1. 多进程并行加速
```python
env = make_vec_env(make_env, n_envs=64, vec_env_cls=SubprocVecEnv)
```
本仓库用 64 个并行环境，是 RL 训练效率的关键。注意：**多进程下只有首个子环境开可视化**（通过标志文件机制实现）。

### 2. TensorBoard 监控
```bash
tensorboard --logdir ./tensorboard/panda_pickup/
```
重点看：
- `rollout/ep_rew_mean`：平均回合奖励（应稳步上升）
- `task/is_success_rate`：成功率（自定义回调 `TaskMetricsCallback` 记录）
- `train/entropy_loss`：策略熵，反映探索性

### 3. 断点恢复
```python
train_ppo(resume_from="path/to/checkpoint")
```
长时间训练务必用断点恢复，避免中断后从头开始。

### 4. 常见问题诊断

| 现象 | 可能原因 | 排查方法 |
|------|---------|----------|
| 奖励不上升 | 奖励设计问题 / 学习率过大 | 检查 reward 各分量、降低 lr |
| 学会接近但学不会抓取 | 接触判定阈值过严 / 夹爪脚本逻辑错 | 打印 `_is_grasped_by_contact()` 返回值 |
| 抖动严重 | 动作平滑惩罚过小 / ent_coef 过大 | 增大平滑惩罚、降低 ent_coef |
| 训练后期崩溃 | 学习率未衰减 / clip_range 过大 | 用线性衰减 lr、降低 clip_range |
| GPU 显存不足 | batch_size 过大 / 网络过深 | 降低 batch_size 或 n_envs |

---

## 七、推荐学习顺序（完整路线）

```
第 1 步：读 README，跑通环境安装
第 2 步：跑 reach_target 的预训练模型，看效果
第 3 步：精读 reach_target 代码（带中文注释）
第 4 步：用小步数（200k）自己训练 reach_target，观察曲线
第 5 步：读 obstacle 代码，理解新增的避障逻辑
第 6 步：读 pickup_cube 代码，重点理解阶梯式奖励与脚本化夹爪
第 7 步：跑 pickup_cube 预训练模型，观察 4 个 stage 行为
第 8 步：尝试修改奖励函数 / 超参数，观察对训练的影响
第 9 步：（进阶）尝试迁移到自己设计的任务
```

---

## 八、延伸资源

- **算法理论**：[OpenAI Spinning Up](https://spinningup.openai.com/) — PPO 推导与实现
- **SB3 文档**：[stable-baselines3.readthedocs.io](https://stable-baselines3.readthedocs.io/) — API 与调参指南
- **MuJoCo 文档**：[mujoco.readthedocs.io](https://mujoco.readthedocs.io/) — MJCF 与 Python API
- **Gymnasium 文档**：[gymnasium.farama.org](https://gymnasium.farama.org/) — 环境接口规范
- **配套视频**：仓库 README 的 Tutorials 表格中有每个文件对应的 B 站讲解视频

---

## 九、仓库 RL 文件速查表

| 文件 | 任务 | 观测维度 | 动作维度 | 难度 | 关键设计 |
|------|------|---------|---------|------|---------|
| [rl_panda_reach_target_high_profile.py](file:///home/playeee/projects/mujoco-learning/rl_panda_reach_target_high_profile.py) | 到达目标点 | 10 | 7 | ⭐ | 稠密奖励 + 姿态惩罚 |
| [rl_panda_obstacle_high_profile.py](file:///home/playeee/projects/mujoco-learning/rl_panda_obstacle_high_profile.py) | 避障到达 | 17 | 7 | ⭐⭐ | 碰撞检测 + 深网络 + lr 衰减 |
| [rl_panda_pickup_cube.py](file:///home/playeee/projects/mujoco-learning/rl_panda_pickup_cube.py) | 抓取放置 | 29 | 7 | ⭐⭐⭐ | 阶梯奖励 + 脚本夹爪 + 助力 + 接触判定 |

---

祝你学习顺利！遇到问题先读代码中的中文注释，再结合 TensorBoard 曲线分析，大部分问题都能定位到原因。
