"""
Franka Emika Panda 机械臂强化学习——抓取并放置方块任务（Pick & Place）

任务说明：
    控制 7 自由度 Panda 机械臂（带夹爪）完成「抓取桌面上的方块 → 抬升 → 搬运 → 放置到目标位置」的完整流程。
    这是一个长时序、多阶段的复杂操作任务，相比 reach_target 与避障任务难度显著提升。

任务阶段（stage）：
    stage0 接近：末端运动到方块上方合适的抓取位姿（横向对准 + 垂直高度 + 朝下姿态 + 偏航对齐）
    stage1 闭合：夹爪闭合抓取方块（脚本化夹爪控制）
    stage2 抬升：将方块抬离桌面到 carry_height
    stage3 放置：搬运方块到目标位置并下放

关键设计：
    - 阶梯式奖励（stage_base + 阶段内稠密奖励）：避免不同阶段奖励尺度冲突
    - 脚本化夹爪控制：夹爪开合不由策略输出，而是根据当前阶段状态自动决定，降低学习难度
    - 助力机制（assist）：在抬升阶段混合 home 动作与策略动作，帮助可靠抬起方块
    - 接触式抓取判定：通过左右手指与方块的接触判断是否真正抓稳

技术栈：MuJoCo + Gymnasium + Stable-Baselines3 (PPO) + PyTorch
"""

import numpy as np
import mujoco
import gymnasium as gym              # 使用新版 Gymnasium（gym 的维护分支）
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback  # 训练回调
from stable_baselines3.common.monitor import Monitor   # 记录回合奖励/长度等信息
import torch.nn as nn
import warnings
import torch
import mujoco.viewer
import time
from typing import Optional
from scipy.spatial.transform import Rotation as R
import os

warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.on_policy_algorithm")

def write_flag_file(flag_filename="rl_visu_flag"):
    """创建标志文件，标记已有进程开启可视化"""
    flag_path = os.path.join("/tmp", flag_filename)
    try:
        with open(flag_path, "w") as f:
            f.write("flag")
        return True
    except Exception:
        return False

def check_flag_file(flag_filename="rl_visu_flag"):
    """检查标志文件是否存在"""
    return os.path.exists(os.path.join("/tmp", flag_filename))

def delete_flag_file(flag_filename="rl_visu_flag"):
    """删除标志文件"""
    flag_path = os.path.join("/tmp", flag_filename)
    if os.path.exists(flag_path):
        os.remove(flag_path)
    return True

# 标志文件机制：多进程训练时仅首个子环境开启可视化窗口

class PandaPickupEnv(gym.Env):
    """
    Panda 机械臂抓取并放置方块强化学习环境。

    状态（观测）：29 维
        7(关节角 q) + 7(关节速度 dq) + 3(末端位置) + 3(方块位置) +
        3(目标位置) + 3(末端到方块向量) + 2(夹爪开合) + 1(抓取阶段标志)
    动作：7 维，归一化 [-1,1]，表示 7 个臂关节的增量（夹爪由脚本化逻辑控制，不由策略输出）
    奖励：阶梯式（stage_base + 阶段内稠密奖励），并叠加姿态、平滑、限位等惩罚
    终止条件：方块掉落（失败）或回合步数超限（截断）
    """

    def __init__(self, visualize: bool = False):
        """初始化环境。visualize 控制是否启用可视化（多进程下仅首个子环境生效）"""
        super().__init__()

        # 标志文件机制：确保只有一个进程开启可视化
        if not check_flag_file():
            write_flag_file()
            self.visualize = visualize
        else:
            self.visualize = False
        self.handle = None

        # 加载含方块与夹爪的场景模型
        self.xml_path = './model/franka_emika_panda/scene_with_rl_pickup_cube.xml'
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        if self.visualize:
            self.handle = mujoco.viewer.launch_passive(self.model, self.data)
            self.handle.cam.distance = 3.2
            self.handle.cam.azimuth = 0.0
            self.handle.cam.elevation = -35.0
            self.handle.cam.lookat = np.array([0.3, 0.0, 0.3])

        self.np_random = np.random.default_rng(None)

        # ===== 通过名称获取关键 body / joint 的 id =====
        self.end_effector_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'ee_center_body'
        )
        self.cube_body_name = "cube"
        self.cube_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.cube_body_name
        )
        self.left_finger_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'left_finger'
        )
        self.right_finger_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'right_finger'
        )
        self.cube_joint_name = "cube_joint"
        self.cube_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, self.cube_joint_name
        )

        # 方块使用自由关节（freejoint）：3 位置 + 4 四元数 = 7 维
        self.freejoint_dim = 7
        # jnt_qposadr 给出该关节在 qpos 数组中的起始索引
        self.cube_qpos_start = self.model.jnt_qposadr[self.cube_joint_id]
        self.cube_qpos_end = self.cube_qpos_start + self.freejoint_dim

        # 机械臂+夹爪维度：7 臂关节 + 2 夹爪关节 = 9
        self.arm_gripper_qpos_dim = 9
        # home 位姿从场景 keyframe 读取
        self.home_joint_pos = self.model.key_qpos[0][:self.arm_gripper_qpos_dim].copy().astype(np.float32)

        mujoco.mj_forward(self.model, self.data)
        # 记录方块默认位置（reset 时在此基础上随机扰动）
        self.cube_default_pos = self.data.body(self.cube_id).xpos.copy().astype(np.float32)
        self.cube_init_pos = self.cube_default_pos.copy()

        # 动作空间：7个臂关节增量（夹爪由脚本控制）
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

        # 观测空间
        # 7(q) + 7(dq) + 3(ee_pos) + 3(cube_pos) + 3(target) + 3(ee_to_cube) + 2(gripper) + 1(grasp_phase) = 29
        self.obs_size = 29
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32
        )

        # ===== 任务参数 =====
        self.target_place_pos = np.array([0.55, -0.12, 0.02], dtype=np.float32)   # 放置目标位置
        self.lift_threshold = 0.05         # 抬升成功判定阈值（方块相对初始高度）
        # 成功阈值
        self.place_threshold = 0.03        # 放置成功判定阈值（方块到目标距离）
        # 助力把方块抬到这个高度再交给策略，避免太起来就放回
        self.carry_height = 0.12           # 搬运高度（抬到该高度后才进入放置阶段）
        # 离目标横向距离小于此值时撤掉助力，让策略自由下放
        self.place_release_dist = 0.08

        # 夹爪控制指令（Panda 夹爪为位置控制，0=全闭，255=全开）
        self.gripper_open = 255.0
        self.gripper_close = 0.0

        # 方块位置随机扰动范围（reset 时在默认位置上加噪声，提升泛化性）
        self.cube_rand_range = 0.01
        # 后面可以根据 episode 增加随机范围（课程学习）
        # self.cube_rand_range = min(0.05, 0.01 + episode * 0.00001)

        self.max_delta_q = 0.05            # 单步关节角度增量上限（增量位置控制）
        self.control_dt = 0.04             # 控制周期（每步 40ms）
        self.sim_dt = self.model.opt.timestep   # 仿真步长

        # 姿态保持权重：让策略倾向保持自然 home 姿态（远离奇异点与限位）
        self.posture_weights = np.array([0.08, 0.08, 0.06, 0.10, 0.18, 0.18, 0.25], dtype=np.float32)
        self.joint_margin = 0.15           # 关节限位安全余量

        # 历史状态缓存（用于奖励计算与平滑性惩罚）
        self.last_action = np.zeros(7, dtype=np.float32)
        self.last_gripper_width = 0.04
        self.last_ee_z = 0.0
        self.last_cube_z = 0.0
        self.cube_init_z = 0.0
        self.episode_steps = 0
        self.max_episode_steps = 250       # 单回合最大步数（截断阈值）

        # ===== 阶段标记（任务进度状态机） =====
        self.touched = False               # 末端是否接触过方块
        self.gripper_opened = False
        self.gripper_opened_step = 0
        self.grasped = False               # 是否已抓稳方块
        self.lifted = False                # 是否已抬升到指定高度
        self.lost_contact_steps = 0        # 丢失接触的连续步数（用于抓取消抖）
        self._assist_on = False            # 是否启用抬升助力
        # 夹爪脚本化状态：0=张开靠近, 1=闭合抓取, 2=提升
        self.grasp_phase = 0
        self.close_steps = 0  # 已闭合的步数

    def _render_scene(self):
        """渲染可视化场景：绿色球=放置目标点，红色球=方块实时位置"""
        if not self.visualize or self.handle is None:
            return

        self.handle.user_scn.ngeom = 0

        # 目标点（绿色）
        if self.handle.user_scn.ngeom < self.handle.user_scn.maxgeom:
            g = self.handle.user_scn.geoms[self.handle.user_scn.ngeom]
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.02, 0.0, 0.0],
                pos=self.target_place_pos,
                mat=np.eye(3).flatten(),
                rgba=np.array([0.1, 0.8, 0.2, 0.7], dtype=np.float32)
            )
            self.handle.user_scn.ngeom += 1

        # 方块实时位置（红色）
        if self.handle.user_scn.ngeom < self.handle.user_scn.maxgeom:
            g = self.handle.user_scn.geoms[self.handle.user_scn.ngeom]
            current_cube_pos = self.data.body(self.cube_id).xpos.copy()
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.02, 0.0, 0.0],
                pos=current_cube_pos,
                mat=np.eye(3).flatten(),
                rgba=np.array([0.8, 0.2, 0.1, 0.7], dtype=np.float32)
            )
            self.handle.user_scn.ngeom += 1

    def reset(self, seed: Optional[int] = None, options=None):
        """
        重置环境到新回合初始状态。

        步骤：
            1. 复位机械臂+夹爪到 home 位姿
            2. 方块位置加随机扰动（仅 XY，Z 不变）
            3. 重置所有阶段标记
            4. 返回初始观测

        返回：(观测, 信息字典)
        """
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        # 复位机械臂+夹爪
        self.data.qpos[:self.arm_gripper_qpos_dim] = self.home_joint_pos.copy()
        self.data.qvel[:self.arm_gripper_qpos_dim] = 0.0
        self.data.ctrl[:7] = self.home_joint_pos[:7]
        self.data.ctrl[7] = self.gripper_open       # 初始夹爪张开

        # 方块位置随机扰动（仅 XY 方向，Z 保持桌面高度）
        cube_rand = self.np_random.uniform(-self.cube_rand_range, self.cube_rand_range, size=3)
        cube_rand[2] = 0.0
        self.cube_init_pos = self.cube_default_pos + cube_rand
        self.data.qpos[self.cube_qpos_start:self.cube_qpos_start+3] = self.cube_init_pos

        # 保持默认四元数（方块无初始旋转）
        default_quat = self.model.qpos0[self.cube_qpos_start+3:self.cube_qpos_end]
        self.data.qpos[self.cube_qpos_start+3:self.cube_qpos_end] = default_quat

        mujoco.mj_forward(self.model, self.data)

        # 重置阶段标记（状态机回到起点）
        self.touched = False
        self.gripper_opened = False
        self.gripper_opened_step = 0
        self.grasped = False
        self.lifted = False
        self.lost_contact_steps = 0
        self._assist_on = False
        self.grasp_phase = 0
        self.close_steps = 0

        self.episode_steps = 0
        self.last_action = np.zeros(7, dtype=np.float32)
        self.last_gripper_width = float(self.data.qpos[7:9].mean())
        ee_init_pos = self.data.body(self.end_effector_id).xpos.copy()
        self.last_ee_z = ee_init_pos[2]
        # 记录方块初始高度，作为抬升判定的基准
        self.cube_init_z = float(self.data.body(self.cube_id).xpos[2])
        self.last_cube_z = self.cube_init_z

        if self.visualize:
            self._render_scene()

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """
        构造 29 维观测向量：
            [7 关节角 q, 7 关节速度 dq, 3 末端位置, 3 方块位置,
             3 目标位置, 3 末端到方块向量, 2 夹爪开合, 1 抓取阶段]

        观测叠加高斯噪声（std=0.001），模拟真实感知噪声，提升鲁棒性。
        """
        q = self.data.qpos[:7].copy().astype(np.float32)
        dq = self.data.qvel[:7].copy().astype(np.float32)

        gripper_q = self.data.qpos[7:9].copy().astype(np.float32)

        ee_pos = self.data.body(self.end_effector_id).xpos.copy().astype(np.float32)
        cube_pos = self.data.body(self.cube_id).xpos.copy().astype(np.float32)
        target_pos = self.target_place_pos.copy()

        ee_to_cube = cube_pos - ee_pos       # 末端指向方块的向量

        obs = np.concatenate([
            q,              # 7
            dq,             # 7
            ee_pos,         # 3
            cube_pos,       # 3
            target_pos,     # 3
            ee_to_cube,     # 3
            gripper_q,      # 2
            np.array([float(self.grasp_phase)], dtype=np.float32),  # 1
        ]).astype(np.float32)

        # 观测噪声
        obs += np.random.normal(0, 0.001, obs.shape).astype(np.float32)
        return obs

    def _is_grasped_by_contact(self) -> bool:
        """
        基于接触点判断是否真正抓稳方块。

        判定条件：
            - 左手指与方块有接触
            - 右手指与方块有接触
            - 夹爪开合度 < 0.035（夹爪已闭合到足够窄）

        这种基于物理接触的判定比单纯依靠夹爪指令更可靠，
        能区分「夹爪闭合但没抓到」与「夹爪闭合且抓稳」两种情况。
        """
        left_contact = False
        right_contact = False

        # 遍历所有接触点，检查是否包含 (方块, 左指) 与 (方块, 右指) 的接触对
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = contact.geom1
            body2 = contact.geom2
            body1_id = self.model.geom_bodyid[body1]
            body2_id = self.model.geom_bodyid[body2]
            bodies = {body1_id, body2_id}
            if self.cube_id in bodies and self.left_finger_id in bodies:
                left_contact = True
            if self.cube_id in bodies and self.right_finger_id in bodies:
                right_contact = True

        gripper_width = self.data.qpos[7:9].mean()
        return left_contact and right_contact and gripper_width < 0.035

    def _calc_reward(self, action: np.ndarray) -> tuple[float, bool, bool, bool]:
        """
        阶梯式奖励函数：stage_base + 阶段内稠密奖励 ∈ [0,1]。

        设计思想：
            将长时序任务拆分为 4 个阶段（stage0~3），每个阶段有一个 base 基础奖励，
            阶段越高 base 越大，从而引导智能体逐级推进任务进度。同一时刻只处于一个 stage。
            阶段内再给出稠密的进度奖励（如接近度、抬升高度、放置横向距离等）。

        阶段结构：
            stage0 接近：base=0 + reach奖励（末端横向/垂直接近方块）
            stage1 闭合：base=1 + 抓取奖励（夹爪闭合且抓稳）
            stage2 抬升：base=1 + 2*lift（方块高度提升）
            stage3 放置：base=3 + 4*place_xy + 0.5*place_z（横向接近目标 + 高度下放）

        返回：(总奖励, 是否终止, 是否成功, 是否掉落)
        """

        # ===== 读取关键状态量 =====
        ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        cube_pos = self.data.body(self.cube_id).xpos.copy()
        gripper_width = self.data.qpos[7:9].mean()

        dist_ee2cube = np.linalg.norm(ee_pos - cube_pos)                          # 末端到方块距离
        dist_cube2target = np.linalg.norm(cube_pos - self.target_place_pos)       # 方块到目标距离
        lateral_to_target = np.linalg.norm(cube_pos[:2] - self.target_place_pos[:2])  # 方块到目标横向距离
        cube_z = cube_pos[2]                                                      # 方块当前高度

        reward = 0.0

        # 接近阶段的几何量
        lateral_dist = np.linalg.norm(ee_pos[:2] - cube_pos[:2])     # 末端到方块横向距离
        vertical_offset = ee_pos[2] - cube_pos[2] - 0.025            # 末端相对方块上方的目标高度偏移
        vertical_gap = ee_pos[2] - cube_pos[2]                       # 末端与方块的垂直高度差

        # ===== 末端姿态度量 =====
        # alignment：末端 Z 轴与世界 -Z 的点积，越接近 1 表示末端越朝下（抓取所需姿态）
        ee_rot = self.data.body(self.end_effector_id).xmat.reshape(3, 3)
        alignment = np.dot(ee_rot[:, 2], np.array([0.0, 0.0, -1.0]))
        # yaw_align：手指 Y 轴在水平面的投影与方块边的对齐程度（避免斜着抓）
        # 计算Y轴水平方向投影
        finger_axis_xy = (-ee_rot[:, 1])[:2]
        finger_axis_xy_norm = np.linalg.norm(finger_axis_xy)
        if finger_axis_xy_norm > 1e-6:
            finger_axis_xy = finger_axis_xy / finger_axis_xy_norm
            # 只关心和方块的一个边平行即可，不是斜着就可以
            yaw_align = max(abs(finger_axis_xy[0]), abs(finger_axis_xy[1]))
        else:
            yaw_align = 0.0

        is_physically_grasped = self._is_grasped_by_contact()

        # ===== 抓取标志位消抖：避免短暂脱接触导致状态频繁跳变 =====
        # 若已标记 grasped，但连续 4 步物理上未抓稳，则取消 grasped 标志
        if self.grasped:
            if is_physically_grasped and gripper_width < 0.035:
                self.lost_contact_steps = 0
            else:
                self.lost_contact_steps += 1
                if self.lost_contact_steps >= 4:
                    self.grasped = False

        # ===== 主奖励：根据当前阶段给出 stage_base + 稠密奖励 =====
        # 每步只落在一个 stage
        if not self.grasped:
            if self.grasp_phase == 0:
                # stage0 接近：base 0 + reach∈[0,1]
                # reach = 横向接近度 * 0.5 + 垂直接近度 * 0.5（tanh 平滑映射）
                reach = 0.5 * (1.0 - np.tanh(6.0 * lateral_dist)) \
                        + 0.5 * (1.0 - np.tanh(6.0 * abs(vertical_offset)))
                reward += reach  # ≤1

                # 当末端接近方块上方时，额外奖励姿态对齐
                if (lateral_dist < 0.018 and 0.01 < vertical_gap < 0.038):
                    pose = 0.5 * alignment + 0.5 * yaw_align
                    reward += pose

                # 里程碑: 到达抓取位姿（横向对准 + 高度合适 + 朝下 + 偏航对齐）
                # 满足后进入 stage1（闭合阶段）
                if (lateral_dist < 0.018 and 0.008 < vertical_gap < 0.038
                        and alignment > 0.965 and yaw_align > 0.93):
                    reward += 1.0
                    self.grasp_phase = 1
                    self.close_steps = 0

                # 记录到 tensorboard 中
                self.alignment = alignment
                self.ee_yaw = yaw_align
            else:
                # stage1 闭合：base 1.0 + 0.5
                reward += 1.0
                if is_physically_grasped:
                    reward += 0.5
                    cube_velocity = np.linalg.norm(
                        self.data.qvel[self.cube_qpos_start:self.cube_qpos_start+3]
                    )
                    # 两个夹抓贴着一边推着走的情况：方块速度小才算真正抓稳
                    if cube_velocity < 0.05:
                        reward += 1.0
                        self.grasped = True
                        self.lost_contact_steps = 0
        else:
            # 防止拿起来放下去：已抬升但方块掉回桌面且离目标远，则回退 lifted 标志
            if (self.lifted and cube_z < self.lift_threshold - 0.02 and lateral_to_target > 0.06):
                self.lifted = False

            if not self.lifted:
                # stage2 抬升中：base 1.0 + 2*lift∈[0,2]
                reward += 1.0
                # 抬升归一化进度：方块相对初始高度的提升比例
                lift_norm = np.clip((cube_z - self.cube_init_z) / self.lift_threshold, 0.0, 1.0)
                reward += 2.0 * lift_norm  # ≤2
                if cube_z > self.lift_threshold:
                    reward += 1.0           # 达到抬升阈值，里程碑奖励
                    self.lifted = True
            else:
                # stage3 搬运放置：base 3.0 + 4*place_xy + 0.5*place_z
                reward += 3.0
                # place_xy：方块横向接近目标的程度（tanh 平滑）
                place_xy = 1.0 - np.tanh(2.5 * lateral_to_target)
                reward += 4.0 * place_xy  # ≤4
                # 高度引导：离目标远时保持 carry_height（防脱手），近时下放到目标高度
                target_blend = 1.0 - np.tanh(8.0 * lateral_to_target)  # lat=0→1, lat=0.15→0
                # lat 大 → desired_z ≈ carry_height（托住，防脱手）
                # lat 小 → desired_z ≈ target_z（鼓励下放）
                desired_z = (self.carry_height * (1.0 - target_blend)
                             + self.target_place_pos[2] * target_blend)
                z_err = abs(cube_z - desired_z)
                reward += 0.5 * (1.0 - np.tanh(8.0 * z_err))  # ≤0.5

        # ===== 成功判定：抓稳 + 抬升 + 方块到达目标 =====
        is_success = self.grasped and self.lifted and dist_cube2target < self.place_threshold
        if is_success:
            reward += 1.0

        # ===== 姿态保持奖励（抬升前鼓励保持朝下姿态） =====
        if not self.lifted:
            reward += 0.3 * (alignment - 1.0)
            reward += 0.2 * (yaw_align - 1.0)

        # ===== 惩罚项 =====
        # 惩罚项 ≤0.1
        # 未抓取时末端贴近方块但方块速度大（推方块而非抓取）的情况
        cube_velocity = np.linalg.norm(self.data.qvel[self.cube_qpos_start:self.cube_qpos_start+3])
        if not self.grasped and dist_ee2cube < 0.05 and cube_velocity > 0.05:
            reward -= 0.3

        # 姿态偏离 home 的惩罚（归一化到关节范围），鼓励保持自然姿态
        q = self.data.qpos[:7].copy()
        dq = self.data.qvel[:7].copy()
        q_home = self.home_joint_pos[:7]
        joint_ranges = self.model.jnt_range[:7]
        range_width = joint_ranges[:, 1] - joint_ranges[:, 0]

        posture_error = (q - q_home) / range_width
        posture_penalty = np.sum(self.posture_weights * np.square(posture_error))
        reward -= 0.05 * posture_penalty

        # 关节限位安全余量惩罚：接近限位时施惩罚，避免到达极限
        lower_margin = q - joint_ranges[:, 0]
        upper_margin = joint_ranges[:, 1] - q
        limit_violation = np.maximum(0.0, self.joint_margin - np.minimum(lower_margin, upper_margin))
        reward -= 0.5 * np.sum(np.square(limit_violation / self.joint_margin))

        # 关节速度惩罚（抑制剧烈运动）+ 动作平滑性惩罚 + 时间惩罚
        reward -= 0.003 * np.linalg.norm(dq)
        reward -= 0.01 * np.linalg.norm(action - self.last_action)
        reward += -0.02  # 时间惩罚

        # 方块掉落惩罚：方块高度低于 -0.02 视为掉落，重罚并终止
        is_drop = cube_z < -0.02
        if is_drop:
            reward -= 5.0

        terminated = is_drop
        return reward, terminated, is_success, is_drop

    def _scripted_gripper(self) -> float:
        """
        脚本化夹爪控制：根据当前阶段自动决定夹爪开合，不由策略输出。

        状态机：
            phase 0 张开靠近：末端到达抓取位姿时切换到 phase 1
            phase 1 闭合抓取：闭合若干步后若抓稳则进入 phase 2；超时则回退 phase 0
            phase 2 提升后保持闭合

        返回：夹爪控制指令（gripper_open 或 gripper_close）

        设计动机：夹爪开合是离散决策，交给策略学习会增加难度；
        通过脚本化可让策略专注于连续的臂关节控制，降低学习难度。
        """
        ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        cube_pos = self.data.body(self.cube_id).xpos.copy()

        lateral_dist = np.linalg.norm(ee_pos[:2] - cube_pos[:2])
        vertical_dist = ee_pos[2] - cube_pos[2]

        ee_rot = self.data.body(self.end_effector_id).xmat.reshape(3, 3)
        alignment = np.dot(ee_rot[:, 2], np.array([0.0, 0.0, -1.0]))

        finger_axis_world = -ee_rot[:, 1]
        finger_axis_xy = finger_axis_world[:2]
        finger_axis_xy_norm = np.linalg.norm(finger_axis_xy)
        if finger_axis_xy_norm > 1e-6:
            finger_axis_xy = finger_axis_xy / finger_axis_xy_norm
            yaw_align = max(abs(finger_axis_xy[0]), abs(finger_axis_xy[1]))
        else:
            yaw_align = 0.0

        if self.grasp_phase == 0:
            # 张开靠近：满足抓取位姿条件后进入闭合阶段
            if lateral_dist < 0.018 and 0.008 < vertical_dist < 0.038 and alignment > 0.965 and yaw_align > 0.93:
                self.grasp_phase = 1
                self.close_steps = 0
        elif self.grasp_phase == 1:
            # 闭合抓取：累计闭合步数，抓稳则进入 phase 2，超时则回退 phase 0
            self.close_steps += 1
            if self.close_steps > 5 and self._is_grasped_by_contact():
                self.grasp_phase = 2
            elif self.close_steps > 15:
                self.grasp_phase = 0

        # phase 0 时张开，其余阶段闭合
        if self.grasp_phase == 0:
            return self.gripper_open
        else:
            return self.gripper_close

    def step(self, action: np.ndarray):
        """
        执行一步环境交互。

        参数：
            action: 7 维归一化动作 [-1,1]，表示臂关节增量方向与幅度

        流程：
            1. 动作裁剪
            2. 助力机制：抬升阶段混合 home 动作，帮助可靠抬起方块
            3. 增量位置控制：动作 -> 关节角度增量 -> 目标角度（裁剪到限位）
            4. 脚本化夹爪控制
            5. 多步物理仿真（control_dt 内多次 mj_step）
            6. 计算奖励、终止/截断、观测

        返回：(观测, 奖励, 是否终止, 是否截断, 信息字典)
        """
        action = np.clip(action, -1.0, 1.0)
        cube_pos_now = self.data.body(self.cube_id).xpos
        cube_z_now = cube_pos_now[2]

        # ===== 助力机制（assist）：在抬升阶段混合 home 动作 =====
        # 已抓稳但尚未抬升，且方块低于 carry_height 时开启助力
        if not self.grasped or self.lifted:
            self._assist_on = False
        elif cube_z_now < self.carry_height - 0.03:
            self._assist_on = True
        elif cube_z_now >= self.carry_height:
            self._assist_on = False

        commanded_action = action.copy()
        if self._assist_on:
            # 计算「回到 home」方向的动作
            current_q = self.data.qpos[:7]
            home_action = np.clip(
                (self.home_joint_pos[:7] - current_q) / self.max_delta_q, -1.0, 1.0
            )
            # 保证从桌面可靠抬起；lifted 后此分支已不再进入
            # 50% home 动作 + 50% 策略动作，混合输出
            blend = 0.5
            commanded_action = np.clip(blend * home_action + (1-blend) * action, -1.0, 1.0)

        # ===== 增量位置控制：动作 -> 关节角度增量 =====
        current_q = self.data.qpos[:7].copy()
        for i in range(7):
            delta = commanded_action[i] * self.max_delta_q
            target = current_q[i] + delta
            jnt_range = self.model.jnt_range[i]
            self.data.ctrl[i] = float(np.clip(target, jnt_range[0], jnt_range[1]))

        # ===== 脚本化夹爪控制 =====
        gripper_target = self._scripted_gripper()
        self.data.ctrl[7] = gripper_target
        if self.model.nu > 8:
            self.data.ctrl[8] = gripper_target

        # ===== 仿真步进：一个控制周期内执行多个仿真子步 =====
        n_steps = int(self.control_dt / self.sim_dt)
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)
            # 方块掉出桌面则提前停止仿真
            if self.data.body(self.cube_id).xpos[2] < -0.05:
                break

        self.episode_steps += 1

        # 可视化
        if self.visualize and self.handle is not None:
            self._render_scene()
            self.handle.sync()
            time.sleep(self.control_dt)

        reward, terminated, is_success, is_drop = self._calc_reward(action)

        # 截断判定：超过最大步数
        truncated = self.episode_steps >= self.max_episode_steps
        if truncated:
            reward -= 1.0

        obs = self._get_observation()

        # 信息字典：用于监控与 TensorBoard 记录
        info = {
            "is_success": is_success,
            "cube_to_target_dist": np.linalg.norm(self.data.body(self.cube_id).xpos - self.target_place_pos),
            "ee_to_cube_dist": np.linalg.norm(self.data.body(self.end_effector_id).xpos - self.data.body(self.cube_id).xpos),
            "cube_z": self.data.body(self.cube_id).xpos[2],
            "collision": self.data.ncon > 3,
            "cube_drop": is_drop,
            "grasped": self.grasped,
            "lifted": self.lifted,
            "alignment": self.alignment,
            "yaw_align": self.ee_yaw
        }

        # 更新历史状态缓存
        self.last_action = action.copy()
        gripper_width = self.data.qpos[7:9].mean()
        self.last_gripper_width = gripper_width
        self.last_ee_z = self.data.body(self.end_effector_id).xpos[2]
        self.last_cube_z = self.data.body(self.cube_id).xpos[2]

        return obs, float(reward), terminated, truncated, info

    def close(self):
        """关闭环境，释放可视化窗口资源"""
        if self.visualize and self.handle is not None:
            self.handle.close()
            self.handle = None


class TaskMetricsCallback(BaseCallback):
    """
    自定义训练回调：将任务相关指标（抓取率、抬升率、成功率、对齐度等）
    汇总记录到 TensorBoard，便于监控训练进度与诊断问题。

    工作方式：每个回合结束时收集 info 中的指标，每隔若干步求均值后记录。
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        # 回合级指标缓冲区
        self.ep_buf = {"grasped": [], "lifted": [],
                       "is_success": [], "cube_z_max": [],
                       "alignment": [], "yaw_align": [],
                       "cube_to_target_dist": [], "ee_to_cube_dist": []}

    def _on_step(self) -> bool:
        """
        每步回调：在回合结束时收集 info 中的任务指标，定时汇总到 TensorBoard。
        返回 True 表示继续训练（返回 False 会提前终止训练）。
        """
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        # 遍历所有并行环境，回合结束时收集指标
        for i, info in enumerate(infos):
            if dones[i] and "episode" in info:
                self.ep_buf["grasped"].append(float(info.get("grasped", False)))
                self.ep_buf["lifted"].append(float(info.get("lifted", False)))
                self.ep_buf["is_success"].append(float(info.get("is_success", False)))
                self.ep_buf["cube_z_max"].append(float(info.get("cube_z", 0.0)))
                self.ep_buf["alignment"].append(float(info.get("alignment", 0.0)))
                self.ep_buf["yaw_align"].append(float(info.get("yaw_align", 0.0)))
                self.ep_buf["cube_to_target_dist"].append(float(info.get("cube_to_target_dist", 0.0)))
                self.ep_buf["ee_to_cube_dist"].append(float(info.get("ee_to_cube_dist", 0.0)))

        # 每 2048 步汇总一次（取均值后记录，并清空缓冲）
        # 每 1000 步
        if self.num_timesteps % 2048 == 0 and len(self.ep_buf["grasped"]) > 0:
            for k, v in self.ep_buf.items():
                if v:
                    self.logger.record(f"task/{k}_rate", np.mean(v))
            self.ep_buf = {k: [] for k in self.ep_buf}
        return True


def train_ppo(
    n_envs: int = 16,
    total_timesteps: int = 5_000_000,
    model_save_path: str = "panda_pickup_ppo",
    visualize: bool = False,
    resume_from: Optional[str] = None
):
    """
    PPO 训练入口。

    参数：
        n_envs: 并行环境数（多进程加速采样）
        total_timesteps: 训练总步数
        model_save_path: 模型保存路径
        visualize: 是否启用可视化（仅首个子环境生效）
        resume_from: 断点恢复路径（None 表示从头训练）

    主要流程：
        1. 构建多进程并行环境（SubprocVecEnv + Monitor）
        2. 配置 PPO 超参数与网络结构
        3. 调用 learn 开始训练
        4. 保存模型
    """
    delete_flag_file()

    ENV_KWARGS = {'visualize': visualize}

    def make_env():
        """构造单个环境：PandaPickupEnv + Monitor（记录回合级信息）"""
        env = PandaPickupEnv(**ENV_KWARGS)
        env = Monitor(env, info_keywords=("is_success", "grasped", "lifted", "cube_z", "ee_to_cube_dist"))
        return env

    # 多进程并行环境：显著加速采样
    env = make_vec_env(
        make_env,
        n_envs=n_envs,
        seed=42,
        vec_env_cls=SubprocVecEnv,
    )

    print(f"观测空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    print(f"并行环境数: {n_envs}")

    if resume_from is not None and os.path.exists(resume_from + ".zip"):
        # 断点恢复：加载已有模型继续训练
        print(f"从 {resume_from} 恢复训练...")
        model = PPO.load(resume_from, env=env)
    else:
        # 策略网络结构：策略网络与价值网络均为 [512, 256, 128] 三层 MLP
        policy_kwargs = dict(
            activation_fn=nn.LeakyReLU,
            net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
            log_std_init=-0.5,
        )

        # 学习率线性衰减到 0：后期更新变小，收敛更稳，抑制末期震荡
        def lr_schedule(progress_remaining: float) -> float:
            return 3e-4 * progress_remaining

        # PPO 超参数说明：
        #   n_steps: 每个环境单次 rollout 步数（rollout buffer 大小 = n_steps * n_envs）
        #   batch_size: 每次优化的 mini-batch 大小
        #   n_epochs: 每次 rollout 数据复用次数
        #   gamma: 折扣因子，0.995 偏向长程回报
        #   gae_lambda: GAE 优势估计的 lambda 参数
        #   ent_coef: 熵正则项系数，鼓励探索
        #   clip_range: PPO 裁剪范围，限制策略更新幅度
        #   target_kl: 早期停止的 KL 阈值，防止策略更新过大
        model = PPO(
            policy="MlpPolicy",
            env=env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            n_steps=256,
            batch_size=1024,
            n_epochs=5,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.005,
            clip_range=0.2,
            max_grad_norm=0.5,
            learning_rate=lr_schedule,
            target_kl=0.03,
            device="cuda:1" if torch.cuda.is_available() else "cpu",
            tensorboard_log="./tensorboard/panda_pickup/",
        )
    eval_env = make_vec_env(make_env, n_envs=1, seed=999)

    print(f"开始训练，总步数: {total_timesteps}")
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        callback=TaskMetricsCallback(),
    )

    model.save(model_save_path)
    env.close()
    eval_env.close()
    print(f"模型已保存至: {model_save_path}")
    delete_flag_file()


def test_ppo(
    model_path: str = "panda_pickup_ppo",
    total_episodes: int = 10,
    render: bool = True
):
    """
    加载训练好的 PPO 模型进行测试评估。

    参数：
        model_path: 模型文件路径（不含 .zip 后缀）
        total_episodes: 测试回合数
        render: 是否启用可视化

    测试使用 deterministic=True（贪心策略），不添加随机性，便于复现评估。
    """
    delete_flag_file()

    env = PandaPickupEnv(visualize=render)
    model = PPO.load(model_path, env=env)
    # model = PPO.load(model_path, device="cpu", env=env)
    # model.to("cuda")

    success_count = 0
    grasp_count = 0
    print(f"开始测试，总轮数: {total_episodes}")

    for ep in range(total_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            # 贪心预测：不带随机性
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            steps += 1

        if info["is_success"]:
            success_count += 1
        if info["grasped"]:
            grasp_count += 1

        print(f"回合{ep+1:2d} | 步数:{steps:3d} | 奖励:{ep_reward:7.2f} | "
              f"成功:{info['is_success']} | 抓取:{info['grasped']} | "
              f"最终高度:{info['cube_z']:.3f}")

    print(f"\n测试完成:")
    print(f"  成功率: {success_count}/{total_episodes} ({success_count/total_episodes*100:.1f}%)")
    print(f"  抓取率: {grasp_count}/{total_episodes} ({grasp_count/total_episodes*100:.1f}%)")

    env.close()
    delete_flag_file()


if __name__ == "__main__":
    """
    主入口：通过 TRAIN_MODE 切换训练/测试。

    训练模式：
        - 64 个并行环境
        - 6000 万步
        - 开启可视化观察首个子环境
    测试模式：
        - 20 个回合评估
        - 渲染可视化
    """
    TRAIN_MODE = False

    MODEL_SAVE_PATH = "assets/model/rl_pickup_checkpoint/panda_pickup_v1_5"
    RESUME_PATH = None

    if TRAIN_MODE:
        # 训练前清空旧的 tensorboard 日志，避免曲线混乱
        import shutil
        if os.path.exists("./tensorboard/panda_pickup/"):
            shutil.rmtree("./tensorboard/panda_pickup/")

        train_ppo(
            n_envs=64,
            total_timesteps=60_000_000,
            model_save_path=MODEL_SAVE_PATH,
            visualize=True,
            resume_from=RESUME_PATH,
        )
    else:
        test_ppo(
            model_path=MODEL_SAVE_PATH,
            total_episodes=20,
            render=True,
        )
    os.system("date")
