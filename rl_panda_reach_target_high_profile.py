"""
Franka Emika Panda 机械臂强化学习——到达目标点任务（Reach Target）

任务说明：
    控制 7 自由度 Panda 机械臂，使其末端执行器从 home 位姿运动到工作空间内随机生成的目标点。
    使用 PPO（Proximal Policy Optimization）算法在 MuJoCo 物理仿真环境中训练策略。

主要组件：
    1. PandaObstacleEnv: 自定义 Gym 环境，封装 MuJoCo 仿真、奖励计算、状态观测等
    2. train_ppo: PPO 训练入口，支持多进程并行采样与断点恢复
    3. test_ppo: 加载已训练模型进行可视化的测试评估

技术栈：MuJoCo（物理仿真）+ Gym（RL 环境接口）+ Stable-Baselines3（PPO 实现）+ PyTorch（神经网络）
"""

import numpy as np
import mujoco                       # MuJoCo 物理引擎，用于机器人仿真
import gym                          # OpenAI Gym，提供标准 RL 环境接口
from gym import spaces              # 定义观测空间与动作空间的数据结构
from stable_baselines3 import PPO   # Stable-Baselines3 提供的 PPO 算法实现
from stable_baselines3.common.env_util import make_vec_env       # 快速创建向量化环境的工具
from stable_baselines3.common.vec_env import SubprocVecEnv       # 多进程向量化环境，加速并行采样
import torch.nn as nn
import warnings
import torch
import mujoco.viewer                # MuJoCo 可视化窗口
import time
from typing import Optional
from scipy.spatial.transform import Rotation as R  # 用于四元数与欧拉角之间的转换

# 忽略stable-baselines3的冗余UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.on_policy_algorithm")

import os

def write_flag_file(flag_filename="rl_visu_flag"):
    """创建标志文件，用于标记当前是否有进程启用了可视化窗口"""
    flag_path = os.path.join("/tmp", flag_filename)
    try:
        with open(flag_path, "w") as f:
            f.write("This is a flag file")
        return True
    except Exception as e:
        return False

def check_flag_file(flag_filename="rl_visu_flag"):
    """检查标志文件是否存在（即是否已有进程在可视化）"""
    flag_path = os.path.join("/tmp", flag_filename)
    return os.path.exists(flag_path)

def delete_flag_file(flag_filename="rl_visu_flag"):
    """删除标志文件，通常在训练/测试结束后调用以清理状态"""
    flag_path = os.path.join("/tmp", flag_filename)
    if not os.path.exists(flag_path):
        return True
    try:
        os.remove(flag_path)
        return True
    except Exception as e:
        return False

# 说明：标志文件机制用于多进程并行训练（SubprocVecEnv）场景。
# MuJoCo 的 GUI 可视化窗口只能由一个进程打开，因此通过 /tmp 下的标志文件
# 确保只有第一个创建的子环境会启动可视化窗口，其余子环境关闭可视化以避免冲突。

class PandaObstacleEnv(gym.Env):
    """
    Panda 机械臂到达目标点强化学习环境。

    该环境继承自 gym.Env，遵循标准的「观测-动作-奖励-终止」交互循环：
        reset() -> 观测
        step(action) -> (观测, 奖励, 终止, 截断, 信息)

    状态（观测）：7 维关节角度 + 3 维目标位置 = 10 维
    动作：7 维，归一化到 [-1, 1]，表示期望的 7 个关节角度（经缩放映射到关节限位内）
    奖励：以末端到目标的距离为主，叠加姿态、动作平滑、碰撞等多项惩罚/奖励
    终止条件：到达目标点（成功）或超时
    """

    def __init__(self, visualize: bool = False):
        """
        初始化环境。

        参数：
            visualize: 是否启用 MuJoCo GUI 可视化窗口。
                       在多进程训练中，仅首个子环境会真正启用可视化。
        """
        super(PandaObstacleEnv, self).__init__()
        # 标志文件机制：若尚未有进程开启可视化，则当前进程可开启；否则强制关闭
        if not check_flag_file():
            write_flag_file()
            self.visualize = visualize
        else:
            self.visualize = False
        self.handle = None

        # 加载 MuJoCo 场景模型与对应的数据容器
        self.model = mujoco.MjModel.from_xml_path('./model/franka_emika_panda/scene.xml')
        self.data = mujoco.MjData(self.model)

        # 启动被动式可视化窗口（passive 表示由用户代码驱动渲染刷新）
        if self.visualize:
            self.handle = mujoco.viewer.launch_passive(self.model, self.data)
            # 相机视角参数：距离、方位角、俯仰角、注视点
            self.handle.cam.distance = 3.0
            self.handle.cam.azimuth = 0.0
            self.handle.cam.elevation = -30.0
            self.handle.cam.lookat = np.array([0.2, 0.0, 0.4])

        # 通过名称获取末端执行器（ee_center_body）的 body id，后续用于读取其位姿
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ee_center_body')
        self.initial_ee_pos = np.zeros(3, dtype=np.float32)
        # 机械臂的 home（初始）关节位姿，每次 reset 会回到此位姿
        self.home_joint_pos = np.array([  # home位姿
            0.0, -np.pi/4, 0.0, -3*np.pi/4,
            0.0, np.pi/2, np.pi/4
        ], dtype=np.float32)

        self.goal_size = 0.03   # 可视化时目标球的半径

        # 约束工作空间：目标点在该范围内随机生成，避免出现不可达或危险目标
        self.workspace = {
            'x': [-0.5, 0.8],
            'y': [-0.5, 0.5],
            'z': [0.05, 0.3]
        }

        # 动作空间与观测空间
        # 动作空间：7 维连续空间，取值 [-1, 1]，分别对应 7 个关节的期望角度（归一化）
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        # 观测空间：7 维关节角度 + 3 维目标位置 = 10 维
        self.obs_size = 7 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32)

        self.goal = np.zeros(3, dtype=np.float32)             # 当前回合的目标点位置
        self.np_random = np.random.default_rng(None)          # 随机数生成器，用于目标点采样
        self.prev_action = np.zeros(7, dtype=np.float32)      # 上一时刻动作，用于动作平滑性惩罚
        self.goal_threshold = 0.005                           # 到达目标的判定阈值（5mm）

    def _get_valid_goal(self) -> np.ndarray:
        """
        在工作空间内采样一个「有效」目标点。

        有效约束：
            - 与初始末端位置的距离在 [0.4, 0.5] 之间（保证任务有一定难度，既不太近也不太远）
            - x 分量 > 0.2 且 z 分量 > 0.2（限定在前上方区域，避免地面/身后目标）
        若不满足则继续重采样。
        """
        while True:
            goal = self.np_random.uniform(
                low=[self.workspace['x'][0], self.workspace['y'][0], self.workspace['z'][0]],
                high=[self.workspace['x'][1], self.workspace['y'][1], self.workspace['z'][1]]
            )
            if 0.4 < np.linalg.norm(goal - self.initial_ee_pos) < 0.5 and goal[0] > 0.2 and goal[2] > 0.2:
                return goal.astype(np.float32)

    def _render_scene(self) -> None:
        """在可视化场景中渲染目标点（蓝色球体），便于直观观察任务目标"""
        if not self.visualize or self.handle is None:
            return
        # user_scn 是用户自定义场景层，ngeom 表示自定义几何体数量，先清零
        self.handle.user_scn.ngeom = 0
        total_geoms = 1
        self.handle.user_scn.ngeom = total_geoms

        # 渲染目标点（蓝色）
        goal_rgba = np.array([0.1, 0.1, 0.9, 0.9], dtype=np.float32)
        mujoco.mjv_initGeom(
            self.handle.user_scn.geoms[0],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[self.goal_size, 0.0, 0.0],
            pos=self.goal,
            mat=np.eye(3).flatten(),     # 旋转矩阵（单位阵表示无旋转）
            rgba=goal_rgba
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        """
        重置环境到一个新回合的初始状态。

        步骤：
            1. 重置 MuJoCo 仿真数据
            2. 将机械臂关节设置到 home 位姿，并前向计算得到末端初始位置
            3. 随机采样一个目标点
            4. 返回初始观测

        返回：(观测, 信息字典)
        """
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # 重置关节到home位姿
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = self.home_joint_pos
        # 前向运动学计算，更新 body 位姿（不进行动力学积分）
        mujoco.mj_forward(self.model, self.data)
        # 记录末端初始位置（作为直线奖励计算的起点）
        self.initial_ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        self.start_ee_pos = self.initial_ee_pos.copy()

        # 生成目标
        self.goal = self._get_valid_goal()
        if self.visualize:
            self._render_scene()

        obs = self._get_observation()
        self.start_t = time.time()        # 记录回合开始时间，用于超时判定
        return obs, {}

    def _get_observation(self) -> np.ndarray:
        """构造观测向量：7 维关节角度 + 3 维目标位置"""
        joint_pos = self.data.qpos[:7].copy().astype(np.float32)
        # ee_pos = self.data.body(self.end_effector_id).xpos.copy().astype(np.float32)
        # ee_quat = self.data.body(self.end_effector_id).xquat.copy().astype(np.float32)
        return np.concatenate([joint_pos, self.goal])

    # def _calc_reward(self, ee_pos: np.ndarray, ee_orient: np.ndarray, joint_angles: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, float]:
    #     dist_to_goal = np.linalg.norm(ee_pos - self.goal)
        
    #     # 非线性距离奖励
    #     if dist_to_goal < self.goal_threshold:
    #         distance_reward = 100.0
    #     elif dist_to_goal < 2*self.goal_threshold:
    #         distance_reward = 50.0
    #     elif dist_to_goal < 3*self.goal_threshold:
    #         distance_reward = 10.0
    #     else:
    #         distance_reward = 1.0 / (1.0 + dist_to_goal)

    #     # 计算起点到目标的向量
    #     start_to_goal = self.goal - self.start_ee_pos
    #     start_to_goal_norm = np.linalg.norm(start_to_goal)
    #     if start_to_goal_norm < 1e-6:  # 避免除以0（理论上不会发生，因目标与起点有距离约束）
    #         linearity_penalty = 0.0
    #     else:
    #         # 计算当前位置到起点的向量
    #         start_to_current = ee_pos - self.start_ee_pos
    #         # 计算当前位置在“起点→目标”直线上的投影比例（0~1之间表示在两点之间）
    #         projection_ratio = np.dot(start_to_current, start_to_goal) / (start_to_goal_norm **2)
    #         projection_ratio = np.clip(projection_ratio, 0.0, 1.0)  # 限制在0~1范围（超出目标点后不再惩罚）
    #         # 计算直线上的投影点
    #         projected_point = self.start_ee_pos + projection_ratio * start_to_goal
    #         # 计算当前位置与投影点的垂直距离（偏离直线的程度）
    #         linearity_error = np.linalg.norm(ee_pos - projected_point)
    #         # 直线性惩罚（距离越大，惩罚越重）
    #         linearity_penalty = 0.7 * linearity_error  # 权重可根据需要调整
                
    #     # 姿态约束：保持末端朝下
    #     target_orient = np.array([0, 0, -1])
    #     ee_orient_norm = ee_orient / np.linalg.norm(ee_orient)
    #     dot_product = np.dot(ee_orient_norm, target_orient)
    #     angle_error = np.arccos(np.clip(dot_product, -1.0, 1.0))
    #     orientation_penalty = 0.3 * angle_error
        
    #     # 动作相关惩罚
    #     action_diff = action - self.prev_action
    #     smooth_penalty = 0.1 * np.linalg.norm(action_diff)
    #     action_magnitude_penalty = 0.05 * np.linalg.norm(action)

    #     contact_reward = 1.0*self.data.ncon
        
    #     # 关节角度限制惩罚
    #     joint_penalty = 0.0
    #     for i in range(7):
    #         min_angle, max_angle = self.model.jnt_range[:7][i]
    #         if joint_angles[i] < min_angle:
    #             joint_penalty += 0.5 * (min_angle - joint_angles[i])
    #         elif joint_angles[i] > max_angle:
    #             joint_penalty += 0.5 * (joint_angles[i] - max_angle)
        
    #     # 时间惩罚
    #     time_penalty = 0.01
        
    #     # v1
    #     # total_reward = distance_reward - contact_reward - smooth_penalty - orientation_penalty       
    #     # v2
    #     # total_reward = distance_reward - contact_reward - smooth_penalty - orientation_penalty - linearity_penalty
    #     # v3
    #     total_reward = distance_reward - contact_reward - smooth_penalty - orientation_penalty - joint_penalty
    #     # print(f"[奖励] 距离目标: {distance_reward:.3f}, [碰撞]: {contact_reward:.3f}, 动作惩罚: {smooth_penalty:.3f}, 姿态: {orientation_penalty:.3f}  总奖励: {total_reward:.3f}")
        
    #     # 更新上一步动作
    #     self.prev_action = action.copy()
        
    #     return total_reward, dist_to_goal, angle_error

    def _calc_reward(self, ee_pos: np.ndarray, ee_orient: np.ndarray, joint_angles: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, float]:
        """
        计算单步奖励。奖励设计采用「稠密奖励 + 多项惩罚」结构，引导智能体：
            1. 快速接近目标（距离奖励）
            2. 沿起点→目标直线运动（直线奖励 + 远离惩罚）
            3. 保持末端朝下姿态（姿态惩罚）
            4. 动作平滑、不碰撞、不超关节限位

        参数：
            ee_pos: 末端位置（世界系）
            ee_orient: 末端姿态（欧拉角 xyz）
            joint_angles: 当前 7 维关节角度
            action: 当前动作（归一化 [-1,1]）

        返回：(总奖励, 到目标距离, 姿态角度误差)
        """
        # ===== 1. 距离奖励：越接近目标奖励越高，呈分段非线性增长 =====
        dist_to_goal = np.linalg.norm(ee_pos - self.goal)

        # 非线性距离奖励（保持不变）
        if dist_to_goal < self.goal_threshold:
            distance_reward = 100.0           # 到达目标：最高奖励
        elif dist_to_goal < 2*self.goal_threshold:
            distance_reward = 50.0            # 非常接近：高奖励
        elif dist_to_goal < 3*self.goal_threshold:
            distance_reward = 10.0            # 较接近：中等奖励
        else:
            distance_reward = 1.0 / (1.0 + dist_to_goal)   # 远离目标：稀疏的小奖励，引导靠近

        # ===== 2. 直线运动引导：鼓励末端沿「起点→目标」直线运动 =====
        # 计算起点到目标的向量及相关参数
        start_to_goal = self.goal - self.start_ee_pos
        start_to_goal_norm = np.linalg.norm(start_to_goal)
        linearity_reward = 0.0
        deviation_penalty = 0.0

        if start_to_goal_norm >= 1e-6:  # 起点和目标不重合时才计算直线相关奖励/惩罚
            # 计算当前位置到起点的向量
            start_to_current = ee_pos - self.start_ee_pos
            # 计算当前位置在“起点→目标”直线上的投影比例（限制在0~1，避免超出目标后惩罚）
            projection_ratio = np.dot(start_to_current, start_to_goal) / (start_to_goal_norm **2)
            projection_ratio = np.clip(projection_ratio, 0.0, 1.0)
            # 计算直线上的投影点，得到当前位置偏离直线的垂直距离
            projected_point = self.start_ee_pos + projection_ratio * start_to_goal
            linearity_error = np.linalg.norm(ee_pos - projected_point)  # 偏离直线的距离

            # 1. 直线接近奖励：离直线越近，奖励越高（非线性递增）
            linearity_reward = 3.0 / (1.0 + linearity_error)  # 系数8.0可根据重要性调整

            # 2. 远离趋势惩罚：检测“先靠近后远离”的行为
            # 初始化或更新历史最小偏离距离（跟踪最近点）
            if not hasattr(self, 'min_linearity_error'):
                self.min_linearity_error = np.inf  # 首次运行初始化
            if linearity_error < self.min_linearity_error:
                self.min_linearity_error = linearity_error  # 更近时更新最小值，无惩罚
            else:
                # 比最近点更远时，惩罚远离的程度（距离差越大，惩罚越重）
                deviation_penalty = 1.0 * (linearity_error - self.min_linearity_error)  # 系数3.0可调整

        # ===== 3. 姿态约束：保持末端朝下（target_orient = -Z 方向） =====
        target_orient = np.array([0, 0, -1])
        ee_orient_norm = ee_orient / np.linalg.norm(ee_orient)
        dot_product = np.dot(ee_orient_norm, target_orient)
        angle_error = np.arccos(np.clip(dot_product, -1.0, 1.0))   # 当前朝向与目标朝向的夹角
        orientation_penalty = 0.3 * angle_error

        # ===== 4. 动作平滑性惩罚：抑制动作抖动 =====
        action_diff = action - self.prev_action
        smooth_penalty = 0.1 * np.linalg.norm(action_diff)
        action_magnitude_penalty = 0.05 * np.linalg.norm(action)

        # ===== 5. 碰撞惩罚：ncon 为当前接触点数，越多惩罚越大 =====
        contact_reward = 1.0 * self.data.ncon

        # ===== 6. 关节角度限制惩罚：超出关节限位时施加惩罚 =====
        joint_penalty = 0.0
        for i in range(7):
            min_angle, max_angle = self.model.jnt_range[:7][i]
            if joint_angles[i] < min_angle:
                joint_penalty += 0.5 * (min_angle - joint_angles[i])
            elif joint_angles[i] > max_angle:
                joint_penalty += 0.5 * (joint_angles[i] - max_angle)

        # 时间惩罚（保持不变）
        time_penalty = 0.01

        # ===== 汇总总奖励：正向奖励 - 各项惩罚 =====
        total_reward = (distance_reward
                    + linearity_reward  # 新增：靠近直线的奖励
                    - contact_reward
                    - smooth_penalty
                    - orientation_penalty
                    - joint_penalty
                    - deviation_penalty)  # 新增：先近后远的惩罚

        # 更新上一步动作
        self.prev_action = action.copy()

        return total_reward, dist_to_goal, angle_error

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.float32, bool, bool, dict]:
        """
        执行一步环境交互。

        参数：
            action: 7 维归一化动作，取值 [-1, 1]

        返回：(观测, 奖励, 是否终止, 是否截断, 信息字典)
            Gym 新版 API（5 元组）：terminated 表示回合自然结束（成功/失败），
            truncated 表示因超时等外部原因截断。
        """
        # ===== 动作缩放：将归一化动作 [-1,1] 映射到各关节的实际角度限位内 =====
        joint_ranges = self.model.jnt_range[:7]
        scaled_action = np.zeros(7, dtype=np.float32)
        for i in range(7):
            # 映射公式：low + (a+1)/2 * (high-low)
            scaled_action[i] = joint_ranges[i][0] + (action[i] + 1) * 0.5 * (joint_ranges[i][1] - joint_ranges[i][0])

        # 执行动作：将目标关节角度写入控制器，并进行一步物理仿真积分
        self.data.ctrl[:7] = scaled_action
        mujoco.mj_step(self.model, self.data)

        # ===== 读取仿真后的状态，计算奖励 =====
        ee_pos = self.data.body(self.end_effector_id).xpos.copy()         # 末端位置
        ee_quat = self.data.body(self.end_effector_id).xquat.copy()       # 末端姿态四元数
        rot = R.from_quat(ee_quat)
        ee_quat_euler_rad = rot.as_euler('xyz')                           # 转为欧拉角，供姿态惩罚使用
        reward, dist_to_goal,_ = self._calc_reward(ee_pos, ee_quat_euler_rad, self.data.qpos[:7], action)
        terminated = False
        collision = False

        # 目标达成判定：末端到目标距离小于阈值则成功终止
        if dist_to_goal < self.goal_threshold:
            terminated = True
        # print(f"[奖励] 距离目标: {dist_to_goal:.3f}, 奖励: {reward:.3f}")

        # 超时判定：单回合超过 20 秒则强制终止并扣分
        if not terminated:
            if time.time() - self.start_t > 20.0:
                reward -= 10.0
                print(f"[超时] 时间过长，奖励减半")
                terminated = True

        # 可视化刷新：被动模式下需手动 sync 同步画面
        if self.visualize and self.handle is not None:
            self.handle.sync()
            time.sleep(0.01)

        obs = self._get_observation()
        info = {
            'is_success': terminated and (dist_to_goal < self.goal_threshold),
            'distance_to_goal': dist_to_goal,
            'collision': collision
        }

        return obs, reward.astype(np.float32), terminated, False, info

    def seed(self, seed: Optional[int] = None) -> list[Optional[int]]:
        """设置环境的随机种子，保证实验可复现"""
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def close(self) -> None:
        """关闭环境，释放可视化窗口等资源"""
        if self.visualize and self.handle is not None:
            self.handle.close()
            self.handle = None
        print("环境已关闭，资源释放完成")


def train_ppo(
    n_envs: int = 24,
    total_timesteps: int = 40_000_000,  # 本次训练的新增步数
    model_save_path: str = "panda_ppo_reach_target",
    visualize: bool = False,
    resume_from: Optional[str] = None
) -> None:
    """
    使用 PPO 算法训练机械臂到达目标点策略。

    参数：
        n_envs: 并行环境数（多进程采样，大幅提升数据采集速度）
        total_timesteps: 本次训练的总采样步数（增量式，支持从断点继续）
        model_save_path: 模型保存路径
        visualize: 是否在训练时可视化（仅首个子环境生效）
        resume_from: 若提供，则从该路径加载已有模型继续训练（断点恢复）
    """
    ENV_KWARGS = {'visualize': visualize}

    # 创建向量化环境：SubprocVecEnv 通过多进程并行运行多个独立环境
    # start_method="fork" 在 Linux 下开销较小，适合 MuJoCo 这种释放 GIL 的场景
    env = make_vec_env(
        env_id=lambda: PandaObstacleEnv(** ENV_KWARGS),
        n_envs=n_envs,
        seed=42,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork"}
    )

    if resume_from is not None:
        # 断点恢复：加载已有模型权重，继续在当前环境上训练
        model = PPO.load(resume_from, env=env)  # 加载时需传入当前环境
    else:
        # 策略网络结构配置：
        #   pi（actor 策略网络）：[256, 128] 两层全连接
        #   vf（critic 价值网络）：[256, 128] 两层全连接
        #   激活函数：ReLU
        POLICY_KWARGS = dict(
            activation_fn=nn.ReLU,
            net_arch=[dict(pi=[256, 128], vf=[256, 128])]
        )
        model = PPO(
            policy="MlpPolicy",          # 使用多层感知机策略（适用于低维状态输入）
            env=env,
            policy_kwargs=POLICY_KWARGS,
            verbose=1,
            n_steps=2048,                # 每个环境每次 rollout 的步数（缓冲区大小 = n_envs * n_steps）
            batch_size=2048,             # 每次更新的小批量大小
            n_epochs=10,                 # 每批数据重复训练的轮数
            gamma=0.99,                  # 折扣因子，衡量未来奖励的重要性
            learning_rate=2e-4,          # Adam 优化器学习率
            device="cuda" if torch.cuda.is_available() else "cpu",  # 自动选择 GPU/CPU
            tensorboard_log="./tensorboard/panda_reach_target/"     # TensorBoard 日志目录
        )

    print(f"并行环境数: {n_envs}, 本次训练新增步数: {total_timesteps}")
    # 开始训练：progress_bar 显示采样进度
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True
    )

    model.save(model_save_path)
    env.close()
    print(f"模型已保存至: {model_save_path}")


def test_ppo(
    model_path: str = "panda_ppo_reach_target",
    total_episodes: int = 5,
) -> None:
    """
    加载已训练的 PPO 模型进行可视化测试，统计成功率。

    参数：
        model_path: 模型文件路径
        total_episodes: 测试回合数
    """
    env = PandaObstacleEnv(visualize=True)
    model = PPO.load(model_path, env=env)

    record_gif = False
    frames = [] if record_gif else None
    render_scene = None
    render_context = None
    pixel_buffer = None
    viewport = None

    success_count = 0
    print(f"测试轮数: {total_episodes}")

    for ep in range(total_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            # deterministic=True：使用策略均值动作（不采样噪声），评估策略的确定性表现
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        if info['is_success']:
            success_count += 1
        print(f"轮次 {ep+1:2d} | 总奖励: {episode_reward:6.2f} | 结果: {'成功' if info['is_success'] else '碰撞/失败'}")

    success_rate = (success_count / total_episodes) * 100
    print(f"总成功率: {success_rate:.1f}%")

    env.close()


if __name__ == "__main__":
    # 程序入口：通过 TRAIN_MODE 切换训练/测试模式
    delete_flag_file()
    TRAIN_MODE = False  # 设为True开启训练模式
    MODEL_PATH = "assets/model/rl_reach_target_checkpoint/panda_ppo_reach_target_v3"
    RESUME_MODEL_PATH = "assets/model/rl_reach_target_checkpoint/panda_ppo_reach_target_v3"
    if TRAIN_MODE:
        train_ppo(
            n_envs=256,                # 训练时使用 256 个并行环境以加速采样
            total_timesteps=500_000_000,
            model_save_path=MODEL_PATH,
            visualize=True,
            resume_from=RESUME_MODEL_PATH
        )
    else:
        test_ppo(
            model_path=MODEL_PATH,
            total_episodes=15,
        )