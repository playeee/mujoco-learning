"""
Franka Emika Panda 机械臂强化学习——避障到达目标点任务（Obstacle Avoidance）

任务说明：
    控制 7 自由度 Panda 机械臂，在存在球形障碍物的工作空间中运动到目标点。
    相比 reach_target 任务，本任务增加了障碍物，智能体需要学会避开障碍物，
    同时保证末端到达目标点。

主要组件：
    1. PandaObstacleEnv: 自定义 Gym 环境，包含障碍物、碰撞检测与避障奖励
    2. train_ppo: PPO 训练入口，采用更深的网络结构与学习率衰减
    3. test_ppo: 加载已训练模型进行可视化测试

与 reach_target 的主要区别：
    - 场景中加入了球形障碍物，每回合随机化其 Y 坐标
    - 观测空间增加了障碍物位置（带噪声）与尺寸
    - 奖励中碰撞惩罚权重更大，碰撞即终止回合
    - 策略网络更深（[512,256,128]），学习率线性衰减
"""

import numpy as np
import mujoco                       # MuJoCo 物理引擎
import gym                          # OpenAI Gym，RL 环境接口
from gym import spaces              # 观测/动作空间定义
from stable_baselines3 import PPO   # PPO 算法实现
from stable_baselines3.common.env_util import make_vec_env       # 向量化环境创建工具
from stable_baselines3.common.vec_env import SubprocVecEnv       # 多进程并行环境
import torch.nn as nn
import warnings
import torch
import mujoco.viewer                # MuJoCo 可视化
import time
from typing import Optional
from scipy.spatial.transform import Rotation as R  # 四元数/欧拉角转换

# 忽略stable-baselines3的冗余UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.on_policy_algorithm")

import os

def write_flag_file(flag_filename="rl_visu_flag"):
    """创建标志文件，标记已有进程开启可视化"""
    flag_path = os.path.join("/tmp", flag_filename)
    try:
        with open(flag_path, "w") as f:
            f.write("This is a flag file")
        return True
    except Exception as e:
        return False

def check_flag_file(flag_filename="rl_visu_flag"):
    """检查标志文件是否存在"""
    flag_path = os.path.join("/tmp", flag_filename)
    return os.path.exists(flag_path)

def delete_flag_file(flag_filename="rl_visu_flag"):
    """删除标志文件，清理可视化状态"""
    flag_path = os.path.join("/tmp", flag_filename)
    if not os.path.exists(flag_path):
        return True
    try:
        os.remove(flag_path)
        return True
    except Exception as e:
        return False

# 标志文件机制：多进程训练时仅首个子环境开启可视化窗口，避免 GUI 冲突

class PandaObstacleEnv(gym.Env):
    """
    Panda 机械臂避障到达目标点强化学习环境。

    状态（观测）：7 维关节角度 + 3 维目标位置 + 3 维障碍物位置 + 1 维障碍物半径 = 14 维
    动作：7 维，归一化到 [-1, 1]，表示期望的 7 个关节角度
    奖励：距离奖励为主，碰撞惩罚较重（碰撞即终止）
    终止条件：到达目标（成功）、发生碰撞（失败）、超时
    """

    def __init__(self, visualize: bool = False):
        """初始化环境。参数 visualize 控制是否启用可视化（多进程下仅首个子环境生效）"""
        super(PandaObstacleEnv, self).__init__()
        # 标志文件机制：确保只有一个进程开启可视化
        if not check_flag_file():
            write_flag_file()
            self.visualize = visualize
        else:
            self.visualize = False
        self.handle = None

        # 加载含障碍物的场景模型
        self.model = mujoco.MjModel.from_xml_path('./model/franka_emika_panda/scene_pos_with_obstacles.xml')
        self.data = mujoco.MjData(self.model)
        # for i in range(self.model.ngeom):
        #     if self.model.geom_group[i] == 3:
        #         self.model.geom_conaffinity[i] = 0

        if self.visualize:
            self.handle = mujoco.viewer.launch_passive(self.model, self.data)
            self.handle.cam.distance = 3.0
            self.handle.cam.azimuth = 0.0
            self.handle.cam.elevation = -30.0
            self.handle.cam.lookat = np.array([0.2, 0.0, 0.4])

        self.np_random = np.random.default_rng(None)

        # 获取末端执行器 body id 与 home 关节位姿（从场景 keyframe 读取）
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ee_center_body')
        self.home_joint_pos = np.array(self.model.key_qpos[0][:7], dtype=np.float32)

        # 动作空间：7 维归一化关节角度
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        # 观测空间：7(关节角) + 3(目标) + 3(障碍物位置) + 1(障碍物半径) = 14 维
        self.obs_size = 7 + 3 + 3 + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32)

        # 目标点位置（Y 坐标在 reset 时随机化）
        # self.goal_position = np.array([0.4, 0.3, 0.4], dtype=np.float32)
        self.goal_position = np.array([0.4, -0.3, 0.4], dtype=np.float32)
        self.goal_arrival_threshold = 0.005   # 到达目标的判定阈值
        self.goal_visu_size = 0.02            # 可视化目标球半径
        self.goal_visu_rgba = [0.1, 0.3, 0.3, 0.8]

        # 在xml中增加障碍物，worldbody 中添加如下
        # <geom name="obstacle_0"
        #     type="sphere"
        #     size="0.060"
        #     pos="0.300 0.200 0.500"
        #     contype="1"
        #     conaffinity="1"
        #     mass="0.0"
        #     rgba="0.300 0.300 0.300 0.800"
        # />
        # 并在init函数中初始化障碍物的位置和大小
        # 通过名称查找障碍物几何体，记录其 id、位置与半径
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name == "obstacle_0":
                self.obstacle_id_1 = i
        self.obstacle_position = np.array(self.model.geom_pos[self.obstacle_id_1], dtype=np.float32)
        self.obstacle_size = self.model.geom_size[self.obstacle_id_1][0]

        self.last_action = self.home_joint_pos   # 用于动作平滑性惩罚

    def _render_scene(self) -> None:
        """在可视化场景中渲染目标点（球体）"""
        if not self.visualize or self.handle is None:
            return
        self.handle.user_scn.ngeom = 0
        total_geoms = 1
        self.handle.user_scn.ngeom = total_geoms

        mujoco.mjv_initGeom(
            self.handle.user_scn.geoms[0],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[self.goal_visu_size, 0.0, 0.0],
            pos=self.goal_position,
            mat=np.eye(3).flatten(),
            rgba=np.array(self.goal_visu_rgba, dtype=np.float32)
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        """
        重置环境到新回合的初始状态。

        每回合随机化：
            - 目标点的 Y 坐标（在 [-0.3, 0.3] 范围内）
            - 障碍物的 Y 坐标（在 [-0.3, 0.3] 范围内）
        这样保证任务多样性，训练出能泛化到不同障碍物配置的策略。

        返回：(观测, 信息字典)
        """
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # 重置关节到home位姿
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = self.home_joint_pos
        self.data.qpos[7:] = [0.04,0.04]      # 夹爪初始开合度
        mujoco.mj_forward(self.model, self.data)

        # 随机化目标点与障碍物的 Y 坐标，增加任务多样性
        self.goal_position = np.array([self.goal_position[0], self.np_random.uniform(-0.3, 0.3), self.goal_position[2]], dtype=np.float32)
        self.obstacle_position = np.array([self.obstacle_position[0], self.np_random.uniform(-0.3, 0.3), self.obstacle_position[2]], dtype=np.float32)
        # 将新的障碍物位置写回模型（MuJoCo 中几何体位置由 model.geom_pos 存储）
        self.model.geom_pos[self.obstacle_id_1] = self.obstacle_position
        mujoco.mj_step(self.model, self.data)

        if self.visualize:
            self._render_scene()

        obs = self._get_observation()
        self.start_t = time.time()            # 记录回合开始时间，用于超时与时间惩罚
        return obs, {}

    def _get_observation(self) -> np.ndarray:
        """
        构造观测向量：[7 关节角, 3 目标位置, 3 障碍物位置(带噪声), 1 障碍物半径] = 14 维

        注意：障碍物位置叠加了高斯噪声（std=0.001），模拟真实感知的不确定性，
        提升策略的鲁棒性，避免对精确障碍物位置的过拟合。
        """
        joint_pos = self.data.qpos[:7].copy().astype(np.float32)
        return np.concatenate([joint_pos, self.goal_position, self.obstacle_position + np.random.normal(0, 0.001, size=3), np.array([self.obstacle_size], dtype=np.float32)])

    def _calc_reward(self, joint_angles: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, float]:
        """
        计算单步奖励。

        奖励组成：
            + 距离奖励（分段非线性，越近越高）
            - 碰撞惩罚（权重 10 * 接触点数，碰撞代价高）
            - 动作平滑惩罚
            - 关节限位惩罚
            - 时间惩罚（鼓励尽快完成任务）

        返回：(总奖励, 到目标距离, 是否发生碰撞)
        """
        now_ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        dist_to_goal = np.linalg.norm(now_ee_pos - self.goal_position)

        # 非线性距离奖励：分段设计，越接近目标奖励越高且增长越快
        if dist_to_goal < self.goal_arrival_threshold:
            distance_reward = 20.0*(1.0+(1.0-(dist_to_goal / self.goal_arrival_threshold)))
        elif dist_to_goal < 2*self.goal_arrival_threshold:
            distance_reward = 10.0*(1.0+(1.0-(dist_to_goal / 2*self.goal_arrival_threshold)))
        elif dist_to_goal < 3*self.goal_arrival_threshold:
            distance_reward = 5.0*(1.0+(1.0-(dist_to_goal / 3*self.goal_arrival_threshold)))
        else:
            distance_reward = 1.0 / (1.0 + dist_to_goal)

        # 平滑惩罚：抑制动作抖动
        smooth_penalty = 0.001 * np.linalg.norm(action - self.last_action)

        # 碰撞惩罚：ncon 为接触点数量，权重较大，强力抑制碰撞
        contact_reward = 10.0 * self.data.ncon

        # 关节角度限制惩罚：超出关节限位时施加惩罚
        joint_penalty = 0.0
        for i in range(7):
            min_angle, max_angle = self.model.jnt_range[:7][i]
            if joint_angles[i] < min_angle:
                joint_penalty += 0.5 * (min_angle - joint_angles[i])
            elif joint_angles[i] > max_angle:
                joint_penalty += 0.5 * (joint_angles[i] - max_angle)

        # 时间惩罚：随时间增长，鼓励高效完成任务
        time_penalty = 0.001 * (time.time() - self.start_t)

        total_reward = (distance_reward
                    - contact_reward
                    - smooth_penalty
                    - joint_penalty
                    - time_penalty)

        self.last_action = action.copy()

        # 返回是否碰撞：接触点数 > 0 视为发生碰撞
        return total_reward, dist_to_goal, self.data.ncon > 0

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.float32, bool, bool, dict]:
        """
        执行一步环境交互。

        返回：(观测, 奖励, 是否终止, 是否截断, 信息字典)

        终止条件：
            - 发生碰撞（失败，额外扣 10 分）
            - 到达目标（成功）
            - 超时（超过 20 秒）
        """
        # 动作缩放：归一化 [-1,1] -> 关节限位
        joint_ranges = self.model.jnt_range[:7]
        scaled_action = np.zeros(7, dtype=np.float32)
        for i in range(7):
            scaled_action[i] = joint_ranges[i][0] + (action[i] + 1) * 0.5 * (joint_ranges[i][1] - joint_ranges[i][0])

        # 写入控制器并执行一步物理仿真
        self.data.ctrl[:7] = scaled_action
        self.data.qpos[7:] = [0.04,0.04]       # 固定夹爪开合度（本任务不使用夹爪）
        mujoco.mj_step(self.model, self.data)

        reward, dist_to_goal, collision = self._calc_reward(self.data.qpos[:7], action)
        terminated = False

        # 碰撞处理：立即终止并额外扣分
        if collision:
            # print("collision happened, ", self.data.ncon)
            # info = {}
            # for i in range(self.data.ncon):
            #     contact = self.data.contact[i]
            #     # 获取几何体对应的body_id
            #     body1_id = self.model.geom_bodyid[contact.geom1]
            #     body2_id = self.model.geom_bodyid[contact.geom2]
            #     # 通过mj_id2name转换body_id为名称
            #     body1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
            #     body2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
            #     info["pair"+str(i)] = {}
            #     info["pair"+str(i)]["geom1"] = contact.geom1
            #     info["pair"+str(i)]["geom2"] = contact.geom2
            #     info["pair"+str(i)]["pos"] = contact.pos.copy()
            #     info["pair"+str(i)]["body1_name"] = body1_name
            #     info["pair"+str(i)]["body2_name"] = body2_name
            # print(info)
            reward -= 10.0
            terminated = True

        # 到达目标：成功终止
        if dist_to_goal < self.goal_arrival_threshold:
            terminated = True
            print(f"[成功] 距离目标: {dist_to_goal:.3f}, 奖励: {reward:.3f}")
        # else:
        #     print(f"[失败] 距离目标: {dist_to_goal:.3f}, 奖励: {reward:.3f}")

        # 超时判定
        if not terminated:
            if time.time() - self.start_t > 20.0:
                reward -= 10.0
                print(f"[超时] 时间过长，奖励减半")
                terminated = True

        # 可视化刷新
        if self.visualize and self.handle is not None:
            self.handle.sync()
            time.sleep(0.01)

        obs = self._get_observation()
        info = {
            'is_success': not collision and terminated and (dist_to_goal < self.goal_arrival_threshold),
            'distance_to_goal': dist_to_goal,
            'collision': collision
        }

        return obs, reward.astype(np.float32), terminated, False, info

    def seed(self, seed: Optional[int] = None) -> list[Optional[int]]:
        """设置随机种子，保证实验可复现"""
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def close(self) -> None:
        """关闭环境，释放可视化窗口资源"""
        if self.visualize and self.handle is not None:
            self.handle.close()
            self.handle = None

def train_ppo(
    n_envs: int = 24,
    total_timesteps: int = 80_000_000,
    model_save_path: str = "panda_ppo_reach_target",
    visualize: bool = False,
    resume_from: Optional[str] = None
) -> None:
    """
    使用 PPO 算法训练避障策略。

    参数：
        n_envs: 并行环境数
        total_timesteps: 总采样步数
        model_save_path: 模型保存路径
        visualize: 是否启用可视化
        resume_from: 断点恢复的模型路径
    """
    ENV_KWARGS = {'visualize': visualize}

    # 创建多进程向量化环境
    env = make_vec_env(
        env_id=lambda: PandaObstacleEnv(** ENV_KWARGS),
        n_envs=n_envs,
        seed=42,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork"}
    )

    if resume_from is not None:
        # 断点恢复
        model = PPO.load(resume_from, env=env)
    else:
        # POLICY_KWARGS = dict(
        #     activation_fn=nn.ReLU,
        #     net_arch=[dict(pi=[256, 128], vf=[256, 128])]
        # )


        # 策略网络：3 层全连接 [512, 256, 128]，LeakyReLU 激活
        # 相比 reach_target 任务，避障任务状态更复杂（含障碍物），故网络更深
        POLICY_KWARGS = dict(
            activation_fn=nn.LeakyReLU,
            net_arch=[
                dict(
                    pi=[512, 256, 128],
                    vf=[512, 256, 128]
                )
            ]
        )

        model = PPO(
            policy="MlpPolicy",
            env=env,
            policy_kwargs=POLICY_KWARGS,
            verbose=1,
            n_steps=2048,          # 每个环境 rollout 步数
            batch_size=2048,       # 小批量大小
            n_epochs=10,           # 每批数据训练轮数
            gamma=0.99,            # 折扣因子
            # ent_coef=0.02,  # 增加熵系数，保留后期探索以提升泛化性
            ent_coef = 0.001,      # 熵正则系数，鼓励探索（值越小越倾向确定性策略）
            clip_range=0.15,       # PPO 裁剪范围，限制策略更新幅度，保证训练稳定
            max_grad_norm=0.5,     # 梯度裁剪阈值，防止梯度爆炸
            learning_rate=lambda f: 1e-4 * (1 - f),  # 学习率线性衰减：f 从 0->1，lr 从 1e-4 -> 0
            device="cuda" if torch.cuda.is_available() else "cpu",
            tensorboard_log="./tensorboard/panda_obstacle_avoidance/"
        )

    print(f"并行环境数: {n_envs}, 本次训练新增步数: {total_timesteps}")
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True
    )

    model.save(model_save_path)
    env.close()
    print(f"模型已保存至: {model_save_path}")


def test_ppo(
    model_path: str = "panda_obstacle_avoidance",
    total_episodes: int = 5,
) -> None:
    """加载已训练模型进行可视化测试，统计成功率"""
    env = PandaObstacleEnv(visualize=True)
    model = PPO.load(model_path, env=env)


    success_count = 0
    print(f"测试轮数: {total_episodes}")

    for ep in range(total_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            # 测试时每步重新获取观测（避免并行环境观测滞后问题）
            obs = env._get_observation()
            # print(f"观察: {obs}")
            action, _states = model.predict(obs, deterministic=True)
            # action += np.random.normal(0, 0.002, size=7)  # 加入噪声
            obs, reward, terminated, truncated, info = env.step(action)
            # print(f"动作: {action}, 奖励: {reward}, 终止: {terminated}, 截断: {truncated}, 信息: {info}")
            episode_reward += reward
            done = terminated or truncated

        if info['is_success']:
            success_count += 1
        print(f"轮次 {ep+1:2d} | 总奖励: {episode_reward:6.2f} | 结果: {'成功' if info['is_success'] else '碰撞/失败'}")

    success_rate = (success_count / total_episodes) * 100
    print(f"总成功率: {success_rate:.1f}%")

    env.close()


if __name__ == "__main__":
    # 程序入口：TRAIN_MODE 切换训练/测试
    TRAIN_MODE = True  # 设为True开启训练模式
    if TRAIN_MODE:
        import os
        os.system("rm -rf /home/dar/mujoco-bin/mujoco-learning/tensorboard*")   # 训练前清理旧日志
    delete_flag_file()
    MODEL_PATH = "assets/model/rl_obstacle_avoidance_checkpoint/panda_obstacle_avoidance_v3"
    RESUME_MODEL_PATH = "assets/model/rl_obstacle_avoidance_checkpoint/panda_obstacle_avoidance_v2"
    if TRAIN_MODE:
        train_ppo(
            n_envs=64,                # 训练时 64 个并行环境
            total_timesteps=60_000_000,
            model_save_path=MODEL_PATH,
            visualize=True,
            # resume_from=RESUME_MODEL_PATH
            resume_from=None          # 从头训练；若需断点恢复则改为 RESUME_MODEL_PATH
        )
    else:
        test_ppo(
            model_path=MODEL_PATH,
            total_episodes=100,
        )
    os.system("date")