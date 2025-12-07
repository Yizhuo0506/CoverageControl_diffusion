# This file is part of the CoverageControl library
#
# Author: Saurav Agarwal
# Contact: sauravag@seas.upenn.edu, agr.saurav1@gmail.com
# Repository: https://github.com/KumarRobotics/CoverageControl
#
# Copyright (c) 2024, Saurav Agarwal
#
# The CoverageControl library is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# The CoverageControl library is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# CoverageControl library. If not, see <https://www.gnu.org/licenses/>.
# @file controller.py
#  @brief Base classes for CVT and neural network based controllers
import coverage_control.nn as cc_nn
import torch
torch.set_float32_matmul_precision('high')

from . import CentralizedCVT
from . import ClairvoyantCVT
from . import DecentralizedCVT
from . import NearOptimalCVT
from .. import IOUtils
from .. import CoverageEnvUtils
from ..core import CoverageSystem
from ..core import Parameters
from ..core import PointVector
from coverage_control.nn.models.diffusion_policy import DiffusionPolicy
import torch.nn.functional as F


__all__ = ["ControllerCVT", "ControllerNN"]

import numpy as np  # 新增

def build_diffusion_schedule(
    num_steps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device: torch.device | str | None = None,
):
    """
    创建与 train_diffusion.py 中一致的线性 beta 调度。
    只用到 betas, alphas, alpha_cumprod 就够了。
    """
    if device is None:
        device = "cpu"

    betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)

    return {
        "num_steps": num_steps,
        "betas": betas,
        "alphas": alphas,
        "alpha_cumprod": alpha_cumprod,
    }

class ControllerCVT:
    """
    Controller class for CVT based controllers
    """

    def __init__(self, config: dict, params: Parameters, env: CoverageSystem):
        """
        Constructor for the CVT controller
        Args:
            config: Configuration dictionary
            params: Parameters object
            env: CoverageSystem object
        """
        self.name = config["Name"]
        self.params = params
        match config["Algorithm"]:
            case "DecentralizedCVT":
                self.alg = DecentralizedCVT(params, env)
            case "ClairvoyantCVT":
                self.alg = ClairvoyantCVT(params, env)
            case "CentralizedCVT":
                self.alg = CentralizedCVT(params, env)
            case "NearOptimalCVT":
                self.alg = NearOptimalCVT(params, env)
            case _:
                raise ValueError(f"Unknown controller type: {controller_type}")

    def step(self, env: CoverageSystem) -> (float, bool):
        """
        Step function for the CVT controller

        Performs three steps:
        1. Compute actions using the CVT algorithm
        2. Get the actions from the algorithm
        3. Step the environment using the actions
        Args:
            env: CoverageSystem object
        Returns:
            Objective value and convergence flag
        """
        self.alg.ComputeActions()
        actions = self.alg.GetActions()
        converged = self.alg.IsConverged()
        error_flag = env.StepActions(actions)

        if error_flag:
            raise ValueError("Error in step")

        return env.GetObjectiveValue(), converged


class ControllerNN:
    """
    Controller class for neural network based controllers
    """

    def __init__(self, config: dict, params: Parameters, env: CoverageSystem):
        """
        Constructor for the neural network controller
        Args:
            config: Configuration dictionary
            params: Parameters object
            env: CoverageSystem object
        """
        self.config = config
        self.params = params
        self.name = self.config["Name"]
        # PolicyType: "LPAC"（默认）或 "DIFFUSION"
        self.policy_type = str(self.config.get("PolicyType", "LPAC")).upper()

        self.use_cnn = self.config["UseCNN"]
        self.use_comm_map = self.config["UseCommMap"]
        self.cnn_map_size = self.config["CNNMapSize"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 归一化统计量（LPAC / Diffusion 都可能会用到）
        self.actions_mean: torch.Tensor | None = None
        self.actions_std: torch.Tensor | None = None

        # 扩散调度结构，只在 DIFFUSION 模式下用
        self.diffusion: dict | None = None

        # --------------------------------------------------------------
        # 1) 如果配置里给的是 ModelFile（原作者 JIT 导出的那种）
        # --------------------------------------------------------------
        if "ModelFile" in self.config:
            self.model_file = IOUtils.sanitize_path(self.config["ModelFile"])
            # 作者这里用的是 torch.load（而不是 torch.jit.load），我们保持行为一致
            self.model = torch.load(self.model_file, map_location=self.device)
        else:
            # ----------------------------------------------------------
            # 2) 否则，用 LearningParams + ModelStateDict 的方式加载
            # ----------------------------------------------------------
            self.learning_params_file = IOUtils.sanitize_path(self.config["LearningParams"])
            self.learning_params = IOUtils.load_toml(self.learning_params_file)

            if self.policy_type == "LPAC":
                # 原始 LPAC GNN policy，保持不变
                self.model = cc_nn.LPAC(self.learning_params).to(self.device)
                # LPAC 内部的 load_model 会自己处理 state_dict 和 actions_mean/std
                self.model.load_model(IOUtils.sanitize_path(self.config["ModelStateDict"]))
                self.actions_mean = self.model.actions_mean.to(self.device)
                self.actions_std = self.model.actions_std.to(self.device)

            elif self.policy_type == "DIFFUSION":
                # 我们训练好的 DiffusionPolicy
                ckpt_path = IOUtils.sanitize_path(self.config["ModelStateDict"])
                ckpt = torch.load(ckpt_path, map_location=self.device)

                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    raw_state_dict = ckpt["model_state_dict"]
                else:
                    raw_state_dict = ckpt

                # 从 state_dict 中把 actions_mean / actions_std 抠出来，
                # 并且从 state_dict 中删除，避免 load_state_dict 报 Unexpected key
                am = raw_state_dict.pop("actions_mean", None)
                as_ = raw_state_dict.pop("actions_std", None)
                if am is not None and as_ is not None:
                    self.actions_mean = am.to(self.device)
                    self.actions_std = as_.to(self.device)

                self.model = DiffusionPolicy(self.learning_params).to(self.device)
                missing, unexpected = self.model.load_state_dict(raw_state_dict, strict=False)
                if unexpected:
                    print(f"[DiffusionPolicy] Ignoring unexpected keys: {unexpected}")
                if missing:
                    print(f"[DiffusionPolicy] Missing keys (usually fine): {missing}")

                # 构造扩散调度（与 train_diffusion.py 一致）
                diff_train_cfg = self.learning_params.get("DiffusionTraining", {})
                num_steps = int(diff_train_cfg.get("NumDiffusionSteps", 1000))
                beta_start = float(diff_train_cfg.get("BetaStart", 1e-4))
                beta_end = float(diff_train_cfg.get("BetaEnd", 2e-2))
                self.diffusion = build_diffusion_schedule(
                    num_steps=num_steps,
                    beta_start=beta_start,
                    beta_end=beta_end,
                    device=self.device,
                )

            else:
                raise ValueError(f"[ControllerNN] Unknown PolicyType: {self.policy_type}")

        # 模型统一放到 device 上并编译
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model = torch.compile(self.model, dynamic=True)

    # ------------------------------------------------------------------
    # ControllerNN.step: 根据 policy_type 分别处理 LPAC / Diffusion
    # ------------------------------------------------------------------
    def step(self, env: CoverageSystem) -> (float, bool):
        """
        Step function for the neural network controller

        Performs three steps:
        1. Get the data from the environment
        2. Get the actions from the model
        3. Step the environment using the actions
        Args:
            env: CoverageSystem object
        Returns:
            Objective value and convergence flag
        """
        if self.policy_type == "LPAC":
            # 原始 LPAC 路径：用 CoverageEnvUtils 构造 torch_geometric 数据
            pyg_data = CoverageEnvUtils.get_torch_geometric_data(
                env,
                self.params,
                True,                 # use_cnn
                self.use_comm_map,
                self.cnn_map_size,
            ).to(self.device)

            with torch.no_grad():
                actions = self.model(pyg_data)  # (N, 2)

            if self.actions_mean is not None and self.actions_std is not None:
                actions = actions * self.actions_std + self.actions_mean

        elif self.policy_type == "DIFFUSION":
            # Diffusion policy：用当前 env 状态跑一条完整的反向扩散链
            with torch.no_grad():
                actions = self._diffusion_step(env)  # (N, 2)

        else:
            raise ValueError(f"[ControllerNN] Unknown PolicyType: {self.policy_type}")

        # 把 (N, 2) 的动作转成 C++ 所需的 PointVector
        actions_np = actions.detach().cpu().numpy()
        point_vector_actions = PointVector(actions_np)
        error_flag = env.StepActions(point_vector_actions)
        if error_flag:
            raise ValueError("Error in env.StepActions(actions)")

        objective_value = env.GetObjectiveValue()
        # 简单收敛判定：所有动作接近 0
        converged = torch.allclose(actions, torch.zeros_like(actions), atol=1e-5)
        return objective_value, converged

    # ------------------------------------------------------------------
    # 私有函数：在当前 env 状态下跑一条扩散采样链，得到 a_0
    # ------------------------------------------------------------------
    def _diffusion_step(self, env: CoverageSystem) -> torch.Tensor:
        """
        使用当前环境状态，跑一次完整的反向扩散链，输出当前时刻的动作 a_0。
        返回形状: (num_robots, 2)
        """
        assert self.diffusion is not None, "[ControllerNN] Diffusion schedule not initialized."

        num_robots = env.GetNumRobots()

        local_maps_list = []
        obstacle_maps_list = []
        positions_list = []

        # 1. 从 env 读取局部地图和机器人位置
        for rid in range(num_robots):
            # C++ 端 API: GetRobotLocalMap(self, rid) -> np.ndarray[float32[m, n]]
            local_map_np = env.GetRobotLocalMap(rid)
            obstacle_map_np = env.GetRobotObstacleMap(rid)

            local_maps_list.append(torch.from_numpy(local_map_np).float())
            obstacle_maps_list.append(torch.from_numpy(obstacle_map_np).float())

            pos_np = np.asarray(env.GetRobotPosition(rid), dtype=np.float32)  # (2,)
            positions_list.append(torch.from_numpy(pos_np))

        local_maps = torch.stack(local_maps_list, dim=0)       # (N, H_raw, W_raw)
        obstacle_maps = torch.stack(obstacle_maps_list, dim=0) # (N, H_raw, W_raw)
        positions = torch.stack(positions_list, dim=0)         # (N, 2)

        # ---- 1.5 确保和训练时 CNN 输入尺寸一致：H = W = self.cnn_map_size ----
        target_size = getattr(self, "cnn_map_size", 32)  # 默认 32，对应 learning_params 里的 CNNMapSize
        if local_maps.dim() != 3:
            raise RuntimeError(
                f"[ControllerNN._diffusion_step] unexpected local_maps.shape={local_maps.shape}"
            )

        N, H_raw, W_raw = local_maps.shape

        if H_raw != target_size or W_raw != target_size:
            # (N, 1, H_raw, W_raw) -> 插值 -> (N, 1, target_size, target_size)
            local_maps = F.interpolate(
                local_maps.unsqueeze(1),
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            obstacle_maps = F.interpolate(
                obstacle_maps.unsqueeze(1),
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            H, W = target_size, target_size
        else:
            H, W = H_raw, W_raw


        # 2. 构造 DiffusionPolicy 所需的 coverage_maps: (B=1, N, C=4, H, W)
        coverage_maps = torch.zeros(
            (1, N, 4, H, W), dtype=torch.float32, device=self.device
        )
        coverage_maps[0, :, 0, :, :] = local_maps.to(self.device)
        coverage_maps[0, :, 1, :, :] = obstacle_maps.to(self.device)

        # 通信图通道（第 3、4 个）目前暂时置 0；
        # 若要严格对齐训练，可以后续根据 env.GetCommunicationMaps() 构造。
        # comm_maps = env.GetCommunicationMaps()  # 若需要，可在这里用

        positions = positions.unsqueeze(0).to(self.device)  # (1, N, 2)

        # 3. 反向扩散采样：从 x_T ~ N(0, I) 迭代到 x_0
        diff = self.diffusion
        num_steps = diff["num_steps"]
        betas = diff["betas"]
        alphas = diff["alphas"]
        alpha_cumprod = diff["alpha_cumprod"]

        B = 1
        actions = torch.randn((B, N, 2), device=self.device)  # x_T

        for step_t in reversed(range(num_steps)):
            t = torch.full((B,), step_t, device=self.device, dtype=torch.long)

            # epsilon_hat = eps_theta(x_t, condition, t)
            eps_hat = self.model(coverage_maps, actions, t, positions)  # (B, N, 2)

            beta_t = betas[step_t]
            alpha_t = alphas[step_t]
            alpha_bar_t = alpha_cumprod[step_t]
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = (1.0 - alpha_t) / sqrt_one_minus_alpha_bar_t

            mean = coef1 * (actions - coef2 * eps_hat)

            if step_t > 0:
                noise = torch.randn_like(actions)
                sigma_t = torch.sqrt(beta_t)
                actions = mean + sigma_t * noise
            else:
                actions = mean  # t == 0

        # 4. 反归一化
        if self.actions_mean is not None and self.actions_std is not None:
            actions = actions * self.actions_std + self.actions_mean

        # 返回 (N, 2)
        return actions.squeeze(0)
