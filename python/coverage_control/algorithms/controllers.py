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

class TokenBuffer:
    """Token cache for decentralized execution.

    each robot shares perceptual token z_i with peers in
    communication range; receivers cache the latest embeddings.
    """

    def __init__(self, num_robots: int, device: torch.device, max_age_steps: int = 0):
        self.num_robots = int(num_robots)
        self.device = device
        self.max_age_steps = int(max_age_steps)

        self._z = [None] * self.num_robots
        self._x = [None] * self.num_robots
        self._t = [-1] * self.num_robots

    def write(self, robot_id: int, z: torch.Tensor, x: torch.Tensor, step: int) -> None:
        rid = int(robot_id)
        self._z[rid] = z.detach()
        self._x[rid] = x.detach()
        self._t[rid] = int(step)

    def write_all(self, z_all: torch.Tensor, x_all: torch.Tensor, step: int) -> None:
        for rid in range(self.num_robots):
            self.write(rid, z_all[rid], x_all[rid], step)

    def is_fresh(self, robot_id: int, step: int) -> bool:
        last = self._t[int(robot_id)]
        if last < 0:
            return False
        if self.max_age_steps <= 0:
            return last == int(step)
        return last >= int(step) - self.max_age_steps

    def get_z(self, robot_id: int) -> torch.Tensor:
        return self._z[int(robot_id)]

    def get_x(self, robot_id: int) -> torch.Tensor:
        return self._x[int(robot_id)]

torch.set_float32_matmul_precision('high')
import pathlib
from . import CentralizedCVT
from . import ClairvoyantCVT
from . import DecentralizedCVT
from . import NearOptimalCVT
from .. import IOUtils
from .. import CoverageEnvUtils
from ..core import CoverageSystem
from ..core import Parameters
from ..core import PointVector
from coverage_control.nn.models.diffusion_policy import DiffusionPolicy, GaussianDiffusion
import torch.nn.functional as F


__all__ = ["ControllerCVT", "ControllerNN"]

import numpy as np  

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
        # PolicyType: "LPAC" / "DIFFUSION"
        self.policy_type = str(self.config.get("PolicyType", "LPAC")).upper()

        self.use_cnn = self.config["UseCNN"]
        self.use_comm_map = self.config["UseCommMap"]
        self.cnn_map_size = self.config["CNNMapSize"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        self.actions_mean: torch.Tensor | None = None
        self.actions_std: torch.Tensor | None = None

        self.diffusion: dict | None = None

        if "ModelFile" in self.config:
            self.model_file = IOUtils.sanitize_path(self.config["ModelFile"])
            self.model = torch.load(self.model_file, map_location=self.device)
        else:
            self.learning_params_file = IOUtils.sanitize_path(self.config["LearningParams"])
            self.learning_params = IOUtils.load_toml(self.learning_params_file)

            if self.policy_type == "LPAC":
                self.model = cc_nn.LPAC(self.learning_params).to(self.device)
                self.model.load_model(IOUtils.sanitize_path(self.config["ModelStateDict"]))
                self.actions_mean = self.model.actions_mean.to(self.device)
                self.actions_std = self.model.actions_std.to(self.device)

            elif self.policy_type == "DIFFUSION":
                # Load diffusion-policy checkpoint
                ckpt_path = IOUtils.sanitize_path(self.config["ModelStateDict"])
                ckpt = torch.load(ckpt_path, map_location=self.device)

                # 1) Extract model state dict
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    raw_state_dict = ckpt["model_state_dict"]
                else:
                    # Backward compatibility: checkpoint is a raw state_dict
                    raw_state_dict = ckpt

                # 2) Extract action normalization statistics if present
                #    and remove them from the state_dict to avoid unexpected keys.
                am = raw_state_dict.pop("actions_mean", None)
                as_ = raw_state_dict.pop("actions_std", None)
                if am is not None and as_ is not None:
                    self.actions_mean = am.to(self.device).float().view(-1)  # (2,)
                    self.actions_std  = as_.to(self.device).float().view(-1) # (2,)

                # 3) Build DiffusionPolicy model from learning_params
                self.model = DiffusionPolicy(self.learning_params).to(self.device)
                missing, unexpected = self.model.load_state_dict(raw_state_dict, strict=False)
                if unexpected:
                    print(f"[DiffusionPolicy] Ignoring unexpected keys: {unexpected}")
                if missing:
                    print(f"[DiffusionPolicy] Missing keys (usually fine): {missing}")

                # 4) Rebuild GaussianDiffusion from checkpoint if available
                diff_state = None
                if isinstance(ckpt, dict) and "diffusion" in ckpt:
                    diff_state = ckpt["diffusion"]

                if diff_state is not None:
                    self.diffusion = GaussianDiffusion.from_state_dict(
                        diff_state, device=self.device
                    )
                else:
                    diff_train_cfg = self.learning_params.get("DiffusionTraining", {})
                    num_steps = int(diff_train_cfg.get("NumDiffusionSteps", 1000))
                    beta_start = float(diff_train_cfg.get("BetaStart", 1e-4))
                    beta_end = float(diff_train_cfg.get("BetaEnd", 2e-2))
                    self.diffusion = GaussianDiffusion(
                        num_steps=num_steps,
                        beta_start=beta_start,
                        beta_end=beta_end,
                        device=self.device,
                    )

                # 5) Store sampling configuration (DDIM steps / eta)
                self.sampling_cfg = self.learning_params.get("DiffusionSampling", {})

                # 6) Paper-aligned decentralized execution parameters
                #    - communication range is matched to the attention radius (default 256 m)
                #    - CommInterval=1 and TokenTTL=0 correspond to ideal wireless (no delay/loss)
                self._step = 0
                self.comm_interval = int(self.sampling_cfg.get("CommInterval", 1))
                self.token_ttl = int(self.sampling_cfg.get("TokenTTL", 0))
                self.comm_radius = float(self.sampling_cfg.get("CommRadius", getattr(self.model, "attn_radius", 256.0) or 256.0))
                self._token_buffer = None

            else:
                raise ValueError(f"[ControllerNN] Unknown PolicyType: {self.policy_type}")
            
            # --- load action normalization stats from dataset directory (delete if re-train) ---
            if self.policy_type == "DIFFUSION" and hasattr(self, "learning_params"):
                if self.actions_mean is None or self.actions_std is None:
                    dataset_root = pathlib.Path(IOUtils.sanitize_path(self.learning_params["DataDir"]))
                    data_dir = (dataset_root / "data") if (dataset_root / "data").exists() else dataset_root

                    mean_path = data_dir / "actions_mean.pt"
                    std_path  = data_dir / "actions_std.pt"

                    if mean_path.exists() and std_path.exists():
                        am = torch.load(mean_path, map_location="cpu").float().view(-1)  # (2,)
                        as_ = torch.load(std_path, map_location="cpu").float().view(-1)  # (2,)
                        self.actions_mean = am.to(self.device)
                        self.actions_std  = as_.to(self.device)

                        with torch.no_grad():
                            if hasattr(self.model, "actions_mean"):
                                self.model.actions_mean.copy_(self.actions_mean.view(1, 1, 2))
                            if hasattr(self.model, "actions_std"):
                                self.model.actions_std.copy_(self.actions_std.view(1, 1, 2))
                    else:
                        print(f"[DiffusionPolicy][WARN] actions_mean/std not found under {data_dir}. "
                            "Diffusion actions will NOT be denormalized.")

        
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model = torch.compile(self.model, dynamic=True)


    def step(self, env: CoverageSystem) -> (float, bool):
        """
        Step function for the neural network controller.

        It performs three steps:
        1. Query the current observation from the environment.
        2. Compute actions using the neural network policy.
        3. Apply actions to the environment.

        Args:
            env: CoverageSystem object.

        Returns:
            A tuple (objective_value, converged_flag).
        """
        if self.policy_type == "LPAC":
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
            # Diffusion policy: run one reverse diffusion chain conditioned on current env state.
            with torch.no_grad():
                actions = self._diffusion_step(env)  # (N, 2)

        else:
            raise ValueError(f"[ControllerNN] Unknown PolicyType: {self.policy_type}")

        # Convert (N, 2) actions to C++ PointVector and step the environment.
        actions_np = actions.detach().cpu().numpy()
        point_vector_actions = PointVector(actions_np)
        error_flag = env.StepActions(point_vector_actions)
        if error_flag:
            raise ValueError("Error in env.StepActions(actions)")

        objective_value = env.GetObjectiveValue()
        # Simple convergence check: all actions close to zero.
        converged = torch.allclose(actions, torch.zeros_like(actions), atol=1e-5)
        return objective_value, converged

    def _diffusion_step(self, env: CoverageSystem) -> torch.Tensor:
        """Decentralized MADP execution.

        - Each robot computes its local perceptual token z_i = CNN(o_i).
        - Robots exchange only z_i within communication range (token buffer).
        - The encoder Ψ_E runs once per sampling call; DDIM denoises for S steps.
        """
        assert self.diffusion is not None, "[ControllerNN] Diffusion schedule not initialized."

        diffusion = self.diffusion
        if isinstance(diffusion, dict):
            diffusion = GaussianDiffusion.from_state_dict(diffusion, device=self.device)
            self.diffusion = diffusion

        num_robots = env.GetNumRobots()

        # Lazy init token buffer once we know N.
        if self._token_buffer is None or self._token_buffer.num_robots != num_robots:
            self._token_buffer = TokenBuffer(num_robots=num_robots, device=self.device, max_age_steps=self.token_ttl)

        # === 1) Build local/obstacle maps  ===
        raw_local_maps = CoverageEnvUtils.get_raw_local_maps(env, self.params)
        raw_obstacle_maps = CoverageEnvUtils.get_raw_obstacle_maps(env, self.params)

        # (B=1, N, H_raw, W_raw) -> (B*N, H, W)
        raw_local_maps = raw_local_maps.unsqueeze(0)
        raw_obstacle_maps = raw_obstacle_maps.unsqueeze(0)

        resized_local = CoverageEnvUtils.resize_maps(raw_local_maps, self.cnn_map_size)
        resized_obstacle = CoverageEnvUtils.resize_maps(raw_obstacle_maps, self.cnn_map_size)

        resized_local = resized_local.view(1, num_robots, self.cnn_map_size, self.cnn_map_size)
        resized_obstacle = resized_obstacle.view(1, num_robots, self.cnn_map_size, self.cnn_map_size)

        local_maps = resized_local[0]       # (N, H, W)
        obstacle_maps = resized_obstacle[0] # (N, H, W)

        # === 2) Build coverage_maps channels (consistent with DiffusionDataset) ===
        C = 4 if self.use_comm_map else 2
        coverage_maps = torch.zeros(
            (1, num_robots, C, self.cnn_map_size, self.cnn_map_size),
            dtype=torch.float32,
            device=self.device,
        )
        coverage_maps[0, :, 0] = local_maps.to(self.device)
        coverage_maps[0, :, 1] = obstacle_maps.to(self.device)

        if self.use_comm_map:
            comm_maps = CoverageEnvUtils.get_communication_maps(env, self.params, self.cnn_map_size)  # (N, 2, H, W)
            coverage_maps[0, :, 2:4] = comm_maps.to(self.device)

        # === 3) Robot positions ===
        positions = CoverageEnvUtils.get_robot_positions(env).unsqueeze(0).to(self.device)  # (1, N, 2)

        # === 4) Local token computation and (ideal) token exchange ===
        z_all = self.model.perceptual_tokens(coverage_maps)  # (1, N, D)

        if (self._step % self.comm_interval) == 0:
            self._token_buffer.write_all(z_all[0], positions[0], step=self._step)

       # === 5) One-shot sampling for all robots (token-only, mask enforces locality) ===
        num_sampling_steps = self.sampling_cfg.get("NumSamplingSteps", None)
        eta = float(self.sampling_cfg.get("Eta", 0.0))
        r = float(self.comm_radius)

        pos = positions[0]  # (N, 2)
        dx = pos[:, None, :] - pos[None, :, :]
        dist2 = (dx * dx).sum(dim=-1)                 # (N, N)
        edge_weights = (dist2 <= (r * r)).float()     # (N, N)
        edge_weights = edge_weights.unsqueeze(0)      # (1, N, N)

        # only tokens are used as conditioning (no raw map leakage beyond CNN->z)
        u_all = self.model.sample_actions_from_tokens(
            z=z_all,                      # (1, N, D)
            positions=positions,           # (1, N, 2)
            diffusion=diffusion,
            edge_weights=edge_weights,     # (1, N, N)  attention mask inside model
            num_steps=num_sampling_steps,
            eta=eta,
        )  # (1, N, 2) normalized

        actions = u_all[0]  # (N, 2)

        self._step += 1

        # === 6) De-normalize if action statistics are available ===
        if self.actions_mean is not None and self.actions_std is not None:
            actions = actions * self.actions_std + self.actions_mean

        return actions
