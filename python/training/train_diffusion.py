"""
Train a diffusion policy model for multi-agent coverage control.

Usage:
    python train_diffusion.py <learning_params.toml> <world_size>
"""

import os
import pathlib
import sys
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from coverage_control import IOUtils
from coverage_control.nn.models.diffusion_policy import DiffusionPolicy


class DiffusionDataset(Dataset):
    """
        <DataDir>/<append-dir>/
            train/
                local_maps.pt
                obstacle_maps.pt
                comm_maps.pt
                robot_positions.pt
                edge_weights.pt
                actions.pt or normalized_actions.pt
            val/
            test/
            actions_mean.pt
            actions_std.pt
    """

    def __init__(
        self,
        data_dir: pathlib.Path,
        split: str,
        use_comm_map: bool,
        world_size: int,
    ) -> None:
        super().__init__()
        self.split = split
        self.data_root = pathlib.Path(data_dir) / split
        self.use_comm_map = use_comm_map

        if not self.data_root.exists():
            raise FileNotFoundError(f"Dataset split directory does not exist: {self.data_root}")

        def _load(name: str) -> torch.Tensor:
            path = self.data_root / name
            if not path.exists():
                raise FileNotFoundError(f"Missing tensor file: {path}")
            t = torch.load(path, map_location="cpu")
            if isinstance(t, torch.Tensor) and t.is_sparse:
                t = t.to_dense()
            return t

        # Core tensors
        robot_positions = _load("robot_positions.pt")       # (M, N, 2)
        local_maps = _load("local_maps.pt")                 # (M, N, H, W)
        obstacle_maps = _load("obstacle_maps.pt")           # (M, N, H, W)
        comm_maps = _load("comm_maps.pt")                   # (M, N, 2, H, W)
        edge_weights = _load("edge_weights.pt")             # (M, N, N)

        # Actions: prefer normalized if present
        actions_path = self.data_root / "normalized_actions.pt"
        if actions_path.exists():
            actions = torch.load(actions_path, map_location="cpu")
            if isinstance(actions, torch.Tensor) and actions.is_sparse:
                actions = actions.to_dense()
            self.actions_normalized = True
        else:
            actions = _load("actions.pt")
            self.actions_normalized = False

        if isinstance(actions, torch.Tensor) and actions.is_sparse:
            actions = actions.to_dense()

        # Sanity checks & shapes
        M, N, H, W = local_maps.shape
        if N != world_size:
            raise ValueError(f"World size mismatch: dataset has {N} robots, but world_size={world_size}")

        # Build coverage maps: (M, N, C, H, W)
        if use_comm_map:
            C = 4
            coverage_maps = torch.zeros((M, N, C, H, W), dtype=local_maps.dtype)
            coverage_maps[:, :, 0, :, :] = local_maps
            coverage_maps[:, :, 1, :, :] = obstacle_maps
            coverage_maps[:, :, 2:4, :, :] = comm_maps
        else:
            C = 2
            coverage_maps = torch.zeros((M, N, C, H, W), dtype=local_maps.dtype)
            coverage_maps[:, :, 0, :, :] = local_maps
            coverage_maps[:, :, 1, :, :] = obstacle_maps

        self.coverage_maps = coverage_maps      # (M, N, C, H, W)
        self.actions = actions                  # (M, N, 2)
        self.robot_positions = robot_positions  # (M, N, 2)
        self.edge_weights = edge_weights        # (M, N, N)

        self.num_samples = M
        self.world_size = N
        self.map_size = H

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            coverage_maps[idx]:   (N, C, H, W)
            actions[idx]:         (N, 2)
            robot_positions[idx]: (N, 2)
            edge_weights[idx]:    (N, N)
        """
        return (
            self.coverage_maps[idx],
            self.actions[idx],
            self.robot_positions[idx],
            self.edge_weights[idx],
        )


# Diffusion utilities

def build_diffusion_schedule(
    num_steps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device: torch.device | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Create a simple linear beta schedule and the corresponding alpha_bar(t).
    """
    device = device or torch.device("cpu")
    betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    return {
        "num_steps": num_steps,
        "betas": betas,
        "alphas": alphas,
        "alpha_cumprod": alpha_cumprod,
    }


# Training / evaluation loops

def train_one_epoch(
    model: nn.Module,
    diffusion: Dict[str, torch.Tensor],
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()

    num_steps = diffusion["num_steps"]
    alpha_cumprod = diffusion["alpha_cumprod"]

    total_loss = 0.0
    total_samples = 0

    for coverage_maps, targets, robot_positions, edge_weights in data_loader:
        coverage_maps = coverage_maps.to(device)      # (B, N, C, H, W)
        targets = targets.to(device)                  # (B, N, 2)
        robot_positions = robot_positions.to(device)  # (B, N, 2)
        # edge_weights = edge_weights.to(device)      # (B, N, N)  # reserved for future use

        B, N, C, H, W = coverage_maps.shape

        actions_0 = targets  # ground-truth at t=0

        # Sample one diffusion time step per graph
        t = torch.randint(
            0, num_steps, (B,), device=device, dtype=torch.long
        )  # (B,)
        alpha_bar_t = alpha_cumprod[t].view(B, 1, 1)  # (B,1,1)

        # Forward diffusion: add Gaussian noise
        noise = torch.randn_like(actions_0)
        actions_t = torch.sqrt(alpha_bar_t) * actions_0 + torch.sqrt(1.0 - alpha_bar_t) * noise

        optimizer.zero_grad(set_to_none=True)
        eps_hat = model(coverage_maps, actions_t, t, robot_positions)  # (B, N, 2)

        loss = F.mse_loss(eps_hat, noise)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        total_samples += B

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    diffusion: Dict[str, torch.Tensor],
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()

    num_steps = diffusion["num_steps"]
    alpha_cumprod = diffusion["alpha_cumprod"]

    total_loss = 0.0
    total_samples = 0

    for coverage_maps, targets, robot_positions, edge_weights in data_loader:
        coverage_maps = coverage_maps.to(device)
        targets = targets.to(device)
        robot_positions = robot_positions.to(device)

        B, N, C, H, W = coverage_maps.shape
        actions_0 = targets

        t = torch.randint(
            0, num_steps, (B,), device=device, dtype=torch.long
        )
        alpha_bar_t = alpha_cumprod[t].view(B, 1, 1)

        noise = torch.randn_like(actions_0)
        actions_t = torch.sqrt(alpha_bar_t) * actions_0 + torch.sqrt(1.0 - alpha_bar_t) * noise

        eps_hat = model(coverage_maps, actions_t, t, robot_positions)
        loss = F.mse_loss(eps_hat, noise)

        total_loss += loss.item() * B
        total_samples += B

    return total_loss / max(total_samples, 1)


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: python train_diffusion.py <learning_params.toml> <world_size>"
        )

    config_file = sys.argv[1]
    world_size = int(sys.argv[2])

    # Load training config
    config = IOUtils.load_toml(config_file)

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Dataset root
    dataset_root = pathlib.Path(IOUtils.sanitize_path(config["DataDir"]))
    data_dir = dataset_root / "data"

    num_workers = int(config.get("NumWorkers", 4))
    use_comm_map = bool(config["ModelConfig"]["UseCommMaps"])

    diff_model_cfg = config["DiffusionModel"]
    diff_train_cfg = config["DiffusionTraining"]

    batch_size = int(diff_train_cfg["BatchSize"])
    num_epochs = int(diff_train_cfg["NumEpochs"])
    lr = float(diff_train_cfg["LearningRate"])
    weight_decay = float(diff_train_cfg["WeightDecay"])
    num_diff_steps = int(diff_train_cfg["NumDiffusionSteps"])

    # Model directory
    model_dir = IOUtils.sanitize_path(diff_model_cfg["Dir"])
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, "diffusion_policy_best.pt")

    # Datasets
    train_dataset = DiffusionDataset(data_dir, "train", use_comm_map, world_size)
    val_dataset = DiffusionDataset(data_dir, "val", use_comm_map, world_size)

    print(
        f"Dataset: train | Size: {len(train_dataset)} "
        f"Coverage Maps: {tuple(train_dataset.coverage_maps.shape)} "
        f"Targets: {tuple(train_dataset.actions.shape)} "
        f"Robot Positions: {tuple(train_dataset.robot_positions.shape)} "
        f"Edge Weights: {tuple(train_dataset.edge_weights.shape)}"
    )
    print(
        f"Dataset: val   | Size: {len(val_dataset)} "
        f"Coverage Maps: {tuple(val_dataset.coverage_maps.shape)} "
        f"Targets: {tuple(val_dataset.actions.shape)} "
        f"Robot Positions: {tuple(val_dataset.robot_positions.shape)} "
        f"Edge Weights: {tuple(val_dataset.edge_weights.shape)}"
    )

    # Load action normalization stats if present (for later sampling / sim2real)
    actions_mean_path = data_dir / "actions_mean.pt"
    actions_std_path = data_dir / "actions_std.pt"
    actions_mean = None
    actions_std = None
    if actions_mean_path.exists() and actions_std_path.exists():
        actions_mean = torch.load(actions_mean_path, map_location="cpu")
        actions_std = torch.load(actions_std_path, map_location="cpu")

    # Create model (uses CNNBackBone + DiffusionModel from config)
    model = DiffusionPolicy(config).to(device)

    if actions_mean is not None and actions_std is not None:
        model.register_buffer("actions_mean", actions_mean.to(device))
        model.register_buffer("actions_std", actions_std.to(device))

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Diffusion schedule
    diffusion = build_diffusion_schedule(num_diff_steps, device=device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, diffusion, train_loader, optimizer, device)
        val_loss = eval_one_epoch(model, diffusion, val_loader, device)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss = {train_loss:.6f} | val_loss = {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state_dict = model.state_dict()
            if actions_mean is not None and actions_std is not None:
                state_dict["actions_mean"] = actions_mean
                state_dict["actions_std"] = actions_std

            torch.save(
                {
                    "model_state_dict": state_dict,
                    "config": diff_model_cfg,
                    "training_cfg": diff_train_cfg,
                },
                best_model_path,
            )

            print(f"  -> Saved best model to: {best_model_path}")

    # Optional: test set evaluation
    test_dataset = DiffusionDataset(data_dir, "test", use_comm_map, world_size)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loss = eval_one_epoch(model, diffusion, test_loader, device)
    print(f"[Test] loss = {test_loss:.6f}")


if __name__ == "__main__":
    main()
