"""Training loop with date-native mini-batches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from respecrank.config import ExperimentConfig
from respecrank.evaluation import evaluate_model
from respecrank.losses import respecrank_loss
from respecrank.model import ReSpecRank
from respecrank.utils import write_json


@dataclass
class EpochRecord:
    epoch: int
    loss: float
    ranking_loss: float
    huber_loss: float
    diversity_loss: float
    validation_rank_ic: float
    validation_ic: float
    router_active: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return self.__dict__.copy()


def create_optimizer(model: ReSpecRank, config: ExperimentConfig) -> torch.optim.Optimizer:
    router_parameters = list(model.router.parameters())
    if model.routing_mode == "direct":
        router_parameters.extend(model.direct_routers.parameters())
    if model.routing_mode == "feature":
        router_parameters.extend(model.feature_conditioner.parameters())
    router_ids = {id(parameter) for parameter in router_parameters}
    main_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in router_ids
    ]
    return torch.optim.AdamW(
        [
            {"params": main_parameters, "lr": config.training.main_learning_rate},
            {"params": router_parameters, "lr": config.training.router_learning_rate},
        ],
        weight_decay=config.training.weight_decay,
    )


def _save_checkpoint(
    path: Path,
    model: ReSpecRank,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    epoch: int,
    validation_rank_ic: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "implementation_version": 2,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config.to_dict(),
            "epoch": epoch,
            "validation_rank_ic": validation_rank_ic,
        },
        path,
    )


def train_model(
    model: ReSpecRank,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
) -> list[EpochRecord]:
    model.to(device)
    optimizer = create_optimizer(model, config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "resolved_config.json", config.to_dict())

    history = []
    best_rank_ic = -float("inf")
    stale_epochs = 0
    for epoch in range(config.training.epochs):
        model.train()
        router_active = epoch >= config.training.router_warmup_epochs
        accumulated = {"total": 0.0, "ranking": 0.0, "huber": 0.0, "diversity": 0.0}
        num_batches = 0
        for date_group in train_loader:
            batches = [item.to(device) for item in date_group]
            outputs = [model(item, force_uniform_router=not router_active) for item in batches]
            targets = [item.targets for item in batches]
            if any(target is None for target in targets):
                raise ValueError("training requires targets")
            terms = respecrank_loss(
                [output.scores for output in outputs],
                targets,
                model.coefficient_bases,
                model.temporal_dilations,
                num_pairs=config.loss.sampled_pairs,
                huber_weight=config.loss.huber_weight,
                diversity_weight=config.loss.diversity_weight,
                huber_delta=config.loss.huber_delta,
                response_grid_size=config.loss.response_grid_size,
                channel_mixing=model.channel_mixing,
            )
            optimizer.zero_grad(set_to_none=True)
            terms.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            optimizer.step()
            accumulated["total"] += float(terms.total.detach())
            accumulated["ranking"] += float(terms.ranking.detach())
            accumulated["huber"] += float(terms.huber.detach())
            accumulated["diversity"] += float(terms.diversity.detach())
            num_batches += 1

        validation = evaluate_model(
            model, validation_loader, device, force_uniform_router=not router_active
        )
        record = EpochRecord(
            epoch=epoch + 1,
            loss=accumulated["total"] / num_batches,
            ranking_loss=accumulated["ranking"] / num_batches,
            huber_loss=accumulated["huber"] / num_batches,
            diversity_loss=accumulated["diversity"] / num_batches,
            validation_rank_ic=validation.summary.rank_ic,
            validation_ic=validation.summary.ic,
            router_active=router_active,
        )
        history.append(record)
        print(json.dumps(record.to_dict(), sort_keys=True), flush=True)

        _save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            config,
            epoch + 1,
            validation.summary.rank_ic,
        )
        if validation.summary.rank_ic > best_rank_ic:
            best_rank_ic = validation.summary.rank_ic
            stale_epochs = 0
            _save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                config,
                epoch + 1,
                validation.summary.rank_ic,
            )
        else:
            stale_epochs += 1
        write_json(output_dir / "history.json", [entry.to_dict() for entry in history])
        patience = config.training.early_stopping_patience
        if patience is not None and stale_epochs >= patience:
            break
    return history
