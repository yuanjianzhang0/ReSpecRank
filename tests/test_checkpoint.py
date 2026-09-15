import torch

from respecrank.checkpoint import load_checkpoint
from respecrank.config import ExperimentConfig
from respecrank.model import ReSpecRank


def test_checkpoint_reconstructs_model(tmp_path) -> None:
    config = ExperimentConfig()
    model = ReSpecRank.from_config(config.data, config.model)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "implementation_version": 2,
            "model_state": model.state_dict(),
            "config": config.to_dict(),
            "epoch": 3,
            "validation_rank_ic": 0.1,
        },
        checkpoint,
    )
    restored, restored_config, payload = load_checkpoint(checkpoint, torch.device("cpu"))
    assert restored_config.model.hidden_dim == 128
    assert payload["epoch"] == 3
    for original, loaded in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(original, loaded)


def test_legacy_checkpoint_requires_retraining(tmp_path) -> None:
    import pytest

    path = tmp_path / "legacy.pt"
    torch.save({"implementation_version": 1}, path)
    with pytest.raises(ValueError, match="retrain"):
        load_checkpoint(path, torch.device("cpu"))
