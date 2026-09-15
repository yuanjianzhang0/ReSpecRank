import json

import numpy as np

from respecrank.data import CrossSectionDataset


def test_processed_date_loading(tmp_path) -> None:
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    np.savez_compressed(
        split_dir / "2025-01-02.npz",
        date=np.asarray("2025-01-02"),
        symbols=np.asarray(["A", "B"]),
        features=np.zeros((2, 20, 12), dtype=np.float32),
        edge_index=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        edge_weight=np.asarray([-1.0, -1.0], dtype=np.float32),
        market_state=np.zeros(5, dtype=np.float32),
        targets=np.asarray([-1.0, 1.0], dtype=np.float32),
    )
    manifest = {
        "splits": {"train": ["train/2025-01-02.npz"]},
        "metadata": {"implementation_version": 2},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    dataset = CrossSectionDataset(tmp_path, "train")
    item = dataset[0]
    assert item.date == "2025-01-02"
    assert item.num_stocks == 2
    assert item.features.shape == (2, 20, 12)
