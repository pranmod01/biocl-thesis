"""Class-IL evaluation: accuracy over the seen-class set, no task ID used.

The coarse signal at eval comes from the same source as training (see DESIGN.md):
for oracle/soft that is the test sample's true superclass (the intended upper-bound
framing); for trained the coarse net runs on the test image. Under aux_loss the
fine prediction ignores the signal anyway.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ..config import NUM_COARSE


@torch.no_grad()
def evaluate_task(model, dataset, coarse_source, seen_mask, device,
                  batch_size: int = 256, num_workers: int = 4,
                  withhold: bool = False) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    correct = total = 0
    for images, fine in loader:
        images = images.to(device)
        fine = fine.to(device)
        if withhold:
            # ablation: hide the real signal, feed a no-information uniform dist
            coarse_dist = images.new_full((images.size(0), NUM_COARSE), 1.0 / NUM_COARSE)
        else:
            coarse_dist = coarse_source.get_coarse_dist(images, fine)
        pred = model.predict(images, coarse_dist, seen_mask)
        correct += (pred == fine).sum().item()
        total += fine.numel()
    return correct / max(total, 1)
