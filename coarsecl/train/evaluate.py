"""Class-IL evaluation: accuracy over the seen-class set, no task ID used.

The coarse signal at eval comes from the same source as training (see DESIGN.md):
for oracle/soft that is the test sample's true superclass (the intended upper-bound
framing); for trained the coarse net runs on the test image. Under aux_loss the
fine prediction ignores the signal anyway.

Each task is scored at two scales from the SAME fine prediction:
- fine:  pred == true fine class.
- super: superclass(pred) == true superclass. Fine errors that stay inside the true
  superclass count as correct here, so fine/super separates "forgot which branch"
  from "forgot which leaf within the branch". Under an oracle gate the prediction is
  masked to the true superclass, so super is ~1.0 by construction, not by knowledge.
- source: argmax of the coarse signal itself vs the true superclass, the reference
  line for super (None when there is no signal).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from ..config import NUM_COARSE
from ..data.cifar100 import coarse_of


@torch.no_grad()
def evaluate_task(model, dataset, coarse_source, seen_mask, device,
                  batch_size: int = 256, num_workers: int = 4,
                  withhold: bool = False) -> Dict[str, Optional[float]]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    correct = super_correct = source_correct = total = 0
    has_source = False
    for images, fine, uid in loader:
        images = images.to(device)
        fine = fine.to(device)
        true_super = coarse_of(fine)
        signal = coarse_source.get_coarse_dist(images, fine, uid)
        if signal is not None:
            has_source = True
            source_correct += (signal.argmax(1) == true_super).sum().item()
        if withhold:
            # ablation: hide the real signal, feed a no-information uniform dist
            coarse_dist = images.new_full((images.size(0), NUM_COARSE), 1.0 / NUM_COARSE)
        else:
            coarse_dist = signal
        pred = model.predict(images, coarse_dist, seen_mask)
        correct += (pred == fine).sum().item()
        super_correct += (coarse_of(pred) == true_super).sum().item()
        total += fine.numel()
    total = max(total, 1)
    return {"fine": correct / total, "super": super_correct / total,
            "source": source_correct / total if has_source else None}
