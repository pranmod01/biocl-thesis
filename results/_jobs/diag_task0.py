"""Diagnostic: is baseline task-0 underperformance an optimization failure or a
generalization gap? Train task 0 (baseline, no coarse) and log BOTH train-set and
test-set accuracy over epochs. High train / low test => data-limited generalization
(expected). Low train too => optimization problem (a real bug to chase).
"""
import torch
from torch.utils.data import DataLoader
from coarsecl.config import load_config, NUM_FINE
from coarsecl.data.cifar100 import get_datasets
from coarsecl.data.task_stream import TaskStream
from coarsecl.models.classifier import build_classifier
from coarsecl.utils import resolve_device, seen_mask_from_classes, set_seed
from coarsecl.coarse.sources import build_coarse_source

cfg = load_config("configs/a_baseline.yaml", "configs/base.yaml")
set_seed(cfg.run.seed)
device = resolve_device(cfg.run.device)
train_set, test_set = get_datasets(cfg.data.data_root)
stream = TaskStream(train_set, test_set, cfg.data.num_tasks, cfg.data.classes_per_task, cfg.data.order_seed)
coarse_source = build_coarse_source(cfg, device)
model = build_classifier(cfg).to(device)

seen = stream.seen_classes(0)
seen_mask = seen_mask_from_classes(seen, NUM_FINE, device)
print("task-0 classes:", sorted(seen))

train_sub = stream.train_subset(0)
test_sub = stream.test_subset(0)
loader = DataLoader(train_sub, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.data.num_workers)

opt = torch.optim.SGD(model.parameters(), lr=cfg.train.lr, momentum=cfg.train.momentum, weight_decay=cfg.train.weight_decay)
EPOCHS = 60
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

@torch.no_grad()
def acc(dataset):
    model.eval()
    dl = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=cfg.data.num_workers)
    c = t = 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        cd = coarse_source.get_coarse_dist(x, y)
        p = model.predict(x, cd, seen_mask)
        c += (p == y).sum().item(); t += y.numel()
    return c / t

print(f"train imgs={len(train_sub)}  test imgs={len(test_sub)}")
for ep in range(EPOCHS):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        cd = coarse_source.get_coarse_dist(x, y)
        opt.zero_grad()
        loss = model.compute_loss(x, y, cd, seen_mask)
        loss.backward(); opt.step()
    sched.step()
    if (ep + 1) % 10 == 0 or ep == 0:
        print(f"epoch {ep+1:3d}  train_acc={acc(train_sub):.3f}  test_acc={acc(test_sub):.3f}")
