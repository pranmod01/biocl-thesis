"""Is the 0.000 retention wall a real floor or a Class-IL measurement ceiling?

Train baseline + rotation with the real loop, then probe the FINAL model on every
task under metrics that have dynamic range where Class-IL top-1 has none:
  - Class-IL top1  (the wall: argmax over all 100)   -> expect ~0
  - Class-IL top5  (true class in top-5 over all 100)
  - Task-IL  top1  (argmax over that task's own 10 classes; chance 0.10) -> latent
                    feature retention, blind to cross-task head bias
  - mean rank of the true class among 100 (0 = top)
If Task-IL >> 0.10 the features still hold old classes and the wall is a ceiling;
rotation-vs-baseline on Task-IL is then the real internalization test.
"""
import torch
from torch.utils.data import DataLoader

from coarsecl.config import load_config
from coarsecl.train.trainer import run_continual


@torch.no_grad()
def fine_logits(model, images):
    outputs = model.forward(images, None)  # source=none for both configs
    return model.strategy.eval_logits(outputs, None)  # [B, 100]


@torch.no_grad()
def probe(model, stream, device, num_tasks):
    model.eval()
    rows = []
    for j in range(num_tasks):
        cls = stream.task_classes[j]
        cls_t = torch.tensor(cls, device=device)
        loader = DataLoader(stream.test_subset(j), batch_size=256, num_workers=4)
        n = c_top1 = c_top5 = t_top1 = 0
        rank_sum = 0.0
        for images, fine in loader:
            images, fine = images.to(device), fine.to(device)
            lg = fine_logits(model, images)                       # [B, 100]
            n += fine.numel()
            c_top1 += (lg.argmax(1) == fine).sum().item()
            top5 = lg.topk(5, dim=1).indices
            c_top5 += (top5 == fine[:, None]).any(1).sum().item()
            # Task-IL: restrict to this task's 10 classes
            sub = lg[:, cls_t]                                    # [B, 10]
            pred = cls_t[sub.argmax(1)]
            t_top1 += (pred == fine).sum().item()
            # rank of the true class among all 100 (0 = highest logit)
            true_logit = lg.gather(1, fine[:, None])
            rank_sum += (lg > true_logit).sum(1).float().sum().item()
        rows.append((j, c_top1 / n, c_top5 / n, t_top1 / n, rank_sum / n))
    return rows


def main():
    import sys
    names = sys.argv[1:] or ["a_baseline", "r_rotation"]
    for name in names:
        cfg = load_config(f"configs/{name}.yaml")
        print(f"\n########## {name} ##########", flush=True)
        _, model, stream, device = run_continual(cfg, return_model=True)
        rows = probe(model, stream, device, cfg.data.num_tasks)
        print(f"{'task':>4} {'ClassIL@1':>10} {'ClassIL@5':>10} {'TaskIL@1':>9} {'mean_rank':>10}")
        for j, c1, c5, t1, rk in rows:
            print(f"{j:>4} {c1:>10.3f} {c5:>10.3f} {t1:>9.3f} {rk:>10.1f}", flush=True)
        old = rows[:-1]  # tasks 0..8, the forgotten ones
        m = lambda i: sum(r[i] for r in old) / len(old)
        print(f"OLD-TASK MEAN  ClassIL@1={m(1):.3f}  ClassIL@5={m(2):.3f}  "
              f"TaskIL@1={m(3):.3f}  mean_rank={m(4):.1f}  (chance: @1=0.01 @5=0.05 TaskIL=0.10 rank=49.5)", flush=True)


if __name__ == "__main__":
    main()
