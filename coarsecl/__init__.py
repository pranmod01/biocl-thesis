"""coarsecl: does a coarse-category signal reduce catastrophic forgetting?

Minimal Class-IL experiment on CIFAR-100. Four conditions (baseline / oracle /
soft proxy / trained coarse net) differ only in how a distribution over the 20
superclasses is produced; that distribution enters the fine classifier through a
swappable conditioning mechanism (concat | aux_loss). See CLAUDE.md.
"""

__all__ = []
