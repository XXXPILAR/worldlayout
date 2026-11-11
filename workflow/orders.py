"""Kept for callers of the earlier interface (infinite-task-world's FastFill bridge imports ``orders.critic_geometry``):
the critic's rules are ``critic.critic_geometry``."""

from .critic import critic_geometry

__all__ = ["critic_geometry"]
