"""Shared wandb helpers for all training scripts."""
import os

import wandb

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass  # dotenv not installed; rely on env var being set externally


def init_wandb(project, entity, run_name, config=None, mode="online"):
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        print("WARNING: WANDB_API_KEY not found — running without wandb logging.")
        return False
    wandb.login(key=api_key)
    wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=config or {},
        mode=mode,
    )
    return True


def log_wandb(data, step=None):
    """Log a metrics dict to the active run; no-op if wandb isn't initialized.

    Safe to call unconditionally from training loops — if `--wandb` wasn't passed
    (or the key was missing) there is no active run and this returns immediately.
    """
    if wandb.run is None:
        return
    if step is not None:
        wandb.log(data, step=step)
    else:
        wandb.log(data)


def finish_wandb():
    wandb.finish()
