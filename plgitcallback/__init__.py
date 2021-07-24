import json
import subprocess
import warnings
from pathlib import Path
from typing import Any, Dict, Union, List, Optional
from contextlib import contextmanager

import git
import pytorch_lightning as pl
from torch._C import Value

from dataclasses import dataclass, asdict, field


class GitRepositoryError(Exception):
    pass


@contextmanager
def open_repo(*args, **kwargs):
    repo = git.Repo(*args, **kwargs)
    try:
        yield repo
    finally:
        repo.close()


@dataclass(frozen=True)
class GitStatus:
    valid: bool = False
    empty: Optional[bool] = None
    dirty: Optional[bool] = None
    commit_hash: Optional[str] = None
    branch_name: Optional[str] = None
    untracked_files: List[str] = field(default_factory=list)
    # repo.remotes.origin.url

    @staticmethod
    def get_status(git_dir: Union[str, Path]) -> "GitStatus":
        path = Path(git_dir).resolve()

        # Check if valid path
        if not path.is_dir():
            return GitStatus(valid=False)

        with open_repo(str(path), search_parent_directories=True) as repo:
            repo: git.Repo
            try:
                _ = repo.git_dir  # Check if valid git-dir
            except git.exc.InvalidGitRepositoryError:
                return GitStatus(valid=False)
            branch = repo.active_branch
            if not branch.is_valid():
                return GitStatus(
                    valid=True, empty=True, untracked_files=repo.untracked_files
                )
            else:
                return GitStatus(
                    valid=True,
                    empty=False,
                    dirty=repo.is_dirty(),
                    commit_hash=repo.head.object.hexsha,
                    branch_name=branch.name,
                    untracked_files=repo.untracked_files,
                )


class GitCommitCallback(pl.Callback):
    """Logs git commit info in training. In particular,
    this callback injects git commit info into model checkpoint
    info and writes a `git_info.json` file to trainer's log dir.

    The git info in the checkpoint can be found in
        ckpt['callbacks'][plgitcallback.GitCommitCallback]

    When `strict` is true, the callback will raise an
    RepositoryError when uncommitted changes have been
    detected and thus stops training.
    """

    def __init__(
        self,
        git_dir: Union[str, Path] = ".",
        ignore_untracked: bool = True,
        strict: bool = False,
    ) -> None:
        super().__init__()
        self.strict = strict
        self.ignore_untracked = ignore_untracked
        self.git_status = _get_git_status(git_dir)

    def on_train_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        ok = self.git_status.test(ignore_untracked=self.ignore_untracked)
        if self.strict and not ok:
            raise GitRepositoryError(
                {
                    "message": "Failed to start training because of git repository errors",
                    "status": self.git_status,
                }
            )
        elif not ok:
            warnings.warn(
                (
                    f"\n----------------------------------------------------------\n"
                    f"Repository contains uncommitted changes:\n"
                    f"{self.git_status}\n"
                    f"For traceability it is recommended to commit before training,\n"
                    f"in order to embed a clean commit hash into checkpoints.\n"
                    f"----------------------------------------------------------"
                )
            )

        if trainer.log_dir is not None:
            log_dir = Path(trainer.log_dir)
            with open(log_dir / "git-status.json", "w") as f:
                f.write(json.dumps(asdict(self.git_status), indent=2, sort_keys=False))

    def on_save_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        checkpoint: Dict[str, Any],
    ) -> dict:
        return asdict(self.git_status)

    def on_load_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        callback_state: Dict[str, Any],
    ) -> None:
        # Here we could check if current and stored commit hash match.
        return super().on_load_checkpoint(trainer, pl_module, callback_state)
