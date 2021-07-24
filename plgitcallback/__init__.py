import json
import subprocess
import warnings
from pathlib import Path
from typing import Any, Dict, Union, List

import git
import pytorch_lightning as pl
from torch._C import Value

from dataclasses import dataclass, asdict, field


@dataclass
class GitStatus:
    valid: bool = False
    empty: bool = False
    dirty: bool = False
    commit_hash: str = None
    untracked_files: List[str] = field(default_factory=list)

    def test(self, ignore_commit: bool = False, ignore_untracked: bool = True):
        return (
            self.valid
            and not self.empty
            and (ignore_commit or self.commit_hash is not None)
            and not self.dirty
            and (ignore_untracked or len(self.untracked_files) == 0)
        )


class RepositoryError(Exception):
    pass


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
            raise RepositoryError(
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


def _get_git_status(git_dir: Union[str, Path]) -> GitStatus:
    repo = git.Repo(str(Path(git_dir).resolve()), search_parent_directories=True)
    try:
        repo.git_dir  # causes exception if not a git-dir
        if not repo.active_branch.is_valid():
            return GitStatus(
                valid=True,
                empty=True,
                dirty=repo.is_dirty(),
                untracked_files=repo.untracked_files,
            )
        else:
            return GitStatus(
                valid=True,
                empty=False,
                dirty=repo.is_dirty(),
                commit_hash=repo.head.object.hexsha,
                untracked_files=repo.untracked_files,
            )
    except git.exc.InvalidGitRepositoryError:
        return GitStatus(valid=False)
    finally:
        repo.close()
