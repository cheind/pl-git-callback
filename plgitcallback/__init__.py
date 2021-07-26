import json
import logging
import time
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import git
import pytorch_lightning as pl


class GitCommitCallbackError(Exception):
    pass


class GitCommitCallbackWarning(UserWarning, ValueError):
    pass


_logger = logging.getLogger(__name__)


@contextmanager
def open_repo(*args, **kwargs):
    repo = git.Repo(*args, **kwargs)
    try:
        yield repo
    finally:
        repo.close()


@dataclass(frozen=True)
class GitStatus:
    empty: Optional[bool] = None  # No commits yet
    dirty: Optional[
        bool
    ] = None  # Like a git-status without untracked files (index or working copy changes that are not committed)
    num_untracked_files: Optional[bool] = None  # Any untracked files
    commit_hash: Optional[str] = None  # Commit hash
    commit_date: Optional[int] = None  # Commit data in number of seconds since epoch
    branch_name: Optional[str] = None  # Current branch name

    def commit_info(self) -> str:
        return f"git(commit={self.commit_hash},commit_date={self.asc_commit_date}"

    @property
    def asc_commit_date(self):
        return time.asctime(time.gmtime(self.commit_date))


class GitCommitCallback(pl.Callback):
    """Logs git repository info in training.

    In particular, this callback injects git commit info into
    model checkpoint info and writes a `git_info.json` file
    to trainer's log dir.

    The git info in the checkpoint can be found in
        ckpt['callbacks'][plgitcallback.GitCommitCallback]

    When `strict` is true, the callback will raise an
    RepositoryError when uncommitted changes have been
    detected and thus stops training.
    """

    def __init__(
        self,
        git_dir: Union[str, Path] = ".",
        mode: Literal["relaxed", "strict"] = "relaxed",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.git_dir = git_dir
        self.git_status = gitstatus_from_repository(git_dir)
        self.disabled = self.git_status is None
        if self.disabled:
            self._warn("GitCommitCallback disabled, not a valid git repository.")

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if not self.disabled and (self.git_status.empty or self.git_status.dirty):
            self._warn_or_raise(
                "Repository is not in a clean state. Please commit before you train."
            )

        if not self.disabled and trainer.log_dir is not None:
            log_dir = Path(trainer.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            with open(log_dir / "git-status.json", "w") as f:
                f.write(json.dumps(asdict(self.git_status), indent=2, sort_keys=False))

        return super().on_fit_start(trainer, pl_module)

    def on_save_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        checkpoint: Dict[str, Any],
    ) -> dict:
        if not self.disabled:
            return asdict(self.git_status) if not self.disabled else None
        else:
            return None

    def on_load_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        callback_state: Dict[str, Any],
    ) -> None:
        if not self.disabled and callback_state is None:
            self._warn_or_raise("Loaded state does not contain git-status information.")
        load_status = GitStatus(**callback_state)
        if not self.disabled and load_status != self.git_status:
            self._warn_or_raise(
                f"Git status mismatch between loaded state {load_status.commit_info()} and current state {self.git_status.commit_info()}"
            )
        return super().on_load_checkpoint(trainer, pl_module, callback_state)

    def _warn_or_raise(self, msg: str):
        if self.mode == "strict":
            self._error(msg)
        else:
            self._warn(msg)

    def _warn(self, msg):
        _logger.warning(msg)
        warnings.warn(message=msg, category=GitCommitCallbackWarning)

    def _error(self, msg):
        _logger.error(msg)
        raise GitCommitCallbackError(msg)


def gitstatus_from_repository(git_dir: Union[str, Path]) -> Optional[GitStatus]:
    path = Path(git_dir).resolve()

    # Check if valid path
    if not path.is_dir():
        return None

    with open_repo(str(path), search_parent_directories=True) as repo:
        repo: git.Repo
        try:
            _ = repo.git_dir  # Check if valid git-dir
        except git.exc.InvalidGitRepositoryError:
            return None
        branch = repo.active_branch
        if not branch.is_valid():
            return GitStatus(empty=True, num_untracked_files=len(repo.untracked_files))
        else:
            return GitStatus(
                empty=False,
                dirty=repo.is_dirty(),
                num_untracked_files=len(repo.untracked_files),
                commit_hash=repo.head.commit.hexsha,
                commit_date=repo.head.commit.committed_date,
                branch_name=branch.name,
            )


def gitstatus_from_json(fp) -> Optional[GitStatus]:
    """Return GitStatus from json-dict like object deserialized from `fp`."""
    try:
        return GitStatus(**json.load(fp))
    except (json.JSONDecodeError, TypeError):
        return None


def gitstatus_from_lightning_checkpoint(ckpt: Dict[Any, Any]):
    return GitStatus(**ckpt["callbacks"][GitCommitCallback])
