import os
import uuid
from pathlib import Path

import git
import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from plgitcallback import GitCommitCallback, GitRepositoryError, GitStatus
import tempfile

IS_TRAVIS = "TRAVIS" in os.environ
THIS_DIR = Path(__file__).parent

# import warnings

# warnings.filterwarnings("ignore")


# class LinearRegressionModel(pl.LightningModule):
#     def __init__(self):
#         super().__init__()
#         self.w = nn.Linear(1, 1, bias=True)
#         self.w.weight.data.fill_(1.7)
#         self.w.bias.data.fill_(0.7)

#     def forward(self, x):
#         # in lightning, forward defines the prediction/inference actions
#         return self.w(x)  # y = k*x + d

#     def training_step(self, batch, batch_idx):
#         loss = self._step(batch)
#         self.log("train_loss", loss)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         loss = self._step(batch)
#         self.log("val_loss", loss)
#         return loss

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=1e-1)

#     def _step(self, batch):
#         x, y = batch
#         yhat = self(x)
#         return F.mse_loss(yhat, y, reduction="sum")


# xs = torch.arange(-10, 10, 0.1).view(-1, 1)
# ys = xs * 2 + 1
# ds_train = torch.utils.data.TensorDataset(xs, ys)


@pytest.fixture
def tmp_git_repo():
    """Fixture that yields a git repository initialized in a temporary directory"""

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        repo = git.Repo.init(Path(tmp))
        yield repo, tmp
        repo.close()


def test_gitstatus(tmp_git_repo):
    repo, rpath = tmp_git_repo
    status = GitStatus.get_status(rpath)
    assert status.valid
    assert status.empty
    assert status.dirty is None
    assert status.commit_hash is None
    assert status.branch_name is None

    open(rpath / "hugo.txt", "w").write("hello")
    status = GitStatus.get_status(rpath)
    assert status.valid
    assert status.empty
    assert status.dirty is None
    assert status.branch_name is None
    assert len(status.untracked_files) == 1

    repo.index.add("hugo.txt")
    status = GitStatus.get_status(rpath)
    assert status.valid
    assert status.empty
    assert status.dirty is None
    assert status.branch_name is None
    assert len(status.untracked_files) == 0

    # Commit file
    repo.git.commit("-m", "test commit")
    status = GitStatus.get_status(rpath)
    assert status.valid
    assert not status.empty
    assert not status.dirty
    assert status.branch_name == "master"
    assert len(status.untracked_files) == 0

    with open(rpath / "hugo.txt", "w") as f:
        f.write("update")
    status = GitStatus.get_status(rpath)
    # repo.index.add("hugo.txt")
    assert status.valid
    assert not status.empty
    assert status.dirty
    assert status.branch_name == "master"
    assert len(status.untracked_files) == 0

    repo.index.add("hugo.txt")
    status = GitStatus.get_status(rpath)
    assert status.valid
    assert not status.empty
    assert status.dirty
    assert status.branch_name == "master"
    assert len(status.untracked_files) == 0

    repo.git.commit("-m", "test commit")
    status = GitStatus.get_status(rpath)
    assert status.valid
    assert not status.empty
    assert not status.dirty
    assert status.branch_name == "master"
    assert len(status.untracked_files) == 0


def test_gitcommitcallback_trainstart(tmp_git_repo):
    repo, rpath = tmp_git_repo

    class MyTrainer:
        def __init__(self, log_dir):
            self.log_dir = log_dir

    with tempfile.TemporaryDirectory() as tmp:
        trainer = MyTrainer(tmp)

        cb = GitCommitCallback(rpath, strict=False, ignore_untracked=True)
        with pytest.warns(UserWarning):
            cb.on_train_start(trainer, None)
            assert (Path(tmp) / "git-status.json").is_file()

        cb = GitCommitCallback(rpath, strict=True, ignore_untracked=True)
        with pytest.raises(RepositoryError):
            cb.on_train_start(trainer, None)

        with open(rpath / "hugo.txt", "w") as f:
            f.write("hello")
        repo.index.add("hugo.txt")
        cb = GitCommitCallback(rpath, strict=True)
        with pytest.raises(RepositoryError):
            cb.on_train_start(trainer, None)

        repo.git.commit("-m", "test commit")
        cb = GitCommitCallback(rpath, strict=True)
        assert cb.git_status.commit_hash == repo.head.object.hexsha
        cb.on_train_start(trainer, None)  # should not raise
