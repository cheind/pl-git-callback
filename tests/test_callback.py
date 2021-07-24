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
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from plgitcallback import (
    GitCommitCallback,
    GitCommitCallbackError,
    GitCommitCallbackWarning,
    GitStatus,
)
import tempfile
import json
import time


import warnings


class LinearRegressionModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(1, 1, bias=True)
        self.w.weight.data.fill_(1.7)
        self.w.bias.data.fill_(0.7)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.w(x)  # y = k*x + d

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-1)

    def _step(self, batch):
        x, y = batch
        yhat = self(x)
        return F.mse_loss(yhat, y, reduction="sum")


xs = torch.arange(-10, 10, 1.0).view(-1, 1)
ys = xs * 2 + 1
ds_train = torch.utils.data.TensorDataset(xs[:15], ys[:15])
ds_val = torch.utils.data.TensorDataset(xs[15:], ys[15:])


@pytest.fixture
def tmp_git_repo():
    """Fixture that yields a git repository initialized in a temporary directory"""

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        repo = git.Repo.init(Path(tmp))
        yield repo, tmp
        repo.close()


def update_file(
    repo: git.Repo, dirpath: Path, fname: str, stage: bool = True, commit: bool = False
):
    with open(dirpath / fname, "w") as f:
        f.write(str(uuid.uuid4()))
    if stage:
        repo.index.add(fname)
    if commit:
        repo.git.commit("-m", "test commit")


def test_gitstatus(tmp_git_repo):
    repo, rpath = tmp_git_repo
    status = GitStatus.get_status(rpath)
    assert status.valid
    assert status.empty
    assert status.dirty is None
    assert status.commit_hash is None
    assert status.branch_name is None

    update_file(repo, rpath, "test.txt", stage=False, commit=False)
    status = GitStatus.get_status(rpath)
    assert status.valid
    assert status.empty
    assert status.dirty is None
    assert status.branch_name is None
    assert status.num_untracked_files == 1

    update_file(repo, rpath, "test.txt", stage=True, commit=False)
    status = GitStatus.get_status(rpath)
    assert status.valid
    assert status.empty
    assert status.dirty is None
    assert status.branch_name is None
    assert status.num_untracked_files == 0

    # Commit file
    update_file(repo, rpath, "test.txt", stage=True, commit=True)
    status = GitStatus.get_status(rpath)
    assert status.valid
    assert not status.empty
    assert not status.dirty
    assert status.branch_name == "master"
    assert status.num_untracked_files == 0

    update_file(repo, rpath, "test.txt", stage=False, commit=False)
    status = GitStatus.get_status(rpath)
    assert status.valid
    assert not status.empty
    assert status.dirty
    assert status.branch_name == "master"
    assert status.num_untracked_files == 0

    import time

    t = time.time()
    update_file(repo, rpath, "test.txt", stage=True, commit=True)
    status = GitStatus.get_status(rpath)
    assert status.valid
    assert not status.empty
    assert not status.dirty
    assert status.branch_name == "master"
    assert status.num_untracked_files == 0
    assert abs(t - status.commit_date) < 1.0


def test_gitcommitcallback_fitstart(tmp_git_repo, tmp_path):
    repo, rpath = tmp_git_repo

    class MyTrainer:
        def __init__(self, log_dir):
            self.log_dir = log_dir

    trainer = MyTrainer(tmp_path)

    # Repo at this point is not clean
    cb = GitCommitCallback(rpath, mode="relaxed")
    with pytest.warns(GitCommitCallbackWarning):
        cb.on_fit_start(trainer, None)
        assert (Path(tmp_path) / "git-status.json").is_file()

    cb = GitCommitCallback(rpath, mode="strict")
    with pytest.raises(GitCommitCallbackError):
        cb.on_fit_start(trainer, None)

    update_file(repo, rpath, "test.txt", stage=True, commit=False)
    cb = GitCommitCallback(rpath, mode="strict")
    with pytest.raises(GitCommitCallbackError):
        cb.on_fit_start(trainer, None)

    update_file(repo, rpath, "test.txt", stage=True, commit=True)
    cb = GitCommitCallback(rpath, mode="strict")
    cb.on_fit_start(trainer, None)
    assert Path(trainer.log_dir / "git-status.json").is_file()
    data = json.load(open(trainer.log_dir / "git-status.json"))
    assert data["branch_name"] == "master"
    assert data["commit_hash"] == repo.head.object.hexsha
    assert abs(data["commit_date"] - time.time()) < 1
    assert data["valid"]
    assert not data["empty"]


def test_gitcommitcallback_saveckpt(tmp_git_repo, tmp_path):
    warnings.filterwarnings("ignore")
    repo, rpath = tmp_git_repo
    update_file(repo, rpath, "test.txt", stage=True, commit=True)

    model = LinearRegressionModel()
    cb = GitCommitCallback(git_dir=rpath, mode="strict")
    cp = ModelCheckpoint(filename="mymodel", mode="min", monitor="val_loss")
    trainer = pl.Trainer(max_epochs=1, callbacks=[cb, cp])
    trainer.fit(model, DataLoader(ds_train), DataLoader(ds_val))
    ckpt = torch.load(Path(trainer.log_dir) / "checkpoints" / "mymodel.ckpt")
    assert GitCommitCallback in ckpt["callbacks"]
    status_saved = GitStatus(**ckpt["callbacks"][GitCommitCallback])
    assert status_saved == cb.git_status


def test_gitcommitcallback_loadckpt(tmp_git_repo, tmp_path):
    warnings.filterwarnings("ignore")
    repo, rpath = tmp_git_repo
    update_file(repo, rpath, "test.txt", stage=True, commit=True)

    model = LinearRegressionModel()
    cb = GitCommitCallback(git_dir=rpath, mode="strict")
    cp = ModelCheckpoint(filename="mymodel", mode="min", monitor="val_loss")
    trainer = pl.Trainer(max_epochs=1, callbacks=[cb, cp])
    trainer.fit(model, DataLoader(ds_train), DataLoader(ds_val))
    ckpt_file = Path(trainer.log_dir) / "checkpoints" / "mymodel.ckpt"

    # later
    cb = GitCommitCallback(git_dir=rpath, mode="strict")
    trainer = pl.Trainer(
        max_epochs=1,
        callbacks=[cb, cp],
        resume_from_checkpoint=ckpt_file,
    )
    trainer.fit(model, DataLoader(ds_train), DataLoader(ds_val))

    # later with changes
    update_file(repo, rpath, "test.txt", stage=True, commit=True)
    cb = GitCommitCallback(git_dir=rpath, mode="strict")
    trainer = pl.Trainer(
        max_epochs=1,
        callbacks=[cb, cp],
        resume_from_checkpoint=ckpt_file,
    )
    with pytest.raises(GitCommitCallbackError):
        trainer.fit(model, DataLoader(ds_train), DataLoader(ds_val))

    cb = GitCommitCallback(git_dir=rpath, mode="relaxed")
    trainer = pl.Trainer(
        max_epochs=1,
        callbacks=[cb, cp],
        resume_from_checkpoint=ckpt_file,
    )
    with pytest.warns(GitCommitCallbackWarning):
        trainer.fit(model, DataLoader(ds_train), DataLoader(ds_val))

    # The following does not call the callback methods
    cb = GitCommitCallback(git_dir=rpath, mode="strict")
    trainer = pl.Trainer(
        max_epochs=1,
        callbacks=[cb, cp],
    )
    trainer.validate(model, val_dataloaders=[DataLoader(ds_val)], ckpt_path=ckpt_file)
