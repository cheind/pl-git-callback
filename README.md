# **pl-git-callback**

This library provides a PyTorch-Lightning callback to increase model reproducibility through enforcing specific git repository states upon training and validation.

Reproducibility is key to the scientific approach. For ML reproducibility is a major concern [1], since tiny details of implementation differences (random seeds, model initialization, etc...) may lead to hugely diverging benchmark results.

PyTorch-Lightning already incorporates mechanisms to increase reproducibility, but (to the best of our knowledge) no mechanism is provided to ensure models have a particular code basis. One might happily change the source code of a model (to some extend), without having existing checkpoints break.

This callback is meant to increase reproducibility on a source code level. For one, it ensures that before training the code repository is in a clean state and no uncommitted changes are present. Second, it injects commit information to checkpoints generated in training, so you can better track associated source code. And finally, the callback ensures that loaded checkpoints are compatible with the current repository state.

## Usage

```python
from plgitcallback import GitCommitCallback

# The following will ensure a clean git repo before
# training starts and also inject repo information to 
# checkpoints.
cb = GitCommitCallback(git_dir='.', mode="strict")

# Default PyTorch-Lightning pipeline
model = MyModel()
cp = ModelCheckpoint(filename="mymodel", mode="min", monitor="val_loss")
trainer = pl.Trainer(max_epochs=1, callbacks=[cb, cp])
trainer.fit(model, DataLoader(ds_train), DataLoader(ds_val))
```

## Install 
```bash
pip install git+https://github.com/cheind/pl-git-callback.git#egg=pl-git-callback[dev]
```

## Operation modes
The callback currently operates in either `strict` or `relaxed`
mode. The difference being that `strict` mode leads to exceptions
when a commit inconsistency is detected whereas `relaxed` causes
only warnings. Hence, in `strict` mode the training usually is
forcefully cancelled when for example uncommitted changes are
detected.

## Logging
This callback injects git commit info into model checkpoints and
writes a `git_info.json` file to trainer's log dir.

To extract GitStatus information from a PyTorch-Lightning
checkpoint file see function `plgitcallback.gitstatus_from_lightning_checkpoint`.

## Open Issues
- PyTorch-Lightning seems to not call `on_load_checkpoint` when only validating/testing a model, hence bypassing the comparison logic of `GitCommitCallback`. See https://github.com/PyTorchLightning/pytorch-lightning/issues/8550

## References
[1] https://sites.google.com/view/icml-reproducibility-workshop/icml2017
