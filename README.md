# **pl-git-callback**

This library provides a PyTorch-Lightning callback to increase model reproducibility through enforcing specific git repository states upon training and validation.

## Problem
Reproducibility is key to the scientific approach. For ML, reproducibility is an equally important concern [1], since minimal differences in implementations (e.g. varying random seeds, model initialisation, etc.) can lead to highly divergent benchmark results.

PyTorch-Lightning already includes mechanisms to increase reproducibility, but to our knowledge no mechanism is yet foreseen to ensure that models conform to a certain code base. One is free to change the source code of a model (to a certain extent) without actually breaking existing checkpoints.

## Contribution

This callback is designed to increase reproducibility at source code level. On the one hand, it ensures that the code repository is in a clean state before training and that there are no uncommitted changes. Secondly, it injects commit information into the checkpoints created during training so that you can better track the associated source code revision. Finally, the callback ensures that loaded checkpoints are compatible with the current state of the repository.
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
```
pip install git+https://github.com/cheind/pl-git-callback.git#egg=pl-git-callback[dev]
```

## Operation modes
The callback currently operates in either `strict` or `relaxed`
mode. The difference being that `strict` mode leads to exceptions
when a commit inconsistency is detected, whereas `relaxed` raises
warnings. Hence, in `strict` mode the training stops when for 
example uncommitted changes are detected.

## Logging
This callback injects git commit info into model checkpoints and
writes a `git_info.json` file to trainer's log dir.

To extract GitStatus information from a PyTorch-Lightning
checkpoint file see function `plgitcallback.gitstatus_from_lightning_checkpoint`.

## Open Issues
- PyTorch-Lightning seems to not call `on_load_checkpoint` when only validating/testing a model, hence bypassing the comparison logic of `GitCommitCallback`. See https://github.com/PyTorchLightning/pytorch-lightning/issues/8550

## Testing
Run all tests via
```bash
pytest
```

## References
[1] https://sites.google.com/view/icml-reproducibility-workshop/icml2017
