# Lightning extension: Fairscale

[![CI testing](https://github.com/Lightning-Devel/PL-Fairscale/actions/workflows/ci-testing.yml/badge.svg?event=push)](https://github.com/Lightning-Devel/PL-Fairscale/actions/workflows/ci-testing.yml)
[![General checks](https://github.com/Lightning-Devel/PL-Fairscale/actions/workflows/ci-checks.yml/badge.svg?event=push)](https://github.com/Lightning-Devel/PL-Fairscale/actions/workflows/ci-checks.yml)
[![Documentation Status](https://readthedocs.org/projects/PL-Fairscale/badge/?version=latest)](https://PL-Fairscale.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Lightning-Devel/PL-Fairscale/main.svg?badge_token=mqheL1-cTn-280Vx4cJUdg)](https://results.pre-commit.ci/latest/github/Lightning-Devel/PL-Fairscale/main?badge_token=mqheL1-cTn-280Vx4cJUdg)

\* the Read-The-Docs is failing as this one leads to the public domain which requires the repo to be public too

PyTorch has it's own version of [FSDP](https://pytorch.org/docs/stable/fsdp.html) which is upstreamed from their [fairscale](https://fairscale.readthedocs.io/en/latest/api/nn/fsdp.html) project.
It was introduced in their [v1.11.0 release](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) but it is recommended to use it with PyTorch v1.12 or more and that's what
Lightning supports.

## Auto Wrapping

Model layers should be wrapped in FSDP in a nested way to save peak memory and enable communication and computation overlapping. The
simplest way to do it is auto wrapping, which can serve as a drop-in replacement for DDP without changing the rest of the code. You don't
have to `wrap` layers manually as in the case of manual wrapping.

While initializing the optimizers inside `configure_optimizers` hook, make sure to use `self.trainer.model.parameters()`, else
PyTorch will raise an error. This is required because when you use auto-wrap, the model layers are sharded and your
`lightning_module.parameters()` will return a generator with no params. This inconvenience will be addressed in the future.

```py
from pl_fairscale.strategies import DDPFullyShardedStrategy
from pytorch_lightning import Trainer
from pytorch_lightning.demos.boring_classes import BoringModel

model = BoringModel()
trainer = Trainer(accelerator="gpu", devices=4, strategy=DDPFullyShardedStrategy(), precision=16)
trainer.fit(model)
```

Read more [here](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/#auto-wrapping).

## Manual Wrapping

Manual wrapping can be useful to explore complex sharding strategies by applying `wrap` selectively to some parts of the model. To activate
parameter sharding with manual wrapping, you can wrap your model using the `wrap` function. Internally in Lightning, we enable a context manager around the `configure_sharded_model` function to make sure the `wrap` parameters are passed correctly.

When not using Fully Sharded these wrap functions are a no-op. This means once the changes have been made, there is no need to remove the changes for other strategies.

`wrap` simply wraps the module with a Fully Sharded Parallel class with the correct parameters from the Lightning context manager.

Here's an example using that uses `wrap` to create your model:

```py
import torch
import torch.nn as nn
from pl_fairscale.strategies import DDPFullyShardedStrategy
from pytorch_lightning import Trainer, LightningModule
from torch.distributed.fsdp.wrap import wrap


class MyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(32, 32)
        self.block = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 32))

    def configure_sharded_model(self):
        # modules are sharded across processes
        # as soon as they are wrapped with `wrap`.
        # During the forward/backward passes, weights get synced across processes
        # and de-allocated once computation is complete, saving memory.

        # Wraps the layer in a Fully Sharded Wrapper automatically
        linear_layer = wrap(self.linear_layer)

        for i, layer in enumerate(self.block):
            self.block[i] = wrap(layer)

        self.model = nn.Sequential(linear_layer, nn.ReLU(), self.block)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters())


model = MyModel()
trainer = Trainer(accelerator="gpu", devices=4, strategy=DDPFullyShardedStrategy(), precision=16)
trainer.fit(model)
```

You can customize the strategy configuration by adjusting the arguments of :class:`~pytorch_lightning.strategies.fully_sharded_native.DDPFullyShardedNativeStrategy` and pass that to the `strategy` argument inside the `Trainer`.

```py
from pytorch_lightning import Trainer
from pl_fairscale.strategies import DDPFullyShardedStrategy

native_fsdp = DDPFullyShardedStrategy(cpu_offload=True)
trainer = Trainer(strategy=native_fsdp, accelerator="gpu", devices=4)
```

Check out [this tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) to learn more about the native support.

______________________________________________________________________

## Activation Checkpointing

Activation checkpointing reduces GPU memory usage by avoiding the storage of intermediate activation tensors in
selected layers. The tradeoff is that computation cost for the backpropagation increases, as the dropped activations
need to be recomputed.

Enable checkpointing on large layers (like Transformers) by providing the layer class/type to the strategy:

```py
from pytorch_lightning import Trainer
from pl_fairscale.strategies import DDPFullyShardedStrategy

fsdp = DDPFullyShardedStrategy(
    activation_checkpointing=MyTransformerBlock,  # or pass a list with multiple types
)
trainer = Trainer(strategy=fsdp, accelerator="gpu", devices=4)
```

## Tests / Docs notes

- We are using [Napoleon style,](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) and we shall use static types...
- It is nice to se [doctest](https://docs.python.org/3/library/doctest.html) as they are also generated as examples in documentation
- For wider and edge cases testing use [pytest parametrization](https://docs.pytest.org/en/stable/parametrize.html) :\]
