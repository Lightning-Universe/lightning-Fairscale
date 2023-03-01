# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from copy import deepcopy
from typing import Mapping
from unittest import mock
from unittest.mock import Mock

import pytest
import torch
from fairscale.nn.data_parallel.sharded_ddp import ShardedDataParallel
from fairscale.optim import OSS
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.demos.boring_classes import BoringModel
from pytorch_lightning.plugins import MixedPrecisionPlugin
from pytorch_lightning.trainer.states import TrainerFn
from torch import Tensor

from lightning_fairscale.strategies import DDPShardedStrategy, DDPSpawnShardedStrategy


class ModelWithAdamOptimizer(BoringModel):
    def configure_optimizers(self):
        return torch.optim.Adam(self.layer.parameters(), lr=0.1)


class CheckModelRestore(ModelWithAdamOptimizer):
    def __init__(self, old_model_state_dict, old_optimizer_states):
        super().__init__()
        self.old_model_state_dict = old_model_state_dict
        self.old_optimizer_states = old_optimizer_states

    def on_train_start(self):
        assert all(
            self._is_equal(actual, expected) for actual, expected in zip(self.state_dict(), self.old_model_state_dict)
        )

        for optimizer, state in zip(self.trainer.optimizers, self.old_optimizer_states):
            optimizer_state = self.trainer.strategy.optimizer_state(optimizer)
            self._is_equal(optimizer_state, state)

    def _is_equal(self, a, b):
        if isinstance(a, Tensor):
            return torch.allclose(a, b)

        if isinstance(a, Mapping):
            return all(self._is_equal(a.get(k, None), b.get(k, None)) for k in b)

        return a == b


@pytest.mark.xfail(AssertionError, reason="Expected 'clip_grad_norm' to have been called.")  # todo
@pytest.mark.parametrize("clip_val", [0, 10])
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This test needs at least single GPU.")
@mock.patch("fairscale.optim.oss.OSS.clip_grad_norm")
def test_ddp_sharded_precision_16_clip_gradients(mock_oss_clip_grad_norm, clip_val, tmpdir):
    """Ensure that clip gradients is only called if the value is greater than 0."""
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        strategy=DDPSpawnShardedStrategy(),
        accelerator="gpu",
        devices=1,
        precision=16,
        fast_dev_run=True,
        gradient_clip_val=clip_val,
    )
    trainer.fit(model)
    if clip_val > 0:
        mock_oss_clip_grad_norm.assert_called()
    else:
        mock_oss_clip_grad_norm.assert_not_called()


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This test needs at least single GPU.")
@pytest.mark.parametrize("strategy", [DDPShardedStrategy, DDPSpawnShardedStrategy])
def test_ddp_choice_sharded_amp(strategy):
    """Test to ensure that plugin native amp plugin is correctly chosen when using sharded."""
    trainer = Trainer(fast_dev_run=True, accelerator="gpu", devices=1, precision=16, strategy=strategy())
    assert isinstance(trainer.precision_plugin, MixedPrecisionPlugin)


def test_ddp_sharded_strategy_checkpoint_cpu(tmpdir):
    """Test to ensure that checkpoint is saved correctly."""
    model = BoringModel()
    trainer = Trainer(strategy=DDPSpawnShardedStrategy(), accelerator="cpu", devices=2, fast_dev_run=True)

    trainer.fit(model)

    checkpoint_path = os.path.join(tmpdir, "model.pt")
    trainer.save_checkpoint(checkpoint_path)
    saved_model = BoringModel.load_from_checkpoint(checkpoint_path)

    # Assert model parameters are identical after loading
    for trained_param, loaded_param in zip(model.parameters(), saved_model.parameters()):
        assert torch.equal(trained_param.to("cpu"), loaded_param)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="This test needs at least 2 GPUs.")
def test_ddp_sharded_strategy_checkpoint_multi_gpu(tmpdir):
    """Test to ensure that checkpoint is saved correctly when using multiple GPUs."""
    model = BoringModel()
    trainer = Trainer(accelerator="gpu", devices=2, strategy=DDPSpawnShardedStrategy(), fast_dev_run=True)

    trainer.fit(model)

    checkpoint_path = os.path.join(tmpdir, "model.pt")
    trainer.save_checkpoint(checkpoint_path)
    saved_model = BoringModel.load_from_checkpoint(checkpoint_path)

    # Assert model parameters are identical after loading
    for trained_param, loaded_param in zip(model.parameters(), saved_model.parameters()):
        assert torch.equal(trained_param.to("cpu"), loaded_param)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="This test needs at least 2 GPUs.")
def test_ddp_sharded_strategy_finetune(tmpdir):
    """Test to ensure that we can save and restart training (simulate fine-tuning)."""
    model = BoringModel()
    trainer = Trainer(accelerator="gpu", devices=2, strategy=DDPSpawnShardedStrategy(), fast_dev_run=True)
    trainer.fit(model)

    checkpoint_path = os.path.join(tmpdir, "model.pt")
    trainer.save_checkpoint(checkpoint_path)
    saved_model = BoringModel.load_from_checkpoint(checkpoint_path)

    trainer = Trainer(fast_dev_run=True)
    trainer.fit(saved_model)


def test_ddp_sharded_strategy_fit_ckpt_path(tmpdir):
    """Test to ensure that resuming from checkpoint works."""
    model = BoringModel()
    trainer = Trainer(strategy=DDPSpawnShardedStrategy(), accelerator="cpu", devices=2, fast_dev_run=True)

    trainer.fit(model)

    checkpoint_path = os.path.join(tmpdir, "model.pt")
    trainer.save_checkpoint(checkpoint_path)

    model = BoringModel()

    trainer = Trainer(strategy=DDPSpawnShardedStrategy(), accelerator="cpu", devices=2, fast_dev_run=True)

    trainer.fit(model, ckpt_path=checkpoint_path)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This test needs at least single GPU.")
def test_ddp_sharded_strategy_fit_ckpt_path_gpu_to_cpu(tmpdir):
    """Test to ensure that resuming from checkpoint works when going from GPUs- > CPU."""
    model = BoringModel()
    trainer = Trainer(strategy=DDPSpawnShardedStrategy(), accelerator="gpu", devices=1, fast_dev_run=True)

    trainer.fit(model)

    checkpoint_path = os.path.join(tmpdir, "model.pt")
    trainer.save_checkpoint(checkpoint_path)

    model = BoringModel()

    trainer = Trainer(strategy=DDPSpawnShardedStrategy(), accelerator="cpu", devices=2, fast_dev_run=True)

    trainer.fit(model, ckpt_path=checkpoint_path)


@pytest.mark.skip(reason="This test is meant to be standalone.")  # todo
@pytest.mark.parametrize(
    "trainer_kwargs",
    [
        {"accelerator": "cpu", "devices": 2},
        pytest.param(
            {"accelerator": "gpu", "devices": 2},
            marks=pytest.mark.skipif(torch.cuda.device_count() < 2, reason="multi-GPU"),
        ),
    ],
)
def test_ddp_sharded_strategy_test_multigpu(trainer_kwargs):
    """Test to ensure we can use validate and test without fit."""
    model = BoringModel()
    trainer = Trainer(
        strategy=DDPSpawnShardedStrategy(),
        fast_dev_run=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        **trainer_kwargs,
    )

    trainer.validate(model)
    trainer.test(model)


class BoringModelSharded(BoringModel):
    def on_train_start(self) -> None:
        """Check if trainer module is wrapped as ShardedDataParallel during training stage."""
        assert isinstance(self.trainer.model, ShardedDataParallel)

    def on_test_start(self) -> None:
        """Check if trainer module remains as LightningModule during test stage."""
        assert isinstance(self.trainer.model, LightningModule)

    def on_validation_start(self) -> None:
        """Check if trainer module remains as LightningModule during test stage."""
        if self.trainer.state.fn == TrainerFn.FITTING:
            assert isinstance(self.trainer.model, ShardedDataParallel)
        else:
            assert isinstance(self.trainer.model, LightningModule)

    def on_predict_start(self) -> None:
        """Check if trainer module remains as LightningModule during prediction stage."""
        assert isinstance(self.trainer.model, LightningModule)


def test_configure_ddp(tmpdir):
    """Tests with ddp sharded strategy."""
    trainer = Trainer(default_root_dir=tmpdir, strategy=DDPSpawnShardedStrategy(), fast_dev_run=True)

    model = BoringModelSharded()

    trainer.fit(model)
    trainer.test(model, dataloaders=model.test_dataloader())
    trainer.validate(model, dataloaders=model.val_dataloader())
    trainer.predict(model, dataloaders=model.predict_dataloader())


@mock.patch("lightning_fairscale.strategies.DDPShardedStrategy._wrap_optimizers", autospec=True)
@pytest.mark.parametrize("cls", [DDPShardedStrategy, DDPSpawnShardedStrategy])
def test_custom_kwargs_sharded(_, cls):
    """Tests to ensure that if custom kwargs are passed, they are set correctly."""
    strategy = cls(reduce_fp16=True)
    strategy._lightning_module = Mock(spec=LightningModule)
    strategy._lightning_module.trainer = Mock()
    strategy.parallel_devices = [Mock()]
    class_name = "sharded" if isinstance(strategy, DDPShardedStrategy) else "sharded_spawn"

    with mock.patch(f"lightning_fairscale.strategies.{class_name}.ShardedDataParallel", autospec=True) as mock_sharded:
        strategy.configure_ddp()
    args, kwargs = mock_sharded.call_args
    assert "reduce_fp16" in kwargs
    assert kwargs["reduce_fp16"]


@mock.patch("lightning_fairscale.strategies.DDPShardedStrategy._wrap_optimizers", autospec=True)
@pytest.mark.parametrize(("params", "expected_buffer_size"), [({}, 0), ({"reduce_buffer_size": 128}, 128)])
@pytest.mark.parametrize("num_nodes", [1, 2])
def test_custom_kwargs_sharded_reduce_buffer_size(_, params, expected_buffer_size, num_nodes):
    """Tests to ensure that ``reduce_buffer_size`` is correctly set based on user kwargs."""
    strategy = DDPShardedStrategy(**params)
    strategy.num_nodes = num_nodes
    strategy._lightning_module = Mock(spec=LightningModule)
    strategy._lightning_module.trainer = Mock()
    strategy.parallel_devices = [Mock()]

    with mock.patch("lightning_fairscale.strategies.sharded.ShardedDataParallel", autospec=True) as mock_sharded:
        strategy.configure_ddp()
    args, kwargs = mock_sharded.call_args
    assert "reduce_buffer_size" in kwargs

    if num_nodes > 1 and len(params) == 0:
        # If user has not specified a buffer size and we're using multiple nodes, check to see if default is set
        assert kwargs["reduce_buffer_size"] == DDPShardedStrategy._REDUCE_BUFFER_SIZE_DEFAULT
    else:
        assert kwargs["reduce_buffer_size"] == expected_buffer_size


def test_block_backward_sync():
    strategy = DDPShardedStrategy()
    model = mock.MagicMock(spec=ShardedDataParallel)
    with mock.patch.object(strategy, "_model", model), strategy.block_backward_sync():
        pass
    model.no_sync.assert_called_once()


@pytest.mark.parametrize(
    ("strategy_name", "expected_ddp_kwargs"),
    [
        ("ddp_sharded", {}),
        ("ddp_sharded_find_unused_parameters_false", {"find_unused_parameters": False}),
        ("ddp_sharded_spawn", {}),
        ("ddp_sharded_spawn_find_unused_parameters_false", {"find_unused_parameters": False}),
    ],
)
def test_ddp_kwargs_from_registry(strategy_name, expected_ddp_kwargs):
    trainer = Trainer(strategy=strategy_name)
    assert trainer.strategy._ddp_kwargs == expected_ddp_kwargs


class BoringFairScaleOptimizerModel(BoringModel):
    def configure_optimizers(self):
        base_optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        return OSS(params=base_optimizer.param_groups, optim=type(base_optimizer), **base_optimizer.defaults)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="This test needs at least 2 GPUs.")
def test_ddp_sharded_strategy_fit_ckpt_path_downsize_gpus(tmpdir):
    model = ModelWithAdamOptimizer()
    trainer = Trainer(
        strategy=DDPSpawnShardedStrategy(),
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        accelerator="gpu",
        devices=2,
    )
    trainer.fit(model)

    checkpoint_path = trainer.checkpoint_callback.best_model_path
    ckpt = torch.load(checkpoint_path)
    old_model_state_dict = deepcopy(ckpt["state_dict"])
    old_optimizer_states = deepcopy(ckpt["optimizer_states"])

    model = CheckModelRestore(old_model_state_dict, old_optimizer_states)
    trainer = Trainer(
        strategy=DDPSpawnShardedStrategy(),
        max_epochs=2,
        limit_train_batches=1,
        limit_val_batches=0,
        accelerator="gpu",
        devices=1,
    )
    trainer.fit(model, ckpt_path=checkpoint_path)
