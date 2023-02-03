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
import pytest
import torch
from pytorch_lightning import Trainer

from pl_fairscale.strategies import DDPShardedStrategy, DDPSpawnShardedStrategy


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="This test needs at least 2 GPUs.")
@pytest.mark.parametrize(
    ("strategy", "strategy_class"),
    [
        ("ddp_sharded", DDPShardedStrategy),
        ("ddp_sharded_spawn", DDPSpawnShardedStrategy),
    ],
)
def test_strategy_choice_gpu_str(strategy, strategy_class):
    with pytest.deprecated_call(match="FairScale has been deprecated in v1.9.0"):
        trainer = Trainer(strategy=strategy, accelerator="gpu", devices=2)
    assert isinstance(trainer.strategy, strategy_class)
