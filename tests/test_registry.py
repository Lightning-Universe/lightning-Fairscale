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
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import StrategyRegistry

from pl_fairscale.strategies import DDPFullyShardedStrategy, DDPShardedStrategy, DDPSpawnShardedStrategy


def test_fsdp_strategy_registry(strategy: str = "fsdp"):
    assert strategy in StrategyRegistry
    assert StrategyRegistry[strategy]["strategy"] == DDPFullyShardedStrategy

    trainer = Trainer(strategy=strategy)

    assert isinstance(trainer.strategy, DDPFullyShardedStrategy)


@pytest.mark.parametrize(
    ("strategy_name", "strategy", "expected_init_params"),
    [
        ("ddp_sharded_spawn_find_unused_parameters_false", DDPSpawnShardedStrategy, {"find_unused_parameters": False}),
        ("ddp_sharded_find_unused_parameters_false", DDPShardedStrategy, {"find_unused_parameters": False}),
    ],
)
def test_ddp_find_unused_parameters_strategy_registry(tmpdir, strategy_name, strategy, expected_init_params):
    trainer = Trainer(default_root_dir=tmpdir, strategy=strategy_name)
    assert isinstance(trainer.strategy, strategy)
    assert strategy_name in StrategyRegistry
    assert StrategyRegistry[strategy_name]["init_params"] == expected_init_params
    assert StrategyRegistry[strategy_name]["strategy"] == strategy
