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
from unittest import mock

import torch
from lightning_utilities.core.rank_zero import rank_zero_only
from pytorch_lightning import Trainer

from lightning_fairscale.strategies import DDPShardedStrategy
from tests.helpers import environment_combinations


@mock.patch("pytorch_lightning.accelerators.cuda.CUDAAccelerator.is_available", return_value=True)
def test_ranks_available_manual_strategy_selection(_):
    """Test that the rank information is readily available after Trainer initialization."""
    num_nodes = 2
    for i, (cluster, variables, expected) in enumerate(environment_combinations()):
        with mock.patch.dict(os.environ, variables):
            strategy = DDPShardedStrategy(
                parallel_devices=[torch.device("cuda", 1), torch.device("cuda", 2)], cluster_environment=cluster
            )
            trainer = Trainer(strategy=strategy, num_nodes=num_nodes)
            assert rank_zero_only.rank == expected["global_rank"]
            assert trainer.global_rank == expected["global_rank"]
            assert trainer.local_rank == expected["local_rank"]
            assert trainer.node_rank == expected["node_rank"]
            assert trainer.world_size == expected["world_size"]
