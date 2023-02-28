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
from lightning_fabric.plugins.environments import LightningEnvironment, SLURMEnvironment, TorchElasticEnvironment


def environment_combinations():
    expected = {"global_rank": 3, "local_rank": 1, "node_rank": 1, "world_size": 4}
    # Lightning
    variables = {"CUDA_VISIBLE_DEVICES": "0,1,2,4", "LOCAL_RANK": "1", "NODE_RANK": "1", "WORLD_SIZE": "8"}
    environment = LightningEnvironment()
    yield environment, variables, expected
    # SLURM
    variables = {
        "CUDA_VISIBLE_DEVICES": "0,1,2,4",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_LOCALID": "1",
        "SLURM_NODEID": "1",
        "SLURM_PROCID": "3",
        "SLURM_NTASKS": "4",
        "SLURM_NTASKS_PER_NODE": "2",
    }
    environment = SLURMEnvironment()
    yield environment, variables, expected
    # TorchElastic
    variables = {
        "CUDA_VISIBLE_DEVICES": "0,1,2,4",
        "LOCAL_RANK": "1",
        "GROUP_RANK": "1",
        "RANK": "3",
        "WORLD_SIZE": "4",
        "LOCAL_WORLD_SIZE": "2",
        "TORCHELASTIC_RUN_ID": "1",
    }
    environment = TorchElasticEnvironment()
    yield environment, variables, expected
