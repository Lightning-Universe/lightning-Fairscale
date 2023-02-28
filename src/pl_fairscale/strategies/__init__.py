"""Lightning strategies."""

from pl_fairscale.strategies.fully_sharded import DDPFullyShardedStrategy  # noqa: F401
from pl_fairscale.strategies.sharded import DDPShardedStrategy  # noqa: F401
from pl_fairscale.strategies.sharded_spawn import DDPSpawnShardedStrategy  # noqa: F401
