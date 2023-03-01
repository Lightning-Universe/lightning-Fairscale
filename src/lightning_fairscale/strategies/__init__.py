"""Lightning strategies."""

from lightning_fairscale.strategies.fully_sharded import DDPFullyShardedStrategy  # noqa: F401
from lightning_fairscale.strategies.sharded import DDPShardedStrategy  # noqa: F401
from lightning_fairscale.strategies.sharded_spawn import DDPSpawnShardedStrategy  # noqa: F401
