__version__ = "0.1.0dev"
__author__ = "Lightning-AI et al."
__author_email__ = "name@lightning.ai"
__license__ = "Apache-2.0"
__copyright__ = f"Copyright (c) 2021-2023, {__author__}."
__homepage__ = "https://github.com/Lightning-Devel/PL-Fairscale"
__docs__ = "PyTorch Lightning Strategy for Fairscale."
# todo: consider loading Readme here...
__long_doc__ = """
Lightning Fairscale
-------------------

When training large models, fitting larger batch sizes, or trying to increase throughput using multi-GPU compute,
 Lightning provides advanced optimized distributed training strategies to support these cases and offer substantial
 improvements in memory usage.

In many cases these strategies are some flavour of model parallelism however we only introduce concepts
 at a high level to get you started. Refer to the FairScale documentation for more information about model parallelism.

Note that some of the extreme memory saving configurations will affect the speed of training.
 This Speed/Memory trade-off in most cases can be adjusted.

Some of these memory-efficient strategies rely on offloading onto other forms of memory, such as CPU RAM or NVMe.
 This means you can even see memory benefits on a single GPU, using a strategy such as DeepSpeed ZeRO Stage 3 Offload.
"""

__all__ = [
    "__author__",
    "__author_email__",
    "__copyright__",
    "__docs__",
    "__long_doc__",
    "__homepage__",
    "__license__",
    "__version__",
]
