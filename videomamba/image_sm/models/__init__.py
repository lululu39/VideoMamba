from .videovit import (
    videovit_tiny,
    videovit_small,
    videovit_middle,
    videovit_base,
)

try:
    from .videomamba import (
        videomamba_tiny,
        videomamba_small,
        videomamba_middle,
        videomamba_base,
    )
    from .videomamba_distill import (
        videomamba_middle_distill,
        videomamba_base_distill,
    )
except ModuleNotFoundError as error:
    if error.name is None or not error.name.startswith("mamba_ssm"):
        raise

from .deit import (
    deit_tiny_patch16_224,
    deit_small_patch16_224,
    deit_base_patch16_224,
)
