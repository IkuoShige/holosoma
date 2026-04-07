from typing import Any, Union

import numpy as np
import numpy.typing as npt
import torch

# Holosoma vendoring patch: jax is an optional FlashSAC dependency only used by
# the genesis / mujoco_playground env wrappers. The hssim conda env does not
# install jax, so we make the import optional and fall back to a synthetic
# placeholder type so that ``Tensor`` remains usable for type hints throughout
# the FlashSAC torch path (which is what we actually exercise).
try:  # pragma: no cover - exercised only when jax is installed
    import jax.numpy as jnp  # type: ignore[import-not-found]

    _JaxArray = jnp.ndarray
except ModuleNotFoundError:  # pragma: no cover - default in hssim env

    class _JaxArrayPlaceholder:  # noqa: D401
        """Placeholder for ``jax.numpy.ndarray`` when jax is not installed."""

    _JaxArray = _JaxArrayPlaceholder

NDArray = npt.NDArray[Any]
F32NDArray = npt.NDArray[np.float32]
Tensor = Union[NDArray, _JaxArray, torch.Tensor]
