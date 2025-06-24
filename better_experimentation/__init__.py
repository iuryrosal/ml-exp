# ignore numba warnings
import warnings
import importlib.util

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silencia logs do TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Evita conflitos com OneDNN
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'  # Reduz logs do XLA

from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=DeprecationWarning)


from better_experimentation.better_experimentation import BetterExperimentation


spec_numba = importlib.util.find_spec("numba")
if spec_numba is not None:
    from numba.core.errors import NumbaDeprecationWarning

    warnings.simplefilter("ignore", category=NumbaDeprecationWarning)

__all__ = [
    "pandas_decorator",
    "BetterExperimentation",
    "__version__",
    "compare",
]

