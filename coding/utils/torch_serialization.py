import sys
import numpy as np
import torch
def _patch_numpy_pickle_compat():
    if 'numpy._core' not in sys.modules:
        sys.modules['numpy._core'] = np.core
    submodules = {
        'numpy._core.multiarray': getattr(np.core, 'multiarray', None),
        'numpy._core.umath': getattr(np.core, 'umath', None),
        'numpy._core.numerictypes': getattr(np.core, 'numerictypes', None),
        'numpy._core._dtype': getattr(np.core, '_dtype', None),
    }
    for name, module in submodules.items():
        if module is not None and name not in sys.modules:
            sys.modules[name] = module
def torch_safe_load(*args, **kwargs):
    _patch_numpy_pickle_compat()
    return torch.load(*args, **kwargs)