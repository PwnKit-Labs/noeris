"""Noeris: Drop-in training accelerator for HuggingFace transformers.

Usage:
    import noeris
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")
    noeris.patch(model)
    # training proceeds as normal, but faster
"""

__version__ = "0.1.0"

from .patch import patch, unpatch

__all__ = ["patch", "unpatch", "__version__"]
