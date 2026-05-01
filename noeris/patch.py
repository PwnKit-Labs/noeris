"""Main patching API: noeris.patch(model) and noeris.unpatch(model).

Monkey-patches HuggingFace transformer models with fast Triton kernels
for training acceleration.
"""

from __future__ import annotations

from typing import Literal

# Marker attribute to track patched state
_NOERIS_MARKER = "_noeris_patched"
_NOERIS_ORIGINALS = "_noeris_originals"


def patch(
    model,
    *,
    kernels: list[Literal["rmsnorm", "geglu", "cross_entropy"]] | None = None,
    verbose: bool = False,
):
    """Monkey-patch a HuggingFace model with fast Triton kernels.

    Args:
        model: A PreTrainedModel (e.g., from AutoModelForCausalLM.from_pretrained).
        kernels: Which kernels to enable. Default: all available for the architecture.
            Options: "rmsnorm", "geglu" (includes swiglu), "cross_entropy".
        verbose: Print which modules were patched.

    Returns:
        The model (modified in-place). Call noeris.unpatch(model) to restore.
    """
    import torch.nn as nn
    from ._detect import detect_architecture
    from ._modules import NoerisRMSNorm, NoerisGatedActivation

    if getattr(model, _NOERIS_MARKER, False):
        if verbose:
            print("noeris: model already patched, skipping")
        return model

    arch = detect_architecture(model)
    if arch is None:
        raise ValueError(
            f"noeris: unsupported model type "
            f"'{getattr(getattr(model, 'config', None), 'model_type', 'unknown')}'. "
            f"Supported: gemma, gemma2, gemma3, llama, mistral, qwen2, qwen3, phi3, olmo2, falcon"
        )

    if kernels is None:
        kernels = ["rmsnorm", "geglu"]

    originals: dict[str, nn.Module | object] = {}
    patched_count = {"rmsnorm": 0, "mlp": 0}

    # Collect all (parent, name, module) triples before mutating
    modules_to_patch = []
    for name, module in model.named_modules():
        modules_to_patch.append((name, module))

    if "rmsnorm" in kernels:
        _patch_rmsnorm(model, arch, originals, patched_count, verbose)

    if "geglu" in kernels:
        _patch_mlp_activation(model, arch, originals, patched_count, verbose)

    setattr(model, _NOERIS_MARKER, True)
    setattr(model, _NOERIS_ORIGINALS, originals)

    if verbose:
        print(
            f"noeris: patched {patched_count['rmsnorm']} RMSNorm modules, "
            f"{patched_count['mlp']} MLP activations "
            f"(arch={arch.model_type}, affine_mode={arch.affine_mode})"
        )

    return model


def unpatch(model, *, verbose: bool = False):
    """Restore a patched model to its original state.

    Args:
        model: A model previously patched with noeris.patch().
        verbose: Print restoration details.
    """
    if not getattr(model, _NOERIS_MARKER, False):
        if verbose:
            print("noeris: model not patched, nothing to restore")
        return model

    originals = getattr(model, _NOERIS_ORIGINALS, {})
    restored = 0

    for key, original in originals.items():
        if key.startswith("module:"):
            module_path = key[len("module:"):]
            _set_module_by_path(model, module_path, original)
            restored += 1
        elif key.startswith("forward:"):
            module_path = key[len("forward:"):]
            target = _get_module_by_path(model, module_path)
            if target is not None:
                target.forward = original
                restored += 1

    delattr(model, _NOERIS_MARKER)
    delattr(model, _NOERIS_ORIGINALS)

    if verbose:
        print(f"noeris: restored {restored} modules")

    return model


def _patch_rmsnorm(model, arch, originals, counts, verbose):
    """Replace all RMSNorm modules with NoerisRMSNorm."""
    from ._modules import NoerisRMSNorm

    norm_class_names = set(arch.norm_classes)

    for name, module in list(model.named_modules()):
        class_name = type(module).__name__
        if class_name in norm_class_names:
            replacement = NoerisRMSNorm(module, affine_mode=arch.affine_mode)
            originals[f"module:{name}"] = module
            _set_module_by_path(model, name, replacement)
            counts["rmsnorm"] += 1
            if verbose:
                print(f"  noeris: {name} ({class_name}) -> NoerisRMSNorm")


def _patch_mlp_activation(model, arch, originals, counts, verbose):
    """Wrap MLP forward to use fused gated activation."""
    from ._modules import NoerisGatedActivation

    mlp_class_names = set(arch.mlp_classes)
    fused_act = NoerisGatedActivation(activation=arch.activation)

    for name, module in list(model.named_modules()):
        class_name = type(module).__name__
        if class_name not in mlp_class_names:
            continue

        # HuggingFace MLP modules have gate_proj, up_proj, down_proj
        # and an act_fn. We wrap forward to fuse gate * act_fn(up).
        gate_proj = getattr(module, "gate_proj", None)
        up_proj = getattr(module, "up_proj", None)
        down_proj = getattr(module, "down_proj", None)
        if gate_proj is None or up_proj is None or down_proj is None:
            continue

        original_forward = module.forward
        originals[f"forward:{name}"] = original_forward

        # Create patched forward that uses fused activation
        def make_patched_forward(mod, fused):
            def patched_forward(x, **kwargs):
                gate = mod.gate_proj(x)
                up = mod.up_proj(x)
                activated = fused(gate, up)
                return mod.down_proj(activated)
            return patched_forward

        module.forward = make_patched_forward(module, fused_act)
        counts["mlp"] += 1
        if verbose:
            print(f"  noeris: {name} ({class_name}) -> fused {arch.activation}")


def _get_module_by_path(model, path):
    """Get a submodule by dot-separated path."""
    parts = path.split(".")
    current = model
    for part in parts:
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _set_module_by_path(model, path, replacement):
    """Set a submodule by dot-separated path."""
    parts = path.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], replacement)
