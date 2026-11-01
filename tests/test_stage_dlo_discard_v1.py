"""Frozen CPU lifecycle checks for the opt-in terminal stage adapter."""
import gc
import weakref
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from tests.diffusion.offloader.helpers import patch_offload_runtime
from h3_omni_adapter import stage_dlo
from vllm_omni.diffusion.model_loader.host_weight_plan import HostWeightPlan, TensorBinding
from vllm_omni.diffusion.offloader.base import OffloadConfig
from vllm_omni.diffusion.offloader.config import OffloadStrategy

pytestmark = [pytest.mark.cpu]


class Blocks(nn.Module):
    _layerwise_offload_blocks_attrs = ["blocks"]

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(3, 3, bias=False) for _ in range(2)])
        for block in self.blocks:
            block.register_buffer("state", torch.arange(3, dtype=torch.float32))

    def forward(self, value):
        for block in self.blocks:
            value = block(value)
        return value


@pytest.fixture
def make_backend(tmp_path, monkeypatch):
    native = stage_dlo.native
    patch_offload_runtime(monkeypatch, native.current_omni_platform, synchronize=True)
    monkeypatch.setattr(stage_dlo, "get_tensor_model_parallel_world_size", lambda: 1)
    created = []

    def build():
        folder = tmp_path / str(len(created))
        folder.mkdir()
        pipeline = nn.Module()
        pipeline.transformer = Blocks()
        pipeline._dit_modules = ["transformer"]
        pipeline._encoder_modules = []
        pipeline._vae_modules = []
        pipeline._resident_modules = []
        expected = {k: v.detach().clone() for k, v in pipeline.state_dict().items()}
        file = folder / "weights.safetensors"
        save_file(expected, file)
        plan = HostWeightPlan("checkpoint_mmap",
                              {name: TensorBinding(name, str(file)) for name in expected},
                              frozenset({"transformer."}))
        config = OffloadConfig(strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,
                               pin_cpu_memory=False, dp_size=1, dlo_use_allgather=False,
                               components=frozenset({"dit"}))
        backend = stage_dlo.StageDLOBackend(config, torch.device("cpu"), host_weight_plan=plan)
        backend.enable(pipeline)
        created.append(backend)
        return backend, pipeline, expected, file
    return build


@pytest.mark.parametrize("method", ["disable", "shutdown"])
def test_default_restores_exact_weights(make_backend, method):
    backend, pipeline, expected, _ = make_backend()
    getattr(backend, method)()
    assert not backend.enabled
    assert not backend._all_hook_groups
    for name, value in pipeline.state_dict().items():
        assert value.device.type == "cpu"
        assert torch.equal(value, expected[name])
    assert backend._poisoned_reason is None


def test_discard_no_restore_no_cpu_copy_and_no_reuse(make_backend, monkeypatch):
    backend, pipeline, _, file = make_backend()
    hooks = [h for group in backend._all_hook_groups for h in group]
    source_refs = [weakref.ref(info["tensor"]) for h in hooks for values in h.cpu_sources.values()
                   for info in values]
    slot_refs = [weakref.ref(v) for slot in hooks[0].cpu_staging_buffers for v in slot.values()]
    restore = Mock(side_effect=AssertionError("must not restore CPU weights"))
    monkeypatch.setattr(stage_dlo.native, "restore_tensor_storage", restore)
    allocations = Mock(wraps=torch.empty_strided)
    monkeypatch.setattr(torch, "empty_strided", allocations)
    backend.shutdown(discard_stage=True)
    assert not backend.enabled and not backend._all_hook_groups
    assert all(v.numel() == 0 for v in pipeline.transformer.parameters())
    assert all(v.numel() == 0 for v in pipeline.transformer.buffers())
    restore.assert_not_called()
    allocations.assert_not_called()
    with pytest.raises(RuntimeError, match="discard"):
        backend.enable(pipeline)
    with pytest.raises(RuntimeError, match="discard"):
        pipeline.transformer(torch.ones(1, 3))
    backend.shutdown(discard_stage=True)
    backend.shutdown()
    backend.disable()
    gc.collect()
    assert all(r() is None for r in source_refs + slot_refs)
    assert str(file) not in Path("/proc/self/maps").read_text()


def test_sync_failure_keeps_sources_and_retry(make_backend, monkeypatch):
    backend, pipeline, _, file = make_backend()
    hooks = [h for group in backend._all_hook_groups for h in group]
    sources = [info["tensor"] for h in hooks for values in h.cpu_sources.values() for info in values]
    with monkeypatch.context() as patch:
        patch.setattr(stage_dlo.native.current_omni_platform, "synchronize",
                      Mock(side_effect=RuntimeError("injected sync failure")))
        with pytest.raises(RuntimeError, match="injected sync"):
            backend.shutdown(discard_stage=True)
    assert backend.enabled and backend._all_hook_groups
    assert not backend._stage_discard_started
    assert backend._poisoned_reason is None
    assert all(v.numel() for v in sources)
    assert str(file) in Path("/proc/self/maps").read_text()
    del sources, hooks
    backend.shutdown(discard_stage=True)
    assert not backend.enabled


def test_partial_hook_failure_preserves_maps_and_cleanup_retry(make_backend, monkeypatch):
    backend, pipeline, _, file = make_backend()
    original = stage_dlo.native.remove_distributed_block_hook
    calls = 0

    def fail_second(block):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected hook removal failure")
        original(block)

    with monkeypatch.context() as patch:
        patch.setattr(stage_dlo.native, "remove_distributed_block_hook", fail_second)
        with pytest.raises(RuntimeError, match="injected hook"):
            backend.shutdown(discard_stage=True)
    assert backend.enabled and backend._all_hook_groups
    assert str(file) in Path("/proc/self/maps").read_text()
    with pytest.raises(RuntimeError, match="discard"):
        backend.shutdown()
    with pytest.raises(RuntimeError, match="discard"):
        backend.disable()
    with pytest.raises(RuntimeError, match="discard"):
        backend.enable(pipeline)
    with pytest.raises(RuntimeError, match="discard"):
        pipeline.transformer(torch.ones(1, 3))
    backend.shutdown(discard_stage=True)
    gc.collect()
    assert not backend.enabled
    assert str(file) not in Path("/proc/self/maps").read_text()


def test_handle_release_failure_is_retryable(make_backend, monkeypatch):
    backend, _, _, file = make_backend()
    with monkeypatch.context() as patch:
        patch.setattr(backend, "_release_mmap_handles",
                      Mock(side_effect=RuntimeError("injected handle failure")))
        with pytest.raises(RuntimeError, match="injected handle"):
            backend.shutdown(discard_stage=True)
    assert backend._all_hook_groups
    assert str(file) in Path("/proc/self/maps").read_text()
    backend.shutdown(discard_stage=True)
    gc.collect()
    assert str(file) not in Path("/proc/self/maps").read_text()


@pytest.mark.parametrize("unsupported", ["tp", "dp", "encoder", "resident", "hwr", "ordinary", "registered"])
def test_unsupported_rejected_before_mutation(make_backend, monkeypatch, unsupported):
    backend, pipeline, _, _ = make_backend()
    groups = backend._all_hook_groups
    before = [dict(block._hook_registry._hooks) for block in pipeline.transformer.blocks]
    with monkeypatch.context() as patch:
        if unsupported == "tp":
            patch.setattr(stage_dlo, "get_tensor_model_parallel_world_size", lambda: 2)
        elif unsupported == "dp":
            patch.setattr(backend, "dp_size", 2)
        elif unsupported == "encoder":
            patch.setattr(backend, "_encoder_modules", [nn.Module()])
        elif unsupported == "resident":
            patch.setattr(backend, "_resident_blocks", [nn.Module()])
        elif unsupported == "hwr":
            patch.setattr(backend, "host_weight_plan", Mock(backing_kind="host_weight_runtime"))
        elif unsupported == "ordinary":
            patch.setattr(backend, "_using_rank_local_mmap", False)
        else:
            patch.setattr(backend, "_using_registered_mmap", True)
        with pytest.raises(ValueError, match="scope"):
            backend.shutdown(discard_stage=True)
    assert backend._poisoned_reason is None and not backend._stage_discard_started
    assert backend._all_hook_groups is groups and backend.enabled
    assert before == [dict(block._hook_registry._hooks) for block in pipeline.transformer.blocks]
    backend.shutdown(discard_stage=True)


def test_invalid_discard_option_is_rejected(make_backend):
    backend, _, _, _ = make_backend()
    with pytest.raises(ValueError):
        backend.shutdown(discard_stage=1)
    backend.shutdown(discard_stage=True)
