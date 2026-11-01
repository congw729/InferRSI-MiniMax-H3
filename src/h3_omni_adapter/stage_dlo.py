"""Explicit terminal-stage cleanup for the local H3 DLO prototype.

Default disable/shutdown retain the native restoration contract. The opt-in
path destroys streamed weights and forbids reuse; it is not a serving API.
"""
from __future__ import annotations

from itertools import chain

import torch
import vllm_omni.diffusion.offloader.distributed_layerwise_backend as native
from torch import nn
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm_omni.diffusion.model_loader.host_weight_plan import HostWeightPlan
from vllm_omni.diffusion.offloader.base import OffloadConfig, run_cleanup_steps
from vllm_omni.diffusion.offloader.tensor_utils import clear_tensor_storage


def _reject_discarded_forward(_module: nn.Module, _inputs: tuple) -> None:
    raise RuntimeError("stage weights were discarded; create and load a new model")


class StageDLOBackend(native.DistributedLayerwiseOffloadBackend):
    """Native transport with a narrow, explicit, nonreusable stage exit."""

    def __init__(
        self,
        config: OffloadConfig,
        device: torch.device,
        host_weight_plan: HostWeightPlan | None = None,
    ):
        super().__init__(config, device, host_weight_plan=host_weight_plan)
        self._stage_discard_started = False
        self._stage_discard_complete = False

    def disable(self) -> None:
        if self._stage_discard_complete:
            return
        if self._stage_discard_started:
            raise RuntimeError("stage discard is pending; retry shutdown(discard_stage=True)")
        super().disable()

    def shutdown(self, *, discard_stage: bool = False) -> None:
        if type(discard_stage) is not bool:
            raise ValueError("discard_stage must be a bool")
        if self._stage_discard_complete:
            return
        if not discard_stage:
            if self._stage_discard_started:
                raise RuntimeError("stage discard is pending; retry shutdown(discard_stage=True)")
            super().shutdown()
            return
        if not self._stage_discard_started:
            self._validate_discard_scope()

        # Stop before any destructive mutation if pending transfers cannot drain.
        native.current_omni_platform.synchronize()
        self._stage_discard_started = True
        self._poisoned_reason = "stage weights were discarded; recreate the backend and reload a new model"

        hooks = list(chain.from_iterable(self._all_hook_groups))
        blocks = list(chain.from_iterable(self._blocks))
        for block in blocks:
            if not getattr(block, "_h3_stage_discarded", False):
                block.register_forward_pre_hook(_reject_discarded_forward)
                block._h3_stage_discarded = True

        # Reuse the native transactional zero-storage primitive. Keep the mmap
        # masters and hook ownership intact until all detach operations succeed.
        targets = {
            id(value): value
            for hook in hooks
            for value in chain(hook.next_block_parameters.values(), hook.next_block_buffers.values())
        }
        clear_tensor_storage(targets.values())
        removal_error = run_cleanup_steps(
            ("removing terminal-stage DLO hook", lambda block=block: native.remove_distributed_block_hook(block))
            for block in blocks
        )
        if removal_error is not None:
            raise removal_error

        # Source tensor views still own their mappings if handle release fails.
        # Retaining the hook groups permits cleanup retry after either failure.
        self._release_mmap_handles()
        for hook in hooks:
            hook.cpu_sources.clear()
            hook.cpu_shards.clear()
            hook.tensor_transforms.clear()
            hook.gpu_buffers = [None, None]
            hook.gpu_shard_buffers = [None, None]
            hook.cpu_staging_buffers = [None, None]
            hook.cpu_staging_events = [None, None]
            hook.ready_events = [None, None]
            hook._prefetch_done = None
            hook._cached_repoint = ()
            hook._materialization_probe = None
            hook._prev_hook = None
        self._blocks.clear()
        self._all_hook_groups.clear()
        self.host_weight_plan = None
        self._using_mmap = False
        self._using_rank_local_mmap = False
        self._using_registered_mmap = False
        self.enabled = False
        self._stage_discard_complete = True

    def _validate_discard_scope(self) -> None:
        hooks = list(chain.from_iterable(self._all_hook_groups))
        supported = (
            self.enabled
            and self.dp_size == 1
            and get_tensor_model_parallel_world_size() == 1
            and self.config.components == frozenset({"dit"})
            and self._using_mmap
            and self._using_rank_local_mmap
            and not self._using_registered_mmap
            and self.host_weight_plan is not None
            and self.host_weight_plan.backing_kind == "checkpoint_mmap"
            and self._host_weight_lease is None
            and self._host_registration is None
            and not self._resident_blocks
            and self._resident_layer_group is None
            and self._residency_pipeline_ref is None
            and not self._encoder_modules
            and not self._staged_components
            and len(self._all_hook_groups) == 1
            and len(hooks) >= 2
            and all(h.dp_size == 1 and h.rank_local_mmap and not h.registered_mmap for h in hooks)
        )
        if not supported:
            raise ValueError("unsupported terminal discard scope")
