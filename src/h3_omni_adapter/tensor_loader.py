"""Bounded synthetic loader with real tensor copies and two host staging slots.

The byte adapter still owns transforms. This is not full-model mmap/offload.
Byte temporaries and both byte/tensor slot pairs contribute to host memory.
"""
import torch

from .byte_adapter import MappedStager


class TensorStager:
    """Copy already transformed bytes into caller-owned CPU/CUDA parameters."""

    def __init__(self, plan, *, device):
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("only CPU and CUDA are supported")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.closed = False
        self.sequence = 0
        self.slots = []
        self.events = [None, None]
        self.byte_stager = MappedStager(plan)
        try:
            capacity = max(max(binding["length"], 4) for binding in plan.values())
            self.slots = [torch.empty(capacity, dtype=torch.uint8, device="cpu",
                                      pin_memory=self.device.type == "cuda") for _ in range(2)]
        except Exception:
            self.close()
            raise

    def load_into(self, binding, parameter):
        if self.closed:
            raise ValueError("stager is closed")
        field = binding["field"]
        dtype = ({"U8": torch.uint8, "BF16": torch.bfloat16, "F32": torch.float32}[binding["dtype"]]
                 if field in {"weight", "scales", "dense"} else torch.float32)
        shape = binding["shape"] if field in {"weight", "scales", "dense"} else (1,)
        if parameter.dtype != dtype or tuple(parameter.shape) != shape:
            raise ValueError("destination dtype or shape mismatch")
        if parameter.device != self.device:
            raise ValueError("destination device mismatch")
        slot_index = self.sequence % 2
        event = self.events[slot_index]
        if event is not None:
            event.synchronize()
        _, actual_shape, data = self.byte_stager.load(binding)
        if actual_shape != shape:
            raise ValueError("transformed shape mismatch")
        slot = self.slots[slot_index][:len(data)]
        slot.copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8))
        view = slot.view(dtype).reshape(shape)
        with torch.no_grad():
            parameter.copy_(view, non_blocking=self.device.type == "cuda")
        if self.device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            self.events[slot_index] = event
        self.sequence += 1

    def synchronize(self):
        for event in self.events:
            if event is not None:
                event.synchronize()

    def stats(self):
        return {**self.byte_stager.stats(), "tensor_host_slot_bytes": sum(x.numel() for x in self.slots),
                "tensor_host_slots_pinned": self.device.type == "cuda",
                "parameter_copies": self.sequence,
                "scope": "slot accounting only; excludes temporaries, parameters and runtime RSS"}

    def close(self):
        if self.closed:
            return
        try:
            self.synchronize()
        finally:
            self.events.clear()
            self.slots.clear()
            self.byte_stager.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()
