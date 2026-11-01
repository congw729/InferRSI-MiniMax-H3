"""Independent bounded sample oracle for the genuine ADALN projection.
No real payload is read on import. Gate choice is frozen before first real run.
"""
from __future__ import annotations
import hashlib, json, math, struct
from pathlib import Path

REVISION = "e5eb578a89295337b8ff433a035929ce0279e0b6"
SOURCE_URL = "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/" + REVISION + "/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors"
HEADER_SHA256 = "7513055dc7e4b8f67e1c8afa2126fb48e899f9c7b70d25655dad599d8c2e6783"
KEY = "blocks.0.adaln_proj.linear.weight"
PREFIX = KEY[:-7]
KEYS = {KEY, PREFIX + ".weight_scale", PREFIX + ".comfy_quant"}
SOURCE_FILE_BYTES = 34038892334
SOURCE_HEADER_BYTES = 108240
SOURCE_RANGES = {
    PREFIX + ".weight_scale": {"absolute_start": 817872, "absolute_end_inclusive": 1204943,
        "bytes": 387072, "dtype": "F32", "shape": [96768, 1]},
    KEY: {"absolute_start": 1765671184, "absolute_end_inclusive": 2025783567,
        "bytes": 260112384, "dtype": "I8", "shape": [96768, 2688]},
    PREFIX + ".comfy_quant": {"absolute_start": 34038874384, "absolute_end_inclusive": 34038874454,
        "bytes": 71, "dtype": "U8", "shape": [71]},
}
LOCAL_OFFSETS = {
    PREFIX + ".weight_scale": [0, 387072],
    KEY: [387072, 260499456],
    PREFIX + ".comfy_quant": [260499456, 260499527],
}
SHAPE = (96768, 2688)
GROUP = 64
PAYLOAD_BYTES = 260499527
SAMPLE_ROWS = (0, 1, 31, 32, 127, 128, 32255, 32256, 48384, 96766, 96767)
SAMPLE_COLUMNS = (0, 1344, 2624)
U32 = 2.0 ** -24
UBF16 = 2.0 ** -8
F32_MIN_NORMAL = 2.0 ** -126
BF16_MIN_SUBNORMAL = 2.0 ** -133

def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()

def verify_sample_identity(path, receipt_path):
    path, receipt_path = Path(path), Path(receipt_path)
    if not path.is_file() or path.stat().st_size != 260499855:
        raise ValueError("explicit bounded sample file required")
    if receipt_path.stat().st_size > 1024**2:
        raise ValueError("sample receipt size bound")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = {
        "complete": True, "sample_downloaded": True, "remote_header_verified": True,
        "revision": REVISION, "source_url": SOURCE_URL,
        "source_header_sha256": HEADER_SHA256,
        "source_file_bytes": SOURCE_FILE_BYTES, "source_header_bytes": SOURCE_HEADER_BYTES,
        "byte_ranges": SOURCE_RANGES, "local_data_offsets_by_key": LOCAL_OFFSETS,
        "full_payload_sha256_verified": False, "tensor_payload_bytes": PAYLOAD_BYTES,
        "sample_basename": path.name, "sample_file_bytes": 260499855, "local_header_bytes": 328,
    }
    for name, value in expected.items():
        if receipt.get(name) != value:
            raise ValueError("sample receipt field mismatch: " + name)
    if set(receipt.get("keys", ())) != KEYS or len(receipt["keys"]) != 3:
        raise ValueError("sample receipt exact tensor set")
    def unique(pairs):
        document = {}
        for key, value in pairs:
            if key in document:
                raise ValueError("duplicate local sample header key")
            document[key] = value
        return document
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("sample prefix missing")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if not 0 < header_bytes <= 4096:
            raise ValueError("bounded local sample header required")
        raw_header = stream.read(header_bytes)
    if len(raw_header) != header_bytes:
        raise ValueError("truncated local header")
    header = json.loads(raw_header, object_pairs_hook=unique)
    expected_header = {key: {"dtype": info["dtype"], "shape": info["shape"],
        "data_offsets": LOCAL_OFFSETS[key]} for key, info in SOURCE_RANGES.items()}
    if header != expected_header:
        raise ValueError("local three-key header differs from frozen ranges/geometry")
    if receipt.get("local_header_bytes") != 8 + header_bytes:
        raise ValueError("sample receipt local header length mismatch")
    if path.stat().st_size != 8 + header_bytes + PAYLOAD_BYTES:
        raise ValueError("sample exact payload length mismatch")
    digest = sha_file(path)
    if digest != receipt.get("sample_sha256"):
        raise ValueError("sample file differs from download receipt")
    return {
        "sample_sha256": digest, "receipt_sha256": sha_file(receipt_path),
        "sample_file_bytes": path.stat().st_size,
        "source_header_sha256": HEADER_SHA256, "source_revision": REVISION,
        "full_remote_payload_sha256_verified": False,
        "scope": "local repacked three-key sample hash and fixed downloader receipt identity",
    }

def _decode_binary_bits(bits, fraction_bits):
    exponent = (bits >> fraction_bits) & 255
    fraction = bits & ((1 << fraction_bits) - 1)
    negative = bool(bits >> (fraction_bits + 8))
    if exponent == 255:
        value = float("nan") if fraction else float("inf")
    elif exponent == 0:
        value = math.ldexp(float(fraction), -126 - fraction_bits)
    else:
        value = math.ldexp(float((1 << fraction_bits) | fraction), exponent - 127 - fraction_bits)
    return -value if negative else value

def _round_binary(value, fraction_bits):
    """Software RNE from an exact Python double, independent of host F32 FTZ."""
    if not math.isfinite(value) or value == 0:
        return value
    absolute = abs(value)
    power = (-126 - fraction_bits if absolute < 2.0**-126
             else math.frexp(absolute)[1] - 1 - fraction_bits)
    numerator, denominator = absolute.as_integer_ratio()
    if power < 0:
        numerator <<= -power
    else:
        denominator <<= power
    integer, remainder = divmod(numerator, denominator)
    if 2 * remainder > denominator or (2 * remainder == denominator and integer & 1):
        integer += 1
    rounded = math.ldexp(float(integer), power)
    if rounded > (2 - 2.0**(-fraction_bits)) * 2.0**127:
        rounded = float("inf")
    return math.copysign(rounded, value)

def _f32(value):
    return _round_binary(value, 23)

def _sign(row, column):
    """Regular B4 digit product, independent of candidate's matrix filling."""
    sign = 1
    for _ in range(3):
        if row % 4 + column % 4 == 3:
            sign = -sign
        row //= 4
        column //= 4
    return sign

def reference_group(q_values, scale):
    if len(q_values) != GROUP or not math.isfinite(scale) or scale < 0:
        raise ValueError("supported source group and finite nonnegative scale required")
    rounded = [_f32(int(value) * scale) for value in q_values]
    if not all(math.isfinite(value) for value in rounded):
        raise ValueError("nonfinite F32 products")
    magnitude = math.fsum(abs(value) / 8.0 for value in rounded)
    reference = [
        math.fsum(rounded[k] * _sign(j, k) / 8.0 for k in range(GROUP))
        for j in range(GROUP)
    ]
    return reference, magnitude

def error_bounds(reference, magnitude):
    gamma = GROUP * U32 / (1.0 - GROUP * U32)
    # Existing non-dyadic mathematical gate, relative to F32 input products.
    normal = UBF16 * abs(reference) + (1.0 + UBF16) * gamma * magnitude + BF16_MIN_SUBNORMAL
    # For subnormal scale DAZ, q*scale may be up to 128*tiny per element.
    # The normalized group sum is therefore bounded by 128*sqrt(g)*tiny.
    # Dot-product and accumulation FTZ add at most 2g*tiny; +2 is margin.
    # The denominator bounds subsequent FP32 accumulation amplification.
    # BF16_MIN_NORMAL (same exponent range) covers final subnormal flush.
    underflow = ((1.0 + UBF16) * (128 * math.sqrt(GROUP) + 2 * GROUP + 2) * F32_MIN_NORMAL /
                 (1.0 - GROUP * U32) + F32_MIN_NORMAL)
    return normal, normal + underflow

def is_within_frozen_bound(value, reference, magnitude):
    """The actual numeric accept/reject predicate used by the real validator."""
    if not math.isfinite(value):
        return False
    return abs(value - reference) <= error_bounds(reference, magnitude)[1]

def validate_sampled_real_projection(source, scales, output):
    import torch
    if (tuple(source.shape) != SHAPE or source.dtype != torch.int8 or source.device.type != "cpu"
            or tuple(scales.shape) != (SHAPE[0], 1) or scales.dtype != torch.float32
            or scales.device.type != "cpu" or tuple(output.shape) != SHAPE
            or output.dtype != torch.bfloat16 or output.device.type != "cpu"):
        raise ValueError("exact real ADALN source/scale/output geometry required")
    values = failed_normal = failed_extended = 0
    maximum_error = maximum_bound = maximum_fraction = 0.0
    exact_bf16_count = 0
    for row in SAMPLE_ROWS:
        scale = _decode_binary_bits(int(scales.view(torch.int32)[row, 0]) & 0xffffffff, 23)
        for column in SAMPLE_COLUMNS:
            q = source[row, column:column + GROUP].tolist()
            reference, magnitude = reference_group(q, scale)
            got = [_decode_binary_bits(bits & 0xffff, 7) for bits in output.view(torch.int16)[row, column:column + GROUP].tolist()]
            reference_bf16 = [_round_binary(_f32(value), 7) for value in reference]
            for value, ref, rounded in zip(got, reference, reference_bf16):
                if not math.isfinite(value):
                    raise AssertionError("nonfinite real sample recovery")
                normal, bound = error_bounds(ref, magnitude)
                error = abs(value - ref)
                values += 1
                failed_normal += int(error > normal)
                failed_extended += int(not is_within_frozen_bound(value, ref, magnitude))
                exact_bf16_count += int(value == rounded)
                maximum_error = max(maximum_error, error)
                maximum_bound = max(maximum_bound, bound)
                maximum_fraction = max(maximum_fraction, error / bound if bound else float(error != 0))
    metrics = {
        "values_checked": values, "sample_rows": list(SAMPLE_ROWS),
        "column_group_starts": list(SAMPLE_COLUMNS),
        "normal_bound_violations_observed": failed_normal,
        "frozen_underflow_extended_bound_violations": failed_extended,
        "bf16_exact_to_fp64_rounded_count_observed": exact_bf16_count,
        "max_absolute_error_to_f32_product_fp64_sum": maximum_error,
        "max_frozen_bound": maximum_bound, "max_error_over_bound": maximum_fraction,
        "reference": "integer-bit decoded source scale; software F32 RNE(q*scale); radix4-sign FP64 fsum /8; BF16 output decoded from integer bits",
        "gate": "all 2112 sampled values finite and within predeclared F32/BF16 bound with underflow extension",
        "not_proven": ["unsampled inverse-rotation values", "original BF16 parity", "quantization quality"],
    }
    if values != 2112 or failed_extended:
        raise AssertionError("real sampled numerical error gate failed: " + json.dumps(metrics))
    return metrics
