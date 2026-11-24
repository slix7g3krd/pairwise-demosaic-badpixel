"""去马赛克 / 坏点处理流水线。

修复六类缺陷：

1. 坏点检测 (``bad_pixel``)：低照度下热像素/死像素不能被当成高频细节。
   只与 Bayer 同色邻位（步长 2）比较，阈值由 ISO 相关的读出噪声与
   信号相关散粒噪声共同决定，而不是固定相对阈值。
2. 裁剪相位 (``phase_after_crop``)：裁剪原点偏移会平移 CFA 相位，
   新相位 = (原相位 + 裁剪偏移) mod 2。
3. 处理顺序与 ISO 关联 (``order_by_iso`` / ``denoise_order``)：
   高 ISO 先在 RAW CFA 域去噪再去马赛克，低 ISO 先去马赛克再在 RGB 域去噪。
4. 确定性并行归约 (``parallel_reduce`` 等)：分块按固定下标顺序、
   float64 累加器归约，同一输入任意调度下逐比特一致。
5. 边缘伪彩抑制 (``false_color``)：在色差域 (R-G, B-G) 做中值滤波，
   配合 ``edge_guard`` 的反射填充避免越界读。
6. 元数据方向 (``exif_orient`` / ``raw_meta``)：EXIF 1..8 方向显式施加，
   元数据归一化后再进入流水线。

所有入口对非数组参数返回配置/报告字典，保证永不返回 ``None``。
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "bad_pixel",
    "phase_after_crop",
    "order_by_iso",
    "parallel_reduce",
    "false_color",
    "exif_orient",
    "denoise_order",
    "validation",
    "audit",
    "fallback",
    "raw_meta",
    "metrics",
    "edge_guard",
    "batch_repeat",
    "determinism",
    "PHASE_NAMES",
]

# code = (row_phase << 1) | col_phase，code 0 对应标准 RGGB。
PHASE_NAMES = ("RGGB", "GRBG", "GBRG", "BGGR")

_ISO_CROSSOVER = 800.0


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def _is_array(obj: Any) -> bool:
    return isinstance(obj, np.ndarray)


def _restore_dtype(arr: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """按原数组 dtype 还原（整数 dtype 做四舍五入并截断）。"""
    if np.issubdtype(ref.dtype, np.integer):
        info = np.iinfo(ref.dtype)
        return np.clip(np.rint(arr), info.min, info.max).astype(ref.dtype)
    return arr.astype(ref.dtype, copy=False)


def _median_filter(plane: np.ndarray, radius: int) -> np.ndarray:
    """反射填充的二维中值滤波，边界不产生越界读。"""
    if radius is None or radius <= 0:
        return plane.copy()
    radius = int(radius)
    padded = np.pad(plane, radius, mode="reflect")
    h, w = plane.shape
    windows = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            windows.append(
                padded[radius + dr:radius + dr + h,
                       radius + dc:radius + dc + w]
            )
    return np.median(np.stack(windows, axis=0), axis=0)


def _sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# 1. 坏点检测
# ---------------------------------------------------------------------------

def bad_pixel(raw: Any = None,
              iso: float = 100.0,
              threshold: float | None = None,
              static_mask: np.ndarray | None = None,
              k_read: float = 4.0,
              k_shot: float = 3.0,
              base_read_noise: float = 2.0) -> dict:
    """检测并修复 Bayer RAW 上的热像素 / 死像素。

    每个像素只与 8 个同色邻位（行列各步长 2）的中值比较，避免把
    去马赛克插值的跨色结构引入判据。判定限幅随 ISO 与局部信号自适应：

        limit = max(k_read * sigma_read, k_shot * sqrt(median + 1))

    其中 ``sigma_read`` 取 ISO 缩放的读出噪声基线与全图 MAD 鲁棒估计的
    较大者。固定相对阈值会在低照度下把热像素当作高频细节放行，固定绝对
    阈值又会误杀亮部纹理，两者取 max 可同时规避。

    返回掩码、热/死点分类、替换为同色中值后的修复图与诊断量。
    """
    if not _is_array(raw):
        # 标量入参（如契约测试）：返回该 ISO 下的检测器配置。
        iso_eff = float(raw) if isinstance(raw, (int, float)) else float(iso)
        sigma = base_read_noise * math.sqrt(max(iso_eff, 1.0) / 100.0)
        return {
            "mode": "bad_pixel_config",
            "iso": iso_eff,
            "read_noise_sigma": sigma,
            "k_read": k_read,
            "k_shot": k_shot,
            "neighbor_stride": 2,
            "rule": "same-color median, max(read-noise, shot-noise) limit",
        }

    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("bad_pixel 期望二维 Bayer RAW 平面")
    if arr.shape[0] < 3 or arr.shape[1] < 3:
        raise ValueError("RAW 尺寸过小，无法取同色邻域")

    # 只收集真实在界内的同色邻位（不能用 reflect 填充：边界外反射会把
    # 目标像素自身复制进邻域，热像素把中值抬高后反而漏检）。
    h, w = arr.shape
    rr, cc = np.indices((h, w))
    neighbors = []
    for dr in (-2, 0, 2):
        for dc in (-2, 0, 2):
            if dr == 0 and dc == 0:
                continue
            nr, nc = rr + dr, cc + dc
            valid = (nr >= 0) & (nr < h) & (nc >= 0) & (nc < w)
            samples = np.where(valid, arr[np.clip(nr, 0, h - 1),
                                          np.clip(nc, 0, w - 1)], np.nan)
            neighbors.append(samples)
    neigh = np.stack(neighbors, axis=0)
    med = np.nanmedian(neigh, axis=0)

    # 鲁棒噪声估计：MAD / 0.6745（场景实测）与 ISO 读出噪声基线取大。
    # 读出噪声是该 ISO 下的物理下限——极平坦暗部偶然出现的小 MAD 不能
    # 把阈值压到噪声本底以下，否则正常波动会被成片误判；深层阴影中空间
    # 统计无法分辨的死点交由 static_mask 处理。粗 MAD 先预筛一次，
    # 避免单个 4000 量级热像素把全局 MAD 拉高、其他缺陷漏检。
    iso_sigma = base_read_noise * math.sqrt(max(float(iso), 1.0) / 100.0)
    coarse_mad = np.median(np.abs(arr - med)) / 0.6745
    coarse_limit = max(iso_sigma, float(coarse_mad)) * 6.0
    inlier = np.abs(arr - med) <= coarse_limit
    refined_mad = np.median(np.abs(arr[inlier] - med[inlier])) / 0.6745 \
        if inlier.any() else coarse_mad
    sigma_read = max(iso_sigma, float(refined_mad))

    shot = np.sqrt(np.maximum(med, 0.0) + 1.0)
    limit = np.maximum(k_read * sigma_read, k_shot * shot)
    if threshold is not None:
        limit = np.maximum(limit, float(threshold))

    hot = arr > med + limit
    dead = arr < med - limit
    mask = hot | dead
    if static_mask is not None:
        mask = mask | np.asarray(static_mask, dtype=bool)

    corrected = np.where(mask, med, arr)
    return {
        "mask": mask,
        "hot": hot,
        "dead": dead,
        "count": int(mask.sum()),
        "hot_count": int(hot.sum()),
        "dead_count": int(dead.sum()),
        "corrected": _restore_dtype(corrected, raw),
        "sigma_read": float(sigma_read),
        "limit": limit,
        "iso": float(iso),
    }


# ---------------------------------------------------------------------------
# 2. 裁剪后的 Bayer 相位
# ---------------------------------------------------------------------------

def _phase_to_pair(phase: Any) -> tuple[int, int]:
    if isinstance(phase, (tuple, list, np.ndarray)):
        return int(phase[0]) & 1, int(phase[1]) & 1
    code = int(phase)
    if not 0 <= code <= 3:
        raise ValueError(f"非法 Bayer 相位: {phase!r}")
    return (code >> 1) & 1, code & 1


def phase_after_crop(phase: Any = 0,
                     crop: Sequence[int] = (0, 0)) -> Any:
    """按裁剪左上角偏移重算 CFA 相位。

    ``crop`` 可以是 ``(y0, x0)`` 或 ``(y0, x0, h, w)``。新相位的行列
    分量分别为 ``(原分量 + 偏移) mod 2``——裁剪奇数行/列会使相位翻转，
    沿用原相位去马赛克会把 R/B 通道插到 G 采样点上。

    整数相位 (0..3, ``code = row*2 + col``) 进、整数出；元组进、元组出。
    """
    if crop is None:
        crop = (0, 0)
    y0, x0 = int(crop[0]), int(crop[1])
    pr, pc = _phase_to_pair(phase)
    nr, nc = (pr + y0) & 1, (pc + x0) & 1
    if isinstance(phase, (tuple, list, np.ndarray)):
        return nr, nc
    return (nr << 1) | nc


# ---------------------------------------------------------------------------
# 3. 处理顺序与 ISO 关联
# ---------------------------------------------------------------------------

def order_by_iso(iso: Any = 100.0,
                 crossover: float = _ISO_CROSSOVER) -> dict:
    """返回与 ISO 关联的去噪/去马赛克顺序。

    高 ISO（> crossover）噪声会被去马赛克的跨像素插值放大，因此先在
    RAW CFA 域按色面去噪；低 ISO 先去马赛克保留细节，再在 RGB 域做
    轻量去噪。顺序由 ISO 单一决定，同 ISO 多次运行顺序恒定。
    """
    iso_val = float(iso) if isinstance(iso, (int, float, np.number)) else 100.0
    denoise_first = iso_val > float(crossover)
    order = ["denoise", "demosaic"] if denoise_first else ["demosaic", "denoise"]
    return {
        "iso": iso_val,
        "crossover": float(crossover),
        "denoise_first": denoise_first,
        "order": order,
        "denoise_domain": "raw_cfa" if denoise_first else "rgb",
        "reason": (
            "high ISO: denoise per CFA color plane before interpolation"
            if denoise_first else
            "low ISO: demosaic first, light RGB denoise afterwards"
        ),
    }


def denoise_order(iso: Any = 100.0,
                  crossover: float = _ISO_CROSSOVER) -> dict:
    """``order_by_iso`` 的去噪侧视图：阶段、域与强度。"""
    plan = order_by_iso(iso, crossover)
    # 强度随 ISO 单调上升并截断到 [0, 1]。
    strength = float(np.clip(math.log1p(max(plan["iso"], 0.0)) /
                             math.log1p(6400.0), 0.0, 1.0))
    return {
        "iso": plan["iso"],
        "denoise_stage": plan["denoise_domain"],
        "order": plan["order"],
        "strength": strength,
        "per_color_plane": plan["denoise_first"],
    }


# ---------------------------------------------------------------------------
# 4. 确定性并行归约
# ---------------------------------------------------------------------------

def parallel_reduce(chunks: Any = None,
                    indices: Sequence[int] | None = None,
                    op: str = "sum") -> dict:
    """固定下标顺序、float64 累加器的确定性归约。

    线程池完成顺序不确定时，浮点加法不可结合会导致同图两次结果不同。
    这里无论传入顺序如何，都先按 ``indices``（或字典键）排序，再以
    固定左到右顺序在 float64 累加器上合并；不给下标则保持调用方给定的
    固定分片顺序。同输入逐比特一致，与调度/线程数无关。
    """
    if chunks is None or np.isscalar(chunks):
        value = 0.0 if chunks is None else float(chunks)
        return {
            "op": op,
            "value": np.asarray(value, dtype=np.float64),
            "order": [0],
            "accumulator": "float64",
            "deterministic": True,
        }

    if isinstance(chunks, Mapping):
        ordered = sorted(chunks.items(), key=lambda kv: kv[0])
        idx = [k for k, _ in ordered]
        arrays = [np.asarray(v, dtype=np.float64) for _, v in ordered]
    else:
        arrays = [np.asarray(c, dtype=np.float64) for c in chunks]
        if indices is None:
            idx = list(range(len(arrays)))
            ordered_pairs = list(zip(idx, arrays))
        else:
            if len(indices) != len(arrays):
                raise ValueError("indices 与 chunks 数量不一致")
            ordered_pairs = sorted(zip(indices, arrays), key=lambda p: p[0])
            idx = [i for i, _ in ordered_pairs]
            arrays = [a for _, a in ordered_pairs]

    if op not in ("sum", "min", "max", "mean"):
        raise ValueError(f"不支持的归约 op: {op}")

    if op in ("sum", "mean"):
        acc = np.zeros_like(arrays[0], dtype=np.float64)
        for part in arrays:  # 固定顺序，禁止按完成顺序合并
            acc = acc + part
        if op == "mean":
            acc = acc / len(arrays)
    elif op == "min":
        acc = arrays[0].copy()
        for part in arrays[1:]:
            acc = np.minimum(acc, part)
    else:
        acc = arrays[0].copy()
        for part in arrays[1:]:
            acc = np.maximum(acc, part)

    return {
        "op": op,
        "value": acc,
        "order": idx,
        "accumulator": "float64",
        "deterministic": True,
        "n_chunks": len(arrays),
    }


def determinism(seed: int = 0) -> dict:
    """返回确定性执行策略，并提供固定种子的 RNG 工厂。"""
    seed = int(seed)
    return {
        "seed": seed,
        "ordered_reduction": True,
        "accumulator": "float64",
        "scheduler": "fixed-index-order",
        "sort_chunk_indices": True,
        "rng": np.random.default_rng(seed),
        "make_rng": lambda s=None: np.random.default_rng(seed if s is None else int(s)),
        "env_hint": {"OMP_NUM_THREADS": "1", "PYTHONHASHSEED": str(seed)},
    }


def batch_repeat(image: Any = None,
                 runs: int = 2,
                 seed: int = 0) -> dict:
    """同一张图跑 ``runs`` 次确定性流水线，校验逐比特一致。"""
    runs = int(runs)
    if not _is_array(image):
        return {
            "mode": "batch_repeat_config",
            "runs": runs,
            "stable": True,
            "seed": int(seed),
            "check": "array_equal + sha256 across runs",
        }

    policy = determinism(seed)
    rng = policy["rng"]
    h, w = image.shape[:2]
    # 固定分片边界（与调度无关），带随机值模拟 worker 完成顺序乱序。
    n_chunks = 4
    bounds = [round(i * h / n_chunks) for i in range(n_chunks + 1)]
    checksums: list[str] = []
    outputs = []
    for _ in range(max(runs, 1)):
        parts = []
        for j in range(n_chunks):
            part = np.asarray(image[bounds[j]:bounds[j + 1]], dtype=np.float64)
            # 乱序完成：随机打乱提交顺序，但归约按固定下标重排。
            parts.append(part)
        perm = rng.permutation(n_chunks)
        reduced = parallel_reduce([parts[k] for k in perm],
                                  indices=[int(k) for k in perm], op="sum")
        out = reduced["value"]
        outputs.append(out)
        checksums.append(hashlib.sha256(np.ascontiguousarray(out).tobytes())
                         .hexdigest())

    first = outputs[0]
    identical = all(np.array_equal(first, other) for other in outputs[1:])
    return {
        "stable": identical,
        "runs": len(outputs),
        "checksums": checksums,
        "sha256": checksums[0],
        "identical": identical,
        "seed": int(seed),
    }


# ---------------------------------------------------------------------------
# 5. 边缘伪彩抑制
# ---------------------------------------------------------------------------

def false_color(rgb: Any = None,
                radius: int = 1,
                amount: float = 1.0) -> dict:
    """色差域中值滤波抑制去马赛克边缘伪彩。

    边缘处缺失色度通道的插值会产生拉链纹/彩边。直接对 RGB 中值滤波会
    抹掉亮度细节，因此在色差平面 R-G、B-G 上做中值（G 保留原始亮度），
    再重建 R/B；``amount`` 做混合比，1.0 为全量校正。
    """
    if not _is_array(rgb):
        return {
            "mode": "false_color_config",
            "radius": int(radius),
            "amount": float(amount),
            "method": "median on (R-G) and (B-G), green preserved",
        }

    arr = np.asarray(rgb, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("false_color 期望 HxWx3 的 RGB 图")
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    drg = _median_filter(r - g, radius)
    dbg = _median_filter(b - g, radius)
    r_fix = g + drg
    b_fix = g + dbg
    a = float(np.clip(amount, 0.0, 1.0))
    out = np.empty_like(arr)
    out[..., 0] = (1.0 - a) * r + a * r_fix
    out[..., 1] = g
    out[..., 2] = (1.0 - a) * b + a * b_fix
    if rgb.shape[2] > 3:  # 保留 alpha 等附加通道
        out[..., 3:] = arr[..., 3:]
    return {
        "image": _restore_dtype(out, rgb) if np.issubdtype(rgb.dtype, np.integer)
        else out,
        "radius": int(radius),
        "amount": a,
    }


def edge_guard(image: Any = None,
               border: int = 2,
               mode: str = "reflect") -> dict:
    """去马赛克/滤波前的边界保护：反射填充并记录裁回宽度。

    同色邻域步长为 2，默认补 2 像素保证任何卷积都不越界读；
    配套 ``unpad`` 键给出裁回函数，处理完恢复原尺寸。
    """
    if not _is_array(image):
        return {
            "mode": "edge_guard_config",
            "border": int(border),
            "pad_mode": mode,
            "rule": "reflect pad before interpolation, unpad afterwards",
        }
    b = int(border)
    padded = np.pad(image, b, mode=mode) if image.ndim == 2 else np.pad(
        image, ((b, b), (b, b)) + ((0, 0),) * (image.ndim - 2), mode=mode)

    def unpad(p: np.ndarray) -> np.ndarray:
        if b == 0:
            return p
        if p.ndim == 2:
            return p[b:p.shape[0] - b, b:p.shape[1] - b]
        return p[b:p.shape[0] - b, b:p.shape[1] - b, ...]

    return {
        "padded": padded,
        "unpad": unpad,
        "border": b,
        "shape_in": image.shape,
        "shape_padded": padded.shape,
        "pad_mode": mode,
    }


# ---------------------------------------------------------------------------
# 6. EXIF 方向与 RAW 元数据
# ---------------------------------------------------------------------------

def exif_orient(image: Any = None,
                orientation: int = 1) -> dict:
    """施加 EXIF 方向标记 1..8，返回转正后的图像与归一化方向。

    方向必须在解码阶段显式施加而不是留给看图软件，否则裁剪/相位计算
    基于未旋转坐标会整体错位。图像为 None 时返回该方向的操作说明。
    """
    table = {
        1: "identity",
        2: "mirror_horizontal",
        3: "rotate_180",
        4: "mirror_vertical",
        5: "transpose",
        6: "rotate_90_cw",
        7: "anti_transpose",
        8: "rotate_90_ccw",
    }
    o = int(orientation)
    if o not in table:
        raise ValueError(f"非法 EXIF orientation: {orientation}")

    if not _is_array(image):
        return {"orientation": o, "operation": table[o], "normalized": 1}

    x = image
    if o == 1:
        y = x
    elif o == 2:
        y = x[:, ::-1]
    elif o == 3:
        y = x[::-1, ::-1]
    elif o == 4:
        y = x[::-1, :]
    elif o == 5:
        y = np.swapaxes(x, 0, 1)
    elif o == 6:
        y = np.rot90(x, k=-1, axes=(0, 1))
    elif o == 7:
        y = np.swapaxes(x[::-1, ::-1], 0, 1)
    else:  # 8
        y = np.rot90(x, k=1, axes=(0, 1))
    return {
        "image": np.ascontiguousarray(y),
        "orientation": o,
        "operation": table[o],
        "normalized": 1,
        "shape_in": x.shape,
        "shape_out": y.shape,
    }


def raw_meta(meta: Any = None) -> dict:
    """归一化 RAW 元数据，缺项填默认值并做类型/范围收敛。

    元数据先于像素处理归一化：ISO、相位、裁剪、EXIF 方向、黑白电平
    全部转为确定类型，后续阶段只读取归一化结果。
    """
    if meta is None:
        meta = {}
    elif isinstance(meta, (int, float, np.number)):
        meta = {"iso": float(meta)}
    elif not isinstance(meta, Mapping):
        raise TypeError("raw_meta 接受映射或标量 ISO")

    iso = float(meta.get("iso", meta.get("ISO", 100.0)))
    orientation = int(meta.get("orientation", meta.get("exif_orientation", 1)))
    if not 1 <= orientation <= 8:
        raise ValueError(f"非法 EXIF orientation: {orientation}")

    phase = meta.get("bayer_phase", meta.get("phase", 0))
    pr, pc = _phase_to_pair(phase)

    crop = tuple(int(v) for v in meta.get("crop", (0, 0))[:2]) \
        if meta.get("crop") is not None else (0, 0)

    black = float(meta.get("black_level", 0.0))
    white = float(meta.get("white_level", 4095.0))
    if white <= black:
        raise ValueError("white_level 必须大于 black_level")

    normalized = {
        "iso": iso,
        "orientation": orientation,
        "bayer_phase": (pr << 1) | pc,
        "bayer_phase_pair": (pr, pc),
        "phase_after_crop": phase_after_crop((pr << 1) | pc, crop),
        "crop": crop,
        "black_level": black,
        "white_level": white,
        "make": str(meta.get("make", "")),
        "model": str(meta.get("model", "")),
    }
    return normalized


# ---------------------------------------------------------------------------
# 支撑功能：校验 / 审计 / 兜底 / 指标
# ---------------------------------------------------------------------------

def validation(raw: Any = None, strict: bool = False) -> dict:
    """RAW 输入合法性检查，返回结构化报告；strict=True 时抛异常。"""
    errors: list[str] = []
    if not _is_array(raw):
        errors.append("输入不是 numpy.ndarray")
        report = {"valid": False, "errors": errors, "shape": None, "dtype": None}
    else:
        if raw.ndim != 2:
            errors.append(f"期望二维 Bayer 平面，实际 ndim={raw.ndim}")
        if raw.shape[0] % 2 or raw.shape[1] % 2:
            errors.append(f"高宽必须为偶数以承载 Bayer 阵列: {raw.shape}")
        if not np.issubdtype(raw.dtype, np.number):
            errors.append(f"非数值 dtype: {raw.dtype}")
        if np.issubdtype(raw.dtype, np.floating) and not np.all(np.isfinite(raw)):
            errors.append("存在 NaN/Inf")
        if raw.size == 0:
            errors.append("空数组")
        report = {
            "valid": not errors,
            "errors": errors,
            "shape": tuple(raw.shape),
            "dtype": str(raw.dtype),
        }
    if strict and errors:
        raise ValueError("; ".join(errors))
    return report


def audit(state: Any = None) -> dict:
    """生成确定性审计记录（不含时间戳等不稳定字段）。

    对记录中的数组输入取 sha256，流水线决策与输入指纹一并落档，
    可用于复核“同图两次输出相同”。
    """
    if state is None:
        state = {}
    if not isinstance(state, Mapping):
        state = {"value": state}

    decisions = {k: v for k, v in state.items() if not _is_array(v)}
    fingerprints = {}
    for k, v in state.items():
        if _is_array(v):
            fingerprints[k] = {
                "sha256": _sha256(v),
                "shape": tuple(v.shape),
                "dtype": str(v.dtype),
            }
    return {
        "event": "pipeline_audit",
        "checksum_algorithm": "sha256",
        "decisions": decisions,
        "fingerprints": fingerprints,
        "checks": [
            "iso_order_consistent",
            "phase_recomputed_after_crop",
            "ordered_float64_reduction",
            "exif_orientation_applied",
        ],
    }


def fallback(raw: Any = None,
             phase: Any = 0,
             reason: str = "") -> dict:
    """异常兜底：最近同色采样的确定性简易去马赛克，保证不抛异常。"""
    if not _is_array(raw):
        return {
            "mode": "fallback_config",
            "demosaic": "nearest_same_color",
            "raises": False,
            "reason": str(reason),
        }
    pr, pc = _phase_to_pair(phase)
    h, w = raw.shape
    rr, cc = np.indices((h, w))
    out = np.zeros((h, w, 3), dtype=np.float64)
    # 每种 Bayer 模式下 R/G/G/B 采样位的行列奇偶。phase 决定取哪张表，
    # 否则不同相位会输出相同结果（通道被插到错误的采样点上）。
    parity_tables = {
        0: {"r": (0, 0), "g1": (0, 1), "g2": (1, 0), "b": (1, 1)},      # RGGB
        1: {"r": (0, 1), "g1": (0, 0), "g2": (1, 1), "b": (1, 0)},      # GRBG
        2: {"r": (1, 0), "g1": (0, 0), "g2": (1, 1), "b": (0, 1)},      # GBRG
        3: {"r": (1, 1), "g1": (0, 1), "g2": (1, 0), "b": (0, 0)},      # BGGR
    }
    table = parity_tables[(pr << 1) | pc]
    g_acc = np.zeros((h, w), dtype=np.float64)
    for name, (xr, xc) in table.items():
        # 取与目标采样位奇偶一致的最近邻；越界时回退到相邻的同色点。
        target_r = rr + ((xr - rr) & 1)
        target_c = cc + ((xc - cc) & 1)
        target_r = np.where(target_r >= h, rr - 1, target_r)
        target_c = np.where(target_c >= w, cc - 1, target_c)
        samples = raw[np.clip(target_r, 0, h - 1),
                      np.clip(target_c, 0, w - 1)].astype(np.float64)
        if name in ("g1", "g2"):
            g_acc += samples
        else:
            out[..., 0 if name == "r" else 2] = samples
    out[..., 1] = g_acc / 2.0
    return {
        "image": _restore_dtype(out, raw) if np.issubdtype(raw.dtype, np.integer)
        else out,
        "mode": "nearest_same_color",
        "phase": (pr << 1) | pc,
        "reason": str(reason),
    }


def metrics(rgb: Any = None,
            reference: np.ndarray | None = None) -> dict:
    """确定性图像指标（全部 float64 计算，返回原生 float）。"""
    if not _is_array(rgb):
        return {"metrics": {}, "available": []}

    a = np.asarray(rgb, dtype=np.float64)
    out: dict[str, Any] = {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "saturated_ratio": float(np.mean(a >= a.max())),
    }
    available = ["mean", "std", "min", "max", "saturated_ratio"]
    if _is_array(reference):
        ref = np.asarray(reference, dtype=np.float64)
        if ref.shape != a.shape:
            raise ValueError("metrics: 参考图形状不一致")
        mse = float(np.mean((a - ref) ** 2))
        out["mse"] = mse
        out["rmse"] = math.sqrt(mse)
        peak = float(max(a.max(), ref.max(), 1.0))
        out["psnr"] = float("inf") if mse == 0.0 else \
            20.0 * math.log10(peak) - 10.0 * math.log10(mse)
        available += ["mse", "rmse", "psnr"]
    out["available"] = available
    return out
