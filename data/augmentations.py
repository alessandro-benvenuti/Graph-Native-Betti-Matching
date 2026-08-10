"""Graph-safe augmentations for channel-first 3D volumes.

The repository uses one coordinate convention everywhere::

    normalized_coordinate = voxel_index / axis_size

For an axis of size ``S``, the largest valid voxel coordinate is therefore
``(S - 1) / S``.  It is deliberately *not* 1.0.  All geometric transforms in
this module operate through voxel coordinates internally, which prevents the
off-by-one errors caused by mixing ``index / size`` and ``index / (size - 1)``.

Volume tensors have shape ``[C, D, H, W]`` and graph nodes use ``[D, H, W]``
coordinate order.  Transform parameters are sampled once and applied to the
image, segmentation, and graph nodes together.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, Sequence, Tuple

import torch
import torch.nn.functional as F


# Keep runtime type aliases compatible with Python 3.8, which is available on
# Gardenia. Built-in generic aliases such as ``tuple[int, ...]`` require 3.9.
SpatialShape = Tuple[int, int, int]
QuarterTurns = Tuple[int, int, int]
FlipAxes = Tuple[bool, bool, bool]


def _shape_tuple(volume_shape: Sequence[int]) -> SpatialShape:
    shape = tuple(int(value) for value in volume_shape)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError(f"Expected a positive 3D spatial shape, got {shape}")
    return shape  # type: ignore[return-value]


def _validate_volume(volume: torch.Tensor, name: str) -> SpatialShape:
    if volume.ndim != 4:
        raise ValueError(
            f"{name} must be channel-first [C, D, H, W], got {tuple(volume.shape)}"
        )
    return _shape_tuple(volume.shape[-3:])


def _validate_nodes(nodes: torch.Tensor) -> None:
    if nodes.ndim != 2 or nodes.shape[-1] < 3:
        raise ValueError(f"nodes must have shape [N, >=3], got {tuple(nodes.shape)}")
    if not torch.is_floating_point(nodes):
        raise TypeError("nodes must use a floating-point dtype")


def _validate_clamp_range(
    clamp_range: tuple[float, float] | None,
) -> None:
    if clamp_range is None:
        return
    if len(clamp_range) != 2:
        raise ValueError("clamp_range must contain exactly two values")
    low, high = (float(value) for value in clamp_range)
    if not low < high:
        raise ValueError(
            f"clamp_range lower bound must be smaller than its upper bound, "
            f"got {clamp_range}"
        )


def axis_max_coordinates(
    volume_shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the largest valid normalized coordinate on each spatial axis."""

    shape = torch.as_tensor(_shape_tuple(volume_shape), dtype=dtype, device=device)
    return (shape - 1.0) / shape


def normalize_voxel_coordinates(
    voxel_coordinates: torch.Tensor,
    volume_shape: Sequence[int],
) -> torch.Tensor:
    """Convert voxel coordinates to the canonical ``index / size`` convention."""

    _validate_nodes(voxel_coordinates)
    shape = torch.as_tensor(
        _shape_tuple(volume_shape),
        dtype=voxel_coordinates.dtype,
        device=voxel_coordinates.device,
    )
    result = voxel_coordinates.clone()
    result[:, :3] = result[:, :3] / shape
    return result


def denormalize_coordinates(
    nodes: torch.Tensor,
    volume_shape: Sequence[int],
) -> torch.Tensor:
    """Convert canonical normalized coordinates to floating voxel coordinates."""

    _validate_nodes(nodes)
    shape = torch.as_tensor(
        _shape_tuple(volume_shape), dtype=nodes.dtype, device=nodes.device
    )
    result = nodes.clone()
    result[:, :3] = result[:, :3] * shape
    return result


def coordinates_to_voxel_indices(
    nodes: torch.Tensor,
    volume_shape: Sequence[int],
    *,
    clamp: bool = True,
) -> torch.Tensor:
    """Round canonical coordinates to integer voxel indices."""

    shape_tuple = _shape_tuple(volume_shape)
    voxel = denormalize_coordinates(nodes, shape_tuple)[:, :3].round().long()
    if clamp:
        lower = torch.zeros(3, dtype=voxel.dtype, device=voxel.device)
        upper = torch.as_tensor(shape_tuple, dtype=voxel.dtype, device=voxel.device) - 1
        # ``torch.maximum``/``minimum`` were added after the PyTorch release
        # installed on Gardenia. The two-input max/min overloads are
        # equivalent here and remain available in current PyTorch releases.
        voxel = torch.max(voxel, lower)
        voxel = torch.min(voxel, upper)
    return voxel


def assert_valid_coordinates(
    nodes: torch.Tensor,
    volume_shape: Sequence[int],
    *,
    atol: float = 1.0e-6,
) -> None:
    """Raise when nodes lie outside the canonical voxel-centre coordinate range."""

    _validate_nodes(nodes)
    if nodes.numel() == 0:
        return
    maxima = axis_max_coordinates(
        volume_shape, dtype=nodes.dtype, device=nodes.device
    )
    spatial = nodes[:, :3]
    if bool((spatial < -atol).any()) or bool((spatial > maxima + atol).any()):
        mins = spatial.min(dim=0)[0].tolist()
        maxs = spatial.max(dim=0)[0].tolist()
        raise ValueError(
            "Graph nodes are outside the valid index/size range: "
            f"min={mins}, max={maxs}, allowed_max={maxima.tolist()}"
        )


def _normalize_quarter_turns(quarter_turns: Sequence[int]) -> QuarterTurns:
    if len(quarter_turns) != 3:
        raise ValueError(
            "quarter_turns must contain rotations for (D,H), (D,W), and (H,W)"
        )
    return tuple(int(value) % 4 for value in quarter_turns)  # type: ignore[return-value]


def degrees_to_quarter_turns(angles_degrees: Sequence[int | float]) -> QuarterTurns:
    """Convert exact multiples of 90 degrees to quarter turns.

    The old ``continuous`` option silently rounded arbitrary angles.  This API
    rejects such inputs instead of pretending they are continuous rotations.
    """

    if len(angles_degrees) != 3:
        raise ValueError("Expected three rotation angles")
    turns: list[int] = []
    for angle in angles_degrees:
        quotient = float(angle) / 90.0
        rounded = round(quotient)
        if abs(quotient - rounded) > 1.0e-7:
            raise ValueError(
                f"Only exact 90-degree rotations are supported, got {angle} degrees"
            )
        turns.append(int(rounded) % 4)
    return tuple(turns)  # type: ignore[return-value]


def _rotate_voxel_coordinates_once(
    voxel_coordinates: torch.Tensor,
    volume_shape: SpatialShape,
    k: int,
    axis_a: int,
    axis_b: int,
) -> tuple[torch.Tensor, SpatialShape]:
    k %= 4
    if k == 0:
        return voxel_coordinates.clone(), volume_shape

    result = voxel_coordinates.clone()
    a = voxel_coordinates[:, axis_a].clone()
    b = voxel_coordinates[:, axis_b].clone()
    size_a = volume_shape[axis_a]
    size_b = volume_shape[axis_b]

    if k == 1:
        result[:, axis_a] = (size_b - 1) - b
        result[:, axis_b] = a
    elif k == 2:
        result[:, axis_a] = (size_a - 1) - a
        result[:, axis_b] = (size_b - 1) - b
    else:
        result[:, axis_a] = b
        result[:, axis_b] = (size_a - 1) - a

    new_shape = list(volume_shape)
    if k % 2 == 1:
        new_shape[axis_a], new_shape[axis_b] = size_b, size_a
    return result, _shape_tuple(new_shape)


def rotate_coordinates(
    nodes: torch.Tensor,
    quarter_turns: Sequence[int],
    volume_shape: Sequence[int],
) -> torch.Tensor:
    """Rotate graph nodes exactly as :func:`rotate_volume`.

    ``volume_shape`` is mandatory.  This is intentional: omitting it was the
    source of the previous ``1 - coordinate`` off-by-one behavior.
    """

    shape = _shape_tuple(volume_shape)
    assert_valid_coordinates(nodes, shape)
    turns = _normalize_quarter_turns(quarter_turns)
    voxel = denormalize_coordinates(nodes, shape)
    spatial = voxel[:, :3]

    for k, (axis_a, axis_b) in zip(turns, ((0, 1), (0, 2), (1, 2))):
        spatial, shape = _rotate_voxel_coordinates_once(
            spatial, shape, k, axis_a, axis_b
        )

    result = nodes.clone()
    shape_tensor = torch.as_tensor(shape, dtype=nodes.dtype, device=nodes.device)
    result[:, :3] = spatial / shape_tensor
    assert_valid_coordinates(result, shape)
    return result


def rotate_volume(
    volume: torch.Tensor,
    quarter_turns: Sequence[int],
) -> torch.Tensor:
    """Apply sequential 90-degree rotations in the model's axis order."""

    _validate_volume(volume, "volume")
    turns = _normalize_quarter_turns(quarter_turns)
    result = volume
    for k, dims in zip(turns, ((1, 2), (1, 3), (2, 3))):
        if k:
            result = torch.rot90(result, k=k, dims=dims)
    return result


def flip_coordinates(
    nodes: torch.Tensor,
    flip_axes: Sequence[bool],
    volume_shape: Sequence[int],
) -> torch.Tensor:
    """Flip graph nodes on selected D/H/W axes without off-by-one shifts."""

    if len(flip_axes) != 3:
        raise ValueError("flip_axes must contain three booleans in D/H/W order")
    shape = _shape_tuple(volume_shape)
    assert_valid_coordinates(nodes, shape)
    voxel = denormalize_coordinates(nodes, shape)
    for axis, enabled in enumerate(flip_axes):
        if enabled:
            voxel[:, axis] = (shape[axis] - 1) - voxel[:, axis]
    result = normalize_voxel_coordinates(voxel, shape)
    assert_valid_coordinates(result, shape)
    return result


def flip_volume(volume: torch.Tensor, flip_axes: Sequence[bool]) -> torch.Tensor:
    """Flip a channel-first volume on selected D/H/W axes."""

    _validate_volume(volume, "volume")
    if len(flip_axes) != 3:
        raise ValueError("flip_axes must contain three booleans in D/H/W order")
    dims = [axis + 1 for axis, enabled in enumerate(flip_axes) if enabled]
    return torch.flip(volume, dims=dims) if dims else volume


def zoom_coordinates(
    nodes: torch.Tensor,
    zoom_factor: float,
    volume_shape: Sequence[int],
) -> torch.Tensor:
    """Scale graph nodes about the voxel-grid centre."""

    if zoom_factor <= 0:
        raise ValueError(f"zoom_factor must be positive, got {zoom_factor}")
    shape = _shape_tuple(volume_shape)
    assert_valid_coordinates(nodes, shape)
    voxel = denormalize_coordinates(nodes, shape)
    centre = torch.as_tensor(
        [(size - 1.0) / 2.0 for size in shape],
        dtype=nodes.dtype,
        device=nodes.device,
    )
    voxel[:, :3] = (voxel[:, :3] - centre) * float(zoom_factor) + centre
    result = normalize_voxel_coordinates(voxel, shape)
    assert_valid_coordinates(result, shape)
    return result


def zoom_volume(
    volume: torch.Tensor,
    zoom_factor: float,
    *,
    mode: str,
    padding_mode: str = "border",
) -> torch.Tensor:
    """Zoom a 3D volume in place while retaining its spatial shape.

    PyTorch calls five-dimensional linear interpolation ``bilinear`` even
    though the actual operation is trilinear.  Segmentations must use
    ``mode='nearest'``.
    """

    _validate_volume(volume, "volume")
    if zoom_factor <= 0:
        raise ValueError(f"zoom_factor must be positive, got {zoom_factor}")
    if mode not in {"bilinear", "nearest"}:
        raise ValueError(f"Unsupported 3D interpolation mode: {mode}")
    if float(zoom_factor) == 1.0:
        return volume

    theta = torch.eye(3, 4, dtype=volume.dtype, device=volume.device).unsqueeze(0)
    theta[:, 0, 0] = 1.0 / float(zoom_factor)
    theta[:, 1, 1] = 1.0 / float(zoom_factor)
    theta[:, 2, 2] = 1.0 / float(zoom_factor)
    batch = volume.unsqueeze(0)
    grid = F.affine_grid(theta, size=batch.shape, align_corners=True)
    return F.grid_sample(
        batch,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    ).squeeze(0)


def scale_intensity(
    image: torch.Tensor,
    output_range: tuple[float, float] = (-0.5, 0.5),
) -> torch.Tensor:
    """Min-max scale an image like MONAI ``ScaleIntensity``."""

    low, high = (float(value) for value in output_range)
    if high <= low:
        raise ValueError(f"Invalid output range: {output_range}")
    image_min = image.min()
    image_max = image.max()
    span = image_max - image_min
    if bool(span == 0):
        return torch.full_like(image, low)
    return (image - image_min) / span * (high - low) + low


def add_gaussian_noise(
    image: torch.Tensor,
    *,
    mean: float,
    std: float,
    clamp_range: tuple[float, float] | None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add already-sampled Gaussian noise and optionally clamp the result."""

    if std < 0:
        raise ValueError(f"std must be non-negative, got {std}")
    if std == 0:
        result = image
    else:
        noise = torch.randn(
            image.shape,
            dtype=image.dtype,
            device=image.device,
            generator=generator,
        )
        result = image + noise * float(std) + float(mean)
    if clamp_range is not None:
        result = result.clamp(float(clamp_range[0]), float(clamp_range[1]))
    return result


def project_2d_to_3d(
    image: torch.Tensor,
    *,
    z_position: float = 0.5,
    thickness: int = 5,
    depth: int | None = None,
    background_value: float = -0.5,
) -> torch.Tensor:
    """Project a channel-first 2D plant image into a thin 3D slab.

    This preserves the active plants pipeline: five adjacent copies centred at
    ``z_position=0.5`` followed by a ``-0.5`` background shift.
    """

    if image.ndim != 3:
        raise ValueError(f"Expected [C,H,W] image, got {tuple(image.shape)}")
    if not 0.0 <= z_position < 1.0:
        raise ValueError("z_position must lie in [0, 1)")
    if thickness <= 0 or thickness % 2 == 0:
        raise ValueError("thickness must be a positive odd number")
    depth = int(depth if depth is not None else image.shape[-1])
    if depth < thickness:
        raise ValueError("depth must be at least as large as thickness")

    result = image.new_zeros((image.shape[0], image.shape[1], image.shape[2], depth))
    centre = round(z_position * depth)
    radius = thickness // 2
    if centre - radius < 0 or centre + radius >= depth:
        raise ValueError("The requested slab extends beyond the output volume")
    for offset in range(-radius, radius + 1):
        result[..., centre + offset] = image
    result += float(background_value)
    return result


def embed_2d_coordinates(
    coordinates_xy: torch.Tensor,
    *,
    z_position: float,
) -> torch.Tensor:
    """Convert plant ``[x,y]`` nodes to the model's ``[D,H,W]=[y,x,z]`` order."""

    if coordinates_xy.ndim != 2 or coordinates_xy.shape[-1] != 2:
        raise ValueError(
            f"Expected plant coordinates [N,2], got {tuple(coordinates_xy.shape)}"
        )
    z = coordinates_xy.new_full((coordinates_xy.shape[0], 1), float(z_position))
    return torch.cat((coordinates_xy[:, [1, 0]], z), dim=1)


def pad_coordinates(
    nodes: torch.Tensor,
    source_shape: Sequence[int],
    target_shape: Sequence[int],
) -> torch.Tensor:
    """Remap canonical coordinates after symmetric spatial padding."""

    source = _shape_tuple(source_shape)
    target = _shape_tuple(target_shape)
    if any(dst < src for src, dst in zip(source, target)):
        raise ValueError(f"target shape {target} cannot be smaller than source {source}")
    voxel = denormalize_coordinates(nodes, source)
    before = torch.as_tensor(
        [(dst - src) // 2 for src, dst in zip(source, target)],
        dtype=nodes.dtype,
        device=nodes.device,
    )
    voxel[:, :3] += before
    result = normalize_voxel_coordinates(voxel, target)
    assert_valid_coordinates(result, target)
    return result


@dataclass(frozen=True)
class AugmentationPolicy:
    """Distribution from which one graph-safe augmentation is sampled."""

    rotate_90: bool = False
    flip_probability: tuple[float, float, float] = (0.0, 0.0, 0.0)
    zoom_range: tuple[float, float] | None = None
    gaussian_noise_probability: float = 0.0
    gaussian_noise_mean: float = 0.0
    gaussian_noise_max_std: float = 0.0
    clamp_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if len(self.flip_probability) != 3:
            raise ValueError("flip_probability must contain three D/H/W values")
        if any(not 0.0 <= probability <= 1.0 for probability in self.flip_probability):
            raise ValueError("Flip probabilities must lie in [0,1]")
        if not 0.0 <= self.gaussian_noise_probability <= 1.0:
            raise ValueError("Gaussian noise probability must lie in [0,1]")
        if self.gaussian_noise_max_std < 0:
            raise ValueError("Gaussian maximum std must be non-negative")
        if self.zoom_range is not None:
            low, high = self.zoom_range
            if low <= 0 or high < low:
                raise ValueError(f"Invalid zoom range: {self.zoom_range}")
        _validate_clamp_range(self.clamp_range)


@dataclass(frozen=True)
class AugmentationParameters:
    """One fully sampled transform shared by image, segmentation, and nodes."""

    quarter_turns: QuarterTurns = (0, 0, 0)
    flip_axes: FlipAxes = (False, False, False)
    zoom_factor: float = 1.0
    add_noise: bool = False
    noise_mean: float = 0.0
    noise_std: float = 0.0
    clamp_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        _normalize_quarter_turns(self.quarter_turns)
        if len(self.flip_axes) != 3:
            raise ValueError("flip_axes must contain three D/H/W booleans")
        if self.zoom_factor <= 0:
            raise ValueError("zoom_factor must be positive")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        _validate_clamp_range(self.clamp_range)

    @property
    def is_identity(self) -> bool:
        return (
            self.quarter_turns == (0, 0, 0)
            and self.flip_axes == (False, False, False)
            and self.zoom_factor == 1.0
            and not self.add_noise
            and self.clamp_range is None
        )


@dataclass(frozen=True)
class GraphSample:
    """Image, segmentation, and graph nodes transformed as one unit."""

    image: torch.Tensor
    segmentation: torch.Tensor
    nodes: torch.Tensor


SYNTHETIC_MRI_TRAIN_POLICY = AugmentationPolicy(
    rotate_90=True,
    zoom_range=(0.6, 1.0),
    gaussian_noise_probability=0.35,
    gaussian_noise_mean=0.0,
    gaussian_noise_max_std=0.015,
    clamp_range=(-0.5, 0.5),
)

PLANTS_TRAIN_POLICY = AugmentationPolicy(
    rotate_90=True,
    flip_probability=(0.5, 0.5, 0.5),
)

EVALUATION_POLICY = AugmentationPolicy()


def policy_for_split(dataset_name: str, split: str) -> AugmentationPolicy:
    """Return the canonical policy and forbid random validation/test transforms."""

    normalized_split = split.strip().lower()
    if normalized_split in {"val", "valid", "validation", "test"}:
        return EVALUATION_POLICY
    if normalized_split != "train":
        raise ValueError(f"Unknown dataset split: {split}")

    normalized_name = dataset_name.strip().lower().replace("_", "")
    if normalized_name in {"syntheticmri", "vessel3d"}:
        return SYNTHETIC_MRI_TRAIN_POLICY
    if normalized_name == "plants":
        return PLANTS_TRAIN_POLICY
    raise ValueError(f"No augmentation policy is defined for dataset '{dataset_name}'")


def sample_augmentation(
    policy: AugmentationPolicy,
    rng: random.Random,
) -> AugmentationParameters:
    """Sample all random parameters once using the supplied worker-local RNG."""

    turns: QuarterTurns = (
        tuple(rng.randint(0, 3) for _ in range(3))  # type: ignore[assignment]
        if policy.rotate_90
        else (0, 0, 0)
    )
    def sample_event(probability: float) -> bool:
        # Do not advance the worker RNG for disabled or mandatory transforms.
        # This keeps the random stream stable when policies contain explicit
        # zero-probability fields.
        if probability == 0.0:
            return False
        if probability == 1.0:
            return True
        return rng.random() < probability

    flips: FlipAxes = tuple(
        sample_event(probability) for probability in policy.flip_probability
    )  # type: ignore[assignment]
    zoom_factor = (
        rng.uniform(*policy.zoom_range) if policy.zoom_range is not None else 1.0
    )
    should_add_noise = sample_event(policy.gaussian_noise_probability)
    # MONAI RandGaussianNoise samples the applied std uniformly from [0, std].
    sampled_std = (
        rng.uniform(0.0, policy.gaussian_noise_max_std)
        if should_add_noise
        else 0.0
    )
    return AugmentationParameters(
        quarter_turns=turns,
        flip_axes=flips,
        zoom_factor=zoom_factor,
        add_noise=should_add_noise,
        noise_mean=policy.gaussian_noise_mean,
        noise_std=sampled_std,
        clamp_range=policy.clamp_range,
    )


def apply_augmentation(
    image: torch.Tensor,
    segmentation: torch.Tensor,
    nodes: torch.Tensor,
    parameters: AugmentationParameters,
    *,
    noise_generator: torch.Generator | None = None,
) -> GraphSample:
    """Apply one sampled augmentation consistently to a graph-labelled volume."""

    image_shape = _validate_volume(image, "image")
    segmentation_shape = _validate_volume(segmentation, "segmentation")
    if image_shape != segmentation_shape:
        raise ValueError(
            f"Image and segmentation shapes differ: {image_shape} vs {segmentation_shape}"
        )
    assert_valid_coordinates(nodes, image_shape)

    transformed_image = rotate_volume(image, parameters.quarter_turns)
    transformed_segmentation = rotate_volume(
        segmentation, parameters.quarter_turns
    )
    transformed_nodes = rotate_coordinates(
        nodes, parameters.quarter_turns, image_shape
    )
    transformed_shape = _shape_tuple(transformed_image.shape[-3:])

    if parameters.zoom_factor != 1.0:
        transformed_image = zoom_volume(
            transformed_image, parameters.zoom_factor, mode="bilinear"
        )
        transformed_segmentation = zoom_volume(
            transformed_segmentation, parameters.zoom_factor, mode="nearest"
        )
        transformed_nodes = zoom_coordinates(
            transformed_nodes, parameters.zoom_factor, transformed_shape
        )

    transformed_image = flip_volume(transformed_image, parameters.flip_axes)
    transformed_segmentation = flip_volume(
        transformed_segmentation, parameters.flip_axes
    )
    transformed_nodes = flip_coordinates(
        transformed_nodes, parameters.flip_axes, transformed_shape
    )

    if parameters.add_noise:
        transformed_image = add_gaussian_noise(
            transformed_image,
            mean=parameters.noise_mean,
            std=parameters.noise_std,
            clamp_range=None,
            generator=noise_generator,
        )

    # The MRI Compose chain always ran its final clamp, including when
    # RandGaussianNoise did not select the sample. Preserve that behavior.
    if parameters.clamp_range is not None:
        transformed_image = transformed_image.clamp(
            float(parameters.clamp_range[0]),
            float(parameters.clamp_range[1]),
        )

    assert_valid_coordinates(transformed_nodes, transformed_shape)
    return GraphSample(
        image=transformed_image,
        segmentation=transformed_segmentation,
        nodes=transformed_nodes,
    )
