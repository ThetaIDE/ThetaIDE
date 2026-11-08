"""Spec-driven sequential networks, for architectures authored in the GUI.

Every other architecture in this package is a Python class with named
hyperparameters, so adding one is a code change. This one takes the
architecture *itself* as configuration: a ``layers`` list of
``{type: ..., **params}`` dicts that the visual model builder writes into
``in/config/model/<name>.yaml``.

Nothing in the pipeline needed to change to support it. ``build_model``
already forwards unrecognised config keys to the constructor, and
``in/config/model/`` is already scanned to populate the panel's model combo,
so a saved design is selectable like any hand-written architecture.

``plan_shapes`` is the other half of the contract: it resolves the tensor
shape after every layer without building anything, so the editor can show a
live shape column and flag a broken stack before training starts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn

from src.app.core.model_registry import register_model

Shape = tuple[int, ...]


class LayerSpecError(ValueError):
    """A layer is unknown, misconfigured, or fed a shape it cannot accept."""


# ── Layer vocabulary ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class LayerKind:
    """One entry in the layer vocabulary.

    ``params`` carries each editable parameter's default, which is what lets
    the editor render a layer's fields generically instead of hardcoding a
    form per layer type. ``ranks`` is the set of input ranks the layer
    accepts, counted without the batch dimension: 1 is a flat feature
    vector, 2 is (C, L), 3 is (C, H, W).
    """

    name: str
    build: Callable[[Shape, dict[str, Any]], nn.Module]
    shape: Callable[[Shape, dict[str, Any]], Shape]
    doc: str
    params: dict[str, Any] = field(default_factory=dict)
    ranks: tuple[int, ...] = (1, 2, 3)
    category: str = "other"


LAYER_KINDS: dict[str, LayerKind] = {}


def register_layer(kind: LayerKind) -> LayerKind:
    LAYER_KINDS[kind.name] = kind
    return kind


def _unchanged(shape: Shape, params: dict[str, Any]) -> Shape:
    return shape


def _int_pair(value: Any) -> tuple[int, int]:
    """Accept ``3`` or ``[3, 5]`` for kernel/stride/padding style parameters."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise LayerSpecError(f"expected one or two numbers, got {list(value)!r}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _conv_out(size: int, kernel: int, stride: int, padding: int, dilation: int = 1) -> int:
    return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


# Activations and other shape-preserving layers, which differ only in the
# module they construct.
_SIMPLE: tuple[tuple[str, Callable[[], nn.Module], str], ...] = (
    ("relu", nn.ReLU, "Rectified linear unit."),
    ("gelu", nn.GELU, "Gaussian error linear unit."),
    ("tanh", nn.Tanh, "Hyperbolic tangent, squashes to (-1, 1)."),
    ("sigmoid", nn.Sigmoid, "Logistic sigmoid, squashes to (0, 1)."),
    ("silu", nn.SiLU, "Sigmoid-weighted linear unit (swish)."),
    ("elu", nn.ELU, "Exponential linear unit."),
    ("identity", nn.Identity, "Passes its input through untouched."),
)

for _name, _ctor, _doc in _SIMPLE:
    register_layer(
        LayerKind(
            name=_name,
            build=lambda shape, params, _c=_ctor: _c(),
            shape=_unchanged,
            doc=_doc,
            category="activation" if _name != "identity" else "other",
        )
    )

register_layer(
    LayerKind(
        name="leaky_relu",
        build=lambda shape, params: nn.LeakyReLU(float(params.get("negative_slope", 0.01))),
        shape=_unchanged,
        doc="Rectified linear unit with a non-zero slope below zero.",
        params={"negative_slope": 0.01},
        category="activation",
    )
)

register_layer(
    LayerKind(
        name="linear",
        build=lambda shape, params: nn.Linear(
            shape[0], int(params["out_features"]), bias=bool(params.get("bias", True))
        ),
        shape=lambda shape, params: (int(params["out_features"]),),
        doc="Fully connected layer over a flat feature vector.",
        params={"out_features": 64, "bias": True},
        ranks=(1,),
        category="dense",
    )
)

register_layer(
    LayerKind(
        name="dropout",
        build=lambda shape, params: nn.Dropout(float(params.get("p", 0.1))),
        shape=_unchanged,
        doc="Zeroes a fraction of its inputs during training only.",
        params={"p": 0.1},
        category="regularisation",
    )
)

register_layer(
    LayerKind(
        name="flatten",
        build=lambda shape, params: nn.Flatten(start_dim=1),
        shape=lambda shape, params: (int(torch.tensor(shape).prod().item()),),
        doc="Collapses every non-batch dimension into one, for a dense layer.",
        ranks=(1, 2, 3),
        category="shape",
    )
)

register_layer(
    LayerKind(
        name="layernorm",
        build=lambda shape, params: nn.LayerNorm(shape[-1]),
        shape=_unchanged,
        doc="Normalises over the last dimension. No batch-size dependence.",
        category="regularisation",
    )
)

register_layer(
    LayerKind(
        name="batchnorm1d",
        build=lambda shape, params: nn.BatchNorm1d(shape[0]),
        shape=_unchanged,
        doc="Normalises each feature or channel across the batch.",
        ranks=(1, 2),
        category="regularisation",
    )
)

register_layer(
    LayerKind(
        name="batchnorm2d",
        build=lambda shape, params: nn.BatchNorm2d(shape[0]),
        shape=_unchanged,
        doc="Normalises each channel of an image across the batch.",
        ranks=(3,),
        category="regularisation",
    )
)


def _conv1d_shape(shape: Shape, params: dict[str, Any]) -> Shape:
    k, _ = _int_pair(params.get("kernel_size", 3))
    s, _ = _int_pair(params.get("stride", 1))
    p, _ = _int_pair(params.get("padding", 0))
    return (int(params["out_channels"]), _conv_out(shape[1], k, s, p))


register_layer(
    LayerKind(
        name="conv1d",
        build=lambda shape, params: nn.Conv1d(
            shape[0],
            int(params["out_channels"]),
            kernel_size=_int_pair(params.get("kernel_size", 3))[0],
            stride=_int_pair(params.get("stride", 1))[0],
            padding=_int_pair(params.get("padding", 0))[0],
        ),
        shape=_conv1d_shape,
        doc="Slides a learned filter along a sequence of shape (C, L).",
        params={"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 0},
        ranks=(2,),
        category="convolution",
    )
)


def _conv2d_shape(shape: Shape, params: dict[str, Any]) -> Shape:
    kh, kw = _int_pair(params.get("kernel_size", 3))
    sh, sw = _int_pair(params.get("stride", 1))
    ph, pw = _int_pair(params.get("padding", 0))
    return (
        int(params["out_channels"]),
        _conv_out(shape[1], kh, sh, ph),
        _conv_out(shape[2], kw, sw, pw),
    )


register_layer(
    LayerKind(
        name="conv2d",
        build=lambda shape, params: nn.Conv2d(
            shape[0],
            int(params["out_channels"]),
            kernel_size=_int_pair(params.get("kernel_size", 3)),
            stride=_int_pair(params.get("stride", 1)),
            padding=_int_pair(params.get("padding", 0)),
        ),
        shape=_conv2d_shape,
        doc="Slides a learned filter over an image of shape (C, H, W).",
        params={"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 0},
        ranks=(3,),
        category="convolution",
    )
)


def _pool_shape(rank: int) -> Callable[[Shape, dict[str, Any]], Shape]:
    def shape_fn(shape: Shape, params: dict[str, Any]) -> Shape:
        kernel = _int_pair(params.get("kernel_size", 2))
        stride = _int_pair(params.get("stride") or params.get("kernel_size", 2))
        spatial = [_conv_out(shape[i + 1], kernel[i], stride[i], 0) for i in range(rank)]
        return (shape[0], *spatial)

    return shape_fn


for _name, _ctor, _rank, _doc in (
    ("maxpool1d", nn.MaxPool1d, 1, "Takes the maximum over each window of a sequence."),
    ("avgpool1d", nn.AvgPool1d, 1, "Averages each window of a sequence."),
    ("maxpool2d", nn.MaxPool2d, 2, "Takes the maximum over each window of an image."),
    ("avgpool2d", nn.AvgPool2d, 2, "Averages each window of an image."),
):
    register_layer(
        LayerKind(
            name=_name,
            build=lambda shape, params, _c=_ctor, _r=_rank: _c(
                kernel_size=_int_pair(params.get("kernel_size", 2))[:_r]
                if _r > 1
                else _int_pair(params.get("kernel_size", 2))[0],
                stride=(
                    _int_pair(params.get("stride") or params.get("kernel_size", 2))[:_r]
                    if _r > 1
                    else _int_pair(params.get("stride") or params.get("kernel_size", 2))[0]
                ),
            ),
            shape=_pool_shape(_rank),
            doc=_doc,
            params={"kernel_size": 2, "stride": None},
            ranks=(_rank + 1,),
            category="pooling",
        )
    )


# ── Shape planning ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LayerPlan:
    """What one layer does to the running shape, or why it cannot.

    The editor renders one row per plan: ``out_shape`` fills the shape column
    and ``error`` turns the row red, so a stack can be inspected while it is
    still invalid.
    """

    index: int
    type: str
    in_shape: Shape | None
    out_shape: Shape | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _describe(shape: Shape) -> str:
    return "(B, " + ", ".join(str(d) for d in shape) + ")"


def plan_shapes(layers: list[dict[str, Any]], input_shape: Shape) -> list[LayerPlan]:
    """Resolve the shape after each layer without constructing anything.

    Never raises: a bad layer yields a plan carrying ``error`` and stops the
    running shape, so the editor can show exactly which row broke the stack
    instead of failing the whole preview.
    """
    plans: list[LayerPlan] = []
    shape: Shape | None = tuple(int(d) for d in input_shape)

    for index, spec in enumerate(layers):
        if not isinstance(spec, dict):
            plans.append(LayerPlan(index, str(spec), shape, None, "layer must be a mapping with a 'type' key"))
            shape = None
            continue

        params = {k: v for k, v in spec.items() if k != "type"}
        name = str(spec.get("type", "")).lower().strip()

        if shape is None:
            plans.append(LayerPlan(index, name, None, None, "skipped: an earlier layer is invalid"))
            continue
        if name not in LAYER_KINDS:
            plans.append(LayerPlan(index, name, shape, None, f"unknown layer type {name!r}"))
            shape = None
            continue

        kind = LAYER_KINDS[name]
        if len(shape) not in kind.ranks:
            wanted = " or ".join(str(r) for r in kind.ranks)
            hint = " Add a flatten layer first." if kind.ranks == (1,) and len(shape) > 1 else ""
            plans.append(
                LayerPlan(
                    index,
                    name,
                    shape,
                    None,
                    f"{name} needs a rank-{wanted} input but got {_describe(shape)}.{hint}",
                )
            )
            shape = None
            continue

        try:
            out = tuple(int(d) for d in kind.shape(shape, params))
        except LayerSpecError as exc:
            plans.append(LayerPlan(index, name, shape, None, str(exc)))
            shape = None
            continue
        except KeyError as exc:
            plans.append(LayerPlan(index, name, shape, None, f"missing required parameter {exc.args[0]!r}"))
            shape = None
            continue
        except (TypeError, ValueError) as exc:
            plans.append(LayerPlan(index, name, shape, None, f"invalid parameter: {exc}"))
            shape = None
            continue

        if any(d <= 0 for d in out):
            plans.append(
                LayerPlan(
                    index,
                    name,
                    shape,
                    None,
                    f"collapses {_describe(shape)} to {_describe(out)}: the input is too small for this kernel or stride",
                )
            )
            shape = None
            continue

        plans.append(LayerPlan(index, name, shape, out))
        shape = out

    return plans


def output_shape(layers: list[dict[str, Any]], input_shape: Shape) -> Shape:
    """The shape the stack produces, or raise naming the first broken layer."""
    plans = plan_shapes(layers, input_shape)
    for plan in plans:
        if not plan.ok:
            raise LayerSpecError(f"layer {plan.index + 1} ({plan.type or 'unnamed'}): {plan.error}")
    return plans[-1].out_shape if plans else tuple(input_shape)


def build_sequential(layers: list[dict[str, Any]], input_shape: Shape) -> tuple[nn.Sequential, Shape]:
    """Construct the stack, raising ``LayerSpecError`` on the first bad layer."""
    plans = plan_shapes(layers, input_shape)
    for plan in plans:
        if not plan.ok:
            raise LayerSpecError(f"layer {plan.index + 1} ({plan.type or 'unnamed'}): {plan.error}")

    modules: list[nn.Module] = []
    for plan, spec in zip(plans, layers):
        params = {k: v for k, v in spec.items() if k != "type"}
        assert plan.in_shape is not None  # guaranteed: every plan validated above
        modules.append(LAYER_KINDS[plan.type].build(plan.in_shape, params))

    out = plans[-1].out_shape if plans else tuple(input_shape)
    assert out is not None
    return nn.Sequential(*modules), out


# ── Model ─────────────────────────────────────────────────────────────────


@register_model("sequential_spec", "custom_sequential", "visual")
class SequentialSpecModel(nn.Module):
    """A network whose architecture comes from config rather than code.

    ``input_shape`` wins when given, which is what lets a convolutional stack
    describe its (C, H, W) input; otherwise the flat ``input_dim`` the
    Lightning modules pass is used. Extra keyword arguments are swallowed
    because ``build_model`` forwards env-derived values such as ``n_actions``
    to every architecture it builds.
    """

    def __init__(
        self,
        input_dim: int | None = None,
        layers: list[dict[str, Any]] | None = None,
        input_shape: list[int] | tuple[int, ...] | None = None,
        squeeze_single_logit: bool = True,
        **_: object,
    ):
        super().__init__()
        if input_shape is not None:
            resolved: Shape = tuple(int(d) for d in input_shape)
        elif input_dim is not None:
            resolved = (int(input_dim),)
        else:
            raise LayerSpecError("SequentialSpecModel needs either input_shape or input_dim")

        specs = [dict(layer) for layer in (layers or [])]
        if not specs:
            raise LayerSpecError("no layers declared: a visually built model needs at least one layer under 'layers:'")

        self.input_shape = resolved
        self.layer_specs = specs
        self.squeeze_single_logit = bool(squeeze_single_logit)
        self.body, self.out_shape = build_sequential(specs, resolved)

    def forward(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        """Run the stack, reshaping the batch to the declared input shape.

        Loaders hand over flat (B, D) features even for a convolutional
        design, so the batch is viewed as the declared shape first. A final
        width of 1 is squeezed to (B,), which is what the shipped
        ClassificationEvalProtocol expects for a binary task.
        """
        out = self.body(x.reshape(x.size(0), *self.input_shape))
        if self.squeeze_single_logit and out.dim() == 2 and out.size(-1) == 1:
            return out.squeeze(-1)
        return out
