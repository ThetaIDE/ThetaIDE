"""Thin long series for drawing. Headless and dependency-free.

Each bucket keeps its lowest and highest value (in x order), so spikes and dips survive,
and the latest point is always kept. Series at or under the limit come back unchanged.
"""

from __future__ import annotations

import math

MAX_DRAWN_POINTS = 2000


def thin_by(points: list, value, limit: int = MAX_DRAWN_POINTS) -> list:
    """At most about ``limit`` of ``points``, bucketed by the number ``value(point)`` returns."""
    if len(points) <= limit:
        return points
    size = math.ceil(len(points) / (limit // 2))
    kept = []
    for start in range(0, len(points), size):
        bucket = points[start : start + size]
        low = min(range(len(bucket)), key=lambda i: value(bucket[i]))
        high = max(range(len(bucket)), key=lambda i: value(bucket[i]))
        kept.extend(bucket[i] for i in sorted({low, high}))
    if kept[-1] is not points[-1]:
        kept.append(points[-1])  # the latest value is always drawn
    return kept


def thin_xy(points: list, limit: int = MAX_DRAWN_POINTS) -> list:
    """Thin ``(x, y)`` pairs by their y value."""
    return thin_by(points, lambda point: point[1], limit)
