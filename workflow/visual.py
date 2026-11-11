"""Annotated bbox observations of the placed layout, before RoomGen mesh generation."""

from __future__ import annotations

import base64
import io
import math

from PIL import Image, ImageDraw

from .geometry import footprint


def layout_image(room: dict, placed: dict[str, dict]) -> str:
    """A PNG data URL in world x/y; the room fixes the scale for pairwise comparisons."""
    polygon = room["floor_polygon_xy_m"]
    x0, y0 = min(p[0] for p in polygon), min(p[1] for p in polygon)
    x1, y1 = max(p[0] for p in polygon), max(p[1] for p in polygon)
    scale = 640 / max(x1 - x0, y1 - y0)

    def point(x, y):
        return (80 + (x - x0) * scale, 720 - (y - y0) * scale)

    image = Image.new("RGB", (800, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 12), "Annotated bbox top-down (not a mesh render)", fill="black")
    draw.text((20, 30), "World metres; arrows = object front; +x right, +y up", fill="black")
    draw.polygon([point(*p) for p in polygon], fill="#f5f4ef", outline="#222222", width=3)
    for opening in room.get("openings", []):
        a = polygon[opening["wall_index"]]
        b = polygon[(opening["wall_index"] + 1) % len(polygon)]
        length = math.dist(a, b)
        t, half = opening["position_on_wall"], opening["width_m"] / (2 * length)
        ends = [point(a[0] + (t + sign * half) * (b[0] - a[0]),
                      a[1] + (t + sign * half) * (b[1] - a[1])) for sign in (-1, 1)]
        draw.line(ends, fill="#00866d" if opening["type"] == "door" else "#4196cf", width=7)
        draw.text(point(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])),
                  opening["type"], fill="black")
    fixed = {obj["id"]: {**obj, "target_size_local_m": obj["size_local_m"]}
             for obj in room.get("fixed_objects", [])}
    boxes = {**fixed, **placed}
    for ident in sorted(boxes, key=lambda i: (boxes[i]["bottom_center_m"][2], i)):
        box = boxes[ident]
        draw.polygon([point(x, y) for x, y in footprint(box).exterior.coords],
                     fill="#c2c5c9" if ident in fixed else "#d5e5f5", outline="#315575", width=2)
        x, y = box["bottom_center_m"][:2]
        yaw = box["yaw_rad"]
        radius = min(box["target_size_local_m"][:2]) * .4
        front = (x + math.cos(yaw) * radius, y + math.sin(yaw) * radius)
        draw.line([point(x, y), point(*front)], fill="#b23830", width=3)
        for angle in (yaw + 2.5, yaw - 2.5):
            draw.line([point(*front), point(front[0] + math.cos(angle) * radius * .4,
                                           front[1] + math.sin(angle) * radius * .4)], fill="#b23830", width=2)
        draw.text(point(x, y), ident, fill="#142333", anchor="mm")
    draw.line([(30, 750), (65, 750)], fill="black", width=2)
    draw.line([(30, 750), (30, 715)], fill="black", width=2)
    draw.text((68, 744), "+x", fill="black")
    draw.text((20, 697), "+y", fill="black")
    draw.text((120, 755), f"Room extent: x [{x0:g}, {x1:g}], y [{y0:g}, {y1:g}] m", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
