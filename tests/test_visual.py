"""Annotated bbox observations have a stable world frame and no mesh claims."""

import base64
import copy
import io
import math
from unittest import mock

from PIL import Image, ImageDraw

from workflow.visual import layout_image
from workflow.solver import _pose


def test_image_uses_world_xy_rotated_footprints_ids_and_fixed_objects():
    room = {"floor_polygon_xy_m": [[-2, -1], [2, -1], [2, 3], [-2, 3]],
            "openings": [{"type": "door", "wall_index": 0, "position_on_wall": .5, "width_m": 1}],
            "fixed_objects": [{"id": "column", "bottom_center_m": [-1, 2, 0],
                               "size_local_m": [.25, .25, 2], "yaw_rad": 0}]}
    placed = {"desk": _pose("desk", [2, 1, .75], 1, 1, 0, math.pi / 2, "floor")}
    frozen = copy.deepcopy((room, placed))
    labels, polygons = [], []
    original_text, original_polygon = ImageDraw.ImageDraw.text, ImageDraw.ImageDraw.polygon

    def text(self, xy, value, *args, **kwargs):
        labels.append(value)
        return original_text(self, xy, value, *args, **kwargs)

    def polygon(self, xy, *args, **kwargs):
        polygons.append(xy)
        return original_polygon(self, xy, *args, **kwargs)

    with mock.patch.object(ImageDraw.ImageDraw, "text", text), \
            mock.patch.object(ImageDraw.ImageDraw, "polygon", polygon):
        url = layout_image(room, placed)
    image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert image.size == (800, 800)
    assert {"desk", "column", "+x", "+y"} <= set(labels)
    assert any("bbox" in label for label in labels)
    assert any("door" in label for label in labels)
    # For the desk yaw=pi/2 its long local x/depth axis points along world +y.
    desk = polygons[-1]
    assert max(y for x, y in desk) - min(y for x, y in desk) > max(x for x, y in desk) - min(x for x, y in desk)
    assert (room, placed) == frozen
    assert layout_image(room, placed) == url
