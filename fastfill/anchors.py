"""Object scope for training: drop structure, infer support anchors, keep a support-closed subset.

anchor = "floor" | "object" | "wall" | "ceiling" | "fixed". Adapters set anchor/parent from source evidence when the
source has it (placement flags, explicit parent ids); otherwise `infer_anchors` derives them from boxes:
  ceiling  hanging categories (pendant, chandelier, ...) whatever their z or source anchor (one at z=0 is
           misplaced), or top within 10 cm of the room height above 1 m
  floor    bottom at most FLOOR_Z = 10 cm above z=0 (sunk boxes too, so build.prep can drop and count them), or
           floor furniture (bed, table, sofa, ...) at most FURNITURE_FLOOR_Z = 50 cm up with nothing under it
           (annotation z-noise, boxes that start above the legs; build.prep snaps them to the floor)
  object   bottom within 5 cm of the top of a floor/object box whose footprint contains its center;
           seats, beds and rugs are never parents (bbox top = backrest / headboard; a rug is not a surface)
  wall     wall art (painting, mirror, ...) with no support below, or anything else elevated without one
  fixed    floor furniture that can hang (cabinet, shelf, vanity...) floating <= 50 cm with nothing below in a
           designer (polygon) room: a wall-hung unit, out of scope to place, shown to the model as a fixed box
Box-level contact is weak evidence of support; build.py counts evidence and inferred supports separately.
`fixed_geometry` picks the structure boxes the model is told about (never placed): doors, windows, columns, stairs,
fireplaces, radiators, partitions, low beams... below FIXED_MAX_Z; surface finishes, sockets/drains, curtains and
slabs covering most of the room are left out, and a "wall" box only counts when it stands inside the room.
"""
import math
import re

from shapely.geometry import Point, Polygon

from fastfill.scene import footprint, head, norm_cat

FLOOR_Z = 0.10   # bottoms up to 10 cm above the floor count as standing on it (build.prep snaps them to z=0)
# Strong architectural words mark structure anywhere in the name; floor/wall/ceiling/column are usually modifiers
# (plant_floor, floor lamp, wall_shelving_unit, locker_column) and mark structure only in purely architectural
# names ("wall", "Wall design", "floor mold").
STRUCTURE = re.compile(r"\b(door|window|curtain|drape|blind|beam|pillar|stair|staircase|stairway|stairwell|railing|rail|"
                       r"handrail|banister|guardrail|balustrade|baseboard|radiator|opening|doorframe|partition|fireplace|"
                       r"chimney|otherstructure|structure|doorway|arch|archway|bulkhead|niche|fence|step|ramp|elevator|duct|"
                       r"drain|pipe|socket|outlet|tile)s?\b")
# furniture whose name borrows a structure word: fireplace_console, pillar_candle, clothes rail, portable_partition_panel
NOT_STRUCTURE = re.compile(r"\b(lamp|light|fan|cabinet|shelf|mat|rug|towel|chair|table|console|stool|candle|figurine|box|"
                           r"plant|planter|basket|display|station|clothes|freestanding|portable|ladder|bench|sofa|couch|"
                           r"bed|wardrobe|desk|seat|bookcase|bookshelf|dresser|sideboard|ottoman)s?\b")
SURFACE = r"(wall|floor|ceiling|column)s?"
ARCH = (r"(panel|paneling|design|combination|mold|molding|moulding|background|trim|skirting|board|tile|cladding|covering|"
        r"decoration|sticker|styling|decorative|decor|structural|support|concrete|low|half|suspended)s?")
HANGING = re.compile(r"\b(pendant|chandelier|ceiling|downlight|hanging)s?\b")
NOT_HANGING = re.compile(r"\b(chair|stool|seat|swing)\b")          # a hanging chair stands on its own frame
# soft covers draped over a bed or sofa (their boxes cover the bed): not objects to place, not floor obstacles
SOFT_COVERS = {"bedding", "comforter", "blanket", "quilt", "duvet", "bedspread", "bedcover", "coverlet"}
# floor furniture whose box bottom sits up to FURNITURE_FLOOR_Z high with nothing under it stands on the floor
# (annotation z-noise, boxes that start above the legs); without this it fell through to 'wall' and vanished
FLOOR_FURNITURE = {"bed", "sofa", "couch", "loveseat", "sectional", "table", "desk", "chair", "armchair", "stool", "bench",
                   "ottoman", "cabinet", "wardrobe", "dresser", "bookcase", "bookshelf", "sideboard", "nightstand",
                   "fridge", "refrigerator", "oven", "stove", "dishwasher", "washer", "dryer", "piano", "tub",
                   "bathtub", "toilet", "console", "stand", "chest", "armoire", "shelf", "cupboard"}
FURNITURE_FLOOR_Z = 0.50   # floating up to here with nothing below: a floor piece (snapped to z=0 by build.prep)
FURNITURE_SUNK_Z = 0.30    # any floor box sunk up to here: same (deeper boxes belong to a lower storey)
# furniture that cannot hang on a wall: floating with nothing below it is on the floor with a noisy z anywhere
GROUND_ONLY = {"bed", "sofa", "couch", "loveseat", "sectional", "chair", "armchair", "table", "desk", "fridge",
               "refrigerator", "oven", "stove", "dishwasher", "stool", "bench", "ottoman", "piano", "tub", "bathtub",
               "washer", "dryer", "wardrobe", "armoire", "bookcase", "dresser"}
SOFT_HANGINGS = {"curtain", "curtains", "drape", "drapes", "blind", "blinds", "shade", "shades", "screening", "valance"}
# structure that has no footprint of its own: finishes, fixtures, the room's own slabs
FINISH = {"floor", "ceiling", "tile", "tiles", "baseboard", "skirting", "trim", "mold", "molding", "moulding", "panel",
          "paneling", "board", "boards", "cladding", "covering", "socket", "outlet", "drain", "pipe", "pipes", "design",
          "decoration", "sticker", "background", "wallpaper"}
FIXED_MAX_Z = 1.5          # structure whose bottom is higher does not shape the floor layout (ceiling beams, ducts)
DIVIDERS = {"partition", "divider", "screen"}   # a "partition panel" is an obstacle, not a wall finish
# scans: box z is noisy, so floating floor furniture is on the floor (not a wall-hung unit), whatever the boundary type
SCANNED = {"Scan2CAD", "MultiScan", "InternScenes_arkit", "InternScenes_scannet", "InternScenes_mp3d",
           "InternScenes_3rscan"}


def hanging(category):
    nc = norm_cat(category)
    return bool(HANGING.search(nc)) and not NOT_HANGING.search(nc)


WALL_ART = re.compile(r"\b(painting|picture|poster|artwork|art|mirror|whiteboard|blackboard|frame)s?\b")
FLAT = {"rug", "carpet", "mat", "doormat", "bathmat", "placemat", "playmat", "floormat", "yogamat", "exercisemat",
        "fitnessmat"}
FLAT |= {w + "s" for w in FLAT}
NO_TOP = {"sofa", "couch", "loveseat", "sectional", "bed", "armchair", "chair", "bench"} | FLAT  # not a support surface


def is_structure(category):
    c = re.sub(r"[^a-z]+", " ", (category or "").lower())
    if NOT_STRUCTURE.search(c):
        return False
    return bool(STRUCTURE.search(c)) or (bool(re.search(rf"\b{SURFACE}\b", c))
                                         and not re.sub(rf"\b({SURFACE}|{ARCH})\b", " ", c).strip())


def infer_anchors(room, tol=0.05):
    objs = room["objects"]
    fps = {o["id"]: Polygon(footprint(o)) for o in objs}
    height = room.get("height")
    hull = room.get("boundary_type") == "hull" or room.get("source") in SCANNED
    for o in sorted(objs, key=lambda o: o["pos"][2]):
        if hanging(o["category"]):       # before the floor rule and any source anchor: a chandelier at z=0 is misplaced
            o["anchor"] = "ceiling"
            continue
        if o.get("anchor"):
            continue
        z, nc = o["pos"][2], norm_cat(o["category"])
        if z <= FLOOR_Z:                 # one-sided: sunk boxes stay floor so build.prep drops and counts them
            o["anchor"] = "floor"
            continue
        c = Point(o["pos"][0], o["pos"][1])
        below = [p for p in objs if p is not o and p.get("anchor") in ("floor", "object") and head(p) not in NO_TOP
                 and abs(p["pos"][2] + p["size"][2] - z) < tol and fps[p["id"]].contains(c)]
        if below:                        # a frame standing on a dresser is on it; art above a sofa is not (NO_TOP)
            o["anchor"], o["parent"] = "object", max(below, key=lambda p: p["pos"][2] + p["size"][2])["id"]
            o["anchor_inferred"] = True
        elif WALL_ART.search(nc):
            o["anchor"] = "wall"
        elif z <= FURNITURE_FLOOR_Z and head(o) in FLOOR_FURNITURE:
            # nothing under it: on the floor with a noisy z (scans; anything that cannot hang), or a wall-hung
            # unit in a designer room (vanity, floating TV console): shown as fixed, not placed
            o["anchor"] = "floor" if hull or head(o) in GROUND_ONLY else "fixed"
        elif height and z + o["size"][2] > height - 2 * tol and z > 1.0:
            o["anchor"] = "ceiling"
        else:
            o["anchor"] = "wall"
    return room


def fixed_geometry(room, band=0.10, intrusion=0.05):
    """Structure boxes to show the model as immovable (see module doc). Returns copies with tilted/anchor cleared.
    Which of them survive next to the reference layout (covered, outside, slabs) is build.prep's decision."""
    out = []
    shell = Polygon(room["boundary"]).buffer(0) if room.get("boundary") else None
    inner = shell.buffer(-band) if shell is not None else None
    hull = room.get("boundary_type") == "hull"
    for o in room["objects"]:
        if not structural(o):
            continue
        words, h = set(norm_cat(o["category"]).split()), head(o)
        finish = (h in FINISH or h.rstrip("s") in FINISH) and not words & DIVIDERS
        if finish or words & SOFT_HANGINGS or o["pos"][2] >= FIXED_MAX_Z:
            continue
        fp = Polygon(footprint(o))
        if shell is not None and fp.area >= 0.8 * shell.area:            # a slab under another name
            continue
        # a "wall" box is the room's own wall unless it stands inside a polygon room (a partition); in a hull room
        # the walls stand inside the hull by construction, so none of them is a partition
        if h == "wall" and (hull or inner is None or fp.intersection(inner).area < intrusion):
            continue
        out.append({k: v for k, v in o.items() if k not in ("anchor", "parent", "tilted", "anchor_inferred")})
    return out


def structural(o):
    """Source-flagged architecture, or a structure name on something that does not stand on another object (a
    power 'outlet' strip or a lever 'arch' file on a desk is an object, not architecture)."""
    return bool(o.get("structure")) or (is_structure(o["category"]) and not o.get("parent"))


def annotate(room):
    """Structure and soft covers removed, anchors inferred: every remaining box with anchor/parent set."""
    objs = [dict(o) for o in room["objects"] if not (structural(o) or head(o) in SOFT_COVERS)]
    return infer_anchors({**room, "objects": objs})["objects"]


def closed(objs, keep=("floor", "object")):
    """Objects whose anchor is in `keep` and whose support chain is entirely kept, in input order."""
    kept, ids = [], set()
    for o in sorted(objs, key=lambda o: o["pos"][2]):   # parents sit lower than children
        if o["anchor"] in keep and (o.get("parent") is None or o["parent"] in ids):
            kept.append(o)
            ids.add(o["id"])
    order = {o["id"]: k for k, o in enumerate(objs)}
    return sorted(kept, key=lambda o: order[o["id"]])


def scope_objects(room, keep=("floor", "object")):
    """Structure removed, anchors inferred, objects outside `keep` dropped together with their dependants."""
    return {**room, "objects": closed(annotate(room), keep)}


if __name__ == "__main__":
    room = {"height": 2.6, "objects": [
        {"id": "desk", "category": "desk", "size": [1.2, .6, .75], "pos": [1, 1, 0], "yaw": 0},
        {"id": "lamp", "category": "table lamp", "size": [.2, .2, .4], "pos": [1.2, 1, .75], "yaw": 0},
        {"id": "pic", "category": "painting", "size": [.8, .05, .6], "pos": [1, 0, 1.4], "yaw": 0},
        {"id": "cl", "category": "pendant lamp", "size": [.5, .5, .3], "pos": [2, 2, 2.0], "yaw": 0},
        {"id": "sofa", "category": "sofa", "size": [2, .9, .8], "pos": [3, 3, 0], "yaw": 0},
        {"id": "art", "category": "picture frame", "size": [.6, .05, .5], "pos": [3, 3, .8], "yaw": 0},
        {"id": "vase", "category": "vase", "size": [.1, .1, .2], "pos": [1, 0, 2.0], "yaw": 0, "anchor": "object", "parent": "pic"},
        {"id": "win", "category": "window", "size": [1, .1, 1.2], "pos": [3, 0, .9], "yaw": 0}]}
    s = scope_objects(room)
    assert [o["id"] for o in s["objects"]] == ["desk", "lamp", "sofa"], s
    assert s["objects"][1]["parent"] == "desk" and s["objects"][1]["anchor"] == "object"
    a = {o["id"]: o["anchor"] for o in infer_anchors({**room, "objects": [dict(o) for o in room["objects"]]})["objects"]}
    assert a["pic"] == "wall" and a["cl"] == "ceiling" and a["art"] == "wall", a   # art above a sofa is not "on" it
    assert is_structure("window") and is_structure("door(screen door, screen)") and not is_structure("floor lamp")
    for c in ("wall", "floor", "Wall design", "ceiling combination", "floor mold", "otherstructure", "room_partition",
              "radiator", "support_column", "vertical core stair 1", "door(screen door, screen)"):
        assert is_structure(c), c
    for c in ("plant_floor", "floor_plant_large", "locker_column", "floor_bookshelf", "bookshelf_wall_unit",
              "floor-standing air purifier", "wall_hung_toilet", "floor lamp", "wall_cabinet", "ceiling fan", "wall_mirror"):
        assert not is_structure(c), c
    sofa = {"id": "s", "category": "sofa", "size": [2, .9, .8], "pos": [1, 1, 0.07], "yaw": 0}
    rug = {"id": "r", "category": "rug", "size": [3, 2, .02], "pos": [1, 1, 0], "yaw": 0}
    got = {o["id"]: (o["anchor"], o.get("parent")) for o in infer_anchors({"objects": [dict(sofa), dict(rug)]})["objects"]}
    assert got == {"s": ("floor", None), "r": ("floor", None)}, got   # 7 cm up = standing, not on the rug
    fx = {"boundary": [[0, 0], [5, 0], [5, 4], [0, 4]], "objects": [
        {"id": "d", "category": "door", "size": [.9, .1, 2.1], "pos": [2, 0, 0], "yaw": 0},
        {"id": "w", "category": "window", "size": [1.2, .1, 1.3], "pos": [4.98, 2, .9], "yaw": math.pi / 2},
        {"id": "c", "category": "support_column", "size": [.3, .3, 2.8], "pos": [2.5, 2, 0], "yaw": 0, "structure": True},
        {"id": "b", "category": "beam", "size": [5, .3, .3], "pos": [2.5, 2, 2.5], "yaw": 0},
        {"id": "f", "category": "floor", "size": [5, 4, .02], "pos": [2.5, 2, 0], "yaw": 0},
        {"id": "k", "category": "curtain", "size": [1.5, .1, 2.4], "pos": [4.95, 2, 0], "yaw": 0},
        {"id": "s", "category": "socket", "size": [.08, .02, .08], "pos": [1, 0.01, .3], "yaw": 0},
        {"id": "w1", "category": "wall", "size": [4, .1, 2.8], "pos": [2.5, 0.05, 0], "yaw": 0},
        {"id": "w2", "category": "wall", "size": [2, .1, 2.8], "pos": [2.5, 2, 0], "yaw": 0},
        {"id": "p", "category": "wall panels", "size": [3, .05, 2.8], "pos": [1.5, 0.02, 0], "yaw": 0},
        {"id": "sc", "category": "window screening", "size": [1.5, .05, 2.4], "pos": [2, 3.98, 0], "yaw": 0},
        {"id": "t", "category": "table", "size": [1.2, .6, .75], "pos": [1, 1, 0], "yaw": 0}]}
    assert [o["id"] for o in fixed_geometry(fx)] == ["d", "w", "c", "w2"], fixed_geometry(fx)
    assert [o["id"] for o in fixed_geometry({**fx, "boundary_type": "hull"})] == ["d", "w", "c"]   # hull: walls are not partitions
    assert not is_structure("step ladder") and is_structure("styling column") and is_structure("otherstructure")
    assert is_structure("elevator shaft") and is_structure("banister") and not is_structure("floor lamp")
    assert not is_structure("elevator_lobby_bench") and is_structure("stair_railing") and is_structure("bedroom_door")
    panel = {"boundary": fx["boundary"], "objects": [{"id": "pp", "category": "low_partition_panel", "size": [.3, 1, 1.7],
                                                      "pos": [2.5, 2, 0], "yaw": 0}, {"id": "wp", "category": "wall panels",
                                                      "size": [3, .05, 2.8], "pos": [1.5, 0.02, 0], "yaw": 0}]}
    assert [o["id"] for o in fixed_geometry(panel)] == ["pp"]            # a partition panel is fixed, a wall panel is not
    # a frame standing on a dresser is on it; a wall-hung cabinet floats -> fixed in a polygon room, floor in a hull
    dr = {"id": "dr", "category": "dresser", "size": [1, .5, .8], "pos": [1, 1, 0], "yaw": 0}
    fr = {"id": "fr", "category": "picture frame", "size": [.2, .05, .25], "pos": [1, 1, .8], "yaw": 0}
    cb = {"id": "cb", "category": "bathroom cabinet", "size": [.6, .4, .5], "pos": [3, 3, .35], "yaw": 0}
    bd = {"id": "bd", "category": "bed", "size": [2, 1.6, .5], "pos": [3, 1, .35], "yaw": 0}
    got = {o["id"]: (o["anchor"], o.get("parent")) for o in annotate({"boundary_type": "polygon", "objects": [dr, fr, cb, bd]})}
    assert got == {"dr": ("floor", None), "fr": ("object", "dr"), "cb": ("fixed", None), "bd": ("floor", None)}, got
    got = {o["id"]: o["anchor"] for o in annotate({"boundary_type": "hull", "objects": [dict(cb)]})}
    assert got == {"cb": "floor"}, got
    assert annotate({"source": "Scan2CAD", "boundary_type": "polygon", "objects": [dict(cb)]})[0]["anchor"] == "floor"
    strip = {"id": "ps", "category": "power_outlet_strip", "size": [.3, .06, .04], "pos": [1, 1, .8], "yaw": 0,
             "anchor": "object", "parent": "dr"}
    assert [o["id"] for o in annotate({"objects": [dict(dr), strip]})] == ["dr", "ps"] and not fixed_geometry(
        {"boundary": fx["boundary"], "objects": [dict(dr), strip]})
    print("anchors.py self-check ok")
