"""Canonical room IR and the text protocol FastFill v1 is trained on.

IR record (one JSON line per room), all lengths in meters, right-handed Z-up, room-local frame
with the boundary AABB min corner at (0, 0):
    uid, source, subset, group          provenance; `group` = leakage identity (house / scan)
    room_type                            str | None
    boundary, boundary_type              [[x, y], ...] CCW | None ; "polygon" | "hull" | "proxy" | None
    height                               float | None
    objects: [{id, category, size: [sx, sy, sz], pos: [x, y, z], yaw, anchor, parent, desc, tilted}]
        size  extents along the object's local X/Y/Z (before yaw)
        pos   bottom center of the object box
        yaw   radians about +Z, from room +X to the object's local +X = its semantic front
        anchor "floor" | "wall" | "ceiling" | "object" | None ; parent = id of supporting object | None
    fixed   [{id, category, size, pos, yaw}] (build output only): immovable boxes already in the room that the
            model is told about but never places (doors, windows, columns, stairs, unlabelled floor boxes)
    meta    front_known (True only when the adapter verified the front convention), n_incomplete, eval_only, ...

Model I/O (spec: fastfill I/O design review, 2026-09-24; fixed geometry added 2026-09-24), one turn, compact JSON:
    system     SYSTEM_PROMPT (constant for training, evaluation and inference)
    user       {"room_type"?, "boundary_type", "boundary", "height"?, "fixed"?: [{id, size, pos, yaw}],
                "objects": [{id, size, desc?}], "constraints"?}
    assistant  {"placements": [{"id", "on"?, "pos": [x, y, z], "yaw": int deg}, ...]}
Fixed items and objects are listed by footprint area (largest first) with ids slug(category)_k over one shared
namespace; the answer lists the objects only, in their listed order.
Compared with OptiScene: polygon instead of area, one id per instance instead of "N description" groups,
local-axis size with a front convention instead of h/w/d, integer degrees instead of 2-decimal radians,
no design-rule meta prompt, no reasoning block (no reasoning data exists), explicit support ("on").
"""
import json
import math
import re

from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

SYSTEM_PROMPT = (
    "You are a room layout model. Place every listed object in the room; do not add, drop or resize objects.\n"
    "Frame: meters, right-handed, Z up, floor z=0. boundary: floor polygon [[x,y],...], counter-clockwise; "
    "boundary_type \"polygon\" = walls, \"hull\" = convex hull of a scanned floor (the real floor may be smaller). "
    "height: ceiling height. A missing key means unknown.\n"
    "fixed: boxes already in the room that cannot move (doors, windows, columns, stairs, ...), each with id, size, "
    "pos and yaw as below; keep placements clear of them and do not block doors; they are not part of the answer.\n"
    "Object: id, size [sx,sy,sz] along its local X,Y,Z (local +X = its front), optional desc.\n"
    "Output only JSON {\"placements\":[{\"id\":id,\"on\":id,\"pos\":[x,y,z],\"yaw\":deg},...]}, "
    "every id once, in the listed order. on: the object it stands on, z = that object's top; "
    "omit on for the floor, z = 0. pos: bottom center of the box. "
    "yaw: integer degrees in [0,360), counter-clockwise from room +X to the object's front.\n"
    "constraints (optional, all must hold): [\"on\",a,b]; [\"faces\",a,b] a's front within 30 degrees of the "
    "direction to b's center; [\"near\",a,b,d] footprint gap at most d; [\"against_wall\",a] one whole side of "
    "a's footprint within 0.1 of a wall; [\"between\",a,b,c] the segment between the centers of b and c crosses a."
)

HASH = re.compile(r"(?=[a-f]*\d)[0-9a-f]{8}\b")                  # SAGE asset hashes: glassnightstand5c8f77c6
GENERIC = re.compile(r"^(other\w*|objects?|unknown)( |$)")      # labels that say nothing: shown as "object"
UNKNOWN_ROOM = {"misc", "other", "other room", "unknown", "undefined", "none"}
REL = re.compile(r"[,;]?\s*\b(positioned|placed|resting|located|situated|sitting on|standing on|lying on|leaning "
                 r"against|next to|beside|in front of|on top of|on a flat surface|on the floor)\b.*$", re.I)
DANGLING = {"a", "an", "the", "and", "or", "with", "of", "in", "on", "to", "for", "at", "by", "featuring", "its"}


# labels whose first synonym is a bare modifier ('corner', 'three seat') or an Infinigen factory name
ALIASES = {"corner/side table": "side table", "three-seat / multi-seat sofa": "three seat sofa",
           "three-seat / multi-person sofa": "three seat sofa", "coffeetablefactory": "coffee table",
           "sidetablefactory": "side table", "boxcomforterfactory": "comforter", "comforterfactory": "comforter"}


def norm_cat(c):
    """Traceable category normalization (raw label stays in IR): first synonym, words only, no hash/instance no."""
    c = (c or "").lower()
    c = ALIASES.get(re.sub(r"\(.*", "", c).strip(), c)
    c = re.split(r"[,(/]", c)[0]
    c = " ".join(re.sub(r"[^a-z0-9]+", " ", c).split())
    c = " ".join(HASH.sub(" ", c).split()).split(" or ")[0]
    return re.sub(r"(?<=[a-z])(?:\s*\d+)+$", "", c)


def head(o):
    """Head noun of an object's category (last word), used by anchor rules and validator buckets."""
    w = norm_cat(o["category"]).split()
    return w[-1] if w else ""


ROOM_WORDS = {"livingroom": "living room", "diningroom": "dining room", "livingdiningroom": "living dining room",
              "masterbedroom": "master bedroom", "secondbedroom": "second bedroom", "kidsroom": "kids room",
              "storageroom": "storage room", "laundryroom": "laundry room", "elderlyroom": "elderly room"}


def norm_room_type(t):
    if not t:
        return None
    t = re.sub(r"\s*/\s*", "/", " ".join(t.lower().replace("_", " ").split())).rstrip(".").strip()
    t = " ".join(ROOM_WORDS.get(w, w) for w in t.split())
    t = re.sub(r"(?:\s+\d+)+$", "", t)                 # "meeting room 1" -> "meeting room"
    # floor / apartment / capacity tags of MansionWorld: "restroom f1", "apartment 2 storage", "meeting room 4p"
    t = " ".join(re.sub(r"\b(?:f\d+|apt\d+|apartment \d+|\d+p)\b", " ", t).split())
    t = re.sub(r"\brestrooms\b", "restroom", t)
    return None if not t or t in UNKNOWN_ROOM else t


def num(x):
    """2-decimal number as shown to the model; integral values without '.0', no '-0'."""
    v = round(float(x), 2) + 0.0
    return int(v) if v == int(v) else v


def shown_size(o):
    return [max(num(v), 0.01) for v in o["size"]]


def short_desc(d, max_words=16):
    """First sentence, no article, no placement clause (scene context, not asset), cut at a clause boundary."""
    s = re.split(r"(?<=[a-z])\.\s", d.strip())[0].rstrip(".")
    s = re.sub(r"^(a|an|the)\s+", "", s, flags=re.I)
    w = REL.sub("", s).strip(" ,;").split()
    if len(w) > max_words:
        w = w[:max_words]
        cut = [i for i, x in enumerate(w) if i >= 4 and (x.endswith(",") or x.lower() in ("with", "featuring", "and"))]
        if cut:
            w = w[:cut[-1] + 1] if w[cut[-1]].endswith(",") else w[:cut[-1]]
    while w and w[-1].lower().strip(",") in DANGLING:
        w.pop()
    return " ".join(w).strip(" ,;")


def _clean_once(b):
    p = Polygon([(round(x, 2), round(y, 2)) for x, y in b]).simplify(0.01, preserve_topology=True)
    if not p.is_valid and p.area > 0:   # cm rounding can pinch a valid ring into a self-touch: repair if harmless
        q = p.buffer(0)
        if q.geom_type == "Polygon" and not q.interiors and abs(q.area - p.area) < 0.01 * p.area:
            p = q
    if not p.is_valid or p.area <= 0:
        return None
    pts = list(orient(p, 1.0).exterior.coords)[:-1]
    k = min(range(len(pts)), key=lambda i: (pts[i][0] ** 2 + pts[i][1] ** 2, pts[i]))
    return [[num(x), num(y)] for x, y in pts[k:] + pts[:k]]


def clean_boundary(b):
    """cm rounding, duplicate/collinear removal (<=1 cm deviation), CCW, start at the vertex nearest (0,0).
    Repeated until nothing changes (simplify never drops a ring's start vertex, and the start moves), so it is
    idempotent: a request echoing a clean boundary keeps it."""
    out = _clean_once(b)
    for _ in range(8):
        nxt = _clean_once(out) if out else None
        if nxt is None or nxt == out:
            break
        out = nxt
    return out


def rot90(room, k):
    """Exact augmentation on cm-rounded rooms: rotate about the origin by k*90 deg, shift back to the min corner.
    Run BEFORE canonical."""
    if k % 4 == 0:
        return room
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k % 4]
    rot = lambda x, y: (c * x - s * y, s * x + c * y)
    b = [rot(x, y) for x, y in room["boundary"]]
    mx, my = min(p[0] for p in b), min(p[1] for p in b)
    b = [[num(x - mx), num(y - my)] for x, y in b]           # rotation keeps CCW; only the start vertex moves
    s0 = min(range(len(b)), key=lambda i: (b[i][0] ** 2 + b[i][1] ** 2, b[i]))
    def turn(o):
        x, y = rot(*o["pos"][:2])
        return {**o, "pos": [num(x - mx), num(y - my), o["pos"][2]], "yaw": (o["yaw"] + k * math.pi / 2) % (2 * math.pi)}
    out = {**room, "boundary": b[s0:] + b[:s0], "objects": [turn(o) for o in room["objects"]]}
    if room.get("fixed"):
        out["fixed"] = [turn(o) for o in room["fixed"]]
    return out


def slug(category):
    nc = norm_cat(category)
    return "object" if not nc or GENERIC.match(nc) else nc.replace(" ", "_")


def canonical(room):
    """Order by footprint area desc (ties: category, size); identical instances by (z, x, y) when poses exist
    (training), else keep input order (inference; the sort is stable). ids become slug(norm_cat)_k, counted over
    the fixed items first, then the objects, so no id is shared between the two lists."""
    def key(o):
        s = [max(round(v, 2), 0.01) for v in o["size"]]    # = shown_size (training sizes are already shown_size)
        k = (-round(s[0] * s[1], 4), norm_cat(o["category"]), [-v for v in s])
        return k + ((round(o["pos"][2], 2), round(o["pos"][0], 2), round(o["pos"][1], 2)) if "pos" in o else ())
    fixed = sorted(room.get("fixed") or [], key=key)
    objs = sorted(room["objects"], key=key)
    count, new = {}, {}
    for o in fixed + objs:
        sl = slug(o["category"])
        count[sl] = count.get(sl, 0) + 1
        new[o["id"]] = f"{sl}_{count[sl]}"
    out = {**room, "objects": [{**o, "id": new[o["id"]], "parent": new.get(o.get("parent"))} for o in objs]}
    if fixed:
        out["fixed"] = [{**o, "id": new[o["id"]]} for o in fixed]
    return out


def user_json(room, constraints=None, with_desc=True):
    d = {}
    rt = norm_room_type(room.get("room_type"))
    if rt:
        d["room_type"] = rt
    d["boundary_type"] = room["boundary_type"]
    d["boundary"] = room["boundary"]
    if room.get("height"):
        d["height"] = num(room["height"])
    if room.get("fixed"):
        d["fixed"] = [{"id": o["id"], "size": shown_size(o), "pos": [num(v) for v in o["pos"]],
                       "yaw": int(round(math.degrees(o["yaw"]))) % 360} for o in room["fixed"]]
    objs = []
    for o in room["objects"]:
        e = {"id": o["id"], "size": shown_size(o)}
        desc = short_desc(o["desc"]) if with_desc and o.get("desc") else ""
        if desc:                     # a desc that is only a placement clause shortens to ""
            e["desc"] = desc
        objs.append(e)
    d["objects"] = objs
    if constraints:
        d["constraints"] = constraints
    return json.dumps(d, ensure_ascii=False, separators=(",", ":"))


def target_json(room):
    out = []
    for o in room["objects"]:
        p = {"id": o["id"]}
        if o.get("parent"):
            p["on"] = o["parent"]
        p["pos"] = [num(v) for v in o["pos"]]
        p["yaw"] = int(round(math.degrees(o["yaw"]))) % 360
        out.append(p)
    return json.dumps({"placements": out}, ensure_ascii=False, separators=(",", ":"))


def messages(room, constraints=None, with_desc=True, with_target=True):
    m = [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": user_json(room, constraints, with_desc)}]
    if with_target:
        m.append({"role": "assistant", "content": target_json(room)})
    return m


def prompt_text(tok, msgs):
    """Chat-template prompt ending right where the assistant answer starts (Qwen3 thinking off)."""
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def _isnum(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


MAX_COORD = 1e3


def parse(text, ids):
    """Never raises. -> (placements {id: {pos, yaw (rad), on}}, error | None).
    error: truncated_or_bad_json | bad_json | bad_item | bad_number | duplicate_id | missing_ids | extra_ids |
    bad_support_ref (unknown, self or cyclic 'on')."""
    s = text.split("</think>", 1)[-1] if isinstance(text, str) else ""
    i = s.find("{")
    try:
        items = json.JSONDecoder().raw_decode(s, i)[0]["placements"] if i >= 0 else None
    except (ValueError, KeyError, TypeError, IndexError, RecursionError):
        return {}, "truncated_or_bad_json"
    if not isinstance(items, list):
        return {}, "bad_json"
    out, err = {}, None
    for p in items:
        pos, yaw, on = (p.get("pos"), p.get("yaw"), p.get("on")) if isinstance(p, dict) else (None, None, None)
        if not (isinstance(p, dict) and isinstance(p.get("id"), str) and isinstance(pos, list) and len(pos) == 3
                and all(map(_isnum, pos)) and _isnum(yaw) and (on is None or isinstance(on, str))):
            err = err or "bad_item"
            continue
        try:
            vals = [float(v) for v in pos + [yaw]]          # OverflowError on absurdly long integers
        except OverflowError:
            vals = [math.inf]
        if not all(math.isfinite(v) and abs(v) < MAX_COORD for v in vals):   # 1 km: nothing in a room is further
            err = err or "bad_number"
            continue
        if p["id"] in out:
            err = err or "duplicate_id"
            continue
        out[p["id"]] = {"pos": vals[:3], "yaw": math.radians(vals[3] % 360), "on": on}
    if err is None and set(out) != set(ids):
        err = "missing_ids" if set(ids) - set(out) else "extra_ids"
    if err is None:
        for pid, p in out.items():
            seen, cur = {pid}, p["on"]
            while cur is not None:
                if cur not in out or cur in seen:
                    return out, "bad_support_ref"
                seen.add(cur)
                cur = out[cur]["on"]
    return out, err


def gt_placements(room):
    """Reference layout in parse() form (evaluate --gt-only)."""
    return {o["id"]: {"pos": o["pos"], "yaw": o["yaw"], "on": o.get("parent")} for o in room["objects"]}


def apply(room, pl):
    """Predicted layout in IR form; parents come ONLY from the prediction; missing objects are dropped."""
    return {**room, "objects": [{**o, "pos": pl[o["id"]]["pos"], "yaw": pl[o["id"]]["yaw"],
                                 "parent": pl[o["id"]]["on"]} for o in room["objects"] if o["id"] in pl]}


def footprint(o):
    """Oriented footprint corners [(x, y)] x4 of an upright object."""
    c, s = math.cos(o["yaw"]), math.sin(o["yaw"])
    hx, hy = o["size"][0] / 2, o["size"][1] / 2
    x, y = o["pos"][0], o["pos"][1]
    return [(x + c * dx - s * dy, y + s * dx + c * dy) for dx, dy in ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy))]


if __name__ == "__main__":
    assert norm_cat("toilet, can, commode") == "toilet" and norm_cat("ceiling lamp(traffic light)") == "ceiling lamp"
    assert norm_cat("pen_1") == "pen" == norm_cat("pen2") and norm_cat("2 seat sofa") == "2 seat sofa"
    assert norm_cat("Lounge Chair / Cafe Chair") == "lounge chair" and norm_cat("King-size Bed") == "king size bed"
    assert norm_cat("glassnightstand5c8f77c6") == "glassnightstand" and norm_cat("plant_or_flower_pot") == "plant"
    assert norm_room_type("Bedroom / Hotel") == "bedroom/hotel" and norm_room_type("Misc.") is None
    assert norm_room_type("LivingRoom") == "living room" == norm_room_type("living_room")
    assert norm_room_type("Meeting Room 1") == "meeting room"
    assert parse('{"placements":[{"id":"a","pos":[' + '9' * 400 + ',1,0],"yaw":0}]}', ["a"])[1] == "bad_number"
    assert short_desc("A wooden table positioned next to a rectangular object. It is old.") == "wooden table"
    assert num(0.0) == 0 and num(-0.001) == 0 and num(2.704) == 2.7
    assert clean_boundary([[0, 0], [2, 0], [4, 0], [4, 3], [4, 3], [0, 3]]) == [[0, 0], [4, 0], [4, 3], [0, 3]]
    wob = [[0, 0], [2.004, 0.006], [4, 0], [4.006, 1.504], [4, 3], [0, 3]]      # near-collinear after cm rounding
    assert clean_boundary(clean_boundary(wob)) == clean_boundary(wob)

    ok = '{"placements":[{"id":"desk_1","pos":[1,1,0],"yaw":90},{"id":"lamp_1","on":"desk_1","pos":[1,1,0.75],"yaw":0}]}'
    ids = ["desk_1", "lamp_1"]
    pl, err = parse("<think>\n\n</think>\n\n" + ok + "\nnote {x}", ids)
    assert err is None and abs(pl["desk_1"]["yaw"] - math.pi / 2) < 1e-9 and pl["lamp_1"]["on"] == "desk_1"
    for bad, want in (('"on":"lamp_1"', "bad_support_ref"), ('"on":"sofa_1"', "bad_support_ref"),
                      ('"on":["desk_1"]', "bad_item"), ('"on":{"id":"desk_1"}', "bad_item")):
        assert parse(ok.replace('"on":"desk_1"', bad), ids)[1] == want, bad
    assert parse(ok.replace('"yaw":0}', '"yaw":NaN}'), ids)[1] == "bad_number"
    assert parse(ok.replace('"yaw":90', '"yaw":true'), ids)[1] == "bad_item"
    assert parse(ok.replace('"pos":[1,1,0]', '"pos":"110"'), ids)[1] == "bad_item"
    assert parse('{"placements":' + "[" * 30000 + "]" * 30000 + "}", ids)[1] in ("truncated_or_bad_json", "bad_item")
    assert parse(ok[:70], ids)[1] == "truncated_or_bad_json" and parse(None, ids)[1] == "bad_json"
    assert parse('{"placements":[{"id":"desk_1","pos":[1,1,0],"yaw":0}]}', ids)[1] == "missing_ids"

    room = {"room_type": "Bed_Room", "boundary_type": "polygon", "boundary": [[0, 0], [4, 0], [4, 3], [0, 3]], "height": 2.7,
            "objects": [{"id": s, "category": "chair", "size": [.5, .5, .9], "pos": [x, y, 0], "yaw": 0}
                        for s, x, y in (("a", 1, 1), ("b", 3, 1), ("c", 2, 2))]
            + [{"id": "d", "category": "desk", "size": [1.2, .6, .75], "pos": [2, .3, 0], "yaw": math.pi / 2},
               {"id": "l", "category": "lamp", "size": [.2, .2, .4], "pos": [2, .3, .75], "yaw": 0, "parent": "d"}]}
    r4 = room
    for k in range(4):   # rot90 before canonical keeps the (z, x, y) numbering of identical chairs
        c = canonical(rot90(room, k))
        chairs = [o for o in c["objects"] if o["category"] == "chair"]
        assert [o["pos"][:2] for o in chairs] == sorted(o["pos"][:2] for o in chairs), k
        r4 = rot90(r4, 1)
    assert r4["boundary"] == room["boundary"] and [o["pos"] for o in r4["objects"]] == [o["pos"] for o in room["objects"]]
    c = canonical(room)
    assert [o["id"] for o in c["objects"]] == ["desk_1", "chair_1", "chair_2", "chair_3", "lamp_1"]
    m = messages(c)
    assert json.loads(m[1]["content"])["room_type"] == "bed room" and "fixed" not in json.loads(m[1]["content"])
    # fixed boxes: shared id namespace (a fixed chair takes chair_1), shown with pose, rotated with the room, not answered
    fx = {**room, "fixed": [{"id": "col", "category": "column", "size": [.3, .3, 2.7], "pos": [3.5, 2.5, 0], "yaw": 0},
                            {"id": "fc", "category": "chair", "size": [.5, .5, .9], "pos": [.5, .5, 0], "yaw": 0},
                            {"id": "u", "category": "unknown", "size": [1, .4, .8], "pos": [2, 2.8, 0], "yaw": math.pi / 2}]}
    cf = canonical(fx)
    assert [o["id"] for o in cf["fixed"]] == ["object_1", "chair_1", "column_1"], cf["fixed"]
    assert [o["id"] for o in cf["objects"]] == ["desk_1", "chair_2", "chair_3", "chair_4", "lamp_1"]
    uj = json.loads(user_json(cf))
    assert uj["fixed"][1] == {"id": "chair_1", "size": [0.5, 0.5, 0.9], "pos": [0.5, 0.5, 0], "yaw": 0} and "fixed" not in target_json(cf)
    r1 = rot90(fx, 1)                                     # (3.5, 2.5) turned by 90 deg about the origin -> (-2.5, 3.5) -> shifted
    assert r1["fixed"][0]["pos"] == [0.5, 3.5, 0] and abs(r1["fixed"][0]["yaw"] - math.pi / 2) < 1e-9, r1["fixed"][0]
    assert slug("otherprop") == "object" == slug("") == slug("Objects") and slug("floor lamp") == "floor_lamp"
    pl, err = parse(m[2]["content"], [o["id"] for o in c["objects"]])
    assert err is None and pl["lamp_1"]["on"] == "desk_1" and pl["lamp_1"]["pos"] == [2.0, 0.3, 0.75]
    assert apply(c, pl)["objects"][-1]["parent"] == "desk_1"
    fp = footprint(c["objects"][0])      # desk rotated 90 deg: 0.6 wide in x, 1.2 deep in y
    xs, ys = [p[0] for p in fp], [p[1] for p in fp]
    assert abs(max(xs) - min(xs) - 0.6) < 1e-9 and abs(max(ys) - min(ys) - 1.2) < 1e-9
    print("scene.py self-check ok")
