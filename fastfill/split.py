"""Leakage-safe train/dev/test assignment.

Rooms are linked when they share
  - a `group` (same house / scan across exports),
  - a meta.group_aliases entry (e.g. several scans of one ARKit visit),
  - a furniture content key (3D-FRONT reuses whole furniture sets across houses; exports repackage rooms),
  - a layout key (same boundary and object poses up to a 90-degree rotation, whatever the assets).
Each connected component goes to one split, chosen by a stable hash; a component holding an eval-only room
(meta.eval_only) goes to test as a whole.
"""
import hashlib
import math

from fastfill.scene import num, rot90

# Procedurally generated sources share one asset library across all plans: equal furniture sets there are not
# repackaged rooms, and linking them fuses thousands of rooms into one component.
GROUP_ONLY = {"MansionWorld"}


def content_key(room, step=0.05):
    """Rotation/translation-invariant fingerprint: multiset of sorted object dims (5 cm grid)."""
    if len(room["objects"]) < 3 or room.get("source") in GROUP_ONLY:
        return None
    dims = sorted(tuple(sorted(round(v / step) for v in o["size"])) for o in room["objects"])
    return hashlib.sha1(repr(dims).encode()).hexdigest()


def layout_key(room):
    """Asset-independent design fingerprint: boundary + poses (cm, integer degrees), min over 4 rotations."""
    if not room.get("boundary"):
        return None
    keys = []
    for k in range(4):
        r = rot90(room, k)
        poses = sorted((num(o["pos"][0]), num(o["pos"][1]), round(math.degrees(o["yaw"])) % 360) for o in r["objects"])
        keys.append(hashlib.sha1(repr((r["boundary"], poses)).encode()).hexdigest())
    return min(keys)


def assign_splits(rooms, dev=0.05, test=0.05):
    """rooms: list of IR dicts -> list of 'train' | 'dev' | 'test' (same order)."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for r in rooms:
        g = "g:" + r["group"]
        find(g)
        for key in ("c:%s" % content_key(r), "l:%s" % layout_key(r)):
            if not key.endswith("None"):
                union(g, key)
        for al in r.get("meta", {}).get("group_aliases") or []:
            union(g, "a:" + al)
    held_out = {find("g:" + r["group"]) for r in rooms if r.get("meta", {}).get("eval_only")}
    # a component is hashed by its smallest GROUP name, not by its union-find root (which may be a furniture or layout
    # key): the split of a house then does not depend on object content, and a prep change does not move houses
    name = {}
    for r in rooms:
        c = find("g:" + r["group"])
        name[c] = min(name.get(c, "g:" + r["group"]), "g:" + r["group"])
    out = []
    for r in rooms:
        c = find("g:" + r["group"])
        h = int(hashlib.sha1(name[c].encode()).hexdigest(), 16) % 10000 / 10000
        out.append("test" if c in held_out or h < test else "dev" if h < test + dev else "train")
    return out


if __name__ == "__main__":
    def room(group, sizes, poses=None, boundary=((0, 0), (5, 0), (5, 4), (0, 4)), source="S", meta=None):
        poses = poses or [(1 + i, 1, 0) for i in range(len(sizes))]
        return {"group": group, "source": source, "boundary": [list(p) for p in boundary], "meta": meta or {},
                "objects": [{"size": s, "pos": [x, y, 0], "yaw": math.radians(d)} for s, (x, y, d) in zip(sizes, poses)]}

    a = room("house1", [[2, 1.6, .5], [.5, .5, .6], [.5, .5, .6]])
    b = room("house2", [[1.6, 2, .5], [.5, .5, .6], [.5, .6, .5]])   # same furniture, other house, axes permuted
    c = room("house1", [[1, 1, 1]])                                  # same house, different room
    v1 = room("vid1", [[1, 2, 3]], meta={"group_aliases": ["visit:9"]})
    v2 = room("vid2", [[3, 2, 1.5]], meta={"group_aliases": ["visit:9"]})    # two videos of one visit
    t1 = room("h_t1", [[1, 1, 1], [2, 2, 2]], [(1, 1, 0), (3, 2, 90)])
    t2 = room("h_t2", [[.9, .9, 1.2], [2, 2, 1]], [(3, 1, 90), (2, 3, 180)],  # same design stored rotated, new assets
              boundary=((0, 0), (4, 0), (4, 5), (0, 5)))
    e1 = room("eval1", [[1, 1, 1], [1, 2, 1], [2, 2, 2]], meta={"eval_only": True})
    e2 = room("twin", [[1, 1, 1], [2, 1, 1], [2, 2, 2]])              # content twin of an eval-only room
    m1 = room("mw1", [[1, 1, 1]] * 3, source="MansionWorld")
    m2 = room("mw2", [[1, 1, 1]] * 3, [(2, 2, 0), (3, 2, 0), (4, 3, 0)], source="MansionWorld")
    rooms = [a, b, c, v1, v2, t1, t2, e1, e2, m1, m2] + [
        room(f"h{i}", [[i / 10 + .1, 1, 1]] * 3, [(1 + i / 1000, 1, 0), (2, 2, 0), (3, 3, 0)]) for i in range(2000)]
    s = assign_splits(rooms)
    assert s[0] == s[1] == s[2] and s[3] == s[4] and s[5] == s[6], s[:7]
    assert s[7] == s[8] == "test", s[7:9]
    assert content_key(m1) is None and layout_key(m1) != layout_key(m2) and layout_key(t1) == layout_key(t2)
    frac = {k: s.count(k) / len(s) for k in ("train", "dev", "test")}
    assert 0.85 < frac["train"] < 0.95 and 0.02 < frac["test"] < 0.08, frac
    assert assign_splits(rooms) == s
    print("split.py self-check ok", frac)
