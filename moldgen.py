#!/usr/bin/env python3
"""
moldgen.py - Generate a 2-part 3D-printable mold around a watertight mesh.

Pipeline (all boolean work done with the Manifold library, which is what
makes this robust and fast compared to naive mesh-boolean approaches):

  1. Load the model, rotate it so the chosen "up" axis becomes +Z.
  2. Build a rectangular blank around its bounding box, padded by `wall`.
  3. Subtract the model from the blank -> mold body with a cavity.
  4. Bore the pour funnel (and optional vent) before splitting, so a
     vertical seam naturally divides the channel between both halves.
  5. Split the body on the seam plane (--seam-axis: z = horizontal/
     classic clamshell, x or y = vertical, for side undercuts) -> two
     halves.
  6. Add registration pins (male on the -axis half, socket on the +axis
     half) across the seam.
  7. Rotate both halves back to the original orientation and export as STL.

ASSUMES THE INPUT MESH IS ALREADY WATERTIGHT/MANIFOLD.
No repair step yet - that's the natural next thing to add once this
core pipeline is solid. A non-watertight input will raise an error
or silently produce a broken (non-manifold) result.

Usage:
    python moldgen.py model.stl --out mold --wall 3 --up z --seam-axis z --seam 0.5

Requires: pip install manifold3d trimesh numpy
"""

import argparse
import re
import sys
import numpy as np
import trimesh
import manifold3d as mf


# --------------------------------------------------------------------------
# Conversion helpers: trimesh <-> manifold3d
# --------------------------------------------------------------------------

def trimesh_to_manifold(tm: trimesh.Trimesh) -> mf.Manifold:
    """Convert a trimesh.Trimesh into a manifold3d.Manifold.

    Assumes `tm` is already a closed, watertight, consistently-wound mesh.
    """
    tm = tm.copy()
    tm.merge_vertices()
    verts = np.ascontiguousarray(tm.vertices, dtype=np.float32)
    tris = np.ascontiguousarray(tm.faces, dtype=np.uint32)
    mesh = mf.Mesh(vert_properties=verts, tri_verts=tris)
    man = mf.Manifold(mesh)
    if man.is_empty():
        raise ValueError(
            "Input mesh did not convert to a valid manifold "
            f"(status={man.status()}). It likely isn't watertight/"
            "consistently oriented. Repair it before running moldgen."
        )
    return man


def manifold_to_trimesh(man: mf.Manifold) -> trimesh.Trimesh:
    mesh = man.to_mesh()
    verts = np.array(mesh.vert_properties)[:, :3]
    tris = np.array(mesh.tri_verts)
    tm = trimesh.Trimesh(vertices=verts, faces=tris, process=False)
    # Long chains of booleans can leave a handful of zero-area "sliver"
    # faces behind (degenerate leftovers at tangent/edge-case contacts,
    # confirmed with heavily-combined option sets) - individually harmless
    # but they show up as tiny disconnected junk shells in the output file.
    # Manifold itself reports no error at any step; this is purely an
    # export-side cleanup.
    tm.update_faces(tm.nondegenerate_faces())
    tm.remove_unreferenced_vertices()
    return tm


def verify_watertight(path: str) -> bool:
    """Check watertightness by reloading the file that was just exported,
    rather than trusting the pre-export in-memory object - confirmed during
    development that the two can disagree in both directions (STL is a
    triangle-soup format with no shared-vertex topology, so export/reload
    reconstructs connectivity from scratch; this is the check that reflects
    what a slicer will actually load)."""
    return trimesh.load(path).is_watertight


# --------------------------------------------------------------------------
# Axis handling: internally we always work with "up" = +Z
# --------------------------------------------------------------------------

_AXIS_TO_Z = {
    "z": np.eye(3),
    # rotate so that +X becomes +Z
    "x": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float),
    # rotate so that +Y becomes +Z
    "y": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float),
}


def rotate_mesh(tm: trimesh.Trimesh, rot3x3: np.ndarray) -> trimesh.Trimesh:
    tm = tm.copy()
    T = np.eye(4)
    T[:3, :3] = rot3x3
    tm.apply_transform(T)
    return tm


# --------------------------------------------------------------------------
# Generic per-axis helpers, used so pins/funnel/hollowing all work whether
# the seam plane is horizontal (perpendicular to Z, the default) or vertical
# (perpendicular to X or Y - needed for models with side undercuts rather
# than top/bottom ones).
# --------------------------------------------------------------------------

_AXES = ("x", "y", "z")


def _axis_index(axis):
    return _AXES.index(axis)


def other_axes(axis):
    """The two axes other than `axis`, in a fixed (x,y,z) order."""
    return [a for a in _AXES if a != axis]


def point_from_axes(values):
    """{'x': .., 'y': .., 'z': ..} (any subset) -> (x, y, z), 0.0 for any axis
    not given. Used to assemble a 3D point/vector/size from per-axis parts
    without hardcoding which axis is which."""
    return (values.get("x", 0.0), values.get("y", 0.0), values.get("z", 0.0))


def bbox_dict(manifold_bbox):
    """(xmin,ymin,zmin,xmax,ymax,zmax) -> {'x': (xmin,xmax), 'y': .., 'z': ..}"""
    xmin, ymin, zmin, xmax, ymax, zmax = manifold_bbox
    return {"x": (xmin, xmax), "y": (ymin, ymax), "z": (zmin, zmax)}


def cylinder_along_axis(height, r1, r2, segments, axis):
    """A cylinder extruding from the origin along +axis, like
    Manifold.cylinder does along +Z by default - for building pins/funnels
    on a seam that isn't the default horizontal (Z) one."""
    cyl = mf.Manifold.cylinder(height, r1, r2, segments, center=False)
    if axis == "z":
        return cyl
    if axis == "x":
        return cyl.rotate((0, 90, 0))
    if axis == "y":
        return cyl.rotate((-90, 0, 0))
    raise ValueError(f"bad axis {axis!r}")


# --------------------------------------------------------------------------
# Multi-part splitting: a list of seams instead of one. Each seam is still a
# full axis-aligned plane, but can now be partial - restricted to only the
# pieces already on one side of an earlier seam (T's stem: a full "-" bar,
# then a "|" stem that only splits the crown above it, leaving the trunk
# below whole). A partial split needs no new geometric primitive - it's
# just an ordinary full-plane split applied to a *subset* of the current
# pieces instead of all of them; the pieces not selected simply pass
# through unsplit. Y's angled, non-axis-aligned planes are a separate,
# bigger step from this.
#
# Each piece is tracked by a `constraints` dict {seam_index: +1/-1} - only
# for seams that actually bound it. A seam absent from a piece's dict means
# that piece spans the *full* extent there (unsplit by that seam), which is
# exactly the case for T's trunk on the stem seam.
# --------------------------------------------------------------------------

def split_multi(body, seams):
    """body, [{'axis':.., 'offset':.., 'applies_to':..}, ...] -> (pieces, constraints)
    constraints[i] is {seam_index: +1/-1} for piece[i] - only seams that
    actually bound that piece. `applies_to`, if not None, is (dep_index,
    dep_side): this seam only splits pieces already on `dep_side` of seam
    `dep_index`; other pieces pass through unsplit (and get no entry for
    this seam's index)."""
    pieces = [body]
    constraints = [{}]
    for i, seam in enumerate(seams):
        applies_to = seam.get("applies_to")
        normal = point_from_axes({seam["axis"]: 1.0})
        new_pieces, new_constraints = [], []
        for p, c in zip(pieces, constraints):
            if applies_to is not None and c.get(applies_to[0]) != applies_to[1]:
                new_pieces.append(p)
                new_constraints.append(c)
                continue
            pos, neg = p.split_by_plane(normal, seam["offset"])
            new_pieces.append(pos)
            new_constraints.append({**c, i: 1})
            new_pieces.append(neg)
            new_constraints.append({**c, i: -1})
        pieces, constraints = new_pieces, new_constraints
    return pieces, constraints


def piece_bounds(blank_bounds, seams, piece_constraints):
    """The analytic (not geometry-derived) bounding box of a piece within
    the padded blank, given which side of each constraining seam it's on -
    used for pin placement so corners land at the piece's true slot in the
    mold rather than wherever carved solid geometry happens to be (which
    can be void in hollow mode). Seams absent from `piece_constraints`
    leave that axis at its full blank extent."""
    b = dict(blank_bounds)
    for i, seam in enumerate(seams):
        if i not in piece_constraints:
            continue
        side = piece_constraints[i]
        lo, hi = b[seam["axis"]]
        b[seam["axis"]] = (seam["offset"], hi) if side > 0 else (lo, seam["offset"])
    return b


def piece_name(seams, piece_constraints):
    return "_".join(
        f"{seams[i]['axis']}{'p' if side > 0 else 'n'}"
        for i, side in sorted(piece_constraints.items())
    )


def add_pins_multi(pieces, constraints_list, seams, blank_bounds, wall,
                    n_pins, pin_dia, fit_clearance, pin_margin):
    """Add registration pins between every pair of pieces that are adjacent
    across a seam: both constrained by that seam on opposite sides, and
    compatible (equal, or one/both unconstrained) on every other seam."""
    pieces = list(pieces)
    for i, seam in enumerate(seams):
        seam_axis = seam["axis"]
        for a in range(len(pieces)):
            ca = constraints_list[a]
            if i not in ca:
                continue
            for b in range(a + 1, len(pieces)):
                cb = constraints_list[b]
                if cb.get(i) != -ca[i]:
                    continue
                if any(
                    j in ca and j in cb and ca[j] != cb[j]
                    for j in range(len(seams)) if j != i
                ):
                    continue
                # a, b are adjacent across seam i - whichever is on the
                # +side is "part_pos" (gets the socket), -side is
                # "part_neg" (gets the male pin), matching the single-seam
                # convention. Pin placement uses the *overlap* of both
                # pieces' slots (not just one), since a partial seam can
                # leave them with different extents on other axes (T's
                # trunk spans the stem axis fully; the crown pieces don't).
                pos_idx, neg_idx = (a, b) if ca[i] > 0 else (b, a)
                bounds_a = piece_bounds(blank_bounds, seams, ca)
                bounds_b = piece_bounds(blank_bounds, seams, cb)
                overlap = {
                    ax: (max(bounds_a[ax][0], bounds_b[ax][0]),
                         min(bounds_a[ax][1], bounds_b[ax][1]))
                    for ax in _AXES
                }
                pos, neg = add_registration_pins(
                    pieces[pos_idx], pieces[neg_idx], overlap, seam_axis,
                    seam["offset"], wall, n_pins, pin_dia, fit_clearance, pin_margin,
                )
                pieces[pos_idx], pieces[neg_idx] = pos, neg
    return pieces


# --------------------------------------------------------------------------
# Feature builders
# --------------------------------------------------------------------------

def add_registration_pins(part_pos, part_neg, bounds, seam_axis, seam_offset,
                           wall, n_pins, pin_dia, fit_clearance, margin):
    """Add male pins to `part_neg` (the -seam_axis side), matching sockets to
    `part_pos` (the +seam_axis side), across the seam plane.

    Pins sit at corners of the model's footprint within the seam plane
    (the two axes other than `seam_axis`), inset by `margin` so they land in
    solid wall material rather than the cavity. For odd/thin shapes, increase
    --pin-margin until they clear the cavity.
    """
    u_axis, v_axis = other_axes(seam_axis)
    u_lo, u_hi = bounds[u_axis]
    v_lo, v_hi = bounds[v_axis]
    corners = [
        (u_lo + margin, v_lo + margin),
        (u_hi - margin, v_hi - margin),
        (u_lo + margin, v_hi - margin),
        (u_hi - margin, v_lo + margin),
    ]
    n_pins = max(2, min(4, n_pins))
    positions = corners[:n_pins]

    pin_r = pin_dia / 2.0
    embed = wall * 0.75
    protrude = wall * 0.75
    socket_extra_depth = 0.3  # so the pin doesn't bottom out and hold the seam open
    seam_overlap = 0.2  # push the socket's cut plane past the seam split rather
    # than starting exactly on it - exact coincidence between a cut boundary
    # and a split plane is a classic degenerate case for boolean robustness
    # (confirmed: without this, the socket can come out as a fully sealed
    # pocket instead of an opening, under just the right combination of
    # other geometry in the chain). part_pos has nothing below seam_offset
    # to begin with, so this overlap is a no-op there beyond fixing that.

    for (uu, vv) in positions:
        male = cylinder_along_axis(embed + protrude, pin_r, pin_r, 24, seam_axis)
        male = male.translate(point_from_axes(
            {seam_axis: seam_offset - embed, u_axis: uu, v_axis: vv}))
        part_neg = part_neg + male

        socket_r = pin_r + fit_clearance
        female = cylinder_along_axis(protrude + socket_extra_depth + seam_overlap,
                                      socket_r, socket_r, 24, seam_axis)
        female = female.translate(point_from_axes(
            {seam_axis: seam_offset - seam_overlap, u_axis: uu, v_axis: vv}))
        part_pos = part_pos - female

    return part_pos, part_neg


def build_funnel_solid(apex_xy, apex_z, wall, funnel_top_dia, funnel_bot_dia,
                        overlap=2.0, pad=0.0):
    """The funnel's own solid volume: a tapered mouth plus a straight,
    constant-radius channel continuing `overlap` mm past the model's
    highest point (see the docstring on why - a real, substantial opening
    rather than a knife-edge connection).

    `pad`, if given, grows every radius by that amount - used to build a
    same-family "outer" version of this shape for `build_funnel_jacket`.
    Only the radius changes; the Z extent is identical regardless of `pad`,
    so an outer (padded) and inner (pad=0) shape always span exactly the
    same height and subtracting the inner from the outer can't leave a
    leftover cap poking past either end (confirmed this the hard way: an
    earlier version also stretched the height by `pad`, which pushed the
    outer shape's top past the mold's own top surface with nothing left to
    clip it back - a visible solid bump sitting proud of the surface).

    Returned as a solid shape rather than subtracted in place, so it can be
    used both to actually open the channel and (via `build_funnel_jacket`)
    to give it a proper wall-thickness tube first.
    """
    r_bot = funnel_bot_dia / 2.0 + pad
    r_top = funnel_top_dia / 2.0 + pad
    taper = mf.Manifold.cylinder(
        wall + 0.02, r_bot, r_top, 32, center=False
    ).translate((apex_xy[0], apex_xy[1], apex_z - 0.01))
    straight = mf.Manifold.cylinder(
        overlap + 0.02, r_bot, r_bot, 32, center=False
    ).translate((apex_xy[0], apex_xy[1], apex_z - overlap - 0.01))
    return taper + straight


def build_funnel_jacket(apex_xy, apex_z, wall, funnel_top_dia, funnel_bot_dia,
                         overlap, model):
    """A `wall`-thick tube wrapped directly around the funnel bore, so
    hollowing doesn't leave the pour channel as just an unsupported hole
    with no walls once it's past cavity_shell's reach around the model.

    Built as two same-family funnel shapes (see `build_funnel_solid`'s
    `pad`) and subtracted, rather than via a Minkowski offset of the bore
    unioned with the model - growing a Minkowski offset around a large
    model unioned with a thin, far-away tube reliably produced bad topology
    (multiple disconnected shells); two direct, well-formed solids don't
    have that problem.
    """
    inner = build_funnel_solid(apex_xy, apex_z, wall, funnel_top_dia, funnel_bot_dia,
                                overlap, pad=0.0)
    outer = build_funnel_solid(apex_xy, apex_z, wall, funnel_top_dia, funnel_bot_dia,
                                overlap, pad=wall)
    return (outer - inner) - model


def build_vent_solid(xy, apex_z, wall, vent_dia, pad=0.0):
    # Same fix as build_funnel_solid: pad widens the radius only, never the
    # Z extent, so outer/inner jacket shapes always share the same height.
    r = vent_dia / 2.0 + pad
    h = wall + 0.02
    return mf.Manifold.cylinder(h, r, r, 20, center=False).translate(
        (xy[0], xy[1], apex_z - 0.01)
    )


def build_vent_jacket(xy, apex_z, wall, vent_dia, model):
    inner = build_vent_solid(xy, apex_z, wall, vent_dia, pad=0.0)
    outer = build_vent_solid(xy, apex_z, wall, vent_dia, pad=wall)
    return (outer - inner) - model


# --------------------------------------------------------------------------
# Hollowing: replace the solid bulk of the mold body with a thin shell
# --------------------------------------------------------------------------

def face_slab(bounds, axis, side, thickness):
    """A thin box, `thickness` deep along `axis` at its 'lo' or 'hi' boundary,
    spanning the FULL extent of the other two axes. One face of a box-shell,
    built individually so a particular face can be left out of the union."""
    u_axis, v_axis = other_axes(axis)
    lo, hi = bounds[axis]
    if side == "lo":
        a0, a1 = lo, lo + thickness
    else:
        a0, a1 = hi - thickness, hi
    dims = {
        axis: a1 - a0,
        u_axis: bounds[u_axis][1] - bounds[u_axis][0],
        v_axis: bounds[v_axis][1] - bounds[v_axis][0],
    }
    origin = {axis: a0, u_axis: bounds[u_axis][0], v_axis: bounds[v_axis][0]}
    return mf.Manifold.cube(point_from_axes(dims), center=False).translate(point_from_axes(origin))


def seam_active_bounds(bounds, seam, seams, wall):
    """The region a seam actually applies to, as a bounds dict - the full
    blank bounds for a full seam (applies_to is None), or restricted along
    the dependency's axis to its side for a partial seam, with a `wall`-deep
    overlap past the dependency's own plane so the two regions meet with a
    real volumetric overlap rather than an exact-coincidence edge (exact
    coincidence at a boolean boundary reliably produces a spuriously sealed/
    disconnected result - confirmed the hard way earlier on the pin
    sockets). Shared between seam_slab and open_back so both restrict a
    partial seam's treatment the same way.
    """
    applies_to = seam.get("applies_to")
    if applies_to is None:
        return dict(bounds)
    dep_idx, dep_side = applies_to
    dep = seams[dep_idx]
    dep_axis = dep["axis"]
    dep_lo, dep_hi = bounds[dep_axis]
    if dep_side > 0:
        restrict_lo, restrict_hi = dep["offset"] - wall, dep_hi
    else:
        restrict_lo, restrict_hi = dep_lo, dep["offset"] + wall
    restricted = dict(bounds)
    restricted[dep_axis] = (restrict_lo, restrict_hi)
    return restricted


def region_box(region_bounds):
    """A solid Manifold cube spanning a bounds dict like {'x': (lo,hi), ...}."""
    dims = {ax: region_bounds[ax][1] - region_bounds[ax][0] for ax in _AXES}
    origin = {ax: region_bounds[ax][0] for ax in _AXES}
    return mf.Manifold.cube(point_from_axes(dims), center=False).translate(point_from_axes(origin))


def cross_brace(region_bounds, seam_axis, cross_width):
    """The '+'-shaped brace for one seam's open-back treatment: two ribs,
    each coplanar with a pair of the region's side walls, running the full
    seam_axis depth of `region_bounds` and crossing through the center of
    the other two axes."""
    u_axis, v_axis = other_axes(seam_axis)
    lo, hi = region_bounds[seam_axis]
    u_lo, u_hi = region_bounds[u_axis]
    v_lo, v_hi = region_bounds[v_axis]
    if cross_width >= (u_hi - u_lo) or cross_width >= (v_hi - v_lo):
        raise ValueError("--cross-width is too large for this mold's footprint.")
    u_mid, v_mid = (u_lo + u_hi) / 2.0, (v_lo + v_hi) / 2.0
    rib_u = mf.Manifold.cube(point_from_axes(
        {seam_axis: hi - lo, u_axis: cross_width, v_axis: v_hi - v_lo}), center=False
    ).translate(point_from_axes({seam_axis: lo, u_axis: u_mid - cross_width / 2.0, v_axis: v_lo}))
    rib_v = mf.Manifold.cube(point_from_axes(
        {seam_axis: hi - lo, u_axis: u_hi - u_lo, v_axis: cross_width}), center=False
    ).translate(point_from_axes({seam_axis: lo, u_axis: u_lo, v_axis: v_mid - cross_width / 2.0}))
    return rib_u + rib_v


def hollow_body(model, blank, wall, skin, seams,
                 open_back=False, cross_width=4.0):
    """Turn a solid `blank - model` body into a shell to save filament.

    Keeps three regions, unions them, then re-clips to the cavity:
      - cavity_shell: material within `wall` of the model surface (this is
        what actually defines the cast - keep it exactly as designed).
      - skin_shell:   material within `skin` of the mold's outer faces
                       (rigidity + a printable, closed outer surface).
      - seam_slab:    a full-footprint solid band around each seam plane in
                       `seams` (whichever axis each is perpendicular to), so
                       every parting face stays flat/sealed and there's
                       solid material for the registration pins to embed
                       into.
    Everything else in the original solid bulk becomes empty void.

    `seams` is a list of {'axis':.., 'offset':.., 'applies_to':..} - one
    full-plane seam slab is added per entry (restricted to that seam's own
    active region if it's partial), so this works unchanged for the
    single-seam case, full multi-part splits, and T-style partial seams.

    With `open_back`, this becomes closer to a vacuum-formed shell. Each
    seam contributes its own open-back treatment, independently, restricted
    to that seam's own active region (see `seam_active_bounds`): the two
    faces perpendicular to its axis (each piece's own "back" on that seam,
    opposite the cavity - the ones with the least functional need to be
    solid, since they don't seal against anything) are left out of
    skin_shell there, replaced by a "+"-shaped brace standing perpendicular
    to them, running the seam's own active depth. Processing seams
    independently this way means a piece's open faces automatically end up
    matching exactly the seams that actually bound it: a full multi-seam
    split opens every seam's pair of faces on every piece, while a T's
    trunk (unconstrained by the partial stem seam) keeps its stem-axis
    faces skinned normally, since that seam's active region never reaches
    it.

    Note: without open_back, for large molds the unsupported span of skin
    over the void can be significant - if it's wide, add internal ribs/
    gyroid infill as a next step, or fall back to solid (--hollow off).

    The funnel/vent bores get their own protective "jacket" added
    separately by the caller (see `build_funnel_jacket`) rather than being
    folded in here - growing a Minkowski offset around the union of a large
    model and a thin, far-away tube reliably produced bad topology (multiple
    disconnected shells), so the jacket is instead built directly as its own
    simple, well-formed shape and just unioned in afterward.
    """
    # Cavity-side shell via a true outward offset (Minkowski sum with a
    # sphere). Simplify first: the offset surface is hidden inside the wall,
    # so full input detail isn't needed and this keeps Minkowski sum fast
    # (its cost scales with the product of face counts on non-convex input).
    simplified = model.simplify(max(wall * 0.1, 0.05))
    ball = mf.Manifold.sphere(1.0, 16).scale((wall, wall, wall))
    grown = simplified.minkowski_sum(ball)
    cavity_shell = grown - model

    bounds = bbox_dict(blank.bounding_box())

    if open_back:
        # Process each axis independently rather than building a full
        # 6-face skin and subtracting seams' open regions from the whole
        # thing - subtracting a same-axis open region from a *different*
        # axis's face would also eat the shared corner/edge strip where the
        # two face-slabs overlap (confirmed this the hard way: it silently
        # shaved material off the *other* faces too). Doing "this face's
        # full extent minus only its own same-axis open portion" per axis
        # avoids that entirely.
        seams_by_axis = {}
        for seam in seams:
            seams_by_axis.setdefault(seam["axis"], []).append(seam)

        skin_shell = None
        for axis in _AXES:
            axis_seams = seams_by_axis.get(axis, [])
            for side in ("lo", "hi"):
                full_face = face_slab(bounds, axis, side, skin)
                if not axis_seams:
                    skin_shell = full_face if skin_shell is None else (skin_shell + full_face)
                    continue
                open_region = None
                for seam in axis_seams:
                    active = seam_active_bounds(bounds, seam, seams, wall)
                    piece = face_slab(active, axis, side, skin)
                    open_region = piece if open_region is None else (open_region + piece)
                remaining = full_face - open_region
                skin_shell = remaining if skin_shell is None else (skin_shell + remaining)
            for seam in axis_seams:
                active = seam_active_bounds(bounds, seam, seams, wall)
                skin_shell = skin_shell + cross_brace(active, axis, cross_width)
    else:
        # Outer skin shell: blank minus a version of itself shrunk inward by `skin`
        inner_dims = {a: (bounds[a][1] - bounds[a][0] - 2 * skin) for a in _AXES}
        if min(inner_dims.values()) <= 0:
            raise ValueError("--skin is too large for this mold's size.")
        inner_origin = {a: bounds[a][0] + skin for a in _AXES}
        inner = mf.Manifold.cube(point_from_axes(inner_dims), center=False).translate(
            point_from_axes(inner_origin)
        )
        skin_shell = blank - inner

    # Seam slab(s): keep a solid band (+-wall) around every parting plane,
    # full extent in the other two axes - unaffected by open_back, these are
    # different faces (the parting lines) and always need to stay solid.
    # A partial seam's slab only needs to exist in its own active region
    # (T's stem slab shouldn't run the full bar-axis range, only the crown
    # side) - see `seam_active_bounds`.
    seam_slab = None
    for seam in seams:
        seam_axis = seam["axis"]
        u_axis, v_axis = other_axes(seam_axis)
        active = seam_active_bounds(bounds, seam, seams, wall)
        slab_bounds = dict(active)
        slab_bounds[seam_axis] = (seam["offset"] - wall, seam["offset"] + wall)
        one_slab = region_box(slab_bounds) ^ blank
        seam_slab = one_slab if seam_slab is None else (seam_slab + one_slab)

    kept = cavity_shell + skin_shell + seam_slab
    kept = (kept ^ blank) - model
    if kept.is_empty():
        raise RuntimeError("Hollowing produced an empty body - reduce --skin or check --wall.")
    return kept


# --------------------------------------------------------------------------
# Sleeve mode: cast a hollow shell instead of a solid part. A separate
# "core" insert occupies the model's interior (offset in from the surface by
# --sleeve-wall) during the pour, leaving just a shell of that thickness.
# The core is inserted from below through an opening cut in the mold's floor
# at the model's base footprint, and pulled back out once the shell has set.
# --------------------------------------------------------------------------

def base_footprint(model, zmin, slab_eps=0.5):
    """2D outline (manifold3d CrossSection) of the model at its base (z=zmin),
    obtained by intersecting with a thin slab there and projecting to XY."""
    bxmin, bymin, bzmin, bxmax, bymax, bzmax = model.bounding_box()
    slab = mf.Manifold.cube(
        (bxmax - bxmin + 2, bymax - bymin + 2, slab_eps), center=False
    ).translate((bxmin - 1, bymin - 1, zmin))
    base_solid = model ^ slab
    if base_solid.is_empty():
        raise ValueError(
            "No solid found at the model's base (z=zmin) - --sleeve assumes "
            "the model has a real flat base area there, not just a point."
        )
    raw = base_solid.project()
    return mf.CrossSection(raw.to_polygons(), mf.FillRule.Positive)


def cut_base_opening(body, base_poly, zmin, clearance):
    """Punch a hole through the mold floor at the base footprint (+clearance/2)
    so the core can pass through and the finished cast has an open base."""
    hole_poly = base_poly.offset(clearance / 2.0)
    hole_h = zmin + 1.2  # from well below the blank up to a hair past zmin
    hole = mf.Manifold.extrude(hole_poly, hole_h).translate((0, 0, -1.0))
    return body - hole


def build_core(model, base_poly, wall, sleeve_wall, zmin, clearance,
                flange_margin, flange_thickness):
    """The insert: model eroded inward by sleeve_wall, plus a stem through
    the base opening ending in a flange that stops flush against the mold's
    exterior bottom (z=0) and doubles as a grip for pulling the core out."""
    simplified = model.simplify(max(sleeve_wall * 0.1, 0.05))
    ball = mf.Manifold.sphere(1.0, 16).scale((sleeve_wall, sleeve_wall, sleeve_wall))
    eroded = simplified.minkowski_difference(ball)
    if eroded.is_empty():
        raise ValueError(
            "--sleeve-wall is too thick for this model - eroding it inward "
            "by that amount leaves nothing. Try a smaller --sleeve-wall."
        )

    stem_poly = base_poly.offset(-clearance / 2.0)
    flange_poly = base_poly.offset(flange_margin)

    stem_top = zmin + sleeve_wall + 1.0  # overlaps into `eroded` for a clean union
    stem_h = stem_top + flange_thickness + 0.02
    stem = mf.Manifold.extrude(stem_poly, stem_h).translate((0, 0, -flange_thickness - 0.01))
    flange = mf.Manifold.extrude(flange_poly, flange_thickness + 0.02).translate(
        (0, 0, -flange_thickness - 0.01)
    )
    core = eroded + stem + flange
    if core.is_empty():
        raise RuntimeError("Core construction produced an empty result.")
    return core


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def generate_mold(input_path, out_prefix, wall, up_axis, seam_axes, seam_fractions,
                   n_pins, pin_dia, fit_clearance, pin_margin,
                   funnel_top_dia, funnel_bot_dia, funnel_overlap,
                   pour_offset_xy, vent_dia, vent_offset_xy,
                   hollow=False, skin=2.0, open_back=False, cross_width=4.0,
                   sleeve=False, sleeve_wall=3.0, sleeve_clearance=0.3,
                   sleeve_flange_margin=3.0, sleeve_flange_thickness=2.5,
                   seam_parents=None):

    if len(seam_axes) != len(seam_fractions):
        raise ValueError("--seam-axis and --seam must have the same number of values.")
    if seam_parents is None:
        seam_parents = [None] * len(seam_axes)
    elif len(seam_parents) != len(seam_axes):
        raise ValueError("--seam-parent must have the same number of values as --seam-axis.")
    for i, parent in enumerate(seam_parents):
        if parent is not None and parent[0] >= i:
            raise ValueError(
                f"--seam-parent entry {i} references seam {parent[0]}, which must "
                "come before it in the list (a seam can only depend on an earlier one)."
            )
    if len(seam_axes) > 1 and sleeve:
        raise ValueError(
            "--sleeve isn't supported with multiple seams yet - it assumes a "
            "single parting line for the base opening/core. Use one --seam-axis "
            "for now."
        )

    src = trimesh.load(input_path, force="mesh")
    if not isinstance(src, trimesh.Trimesh):
        raise ValueError("Input did not load as a single mesh.")
    if not src.is_watertight:
        print(
            "WARNING: input is not watertight. Proceeding anyway per "
            "current assumption, but the boolean result may be invalid.",
            file=sys.stderr,
        )

    # Rotate into canonical Z-up working frame
    rot = _AXIS_TO_Z[up_axis]
    work = rotate_mesh(src, rot)

    model = trimesh_to_manifold(work)
    # Lift the model up by `wall` so the blank can start flush at z=0 (bed-
    # ready) while still leaving a full wall-thickness floor beneath it.
    # Without this, a model with a genuinely flat base (a real contact area,
    # not just a tangent point) ends up with NO floor under it at all - the
    # mold's bottom face and the model's base face are coincident, so the
    # boolean subtraction leaves a hole straight through and liquid poured
    # in would leak out the bottom.
    model = model.translate((0, 0, wall))
    xmin, ymin, zmin, xmax, ymax, zmax = model.bounding_box()
    model_bounds = bbox_dict(model.bounding_box())

    # 1. Blank block around the model, padded by wall thickness on all sides,
    #    sitting on the print bed (z=0) up to the top of the model + wall.
    blank = mf.Manifold.cube(
        (xmax - xmin + 2 * wall, ymax - ymin + 2 * wall, zmax + wall),
        center=False,
    ).translate((xmin - wall, ymin - wall, 0))

    # Seam planes: perpendicular to each axis in `seam_axes` (default z =
    # horizontal, the classic clamshell; x or y gives a vertical seam for
    # models with side undercuts), at the matching fraction of the model's
    # extent along that axis. One seam -> 2 pieces (top/bottom); two full
    # seams -> 4 pieces (a +/X split); three -> 8. A seam can instead be
    # partial via `seam_parents` - restricted to only the pieces already on
    # one side of an earlier seam (T's stem, restricted to the crown side
    # of its full "-" bar).
    seams = []
    for axis, frac, parent in zip(seam_axes, seam_fractions, seam_parents):
        lo, hi = model_bounds[axis]
        seams.append({"axis": axis, "offset": lo + frac * (hi - lo), "applies_to": parent})

    # Build the funnel (and vent) solids now, before hollowing - so that if
    # hollowing is on, it can add a proper wall-thickness jacket around the
    # pour channel, rather than the channel just dissolving into the
    # general void wherever it isn't already inside cavity_shell's reach.
    default_xy = ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)
    apex_xy = (default_xy[0] + pour_offset_xy[0], default_xy[1] + pour_offset_xy[1])
    funnel_solid = build_funnel_solid(apex_xy, zmax, wall, funnel_top_dia, funnel_bot_dia,
                                       overlap=funnel_overlap)
    vent_solid = None
    vent_xy = None
    if vent_dia > 0:
        vent_xy = (default_xy[0] + vent_offset_xy[0], default_xy[1] + vent_offset_xy[1])
        vent_solid = build_vent_solid(vent_xy, zmax, wall, vent_dia)

    # 2. Carve the cavity
    solid_body = blank - model
    if solid_body.is_empty():
        raise RuntimeError("Cavity subtraction produced an empty result.")

    # 2b. Optionally hollow the bulk out into a shell to save filament
    if hollow:
        body = hollow_body(model, blank, wall, skin, seams,
                            open_back=open_back, cross_width=cross_width)
        # Give the pour channel(s) a proper wall-thickness jacket, since
        # otherwise whatever part of them is past cavity_shell's reach
        # around the model just dissolves into the general void. Clipped to
        # blank as a safety net - e.g. a very wide --funnel-top-dia could
        # otherwise poke sideways past the mold's own footprint too.
        body = body + (build_funnel_jacket(apex_xy, zmax, wall, funnel_top_dia,
                                            funnel_bot_dia, funnel_overlap, model) ^ blank)
        if vent_solid is not None:
            body = body + (build_vent_jacket(vent_xy, zmax, wall, vent_dia, model) ^ blank)
        print(f"hollowed: {solid_body.volume():.0f} mm3 -> {body.volume():.0f} mm3 "
              f"({100 * (1 - body.volume() / solid_body.volume()):.0f}% material saved)")
    else:
        body = solid_body

    # 2c. Optionally cut a base opening and build the core insert, so the
    #     mold casts a hollow shell rather than a solid part. (Always tied
    #     to the up-axis/floor, independent of seam axis; guarded above to
    #     single-seam only.)
    core = None
    if sleeve:
        base_poly = base_footprint(model, zmin)
        body = cut_base_opening(body, base_poly, zmin, sleeve_clearance)
        core = build_core(
            model, base_poly, wall, sleeve_wall, zmin, sleeve_clearance,
            sleeve_flange_margin, sleeve_flange_thickness,
        )

    # 3. Actually open the pour funnel and vent up now, subtracting the same
    #    solids used above - bored before splitting, so with a vertical seam
    #    this naturally divides the channel between the pieces on either
    #    side of it (the standard way real 2-part molds route a sprue
    #    across a vertical parting line), and for the default horizontal
    #    seam it lands entirely in the top piece exactly as before.
    body = body - funnel_solid
    if vent_solid is not None:
        body = body - vent_solid

    # 4. Split on every seam plane -> 2^len(seams) pieces (fewer if any
    #    seam is partial and only splits some of them)
    pieces, constraints = split_multi(body, seams)

    # 5. Registration pins between every pair of pieces adjacent across a seam
    blank_bounds = bbox_dict(blank.bounding_box())
    pieces = add_pins_multi(pieces, constraints, seams, blank_bounds, wall,
                             n_pins, pin_dia, fit_clearance, pin_margin)

    if any(p.is_empty() for p in pieces):
        raise RuntimeError("A mold piece came out empty - check parameters.")

    # 6. Back to trimesh, rotate back to original orientation, export.
    # Single seam keeps the familiar _top/_bottom names; multi-seam uses a
    # name built from each piece's side of every seam that bounds it
    # (e.g. "zp_xp").
    inv_rot = rot.T  # rotation matrices are orthonormal
    piece_paths, piece_tms = [], []
    for p, c in zip(pieces, constraints):
        tm = rotate_mesh(manifold_to_trimesh(p), inv_rot)
        if len(seams) == 1:
            label = "top" if c[0] > 0 else "bottom"
        else:
            label = piece_name(seams, c)
        path = f"{out_prefix}_{label}.stl"
        tm.export(path)
        print(f"{label + ':':13s}{path}  watertight={verify_watertight(path)}  "
              f"volume={p.volume():.1f} mm3")
        piece_paths.append(path)
        piece_tms.append(tm)

    core_tm = None
    core_path = None
    if core is not None:
        if core.is_empty():
            raise RuntimeError("Core insert came out empty - check parameters.")
        core_tm = rotate_mesh(manifold_to_trimesh(core), inv_rot)
        core_path = f"{out_prefix}_core.stl"
        core_tm.export(core_path)
        print(f"core insert: {core_path}  watertight={verify_watertight(core_path)}  "
              f"volume={core.volume():.1f} mm3")
        print("Insert the core from below (flange side) until the flange "
              "seats against the mold's bottom; pull it back out once the "
              "shell has set.")

    return piece_paths, piece_tms, constraints, seams, core_path, core_tm


# --------------------------------------------------------------------------
# Preview rendering (headless, no GPU needed - matplotlib with manual
# per-face Lambertian shading, since there's no pyrender/OpenGL to rely on
# in a plain server/CLI environment).
# --------------------------------------------------------------------------

def final_frame_component(up_axis, axis):
    """(index, sign) - which output axis (0=x,1=y,2=z), and with which
    sign, `axis` (named in the up=Z working frame) maps to after rotating
    back to the model's original orientation. Used for preview rendering."""
    rot = _AXIS_TO_Z[up_axis]
    v = np.zeros(3)
    v[_axis_index(axis)] = 1.0
    v2 = rot.T @ v
    idx = int(np.argmax(np.abs(v2)))
    sign = 1.0 if v2[idx] >= 0 else -1.0
    return idx, sign


_PIECE_PALETTE = [
    (0.35, 0.65, 0.85), (0.95, 0.65, 0.25), (0.80, 0.35, 0.55), (0.45, 0.75, 0.45),
    (0.75, 0.55, 0.85), (0.90, 0.75, 0.30), (0.40, 0.70, 0.75), (0.85, 0.45, 0.35),
]


def render_preview(piece_tms, constraints, seams, out_path, up_axis="z",
                    explode=None, core_tm=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    def shaded_facecolors(mesh, base_rgb, light_dir=(0.5, 0.4, 0.75)):
        light_dir = np.array(light_dir, dtype=float)
        light_dir /= np.linalg.norm(light_dir)
        diffuse = np.clip(mesh.face_normals @ light_dir, 0, 1)
        brightness = 0.35 + 0.65 * diffuse  # ambient + diffuse term
        base = np.array(base_rgb)
        colors = np.clip(base[None, :] * brightness[:, None], 0, 1)
        return np.hstack([colors, np.ones((len(colors), 1))])

    def draw(ax, parts, azim, elev=18):
        all_pts = []
        for mesh, base_rgb in parts:
            colors = shaded_facecolors(mesh, base_rgb)
            pc = Poly3DCollection(mesh.vertices[mesh.faces], facecolor=colors,
                                   edgecolor=(0, 0, 0, 0.12), linewidths=0.1, zsort="min")
            ax.add_collection3d(pc)
            all_pts.append(mesh.vertices)
        pts = np.vstack(all_pts)
        mins, maxs = pts.min(0), pts.max(0)
        center = (mins + maxs) / 2
        span = (maxs - mins).max() / 2 * 1.08
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(center[2] - span, center[2] + span)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    if core_tm is not None:
        # Sleeve mode (single-seam only): an exploded view would hide the
        # fit, so instead cut each assembled piece in half and view into
        # the cut to show the core sitting inside the cavity with the
        # shell gap around it. Cut via manifold3d's own trim_by_plane
        # rather than trimesh's slice_mesh_plane(cap=True), which pulls in
        # scipy/shapely/rtree/networkx just to cap a triangle hole -
        # manifold3d is already a hard dependency and produces a clean
        # capped cut natively.
        top_tm, bottom_tm = (
            (piece_tms[0], piece_tms[1]) if constraints[0][0] > 0 else (piece_tms[1], piece_tms[0])
        )

        def half(mesh):
            man = trimesh_to_manifold(mesh)
            trimmed = man.trim_by_plane((0, -1, 0), 0.0)
            return manifold_to_trimesh(trimmed)
        parts = [
            (half(bottom_tm), (0.35, 0.65, 0.85)),
            (half(top_tm), (0.95, 0.65, 0.25)),
            (half(core_tm), (0.8, 0.2, 0.2)),
        ]
        draw(ax, parts, azim=90)
    else:
        if explode is None:
            explode = max(tm.extents.max() for tm in piece_tms) * 0.4
        # Each seam's axis (named in the up=Z working frame) maps to some
        # signed output axis once rotated back; a piece's explode direction
        # is the sum of its side (+/-1) on every seam that bounds it, in
        # that seam's mapped direction - so a 4-piece +/X split fans
        # diagonally outward, a piece unconstrained by some seam (T's
        # trunk on the stem seam) simply doesn't shift along that axis, and
        # the single-seam case reduces to exactly the old behavior.
        components = [final_frame_component(up_axis, seam["axis"]) for seam in seams]
        parts = []
        for i, (tm, c) in enumerate(zip(piece_tms, constraints)):
            shift = np.zeros(3)
            for seam_idx, (idx, sign) in enumerate(components):
                shift[idx] += explode * sign * c.get(seam_idx, 0)
            m = tm.copy()
            m.apply_translation(shift)
            parts.append((m, _PIECE_PALETTE[i % len(_PIECE_PALETTE)]))
        draw(ax, parts, azim=-50, elev=22)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"preview:     {out_path}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="Path to a watertight STL/OBJ/3MF model")
    p.add_argument("--out", default="mold", help="Output filename prefix")
    p.add_argument("--wall", type=float, default=3.0, help="Mold wall thickness (mm)")
    p.add_argument("--up", choices=["x", "y", "z"], default="z",
                    help="Which model axis points 'up' toward the pour side")
    p.add_argument("--seam-axis", choices=["x", "y", "z"], default=["z"], nargs="+",
                    help="Which axis each parting plane is perpendicular to. One "
                         "value (default z) = the classic 2-piece clamshell; x or y "
                         "= a vertical seam instead, for undercuts on the sides "
                         "rather than top/bottom. Two values = a 4-piece +/X split "
                         "(both seams full planes); three = 8 pieces. Pair "
                         "positionally with --seam.")
    p.add_argument("--seam", type=float, default=[0.5], nargs="+",
                    help="Seam position(s) as a fraction (0-1) of the model's "
                         "extent along the matching --seam-axis entry")
    p.add_argument("--seam-parent", default=None, nargs="+",
                    help="Makes a seam partial: '-' (default) = a full plane "
                         "applying to every piece so far. 'N+' or 'N-' = only "
                         "split pieces already on the +/- side of seam N (N is "
                         "0-based, must be an earlier seam). This is how a T "
                         "split works: --seam-axis z x --seam 0.5 0.5 "
                         "--seam-parent - 0+ gives a full horizontal bar (seam "
                         "0) and a vertical stem (seam 1) that only splits the "
                         "crown above the bar, leaving the trunk below it whole.")
    p.add_argument("--pins", type=int, default=2, choices=[2, 3, 4],
                    help="Number of registration pins")
    p.add_argument("--pin-dia", type=float, default=5.0, help="Pin diameter (mm)")
    p.add_argument("--fit-clearance", type=float, default=0.15,
                    help="Radial clearance added to sockets/keys (mm)")
    p.add_argument("--pin-margin", type=float, default=None,
                    help="Inset of pins from the mold's outer footprint corners "
                         "(mm). Defaults to 1.5x wall thickness.")
    p.add_argument("--funnel-top-dia", type=float, default=10.0)
    p.add_argument("--funnel-bottom-dia", type=float, default=4.0)
    p.add_argument("--funnel-overlap", type=float, default=2.0,
                    help="How far the funnel bore extends past the model's "
                         "highest point, as a straight constant-radius channel "
                         "(mm). Raises this to widen the connection into the "
                         "cavity for thicker/more viscous pours. Reduce it for "
                         "a very slender/pointed model tip, where too much "
                         "overlap could bore sideways out through the wall.")
    p.add_argument("--pour-offset", type=float, nargs=2, default=(0.0, 0.0),
                    metavar=("DX", "DY"),
                    help="Offset the pour funnel from the model's footprint centre")
    p.add_argument("--vent-dia", type=float, default=0.0,
                    help="Vent hole diameter (mm), 0 = no vent")
    p.add_argument("--vent-offset", type=float, nargs=2, default=(0.0, 0.0),
                    metavar=("DX", "DY"))
    p.add_argument("--hollow", action="store_true",
                    help="Hollow the mold bulk into a shell (keeps the --wall thickness "
                         "at the cavity, a --skin skin outside, void between) to save "
                         "filament. Most worthwhile on larger molds.")
    p.add_argument("--skin", type=float, default=2.0,
                    help="Outer skin thickness when --hollow is set (mm)")
    p.add_argument("--open-back", action="store_true",
                    help="With --hollow: leave off each half's outer face "
                         "(opposite the cavity) entirely instead of skinning "
                         "it, replacing it with a '+' shaped brace standing "
                         "perpendicular to it for rigidity - a vacuum-formed-"
                         "shell look rather than a fully boxed-in shell.")
    p.add_argument("--cross-width", type=float, default=4.0,
                    help="Width of each brace rib when --open-back is set (mm)")
    p.add_argument("--preview", action="store_true",
                    help="Save a shaded preview PNG (exploded view normally, "
                         "or a cutaway if --sleeve is set)")
    p.add_argument("--sleeve", action="store_true",
                    help="Cast a hollow shell instead of a solid part: cuts a base "
                         "opening and generates a separate core insert STL. Assumes "
                         "the model has a real flat base area at z=zmin.")
    p.add_argument("--sleeve-wall", type=float, default=3.0,
                    help="Thickness of the cast shell (mm)")
    p.add_argument("--sleeve-clearance", type=float, default=0.3,
                    help="Total diametral clearance between the core's stem and "
                         "the mold's base opening (mm)")
    p.add_argument("--sleeve-flange-margin", type=float, default=3.0,
                    help="How far the core's flange extends past the base opening (mm)")
    p.add_argument("--sleeve-flange-thickness", type=float, default=2.5)
    args = p.parse_args()

    pin_margin = args.pin_margin if args.pin_margin is not None else args.wall * 1.5

    seam_parent_raw = args.seam_parent if args.seam_parent is not None else ["-"] * len(args.seam_axis)
    if len(seam_parent_raw) != len(args.seam_axis):
        p.error("--seam-parent must have the same number of values as --seam-axis.")
    seam_parents = []
    for i, raw in enumerate(seam_parent_raw):
        if raw == "-":
            seam_parents.append(None)
            continue
        m = re.fullmatch(r"(\d+)([+-])", raw)
        if not m:
            p.error(f"--seam-parent entry {i!r} must be '-' or like '0+'/'1-'.")
        dep_idx, dep_sign = int(m.group(1)), (1 if m.group(2) == "+" else -1)
        seam_parents.append((dep_idx, dep_sign))

    piece_paths, piece_tms, constraints, seams, core_path, core_tm = generate_mold(
        input_path=args.input,
        out_prefix=args.out,
        wall=args.wall,
        up_axis=args.up,
        seam_axes=args.seam_axis,
        seam_fractions=args.seam,
        n_pins=args.pins,
        pin_dia=args.pin_dia,
        fit_clearance=args.fit_clearance,
        pin_margin=pin_margin,
        funnel_top_dia=args.funnel_top_dia,
        funnel_bot_dia=args.funnel_bottom_dia,
        funnel_overlap=args.funnel_overlap,
        pour_offset_xy=tuple(args.pour_offset),
        vent_dia=args.vent_dia,
        vent_offset_xy=tuple(args.vent_offset),
        hollow=args.hollow,
        skin=args.skin,
        open_back=args.open_back,
        cross_width=args.cross_width,
        sleeve=args.sleeve,
        sleeve_wall=args.sleeve_wall,
        sleeve_clearance=args.sleeve_clearance,
        sleeve_flange_margin=args.sleeve_flange_margin,
        sleeve_flange_thickness=args.sleeve_flange_thickness,
        seam_parents=seam_parents,
    )

    if args.preview:
        render_preview(piece_tms, constraints, seams, f"{args.out}_preview.png",
                        up_axis=args.up, core_tm=core_tm)


if __name__ == "__main__":
    main()
