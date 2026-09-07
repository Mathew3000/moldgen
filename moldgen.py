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


def final_frame_axis(up_axis, axis):
    """Which output axis (0=x,1=y,2=z), after rotating back from the internal
    up=Z working frame to the model's original orientation, `axis` (named in
    that working frame) ends up along. Used only for preview rendering."""
    rot = _AXIS_TO_Z[up_axis]
    v = np.zeros(3)
    v[_axis_index(axis)] = 1.0
    v2 = rot.T @ v
    return int(np.argmax(np.abs(v2)))


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

    `pad`, if given, grows every radius (and stretches the top/bottom
    accordingly) by that amount - used to build a same-family "outer"
    version of this shape for `build_funnel_jacket` rather than subtracted
    directly.

    Returned as a solid shape rather than subtracted in place, so it can be
    used both to actually open the channel and (via `build_funnel_jacket`)
    to give it a proper wall-thickness tube first.
    """
    r_bot = funnel_bot_dia / 2.0 + pad
    r_top = funnel_top_dia / 2.0 + pad
    taper = mf.Manifold.cylinder(
        wall + 0.02 + 2 * pad, r_bot, r_top, 32, center=False
    ).translate((apex_xy[0], apex_xy[1], apex_z - 0.01 - pad))
    straight = mf.Manifold.cylinder(
        overlap + 0.02 + pad, r_bot, r_bot, 32, center=False
    ).translate((apex_xy[0], apex_xy[1], apex_z - overlap - 0.01 - pad))
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
    r = vent_dia / 2.0 + pad
    h = wall + 0.02 + 2 * pad
    return mf.Manifold.cylinder(h, r, r, 20, center=False).translate(
        (xy[0], xy[1], apex_z - 0.01 - pad)
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


def hollow_body(model, blank, wall, skin, seam_axis, seam_offset,
                 open_back=False, cross_width=4.0):
    """Turn a solid `blank - model` body into a shell to save filament.

    Keeps three regions, unions them, then re-clips to the cavity:
      - cavity_shell: material within `wall` of the model surface (this is
        what actually defines the cast - keep it exactly as designed).
      - skin_shell:   material within `skin` of the mold's outer faces
                       (rigidity + a printable, closed outer surface).
      - seam_slab:    a full-footprint solid band around the seam plane
                       (whichever axis it's perpendicular to), so the
                       parting face stays flat/sealed and there's solid
                       material for the registration pins to embed into.
    Everything else in the original solid bulk becomes empty void.

    With `open_back`, this becomes closer to a vacuum-formed shell: the two
    outer faces perpendicular to seam_axis (each half's own "back", opposite
    the cavity - the ones with the least functional need to be solid, since
    they don't seal against anything) are left off skin_shell entirely, and
    replaced with a "+"-shaped brace standing perpendicular to them (in the
    same two planes as the four side walls, running the full depth) for
    rigidity in place of a full panel.

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
    u_axis, v_axis = other_axes(seam_axis)

    if open_back:
        # Only the 4 side faces (S) - leave both seam_axis-facing faces (O,
        # each half's own "back") out entirely.
        skin_shell = (
            face_slab(bounds, u_axis, "lo", skin) + face_slab(bounds, u_axis, "hi", skin)
            + face_slab(bounds, v_axis, "lo", skin) + face_slab(bounds, v_axis, "hi", skin)
        )
        lo, hi = bounds[seam_axis]
        u_lo, u_hi = bounds[u_axis]
        v_lo, v_hi = bounds[v_axis]
        if cross_width >= (u_hi - u_lo) or cross_width >= (v_hi - v_lo):
            raise ValueError("--cross-width is too large for this mold's footprint.")
        u_mid, v_mid = (u_lo + u_hi) / 2.0, (v_lo + v_hi) / 2.0
        rib_u = mf.Manifold.cube(point_from_axes(
            {seam_axis: hi - lo, u_axis: cross_width, v_axis: v_hi - v_lo}), center=False
        ).translate(point_from_axes({seam_axis: lo, u_axis: u_mid - cross_width / 2.0, v_axis: v_lo}))
        rib_v = mf.Manifold.cube(point_from_axes(
            {seam_axis: hi - lo, u_axis: u_hi - u_lo, v_axis: cross_width}), center=False
        ).translate(point_from_axes({seam_axis: lo, u_axis: u_lo, v_axis: v_mid - cross_width / 2.0}))
        skin_shell = skin_shell + rib_u + rib_v
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

    # Seam slab: keep a solid band (+-wall) around the parting plane, full
    # extent in the other two axes - unaffected by open_back, this is a
    # different face (the parting line) and always needs to stay solid.
    slab_dims = {
        seam_axis: 2 * wall,
        u_axis: bounds[u_axis][1] - bounds[u_axis][0],
        v_axis: bounds[v_axis][1] - bounds[v_axis][0],
    }
    slab_origin = {
        seam_axis: seam_offset - wall,
        u_axis: bounds[u_axis][0],
        v_axis: bounds[v_axis][0],
    }
    seam_box = mf.Manifold.cube(point_from_axes(slab_dims), center=False).translate(
        point_from_axes(slab_origin)
    )
    seam_slab = seam_box ^ blank

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

def generate_mold(input_path, out_prefix, wall, up_axis, seam_axis, seam_fraction,
                   n_pins, pin_dia, fit_clearance, pin_margin,
                   funnel_top_dia, funnel_bot_dia, funnel_overlap,
                   pour_offset_xy, vent_dia, vent_offset_xy,
                   hollow=False, skin=2.0, open_back=False, cross_width=4.0,
                   sleeve=False, sleeve_wall=3.0, sleeve_clearance=0.3,
                   sleeve_flange_margin=3.0, sleeve_flange_thickness=2.5):

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

    # Seam plane: perpendicular to `seam_axis` (default z = horizontal, the
    # classic clamshell; x or y gives a vertical seam for models with side
    # undercuts instead of top/bottom ones), at `seam_fraction` of the
    # model's extent along that axis.
    seam_lo, seam_hi = model_bounds[seam_axis]
    seam_offset = seam_lo + seam_fraction * (seam_hi - seam_lo)

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
        body = hollow_body(model, blank, wall, skin, seam_axis, seam_offset,
                            open_back=open_back, cross_width=cross_width)
        # Give the pour channel(s) a proper wall-thickness jacket, since
        # otherwise whatever part of them is past cavity_shell's reach
        # around the model just dissolves into the general void.
        body = body + build_funnel_jacket(apex_xy, zmax, wall, funnel_top_dia,
                                           funnel_bot_dia, funnel_overlap, model)
        if vent_solid is not None:
            body = body + build_vent_jacket(vent_xy, zmax, wall, vent_dia, model)
        print(f"hollowed: {solid_body.volume():.0f} mm3 -> {body.volume():.0f} mm3 "
              f"({100 * (1 - body.volume() / solid_body.volume()):.0f}% material saved)")
    else:
        body = solid_body

    # 2c. Optionally cut a base opening and build the core insert, so the
    #     mold casts a hollow shell rather than a solid part. (Always tied
    #     to the up-axis/floor, independent of seam_axis.)
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
    #    this naturally divides the channel between both halves (the
    #    standard way real 2-part molds route a sprue across a vertical
    #    parting line), and for the default horizontal seam it lands
    #    entirely in the top half exactly as before.
    body = body - funnel_solid
    if vent_solid is not None:
        body = body - vent_solid

    # 4. Split on the seam plane
    seam_normal = point_from_axes({seam_axis: 1.0})
    top, bottom = body.split_by_plane(seam_normal, seam_offset)

    # 5. Registration pins
    blank_bounds = bbox_dict(blank.bounding_box())
    top, bottom = add_registration_pins(
        top, bottom, blank_bounds, seam_axis, seam_offset, wall,
        n_pins, pin_dia, fit_clearance, pin_margin,
    )

    if top.is_empty() or bottom.is_empty():
        raise RuntimeError("A mold half came out empty - check parameters.")

    # 6. Back to trimesh, rotate back to original orientation, export
    inv_rot = rot.T  # rotation matrices are orthonormal
    top_tm = rotate_mesh(manifold_to_trimesh(top), inv_rot)
    bottom_tm = rotate_mesh(manifold_to_trimesh(bottom), inv_rot)

    top_path = f"{out_prefix}_top.stl"
    bottom_path = f"{out_prefix}_bottom.stl"
    top_tm.export(top_path)
    bottom_tm.export(bottom_path)

    print(f"top half:    {top_path}  watertight={verify_watertight(top_path)}  "
          f"volume={top.volume():.1f} mm3")
    print(f"bottom half: {bottom_path}  watertight={verify_watertight(bottom_path)}  "
          f"volume={bottom.volume():.1f} mm3")

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

    return top_path, bottom_path, top_tm, bottom_tm, core_path, core_tm


# --------------------------------------------------------------------------
# Preview rendering (headless, no GPU needed - matplotlib with manual
# per-face Lambertian shading, since there's no pyrender/OpenGL to rely on
# in a plain server/CLI environment).
# --------------------------------------------------------------------------

def render_preview(top_tm, bottom_tm, out_path, up_axis="z", seam_axis="z",
                    explode=None, core_tm=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # The two returned pieces separate along `seam_axis` as named in the
    # internal up=Z working frame; top_tm/bottom_tm have already been
    # rotated back to the model's original orientation, so map that axis
    # through the same rotation to find which output axis it lands on.
    axis_idx = final_frame_axis(up_axis, seam_axis)

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
        # Sleeve mode: an exploded view would hide the fit, so instead cut
        # each assembled piece in half and view into the cut to show the
        # core sitting inside the cavity with the shell gap around it.
        # Cut via manifold3d's own trim_by_plane rather than trimesh's
        # slice_mesh_plane(cap=True), which pulls in scipy/shapely/rtree/
        # networkx just to cap a triangle hole - manifold3d is already a
        # hard dependency and produces a clean capped cut natively.
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
            explode = max(top_tm.extents.max(), bottom_tm.extents.max()) * 0.4
        parts = []
        for mesh, base_rgb, sign in [(bottom_tm, (0.35, 0.65, 0.85), -1.0),
                                      (top_tm, (0.95, 0.65, 0.25), 1.0)]:
            m = mesh.copy()
            shift = np.zeros(3)
            shift[axis_idx] = explode * sign
            m.apply_translation(shift)
            parts.append((m, base_rgb))
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
    p.add_argument("--seam-axis", choices=["x", "y", "z"], default="z",
                    help="Which axis the parting plane is perpendicular to: "
                         "z = horizontal/classic clamshell (default), x or y = "
                         "vertical seam, for models with undercuts on the sides "
                         "rather than top/bottom")
    p.add_argument("--seam", type=float, default=0.5,
                    help="Seam position as a fraction (0-1) of the model's "
                         "extent along --seam-axis")
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

    top_path, bottom_path, top_tm, bottom_tm, core_path, core_tm = generate_mold(
        input_path=args.input,
        out_prefix=args.out,
        wall=args.wall,
        up_axis=args.up,
        seam_axis=args.seam_axis,
        seam_fraction=args.seam,
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
    )

    if args.preview:
        render_preview(top_tm, bottom_tm, f"{args.out}_preview.png",
                        up_axis=args.up, seam_axis=args.seam_axis, core_tm=core_tm)


if __name__ == "__main__":
    main()
