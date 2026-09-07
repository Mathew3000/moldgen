# moldgen

A command-line tool that turns a watertight 3D model into a 2-part,
3D-printable mold. Give it an STL/OBJ/3MF, get back a top half and a bottom
half (plus, optionally, a core insert) as print-ready STLs.

No browser, no account, no upload — it's a single Python script that runs
locally.

```
python moldgen.py duck.stl --out duck_mold --wall 3 --pins 4 --preview
```

## What it does

For a model you'd like to cast in wax, soap, resin, plaster, or similar:

1. Builds a block around the model, padded by a wall thickness
2. Subtracts the model to carve out the cavity
3. Splits the block into two halves along a parting plane
4. Adds registration pins so the halves close in register
5. Bores a pour funnel down to the cavity, and an optional air vent
6. Exports both halves as print-ready, watertight STL files

All of the geometry work is boolean CSG done with the [Manifold](https://github.com/elalish/manifold)
library, which is what makes it fast and robust instead of falling over on
real-world meshes the way naive mesh-boolean code tends to.

## Requirements

```
pip install -r requirements.txt
```

- `numpy`
- `trimesh` — STL/OBJ/3MF import and export
- `manifold3d` — the boolean CSG engine
- `matplotlib` — only needed for `--preview`

No scipy, shapely, rtree, or networkx — the whole pipeline, including the
preview renderer, is built on the four packages above.

## Assumptions

- **The input mesh must already be watertight/manifold.** There's no repair
  step yet. A non-watertight input will either raise an error or silently
  produce a broken result — fix the mesh first (e.g. in your slicer's repair
  tool, Blender, or similar).
- **`--sleeve` mode assumes the model has a real flat base area**, not just a
  point of contact with the build plate, since it needs an actual footprint
  to cut the core opening from.

## Usage

```
python moldgen.py MODEL --out PREFIX [options]
```

### Core options

| Flag | Default | Description |
|---|---|---|
| `--out` | `mold` | Output filename prefix |
| `--wall` | `3.0` | Mold wall thickness (mm) |
| `--up` | `z` | Which model axis points "up" toward the pour side |
| `--seam-axis` | `z` | Which axis the parting plane is perpendicular to. `z` = horizontal/classic clamshell. `x` or `y` = vertical seam, for models with undercuts on the sides rather than top/bottom |
| `--seam` | `0.5` | Seam position as a fraction (0–1) of the model's extent along `--seam-axis` |
| `--preview` | off | Save a shaded preview PNG next to the STLs (exploded view, or a cutaway if `--sleeve` is set) |

`--up` and `--seam-axis` are independent: `--up` decides which way the
model sits and where the pour funnel goes; `--seam-axis` decides where the
mold splits. A tall figure might pour from the top (`--up z`) but still
need a vertical parting line (`--seam-axis x`) to release properly.

### Registration pins

| Flag | Default | Description |
|---|---|---|
| `--pins` | `2` | Number of pins: `2`, `3`, or `4` |
| `--pin-dia` | `5.0` | Pin diameter (mm) |
| `--fit-clearance` | `0.15` | Radial clearance added to sockets (mm) |
| `--pin-margin` | `1.5 × wall` | Inset of pins from the mold's outer footprint corners (mm) |

### Pour funnel and vent

| Flag | Default | Description |
|---|---|---|
| `--funnel-top-dia` | `10.0` | Funnel mouth diameter (mm) |
| `--funnel-bottom-dia` | `4.0` | Funnel diameter where it meets the cavity (mm) |
| `--funnel-overlap` | `2.0` | How far the funnel bore extends past the model's highest point as a straight channel (mm). Raise for thicker/more viscous pours; lower for a very slender/pointed model tip |
| `--pour-offset DX DY` | `0 0` | Offset the funnel from the model's footprint centre |
| `--vent-dia` | `0.0` | Vent hole diameter (mm), `0` = no vent |
| `--vent-offset DX DY` | `0 0` | Offset the vent from the model's footprint centre |

### Hollow mode

Replaces the solid mold body with a shell — a wall's thickness of material
around the cavity, a skin on the outside, void in between — to cut filament
use on larger molds.

| Flag | Default | Description |
|---|---|---|
| `--hollow` | off | Hollow the mold bulk into a shell |
| `--skin` | `2.0` | Outer skin thickness when `--hollow` is set (mm) |
| `--open-back` | off | With `--hollow`: leave off each half's outer face entirely (the one opposite the cavity, parallel to it) instead of skinning it over, replacing it with a "+"-shaped brace standing perpendicular to it — a vacuum-formed-shell look (contoured wall + side walls + open back) rather than a fully boxed-in shell |
| `--cross-width` | `4.0` | Width of each brace rib when `--open-back` is set (mm) |

Material savings scale with mold size: a small mold is mostly wall already,
so there's little to remove; a large one can save roughly half its volume,
more with `--open-back`. For a very wide, unsupported span of skin over the
void, consider adding internal ribs manually before printing, or leave
`--hollow` off for that mold.

### Sleeve mode (hollow casts)

Casts a hollow shell instead of a solid part. A separate core insert
occupies the model's interior during the pour; you pull it out afterward
through an opening in the mold's base.

| Flag | Default | Description |
|---|---|---|
| `--sleeve` | off | Cast a hollow shell; generates an extra `_core.stl` |
| `--sleeve-wall` | `3.0` | Thickness of the cast shell (mm) |
| `--sleeve-clearance` | `0.3` | Total diametral clearance between the core's stem and the mold's base opening (mm) |
| `--sleeve-flange-margin` | `3.0` | How far the core's flange extends past the base opening (mm) |
| `--sleeve-flange-thickness` | `2.5` | Flange thickness (mm) |

Workflow: clamp the two mold halves together, insert the core from below
(flange side) until the flange seats flush against the mold's underside,
pour, let it set, then pull the core back out. The flange doubles as both
a depth stop and a grip.

## Output files

| File | Produced when |
|---|---|
| `PREFIX_top.stl` | always |
| `PREFIX_bottom.stl` | always |
| `PREFIX_core.stl` | `--sleeve` |
| `PREFIX_preview.png` | `--preview` |

"Top"/"bottom" mean the +axis/−axis side of `--seam-axis`, whatever that
axis is — the names are kept for both a horizontal and a vertical seam for
consistency.

## Examples

Simple two-part mold, default settings:
```
python moldgen.py figurine.stl --out figurine_mold
```

Larger mold, hollowed to save filament, 4 pins, with a preview render:
```
python moldgen.py vase.stl --out vase_mold --wall 3 --hollow --skin 2 --pins 4 --preview
```

Vertical seam for a model with side undercuts:
```
python moldgen.py bust.stl --out bust_mold --seam-axis x --pins 4
```

Hollow-cast candle with a removable core:
```
python moldgen.py candle_master.stl --out candle_mold --sleeve --sleeve-wall 3 --preview
```

## Print settings

Whatever you're casting, favor more perimeters and finer layers over the
seam-facing surfaces — that's what actually keeps liquid from weeping
through the parting line, not infill. PETG over PLA if you're pouring
anything hot (e.g. wax).

## Known limitations / next steps

- No mesh repair — watertight input only, for now.
- `--pins` tops out at 4, placed at footprint corners; no auto-orientation
  or undercut detection to choose piece count automatically.
- Hollow mode leaves the void unsupported; very large spans may need
  internal ribs added separately.
- Pin placement is corner-based and can land in the cavity on thin or
  oddly-shaped parts — increase `--pin-margin` if that happens.
- `--open-back` combined with `--sleeve`: the core's flange is designed to
  seat flush against a solid outer face. With the outer face open, it only
  finds solid material where it lands on the side-wall ring or the cross,
  not across its whole area — usually fine, but check the flange size/
  position for your model if you combine the two.

### A note on watertightness checks

Earlier versions of this script trusted the in-memory mesh's own
`is_watertight` check when reporting status. That turned out to be
unreliable in both directions - a mesh could show `True` in memory and
fail once exported and reloaded (STL has no shared-vertex topology, so
export/reload reconstructs it from scratch and can expose a handful of
near-miss vertices or degenerate leftover faces from a long boolean chain),
or the reverse. The script now reloads each exported file to report its
actual status, and does a degenerate-face cleanup pass before export. If
you're on an older copy of this script and see `--hollow` output that
looks off in a slicer despite a "watertight=True" printout, that's why -
grab the current version.
