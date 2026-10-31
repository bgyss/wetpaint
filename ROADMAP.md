# Roadmap

Where wetpaint goes after it matches a desktop natural-media painter: painterly looks for
animation and 3D, in Blender first and in 2D animation packages after that.

## Why this is feasible

Most painterly 3D looks (Spider-Verse, Arcane, *Puss in Boots: The Last Wish*) are painted
or composited per shot by hand. What's missing in open tools is a *simulated* medium that:

- is deterministic (same strokes, same frame, same pixels): `strokes_replay_deterministically`
  already tests this;
- runs headless and in batch;
- takes its strokes from animation data (a rig, Grease Pencil, or render passes) rather
  than from a mouse;
- stays coherent from frame to frame.

The engine already has the first two in part. The rest is the plan below.

## Phase 0: engine foundations

| Item | Why |
|---|---|
| Pen input: pressure, tilt and twist on macOS, Windows, Linux and the web | Rotating and tilting brushes, pressure curves. Needed by every brush feature below. |
| Bristle brushes that rotate with the shape, carve and build relief, and pick up paint per bristle | The biggest visible gap in oil and acrylic. |
| Paint loading and run-out along a stroke | Dry-brush endings; real-looking strokes. |
| Heightmap in physical units, with valleys, optional paper relief, and a normal-map export | Displacement and normal maps for 3D. |
| GPU simulation in the browser (async readbacks across frames) | Speed on the web. |
| A default brush library authored in Brush Creator | The engine needs good brushes to show what it can do. |
| `wetpaint-cli`: replay a stroke file (JSON / SVG paths with pressure, tilt and time) onto a document; simulate N steps; write PNG/EXR (colour, heightmap, normals, wetness) | The base of every integration below. |

## Phase 1: Python and Blender

1. **Python bindings** (`crates/py`, PyO3 + maturin). Covers documents, layers, brushes,
   `stroke(points)`, `step(n)`, and render to numpy and EXR. Build as an abi3 wheel per
   platform.
2. **Blender extension** (Blender 4.2+ extension format, bundling the wheel in
   `blender_manifest.toml`; archive root holds `__init__.py` and the manifest, relative
   imports only):
   - **Grease Pencil to paint.** Render Grease Pencil strokes through wetpaint brushes.
     Each GP point's position, pressure, strength and time becomes a stroke sample. The
     painted result is cached per frame as an image sequence, laid back on a plane or
     output as a render layer. This is the shortest path to painterly 2.5D rigs, which are
     built from GP drawings and textured planes driven by bones, shape keys and the view
     angle (see below).
   - **Texture-space painting.** Paint an object's UV texture with wetpaint, with water
     and drips flowing in UV space. Seams need care, tracked as research below.
   - **Screen-space stylisation** (a render post-process). Take Blender's render passes
     (colour, normals, depth, object/material IDs, vector) and have wetpaint lay strokes
     that follow surface direction and shading, then simulate wet media over them.
     Temporal coherence comes from carrying strokes forward with the vector pass.
3. **Exports for 3D.** Colour, heightmap and normal maps from a wetpaint painting as a
   Principled BSDF material: painted textures with real impasto relief.

### View-angle 2D rigs

The target is a hand-drawn puppet that redraws correctly as the camera orbits it. It
works like the classic multi-angle head turns of Flash and Harmony, but continuously and
live in the viewport. Drawn parts (head, hair, eyes, limbs) are keyed at a few view angles.
The camera's angle to the character selects, swaps or blends them, through drivers, shape
keys, Grease Pencil frames or geometry nodes. Today these rigs are built by hand, and
people keep asking how. wetpaint can help in three ways:

1. **Paint the angle keys.** Each part's drawing at each key angle is a wetpaint
   document. The rig's planes show the painted colour, plus normals and relief from the
   heightmap, so the puppet carries real paint texture and lights like paint.
2. **A rig kit.** A Blender extension that sets up the view-angle machinery itself:
   - compute the camera-relative azimuth and elevation per part;
   - map those angles to key drawings (nearest or blended);
   - control draw order per angle (an ear passes behind the head as the view turns);
   - a sheet that shows the angle keys for each part.

   This can be generic, independent of the paint engine, and still useful on its own.
3. **Generate keys from a 3D model** (the reverse direction). Render a 3D model or proxy
   at the key angles. Then lay painted strokes over each render by the screen-space
   method above, giving every angle a drawing in a consistent painted style. An artist
   then retouches the drawings, rather than painting them all from scratch.

The hard part is continuous motion between keys. Swapping drawings pops, and
cross-fading looks ghostly. Options to research:
- morphing between key drawings (the parameter-mesh deformation of Live2D-style tools);
- interpolating the stroke data itself, since wetpaint keeps strokes as paths: match
  strokes between keys, interpolate them and re-simulate the paint.

## Phase 2: 2D animation packages

- **Toon Boom Harmony:**
  - Batch round trip: Harmony exports drawing layers or vector strokes per frame, then
    `wetpaint-cli` paints and simulates them and writes EXR/PNG sequences back for
    compositing.
  - Harmony's scripting interface can automate the export and import.
  - A native node through Harmony's SDK is a later step, depending on SDK access.
- **OpenToonz, Krita:** open source, and both have plug-in or scripting routes. Krita's
  brush engine API is a natural home for a wetpaint brush.
- **Image-sequence workflow for any package:** frames plus a stroke file in, painted
  frames out.

## Phase 3: beyond a desktop painter

- **Animated wet media.** Simulation time follows the timeline. Wet washes spread over
  frames as an effect, and drying is art-directable per frame.
- **Stroke-level art direction.** Strokes are kept as editable data (path, pressure,
  brush, seed) and re-simulated on demand, so a look can be tuned after animating.
- **Resolution independence.** Tile-based simulation and replay at any output size.
- **Style transfer by example.** Fit brush parameters to a reference painting.

## Research list

1. Temporal coherence for stroke-based rendering:
   - Meier, *Painterly Rendering for Animation* (SIGGRAPH 1996): strokes attached to
     particles on 3D surfaces.
   - Bénard et al., *Stylizing Animation by Example* (SIGGRAPH 2013).
   - Bénard et al., *A Dynamic Noise Primitive for Coherent Stylization* (2010).
2. Watercolour stylisation of 3D:
   - Montesdeoca et al., *Art-directed Watercolor Stylization of 3D Animations in
     Real-time* (2017).
   - The MNPR framework (Maya, open source).
   - Bousseau et al., *Interactive Watercolor Rendering with Temporal Coherence and
     Abstraction* (NPAR 2006).
3. Painting in 3D space: Schmid et al., *OverCoat: An Implicit Canvas for 3D Painting*
   (SIGGRAPH 2011).
4. Stroke placement from images: Hertzmann, *Painterly Rendering with Curved Brush Strokes
   of Multiple Sizes* (SIGGRAPH 1998).
5. Blender integration details:
   - Grease Pencil v3 data access (Blender 4.3+);
   - shipping native wheels in extensions;
   - whether a custom `RenderEngine` or a post-render handler suits the screen-space
     path;
   - caching per frame in the blend file.
6. Harmony: what its scripting interface can export (vector stroke data or bitmaps only),
   and what access its SDK allows.
7. UV-space flow: water crossing UV seams; anisotropic paper on curved surfaces.
8. View-angle rigs:
   - how existing multi-angle rigs key and blend views (Harmony head turns; 2.5D
     multi-angle sprites in games);
   - parameter-mesh deformation of 2D art (Live2D-style);
   - stroke correspondence and interpolation between drawn keys: vector inbetweening,
     e.g. Whited et al., *BetweenIT* (2010).
