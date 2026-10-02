# Third-party notices

WasserMan depends on software installed separately and includes a small number of
third-party simulation assets. Each component retains its own terms.

## Isaac Lab

Copyright NVIDIA CORPORATION and contributors. BSD-3-Clause.

WasserMan uses an unmodified source checkout of the `release/3.0.0` branch pinned
at commit `76c7c60de65eb5196dbfacac91d3febbff6044d2` as an editable dependency.

Source: <https://github.com/isaac-sim/IsaacLab>

## Isaac Sim

Copyright NVIDIA CORPORATION. Distributed under NVIDIA's applicable license
and EULA. Isaac Sim is installed separately and is not redistributed here.

Source: <https://developer.nvidia.com/isaac/sim>

## AM-Bench push-button asset

- Vendored path: `src/wasman/assets/data/objects/ambench/push_button.usd`
- Upstream path: `source/ambench/ambench/assets/objects/push_button.usd`
- License: Apache-2.0
- Copyright: 2026, The AM-Bench Contributors

The USD is redistributed unchanged. Its retained notice is stored in
`src/wasman/assets/data/objects/ambench/LICENSE.txt`.

Source: <https://github.com/ambench/ambench>

## Angler robot description

The composite BlueROV2 Heavy + Reach Alpha 5 URDF adapts robot topology,
kinematics, inertial data, and collision approximations from Angler and its
Blue/Alpha descriptions.

- License: MIT
- Copyright: 2023, Evan Palmer
- Vendored output: `src/wasman/assets/data/robots/bluerov2_alpha/bluerov2_alpha.urdf`

Sources: <https://github.com/Robotic-Decision-Making-Lab/angler>,
<https://github.com/evan-palmer/blue>, and
<https://github.com/evan-palmer/alpha>

## BlueROV2 Heavy visual mesh

- Vendored path: `src/wasman/assets/data/robots/bluerov2_alpha/meshes/blue/bluerov2_heavy_reach.dae`
- Separate original T200 rotor meshes: `meshes/blue/cw_prop.dae` and
  `meshes/blue/ccw_prop.dae`, copied from the same Blue repository's
  `blue_description/meshes/t200/`. Mount orientation and handedness follow its
  Heavy Reach / T200 descriptions; a measured CAD registration offset aligns
  their centers with this project's imported housing mesh.
- License: MIT
- Copyright: 2022, Evan Palmer

The full retained license is in
`src/wasman/assets/data/robots/bluerov2_alpha/BLUE_LICENSE.txt`.

## T200 static performance samples

The numerical sample table in `src/wasman/physics/data/t200_16v.json` is derived
from Blue Robotics' public 16 V measurement sheet:
<https://cad.bluerobotics.com/T200-Public-Performance-Data-10-20V-September-2019.xlsx>.
Manufacturer attribution, workbook checksum and units are retained in the JSON.
The full workbook is cached locally, not redistributed. These static samples
do not establish the assumed motor delay/time constant used by WasserMan-Bench.

## Reach Alpha visual meshes — restricted terms

The Alpha arm meshes below are **not under a permissive open-source license**:

`src/wasman/assets/data/robots/bluerov2_alpha/meshes/alpha/`

Their supplied license limits use to products manufactured or developed by
Reach Robotics Pty Ltd and prohibits other redistribution or integration
without prior written authorization. The exact retained terms are in
`src/wasman/assets/data/robots/bluerov2_alpha/REACH_MESH_LICENSE.txt`.

These visual CAD files are deliberately excluded from Git and release payloads.
Install separately only when your authorization permits the intended use. Their
absence does not grant permission to use or redistribute them. The benchmark's
original geometry is preserved; no replacement physics or success predicate is
introduced by packaging.
Source: <https://github.com/evan-palmer/alpha>

## Open procedural arm profile

The separately versioned `open-procedural-v1` profile uses the MIT-derived
joint/inertial description above, primitive arm visuals and newly specified
box-segment grasp fingers. The converter neither reads nor approximates the
restricted Reach meshes. This profile changes images and finger collisions;
it is not a reproduction of the historical CAD configuration. New datasets and
results identify their profile explicitly. The historical packaging statement
above continues to apply to `historical-cad-v1` only.

## MarineGym

Copyright MarineGym contributors. MIT.

MarineGym's hydrodynamic decomposition and BlueROV-class coefficient scale
were used as a scientific reference. WasserMan implements its own batched model
against the Isaac Lab 3.0 wrench API and does not include MarineGym source.

Source: <https://github.com/Marine-RL/MarineGym>

## Industrial valve — Poly Haven

Jorge Camacho, **Modular Industrial Pipes 01**, CC0-1.0.
Source: <https://polyhaven.com/a/modular_industrial_pipes_01>.
License: <https://polyhaven.com/license>.

`src/wasman/assets/data/objects/industrial_valve/` contains an adapted valve
module, separated into fixed body and rotating wheel, and unchanged 2K base
color/roughness/metallic/normal maps. Uniform scale is 0.65; source URLs and
SHA-256 hashes are in `provenance.json`. Contact proxies and assumed physical
parameters are WasserMan additions, not Poly Haven measurements or a certified
commercial subsea valve specification.


## UUV Simulator / uuv_manipulators

The RexROV2 and Oberon7 descriptions and the referenced hydrodynamic data retain
Apache-2.0 terms and their original contributor notices. The full licenses and
notices are kept next to the derived assets in
`src/wasman/assets/data/robots/rexrov2_oberon7/` and
`src/wasman/physics/data/rexrov2/`. Project-specific rendering and controller
adaptations do not constitute hardware validation.

Sources: <https://github.com/uuvsimulator/uuv_simulator> and
<https://github.com/uuvsimulator/uuv_manipulators>.

## Policy implementations and pretrained encoders

ACT (LeRobot) and Diffusion Policy (AM-Bench) sources are fetched separately at
exact commits with verified file hashes. Their licenses and upstream notices are
retained by those installations; this project's license does not replace them.
See `research/policy-sources.json`, `research/policy-requirements.txt` and the
HotStab runtime's `requirements/visual-policy-sources.json`. Selected research
checkpoints include their trained image encoders; the result registry records
provenance. No Isaac Sim binaries or licensed robot CAD are bundled here.

## Website fonts

The separate website archive retains the Manrope and DM Mono font license files
alongside its source fonts. Its rendered robot demonstrations do not include the
underlying restricted CAD source files.
