# 3D models

Files for 3D printers, laser cutters and CNC machines used to build the
vehicle.

The chassis CAD currently lives in [`../src/Hardware/`](../src/Hardware):

| File | What it is |
|---|---|
| `Chassis Vertikal.f3d` | Fusion 360 source (tracked via Git LFS) |
| `Chassis Vertikal.step` | neutral exchange format |
| `Chassis Vertikal.stl` | mesh for printing |

- [ ] move or copy the printable parts here, so the models sit where the
      judges look for them
- [ ] add one file per printed part instead of a single assembly, if the
      parts are printed separately

> Note: `src/` is checked out on the robot itself. Large CAD files placed
> there are pulled onto the Jetson on every clone even though it never
> needs them — this folder is the better home for them.
