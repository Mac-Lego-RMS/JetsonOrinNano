# JetsonOrinNano — WRO Future Engineers

Repository of the self-driving vehicle: control software, hardware
documentation and the material required by the WRO Future Engineers rules.

## Structure

| Folder | Content |
|---|---|
| [`src/`](src) | ROS 2 workspace — all control software |
| [`models/`](models) | files for 3D printing, laser cutting, CNC |
| [`schemes/`](schemes) | schematic diagrams of the electronics |
| [`t-photos/`](t-photos) | team photos |
| [`v-photos/`](v-photos) | vehicle photos from every side |
| [`video/`](video) | `video.md` with the link to the driving demonstration |
| [`other/`](other) | datasets, specifications, protocol descriptions |

Each folder carries a README listing what still has to go in.

## Jetson

On the robot the repository lives in `~/ros2_ws` with a sparse checkout on
`src/`, so `git pull` there fetches the workspace only — the photos, models
and documentation stay off the vehicle.

`src/` contains files tracked with **Git LFS** (`*.f3d`). Install it before
cloning, otherwise Git aborts on every command:

```bash
brew install git-lfs && git lfs install     # macOS
sudo apt install git-lfs && git lfs install # Ubuntu / Jetson
```
