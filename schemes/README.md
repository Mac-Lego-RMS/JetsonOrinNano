# Schematic diagrams

Diagrams of the electromechanical components: every electronic component
and motor used in the vehicle, and how they are wired together.

- [ ] Jetson Orin Nano ↔ ESP32-S3 (UART, `/dev/ttyTHS1`, 115200 baud)
- [ ] ESP32-S3 ↔ drive motor and steering servo
- [ ] sensors: RPLIDAR S3, BNO055 IMU, USB camera
- [ ] power distribution (battery, regulators, fusing)

A photo of a clean hand-drawn diagram is acceptable; a KiCad/Fritzing
export is nicer.

## Contents

| Path | Content |
|---|---|
| [`MainPCB-schematic.pdf`](MainPCB-schematic.pdf) | schematic of the main board, exported from KiCad |
| [`MainPCB/`](MainPCB) | KiCad 10 project of the main board (ESP32-S3, motor driver, servo bus, Jetson/lidar connectors, power supply) |
| `MainPCB/libraries/` | parts that are not in the stock KiCad libraries (MAX17504 buck converter, XT30 connector) |
| `MainPCB/3d/` | STEP model of the assembled board |
| `MainPCB/production/` | BOM and placement files of the last order (V5) |

The project came over from the FutureEngineers repository together with
its commit history (`git log -- schemes/MainPCB`).

### Opening the project

Open `MainPCB/MainPCB.kicad_pro` in KiCad 10. The two local libraries are
registered in the project's own `fp-lib-table` / `sym-lib-table`, so
nothing needs to be set up. Some parts come from **easyeda2kicad**; their
footprints and symbols are embedded in the board and schematic, only their
3D models need the `EASYEDA2KICAD` path variable.

### Updating the PDF

After changing the schematic, regenerate the export:

```bash
kicad-cli sch export pdf -o schemes/MainPCB-schematic.pdf schemes/MainPCB/MainPCB.kicad_sch
```

Gerbers are not kept in the repo; generate them from the board when
ordering.
