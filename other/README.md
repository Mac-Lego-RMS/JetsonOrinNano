# Other

Everything else needed to understand how the vehicle is prepared for the
competition. These files live next to the code or the electronics they
describe; this page points to them.

| What | Where |
|---|---|
| Serial protocol Jetson ↔ ESP32-S3 | [`src/esp_firmware/docs/JETSON_BRIDGE.md`](../src/esp_firmware/docs/JETSON_BRIDGE.md) |
| Bill of materials and placement file of the main PCB | [`schemes/MainPCB/production/`](../schemes/MainPCB/production) |
| Setting up the Jetson: Docker image, device rules, autostart | [`setup/README.md`](../setup/README.md) |
| Calibration files loaded at run time | [`config/`](../config) |
| Measurement data behind the journal | [`docs/data/`](../docs/data), workbook [`mobility_measurements.xlsx`](../docs/data/manual/mobility_measurements.xlsx) |
| Scripts that turn recorded runs into figures | [`docs/analysis/`](../docs/analysis) |
