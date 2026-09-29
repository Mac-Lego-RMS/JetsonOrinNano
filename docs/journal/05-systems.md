# Systems thinking and engineering decisions

<!--
Owner: all. Rubric criterion 4.
4 points: subsystems mapped and their interactions explained; constraints
mentioned.
6 points: explicit constraints; trade-offs; iteration cycles; risks and failure
modes with mitigation; "we chose X instead of Y because ..." based on data.
DRAFT: software rows filled from the team's answers; mechanics / power rows
are for Clemens and Jannik.
-->

## Subsystems and how they interact

<!-- TODO: one block diagram mechanics / power / sensors / compute / software
and a paragraph per interface. Examples that already exist:
- steering play and servo curve -> measured steering table in the bridge
- camera frame rate drops while driving -> scan halt in lap 1
- LiDAR blocked behind by the mount -> 270 deg usable view
- start pose in the bay is the map origin -> parking accuracy depends on it -->

## Constraints

| Constraint | Consequence |
|---|---|
| Ties are broken by time (national final: same points as two other teams, 4th place on time) | faster driving needed a pose that does not depend on every single scan -> EKF |
| Colour is only reliable at standstill (frame rate 15.5 -> 2.5 Hz, coloured points 38 -> 2 % while driving) | scan halts in lap 1 only; map frozen after lap 1 |
| Colour only trusted up to 1.60 m | votes from further away count as "something there" only |
| Six CPU cores shared by fusion, estimation and control | CPU load reduced from ~92 % to ~53 % (commit a14524e) |
| The field is fourfold symmetric | global scan matching finds poses rotated by 90°; the start pose must come from start detection |
| Start procedure (rules 9.10–9.14): one switch, one start button, nothing measured before it | container and controller start from the autostart (boot 85–90 s) and wait for the button; direction, position and bay are detected after it |

## Decisions and trade-offs

| Decision | Alternatives considered | Why | Evidence |
|---|---|---|---|
| EKF with wall features | pose from every scan (wall follower, national final); ICP scan matching; particle filter | gyro and encoder bridge missing or wrong scans; the field is known and simple, so a few walls per scan are enough; the innovation of every wall is visible | ICP ran into local minima in our evaluations; global search gave 90° rotated poses (field symmetry) |
| Split-and-merge + SVD fit | RANSAC, Hough transform | uses the angular order of the scan, deterministic, returns segment end points (bay wall vs. front wall) | scan callback ~6 ms median; heading drift 25° -> ±1.5° after splitting at corners |
| Colour per LiDAR point (fusion) | YOLOv11n on the camera image (national final) | latency; distance and colour in one measurement | TODO latency comparison |
| 270° fisheye | 120° CSI camera | sees the next straight before the corner | 3 vs. 6 of 6 seats at the scan halt (fov_coverage) |
| RPLIDAR S3 | LD09 | higher resolution and scan rate at a similar size | – |
| Stanley | PD on lateral and heading error | one law for straights, arcs and ramps, steering angle directly | ±8° oscillation removed with dead-time prediction (260 ms) |
| Tangential arcs, cosine ramps | front-loaded ramp | constant curvature = constant feed-forward; ramps tangential at both ends | front-loaded ramp: ~15° kink |
| Staged gate for wall matching | gate from the EKF covariance | covariance 0.2 -> 5.8 cm in 30 s while the real error reached metres | – |
| Collect detections over several views while driving, scan halts in lap 1 | stop at the end of every straight in every lap | stopping costs time; one view cannot resolve two pillars in one row | TODO A/B test scan halt |
| Controlled approach to the parking start pose (Stanley) | blind 55 cm ESP move | the blind move ended 12 cm beside and 18° skewed | – |
| One reverse move with continuous steering | about nine 5 cm ESP moves | simulation: 0.4 cm instead of 3.4 cm final error | – |
| Unpark variant from the nearest pillar in front | pillar row 1.5 m before the front wall | the bay position varies with the layout; a green pillar right in front was missed | – |
| Mask the bay geometrically | detect the magenta bay walls with the camera | magenta not detected reliably enough | – |
| Jetson Orin Nano + ROS 2 | Raspberry Pi 4 with plain Python classes (last season) | compute for fusion; ROS 2 as industry standard, bags for offline testing | – |

## Iterations

| Version | When | What changed | Result |
|---|---|---|---|
| v0 | season 2025 | Raspberry Pi 4 -> Jetson | – |
| v1.0 (tag, commit 40f0dad) | national final, June 2026 | LiDAR wall follower (PID), YOLOv11n, IMU turn counting | full driving score, 29/30 documentation, 4th place on time |
| v2 (now) | after the national final | EKF + map, RPLIDAR S3, fisheye, camera-LiDAR fusion, Stanley, encoder | TODO numbers from runs.csv |

Parking alone went through three iterations in the test runs (table in the
software chapter): 6/8, 2/6 and 5/5 runs within the 2 cm rule.

## Risks and failure modes

| Failure mode | Effect | Detection | Mitigation |
|---|---|---|---|
| Colour misread while moving | pillar passed on the wrong side | – | colour only at standstill / low yaw rate, votes, 1.60 m limit |
| Phantom pillar | unnecessary lane change, crash into the inner wall | LiDAR sees through the seat | seat cleared after 6 see-throughs |
| Localisation lost | wrong path | localisation state | staged gate, slow down, emergency stop after 2 s |
| Gyro failure (I2C) | heading drifts | watchdog on messages / zeros | `gyro_ok` -> localisation lost -> stop |
| Robot moved before the start / map from the previous run | parks at the wrong place | – | estimation restarted for every run, controller waits until gyro, bay and localisation are ok (40 s timeout) |
| Second estimation node running | map latched 8° rotated (run parken_test_14) | only doubled log lines | single-instance check at node start |
| Camera re-enumerates on USB | no colour | – | fixed device name via udev |
| ESP reboot / clock jump | wrong time stamps | time sync detects the reboot | resync |
| Wall contact | robot pushes against the wall | LiDAR < 4 cm in front | stop and back up (max. 2 per corner) |

## Known limitations

- Colour detection while driving is not reliable; the robot needs the scan
  halts.
- No de-skewing of the LiDAR scan during fast turns.
- Parking depends on the start pose in the bay.
- Parts of the algorithms are complex and need good sensor data.
