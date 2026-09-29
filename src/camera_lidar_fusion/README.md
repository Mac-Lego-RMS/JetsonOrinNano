# camera_lidar_fusion

Marries the horizontally mounted 360 degree fisheye camera (USB, `/video_source/raw`,
1280x960) with the 2D lidar. Two nodes:

| Node | Purpose |
| --- | --- |
| `lidar_pixel_mapper` | Colour per lidar point -> CSV, PointCloud2, debug image |
| `rotation_calibration` | Measure the image circle and determine the rotation of the camera |

## Geometry

Robot/lidar frame according to REP-103: **X front, Y left, Z up**.

The camera lies on its back, so the optical axis points up. With a
270 degree opening angle the field of view reaches 45 degrees below the horizon -- so
it sees all round and slightly downwards. That is exactly what we need to pick up
obstacles already at the corner entry.

For the nominal case (axis exactly vertical) the optical frame coincides with the
robot frame, the rotation matrix is the identity. The
calibration angles only describe the deviation:

* `yaw_deg` -- rotation about the optical axis. **This is the rotation this is
  all about.** It only depends on the azimuth, not on the height of the target object,
  and can therefore be solved in closed form.
* `pitch_deg` / `roll_deg` -- tilt of the axis from the vertical.

Projection: `P_cam = R @ (P_robot - t)`, then equidistant fisheye
`r = f * theta` with `f = radius_px / (fov/2)`. Set `poly_coeffs` if needed.

Important: the lidar plane lies **below** the camera, so `theta > 90 deg`
for all ground points. That is correct and lies within the 135 degrees.

### Where in the image it samples: `sample_mode`

**`horizon` (default).** The lidar point is sampled at lens height. The
height difference to the camera is then zero, `theta` exactly 90 degrees and the
image radius constant `f*pi/2` = 301.3 px -- independent of the distance. Only
the azimuth is left, i.e. **a fixed circle in the image**.

That is enough for pylons as long as the lens sits between the mat and the
pylon top: a pylon that pierces the horizontal plane through the lens
lies on this ring at *every* distance. Measured on real
lidar data (2203 points, 0.05 to 2.96 m): radius span 0.013 px.

The gain is not only simplicity -- radially two error sources drop out
completely: the range measurement of the lidar and a wrong `cam_z`. Only
`yaw` is left. Exactly what you need for distant obstacles.

How much margin the ring has inside the pylon (10 cm pylon, distance from the ring to the top and
bottom edge in px):

| Lens height | 0.3 m | 1.0 m | 2.0 m |
| --- | --- | --- | --- |
| 2 cm | 50.0 / 12.8 | 15.3 / 3.8 | 7.7 / 1.9 |
| **5 cm** | **31.7 / 31.7** | **9.6 / 9.6** | **4.8 / 4.8** |
| 8 cm | 12.8 / 50.0 | 3.8 / 15.3 | 1.9 / 7.7 |
| 11 cm | miss | miss | miss |

At half the pylon height the distance to both edges is largest -- **that is where
the lens should sit**. Above the pylon top the ring misses the
pylon and reads the wall behind it; then use `height`.

**`height`.** Sampling at a fixed height `sample_height_m` above the lidar plane,
the image radius depends on the distance. Only needed if the lens does not sit between
the mat and the pylon top.

### A zone instead of a line

A single sampling radius is fragile: depending on distance and
calibration error it hits the pylon one time, the wall behind it the next, the floor in front of it the next. With
a zone set, a piece of the radial line is scanned instead and
it is counted which fraction of the pixels matches which colour; from
`sample_zone_min_frac` on a colour wins.

**Vote, do not average.** A median over a segment that lies half on the
pylon and half on the wall gives a mishmash. The vote fraction stays
meaningful as long as the pylon fills a significant part of the segment.
Cross-check on the setup: if you simply take the most saturated pixel instead of the vote,
you find something in almost every line and produce
clusters 30 degrees wide where a pylon would have 5 degrees.

**The zone is not of constant thickness.** A wall band of fixed height does not appear in the
fisheye as a circular band of equal thickness -- and that is the heart of the matter. If
the lens sits at the height of the top edge of the wall band, the height difference for this edge is
zero, `theta` therefore exactly 90 degrees and the image radius constant:
the top edge runs as a straight line. The bottom edge lies one band height
lower, its `theta` approaches 90 degrees from above as the distance grows,
so its radius approaches that of the top edge from outside:

| Distance | Zone | Thickness |
| --- | --- | --- |
| 0.3 m | 392 .. 445 px | 53 px |
| 1.0 m | 398 .. 413 px | 15 px |
| 3.0 m | 399 .. 405 px | 6 px |

A width fixed in pixels would therefore be much too narrow near and too
wide far away -- far away it sticks out above the wall band and also collects the bright mat or
the wall, so that points wrongly come out as `unknown` instead of `black`.

Where the zone limits come from is decided in this order:

1. **`zone_from_band: true`** -- bottom edge live from the image (see below),
   top edge from the calibration. The best there is.
2. **Calibrated curve** in the calibration file (`zone` / `zonefit`).
3. **`sample_zone_low_m` / `sample_zone_high_m`** -- computed from two heights.
   Only as a fallback, see model error further below.

```bash
ros2 param set /lidar_pixel_mapper sample_zone_min_frac 0.5   # vote fraction
ros2 param set /lidar_pixel_mapper sample_zone_steps 13       # samples
ros2 param set /lidar_pixel_mapper sample_zone_use 1.0        # 0.4 = middle third
```

`sample_zone_use` only scans the middle part of the zone. If the limits sit
cleanly, the middle is the purest spot -- the edges contribute mixed pixels.
Going all the way down to one line is risky though: at 3 m the whole zone is only
6 px thick, 40 percent of that are two pixels, and then everything again depends on
the zone sitting right to the pixel.

The vote fraction must match the zone width. Measured on the setup, with two
green and two red pylons on the field:

| `sample_zone_min_frac` | green clusters | red clusters |
| --- | --- | --- |
| 0.20 | 7 | 3 |
| 0.40 | 4 | 2 |
| **0.50** | **2** | **2** |

At 0.50 exactly the four real pylons were left, without false hits.

### Measuring the zone instead of computing it: `zone` and `zonefit`

COMPUTING the zone from `cam_z` and the focal length does not work well
enough. The reason is a model error that grows towards the image edge -- exactly
where we work: `poly_coeffs` is empty, so the model computes strictly
equidistant with `r = f*theta`, and real fisheye lenses deviate from that at the
edge. On the setup that was **9 to 12 px**. You can see the effect in
that a back-calculation of the top edge of the wall band gave three different heights for
the same edge: +10.1 cm at 0.5 m, +5.5 cm at 1.0 m, +12.7 cm at
2.5 m. With a single height, near and far can therefore not be hit at the
same time.

That is why the zone is measured. The commands run in `rotation_calibration`:

```bash
# place the pylon at one distance, then:
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: zone"
# move it, repeat -- 5 to 6 positions from 0.3 to 2.5 m

ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: zonelist"
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: zonedel 3 7"
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: zonefit"
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: save"
ros2 topic pub --once /camera_lidar/reload std_msgs/msg/Empty "{}"
```

`zone` scans the radial line through the lidar cluster and measures over
which radius range the pylon colour stands there -- i.e. exactly the range that
the mapper is to sample later. `zonefit` fits

    r(rho) = r0 + k / rho

through the measurements, separately for inner and outer. This form is not
guessed, it follows from the geometry: for `theta` near 90 degrees
`atan2(rho, dz)` is roughly `pi/2 - dz/rho`, so `r` is roughly
`f*pi/2 - f*dz/rho`. The 1/rho part carries the height of the edge, the constant
part the focal length -- **and with it the model error as well**, which you cannot
get rid of anywhere else. Exactly that is the gain over computing.

On the setup, 9 samples from 0.27 to 2.14 m:

```
r_inner = 399.9  -2.31/rho     RMS 1.0 px
r_outer = 400.1 +13.33/rho     RMS 1.6 px
```

The computed model puts the horizon ring at 412.3 px, the fit tends towards
399.9 px -- the difference of 12.4 px is the model error. And `zone_k_in` of
-2.31 corresponds to only 0.9 cm: the lens sits practically at the height of the
top edge of the wall band, whose image radius is therefore constant.

**Spread widely.** The fit separates a constant part from a 1/rho part and
needs near AND far for that. Below a spread factor of 2.5 `zonefit` warns.

For the commands: first `background` (otherwise the node finds our own
build instead of the pylon, see below), and set `target_label` to the pylon colour.
`target_range_max_m` is at 1.5 m -- raise it for samples further out.

### Finding the wall band live: `band_detect`

The most reliable edge in the image is the **bottom edge** of the black wall band,
i.e. the transition to the bright mat: behind it there is always the same thing, whatever the
direction. The top edge is no good for this, behind it there is white
wall, dark couch or wood. Measured on 1362 edge pairs:

| Edge | RMS of the fit |
| --- | --- |
| bottom edge (against the mat) | 5.3 px |
| top edge (against the room) | 12.8 px |
| for comparison: pylon colour (`zonefit`) | 1.0 / 1.6 px |

`_find_band` walks from the inside outwards per azimuth and takes the first
spot where it stays bright -- `band_run` pixels in a row. That way
a single highlight on the wall band does not trigger the edge early. On the setup
this hits **349 of 360 azimuths**.

Two outlier filters, both physically justified: the bottom edge must always
lie further out than the top edge (the wall band is about 10 cm high), and
neighbouring azimuths must be similar (the wall band does not jump). The filter
shrank the detected radius range from 368..436 to 403..436 px.

```bash
ros2 param set /lidar_pixel_mapper band_detect true
ros2 param set /lidar_pixel_mapper zone_from_band true   # align the zone to it
ros2 param set /lidar_pixel_mapper band_steps 360        # azimuth resolution
ros2 param set /lidar_pixel_mapper band_dark_max 60      # this dark is the wall band
ros2 param set /lidar_pixel_mapper band_bright_min 100   # this bright is the mat
ros2 param set /lidar_pixel_mapper band_run 4            # bright pixels in a row
ros2 param set /lidar_pixel_mapper band_smooth 9         # median window
ros2 param set /lidar_pixel_mapper band_max_dev 12.0     # max deviation [px]
```

With `zone_from_band` the bottom edge of the zone comes live from the image, the
top edge stays constant. Where no edge was found, the
calibration curve applies -- so the band search can only improve, never make it worse.

### If the ring sits too high: calibrate the focal length

The ring lies at `r = f*pi/2`, and `f = radius_px / (fov/2)`. Up to this point the FOV was
an **assumption** (270 degrees from the product description), never measured.
If it is wrong, the ring sits at the wrong radius -- and because an FOV assumed too large
makes `f` too small, it then sits too far **inside**, i.e. too high
in the room, and looks over the pylons.

| assumed FOV | ring at |
| --- | --- |
| 270 deg | 301 px |
| 240 deg | 339 px |
| 220 deg | 370 px |
| 200 deg | 407 px |
| 180 deg | 452 px (= edge of the image circle) |

**Quick way -- set the ring by hand.** The ring is drawn in orange in the debug image.
Move it until it lies at pylon height:

```bash
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: ring"
ros2 param set /camera_rotation_calibration horizon_radius_px 370
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: save"
```

**Clean way -- `radial`.** Sample one pylon at several distances
(important: **spread widely**, e.g. 0.2 to 1.5 m) and then:

```bash
ros2 param set /camera_rotation_calibration pylon_height_m 0.10
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: radial"
```

The foot point of the pylon stands on the mat, i.e. always `L` below the lens:
`theta_foot = pi - atan(L/rho)`. If the pylon moves from 0.2 to 2 m, this
angle runs from about 158 to 92 degrees -- this spread makes `f` and `L`
jointly determinable. The top edge gives the same equation with `L - H`.

`radial` therefore also gives you **the lens height above the mat** -- and tells
you directly whether the horizon ring can work at all or whether the camera
has to go lower.

### Tilting the ring down: `sample_depression_deg`

Tilts the ring down by X degrees; the horizontal plane becomes a cone.
The image radius stays constant (`f*(pi/2 + X)`), but the **sampling depth below
the lens grows with the distance**: `rho * tan(X)`.

| `sample_depression_deg` | 0.3 m | 1.0 m | 2.0 m |
| --- | --- | --- | --- |
| 0.5 deg | 0.3 cm | 0.9 cm | 1.7 cm |
| 1.0 deg | 0.5 cm | 1.7 cm | 3.5 cm |
| 3.0 deg | 1.6 cm | 5.2 cm | 10.5 cm (below the mat) |

For 10 cm pylons only fractions of a degree are usable. And the
hard case: if the lens sits **above** the pylon top, there is no
angle at all that hits near and far at the same time -- at 13 cm lens height
0.3 m needs between 5.7 and 23.4 degrees, but 2.0 m between 0.9 and 3.7 degrees. The
windows do not overlap. Then only `height` helps.

Hence: bringing the lens to pylon height is the solution, not the tilt angle.

### Averaging instead of one pixel: `sample_band_m`

Instead of a single pixel, `sample_band_count` samples are read along
the **radial** line through the point -- which runs along the pylon in the fisheye
-- and their **median** is taken (not the mean: the
median holds out when one end of the band slips past the pylon edge).

The band width is given in metres of pylon height and converted per point from the
distance to pixels (`f*atan(band_m/rho)`). Far away the band therefore shrinks
by itself and automatically stays inside the pylon.

```bash
ros2 param set /lidar_pixel_mapper sample_band_m 0.03   # default: +-3 cm
ros2 param set /lidar_pixel_mapper sample_band_m 0.0    # off, one pixel
ros2 param set /lidar_pixel_mapper sample_band_count 5
```

**On compute time:** `patch_px` filters the *whole* image via `medianBlur` and
costs about 26 ms per scan on 1280x960 -- a good 40 percent of a core at 15 Hz.
As long as the band is active that is superfluous, which is why `patch_px` is set to
1. Only raise it if you set `sample_band_m` to 0.

| Step | Time per scan (2200 points) |
| --- | --- |
| `medianBlur` 1280x960 | 26.4 ms |
| band sampling, 5 samples | 1.8 ms |
| classification | 0.7 ms |
| projection | 0.4 ms |

See also the section **Compute time in operation** further below -- it has
the numbers for zone, wall band detection and debug image.

```bash
ros2 param set /lidar_pixel_mapper sample_mode horizon
ros2 param set /lidar_pixel_mapper sample_mode height
ros2 param set /lidar_pixel_mapper sample_height_m 0.0   # only with height
```

Both parameters are read again on every scan, so they take effect immediately.

### The z height

`cam_z` (camera above the lidar plane) is part of the translation and is therefore
fully accounted for -- it determines `theta` and thus the **image radius**. Yaw,
on the other hand, only determines the **angle**. The two are orthogonal and
do not interfere with each other.

How strongly z acts depends on the distance:

| Error | 0.2 m | 0.5 m | 1.0 m | 2.0 m |
| --- | --- | --- | --- | --- |
| `cam_z` 1 cm off | 8.9 px | 3.8 px | 1.9 px | 1.0 px |
| `cam_z` 2 cm off | 17.6 px | 7.6 px | 3.8 px | 1.9 px |
| `yaw` 1 deg off | 6.1 px | 5.6 px | 5.4 px | 5.3 px |

So: near, z counts, far away it vanishes -- for obstacles at the corner entry
(i.e. far away) yaw is what matters. A centimetre of measurement error with the
ruler costs you less than 2 pixels at 1 m.

That applies to `sample_mode: height`. With `horizon` the influence of
`cam_z` on the sampling drops out completely -- there `cam_z` is only needed
to hit the lens height, and the ring stays the same.

From images, only the **height difference** between camera and target mark can ever be
determined, never both separately. `height` therefore solves `cam_z` under the
assumption that `target_height_m` (height of the colour blob centroid above the
lidar plane) is right. Measuring with a ruler is more accurate; `height` is the
cross-check.

### The zero point of the colour index: `white_point`

`rg_index` silently assumes that a colourless surface gives
`z = (G-R)/max(B,G,R) = 0`. It only does so if the white balance of the
camera matches the light. Measured again on 16.09.2026 (bag `wb_test`, 266 frames,
`CAM_WB_TEMP=4600`): the **white mat** gives B=167 G=212 R=194, i.e.
**z = +0.084 instead of 0**.

So the symmetric thresholds `+-rg_z_min` are in truth completely
asymmetric:

| Colour | required colour swing |
| --- | --- |
| green | 0.150 - 0.084 = **0.066** |
| red | 0.150 + 0.084 = **0.234** |

Red therefore has to be three and a half times as strong as green. Near, this does not
show (a red pylon is at z = -0.58), but the further away, the more
the few pylon pixels mix with the background -- and red drops
below the threshold first. That was exactly the red/green mix-up at
distance. Found in the bag: a red pylon at azimuth 85 degrees measures
z = -0.134 and was discarded as `unknown`.

**The cast is not the same all round.** Measured over 12 sectors it runs from
+0.046 to +0.121, a span of 0.075 -- half the threshold. The cause is directional
light plus the colour drift of the fisheye towards the edge. A single global number (or
a different Kelvin value) cannot hit that, which is why it is measured **per azimuth sector**.

Over time, on the other hand, it is rock-stable: spread per sector at most 0.006 over
14 seconds of driving. So it is not an exposure problem but a
fixed misadjustment -- and therefore cleanly measurable.

A ring of pixels on the **mat** is sampled, just outside the wall band
(automatically `zone_r0_out + 12` to `radius_px - 15`, 408..446 px on the setup).
Per sector the bright, almost colourless pixels are taken from it and their
mean z is formed -- **mean, not median**: with a median of integers
the result stays an integer.

Why not use the black wall band as a second reference point as well: it measures
B=6 G=13 R=13. With numbers this small a single digit flips the index by
0.077; `z_band` jumped back and forth between 0.000 and -0.077 in the measurement.
The mat at around 200 is the reliable reference.

**The measurement is corrected, not the image.** `z0` is subtracted from `zz`
and correspondingly `z0 * mx` from `dd` (the offset in `G-R` grows with the
brightness, because `z = (G-R)/mx`). `rg_z_min`, `rg_s_min` and `rg_d_min`
thereby keep their meaning and their established tuning, and it costs
no noise -- unlike scaling up the channels.

Checked against `wb_test`, 219 scans with 2 red and 3 green pylons:

| | before | after |
| --- | --- | --- |
| red pylons, stable clusters | 3 | 3 (unchanged) |
| red pylon 2 m, wrongly green points | 13.1 | **5.7** |
| red pylon 2 m, red points | 22.4 | 22.8 |
| green pylon 2.55 m | 11.5 | 12.5 |
| green pylon 0.56 m | 16.1 | 10.6 |

The last row is the price: a near green pylon loses about a
third of its points (but at 10.6 stays well above any cluster threshold).

```bash
ros2 param set /lidar_pixel_mapper white_point true
ros2 param set /lidar_pixel_mapper white_point_sectors 12   # 30 deg per sector
ros2 param set /lidar_pixel_mapper white_point_step 3       # pixel thinning
ros2 param set /lidar_pixel_mapper white_point_r_min 0      # 0 = automatic
ros2 param set /lidar_pixel_mapper white_point_r_max 0
```

The measured value is in the periodic `Sync:` log (`white point z0: ...`).
If it drifts or the span gets large, you see it there first.

Compute time on the Jetson, 1280x960:

| `white_point_step` | pixels in the ring | time per frame | measured z0 |
| --- | --- | --- | --- |
| 1 | 100596 | 18.07 ms | +0.105 |
| **3** (default) | **11196** | **5.18 ms** | **+0.106** |
| 4 | 6300 | 3.84 ms | +0.104 |

The result practically does not depend on the sampling density -- if you need CPU,
you can safely go to 4.

## Why a reference scan is needed

On the S3, cables and electronics cover part of the field of view. There the
scanner measures itself -- a few centimetres -- and those are therefore ALWAYS the
nearest points. A search for the nearest object thus never finds the
pylon.

Blind sectors (``blind``) catch the core of these regions, but not the
edge: there the beam grazes the build and gives e.g. 0.23 m, i.e. above the
threshold. That is why the reference scan (``background``) is the actual
tool -- it records the empty surroundings once, and after that only
what measures CLOSER than this reference counts as a target. Cables, electronics, table edges and
walls all drop out by themselves that way.

## Order of setup

Measure `cam_z` (height of the camera above the lidar plane) once and enter it in the
calibration file -- that is the only value no node can guess.

```bash
ros2 run camera_lidar_fusion rotation_calibration

# 0a. Measure the blocked lidar sectors (send twice: collect, evaluate)
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: blind"
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: blind"

# 0b. Reference scan of the EMPTY surroundings -- remove the pylon! (also twice)
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: background"
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: background"

# 0c. Pin the colour of the calibration pylon -- otherwise the largest
#     colour blob in the room wins instead of the pylon.
ros2 param set /camera_rotation_calibration target_label green

# 1. Measure the image circle automatically
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: circle"

# 2. Measure the rotation: put down ONE red/green block, nothing else in the
#    near range. Sample per position, move the block all round (>= 3 positions).
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: sample"

# 3. Solve, check, save
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: solve"
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: verify"
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: save"

# 3b. Measure the sampling zone -- place the pylon at 5 to 6 distances from 0.3 to 2.5 m
#     and send "zone" at each position. Raise target_range_max_m beforehand,
#     otherwise the node only sees up to 1.5 m.
ros2 param set /camera_rotation_calibration target_range_max_m 3.0
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: zone"
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: zonelist"
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: zonefit"
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: save"
ros2 topic pub --once /camera_lidar/reload std_msgs/msg/Empty "{}"

# 4. Optional: cross-check the camera height (needs near samples, < 0.5 m)
ros2 param set /camera_rotation_calibration target_height_m 0.05
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: height"
```

Check in Foxglove: `/camera_lidar/calib_debug`. The orange X (projected
lidar cluster) must lie on the colour ring (camera blob). More commands:
`list`, `clear`, `reload`, `auto` (collects by itself as soon as the block has been moved far
enough).

Everything can also be adjusted live by hand -- the debug image follows immediately:

```bash
ros2 param set /camera_rotation_calibration yaw_deg 12.5
ros2 topic pub --once /camera_lidar/calib_cmd std_msgs/msg/String "data: save"
```

## Colour per lidar point

```bash
ros2 run camera_lidar_fusion lidar_pixel_mapper
ros2 topic pub --once /camera_lidar/capture std_msgs/msg/Empty "{}"
```

Writes `/workspace/lidar_color_logs/lidar_pixels_<time>.csv` plus the
calibration used as `_calib.yaml` next to it. Columns:

```
stamp_sec, idx, angle_deg, range_m, x_m, y_m, z_m,
u_px, v_px, theta_deg, phi_deg, b, g, r, h, s, v, label
```

`label` is `red`, `green`, `magenta`, `black` or `unknown`.

Which colours are searched for at all is controlled by `active_labels`. The parameter
is read on every scan, so it can be switched while running -- unlike
the thresholds in `color.*`, which are frozen at start-up:

```bash
ros2 param set /lidar_pixel_mapper active_labels "[red,green]"
ros2 param set /lidar_pixel_mapper active_labels "[red,green,magenta]"
```

Magenta easily produces false hits at larger distances and only gets in the way
as long as the parking zone is not needed.
`csv_mode:=continuous` instead appends every scan to one file,
`csv_mode:=off` switches the CSV off completely.

## Debug view in Foxglove

The parameter `debug` (default `true`) is the main switch for the display:

```bash
ros2 param set /lidar_pixel_mapper debug true    # on
ros2 param set /lidar_pixel_mapper debug false   # off, saves CPU in the run
```

If it is on, two topics go out:

* **`/camera_lidar/colored_scan`** -- `PointCloud2` with RGB: every lidar point at
  its real x/y position. What it is coloured with is decided by
  `cloud_color_mode`:

  | Mode | Colour | what for |
  | --- | --- | --- |
  | `label` (default) | strong per label, rest dark grey | finding pylons |
  | `raw` | the measured pixel colour | checking calibration and thresholds |

  `label` uses the palette `CLOUD_BGR` from `colors.py`: red `0xFF0000`, green
  `0x00FF00`, magenta `0xFF00FF`, black `0x2D2D2D`, unknown `0x555555`.
  The values are exact, so a consumer can check for them directly instead of
  guessing colour ranges.

  Why this is needed: measured on the real setup, in `raw` mode
  practically all points lie at R/G/B around 20 to 25 -- 1028 different colour values,
  but all of them dark grey mush in which red and green can hardly be told
  apart. In `label` mode there are 4 unambiguous values.

  `raw` still remains the view in which you see whether the calibration
  is right: if the red points are on the red block, yaw is right.

  In Foxglove open a 3D panel, subscribe to the topic, set the colour mode to `RGB`.
  The frame is the lidar's (for `sllidar` = `laser`).

  ```bash
  ros2 param set /lidar_pixel_mapper cloud_color_mode raw
  ros2 param set /lidar_pixel_mapper cloud_color_mode label
  ```

  The parameter is read again on every scan, so it takes effect immediately -- unlike
  the colour thresholds `color.*`, which are only read at start-up.
* **`/camera_lidar/debug_image`** -- the same the other way round: the fisheye image with
  the projections drawn in, the image circle and an arrow to the front.

Finer control with `publish_cloud`, `publish_debug_image` and `debug_rate_hz`
(default 5 Hz for the image; the PointCloud goes out with every scan).
`/camera_lidar/summary` only counts the labels and always runs.

The calibration node has the same switch for `/camera_lidar/calib_debug`.

### The debug image: round plus unwrapped

`/camera_lidar/debug_image` delivers two views on top of each other. At the top the
round fisheye with image circle, horizon ring, zone limits, the sampled
segments in label colour and the detected wall band edge (magenta). Below it an
**unwrapped strip**: azimuth horizontal, image radius vertical.

The strip is the more useful view. In the round image everything
interesting lies at the outer edge and is squeezed into a few pixels
there; unrolled, the layers sit cleanly on top of each other -- at the top
the room, below it the black wall band, at the very bottom the bright mat. Whether the
sampling zone sits on the wall band or reaches over it, you see there at
a glance, in the round image you do not.

```bash
ros2 param set /lidar_pixel_mapper debug_polar true
ros2 param set /lidar_pixel_mapper debug_polar_height 150
```

The header line names the mode (`zone: wall band live` / `zone: calibrated` /
`zone: ... computed`), the vote fraction, the point count and how many azimuths
the band search has hit.

## Compute time in operation

Measured on the Jetson (instantaneous load via /proc, 2100 points per scan, 15 Hz):

| Configuration | CPU |
| --- | --- |
| classification only | 28 % of a core |
| + wall band detection (`band_detect`, 15 Hz) | 56 % |
| + debug image with polar strip (5 Hz) | 94 % |
| the same with `band_steps: 180` | 84 % |

Two things are remarkable about this. The **debug image is the most expensive item**
at 38 percent, although it only runs at 5 Hz -- drawing and polar unwrapping
on 1280x960 cost. So in the competition run `debug:=false`, which saves the 38
percent immediately. And the **wall band detection costs 28 percent**, because it runs on
every scan; `band_steps: 180` instead of 360 gets 10 percent of that back,
still dense enough for a wall band at 2 degrees between samples.

Careful with the measuring method: `ps -o pcpu` gives the average over the
whole lifetime of the process and is useless for a before/after comparison.
The numbers above come from the difference of `utime + stime` in
`/proc/<pid>/stat` over a fixed interval.

By the way, when optimising, the biggest chunk was not what you would expect:
the median filter of the wall band detection cost **17.7 of
24 ms** as a Python loop; vectorised over a sliding window it is 1.1 ms. The V channel,
on the other hand, stays with `cv2.cvtColor` (3.8 ms) -- `img.max(axis=2)` does give
the same result, but needs 34.9 ms.

## Launch

```bash
ros2 launch camera_lidar_fusion camera_lidar.launch.py
ros2 launch camera_lidar_fusion camera_lidar.launch.py mode:=calib
ros2 launch camera_lidar_fusion camera_lidar.launch.py scan_topic:=/ldlidar_node/scan
```

`scan_topic` is set to `/scan` (what `sllidar_s3_launch.py` publishes). The
older code in `robot_vision` partly still hangs on `/ldlidar_node/scan` --
if in doubt, ask `ros2 topic list`.

## Calibration file

`/workspace/config/fisheye_calib.yaml` is read and written
(parameter `calib_file`). If it does not exist, the default shipped in
`share/camera_lidar_fusion/config/fisheye_calib.yaml` applies.

Besides image circle, pose and blind sectors it holds the four coefficients of the
sampling zone:

```yaml
zone_r0_in:  399.9    # r_inner(rho) = zone_r0_in  + zone_k_in  / rho
zone_k_in:    -2.31
zone_r0_out: 400.1    # r_outer(rho) = zone_r0_out + zone_k_out / rho
zone_k_out:  +13.33
```

If `zone_r0_out` and `zone_k_out` are both 0, the zone counts as not
calibrated and the mapper falls back to the computed heights. The
start-up message says `MEASURED` or `COMPUTED`, so you do not have to guess.

## Tests

The projection model runs without hardware:

```bash
cd /workspace/src/camera_lidar_fusion && python3 -m pytest test/test_fisheye_model.py -q
```

`test/fake_scan.py` publishes a synthetic 360 degree scan on `/scan`,
so that `lidar_pixel_mapper` can also be tested without a running lidar.
