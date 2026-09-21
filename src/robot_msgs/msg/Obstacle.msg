# One traffic sign (pillar) on the track.

uint8 COLOR_UNKNOWN=0
uint8 COLOR_RED=1
uint8 COLOR_GREEN=2

int32 id                      # seat index 0..23, stable across updates
geometry_msgs/Point position  # map frame; x/y used, z=0. CENTRE of the pillar
                              # (44 mm edge length -- add the extent yourself)
uint8 color                   # see COLOR_* above; UNKNOWN = seen but colour
                              # not yet certain -> treat as an obstacle anyway
int32 wall_idx                # straight it belongs to, same indexing as
                              # corner_geometry; -1 if not assignable