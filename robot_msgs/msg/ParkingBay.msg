# The parking lot, as the two bounding walls measured on the track.
#
# Segments, not lines: the walls are short (about 20 cm) and it is their ENDS
# that define the bay, so an infinite HNF line would lose exactly the
# information the manoeuvre needs. No target pose either -- where inside the
# bay to stop is a planning decision, and the controller also needs the real
# wall positions for collision checks while swinging in.

std_msgs/Header header             # frame_id = 'map' (start-anchored)

bool detected                      # false: no bay found (fields undefined)

geometry_msgs/Point wall_a_start   # segment A, both endpoints; x/y used, z=0
geometry_msgs/Point wall_a_end
geometry_msgs/Point wall_b_start   # segment B
geometry_msgs/Point wall_b_end

int32 wall_idx                     # straight it sits on, indexed as in
                                   # corner_geometry; -1 if not yet known