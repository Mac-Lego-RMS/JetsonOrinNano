# The outer track box (fixed 3x3 m rim) in the start-anchored map frame.
# Direction-agnostic: corners in a fixed CCW winding, index 0 = the corner with
# the largest x (furthest ahead in driving direction). The controller uses the
# separate /race_direction topic to decide which way to traverse the indices.
std_msgs/Header header

geometry_msgs/Point[4] corners
WallHNF[4] walls
float64 edge_length