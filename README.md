# JetsonOrinNano — WRO Repository

## Aufbau

- `src/` — ROS 2 Workspace. Wird auf dem Jetson nach `~/ros2_ws/src` ausgecheckt.
- Weitere Ordner koennen daneben liegen (Doku, Modelle, Bilder, Videos, ...).
  Diese werden auf dem Jetson bewusst **nicht** ausgecheckt.

## Jetson

Das Repo liegt dort in `~/ros2_ws` mit Sparse Checkout auf `src/`,
`git pull` holt also nur den Inhalt von `src/`.
