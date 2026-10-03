# Multi-target online frame protocol

Protocol version: `MULTITARGET-ONLINE-1.0`.

The algorithm consumes one frame at a time. Inputs contain observer states and
target-free detections; they never contain a physical `targetId`. The tracker
creates and maintains its own target and track identifiers.

## Input frame

```json
{
  "frameIndex": 0,
  "timestamp": 0.0,
  "observers": [
    {
      "observerId": "observer-1",
      "stateEci": [7000000.0, 0.0, 0.0, 0.0, 7500.0, 0.0],
      "qEci2PriWxyz": [1.0, 0.0, 0.0, 0.0]
    }
  ],
  "detections": [
    {
      "messageId": "radar-0001",
      "observerId": "observer-1",
      "modality": "RADAR",
      "detectionGroupId": "blob-0001",
      "measurement": [120000.0, -2.5],
      "covariance": [[100.0, 0.0], [0.0, 0.0025]],
      "confidence": 1.0,
      "valid": true
    },
    {
      "messageId": "los-0001",
      "observerId": "observer-1",
      "modality": "LOS",
      "detectionGroupId": "blob-0001",
      "measurement": [0.98, 0.18, 0.08],
      "covariance": [[4e-10, 0.0], [0.0, 4e-10]],
      "metadata": {"sourceModality": "INFRARED"}
    }
  ]
}
```

`stateEci` is `[x, y, z, vx, vy, vz]` in J2000 SI units. `RADAR` measurement
is `[range_m, range_rate_mps]`. `LOS` is a normalized J2000 direction. The same
observer uses `detectionGroupId` to state that its radar and LOS detections at
one epoch belong to the same local blob; it is not a target identity and need
not remain stable across epochs.

Raw `OPTICAL` and `INFRARED` two-component measurements are also accepted when
their metadata contains `quaternion_i2b_wxyz`.

## Output frame

Each output has type `MULTITARGET_ESTIMATION_FRAME`. Its `targets` array contains
the internally assigned `targetId`, stable `trackId`, lifecycle, J2000
`x/y/z/vx/vy/vz`, covariance diagonal, and contributing observer IDs. Association
counts and newly initialized target IDs are included for runtime diagnostics.

## Local replay

```powershell
python -m experiments.run_online_multitarget_replay `
  --input path\to\input.json `
  --output results\online_replay.json
```
