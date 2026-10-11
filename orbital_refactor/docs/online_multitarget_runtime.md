# Online multi-target estimation runtime

`OnlineMultiTargetRuntime` is the target-free online path.  It is separate
from prepared trajectory replay and processes exactly one sensor epoch at a
time.

Each input frame uses `MULTITARGET-ONLINE-1.0` semantics:

- current J2000 observer states and SPRI attitudes;
- unlabeled radar, infrared, or optical detections;
- a local `detectionGroupId` connecting modalities from the same detection;
- no physical target ID and no externally supplied target initial state.

The runtime executes the following sequence:

1. parse and validate the frame and one-second (or configured) clock interval;
2. associate detections with existing tracks;
3. buffer unassigned multimodal detections for autonomous IOD;
4. hand successful J2000 IOD states and covariances to continuous tracking;
5. run local multimodal filtering, maneuver detection, lifecycle management,
   duplicate suppression, and cooperative fusion;
6. update persistent target-keyed direction-ring and RT place-cell states;
7. produce one `TRAJECTORY_FRAME` and one `DISPLAY_TELEMETRY` message.

The trajectory state remains:

```text
x, y, z, vx, vy, vz
```

Before IOD succeeds, `satelliteList` and `cannBySatellite` are empty.  The
runtime does not publish a guessed target state.  After a target is born its
numeric `satId` remains stable for the lifetime of the runtime.

Input timestamps may be relative seconds or an absolute continuous time. The
first received timestamp is aligned with `start_time_ms`; later frames use the
elapsed source time, so an absolute timestamp is not added twice.

The tracker accepts an optional measurement-preprocessor factory. One
processor is retained for every `(observer_id, target_id)` stream. A processor
may replace only measurement-domain values and covariance; stream length,
observer, target, modality, and timestamp are checked before filtering. This
adapts the original CANN measurement front end without granting it ownership of
the target state, covariance, or CI result.

`OnlineMultiTargetRuntime.publish` sends both messages to the same configured
UDP destination.  Receivers distinguish them using `messageType`.

`OnlineMultiTargetRuntimeEndpoint` adds the following non-conflicting route to
the existing scene-control server:

```text
POST /api/external/scene/measurement-frame
```

Frames are accepted only while the scene clock is `RUNNING`. Exact retries of
the last frame are idempotent and do not duplicate UDP output; reusing the same
`frameIndex` with different content returns a conflict. Existing activation,
run-state, speed, stop, status, and prepared-replay behavior is unchanged.
The HTTP acknowledgement reports `algorithmProcessingMs`, `udpPublicationMs`,
and `withinSampleInterval` so deployment tests can verify the one-second budget
on the actual target computer and scenario size.

The deployable entry point is:

```text
python -m examples.run_online_multitarget_service \
  --udp-host 127.0.0.1 --udp-port 5010 \
  --scene-id demo --run-id run-001 --time-base-id TB-001 \
  --start-time 2026-09-18T04:00:00Z \
  --end-time 2026-09-18T04:20:00Z
```

It listens on control port 8089 by default, requires activation, waits for
`START`, then accepts online measurement frames through the route above. After
`STOP` or natural completion, the process stays alive, resets both the scene
controller and all target/IOD/CANN runtime state, and waits for the next
activation.

Activation uses the existing structured orbital-element scene format. It
validates and converts the supplied satellite initial conditions and checks the
scene/time base. For the current online contract, however, each measurement
frame remains authoritative for observer J2000 state and SPRI attitude; this is
explicitly reported as `observerStateSource=ONLINE_MEASUREMENT_FRAMES` in the
activation response. Target states are never accepted from activation and must
still be established by IOD.
