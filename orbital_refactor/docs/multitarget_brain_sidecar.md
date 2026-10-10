# Multi-target brain-inspired sidecar

The multi-target brain-inspired representation is intentionally kept outside
the probabilistic estimator.  It consumes the target-level J2000 posterior
history after cooperative identity resolution and does not feed synthetic
measurements back into the filter.

`build_multitarget_brain_sidecar` keys every history by persistent cooperative
target ID.  Observer-local target names are not used.  Consequently an
observer joining or leaving the network does not reset, exchange, or duplicate
the brain-inspired state of a physical target.

For each cooperative target the sidecar currently provides:

- an absolute orbital-phase direction ring with 180 neurons;
- a decoded radial/along-track offset relative to propagation of the first
  posterior (no truth trajectory is required);
- a normalized 3 x 3 radial/along-track place-cell heatmap;
- decoded phase, decoded RT coordinates, and validity flags.

The direction-ring zero is the orbit's ascending node.  It is not the target's
initial position, so targets at different absolute orbital phases produce
different direction-ring activity.

The dynamic observer membership and brain-sidecar acceptance can be run with:

```powershell
python -m experiments.run_multitarget_dynamic_membership_acceptance
```

This acceptance checks that global target identities remain unchanged while
observer A leaves, observer C joins, and observer B later leaves.  It also
checks that all target-keyed brain states remain valid, that the direction
ring has 180 neurons, and that each 3 x 3 heatmap is normalized.  UDP
serialization and real-time transport are deliberately outside this stage.
