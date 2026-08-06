# Orbital Refactor

This project restructures the original `orbital/` research code into reusable
filters, adapters, pipelines, interfaces, and cooperative-estimation tools.
The refactor preserves the numerical behavior of the original EKF, covariance
intersection (CI), NIS gating, and experiment workflows wherever regression
compatibility is required.

The original scripts are retained in `legacy/` as numerical-regression
references. They are not the primary application interface.

## Main interfaces

Run a state-awareness task through the stable module interface:
 
```python
output = StateAwarenessModule().run(module_input)
```

Run a complete observation history with:

```python
output = StateAwarenessModule().run_history(module_input)
```

Select the single-satellite fusion architecture through the filter
configuration:

```python
config["filter"]["architecture"] = "centralized"    # Centralized EKF
# or
config["filter"]["architecture"] = "federated_ci"   # Federated EKF + CI (default)
```

Run the reusable multi-node cooperative pipeline with:

```python
from cooperative.multi_sat_pipeline import run_cooperative_pipeline
```

Export a run to CSV, NPZ, and JSON with:

```python
from exporters import export_run_bundle

paths = export_run_bundle(
    history,
    module_output,
    output_directory="outputs/run_001",
    stem="centralized_result",
)
```

## Data flow

Precomputed neural-network predictions can be used without checkpoint files:

```text
SHIRT JSON + predictions.npz
        -> data-source adapters
        -> ModuleInput / Observation
        -> StateAwarenessModule
        -> ModuleOutput
```

Checkpoint files are required only by the separate neural-network inference
and training tools.

## Examples

Run examples from the project root:

```bash
python examples/run_standard_interface.py
python examples/run_multi_sat_scenario.py
python examples/run_multi_sat_interface.py
python examples/run_v13_1_baseline.py
python examples/run_v13_2_fleet_ci.py
python examples/run_v13_4_attitude_coupling.py
```

Run the v13.4 attitude-coupling Monte Carlo validation with:

```bash
python examples/run_v13_4_attitude_monte_carlo.py --seeds 30
```

Validate the independent MEKF covariance before tuning orbit/attitude
interface weights:

```bash
python examples/run_v13_4_attitude_consistency_monte_carlo.py --seeds 30
```

Run distributed Fleet-State CI validation across packet-loss and delay grids:

```bash
python examples/run_v13_4_distributed_monte_carlo.py --seeds 30 \
  --packet-loss-rates 0.0 0.2 --delays 0.0 4.0
```

Run the configurable noise and process Monte Carlo sweep with:

```bash
python examples/run_parameter_sweep.py --seeds 5 \
  --modes combined --angle-sigmas-deg 1.0 2.0 \
  --process-noises 1e-7
```

The sweep exports one raw CSV row per seed and algorithm, plus an aggregated
`*_summary.csv` containing means and standard deviations.

## Tests

Run the complete test suite with:

```bash
python -m pytest
```

---

# Version Change Log

## v13.4 - Attitude-aware BODY angle fusion

### Added

- Added BODY-frame inter-satellite azimuth/elevation prediction using explicit
  source-satellite `wxyz` inertial-to-body quaternions.
- Added BODY-angle Jacobians with respect to the orbital states and the MEKF
  left-multiplicative small-angle attitude error.
- Added propagation of attitude uncertainty into the angle covariance through
  `R_effective = R_sensor + J_attitude P_attitude J_attitude^T`.
- Connected epoch- and satellite-matched `AttitudeEstimate` objects to the
  centralized Fleet EKF and distributed Fleet-State CI paths.
- Added a reproducible comparison of truth attitude, MEKF attitude with
  covariance, and MEKF attitude without covariance.
- Added multi-seed Monte Carlo export with per-run metrics, aggregate
  statistics, percentiles, failure rates, and paired covariance-propagation
  win rates.
- Added distributed Fleet-State CI Monte Carlo validation across configurable
  packet-loss and communication-delay grids while keeping attitude as an
  external or independently estimated auxiliary input.
- Added Pre-CI and Post-CI state/covariance histories, per-epoch CI gain
  diagnostics, and a local-only control mode for separating measurement-update
  behavior from communication-fusion effects.
- Added independent attitude NEES, gyro NIS, and star-tracker NIS Monte Carlo
  diagnostics so MEKF covariance calibration can be assessed before any
  orbit/attitude interface scaling is introduced.

### Validation

- The 120-second comparison produces a mean MEKF attitude error of about
  `0.051 deg`.
- Truth-attitude BODY angles produce a mean two-dimensional angle NIS near the
  expected value of `2`.
- Propagating MEKF attitude covariance prevents the strongly overconfident NIS
  observed when the same attitude estimate is used with zero attitude
  covariance.

## v13.3 - Attitude MEKF migration

### Added

- Added `orbital_core/attitude.py` with explicit `wxyz` quaternion operations,
  rigid-body angular dynamics, RK4 angular-rate integration, and attitude-error
  utilities migrated from the original thesis implementation.
- Added `orbital_core/attitude_filter.py` with a nine-dimensional
  multiplicative error state:

  ```text
  [attitude error, angular-velocity error, gyro-bias error]
  ```

- Added gyro, star-tracker, and attitude-estimate interface objects.
- Added reproducible attitude truth, gyro white noise, gyro-bias random walk,
  and star-tracker measurement simulation.
- Added rigid-body angular-rate propagation, star-tracker quaternion updates,
  body-vector updates, NIS output, and Joseph covariance updates.
- Added quaternion-convention, sensor-reproducibility, MEKF-convergence,
  gyro-bias, and covariance-positive-semidefinite tests.

### Changed

- Extended the original six-dimensional attitude and angular-rate MEKF with an
  explicit three-dimensional gyro-bias state.
- Documented attitude estimation as an independent auxiliary subsystem. It is
  not yet coupled to the orbital angle-measurement update; a future integration
  can provide the estimated inertial-to-body DCM and attitude covariance.

### Validation

- Attitude migration tests: `3 passed`.
- Regression suite excluding two sandbox-incompatible temporary-directory test
  files: `72 passed`.

## v13.2 - Fleet-State CI and statistical validation

### Added

- Added a distributed validation path in which every satellite maintains a
  local copy of the same `6N` fleet state.
- Added validated `FleetStateMessage` exchange between nodes.
- Restricted CI to messages with identical state dimensions and satellite
  ordering.
- Added optional absolute ECI position anchors to remove the global drift mode
  of a relative-only constellation.
- Added symmetric fleet scenarios, a centralized `6N` EKF, distributed
  Fleet-State CI, RTN angle measurements, RMSE/NIS/NEES evaluation, and
  configurable Monte Carlo parameter sweeps.
- Added reproducible Gaussian white noise for inter-satellite range,
  range-rate, RTN azimuth/elevation, and low-rate GNSS position anchors.
- Matched generated covariance matrices to the configured measurement-noise
  standard deviations.

### Fixed

- Included neighbor-state covariance in distributed inter-satellite updates.
- Rejected direct CI between absolute states of different physical satellites
  in the primary fleet-filter path.

## v13.1 - Symmetric three-satellite baseline

### Added

- Added `scenarios/fleet_scenario.py` for symmetric N-satellite truth
  scenarios.
- Added `orbital_core/fleet_centralized_ekf.py` for the centralized `6N` state.
- Added `pipelines/fleet_centralized.py` for centralized inter-satellite EKF
  execution.
- Added per-satellite absolute orbit propagation, range/range-rate observation
  updates, fully connected topology support, and three-satellite acceptance
  examples.
- Included neighbor covariance in distributed inter-satellite updates through
  `H_j P_j H_j^T`.
- Rejected direct CI between different satellites' absolute states.

## v12.4 - Cooperative experiment framework

### Added

- Added a unified cooperative experiment entry script.
- Added JSON-based cooperative experiment configuration.
- Added cooperative result-summary export.

### Purpose

- Improved reproducibility and prepared delay, dropout, and packet-loss
  comparative experiments.

## v12.3 - Age-aware CI weighting

### Added

- Added information-age-based covariance inflation.
- Added optional age-aware CI weighting for asynchronous fusion.

### Changed

- Extended cooperative pipeline interfaces with:
  - `age_aware`
  - `age_penalty`

### Purpose

- Reduced the influence of older asynchronous information during CI fusion.

## v12.2 - Orbital-dynamics time alignment

### Added

- Replaced constant-velocity delayed-state alignment with two-body + J2 RK4
  propagation.
- Added an independent message-buffer lifecycle for each fusion run.

### Fixed

- Prevented communication-buffer state from leaking between repeated
  experiments.

## v12.1 - Asynchronous time-alignment fusion

### Added

- Added `cooperative/time_alignment.py`.
- Added delayed-state alignment before CI fusion.
- Added a time-propagation interface for future orbital-dynamics-based
  compensation.

### Changed

- Updated the asynchronous fusion pipeline so that:
  - `MessageBuffer` releases delayed reports;
  - released reports are propagated to the current fusion epoch;
  - aligned reports are used for CI fusion.
- Kept example entry scripts on the project-root path-insertion pattern.

### Notes

- This version used a constant-velocity transition model as the initial
  time-alignment baseline.

## v12 - Asynchronous fusion base

### Added

- Added message-buffer-based delayed communication processing.
- Added arrival-timestamp-based report release.

## v11.1 - Communication logging extension

### Added

- Added delay-configuration output to `run_multi_sat_delay.py`.
- Added received-node-history inspection and recording.
- Added source and arrival timestamp output for delay debugging.

### Notes

- This version did not yet include time-alignment compensation.

## v10 - Communication extension

### Added

- Added `cooperative/communication_channel.py`.
- Added stochastic packet-loss simulation.
- Extended the CI pipeline with `communication_channel`.
- Added `received_node_history`.
- Added `run_multi_sat_packet_loss.py`.

### Notes

- Node dropout remained supported through `CommunicationConfig`.

## v9 - Preliminary communication changes

### Existing in v8

- `NodeReport`.
- `active_node_history`.
- `validity_history_by_node`.

### Added

- Added `cooperative/communication.py`.
- Added `examples/run_multi_sat_node_dropout.py`.

### Notes

- The communication layer supported deterministic dropout.
- Packet loss, delay buffering, and asynchronous fusion were identified as
  subsequent extensions.

## v8 - Multi-satellite cooperative estimation

### Added

- Added an end-to-end path from generated orbital truth through local filtering
  to epoch-wise multi-node covariance intersection.
- Added absolute two-body + J2 orbit propagation.
- Added Keplerian-elements-to-ECI initialization.
- Added per-observer RTN/PRI frame histories.
- Added target and observer absolute and relative trajectory generation.
- Added synthetic optical, infrared, radar, and NN-surrogate observations.
- Added `StateAwarenessModule.run_history()` while preserving the stable
  `run()` API.
- Added multi-node history execution helpers.
- Added conversion from node-relative estimates to a common target absolute ECI
  state.
- Added simultaneous covariance-intersection fusion for one to three nodes.
- Added complete communication-outage handling by holding the last cooperative
  posterior after initialization.
- Added `node_validity_by_node` support for node and link failure simulation.

### Example workflow

The complete multi-satellite interface example:

1. Propagates one target and three observer satellites with the two-body + J2
   model.
2. Generates heterogeneous infrared, radar, and synthetic learning-enhanced
   observations.
3. Builds an independent `ModuleInput` and single-node filter for each
   observer.
4. Converts every local target-relative estimate into a target absolute ECI
   state.
5. Fuses available node reports by covariance intersection at each epoch.
6. Reports local and cooperative position/velocity RMSE and final node weights.

## v6 - Centralized fusion and unified result export

### Added

- Added a centralized multimodal EKF pipeline while preserving the external
  `ModuleInput` and `ModuleOutput` interfaces.
- Added `orbital_core/centralized_filter.py` with centralized prediction,
  per-modality NIS prechecks, and joint measurement updates.
- Added `pipelines/centralized.py` with the centralized execution loop,
  abnormal-event handling, and standard output packaging.
- Added `exporters/result_exporter.py` for unified CSV, NPZ, and JSON export.
- Extended `adapters/module_input_adapter.py` with centralized input adaptation
  through the existing standard interface.
- Added support for traditional optical, infrared, and radar observations, as
  well as learning-enhanced optical observations.

### Preserved behavior

- Preserved the legacy fixed finite-difference step, pseudoinverse, Joseph
  covariance update, and NIS soft/hard gating behavior.

### Output

- `centralized_result.csv`: per-epoch position, velocity, acceleration, NIS,
  gating, and CI weights when applicable.
- `centralized_result.npz`: complete state, covariance, and diagnostic history.
- `centralized_result.json`: standard `ModuleOutput` for upstream systems and
  visualization.

### Validation

- Regression and interface tests: `20 passed`.
- Centralized trajectories and covariances matched the legacy scripts, with
  only floating-point differences of approximately `1e-8` to `1e-7`.

## v5 - SHIRT and NN-prediction data adapters

### Added

- Added a data-source layer that converts legacy experiment files into the
  documented interface objects without changing the filter or CI
  implementation.
- Added `adapters/shirt_data_adapter.py` to:
  - load SHIRT `metadata.json` and `roe*.json`;
  - preserve the verified legacy quaternion and coordinate convention;
  - construct runtime auxiliary histories and a standard `ModuleInput`.
- Added `adapters/nn_prediction_adapter.py` to:
  - read `predictions.npz` through the existing `image_path` / `t_pred`
    contract;
  - align predictions with SHIRT filenames;
  - optionally build pseudo velocity;
  - emit learning-enhanced optical `Observation` objects.
- Added `adapters/synthetic_measurement_adapter.py` to:
  - create infrared azimuth/elevation and radar range/range-rate observations;
  - support explicit dropout windows;
  - emit standard traditional `Observation` objects.

### Validation

- Regression and adapter tests: `18 passed`.

## v4 - Standard input adaptation and learning-enhanced observations

### Added

- Added `adapters/module_input_adapter.py`.
- Added the unified `interfaces/state_awareness_module.py` entry point.
- Added automatic mapping of
  `Observation(modality="OPTICAL", source_type="LEARNING")` to `nn`.
- Added three-dimensional position observations in ECI and SPRI frames.
- Added six-dimensional position and pseudo-velocity observations in ECI and
  SPRI frames.
- Compared NN EKF numerical behavior with the original
  `federated_ci3_nn_ir_rad_fusion_ekf.py`.

### Implementation notes

- The adapter converts documented `ModuleInput` and `Observation` objects into
  arrays and local filters used by the federated CI pipeline.
- Primary-satellite state history and `q_eci2pri` history remain auxiliary
  model data in `config.runtime` to minimize algorithm changes.
- A modality uses a fixed measurement covariance within one task to maintain
  compatibility with the legacy implementation.
- Time-varying measurement covariance and confidence-based dynamic scaling were
  deferred as optional features that would not change default behavior.

## v3 - Federated CI pipeline and standard output

### Added

- Added `pipelines/federated_ci.py`.
- Organized traditional optical, infrared, radar, and other local EKFs into a
  unified federated pipeline.
- Preserved valid-posterior selection for CI, holding the previous fused result
  when every modality is missing, and optional feedback behavior.
- Added local states, fused states, covariance, NIS, gating flags, CI weights,
  and statistics to pipeline output.
- Converted modality loss, hard-gate rejection, and soft-gate down-weighting
  into `AbnormalEvent` objects.
- Added direct generation of `LocalEstimate`, `SingleFusionResult`, and
  `ModuleOutput`.

### Validation

- Regression tests directly called
  `legacy/federated_ci_dynamics_fusion_ekf.py` and compared fused states, fused
  covariances, and every local-filter result.

## v2 - Shared filters and CI utilities

### Added

- Added `orbital_core/filters.py` to unify single-modality and federated local
  orbital EKFs.
- Added `orbital_core/ci_fusion.py` to unify two-way and three-way covariance
  intersection and return named weights.
- Added `orbital_core/quality.py` with non-intrusive quality-scoring helpers.
- Added `pipelines/single_modal.py` to extract the single-modality execution
  loop and return history plus a standard `LocalEstimate`.
- Added regression tests that load the legacy `DynamicsEKF` directly and verify
  identical prediction and update results.

### Preserved behavior

- Preserved the fixed-step numerical Jacobian, pseudoinverse, Joseph covariance
  update, and NIS soft/hard gating behavior.
- Kept legacy scripts unchanged.

### Validation

- Test suite at this stage: `8 passed`.

## v1 - Initial engineering refactor

### Added

- Preserved the original scripts under `legacy/` as regression references.
- Extracted Earth constants, coordinate transformations, orbital dynamics,
  measurement models, and metric calculations.
- Added interface objects including `ModuleInput`, `Observation`,
  `LocalEstimate`, `SingleFusionResult`, `NodeReport`, and `ModuleOutput`.
- Added foundational unit tests.

### Preserved behavior

- Did not change the numerical logic of the original EKF, CI, NIS, or experiment
  workflows.
