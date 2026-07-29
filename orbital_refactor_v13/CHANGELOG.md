# Orbital Refactor Change Log

## v13.3 - Attitude MEKF migration

### Added
- Added `orbital_core/attitude.py` with explicit `wxyz` quaternion operations,
  rigid-body angular dynamics, RK4 angular-rate integration, and attitude-error
  utilities migrated from the original thesis implementation.
- Added `orbital_core/attitude_filter.py` with a nine-dimensional MEKF error
  state: attitude error, angular-velocity error, and gyro-bias error.
- Added gyro, star-tracker, and attitude-estimate interface objects.
- Added reproducible attitude truth, gyro white-noise/bias-random-walk, and
  star-tracker measurement simulation.
- Added quaternion-convention, sensor reproducibility, MEKF convergence,
  gyro-bias, and covariance-positive-semidefinite tests.

### Changed
- Extended the original six-dimensional attitude/angular-rate MEKF with an
  explicit three-dimensional gyro-bias state.
- Documented the attitude subsystem as independent auxiliary estimation; it is
  not yet coupled into the orbital angle-measurement update.

### Validation
- Attitude migration tests: `3 passed`.
- Regression suite excluding two sandbox-incompatible temporary-directory test
  files: `72 passed`.

## v13.2 - Fleet-State CI and statistical validation

### Added
- Added symmetric fleet scenarios, absolute position anchors, centralized `6N`
  EKF, distributed Fleet-State CI, RTN angle measurements, RMSE/NIS/NEES
  evaluation, and configurable Monte Carlo parameter sweeps.
- Added real Gaussian noise sampling for inter-satellite measurements and GNSS
  anchors with covariance matched to the generation model.

### Fixed
- Included neighbor-state covariance in distributed inter-satellite updates.
- Rejected direct CI between absolute states of different physical satellites
  in the primary fleet-filter path.

## v13.1 - Symmetric three-satellite baseline

### Added
- Added per-satellite absolute orbit propagation, range/range-rate observation
  updates, fully connected topology support, and three-satellite acceptance
  examples.

## v12.1 - Asynchronous Time Alignment Fusion

### Added
- Added `cooperative/time_alignment.py`.
- Added delayed state alignment before CI fusion.
- Added a time propagation interface for future orbital dynamics based compensation.

### Modified
- Updated asynchronous fusion pipeline:
  - delayed reports are released by `MessageBuffer`;
  - released reports are propagated to current fusion epoch;
  - aligned reports are used for CI fusion.
- Kept example entry scripts using the project-root path insertion pattern.

### Notes
- Current time alignment uses a constant-velocity transition model as a baseline.
- Future versions can replace it with the full orbital dynamics/J2 propagation model.

## v12 - Asynchronous Fusion Base

### Added
- Message buffer based delayed communication processing.
- Arrival timestamp based report release.

## v11.1 - Delay Logging

### Added
- Delay configuration output.
- Source and arrival timestamp logging.
- Received node history recording.


## v12.2 - Orbital dynamics time alignment

### Added
- Replaced constant-velocity delayed-state alignment with two-body + J2 RK4 propagation.
- Added independent message buffer lifecycle per fusion run.

### Fixed
- Avoided communication buffer state leaking between repeated experiments.


## v12.3 - Age-aware CI weighting

### Added
- Added information-age based covariance inflation.
- Added optional age-aware CI weighting for asynchronous fusion.

### Modified
- Extended cooperative pipeline interfaces with:
  - `age_aware`
  - `age_penalty`

### Purpose
Reduce the influence of older asynchronous information during CI fusion.


## v12.4 - Cooperative experiment framework

### Added
- Added unified cooperative experiment entry script.
- Added JSON based cooperative experiment configuration.
- Added cooperative result summary export.

### Purpose
- Improve reproducibility and prepare delay/dropout/packet-loss comparative experiments.
