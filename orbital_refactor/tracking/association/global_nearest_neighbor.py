from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment

from orbital_core.dynamics import numerical_jacobian_discrete, rk4_step_absolute
from tracking.data_contracts import GlobalTargetEstimate, TargetInitialState
from tracking.initial_orbit.data_contracts import IODObservation

from .data_contracts import AssociationMatch, AssociationResult, UnlabeledIODObservation


def associate_to_tracks(
    observations,
    tracks: Mapping[str, GlobalTargetEstimate | TargetInitialState],
    *,
    gate_squared_mahalanobis: float = 13.815510557964274,
    minimum_cost_margin: float = 2.0,
    prediction_position_sigma_floor_m: float = 50.0,
    prediction_velocity_sigma_floor_mps: float = 0.25,
) -> AssociationResult:
    """Joint multimodal GNN association with conservative ambiguity rejection."""

    values = tuple(observations)
    if gate_squared_mahalanobis <= 0.0:
        raise ValueError("gate_squared_mahalanobis must be positive.")
    if minimum_cost_margin < 0.0:
        raise ValueError("minimum_cost_margin must be non-negative.")
    if prediction_position_sigma_floor_m < 0.0:
        raise ValueError("prediction_position_sigma_floor_m must be non-negative.")
    if prediction_velocity_sigma_floor_mps < 0.0:
        raise ValueError("prediction_velocity_sigma_floor_mps must be non-negative.")
    if any(not isinstance(item, UnlabeledIODObservation) for item in values):
        raise TypeError("associate_to_tracks requires UnlabeledIODObservation values.")
    channels = defaultdict(list)
    for index, item in enumerate(values):
        if item.valid_flag:
            channels[(item.timestamp, item.observer_id)].append(index)

    matches = []
    observed_targets = set()
    assigned_indices = set()
    ambiguous_indices = set()
    target_items = tuple(sorted(tracks.items()))
    prediction_cache = {}
    for indices in channels.values():
        if not target_items:
            continue
        timestamp = values[indices[0]].timestamp
        predicted_target_items = []
        for target_id, estimate in target_items:
            key = (timestamp, target_id)
            if key not in prediction_cache:
                prediction_cache[key] = _predict_estimate(
                    estimate,
                    timestamp=timestamp,
                    prediction_position_sigma_floor_m=(
                        prediction_position_sigma_floor_m
                    ),
                    prediction_velocity_sigma_floor_mps=(
                        prediction_velocity_sigma_floor_mps
                    ),
                )
            predicted_target_items.append((target_id, prediction_cache[key]))
        predicted_target_items = tuple(predicted_target_items)
        grouped = defaultdict(list)
        ungrouped = defaultdict(list)
        for index in indices:
            item = values[index]
            if item.detection_group_id is None:
                ungrouped[item.modality].append([index])
            else:
                grouped[item.detection_group_id].append(index)
        unit_sets = []
        if grouped:
            unit_sets.append(list(grouped.values()))
        unit_sets.extend(ungrouped.values())
        for units in unit_sets:
            unit_matches, unit_ambiguous = _assign_units(
                units,
                predicted_target_items,
                values,
                gate_squared_mahalanobis=gate_squared_mahalanobis,
                minimum_cost_margin=minimum_cost_margin,
            )
            for index, target_id, distance in unit_matches:
                matches.append(AssociationMatch(index, target_id, distance))
                assigned_indices.add(index)
                observed_targets.add(target_id)
            ambiguous_indices.update(unit_ambiguous)
    return AssociationResult(
        matches=tuple(sorted(matches, key=lambda item: item.observation_index)),
        unassigned_observation_indices=tuple(
            index for index in range(len(values)) if index not in assigned_indices
        ),
        ambiguous_observation_indices=tuple(sorted(ambiguous_indices)),
        unobserved_target_ids=tuple(sorted(set(tracks) - observed_targets)),
    )


def _assign_units(
    units,
    target_items,
    values,
    *,
    gate_squared_mahalanobis,
    minimum_cost_margin,
):
    costs = np.full((len(units), len(target_items)), np.inf)
    individual_costs = {}
    geometry_cache = {}
    for row, indices in enumerate(units):
        for column, (_, prediction) in enumerate(target_items):
            distances = []
            for index in indices:
                observation = values[index]
                geometry_key = (
                    column,
                    observation.observer_id,
                    "RADAR" if observation.modality == "RADAR" else "LOS",
                )
                if prediction[0] is None:
                    geometry = None
                elif geometry_key not in geometry_cache:
                    geometry_cache[geometry_key] = _measurement_geometry(
                        observation, prediction[0],
                    )
                    geometry = geometry_cache[geometry_key]
                else:
                    geometry = geometry_cache[geometry_key]
                distances.append(_innovation_distance_from_prediction(
                    observation,
                    *prediction,
                    geometry=geometry,
                    # Unit cost is the mean across its modalities.  A single
                    # term can therefore reach N times the mean gate without
                    # making the whole unit impossible.
                    gate_squared_mahalanobis=(
                        gate_squared_mahalanobis * len(indices)
                    ),
                ))
            distances = tuple(distances)
            individual_costs[(row, column)] = distances
            if all(np.isfinite(value) for value in distances):
                costs[row, column] = float(np.mean(distances))
    safe = np.where(
        np.isfinite(costs), costs, gate_squared_mahalanobis + 1e9,
    )
    rows, columns = linear_sum_assignment(safe)
    matches = []
    ambiguous = set()
    for row, column in zip(rows, columns):
        distance = float(costs[row, column])
        if not np.isfinite(distance) or distance > gate_squared_mahalanobis:
            continue
        alternatives = np.delete(costs[row], column)
        finite_alternatives = alternatives[np.isfinite(alternatives)]
        margin = (
            float("inf")
            if finite_alternatives.size == 0
            else float(np.min(finite_alternatives) - distance)
        )
        if margin < minimum_cost_margin:
            ambiguous.update(units[row])
            continue
        target_id = target_items[column][0]
        for index, individual_distance in zip(
            units[row], individual_costs[(row, column)],
        ):
            matches.append((index, target_id, float(individual_distance)))
    return matches, ambiguous


def label_matches(observations, result: AssociationResult) -> tuple[IODObservation, ...]:
    values = tuple(observations)
    labeled = []
    for match in result.matches:
        item = values[match.observation_index]
        labeled.append(IODObservation(
            timestamp=item.timestamp,
            observer_id=item.observer_id,
            target_id=match.target_id,
            modality=item.modality,
            observer_state_eci=item.observer_state_eci,
            measurement=item.measurement,
            covariance=item.covariance,
            valid_flag=item.valid_flag,
            metadata={
                **item.metadata,
                "associationSquaredMahalanobis": match.squared_mahalanobis,
                "associationMethod": "GNN",
            },
        ))
    return tuple(labeled)


def _innovation_distance(
    observation,
    estimate,
    *,
    prediction_position_sigma_floor_m=0.0,
    prediction_velocity_sigma_floor_mps=0.0,
):
    prediction = _predict_estimate(
        estimate,
        timestamp=observation.timestamp,
        prediction_position_sigma_floor_m=prediction_position_sigma_floor_m,
        prediction_velocity_sigma_floor_mps=prediction_velocity_sigma_floor_mps,
    )
    return _innovation_distance_from_prediction(observation, *prediction)


def _predict_estimate(
    estimate,
    *,
    timestamp,
    prediction_position_sigma_floor_m,
    prediction_velocity_sigma_floor_mps,
):
    dt = float(timestamp - estimate.timestamp)
    if dt < 0.0:
        return None, None
    state = np.asarray(estimate.state_eci, dtype=float)
    covariance = np.asarray(estimate.covariance_eci, dtype=float)
    if dt > 0.0:
        propagate = lambda value: rk4_step_absolute(value, dt)
        transition = numerical_jacobian_discrete(propagate, state)
        state = propagate(state)
        covariance = transition @ covariance @ transition.T
    covariance = covariance + np.diag(
        [float(prediction_position_sigma_floor_m) ** 2] * 3
        + [float(prediction_velocity_sigma_floor_mps) ** 2] * 3
    )
    return state, covariance


def _innovation_distance_from_prediction(
    observation,
    state,
    covariance,
    *,
    geometry=None,
    gate_squared_mahalanobis=None,
):
    if state is None or covariance is None:
        return np.inf
    predicted, measured, jacobian = _linearized_measurement(
        observation, state, geometry=geometry,
    )
    if not (
        np.all(np.isfinite(predicted))
        and np.all(np.isfinite(measured))
        and np.all(np.isfinite(jacobian))
    ):
        return np.inf
    residual = measured - predicted
    projected_covariance = jacobian @ covariance @ jacobian.T
    if gate_squared_mahalanobis is not None:
        # For positive-definite S, r.T @ inv(S) @ r >= r_i**2 / S_ii.
        # Therefore this inexpensive component-wise test can only reject a
        # candidate that the full Mahalanobis gate would reject as well.
        marginal_variance = np.diag(projected_covariance) + np.diag(
            observation.covariance
        )
        if (
            np.all(np.isfinite(marginal_variance))
            and np.all(marginal_variance > 0.0)
            and np.any(
                residual * residual
                > float(gate_squared_mahalanobis) * marginal_variance
            )
        ):
            return np.inf
    innovation_covariance = projected_covariance + observation.covariance
    innovation_covariance = 0.5 * (innovation_covariance + innovation_covariance.T)
    try:
        return float(residual @ np.linalg.solve(innovation_covariance, residual))
    except np.linalg.LinAlgError:
        return np.inf


def _linearized_measurement(observation, state, *, geometry=None):
    """Return association-space measurement terms with analytic Jacobians.

    Association evaluates every observation against every candidate track, so
    finite-difference Jacobians dominate runtime at swarm scale.  Both current
    measurement models have compact closed forms and do not need repeated
    perturb-and-evaluate calls.
    """

    if geometry is None:
        geometry = _measurement_geometry(observation, state)
    predicted, jacobian, projection = geometry
    if observation.modality == "RADAR":
        measured = np.asarray(observation.measurement, dtype=float)
    else:
        measured = projection @ np.asarray(observation.measurement, dtype=float)
    return predicted, measured, jacobian


def _measurement_geometry(observation, state):
    observer = np.asarray(observation.observer_state_eci, dtype=float)
    reference = np.asarray(state, dtype=float)
    relative_position = reference[:3] - observer[:3]
    distance = float(np.linalg.norm(relative_position))
    if not np.isfinite(distance) or distance <= np.finfo(float).eps:
        invalid = np.full(2, np.nan)
        return invalid, np.full((2, 6), np.nan), np.full((2, 3), np.nan)
    line = relative_position / distance

    if observation.modality == "RADAR":
        relative_velocity = reference[3:] - observer[3:]
        range_rate = float(line @ relative_velocity)
        jacobian = np.zeros((2, 6), dtype=float)
        jacobian[0, :3] = line
        jacobian[1, :3] = (
            relative_velocity - line * range_rate
        ) / distance
        jacobian[1, 3:] = line
        return np.array([distance, range_rate]), jacobian, None

    tangent_one, tangent_two = _tangent_basis(line)
    jacobian = np.zeros((2, 6), dtype=float)
    jacobian[0, :3] = tangent_one / distance
    jacobian[1, :3] = tangent_two / distance
    projection = np.vstack((tangent_one, tangent_two))
    return np.zeros(2, dtype=float), jacobian, projection


def _measurement_function(observation, reference_state):
    observer = observation.observer_state_eci
    if observation.modality == "RADAR":
        def radar(state):
            relative_position = state[:3] - observer[:3]
            relative_velocity = state[3:] - observer[3:]
            distance = np.linalg.norm(relative_position)
            return np.array([distance, relative_position @ relative_velocity / distance])
        return radar
    predicted_line = reference_state[:3] - observer[:3]
    predicted_line /= np.linalg.norm(predicted_line)
    tangent_one, tangent_two = _tangent_basis(predicted_line)

    def los(state):
        line = state[:3] - observer[:3]
        line /= np.linalg.norm(line)
        return np.array([line @ tangent_one, line @ tangent_two])
    return los


def _association_measurement(observation, reference_state):
    if observation.modality == "RADAR":
        return observation.measurement
    predicted_line = reference_state[:3] - observation.observer_state_eci[:3]
    predicted_line /= np.linalg.norm(predicted_line)
    tangent_one, tangent_two = _tangent_basis(predicted_line)
    return np.array([
        observation.measurement @ tangent_one,
        observation.measurement @ tangent_two,
    ])


def _tangent_basis(line):
    x, y, z = np.asarray(line, dtype=float)
    if abs(float(z)) <= 0.9:
        scale = float(np.hypot(x, y))
        first = np.array([y / scale, -x / scale, 0.0])
    else:
        scale = float(np.hypot(x, z))
        first = np.array([-z / scale, 0.0, x / scale])
    second = np.array([
        y * first[2] - z * first[1],
        z * first[0] - x * first[2],
        x * first[1] - y * first[0],
    ])
    return first, second


def _numerical_jacobian(function, state):
    state = np.asarray(state, dtype=float)
    baseline = function(state)
    result = np.empty((baseline.size, state.size))
    for axis in range(state.size):
        step = max(1e-4, abs(state[axis]) * 1e-7)
        delta = np.zeros_like(state)
        delta[axis] = step
        result[:, axis] = (function(state + delta) - function(state - delta)) / (2.0 * step)
    return result

