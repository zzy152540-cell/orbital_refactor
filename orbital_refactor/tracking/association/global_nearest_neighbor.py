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
) -> AssociationResult:
    """Joint multimodal GNN association with conservative ambiguity rejection."""

    values = tuple(observations)
    if gate_squared_mahalanobis <= 0.0:
        raise ValueError("gate_squared_mahalanobis must be positive.")
    if minimum_cost_margin < 0.0:
        raise ValueError("minimum_cost_margin must be non-negative.")
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
    for indices in channels.values():
        if not target_items:
            continue
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
                target_items,
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
    for row, indices in enumerate(units):
        for column, (_, estimate) in enumerate(target_items):
            distances = tuple(
                _innovation_distance(values[index], estimate) for index in indices
            )
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


def _innovation_distance(observation, estimate):
    dt = float(observation.timestamp - estimate.timestamp)
    if dt < 0.0:
        return np.inf
    state = np.asarray(estimate.state_eci, dtype=float)
    covariance = np.asarray(estimate.covariance_eci, dtype=float)
    if dt > 0.0:
        propagate = lambda value: rk4_step_absolute(value, dt)
        transition = numerical_jacobian_discrete(propagate, state)
        state = propagate(state)
        covariance = transition @ covariance @ transition.T
    measurement_function = _measurement_function(observation, state)
    predicted = measurement_function(state)
    measured = _association_measurement(observation, state)
    jacobian = _numerical_jacobian(measurement_function, state)
    innovation_covariance = jacobian @ covariance @ jacobian.T + observation.covariance
    innovation_covariance = 0.5 * (innovation_covariance + innovation_covariance.T)
    residual = measured - predicted
    try:
        return float(residual @ np.linalg.solve(innovation_covariance, residual))
    except np.linalg.LinAlgError:
        return np.inf


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
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(line @ reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    first = np.cross(line, reference)
    first /= np.linalg.norm(first)
    return first, np.cross(line, first)


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

