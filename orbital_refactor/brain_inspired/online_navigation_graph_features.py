from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brain_inspired.hierarchical_navigation_place_cells import (
    HierarchicalNavigationPlaceCellEncoder,
)
from brain_inspired.navigation_shadow_quality import NavigationShadowQualityConfig
from brain_inspired.orbital_direction_state import OrbitalDirectionState
from brain_inspired.orbital_phase_adapter import OrbitalPlaneFrame
from brain_inspired.orbital_radial_state import OrbitalRadialState
from brain_inspired.orbital_rt_grid_state import OrbitalRTGridConfig, OrbitalRTGridState
from orbital_core.dynamics import rk4_step_absolute


@dataclass(frozen=True)
class OnlineNavigationGraphFeatureConfig:
    anchor_interval_epochs: int = 2
    rt_config: OrbitalRTGridConfig = OrbitalRTGridConfig()
    quality_config: NavigationShadowQualityConfig = NavigationShadowQualityConfig()

    def validate(self):
        if self.anchor_interval_epochs < 1:
            raise ValueError("anchor_interval_epochs must be positive.")
        self.rt_config.validate()
        self.quality_config.validate()


class OnlineNavigationGraphFeatureProvider:
    """Persistent truth-free CANN sidecars for topology-decision features."""

    def __init__(self, *, initial_state_by_node, initial_timestamp=0.0,
                 config=None):
        self.config = config or OnlineNavigationGraphFeatureConfig()
        self.config.validate()
        states = {
            str(node): np.asarray(state, dtype=float).copy()
            for node, state in initial_state_by_node.items()
        }
        if (not states or any(state.shape != (6,) or np.any(~np.isfinite(state))
                              for state in states.values())):
            raise ValueError("Initial states must be finite six-vectors.")
        self._frames = {
            node: OrbitalPlaneFrame.from_state_eci(state)
            for node, state in states.items()
        }
        self._direction = {
            node: OrbitalDirectionState(node_id=node, frame=self._frames[node])
            for node in states
        }
        self._radial = {
            node: OrbitalRadialState(node_id=node, frame=self._frames[node])
            for node in states
        }
        self._rt = {
            node: OrbitalRTGridState(node_id=node, config=self.config.rt_config)
            for node in states
        }
        self._place = {
            node: HierarchicalNavigationPlaceCellEncoder() for node in states
        }
        self._reference = states
        self._last_posterior = None
        self._last_timestamp = float(initial_timestamp)
        self._epoch_count = 0
        self._previous_quality = {node: 1.0 for node in states}

    def update(self, *, timestamp, state_by_node, covariance_by_node,
               anchor_trusted_by_node=None):
        states = {str(node): np.asarray(value, dtype=float)
                  for node, value in state_by_node.items()}
        covariance = {str(node): np.asarray(value, dtype=float)
                      for node, value in covariance_by_node.items()}
        if set(states) != set(self._direction) or set(covariance) != set(states):
            raise ValueError("Online CANN features require identical node sets.")
        trusted = anchor_trusted_by_node or {}
        if self._last_posterior is None:
            result = self._initialize(float(timestamp), states)
        else:
            result = self._advance(
                float(timestamp), states, covariance, trusted,
            )
        self._last_posterior = {
            node: state.copy() for node, state in states.items()
        }
        self._last_timestamp = float(timestamp)
        self._epoch_count += 1
        return result

    def _initialize(self, timestamp, states):
        metrics = {}
        for node, state in states.items():
            direction = self._direction[node].initialize_from_state(
                timestamp=timestamp, state_eci=state,
            )
            radial = self._radial[node].initialize_from_state(
                timestamp=timestamp, state_eci=state,
            )
            rt = self._rt[node].initialize(
                timestamp=timestamp, state_eci=state,
                reference_state_eci=self._reference[node],
            )
            place = self._place[node].encode(
                phase=direction.decoded_phase,
                radial_position=rt.decoded_rt[0],
                along_track_position=rt.decoded_rt[1],
            )
            metrics[node] = self._metrics(
                direction=direction, radial=radial, rt=rt, place=place,
                quality=1.0, delayed_quality=1.0,
            )
        return metrics

    def _advance(self, timestamp, states, covariance, trusted):
        dt = timestamp - self._last_timestamp
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("Online CANN timestamps must be strictly increasing.")
        anchor_epoch = self._epoch_count % self.config.anchor_interval_epochs == 0
        metrics = {}
        for node, state in states.items():
            reference = rk4_step_absolute(self._reference[node], dt)
            prior = rk4_step_absolute(self._last_posterior[node], dt)
            confidence = _anchor_confidence(covariance[node])
            use_anchor = bool(anchor_epoch and trusted.get(node, True))
            direction_prediction = self._direction[node].predict_from_state(
                timestamp=timestamp, predicted_state_eci=prior,
            )
            direction = self._direction[node].anchor_from_state(
                timestamp=timestamp, posterior_state_eci=state,
                confidence=confidence, trusted=use_anchor,
            )
            radial_prediction = self._radial[node].predict_from_state(
                timestamp=timestamp, predicted_state_eci=prior,
            )
            radial = self._radial[node].anchor_from_state(
                timestamp=timestamp, posterior_state_eci=state,
                confidence=confidence, trusted=use_anchor,
            )
            self._rt[node].predict(
                timestamp=timestamp, predicted_state_eci=prior,
                reference_state_eci=reference,
            )
            rt = self._rt[node].anchor(
                timestamp=timestamp, posterior_state_eci=state,
                reference_state_eci=reference, confidence=confidence,
                trusted=use_anchor,
            )
            quality = self._quality(
                direction_prediction, radial_prediction, direction, radial,
            )
            delayed = self._previous_quality[node]
            self._previous_quality[node] = quality
            place = self._place[node].encode(
                phase=direction.decoded_phase,
                radial_position=rt.decoded_rt[0],
                along_track_position=rt.decoded_rt[1],
            )
            metrics[node] = self._metrics(
                direction=direction, radial=radial, rt=rt, place=place,
                quality=quality, delayed_quality=delayed,
            )
            self._reference[node] = reference
        return metrics

    def _quality(self, direction_prediction, radial_prediction, direction,
                 radial):
        settings = self.config.quality_config
        d_direction = abs(direction_prediction.phase_residual) / (
            settings.direction_residual_scale
        )
        d_radial = abs(radial_prediction.displacement_residual) / (
            settings.radial_residual_scale
        )
        consistency = np.exp(-0.5 * (d_direction ** 2 + d_radial ** 2))
        freshness = np.exp(
            -max(direction.anchor_age, radial.anchor_age)
            / settings.anchor_age_scale
        )
        quality = float(np.clip(
            consistency * freshness, settings.minimum_quality, 1.0,
        ))
        if (not direction.valid or not radial.valid
                or radial.saturated_at_boundary):
            quality = settings.minimum_quality
        return quality

    @staticmethod
    def _metrics(*, direction, radial, rt, place, quality, delayed_quality):
        return {
            "cann_direction_concentration": direction.bump_concentration,
            "cann_abs_direction_residual_rad": abs(direction.phase_residual),
            "cann_rt_residual_norm_m": float(np.linalg.norm(rt.residual_rt)),
            "cann_anchor_age_s": max(direction.anchor_age, rt.anchor_age),
            "cann_shadow_quality": quality,
            "cann_delayed_feedback_quality": delayed_quality,
            "cann_boundary_saturated": float(
                radial.saturated_at_boundary or rt.saturated_at_boundary
            ),
            "cann_anchor_rejected": float(rt.anchor_rejected),
            "cann_reference_rebased": float(np.any(rt.reference_rebased)),
            "place_fine_scale_active": float(place.fine_scale_active),
            "place_boundary_saturated": float(place.boundary_saturated),
            "place_scale_transition": float(place.scale_transition),
        }


def _anchor_confidence(covariance):
    if covariance.shape != (6, 6) or np.any(~np.isfinite(covariance)):
        raise ValueError("Online CANN covariance must be finite 6x6.")
    sigma = np.sqrt(max(0.0, float(np.trace(covariance[:3, :3]) / 3.0)))
    return float(np.clip(10.0 / (10.0 + sigma), 0.0, 1.0))
