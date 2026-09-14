"""Shared adapters for feeding pre-generated cases into online filters."""

from cooperative.network_schmidt_orchestrator import TransportSourceUpdate


def source_updates_from_messages(messages, node_ids):
    updates = {}
    for message in messages:
        source = str(message.source_node_id)
        for event in message.transport_events:
            key = (source, float(event.timestamp))
            updates.setdefault(key, TransportSourceUpdate(
                state=event.state_estimate,
                error_transition=(
                    message.error_transition
                    if event.source_error_transition is None
                    else event.source_error_transition
                ),
                independent_process_noise=(
                    message.accumulated_process_noise
                    if event.source_process_noise is None
                    else event.source_process_noise
                ),
                information_ids=event.information_ids,
                event_error_transition=event.error_transition,
                event_process_noise=event.independent_process_noise,
            ))
    missing_sources = set(node_ids) - {source for source, _ in updates}
    if missing_sources:
        raise RuntimeError("Source updates are unavailable for some nodes.")
    return updates


def items_by_timestamp(items):
    result = {}
    for item in items:
        result.setdefault(float(item.timestamp), []).append(item)
    return result
