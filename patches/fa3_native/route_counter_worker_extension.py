from __future__ import annotations


class SparseRouteCounterWorkerExtension:
    """Minimal named RPC surface for cold route-counter boundaries."""

    def sfi_reset_sparse_route_counters_for_measurement(
        self,
    ) -> dict[str, object]:
        from patches.fa3_native.install import (
            reset_rank_local_route_counter_slot_for_measurement,
        )

        return reset_rank_local_route_counter_slot_for_measurement()

    def sfi_snapshot_sparse_route_counters_after_measurement(
        self,
    ) -> dict[str, object]:
        from patches.fa3_native.install import (
            snapshot_rank_local_route_counter_slot_after_measurement,
        )

        return snapshot_rank_local_route_counter_slot_after_measurement()
