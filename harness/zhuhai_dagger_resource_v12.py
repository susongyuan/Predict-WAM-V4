"""Do not confuse old swapped desktop pages with missing live RAM headroom."""
from harness.zhuhai_dagger_parallel import GIB, can_expand
from harness.zhuhai_dagger_resource_v10 import ResourceAdmission as PreviousAdmission


class ResourceAdmission(PreviousAdmission):
    def block_reasons(self, row, active, maximum=4):
        if row is None:
            return ['resource_snapshot_pending']
        floor = max(2*GIB, .10*row['total'])
        reasons = []
        if self.clock() < self.cooldown_until:
            reasons.append('resource_cooldown_60_seconds')
        if active >= maximum:
            reasons.append('worker_limit_or_pilot_gate')
        reserve = (8.5 if active else 7)*GIB + floor
        if row['available'] < reserve:
            reasons.append('cold_start_RAM_reserve')
        if row['gpu_free_bytes'] < 6*GIB:
            reasons.append('GPU_free_below_6GiB')
        if row['swap_fraction'] >= (.85 if active >= 3 else .92):
            reasons.append('fourth_worker_swap_85_percent' if active >= 3 else 'swap_92_percent')
        return reasons

    def pressure(self):
        # The user rejected a prolonged single-worker fallback. Let existing
        # safe workers continue; briefly cool new admission, then fill capacity
        # again under the unchanged hard guards and cold-start RAM reserve.
        self.cooldown_until = self.clock() + 60.
        self.serial_until = 0.

    def allowed(self, row, active, maximum=4):
        now = self.clock()
        if now < self.cooldown_until or not can_expand(row, active, maximum=maximum):
            return False
        if not active:
            return row['swap_fraction'] < .92
        if now < self.serial_until or row['swap_fraction'] >= (.85 if active >= 3 else .92):
            return False
        # The unchanged runtime guards still exit at 95% swap or the RAM floor.
        # Second/third admission keeps a further 8.5 GiB for environment + WAM;
        # old swapped desktop pages alone no longer reduce collection to one.
        floor = max(2*GIB, .10*row['total'])
        return row['available'] >= 8.5*GIB + floor
