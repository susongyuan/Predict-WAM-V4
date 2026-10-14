"""Causal commanded-closure/stillness clocks, not physical object-hold evidence."""
import numpy as np


def _finite_array(value, name, shape):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in 'fiu':
        raise ValueError(f'{name} must be a finite real array of shape {shape}')
    with np.errstate(over='ignore', invalid='ignore'):
        array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must contain only finite values')
    return array


class EventClock:
    """Float32 features, left then right: [closed tau, still tau, closed flag].

    Tau is seconds / max_seconds, capped at 1. State order is [L q7, L grip,
    R q7, R grip]. Event entry has tau=0; stillness needs a measured interval.
    Measured grippers may overshoot [0, 1]; no state or command is clipped.
    """

    def __init__(self, dt=.1, still_velocity_rad_s=.02, max_seconds=10.,
                 close_threshold=.5):
        params = _finite_array(
            [dt, still_velocity_rad_s, max_seconds, close_threshold],
            'clock parameters', (4,))
        if (params[0] <= 0 or params[1] < 0 or params[2] <= 0
                or not 0 <= params[3] <= 1):
            raise ValueError('require dt/max_seconds > 0, still_velocity_rad_s >= 0, '
                             'and close_threshold in [0, 1]')
        self.dt, self.still_velocity_rad_s, self.max_seconds, self.close_threshold = params
        self.reset()

    def reset(self):
        self._previous_q = None
        self._flags = np.zeros((2, 2), dtype=bool)
        self._seconds = np.zeros((2, 2), dtype=np.float64)

    def update(self, measured_state16, previous_target16=None):
        """Return (6,) float32; pass the last issued command, never a future action.

        Only t0 may omit previous_target16: q0 supplies its closure flags, with
        all times zero. Later calls must explicitly supply the previous command.
        """
        state = _finite_array(measured_state16, 'measured_state16', (16,))
        if previous_target16 is None:
            if self._previous_q is not None:
                raise ValueError('previous_target16 (last issued command) is required after t0')
            target = state
        else:
            target = _finite_array(previous_target16, 'previous_target16', (16,))

        q = state.reshape(2, 8)[:, :7]
        closed = target[[7, 15]] < self.close_threshold
        still = np.zeros(2, dtype=bool)
        with np.errstate(over='ignore'):
            if self._previous_q is not None:
                velocity = np.abs(q - self._previous_q) / self.dt
                still = (velocity <= self.still_velocity_rad_s).all(axis=1)
            elapsed = np.minimum(self._seconds + self.dt, self.max_seconds)
        flags = np.column_stack((closed, still))
        self._seconds = np.where(flags & self._flags, elapsed, 0.)
        self._flags = flags
        self._previous_q = q.copy()
        return np.column_stack((self._seconds / self.max_seconds, closed)).astype(
            np.float32).reshape(6)


def episode_event_features(state, action, *, initial_target=None, **clockkwargs):
    """Return (N, 6) float32; no mutations or file access.

    At t0 use initial_target (q0 only if None); later use action[t-1]. Each call
    starts all times at zero, including when initial_target seeds a data slice.
    """
    state = np.asarray(state)
    if state.ndim != 2 or state.shape[1] != 16:
        raise ValueError('state must have shape (N, 16)')
    state = _finite_array(state, 'state', state.shape)
    action = _finite_array(action, 'action', state.shape)
    if initial_target is not None:
        initial_target = _finite_array(initial_target, 'initial_target', (16,))
    clock = EventClock(**clockkwargs)
    features = np.empty((len(state), 6), dtype=np.float32)
    for t in range(len(state)):
        features[t] = clock.update(state[t], action[t - 1] if t else initial_target)
    return features
