"""Interval-equivalent local velocity from holonomic actuator encoder increments.

No world/object poses are inputs. Absolute encoder offsets cancel. The midpoint
frame matches docking_controller.Odometry's existing integration convention.
"""
import numpy as np


def interval_base_velocity(before, after, dt):
    before, after = np.asarray(before, float), np.asarray(after, float)
    if before.shape != (6,) or after.shape != (6,) or not np.isfinite([before, after]).all():
        raise ValueError('expected finite six-base-joint encoder positions')
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('encoder interval must be positive')
    yaw_delta = (after[5] - before[5] + np.pi) % (2*np.pi) - np.pi
    yaw_mid = before[5] + .5*yaw_delta
    dx, dy = after[:2] - before[:2]
    c, s = np.cos(yaw_mid), np.sin(yaw_mid)
    return np.array([(c*dx+s*dy)/dt, (-s*dx+c*dy)/dt, yaw_delta/dt])


def replace_base_velocity(obs, robot_name, velocity):
    """Copy the proprio field rather than modifying the simulator's cached sensor tensor."""
    velocity = np.asarray(velocity, float)
    if velocity.shape != (3,) or not np.isfinite(velocity).all():
        raise ValueError('invalid interval velocity')
    result = dict(obs)
    if robot_name in result and isinstance(result[robot_name], dict) and 'proprio' in result[robot_name]:
        target = result[robot_name] = dict(result[robot_name])
        key = 'proprio'
    else:
        target, key = result, f'{robot_name}::proprio'
    value = target[key]
    if tuple(value.shape) != (61,):
        raise ValueError('interval adapter requires the R1Pro 61D proprio contract')
    if hasattr(value, 'clone'):
        value = value.clone()
        value[:3] = value.new_tensor(velocity)
    else:
        value = np.array(value, copy=True)
        value[:3] = velocity
    target[key] = value
    return result
