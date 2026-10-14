"""Load/unload plates to/from dishwasher."""
from abc import ABC

import numpy as np
import quaternion
from pyquaternion import Quaternion

from bigym.bigym_env import BiGymEnv
from bigym.const import PRESETS_PATH
from bigym.envs.props.holders import DishDrainer
from bigym.envs.props.dishwasher import Dishwasher
from bigym.envs.props.cabintets import WallCabinet
from bigym.envs.props.kitchenware import Plate
from bigym.utils.env_utils import get_random_sites


class _DishwasherPlatesEnv(BiGymEnv, ABC):
    """Base plates environment."""

    RESET_ROBOT_POS = np.array([0, -0.6, 0])

    _PRESET_PATH = PRESETS_PATH / "counter_dishwasher.yaml"

    _RACK_POSITION = np.array([0.6, -0.6, 0.86])
    _RACK_BOUNDS = np.array([0.05, 0.05, 0])

    _PLATES_COUNT = 2
    _PLATE_ROTATION = quaternion.from_euler_angles(np.pi / 2, np.pi / 2, 0)
    _PLATE_OFFSET_POS = np.array([0, -0.01, 0.05])

    _SITES_STEP = 2
    _SITES_SLICE = 4

    _TOLERANCE = np.deg2rad(20)

    def _initialize_env(self):
        self.dishwasher = self._preset.get_props(Dishwasher)[0]
        self.drainer: DishDrainer = DishDrainer(self._mojo)
        self.plates: list[Plate] = []
        for _ in range(self._PLATES_COUNT):
            self.plates.append(Plate(self._mojo))

    def _fail(self) -> bool:
        if super()._fail():
            return True
        for plate in self.plates:
            if plate.is_colliding(self.floor):
                return True
        return False

    def _on_reset(self):
        self.dishwasher.set_state(door=1, bottom_tray=1, middle_tray=0)
        offset = np.random.uniform(-self._RACK_BOUNDS, self._RACK_BOUNDS)
        self.drainer.body.set_position(self._RACK_POSITION + offset)


class DishwasherUnloadPlates(_DishwasherPlatesEnv):
    """Unload plates from dishwasher task."""

    def _success(self) -> bool:
        for plate in self.plates:
            if not plate.is_colliding(self.drainer):
                return False
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(plate, side):
                    return False
        return True

    def _on_reset(self):
        super()._on_reset()
        sites = self.dishwasher.tray_bottom.site_sets[0]
        sites = get_random_sites(
            sites, len(self.plates), self._SITES_STEP, self._SITES_SLICE
        )
        for plate, site in zip(self.plates, sites):
            plate.body.set_quaternion(quaternion.as_float_array(self._PLATE_ROTATION))
            plate.body.set_position(site.get_position() + self._PLATE_OFFSET_POS)


class DishwasherUnloadPlatesLong(DishwasherUnloadPlates):
    """Unload plate from dishwasher in wall cabinet task."""

    _PRESET_PATH = PRESETS_PATH / "counter_dishwasher_wall_cabinet.yaml"

    _PLATES_COUNT = 1
    _RACK_POSITION = np.array([0.8, -0.6, 1.474])
    _RACK_BOUNDS = np.array([0.01, 0.05, 0])

    _TOLERANCE = 0.1

    def _initialize_env(self):
        super()._initialize_env()
        self.wall_cabinet = self._preset.get_props(WallCabinet)[0]

    def _success(self) -> bool:
        if not np.allclose(self.dishwasher.get_state(), 0, atol=self._TOLERANCE):
            return False
        if not np.allclose(self.wall_cabinet.get_state(), 0, atol=self._TOLERANCE):
            return False
        return super()._success()


class DishwasherLoadPlates(_DishwasherPlatesEnv):
    """Load plates to dishwasher task."""

    def _success(self) -> bool:
        up = np.array([0, 0, 1])
        right = np.array([0, -1, 0])
        for plate in self.plates:
            plate_up = Quaternion(plate.body.get_quaternion()).rotate(up)
            angle_to_right = np.arccos(np.clip(np.dot(plate_up, right), -1.0, 1.0))
            angle_to_left = np.arccos(np.clip(np.dot(plate_up, -right), -1.0, 1.0))
            if not (
                angle_to_right <= self._TOLERANCE or angle_to_left <= self._TOLERANCE
            ):
                return False
            if not plate.is_colliding(self.dishwasher.tray_bottom.colliders):
                return False
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(plate, side):
                    return False
        return True

    def _on_reset(self):
        super()._on_reset()
        sites = self.drainer.sites
        sites = get_random_sites(
            sites, len(self.plates), self._SITES_STEP, self._SITES_SLICE
        )
        for plate, site in zip(self.plates, sites):
            plate.body.set_quaternion(quaternion.as_float_array(self._PLATE_ROTATION))
            plate.body.set_position(site.get_position() + self._PLATE_OFFSET_POS)
