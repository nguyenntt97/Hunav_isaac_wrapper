#!/usr/bin/env python3
"""
robots/drivers.py

One interface over the two ways this wrapper can move a robot.

HuNavManager reads the robot through ``get_world_pose()``,
``get_linear_velocity()`` and ``get_angular_velocity()`` and nothing else, all
duck-typed, so a driver can stand in for the ``WheeledRobot`` it used to be
handed without HuNavManager changing at all.

The split that matters is *when* control is applied:

  * A differential drive is a velocity target. Writing it once per rendered
    frame (20 Hz) is what the wrapper has always done and is plenty.
  * A learned locomotion policy is a closed loop around joint positions. It has
    to run every PhysX step, at the rate it was trained at, or the robot falls
    over. That is why ``on_physics_step`` exists separately from
    ``on_render_step``.
"""

import numpy as np

from .specs import GO2_MAX_ANGULAR, GO2_MAX_LINEAR


def _to_numpy(array):
    """Coerce a warp array, torch tensor or sequence to numpy.

    Isaac Sim 6.0's experimental Articulation returns warp arrays, which do not
    support Python item indexing -- and HuNavManager indexes and float()s
    everything it reads, so the conversion has to happen here rather than
    surfacing as an abort deep inside rosidl.
    """
    if isinstance(array, np.ndarray):
        return array
    if hasattr(array, "detach"):  # torch.Tensor
        return array.detach().cpu().numpy()
    if hasattr(array, "numpy"):  # warp.array
        return array.numpy()
    return np.asarray(array)


def _clamp(value, limit):
    return max(-limit, min(limit, float(value)))


class WheeledDriver:
    """Differential drive: the wrapper's original robot path, unchanged.

    ``lin_y`` is accepted and ignored; a differential base cannot strafe, and
    Nav2's controllers never ask it to.
    """

    def __init__(self, world, spec, usd_path, position):
        from isaacsim.robot.wheeled_robots.controllers.differential_controller import (
            DifferentialController,
        )
        from isaacsim.robot.wheeled_robots.robots import WheeledRobot

        self.spec = spec
        self.prim_path = f"/World/{spec.prim_name}"
        self.robot = world.scene.add(
            WheeledRobot(
                prim_path=self.prim_path,
                name="Robot",
                wheel_dof_names=list(spec.wheel_dof_names),
                create_robot=True,
                usd_path=usd_path,
                position=position,
                orientation=list(spec.spawn_orientation),
            )
        )
        self.diff_controller = DifferentialController(
            name="diff_drive_controller",
            wheel_radius=spec.wheel_radius,
            wheel_base=spec.wheel_base,
        )
        self._cmd_lin_x = 0.0
        self._cmd_ang_z = 0.0

    def set_command(self, lin_x, lin_y, ang_z):
        self._cmd_lin_x = float(lin_x)
        self._cmd_ang_z = float(ang_z)

    def on_physics_step(self, dt):
        pass

    def on_render_step(self):
        self.robot.apply_wheel_actions(
            self.diff_controller.forward([self._cmd_lin_x, self._cmd_ang_z])
        )

    def get_world_pose(self):
        return self.robot.get_world_pose()

    def get_linear_velocity(self):
        return self.robot.get_linear_velocity()

    def get_angular_velocity(self):
        return self.robot.get_angular_velocity()

    def joint_state(self):
        return None


def _deinstance(prim_path):
    """Make the robot's link geometry non-instanceable.

    The Isaac Lab Go2 authors every link's ``visuals`` and ``collisions`` as an
    instanceable prim, so the geometry lives in a shared prototype. Prototypes
    are shared, which means a prototype cannot carry a per-instance transform --
    and the articulation's per-link poses never reach them. The trunk still
    draws in the right place because it is the articulation root, whose
    transform is authored on the prim itself, so the robot renders as a body
    with its legs left behind at the authored bind pose.

    That is invisible in a world where the robot spawns at the origin (the legs
    are stranded exactly where they belong) and obvious anywhere else: in
    brownstone, which spawns at (4, -43), the legs sit 43 m away at the origin.

    De-instancing costs a little memory for one robot and makes each link's
    geometry a real prim that follows its own transform.
    """
    import omni.usd
    from pxr import Usd

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        return 0
    count = 0
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.IsInstance():
            prim.SetInstanceable(False)
            count += 1
    if count:
        print(f"[hunav] de-instanced {count} prims under {prim_path} so each "
              "link's geometry follows its own transform", flush=True)
    return count


class Go2PolicyDriver:
    """Unitree Go2, driven by the flat-terrain policy Isaac Sim ships.

    ``isaacsim.robot.policy.examples.robots.Go2FlatTerrainPolicy`` wraps a
    TorchScript network that maps a 48-element observation to 12 joint position
    targets, and takes its command as (v_x, v_y, w_z) -- exactly a Twist. Unlike
    the go2_omniverse submodule, which owns an entire Isaac Lab environment and
    step loop, this references itself into whatever stage is already open.
    """

    def __init__(self, world, spec, usd_path, position):
        from isaacsim.robot.policy.examples.robots import Go2FlatTerrainPolicy

        self.spec = spec
        self.prim_path = f"/World/{spec.prim_name}"
        # usd_path=None lets the policy class pick its own default asset; we
        # pass the path from asset_paths so there is one place to fix when the
        # next Isaac release moves it.
        self.policy = Go2FlatTerrainPolicy(
            prim_path=self.prim_path,
            usd_path=usd_path,
            position=list(position),
            orientation=list(spec.spawn_orientation),
            policy_path=spec.policy_path,
            env_config_path=spec.env_config_path,
        )
        _deinstance(self.prim_path)
        self._physics_ready = False
        self._command = None
        self._cmd = (0.0, 0.0, 0.0)
        self._explicit = spec.actuation == "explicit"
        self._target = None
        self._gains = None

    def _init_explicit_actuation(self):
        """Zero PhysX's PD drives and take over torque generation.

        ``PolicyController.initialize`` normally hands the position targets to
        PhysX's implicit joint drives. Isaac Lab, where the policy was trained,
        does something different: its DCMotor actuator computes the torque in
        Python every physics step and applies it as an effort, with a
        velocity-dependent clamp that a plain symmetric effort limit does not
        reproduce. This makes inference use the same actuator as training.
        """
        import torch
        from isaacsim.robot.policy.examples.controllers.config_loader import (
            get_robot_joint_properties,
        )

        params = self.policy.policy_env_params
        robot = self.policy.robot
        effort_limit, velocity_limit, stiffness, damping, _, _, _ = get_robot_joint_properties(
            params, robot.dof_names
        )
        device = torch.device(str(robot._device))
        t = lambda v: torch.tensor(v, dtype=torch.float32, device=device)

        # saturation_effort is the DC motor's stall torque; it is not one of the
        # values get_robot_joint_properties returns, so read it off the config.
        saturation = dict()
        for actuator in params["scene"]["robot"]["actuators"].values():
            value = actuator.get("saturation_effort", actuator.get("effort_limit"))
            for expr in actuator.get("joint_names_expr", []):
                saturation[expr] = float(value)
        # One actuator group covers every leg joint on the Go2, so the single
        # value is unambiguous; fall back to the effort limit otherwise.
        sat = float(next(iter(saturation.values()))) if saturation else None

        self._gains = {
            "kp": t(stiffness),
            "kd": t(damping),
            "effort_limit": t(effort_limit),
            "velocity_limit": t(velocity_limit),
            "saturation": t([sat] * len(robot.dof_names)) if sat else t(effort_limit),
        }
        robot.switch_dof_control_mode("effort")
        print(
            f"[hunav] Go2 explicit actuation: kp={stiffness[0]} kd={damping[0]} "
            f"effort_limit={effort_limit[0]} saturation={sat}",
            flush=True,
        )

    def _apply_explicit_torque(self):
        """One DCMotor step: PD, then the four-quadrant torque-speed clamp."""
        import torch
        import warp as wp

        if self._target is None:
            return
        robot = self.policy.robot
        g = self._gains
        q = wp.to_torch(robot.get_dof_positions()).reshape(-1)
        qd = wp.to_torch(robot.get_dof_velocities()).reshape(-1)
        torque = g["kp"] * (self._target - q) - g["kd"] * qd

        # isaaclab.actuators.actuator_pd.DCMotor._clip_effort
        max_effort = torch.clamp(
            g["saturation"] * (1.0 - qd / g["velocity_limit"]), max=g["effort_limit"]
        )
        min_effort = torch.clamp(
            g["saturation"] * (-1.0 - qd / g["velocity_limit"]), min=-g["effort_limit"]
        )
        torque = torch.clamp(torque, min=min_effort, max=max_effort)
        robot.set_dof_efforts(wp.from_torch(torque.contiguous()))

    def _forward_explicit(self, dt):
        """PolicyController.forward, but writing torque instead of targets."""
        import torch

        policy = self.policy
        device = torch.device(str(policy.robot._device))
        if policy._previous_action is None:
            policy._previous_action = torch.zeros(12, device=device)
            policy._current_action = torch.zeros(12, device=device)

        if policy._policy_counter % policy._decimation == 0:
            obs = policy._compute_observation(self._command_tensor())
            policy._current_action = policy._compute_action(obs)
            policy._previous_action = policy._current_action.clone()
            self._target = policy.default_pos + policy._current_action * policy._action_scale
        policy._policy_counter += 1

        # Torque is recomputed every physics step from fresh joint state, which
        # is what Articulation.write_data_to_sim does inside Isaac Lab's loop.
        self._apply_explicit_torque()

    @property
    def robot(self):
        return self.policy.robot

    def set_command(self, lin_x, lin_y, ang_z):
        self._cmd = (
            _clamp(lin_x, GO2_MAX_LINEAR),
            _clamp(lin_y, GO2_MAX_LINEAR),
            _clamp(ang_z, GO2_MAX_ANGULAR),
        )

    def _command_tensor(self):
        import torch

        device = torch.device(str(self.policy.robot._device))
        if self._command is None or self._command.device != device:
            self._command = torch.zeros(3, device=device)
        self._command[0] = self._cmd[0]
        self._command[1] = self._cmd[1]
        self._command[2] = self._cmd[2]
        return self._command

    def on_physics_step(self, dt):
        """Run one policy step. Mirrors Isaac's own Go2 example.

        The articulation's physics tensor entity is only valid once the
        timeline is playing, and it goes invalid again across a stop/reset --
        hence re-initialising rather than assuming a single setup.
        """
        if not self.policy.robot.is_physics_tensor_entity_valid():
            self._physics_ready = False

        if self._physics_ready:
            if self._explicit:
                self._forward_explicit(dt)
            else:
                self.policy.forward(dt, self._command_tensor())
        else:
            self._physics_ready = True
            if self._explicit:
                # set_gains=False leaves the implicit drives at zero so they do
                # not fight the torques computed in _apply_explicit_torque.
                self.policy.initialize(control_mode="effort", set_gains=False)
                self._init_explicit_actuation()
            else:
                self.policy.initialize()
            self.policy.post_reset()
            self._target = None

    def on_render_step(self):
        pass

    def get_world_pose(self):
        positions, orientations = self.policy.robot.get_world_poses()
        # (N, 3) and (N, 4) wxyz -- the convention HuNavManager already expects.
        return _to_numpy(positions)[0], _to_numpy(orientations)[0]

    def get_linear_velocity(self):
        linear, _ = self.policy.robot.get_velocities()
        return _to_numpy(linear)[0]

    def get_angular_velocity(self):
        _, angular = self.policy.robot.get_velocities()
        return _to_numpy(angular)[0]

    def joint_state(self):
        try:
            names = list(self.policy.robot.dof_names)
            positions = _to_numpy(self.policy.robot.get_dof_positions())[0]
        except Exception:
            # Before the first physics step there is no articulation view yet.
            return None
        return names, positions


_DRIVERS = {
    "wheeled": WheeledDriver,
    "go2_policy": Go2PolicyDriver,
}


def make_driver(spec, world, usd_path, position):
    """Build the driver named by ``spec.driver``."""
    try:
        cls = _DRIVERS[spec.driver]
    except KeyError:
        raise ValueError(
            f"Robot '{spec.key}' asks for unknown driver '{spec.driver}'"
        ) from None
    return cls(world, spec, usd_path, position)
