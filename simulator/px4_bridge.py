"""
PX4 MAVLink SITL Bridge

Bridges this fixed-wing physics simulator with PX4 via the MAVLink
Simulator API (HIL_* messages + HIL_ACTUATOR_CONTROLS).
"""

from __future__ import annotations

import argparse
import inspect
import math
import time
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from .aircraft import AircraftConfig
from .dynamics import FlightDynamics, SimulationConfig
from .environment import Environment
from .intruders import IntruderConfig, IntruderManager
from .state import ControlInputs
from .trim import TrimCondition, compute_trim

try:
    from pymavlink import mavutil

    HAS_PYMAVLINK = True
except Exception:
    HAS_PYMAVLINK = False


EARTH_RADIUS_M = 6378137.0


@dataclass
class GeoReference:
    """Local NED origin in geodetic coordinates."""

    lat_deg: float = 37.4276
    lon_deg: float = -122.1697
    alt_m_msl: float = 30.0


@dataclass
class ActuatorMapping:
    """Maps PX4 HIL actuator channels into sim controls."""

    aileron_idx: int = 0
    elevator_idx: int = 1
    rudder_idx: int = 2
    throttle_idx: int = 3

    reverse_aileron: bool = False
    reverse_elevator: bool = False
    reverse_rudder: bool = False

    throttle_bipolar: bool = True


@dataclass
class PX4BridgeConfig:
    """Runtime configuration for bridge loop and MAVLink link."""

    connection: str = "tcp:127.0.0.1:4560"
    source_system: int = 245
    source_component: int = 191
    heartbeat_hz: float = 1.0
    obstacle_hz: float = 5.0
    timeout_s: float = 10.0


class ControlInterventionPolicy(Protocol):
    """Optional control intervention layer for DAA experiments."""

    def intervene(
        self,
        controls: ControlInputs,
        dynamics: FlightDynamics,
        intruder_manager: Optional[IntruderManager],
    ) -> ControlInputs:
        ...


@dataclass
class SimpleBearingAvoidancePolicy:
    """
    Simple geometry-based DAA intervention.

    This is a placeholder policy to exercise intervention plumbing before
    camera-based `.pt` inference is integrated.
    """

    trigger_distance_m: float = 220.0
    forward_cone_deg: float = 45.0
    bank_command: float = 0.35
    climb_elevator: float = -0.08

    def intervene(
        self,
        controls: ControlInputs,
        dynamics: FlightDynamics,
        intruder_manager: Optional[IntruderManager],
    ) -> ControlInputs:
        if intruder_manager is None or not intruder_manager.intruders:
            return controls

        own = dynamics.state
        R_ned_to_body = own.quaternion.to_dcm().T
        cone = math.radians(self.forward_cone_deg)

        nearest = None
        nearest_az = 0.0
        for intruder in intruder_manager.intruders:
            rel_ned = intruder.dynamics.state.position - own.position
            rel_body = R_ned_to_body @ rel_ned
            x_fwd = rel_body[0]
            y_right = rel_body[1]
            if x_fwd <= 0.0:
                continue
            rng = float(np.linalg.norm(rel_body))
            az = math.atan2(y_right, x_fwd)
            if abs(az) <= cone and (nearest is None or rng < nearest):
                nearest = rng
                nearest_az = az

        if nearest is None or nearest > self.trigger_distance_m:
            return controls

        # Turn away from intruder bearing and bias to climb.
        turn_sign = -1.0 if nearest_az >= 0.0 else 1.0
        return ControlInputs(
            elevator=float(np.clip(controls.elevator + self.climb_elevator, -dynamics.aircraft.max_elevator, dynamics.aircraft.max_elevator)),
            aileron=float(np.clip(turn_sign * self.bank_command, -dynamics.aircraft.max_aileron, dynamics.aircraft.max_aileron)),
            rudder=float(np.clip(turn_sign * 0.15, -dynamics.aircraft.max_rudder, dynamics.aircraft.max_rudder)),
            throttle=controls.throttle,
        )


class PX4MavlinkBridge:
    """Bidirectional MAVLink bridge between PX4 SITL and this simulator."""

    def __init__(
        self,
        dynamics: FlightDynamics,
        environment: Environment,
        geo_reference: GeoReference,
        bridge_config: PX4BridgeConfig,
        actuator_mapping: ActuatorMapping,
        intruder_manager: Optional[IntruderManager] = None,
        intervention_policy: Optional[ControlInterventionPolicy] = None,
    ):
        if not HAS_PYMAVLINK:
            raise ImportError("pymavlink is required. Install with: pip install pymavlink")

        self.dynamics = dynamics
        self.environment = environment
        self.geo_reference = geo_reference
        self.bridge_config = bridge_config
        self.actuator_mapping = actuator_mapping
        self.intruder_manager = intruder_manager
        self.intervention_policy = intervention_policy

        self._master = None
        self._last_controls = ControlInputs()
        self._last_heartbeat_t = 0.0
        self._last_obstacle_t = 0.0
        self._boot_wall_t = time.time()

    @staticmethod
    def _fn_arity(fn) -> int:
        """Return argument count for bound MAVLink send function."""
        return len(inspect.signature(fn).parameters)

    def connect(self) -> None:
        """Connect to PX4 MAVLink simulator endpoint."""
        self._master = mavutil.mavlink_connection(
            self.bridge_config.connection,
            source_system=self.bridge_config.source_system,
            source_component=self.bridge_config.source_component,
            autoreconnect=True,
        )

        print(f"Connecting to PX4 at {self.bridge_config.connection}...")
        hb = self._master.wait_heartbeat(timeout=self.bridge_config.timeout_s)
        if hb is None:
            raise TimeoutError("Timed out waiting for PX4 heartbeat")
        print("PX4 heartbeat received")

    def _control_from_actuator(self, msg) -> ControlInputs:
        controls = list(msg.controls)

        def channel(idx: int, default: float = 0.0) -> float:
            if idx < 0 or idx >= len(controls):
                return default
            value = controls[idx]
            if value is None or math.isnan(value):
                return default
            return float(value)

        aileron = channel(self.actuator_mapping.aileron_idx)
        elevator = channel(self.actuator_mapping.elevator_idx)
        rudder = channel(self.actuator_mapping.rudder_idx)
        throttle_raw = channel(self.actuator_mapping.throttle_idx)

        if self.actuator_mapping.reverse_aileron:
            aileron *= -1.0
        if self.actuator_mapping.reverse_elevator:
            elevator *= -1.0
        if self.actuator_mapping.reverse_rudder:
            rudder *= -1.0

        if self.actuator_mapping.throttle_bipolar:
            throttle = 0.5 * (throttle_raw + 1.0)
        else:
            throttle = throttle_raw

        return ControlInputs(
            aileron=float(np.clip(aileron * self.dynamics.aircraft.max_aileron, -self.dynamics.aircraft.max_aileron, self.dynamics.aircraft.max_aileron)),
            elevator=float(np.clip(elevator * self.dynamics.aircraft.max_elevator, -self.dynamics.aircraft.max_elevator, self.dynamics.aircraft.max_elevator)),
            rudder=float(np.clip(rudder * self.dynamics.aircraft.max_rudder, -self.dynamics.aircraft.max_rudder, self.dynamics.aircraft.max_rudder)),
            throttle=float(np.clip(throttle, 0.0, 1.0)),
        )

    def _drain_actuators(self) -> Optional[ControlInputs]:
        latest = None
        while True:
            msg = self._master.recv_match(type="HIL_ACTUATOR_CONTROLS", blocking=False)
            if msg is None:
                break
            latest = self._control_from_actuator(msg)
        return latest

    def _to_gps(self):
        state = self.dynamics.state
        vn, ve, vd = state.velocity_ned
        groundspeed = math.sqrt(vn * vn + ve * ve)
        cog_deg = (math.degrees(math.atan2(ve, vn)) + 360.0) % 360.0

        lat = self.geo_reference.lat_deg + math.degrees(state.p_north / EARTH_RADIUS_M)
        lon = self.geo_reference.lon_deg + math.degrees(
            state.p_east / (EARTH_RADIUS_M * math.cos(math.radians(self.geo_reference.lat_deg)))
        )
        alt_msl = self.geo_reference.alt_m_msl + state.altitude

        return {
            "lat_e7": int(round(lat * 1e7)),
            "lon_e7": int(round(lon * 1e7)),
            "alt_mm": int(round(alt_msl * 1000.0)),
            "vn_cms": int(round(vn * 100.0)),
            "ve_cms": int(round(ve * 100.0)),
            "vd_cms": int(round(vd * 100.0)),
            "vel_cms": int(round(groundspeed * 100.0)),
            "cog_cdeg": int(round(cog_deg * 100.0)),
        }

    def _send_heartbeat(self, now: float) -> None:
        period = 1.0 / max(self.bridge_config.heartbeat_hz, 0.1)
        if now - self._last_heartbeat_t < period:
            return
        self._last_heartbeat_t = now

        self._master.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GENERIC,
            mavutil.mavlink.MAV_AUTOPILOT_GENERIC,
            0,
            0,
            mavutil.mavlink.MAV_STATE_ACTIVE,
        )

    def _send_hil_messages(self) -> None:
        now_us = int((time.time() - self._boot_wall_t) * 1e6)
        state = self.dynamics.state
        fm = self.dynamics.forces_moments
        atmo = self.environment.get_atmosphere(state.altitude)

        gps = self._to_gps()

        accel = fm.force / self.dynamics.aircraft.mass_properties.mass
        pressure_alt = float(state.altitude)
        abs_pressure_mbar = float(atmo.pressure / 100.0)

        airspeed = max(fm.airspeed, 0.0)
        diff_pressure_pa = 0.5 * atmo.density * airspeed * airspeed
        diff_pressure_mbar = float(diff_pressure_pa / 100.0)

        hil_sensor_args = [
            now_us,
            float(accel[0]),
            float(accel[1]),
            float(accel[2]),
            float(state.p),
            float(state.q),
            float(state.r),
            0.0,
            0.0,
            0.0,
            abs_pressure_mbar,
            diff_pressure_mbar,
            pressure_alt,
            float(atmo.temperature - 273.15),
            int(
                mavutil.mavlink.HIL_SENSOR_UPDATED_XACC
                | mavutil.mavlink.HIL_SENSOR_UPDATED_YACC
                | mavutil.mavlink.HIL_SENSOR_UPDATED_ZACC
                | mavutil.mavlink.HIL_SENSOR_UPDATED_XGYRO
                | mavutil.mavlink.HIL_SENSOR_UPDATED_YGYRO
                | mavutil.mavlink.HIL_SENSOR_UPDATED_ZGYRO
                | mavutil.mavlink.HIL_SENSOR_UPDATED_ABS_PRESSURE
                | mavutil.mavlink.HIL_SENSOR_UPDATED_DIFF_PRESSURE
            ),
        ]
        arity = self._fn_arity(self._master.mav.hil_sensor_send)
        if arity >= 16:
            hil_sensor_args.append(0)  # id extension
        self._master.mav.hil_sensor_send(*hil_sensor_args[:arity])

        hil_gps_args = [
            now_us,
            3,
            gps["lat_e7"],
            gps["lon_e7"],
            gps["alt_mm"],
            100,
            100,
            gps["vel_cms"],
            gps["vn_cms"],
            gps["ve_cms"],
            gps["vd_cms"],
            gps["cog_cdeg"],
            10,
            0,  # id extension
            0,  # yaw extension
        ]
        arity = self._fn_arity(self._master.mav.hil_gps_send)
        self._master.mav.hil_gps_send(*hil_gps_args[:arity])

        hil_state_args = [
            now_us,
            state.quaternion.to_array(),
            float(state.p),
            float(state.q),
            float(state.r),
            gps["lat_e7"],
            gps["lon_e7"],
            gps["alt_mm"],
            gps["vn_cms"],
            gps["ve_cms"],
            gps["vd_cms"],
            int(round(airspeed * 100.0)),  # indicated airspeed (cm/s)
            int(round(airspeed * 100.0)),  # true airspeed (cm/s)
            int(round(float(accel[0]) * 1000.0)),  # mG
            int(round(float(accel[1]) * 1000.0)),  # mG
            int(round(float(accel[2]) * 1000.0)),  # mG
        ]
        arity = self._fn_arity(self._master.mav.hil_state_quaternion_send)
        self._master.mav.hil_state_quaternion_send(*hil_state_args[:arity])

    def _send_obstacle_distance(self, now: float) -> None:
        if self.intruder_manager is None:
            return
        period = 1.0 / max(self.bridge_config.obstacle_hz, 0.1)
        if now - self._last_obstacle_t < period:
            return
        self._last_obstacle_t = now

        distances_cm = [65535] * 72
        min_cm = 100
        max_cm = 12000
        increment_deg = 5
        own = self.dynamics.state
        R_ned_to_body = own.quaternion.to_dcm().T

        for intruder in self.intruder_manager.intruders:
            intr = intruder.dynamics.state
            rel_ned = intr.position - own.position
            rel_body = R_ned_to_body @ rel_ned
            x_fwd = rel_body[0]
            y_right = rel_body[1]
            if x_fwd <= 0.0:
                continue
            range_m = float(np.linalg.norm(rel_body))
            az_deg = (math.degrees(math.atan2(y_right, x_fwd)) + 360.0) % 360.0
            idx = int(az_deg // increment_deg) % len(distances_cm)
            cm = int(round(range_m * 100.0))
            cm = max(min_cm, min(max_cm, cm))
            distances_cm[idx] = min(distances_cm[idx], cm)

        obstacle_args = [
            int((time.time() - self._boot_wall_t) * 1e6),
            mavutil.mavlink.MAV_DISTANCE_SENSOR_UNKNOWN,
            distances_cm,
            increment_deg,
            min_cm,
            max_cm,
            0.0,
            0.0,
            mavutil.mavlink.MAV_FRAME_BODY_FRD,
        ]
        arity = self._fn_arity(self._master.mav.obstacle_distance_send)
        self._master.mav.obstacle_distance_send(*obstacle_args[:arity])

    def run(self, duration_s: Optional[float] = None) -> None:
        """Run bridge loop in real-time."""
        dt = self.dynamics.sim_config.dt
        end_t = None if duration_s is None else (self.dynamics.state.time + duration_s)

        print(f"Bridge running at {1.0 / dt:.1f} Hz")
        while True:
            loop_start = time.time()

            new_controls = self._drain_actuators()
            if new_controls is not None:
                self._last_controls = new_controls

            commanded_controls = self._last_controls
            if self.intervention_policy is not None:
                commanded_controls = self.intervention_policy.intervene(
                    commanded_controls,
                    self.dynamics,
                    self.intruder_manager,
                )

            self.dynamics.step(commanded_controls)

            if self.intruder_manager is not None:
                if self.intruder_manager.should_spawn_intruder(dt, self.dynamics.state):
                    self.intruder_manager.spawn_intruder(self.dynamics.state)
                self.intruder_manager.update_intruders(dt, self.dynamics.state)

            self._send_hil_messages()
            now = time.time()
            self._send_heartbeat(now)
            self._send_obstacle_distance(now)

            if self.dynamics.crash_state.crashed:
                print(f"Simulation terminated: {self.dynamics.crash_state.crash_message}")
                break

            if end_t is not None and self.dynamics.state.time >= end_t:
                break

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, dt - elapsed))


def build_default_dynamics(aircraft: AircraftConfig) -> tuple[FlightDynamics, Environment]:
    """Create and trim dynamics for PX4 bridge usage."""
    environment = Environment()
    sim_config = SimulationConfig(dt=0.01)
    dynamics = FlightDynamics(aircraft, environment, sim_config)

    trim = compute_trim(
        TrimCondition(airspeed=25.0, altitude=100.0),
        aircraft,
        environment,
    )
    if trim.success:
        dynamics.reset(trim.state)
        dynamics.controls = trim.controls
        print("Initialized from trim condition")
    else:
        dynamics.reset()
        print("Trim failed; using default initial state")

    return dynamics, environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PX4 SITL bridge for fixed-wing UAS simulator")
    parser.add_argument("--aircraft", type=str, help="Path to aircraft YAML config")
    parser.add_argument("--connection", type=str, default="tcp:127.0.0.1:4560", help="MAVLink connection string")
    parser.add_argument("--duration", type=float, default=0.0, help="Run duration in seconds (0 = forever)")

    parser.add_argument("--home-lat", type=float, default=37.4276, help="Home latitude in degrees")
    parser.add_argument("--home-lon", type=float, default=-122.1697, help="Home longitude in degrees")
    parser.add_argument("--home-alt", type=float, default=30.0, help="Home altitude MSL in meters")

    parser.add_argument("--enable-intruders", action="store_true", help="Enable intruder generation")
    parser.add_argument("--intruder-rate", type=float, default=0.1, help="Intruder spawn rate (per second)")
    parser.add_argument("--simple-daa", action="store_true", help="Enable simple geometry-based control intervention")
    parser.add_argument(
        "--policy",
        type=str,
        help="Path to PyTorch checkpoint (.pt) for ModelPolicy intervention (RFC 001)",
    )
    parser.add_argument(
        "--policy-device",
        type=str,
        default="cpu",
        help="Torch device for --policy (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--training-fidelity",
        action="store_true",
        help="Use scratch_built_daa presets for policy observations and trim",
    )

    parser.add_argument("--reverse-aileron", action="store_true", help="Invert aileron sign")
    parser.add_argument("--reverse-elevator", action="store_true", help="Invert elevator sign")
    parser.add_argument("--reverse-rudder", action="store_true", help="Invert rudder sign")
    parser.add_argument("--throttle-unipolar", action="store_true", help="Treat throttle as [0,1] instead of [-1,1]")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.aircraft:
        aircraft = AircraftConfig.from_yaml(args.aircraft)
        print(f"Loaded aircraft: {aircraft.name}")
    else:
        aircraft = AircraftConfig(name="Generic UAS")
        print("Using default aircraft config")

    dynamics, environment = build_default_dynamics(aircraft)

    intruder_manager = None
    if args.enable_intruders:
        intruder_cfg = IntruderConfig(spawn_rate=args.intruder_rate)
        intruder_manager = IntruderManager(intruder_cfg, environment)
        print("Intruders enabled")

    intervention_policy = None
    if args.policy:
        from .policy import TrainingFidelityConfig, TrimAssistedModelPolicy
        from .trim import TrimCondition, compute_trim

        if args.training_fidelity:
            tf = TrainingFidelityConfig.defaults()
            trim = compute_trim(
                TrimCondition(airspeed=tf.cruise_speed_mps, altitude=1000.0),
                dynamics.aircraft,
                environment,
            )
            intervention_policy = TrimAssistedModelPolicy.from_checkpoint(
                args.policy,
                trim.controls,
                device=args.policy_device,
                renderer_backend=tf.renderer_backend,
                throttle_mode=tf.throttle_mode,
                surface_scale=tf.surface_scale,
            )
        else:
            trim = compute_trim(
                TrimCondition(airspeed=25.0, altitude=100.0),
                dynamics.aircraft,
                environment,
            )
            intervention_policy = TrimAssistedModelPolicy.from_checkpoint(
                args.policy,
                trim.controls,
                device=args.policy_device,
            )
        print(f"TrimAssistedModelPolicy loaded from {args.policy} (device={args.policy_device})")
    elif args.simple_daa:
        intervention_policy = SimpleBearingAvoidancePolicy()

    bridge = PX4MavlinkBridge(
        dynamics=dynamics,
        environment=environment,
        geo_reference=GeoReference(args.home_lat, args.home_lon, args.home_alt),
        bridge_config=PX4BridgeConfig(connection=args.connection),
        actuator_mapping=ActuatorMapping(
            reverse_aileron=args.reverse_aileron,
            reverse_elevator=args.reverse_elevator,
            reverse_rudder=args.reverse_rudder,
            throttle_bipolar=not args.throttle_unipolar,
        ),
        intruder_manager=intruder_manager,
        intervention_policy=intervention_policy,
    )

    bridge.connect()
    bridge.run(duration_s=None if args.duration <= 0.0 else args.duration)


if __name__ == "__main__":
    main()
