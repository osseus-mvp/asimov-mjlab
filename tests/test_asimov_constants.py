"""Tests for asimov_constants.py.

These verify the requirement register published on the Asimov v1 hardware
repository, osseus-mvp/asimov-1, at tests/requirements/requirements.yaml. Each
test names the requirement it verifies, and the register's `verifies` field
points back at the test by node id.

The expected values below are transcribed by hand from
src/mjlab/asset_zoo/robots/asimov/README.md, which is the motor table this
package's constants were derived from. They are deliberately not imported from
asimov_constants: a test that reads its expectation out of the module it is
checking would pass no matter what the module said.
"""

import math
import re

import mujoco
import pytest

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.asimov import asimov_constants
from mjlab.entity import Entity

# Motor Specifications table, README.md: model, gear ratio, peak torque (Nm).
# REQ-L3-MOTOR-HIP-PITCH-PEAK, REQ-L3-MOTOR-KNEE-PEAK, REQ-L3-MOTOR-ANKLE-PEAK.
PEAK_TORQUE_NM = {
  "hip_pitch": ("EC-A6416-P2-25", 120.0),
  "hip_roll": ("EC-A5013-H17-100", 90.0),
  "hip_yaw": ("EC-A3814-H14-107", 60.0),
  "knee": ("EC-A4315-P2-36", 75.0),
  "ankle": ("EC-A4310-P2-36", 36.0),
}

# Armature table, README.md: rotor inertia (kg*mm^2) and gear ratio.
ROTOR_INERTIA_KG_MM2 = {
  "hip_pitch": (104.395, 25),
  "hip_roll": (10.0, 100),
  "hip_yaw": (3.0, 107),
  "knee": (25.5, 36),
  "ankle": (18.2, 36),
}

# REQ-L2-DAMPING-CEILING. README.md: "we cap KD at 5.0 Nm-s/rad for all joints
# due to hardware limitations."
DAMPING_HARDWARE_MAX = 5.0

# REQ-L2-REFLECTED-INERTIA. The published armatures are given to three
# significant figures, so half-ulp rounding alone reaches ~0.15 percent.
ARMATURE_TOLERANCE_PERCENT = 0.5

# REQ-L2-ACTUATOR-COVERAGE. asimov_constants: "Asimov is a 12-DOF bipedal
# robot with 6 joints per leg."
LOCOMOTION_JOINT_COUNT = 12


@pytest.fixture(scope="module")
def asimov_entity() -> Entity:
  return Entity(asimov_constants.get_asimov_robot_cfg())


@pytest.fixture(scope="module")
def asimov_model(asimov_entity: Entity) -> mujoco.MjModel:
  return asimov_entity.spec.compile()


def _actuator_configs() -> tuple[BuiltinPositionActuatorCfg, ...]:
  configs = asimov_constants.ASIMOV_ARTICULATION.actuators
  assert all(isinstance(cfg, BuiltinPositionActuatorCfg) for cfg in configs)
  return tuple(cfg for cfg in configs if isinstance(cfg, BuiltinPositionActuatorCfg))


# fmt: off
@pytest.mark.parametrize(
  "joint,actuator_config,effort",
  [
    ("hip_pitch", asimov_constants.ASIMOV_ACTUATOR_HIP_PITCH, asimov_constants.EFFORT_HIP_PITCH),
    ("hip_roll", asimov_constants.ASIMOV_ACTUATOR_HIP_ROLL, asimov_constants.EFFORT_HIP_ROLL),
    ("hip_yaw", asimov_constants.ASIMOV_ACTUATOR_HIP_YAW, asimov_constants.EFFORT_HIP_YAW),
    ("knee", asimov_constants.ASIMOV_ACTUATOR_KNEE, asimov_constants.EFFORT_KNEE),
    ("ankle", asimov_constants.ASIMOV_ACTUATOR_ANKLE, asimov_constants.EFFORT_ANKLE),
  ],
)
# fmt: on
def test_effort_limits_match_datasheet_peak_torques(
  asimov_model, joint, actuator_config, effort
):
  """REQ-L2-EFFORT-* and REQ-L3-MOTOR-*-PEAK: the limit is the motor's peak.

  An effort limit above the motor's peak torque is a command the hardware
  cannot execute, and a policy trained against it learns a robot that does not
  exist. An effort limit below it silently gives away capability the robot was
  specified to have, so the comparison is equality rather than an upper bound.
  """
  motor, peak = PEAK_TORQUE_NM[joint]

  assert effort == peak, (
    f"{joint} effort limit is {effort} Nm but {motor} peaks at {peak} Nm"
  )
  assert actuator_config.effort_limit == peak

  # The limit has to survive compilation, not just sit in the config: MuJoCo
  # enforces it as the actuator's force range.
  matched = 0
  for index in range(asimov_model.nu):
    actuator = asimov_model.actuator(index)
    if any(re.match(p, actuator.name) for p in actuator_config.joint_names_expr):
      matched += 1
      assert actuator.forcerange[0] == -peak
      assert actuator.forcerange[1] == peak
  assert matched > 0, f"no compiled actuator matches {actuator_config.joint_names_expr}"


def test_damping_never_exceeds_the_hardware_maximum(asimov_model):
  """REQ-L2-DAMPING-CEILING: no actuator asks for more KD than firmware can give.

  This is the requirement that only ever existed as a comment. Every constant
  sits exactly at 5.0, so the test has no headroom to absorb a change: raising
  any damping value fails here before it can reach a trained policy that will
  not deploy.
  """
  declared = {
    "hip_pitch": asimov_constants.DAMPING_HIP_PITCH,
    "hip_roll": asimov_constants.DAMPING_HIP_ROLL,
    "hip_yaw": asimov_constants.DAMPING_HIP_YAW,
    "knee": asimov_constants.DAMPING_KNEE,
    "ankle": asimov_constants.DAMPING_ANKLE,
  }
  above = {name: value for name, value in declared.items() if value > DAMPING_HARDWARE_MAX}
  assert not above, (
    f"damping constants above the {DAMPING_HARDWARE_MAX} Nm-s/rad hardware max: {above}"
  )

  for config in _actuator_configs():
    assert config.damping <= DAMPING_HARDWARE_MAX, (
      f"{config.joint_names_expr} sets damping {config.damping}, above the"
      f" {DAMPING_HARDWARE_MAX} Nm-s/rad hardware max"
    )

  # biasprm[2] carries -KD in a compiled position actuator.
  for index in range(asimov_model.nu):
    actuator = asimov_model.actuator(index)
    assert -actuator.biasprm[2] <= DAMPING_HARDWARE_MAX, (
      f"compiled actuator {actuator.name} has damping {-actuator.biasprm[2]}, above the"
      f" {DAMPING_HARDWARE_MAX} Nm-s/rad hardware max"
    )


def test_stiffness_equals_armature_times_natural_frequency_squared():
  """REQ-L2-STIFFNESS-FORMULA: KP = J_reflected * omega_n^2 at every joint.

  The point of the rule is that every joint ends up with the same closed-loop
  bandwidth. A stiffness typed in by hand would break that quietly, because a
  wrong gain still produces a robot that walks in simulation.
  """
  # The module truncates pi, so the bandwidth is checked against math.pi with
  # a tolerance and the per-joint products are checked against the module's own
  # constant exactly.
  assert asimov_constants.NATURAL_FREQ == pytest.approx(10 * 2.0 * math.pi, rel=1e-9)
  natural_freq = asimov_constants.NATURAL_FREQ

  pairs = [
    (asimov_constants.STIFFNESS_HIP_PITCH, asimov_constants.ARMATURE_HIP_PITCH),
    (asimov_constants.STIFFNESS_HIP_ROLL, asimov_constants.ARMATURE_HIP_ROLL),
    (asimov_constants.STIFFNESS_HIP_YAW, asimov_constants.ARMATURE_HIP_YAW),
    (asimov_constants.STIFFNESS_KNEE, asimov_constants.ARMATURE_KNEE),
    (asimov_constants.STIFFNESS_ANKLE_PITCH, asimov_constants.ARMATURE_ANKLE_PITCH),
    (asimov_constants.STIFFNESS_ANKLE_ROLL, asimov_constants.ARMATURE_ANKLE_ROLL),
  ]
  for stiffness, armature in pairs:
    assert stiffness == pytest.approx(armature * natural_freq**2, rel=1e-9)

  configs = _actuator_configs()
  assert len(configs) == 5
  for config in configs:
    assert config.stiffness == pytest.approx(
      config.armature * natural_freq**2, rel=1e-9
    ), f"{config.joint_names_expr} stiffness is not armature * natural_freq^2"


# fmt: off
@pytest.mark.parametrize(
  "joint,armature",
  [
    ("hip_pitch", asimov_constants.ARMATURE_HIP_PITCH),
    ("hip_roll", asimov_constants.ARMATURE_HIP_ROLL),
    ("hip_yaw", asimov_constants.ARMATURE_HIP_YAW),
    ("knee", asimov_constants.ARMATURE_KNEE),
    ("ankle", asimov_constants.ARMATURE_ANKLE_PITCH),
  ],
)
# fmt: on
def test_armature_equals_rotor_inertia_times_gear_ratio_squared(joint, armature):
  """REQ-L2-REFLECTED-INERTIA: armature = J_rotor * gear_ratio^2, within 0.5%.

  Armature mismatch is the classic sim2real failure: the dynamics differ, and
  a policy trained on the wrong reflected inertia falls over on the first step.
  Checking against the rotor inertia and gear ratio rather than against a
  stored number means a regeared joint fails here.
  """
  rotor_kg_mm2, gear_ratio = ROTOR_INERTIA_KG_MM2[joint]
  expected = rotor_kg_mm2 * 1e-6 * gear_ratio**2
  deviation = abs(armature - expected) / expected * 100

  assert deviation <= ARMATURE_TOLERANCE_PERCENT, (
    f"{joint} armature is {armature} kg-m^2 but {rotor_kg_mm2} kg-mm^2 at"
    f" {gear_ratio}:1 gives {expected} kg-m^2, a deviation of {deviation:.3f} percent"
  )


def test_actuator_patterns_cover_every_actuated_joint_exactly_once(asimov_entity):
  """REQ-L2-ACTUATOR-COVERAGE: all 12 joints matched, none matched twice.

  Both directions are failures. A joint matched by no pattern is unactuated in
  training and actuated on the robot; a joint matched by two takes whichever
  configuration was applied last, which is a gain nobody chose.
  """
  joint_names = asimov_entity.joint_names
  assert len(joint_names) == LOCOMOTION_JOINT_COUNT

  matches = {
    name: [
      pattern
      for config in _actuator_configs()
      for pattern in config.joint_names_expr
      if re.match(pattern, name)
    ]
    for name in joint_names
  }

  uncovered = sorted(name for name, patterns in matches.items() if not patterns)
  assert not uncovered, f"actuated joints no actuator pattern reaches: {uncovered}"

  doubled = {name: patterns for name, patterns in matches.items() if len(patterns) > 1}
  assert not doubled, f"joints matched by more than one actuator pattern: {doubled}"

  assert len(matches) == LOCOMOTION_JOINT_COUNT


def test_every_locomotion_joint_carries_a_finite_effort_limit(asimov_entity, asimov_model):
  """REQ-L1-TORQUE-LIMITED: all 12 joints are torque limited, none left open.

  An actuator with no effort limit can command any torque the solver asks for,
  which is the one thing the torque-limiting safety function exists to prevent.
  """
  for config in _actuator_configs():
    assert config.effort_limit is not None, (
      f"{config.joint_names_expr} sets no effort limit"
    )
    assert math.isfinite(config.effort_limit)

  assert asimov_model.nu == LOCOMOTION_JOINT_COUNT, (
    f"the compiled model has {asimov_model.nu} actuators, expected"
    f" {LOCOMOTION_JOINT_COUNT}"
  )
  for index in range(asimov_model.nu):
    actuator = asimov_model.actuator(index)
    assert asimov_model.actuator_forcelimited[index] == 1, (
      f"actuator {actuator.name} is not force limited"
    )
    assert actuator.forcerange[1] > 0


def test_locomotion_model_declares_twelve_actuated_joints(asimov_entity):
  """REQ-L2-ACTUATOR-COVERAGE: the model is the 12-DOF locomotion subset.

  Worth asserting on its own because of the discrepancy it records. The
  hardware repository declares 25 actuated joints for the whole robot; this
  package models the biped only, so 13 of those joints have no gains, no
  effort limit and no coverage here. Pinning 12 keeps that gap visible instead
  of letting the number drift towards looking complete.
  """
  assert asimov_entity.num_joints == LOCOMOTION_JOINT_COUNT
  assert asimov_entity.num_actuators == LOCOMOTION_JOINT_COUNT
  assert asimov_entity.is_actuated
  assert not asimov_entity.is_fixed_base
