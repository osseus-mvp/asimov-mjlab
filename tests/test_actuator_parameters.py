"""Tests for actuator_parameters.yaml.

The declaration exists so that a tool outside this repository can read the
actuator constants as data. It states no numbers — only which constant holds
which quantity, in which unit, for which joints — so the way it goes wrong is
by drifting: a constant renamed, a joint added, a gain moved from a literal to
an expression. Every one of those turns a correct-looking declaration into one
that names something that is not there.

These tests are what stop that. They compare the declaration against
asimov_constants.py and against the compiled model, and nothing here reads
README.md: the prose in this repository is already out of date about damping
(the "Resulting Gains" table says KD 5.0 and DAMPING_KNEE is 6.0), which is
exactly the failure mode the declaration is meant to avoid.
"""

import ast
import pathlib

import pytest
import yaml

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.asimov import asimov_constants
from mjlab.entity import Entity

DECLARATION_PATH = (
  pathlib.Path(asimov_constants.__file__).parent / "actuator_parameters.yaml"
)

SCHEMA = "osseus-actuator-parameters/v1"

# The measures a consumer is expected to find, and the unit each is stated in.
# Written out here rather than read from the declaration, so that changing a
# unit in the declaration alone fails this file.
EXPECTED_UNITS = {
  "effort_limit_nm": "N·m",
  "damping_nm_s_per_rad": "N·m·s/rad",
}


@pytest.fixture(scope="module")
def declaration() -> dict:
  return yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def asimov_entity() -> Entity:
  return Entity(asimov_constants.get_asimov_robot_cfg())


def _actuator_configs() -> tuple[BuiltinPositionActuatorCfg, ...]:
  configs = asimov_constants.ASIMOV_ARTICULATION.actuators
  return tuple(cfg for cfg in configs if isinstance(cfg, BuiltinPositionActuatorCfg))


def test_the_declaration_states_this_schema_and_points_at_the_constants(declaration):
  assert declaration["schema"] == SCHEMA
  module = DECLARATION_PATH.parent / declaration["constants_module"]
  assert module.resolve() == pathlib.Path(asimov_constants.__file__).resolve()


def test_every_named_constant_exists_and_is_a_module_level_literal(declaration):
  """A consumer reads these by parsing, not by importing, so a constant that
  is computed is unreadable to it even though it exists here."""
  source = pathlib.Path(asimov_constants.__file__).read_text(encoding="utf-8")
  literals = set()
  for statement in ast.parse(source).body:
    if not isinstance(statement, ast.Assign):
      continue
    for target in statement.targets:
      if not isinstance(target, ast.Name):
        continue
      try:
        ast.literal_eval(statement.value)
      except (ValueError, TypeError, SyntaxError):
        continue
      literals.add(target.id)

  for actuator in declaration["actuators"]:
    for parameter in actuator["parameters"]:
      name = parameter["constant"]
      assert hasattr(asimov_constants, name), (
        f"{actuator['id']} names {name}, which asimov_constants does not define"
      )
      assert name in literals, (
        f"{actuator['id']} names {name}, which is not a module-level literal —"
        " a reader that parses rather than imports cannot get its value"
      )


def test_every_declared_value_matches_the_constant_it_names(declaration):
  """The declaration must describe the module it points at, not a past one."""
  for actuator in declaration["actuators"]:
    for parameter in actuator["parameters"]:
      value = getattr(asimov_constants, parameter["constant"])
      assert isinstance(value, (int, float)) and not isinstance(value, bool), (
        f"{parameter['constant']} is {type(value).__name__}, which is not a"
        " number a limit can be compared against"
      )


def test_units_are_stated_and_are_the_ones_a_consumer_expects(declaration):
  for actuator in declaration["actuators"]:
    for parameter in actuator["parameters"]:
      measure = parameter["measure"]
      assert measure in EXPECTED_UNITS, f"unknown measure {measure!r}"
      assert parameter["unit"] == EXPECTED_UNITS[measure], (
        f"{actuator['id']}'s {measure} is stated in {parameter['unit']!r};"
        f" a consumer compares units as exact strings and expects"
        f" {EXPECTED_UNITS[measure]!r}"
      )


def test_the_declared_joints_are_exactly_the_models_actuated_joints(
  declaration, asimov_entity
):
  """Both directions, because both are wrong.

  A joint the declaration omits has an effort limit nothing outside this
  repository can see; a joint it invents publishes a limit for a motor that
  does not exist.
  """
  declared = [joint for a in declaration["actuators"] for joint in a["joints"]]

  assert len(declared) == len(set(declared)), (
    f"a joint is declared under two actuators: {sorted(declared)}"
  )
  assert sorted(declared) == sorted(asimov_entity.joint_names)


def test_every_actuator_configuration_is_declared(declaration):
  """Five configurations in the module, five entries here."""
  assert len(declaration["actuators"]) == len(_actuator_configs())


def test_the_declaration_states_no_values(declaration):
  """The property that stops it drifting: there is no number to go stale."""
  for actuator in declaration["actuators"]:
    for parameter in actuator["parameters"]:
      assert set(parameter) == {"constant", "measure", "unit"}, (
        f"{actuator['id']} states {sorted(set(parameter) - {'constant', 'measure', 'unit'})}"
        " beside the constant it names. A value written here can go stale"
        " against asimov_constants.py, which is the whole thing this file"
        " avoids by naming the constant instead"
      )
