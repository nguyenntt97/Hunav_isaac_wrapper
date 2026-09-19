"""
Scenario authoring for HuNav agents.

One in-memory `ScenarioSpec` is the source of truth for a scene's agents. The
two artifacts HuNav actually consumes -- the scenario YAML read by
`hunav_loader`, and one behavior tree XML per agent read by
`hunav_agent_manager` -- are both emitted from it, never edited independently.

See ``new_behavior/tasks/hunav_runtime_contract.md`` for why that matters: the
YAML's ``goals:`` list is read only by tree generators, so a YAML edited without
its trees regenerated leaves agents walking the previous ring.

Nothing in `spec` or `bt_emit` imports Isaac Sim, so both are testable with a
plain interpreter. `sampling` and `authoring` need a live stage.
"""

from .spec import (
    BEHAVIOR_TYPES,
    BEHAVIOR_NAMES,
    BehaviorSpec,
    AgentSpec,
    Pose,
    Problem,
    ScenarioSpec,
    FORCE_FACTOR_RANGES,
    VEL_RANGE,
)
from .bt_emit import emit_tree, emit_all_trees, bt_filename

__all__ = [
    "BEHAVIOR_TYPES",
    "BEHAVIOR_NAMES",
    "BehaviorSpec",
    "AgentSpec",
    "Pose",
    "Problem",
    "ScenarioSpec",
    "FORCE_FACTOR_RANGES",
    "VEL_RANGE",
    "emit_tree",
    "emit_all_trees",
    "bt_filename",
]
