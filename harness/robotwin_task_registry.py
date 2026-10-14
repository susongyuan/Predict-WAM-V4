"""Task definitions and independently registered providers for native14 reuse.

Registration does not create robot perception or convert Unknown into evidence.
All complete-system providers must be explicitly admitted outside model output.
"""
from dataclasses import dataclass
import json
from pathlib import Path
from harness.robotwin_task_supervisor import ProviderBinding
from harness.robotwin_system_adapter import reject_gt


@dataclass(frozen=True)
class TaskDefinition:
    task: str
    targets: tuple
    predicates: tuple
    max_recoveries: int = 2
    max_age_s: float = 30.
    robot_version: str = 'aloha-agilex-native14'

    def __post_init__(self):
        if (not self.task or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in self.task)
                or not self.targets or len(set(self.targets)) != len(self.targets)
                or not self.predicates or len(set(self.predicates)) != len(self.predicates)
                or any(not isinstance(v, str) or not v for v in self.targets+self.predicates)
                or type(self.max_recoveries) is not int or self.max_recoveries < 0
                or not 0 < self.max_age_s <= 300):
            raise ValueError('invalid task definition')

    @property
    def target_set_key(self):
        """Semantic registry key, NEVER a visible physical object/track ID."""
        return 'task-target-set:' + self.task + ':' + '|'.join(self.targets)

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text())
        if value.get('format') != 'robotwin-task-definition-v1':
            raise ValueError('unsupported task definition')
        return cls(value['task'], tuple(value['targets']), tuple(value['predicates']),
                   max_recoveries=value.get('max_recoveries', 2), max_age_s=value.get('max_age_s', 30.),
                   robot_version=value.get('robot_version', 'aloha-agilex-native14'))


def aggregate_target_predicates(definition, per_target):
    """AND each predicate over the exact target set; missing evidence is Unknown.

    A known False remains False even when another target is Unknown. Whether a
    complete terminal failure is established remains the strict supervisor's job.
    """
    reject_gt(per_target)
    if not isinstance(per_target, dict) or set(per_target) != set(definition.targets):
        raise ValueError('exact target set required; missing identities are not success')
    for row in per_target.values():
        if (not isinstance(row, dict) or set(row) != set(definition.predicates)
                or any(v is not None and type(v) is not bool for v in row.values())):
            raise ValueError('complete three-valued predicate schema required')
    result = {}
    for name in definition.predicates:
        values = [per_target[target][name] for target in definition.targets]
        result[name] = False if False in values else None if None in values else True
    return result  # Fresh isolated dict; Outcome serialization must remain supported.


class TaskRegistry:
    def __init__(self):
        self._tasks = {}

    def register(self, definition):
        if not isinstance(definition, TaskDefinition) or definition.task in self._tasks:
            raise ValueError('unique typed task registration required')
        self._tasks[definition.task] = definition

    def get(self, task):
        try:
            return self._tasks[task]
        except KeyError as exc:
            raise ValueError('task has not been registered: ' + str(task)) from exc


def require_full_providers(scope, grounder, verifier, visible_tracks):
    for name, provider in (('grounder', grounder), ('verifier', verifier)):
        if (not isinstance(provider, ProviderBinding) or not callable(provider.callback)
                or not provider.provider_id or not provider.version or provider.scope_key != scope.key()
                or provider.trusted is not True or provider.admitted is not True
                or name == 'verifier' and provider.independent is not True):
            raise ValueError('full system requires scoped, admitted ' + name)
    if not callable(visible_tracks):
        raise ValueError('sensor-only visible track producer required')
