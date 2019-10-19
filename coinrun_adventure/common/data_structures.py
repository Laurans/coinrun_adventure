from dataclasses import dataclass
from dataclasses_serialization.json import JSONSerializerMixin


@dataclass
class Step(JSONSerializerMixin):
    timestep: int = None
    imagename: str = None
    reward: float = None
    done: bool = None
    actions: list = None
    state_value: float = None
    pi_raw: list = None


@dataclass
class Metadata(JSONSerializerMixin):
    game_name: str = None
    action_names: list = None
    sequence_folder_name: str = None
    images_folder_name: str = None
    images_explain_name: str = None
