from dataclasses import dataclass, field
from typing import List, Optional

from synflownet.utils.misc import StrictDataClass


@dataclass
class SEHTaskConfig(StrictDataClass):
    reduced_frag: bool = False


@dataclass
class ReactionTaskConfig:
    templates_filename: str = "hb.txt"
    reverse_templates_filename: Optional[str] = None
    reward: Optional[str] = None
    building_blocks_filename: str = "enamine_bbs.txt"
    precomputed_bb_masks_filename: str = "precomputed_bb_masks_enamine_bbs.pkl"
    building_blocks_costs: Optional[str] = None
    sanitize_building_blocks: bool = False


@dataclass
class VinaConfig(StrictDataClass):
    opencl_binary_path: str = (
        "bin/QuickVina2-GPU-2-1"  # needed if you use VINA for rewards
    )
    vina_path: str = (
        "bin/QuickVina2-GPU-2-1/Vina-GPU"  # path to VINA executable, needed if you use VINA for rewards
    )
    target: str = "kras"  # kras, 2bm2


@dataclass
class TasksConfig(StrictDataClass):
    reactions_task: ReactionTaskConfig = field(default_factory=ReactionTaskConfig)
