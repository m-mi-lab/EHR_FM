from enum import StrEnum
from importlib.resources import files
from pathlib import Path

from enum import StrEnum
from importlib.resources import files
from pathlib import Path
import os

PROJECT_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
MAPPINGS_DIR = PROJECT_ROOT / "tokenizer" / "maps"


class SpecialToken(StrEnum):
    DOB = "MEDS_BIRTH"
    DEATH = "MEDS_DEATH"
    TIMELINE_END = "TIMELINE_END"

    # hospital adm and dc
    ADMISSION = "HOSPITAL_ADMISSION"
    DISCHARGE = "HOSPITAL_DISCHARGE"

    # only MIMIC
    ICU_ADMISSION = "ICU_ADMISSION"
    ICU_DISCHARGE = "ICU_DISCHARGE"
    ED_ADMISSION = "ED_REGISTRATION"
    ED_DISCHARGE = "ED_OUT"
    SOFA = "SOFA"


STATIC_DATA_FN = "static_data.pickle"