from dataclasses import dataclass
from typing import Union


@dataclass
class PhotoPaths:
    """
    Stores the file paths for four specific types of photos associated with a patient visit.
    D, C, N, P represent different imaging modalities.
    """
    photo_pathD: str
    photo_pathC: str
    photo_pathN: str
    photo_pathP: str


@dataclass
class TimePoint:
    """
    Stores the data for a single patient visit/time point, including the patient's ID,
    the associated photo paths, and the clinical grade.
    """
    patient_id: str
    photo_paths: PhotoPaths
    grade: float


@dataclass
class TimePoints:
    """
    Stores longitudinal data for a single patient across up to 9 time points.
    Missing time points are represented as None.
    """
    patient_id: str
    point_1: Union[TimePoint, None]
    point_2: Union[TimePoint, None]
    point_3: Union[TimePoint, None]
    point_4: Union[TimePoint, None]
    point_5: Union[TimePoint, None]
    point_6: Union[TimePoint, None]
    point_7: Union[TimePoint, None]
    point_8: Union[TimePoint, None]
    point_9: Union[TimePoint, None]