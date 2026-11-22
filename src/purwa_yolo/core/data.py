"""Plain data containers passed between the vision layer and the UI.

No heavy imports here, so every layer can use them.
"""
from dataclasses import dataclass, field


@dataclass
class VehicleCrops:
    """One counted vehicle and its best crops (used later for plate reading)."""
    vehicle_id: int
    cls: str
    crops: list = field(default_factory=list)


@dataclass
class DetectionResult:
    """Output of section 1 (YOLO)."""
    kind: str
    media: object
    counts: dict
    density: int
    vehicles: list
    frames: int = 0
    h264: bool = True
    tracked: bool = True
    run_id: str = ""
    params: tuple = ()
    elapsed: float = 0.0

    @property
    def density_basis(self):
        return "in the image" if self.kind == "image" else "at most in a single frame"


@dataclass
class PlateReading:
    """Best plate reading of one vehicle (text is None when unreadable)."""
    vehicle_id: int
    cls: str
    text: str | None
    conf: float


@dataclass
class OcrResult:
    """Output of section 2 (plate OCR)."""
    run_id: str
    plates: list
    elapsed: float


@dataclass
class OcrStats:
    """Counters describing what happened during plate reading (for the UI)."""
    crops: int = 0
    errors: int = 0
    first_error: str | None = None
    plates_found: int = 0
    valid_reads: int = 0
    shapes: list = field(default_factory=list)
