"""Plain data containers passed between the vision layer and the UI.

No heavy imports here, so every layer can use them.
"""
from dataclasses import dataclass, field


@dataclass
class VehicleCrops:
    """One counted vehicle and its best crops (used later for plate reading)."""
    vehicle_id: int                 # image: 1, 2, 3...  video: the tracker id
    cls: str                        # "car", "van", "bus"
    crops: list = field(default_factory=list)   # JPEG bytes, largest first


@dataclass
class DetectionResult:
    """Output of section 1 (YOLO)."""
    kind: str                       # "image" or "video"
    # annotated RGB array (image) or mp4 bytes (video)
    media: object
    counts: dict                    # {class name: number of vehicles}
    density: int                    # vehicles used for the density level
    # list[VehicleCrops], one per counted vehicle
    vehicles: list
    frames: int = 0                 # video only: frames processed
    h264: bool = True               # video only: playable in the browser?
    tracked: bool = True            # video only: unique vehicles via tracking?
    # filled in by the app after the run
    run_id: str = ""                # new value on every detection run
    params: tuple = ()              # inputs used; the result is hidden if they change
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
    run_id: str                     # the DetectionResult it was computed from
    plates: list                    # list[PlateReading]
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
