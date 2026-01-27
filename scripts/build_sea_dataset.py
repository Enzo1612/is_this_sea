import zipfile
import pathlib

SEA_CATEGORIES = {
    "ocean",
    "wave",
    "underwater/ocean_deep",
    "beach",
    "coast",
    "harbor",
    "pier",
    "lagoon",
    "islet",
}

NON_SEA_CATEGORIES = {
    "lake/natural",
    "river",
    "pond",
    "waterfall",
    "fishpond",
    "moat/water",
    "swimming_hole",
    "desert/sand",
    "valley",
    "mountain",
    "mountain_snowy",
    "tundra",
    "sky",
    "snowfield",
    "iceberg",
    "ice_floe",
}