"""Document unit conversions. Canvas space is 1 unit = 1 px."""

UNITS = ("PX", "PT", "MM", "CM", "IN")


def units_per_px(unit: str, dpi: float = 96.0) -> float:
    """How many ``unit`` correspond to one canvas px at ``dpi``."""
    inch_per_px = 1.0 / float(dpi)
    return {
        "PX": 1.0,
        "PT": inch_per_px * 72.0,
        "MM": inch_per_px * 25.4,
        "CM": inch_per_px * 2.54,
        "IN": inch_per_px,
    }[unit]


def px_to_unit(value_px: float, unit: str, dpi: float = 96.0) -> float:
    return value_px * units_per_px(unit, dpi)


def unit_to_px(value: float, unit: str, dpi: float = 96.0) -> float:
    return value / units_per_px(unit, dpi)
