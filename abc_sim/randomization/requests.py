"""Reset request models and task-local sampling exceptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DishRackResetRequest:
    """Optional reset controls for the dishrack randomizer."""

    plate_variant: str | None = None
    plate_count: int | None = None
    dish_rack_variant: str | None = None
    cycle_plate: int = 0
    cycle_dish_rack: int = 0
    randomize_variants: bool | None = None
    randomize_scales: bool | None = None
    randomize_arm_bases: bool = False

    @classmethod
    def from_value(cls, value: Any | None) -> "DishRackResetRequest":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError(
                "DishRack reset request must be a dict or DishRackResetRequest, "
                f"got {type(value).__name__}"
            )

        def _optional_str(key: str) -> str | None:
            raw = value.get(key)
            if raw is None:
                return None
            return str(raw)

        def _int_value(key: str) -> int:
            raw = value.get(key, 0)
            if raw is None:
                return 0
            return int(raw)

        randomize_variants = value.get("randomize_variants")
        if randomize_variants is not None:
            randomize_variants = bool(randomize_variants)

        randomize_scales = value.get("randomize_scales")
        if randomize_scales is not None:
            randomize_scales = bool(randomize_scales)

        return cls(
            plate_variant=_optional_str("plate_variant"),
            plate_count=value.get("plate_count"),
            dish_rack_variant=_optional_str("dish_rack_variant"),
            cycle_plate=_int_value("cycle_plate"),
            cycle_dish_rack=_int_value("cycle_dish_rack"),
            randomize_variants=randomize_variants,
            randomize_scales=randomize_scales,
            randomize_arm_bases=bool(value.get("randomize_arm_bases", False)),
        )


@dataclass(frozen=True)
class MugResetRequest:
    """Optional reset controls for mug asset randomization tasks."""

    mug_variant: str | None = None
    mug_variants: tuple[str, ...] | None = None
    mug_variant_pool: tuple[str, ...] | None = None
    mug_variant_combos: tuple[tuple[str, ...], ...] | None = None
    mug_count: int | None = None
    cycle_mug: int = 0
    randomize_variants: bool | None = None
    randomize_scales: bool | None = None

    @classmethod
    def from_value(cls, value: Any | None) -> "MugResetRequest":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError(
                "Mug reset request must be a dict or MugResetRequest, "
                f"got {type(value).__name__}"
            )

        raw_variant = value.get("mug_variant")
        raw_variants = value.get("mug_variants")
        if raw_variants is None:
            mug_variants = None
        elif isinstance(raw_variants, str):
            mug_variants = (raw_variants,)
        else:
            mug_variants = tuple(str(variant) for variant in raw_variants)

        raw_variant_pool = value.get("mug_variant_pool")
        if raw_variant_pool is None:
            mug_variant_pool = None
        elif isinstance(raw_variant_pool, str):
            mug_variant_pool = tuple(
                item.strip() for item in raw_variant_pool.split(",") if item.strip()
            )
        else:
            mug_variant_pool = tuple(str(variant) for variant in raw_variant_pool)

        raw_variant_combos = value.get("mug_variant_combos")
        if raw_variant_combos is None:
            mug_variant_combos = None
        elif isinstance(raw_variant_combos, str):
            mug_variant_combos = tuple(
                tuple(item.strip() for item in combo.split(",") if item.strip())
                for combo in raw_variant_combos.split(";")
                if combo.strip()
            )
        else:
            combos: list[tuple[str, ...]] = []
            for raw_combo in raw_variant_combos:
                if isinstance(raw_combo, str):
                    combo = tuple(item.strip() for item in raw_combo.split(",") if item.strip())
                else:
                    combo = tuple(str(variant) for variant in raw_combo)
                combos.append(combo)
            mug_variant_combos = tuple(combos)

        raw_count = value.get("mug_count")
        raw_cycle = value.get("cycle_mug", 0)
        randomize_variants = value.get("randomize_variants")
        if randomize_variants is not None:
            randomize_variants = bool(randomize_variants)

        randomize_scales = value.get("randomize_scales")
        if randomize_scales is not None:
            randomize_scales = bool(randomize_scales)

        return cls(
            mug_variant=None if raw_variant is None else str(raw_variant),
            mug_variants=mug_variants,
            mug_variant_pool=mug_variant_pool,
            mug_variant_combos=mug_variant_combos,
            mug_count=None if raw_count is None else int(raw_count),
            cycle_mug=0 if raw_cycle is None else int(raw_cycle),
            randomize_variants=randomize_variants,
            randomize_scales=randomize_scales,
        )


@dataclass(frozen=True)
class WaterBottleResetRequest:
    """Optional reset controls for the RoboCasa water-bottle randomizer."""

    bottle_variant: str | None = None
    bottle_variants: tuple[str, ...] | None = None
    bottle_count: int | None = None
    cycle_bottle: int = 0
    randomize_variants: bool | None = None
    randomize_scales: bool | None = None

    @classmethod
    def from_value(cls, value: Any | None) -> "WaterBottleResetRequest":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError(
                "WaterBottle reset request must be a dict or WaterBottleResetRequest, "
                f"got {type(value).__name__}"
            )

        raw_variant = value.get("bottle_variant")
        raw_variants = value.get("bottle_variants")
        if raw_variants is None:
            bottle_variants = None
        elif isinstance(raw_variants, str):
            bottle_variants = (raw_variants,)
        else:
            bottle_variants = tuple(str(variant) for variant in raw_variants)

        randomize_variants = value.get("randomize_variants")
        if randomize_variants is not None:
            randomize_variants = bool(randomize_variants)

        randomize_scales = value.get("randomize_scales")
        if randomize_scales is not None:
            randomize_scales = bool(randomize_scales)

        raw_count = value.get("bottle_count")
        raw_cycle = value.get("cycle_bottle", 0)
        return cls(
            bottle_variant=None if raw_variant is None else str(raw_variant),
            bottle_variants=bottle_variants,
            bottle_count=None if raw_count is None else int(raw_count),
            cycle_bottle=0 if raw_cycle is None else int(raw_cycle),
            randomize_variants=randomize_variants,
            randomize_scales=randomize_scales,
        )


@dataclass(frozen=True)
class ChessResetRequest:
    """Optional reset controls for the chess setup randomizer."""

    scenario: str | None = None
    target_count: int | None = None
    color_mode: str | None = None
    tin_variant: str | None = None
    cycle_scenario: int = 0
    cycle_color_mode: int = 0
    cycle_tin: int = 0
    randomize_variants: bool | None = None
    randomize_scales: bool | None = None

    @classmethod
    def from_value(cls, value: Any | None) -> "ChessResetRequest":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError(
                "Chess reset request must be a dict or ChessResetRequest, "
                f"got {type(value).__name__}"
            )

        scenario = value.get("scenario")
        color_mode = value.get("color_mode")
        tin_variant = value.get("tin_variant")
        target_count = value.get("target_count")
        raw_cycle_scenario = value.get("cycle_scenario", 0)
        raw_cycle_color_mode = value.get("cycle_color_mode", 0)
        raw_cycle_tin = value.get("cycle_tin", 0)
        randomize_variants = value.get("randomize_variants")
        if randomize_variants is not None:
            randomize_variants = bool(randomize_variants)
        randomize_scales = value.get("randomize_scales")
        if randomize_scales is not None:
            randomize_scales = bool(randomize_scales)

        return cls(
            scenario=None if scenario is None else str(scenario),
            target_count=None if target_count is None else int(target_count),
            color_mode=None if color_mode is None else str(color_mode),
            tin_variant=None if tin_variant is None else str(tin_variant),
            cycle_scenario=0 if raw_cycle_scenario is None else int(raw_cycle_scenario),
            cycle_color_mode=0 if raw_cycle_color_mode is None else int(raw_cycle_color_mode),
            cycle_tin=0 if raw_cycle_tin is None else int(raw_cycle_tin),
            randomize_variants=randomize_variants,
            randomize_scales=randomize_scales,
        )


@dataclass(frozen=True)
class SweepResetRequest:
    """Optional reset controls for the sweep randomizer."""

    trash_count: int | None = None
    randomize_scales: bool | None = None

    @classmethod
    def from_value(cls, value: Any | None) -> "SweepResetRequest":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError(
                "Sweep reset request must be a dict or SweepResetRequest, "
                f"got {type(value).__name__}"
            )
        raw_trash_count = value.get("trash_count")
        randomize_scales = value.get("randomize_scales")
        if randomize_scales is not None:
            randomize_scales = bool(randomize_scales)
        return cls(
            trash_count=None if raw_trash_count is None else int(raw_trash_count),
            randomize_scales=randomize_scales,
        )


class _SweepPlacementFailure(RuntimeError):
    """Internal signal that one sweep placement attempt should be retried."""


class _WaterBottlePlacementFailure(RuntimeError):
    """Internal signal that one water-bottle placement attempt should be retried."""


class _ChessPlacementFailure(RuntimeError):
    """Internal signal that one chess placement attempt should be retried."""


__all__ = [name for name in globals() if not name.startswith('__')]
