from datetime import timezone

import pytest

from kindex.temporal import (
    normalize_datetime,
    normalize_iso_datetime,
    normalize_time_range,
    parse_iso_datetime,
)


def test_parse_iso_datetime_treats_naive_values_as_utc():
    parsed = parse_iso_datetime("2030-01-02T03:04:05.123456")

    assert parsed.tzinfo is timezone.utc
    assert parsed.isoformat() == "2030-01-02T03:04:05.123456+00:00"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2030-01-02T03:04:05.123499Z", "2030-01-02T03:04:05.123Z"),
        ("2030-01-02T03:04:05.123500Z", "2030-01-02T03:04:05.124Z"),
    ],
)
def test_normalize_iso_datetime_rounds_sub_milliseconds_half_up(value, expected):
    assert normalize_iso_datetime(value) == expected


def test_normalize_iso_datetime_converts_offsets():
    assert normalize_iso_datetime("2030-01-02T03:04:05.123456+02:30") == (
        "2030-01-02T00:34:05.123Z"
    )


def test_normalize_iso_datetime_carries_rounding_into_next_second():
    assert normalize_iso_datetime("2030-01-02T03:04:05.999500Z") == (
        "2030-01-02T03:04:06.000Z"
    )


def test_normalize_datetime_returns_aware_utc_milliseconds():
    normalized = normalize_datetime(
        parse_iso_datetime("2030-01-02T03:04:05.123500+02:30")
    )

    assert normalized.isoformat() == "2030-01-02T00:34:05.124000+00:00"


def test_parse_iso_datetime_error_names_field():
    with pytest.raises(ValueError, match="since must be an ISO date or timestamp"):
        parse_iso_datetime("invalid", "since")


def test_normalize_time_range_compares_instants_not_wire_values():
    since, until = normalize_time_range(
        "2030-01-02T01:00:00+01:00",
        "2030-01-02T00:00:00Z",
    )

    assert since == until == "2030-01-02T00:00:00.000Z"


def test_normalize_time_range_accepts_boundaries_equal_after_normalization():
    since, until = normalize_time_range(
        "2030-01-02T00:00:00.000499Z",
        "2030-01-02T00:00:00.000001Z",
    )

    assert since == until == "2030-01-02T00:00:00.000Z"


def test_normalize_time_range_rejects_boundaries_inverted_after_normalization():
    with pytest.raises(ValueError, match="since must be earlier than or equal to until"):
        normalize_time_range(
            "2030-01-02T00:00:00.001500Z",
            "2030-01-02T00:00:00.001499Z",
        )
