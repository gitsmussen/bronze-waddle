#!/usr/bin/env python3

import json
import logging
import math
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import (
    HTTPRedirectHandler,
    Request,
    build_opener,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))
LOCATIONS_FILE = Path(
    os.environ.get("LOCATIONS_FILE", BASE_DIR / "locations.jsonl")
)

API_URL = "https://api.met.no/weatherapi/locationforecast/2.0/complete"
USER_AGENT = (
    "bronze-waddle/1.0 "
    "(+https://github.com/gitsmussen/bronze-waddle; "
    "contact: git@smussen.no)"
)

REQUEST_TIMEOUT = 30
REQUEST_GAP_SECONDS = 1
MAX_RESPONSE_BYTES = 120 * 1024
SCHEMA_VERSION = 1

UTC_TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class NoRedirectHandler(HTTPRedirectHandler):
    """Reject HTTP redirects instead of following them."""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


HTTP_OPENER = build_opener(NoRedirectHandler())


def utc_timestamp() -> str:
    """Return the current UTC time in ISO 8601 Z format."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def parse_utc_timestamp(value: object) -> datetime:
    """Parse the exact UTC timestamp format used by MET Norway."""
    if not isinstance(value, str) or not UTC_TIMESTAMP_PATTERN.fullmatch(value):
        raise ValueError(f"Expected a UTC timestamp ending in Z, got {value!r}")

    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise ValueError(f"Invalid UTC timestamp: {value!r}") from error


def append_status(
    status: str,
    failures: list[dict[str, str]],
    failure_types: list[str],
) -> None:
    """Append one JSON status record for this run."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "time": utc_timestamp(),
        "status": status,
        "failure_types": failure_types,
        "failures": failures,
    }

    path = DATA_DIR / "log.jsonl"
    payload = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"

    with path.open("a", encoding="utf-8") as file:
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())


def load_locations(path: Path) -> list[dict[str, Any]]:
    """Load and validate one JSON location object per non-empty line."""
    locations: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                location = json.loads(line)
                if not isinstance(location, dict):
                    raise ValueError("line must contain a JSON object")

                location_id = location["id"]
                name = location["name"]
                lat = float(location["lat"])
                lon = float(location["lon"])
                archive = location.get("archive", False)
                altitude = location["altitude"]

                if isinstance(altitude, bool) or not isinstance(altitude, int):
                    raise ValueError("altitude must be an integer number of meters")

                if not isinstance(location_id, str) or not re.fullmatch(
                    r"[A-Za-z0-9_-]+", location_id
                ):
                    raise ValueError(
                        "id must contain only letters, digits, _ or -"
                    )

                if not isinstance(name, str) or not name.strip():
                    raise ValueError("name must be a non-empty string")

                if location_id in seen_ids:
                    raise ValueError(f"duplicate id {location_id!r}")

                if not math.isfinite(lat) or not -90 <= lat <= 90:
                    raise ValueError("latitude is invalid or out of range")

                if not math.isfinite(lon) or not -180 <= lon <= 180:
                    raise ValueError("longitude is invalid or out of range")

                if not math.isfinite(altitude):
                    raise ValueError("altitude must be finite")

                if not isinstance(archive, bool):
                    raise ValueError("archive must be true or false")

            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error

            seen_ids.add(location_id)
            locations.append(
                {
                    "id": location_id,
                    "name": name.strip(),
                    "lat": lat,
                    "lon": lon,
                    "altitude": altitude,
                    "archive": archive,
                }
            )

    if not locations:
        raise ValueError(f"No locations found in {path}")

    return locations


def validate_forecast(forecast: object) -> dict[str, Any]:
    """Validate the API response shape needed by all saved forecast files."""
    if not isinstance(forecast, dict):
        raise ValueError("API response is not a JSON object")

    properties = forecast.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("API response is missing properties")

    meta = properties.get("meta")
    units = meta.get("units") if isinstance(meta, dict) else None
    if not isinstance(units, dict) or not all(
        isinstance(key, str) and isinstance(unit, str)
        for key, unit in units.items()
    ):
        raise ValueError("API response is missing a valid properties.meta.units")

    timeseries = properties.get("timeseries")
    if not isinstance(timeseries, list) or not timeseries:
        raise ValueError("API response contains no forecast timeseries")

    first = timeseries[0]
    if not isinstance(first, dict):
        raise ValueError("First timeseries entry is not an object")

    parse_utc_timestamp(first.get("time"))

    data = first.get("data")
    instant = data.get("instant") if isinstance(data, dict) else None
    details = instant.get("details") if isinstance(instant, dict) else None
    if not isinstance(details, dict):
        raise ValueError("First timeseries entry has no instant.details object")

    if not all(isinstance(key, str) for key in details):
        raise ValueError("instant.details contains a non-string key")

    return forecast


def fetch_forecast(location: dict[str, Any]) -> dict[str, Any]:
    """Fetch one location once; redirects and failed requests are not retried."""

    """Looks like MET api does not accept decimal places on altitude, turning float>int"""
    altitude = location["altitude"]
    if altitude.is_integer():
        altitude = int(altitude)


    query = urlencode(
        {
            "altitude": location["altitude"],
            "lat": location["lat"],
            "lon": location["lon"],
        }
    )

    request = Request(
        f"{API_URL}?{query}",
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )

    with HTTP_OPENER.open(request, timeout=REQUEST_TIMEOUT) as response:
        payload = response.read(MAX_RESPONSE_BYTES + 1)

    if len(payload) > MAX_RESPONSE_BYTES:
        raise ValueError(
            f"Response exceeded the {MAX_RESPONSE_BYTES}-byte limit"
        )

    try:
        decoded = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"Response is not valid UTF-8 JSON: {error}") from error

    return validate_forecast(decoded)


def forecasts_are_equal(path: Path, forecast: dict[str, Any]) -> bool:
    """Compare parsed JSON, ignoring formatting and object key ordering."""
    if not path.exists():
        return False

    try:
        with path.open("r", encoding="utf-8") as file:
            existing = json.load(file)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False

    return existing == forecast


def replace_forecast(path: Path, forecast: dict[str, Any]) -> None:
    """Atomically replace a forecast after a successful validated fetch."""
    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temp_path = Path(file.name)
            json.dump(forecast, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        os.replace(temp_path, path)

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def get_first_sample(
    forecast: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    properties = forecast["properties"]
    first = properties["timeseries"][0]
    details = first["data"]["instant"]["details"]
    units = properties["meta"]["units"]

    return first["time"], details, units


def archive_metadata(
    location: dict[str, Any],
    units: dict[str, Any],
    values: dict[str, Any],
) -> dict[str, Any]:
    """Construct the self-describing first JSONL line."""
    return {
        "type": "metadata",
        "schema": SCHEMA_VERSION,
        "id": location["id"],
        "name": location["name"],
        "lat": location["lat"],
        "lon": location["lon"],
        "altitude": location["altitude"],
        "units": units,
        "value_fields": sorted(values),
    }


def read_archive_tail(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Read and validate metadata and the last sample without loading all history."""
    with path.open("rb") as file:
        file.seek(0, os.SEEK_END)
        size = file.tell()

        if size == 0:
            raise ValueError(f"{path.name} is empty")

        file.seek(0)
        metadata_line = file.readline()

        if not metadata_line.endswith(b"\n"):
            raise ValueError(f"{path.name}: metadata line is incomplete")

        try:
            metadata = json.loads(metadata_line)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(f"{path.name}: invalid metadata JSON") from error

        if not isinstance(metadata, dict) or metadata.get("type") != "metadata":
            raise ValueError(f"{path.name}: first line must be metadata")

        if file.tell() == size:
            return metadata, None

        file.seek(-1, os.SEEK_END)
        if file.read(1) != b"\n":
            raise ValueError(f"{path.name}: file does not end with a newline")

        # Seek backward to the newline preceding the last JSONL record.
        position = size - 2
        while position >= 0:
            file.seek(position)
            if file.read(1) == b"\n":
                break
            position -= 1

        file.seek(position + 1)
        last_line = file.readline()

    if not last_line.strip():
        raise ValueError(f"{path.name}: last record is blank")

    try:
        last_record = json.loads(last_line)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{path.name}: last record is invalid JSON") from error

    if not isinstance(last_record, dict):
        raise ValueError(f"{path.name}: last record is not an object")

    parse_utc_timestamp(last_record.get("time"))

    values = last_record.get("values")
    if not isinstance(values, dict):
        raise ValueError(f"{path.name}: last record has no values object")

    return metadata, last_record


def validate_archive_metadata(
    metadata: dict[str, Any],
    location: dict[str, Any],
    units: dict[str, Any],
    values: dict[str, Any],
    path: Path,
) -> None:
    if metadata.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"{path.name}: unsupported archive schema")

    for key in ("id", "name", "lat", "lon", "altitude"):
        if key not in metadata or metadata[key] != location[key]:
            raise ValueError(
                f"{path.name}: location metadata mismatch for {key}"
            )

    if metadata.get("units") != units:
        raise ValueError(f"{path.name}: units changed; archive not updated")

    saved_fields = metadata.get("value_fields")
    if not isinstance(saved_fields, list) or not all(
        isinstance(field, str) for field in saved_fields
    ):
        raise ValueError(f"{path.name}: invalid value_fields metadata")

    if saved_fields != sorted(values):
        raise ValueError(
            f"{path.name}: archived value fields changed; archive not updated"
        )


def write_new_archive(
    path: Path,
    metadata: dict[str, Any],
    timestamp: str,
    values: dict[str, Any],
) -> None:
    """Create metadata and its first sample together, then atomically install."""
    metadata_line = json.dumps(
        metadata,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    sample_line = json.dumps(
        {"time": timestamp, "values": values},
        ensure_ascii=False,
        separators=(",", ":"),
    )

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temp_path = Path(file.name)
            file.write(metadata_line + "\n")
            file.write(sample_line + "\n")
            file.flush()
            os.fsync(file.fileno())

        os.replace(temp_path, path)

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def append_sample(
    path: Path,
    timestamp: str,
    values: dict[str, Any],
) -> None:
    """Append one sample, rolling back to the prior file size on write errors."""
    payload = (
        json.dumps(
            {"time": timestamp, "values": values},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")

    with path.open("r+b") as file:
        file.seek(0, os.SEEK_END)
        original_size = file.tell()

        if original_size:
            file.seek(-1, os.SEEK_END)
            if file.read(1) != b"\n":
                raise ValueError(f"{path.name}: file does not end with a newline")

        file.seek(original_size)
        try:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        except OSError:
            file.truncate(original_size)
            file.flush()
            os.fsync(file.fileno())
            raise


def update_history(
    path: Path,
    location: dict[str, Any],
    timestamp: str,
    values: dict[str, Any],
    units: dict[str, Any],
) -> str:
    """Create a history file or append a strictly newer sample."""
    new_time = parse_utc_timestamp(timestamp)
    metadata = archive_metadata(location, units, values)

    if not path.exists():
        write_new_archive(path, metadata, timestamp, values)
        return "created"

    saved_metadata, last_record = read_archive_tail(path)
    validate_archive_metadata(
        saved_metadata,
        location,
        units,
        values,
        path,
    )

    if last_record is None:
        # A metadata-only file is valid; add its first data row.
        append_sample(path, timestamp, values)
        return "appended"

    last_time = parse_utc_timestamp(last_record["time"])
    if new_time <= last_time:
        return "not newer"

    append_sample(path, timestamp, values)
    return "appended"


def make_status(
    location_count: int,
    failures: list[dict[str, str]],
) -> tuple[str, list[str]]:
    if not failures:
        return "FETCH TOTAL SUCCESS", []

    failed_ids = {failure["id"] for failure in failures}
    failure_types = sorted({failure["stage"] for failure in failures})
    type_label = " AND ".join(item.upper() for item in failure_types)

    if len(failed_ids) >= location_count:
        return f"FETCH TOTAL FAILURE ({type_label})", failure_types

    return (
        f"FETCH {len(failed_ids)} PROBLEMS ({type_label})",
        failure_types,
    )


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        locations = load_locations(LOCATIONS_FILE)
    except (OSError, ValueError) as error:
        logger.error("Could not load locations: %s", error)
        try:
            append_status(
                "FETCH TOTAL FAILURE (CONFIGURATION)",
                [{"id": "*", "stage": "configuration", "reason": str(error)}],
                ["configuration"],
            )
        except OSError as log_error:
            logger.error("Could not append status log: %s", log_error)
        return 1

    failures: list[dict[str, str]] = []

    for index, location in enumerate(locations):
        if index:
            time.sleep(REQUEST_GAP_SECONDS)

        location_id = location["id"]
        forecast_path = DATA_DIR / f"{location_id}.json"

        try:
            forecast = fetch_forecast(location)
        except (
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
            ValueError,
        ) as error:
            reason = f"{type(error).__name__}: {error}"
            logger.error("%s: forecast failed: %s", location_id, reason)
            failures.append(
                {"id": location_id, "stage": "forecast", "reason": reason}
            )
            continue

        try:
            if forecasts_are_equal(forecast_path, forecast):
                logger.info("%s: forecast unchanged", forecast_path.name)
            else:
                replace_forecast(forecast_path, forecast)
                logger.info("%s: forecast updated", forecast_path.name)
        except OSError as error:
            reason = f"{type(error).__name__}: {error}"
            logger.error("%s: saving forecast failed: %s", location_id, reason)
            failures.append(
                {"id": location_id, "stage": "forecast", "reason": reason}
            )
            # Do not archive data if the forecast could not be saved.
            continue

        if not location["archive"]:
            continue

        try:
            timestamp, values, units = get_first_sample(forecast)
            history_path = DATA_DIR / f"hist.{location_id}.jsonl"
            result = update_history(
                history_path,
                location,
                timestamp,
                values,
                units,
            )
            logger.info("%s: history %s", history_path.name, result)
        except (OSError, ValueError, KeyError, TypeError) as error:
            reason = f"{type(error).__name__}: {error}"
            logger.error("%s: archive failed: %s", location_id, reason)
            failures.append(
                {"id": location_id, "stage": "archive", "reason": reason}
            )

    status, failure_types = make_status(len(locations), failures)
    logger.info("Run result: %s", status)

    try:
        append_status(status, failures, failure_types)
    except OSError as error:
        logger.error("Could not append status log: %s", error)
        return 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())