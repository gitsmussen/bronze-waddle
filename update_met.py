#!/usr/bin/env python3

import json
import logging
import math
import os
import re
import sys
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import Message
from email.utils import parsedate_to_datetime
from enum import Enum
from http.client import HTTPException
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import (
    HTTPRedirectHandler,
    Request,
    build_opener,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))

# A local path OR an https:// URL (for example a raw.githubusercontent.com link).
LOCATIONS_FILE = os.environ.get("LOCATIONS_FILE", str(BASE_DIR / "locations.jsonl"))
# Last locations list that downloaded and validated; used if the URL fails later.
LOCATIONS_CACHE = DATA_DIR / "locations.cache.jsonl"
ALLOWED_LOCATION_URL_PREFIXES = ("https://",)
MAX_LOCATIONS = 200
MAX_LOCATIONS_BYTES = 256 * 1024

# Remembers Last-Modified / Expires per forecast URL between runs.
STATE_FILE = DATA_DIR / "fetch-state.json"

API_URL = "https://api.met.no/weatherapi/locationforecast/2.0/complete"
USER_AGENT = (
    "bronze-waddle/1.0 "
    "(+https://github.com/gitsmussen/bronze-waddle; "
    "contact: git@smussen.no)"
)

REQUEST_TIMEOUT = 30
REQUEST_GAP_SECONDS = 1
MAX_ATTEMPTS = 2                # the first try plus a single retry
RETRY_DELAY = 10                # seconds before the retry
MAX_RETRY_WAIT = 60             # give up if MET asks us to wait longer than this
MAX_CONSECUTIVE_FAILURES = 3    # stop the run if MET seems down
MAX_RESPONSE_BYTES = 512 * 1024
SCHEMA_VERSION = 1

# MET asks for at most 4 decimals in coordinates.
COORDINATE_DECIMALS = 4

UTC_TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
)

# Stages that count towards MAX_CONSECUTIVE_FAILURES.
FETCH_STAGES = {"forecast", "unexpected"}

# Files in DATA_DIR that are not location files (used by the orphan check).
NON_LOCATION_JSON_FILES = {STATE_FILE.name}


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


class BlockedError(Exception):
    """MET refused us (403, or 429 that does not clear); stop the whole run."""


class FetchStatus(Enum):
    DOWNLOADED = "downloaded"
    NOT_MODIFIED = "not modified"


@dataclass
class FetchResult:
    status: FetchStatus
    forecast: dict[str, Any] | None = None
    validators: dict[str, str] | None = None


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


def write_text_atomically(path: Path, text: str) -> None:
    """Write a text file through a same-directory temp file."""
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
            file.write(text)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temp_path, path)

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def append_status(
    status: str,
    failures: list[dict[str, str]],
    failure_types: list[str],
    warnings: list[str] | None = None,
) -> None:
    """Append one JSON status record for this run."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "time": utc_timestamp(),
        "status": status,
        "failure_types": failure_types,
        "failures": failures,
    }
    if warnings:
        record["warnings"] = warnings

    path = DATA_DIR / "log.jsonl"
    payload = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"

    with path.open("a", encoding="utf-8") as file:
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())

# --------------------------------------------------
# Fetch state (Last-Modified / Expires)
# --------------------------------------------------

# Keyed by forecast URL: {"last_modified": "...", "expires": "...", "etag": "..."}
FETCH_STATE: dict[str, dict[str, str]] = {}


def load_fetch_state() -> dict[str, dict[str, str]]:
    """Read saved validators; a missing or broken file just means 'fetch everything'."""
    try:
        with STATE_FILE.open("r", encoding="utf-8") as file:
            raw: object = json.load(file)
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as error:
        logger.warning(
            "Could not read %s (%s); fetching everything fresh",
            STATE_FILE.name,
            error,
        )
        return {}

    if not isinstance(raw, dict):
        return {}

    state: dict[str, dict[str, str]] = {}
    for url, saved in raw.items():
        if isinstance(url, str) and isinstance(saved, dict):
            state[url] = {
                key: value
                for key, value in saved.items()
                if isinstance(key, str) and isinstance(value, str)
            }
    return state


def save_fetch_state() -> None:
    text = json.dumps(FETCH_STATE, indent=2, sort_keys=True) + "\n"
    write_text_atomically(STATE_FILE, text)


def extract_validators(headers: Message) -> dict[str, str]:
    validators: dict[str, str] = {}

    last_modified = headers.get("Last-Modified")
    if last_modified:
        validators["last_modified"] = last_modified

    expires = headers.get("Expires")
    if expires:
        validators["expires"] = expires

    etag = headers.get("ETag")
    if etag:
        validators["etag"] = etag

    return validators


def remember_validators(url: str, validators: dict[str, str] | None) -> None:
    """After a full download: replace what we remembered (call only after saving)."""
    if validators:
        FETCH_STATE[url] = validators
    else:
        FETCH_STATE.pop(url, None)


def merge_validators(url: str, validators: dict[str, str] | None) -> None:
    """After a 304: keep Last-Modified, but take over the new Expires."""
    if validators:
        FETCH_STATE[url] = {**FETCH_STATE.get(url, {}), **validators}


def conditional_headers(url: str, local_copy: Path) -> dict[str, str]:
    """Ask the server to answer 304 if nothing changed.

    Only sent when we really have the forecast on disk. MET wants the
    If-Modified-Since value to be exactly the previous Last-Modified value,
    which is why it is stored and sent back unchanged.
    """
    if not local_copy.exists() or local_copy.stat().st_size == 0:
        return {}

    saved = FETCH_STATE.get(url, {})
    headers: dict[str, str] = {}
    if "last_modified" in saved:
        headers["If-Modified-Since"] = saved["last_modified"]
    if "etag" in saved:
        headers["If-None-Match"] = saved["etag"]
    return headers


def expires_in_future(saved: dict[str, str]) -> bool:
    """True while MET's Expires time has not passed (no new request allowed)."""
    value = saved.get("expires")
    if not value:
        return False

    try:
        expires = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return False

    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)

    return datetime.now(timezone.utc) < expires

# --------------------------------------------------
# Locations
# --------------------------------------------------

def parse_locations(lines: Iterable[str], source: str) -> list[dict[str, Any]]:
    """Validate one JSON location object per non-empty line."""
    locations: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for line_number, line in enumerate(lines, start=1):
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

            # Compared ignoring case: Oslo and oslo would collide on Windows.
            if location_id.lower() in seen_ids:
                raise ValueError(
                    f"duplicate id {location_id!r} (ids are compared ignoring case)"
                )

            if not math.isfinite(lat) or not -90 <= lat <= 90:
                raise ValueError("latitude is invalid or out of range")

            if not math.isfinite(lon) or not -180 <= lon <= 180:
                raise ValueError("longitude is invalid or out of range")

            if not isinstance(archive, bool):
                raise ValueError("archive must be true or false")

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{source}:{line_number}: {error}") from error

        rounded_lat = round(lat, COORDINATE_DECIMALS)
        rounded_lon = round(lon, COORDINATE_DECIMALS)
        if (rounded_lat, rounded_lon) != (lat, lon):
            logger.warning(
                "%s: coordinates rounded to %d decimals (required by MET)",
                location_id,
                COORDINATE_DECIMALS,
            )

        seen_ids.add(location_id.lower())
        locations.append(
            {
                "id": location_id,
                "name": name.strip(),
                "lat": rounded_lat,
                "lon": rounded_lon,
                "altitude": altitude,
                "archive": archive,
            }
        )

        if len(locations) > MAX_LOCATIONS:
            raise ValueError(
                f"{source}: more than {MAX_LOCATIONS} locations; refusing to use it"
            )

    if not locations:
        raise ValueError(f"No locations found in {source}")

    return locations


def load_locations_file(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as file:
        return parse_locations(file, str(path))


def is_url(value: str) -> bool:
    return value.lower().startswith(("http://", "https://"))


def download_locations(url: str) -> str:
    """Download the locations list (redirects refused, size limited)."""
    if not url.lower().startswith(ALLOWED_LOCATION_URL_PREFIXES):
        raise ValueError("LOCATIONS_FILE URL must start with https://")

    request = Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/plain, */*"},
    )

    with HTTP_OPENER.open(request, timeout=REQUEST_TIMEOUT) as response:
        payload = response.read(MAX_LOCATIONS_BYTES + 1)

    if len(payload) > MAX_LOCATIONS_BYTES:
        raise ValueError(
            f"Locations file exceeded the {MAX_LOCATIONS_BYTES}-byte limit"
        )

    return payload.decode("utf-8-sig")


def get_locations() -> tuple[list[dict[str, Any]], list[str]]:
    """Return (locations, warnings).

    With a URL, the downloaded list is validated and saved as the last good
    copy. If the download or validation fails later, that copy is used so a
    broken edit or a network problem does not stop all forecasts.
    """
    warnings: list[str] = []

    if not is_url(LOCATIONS_FILE):
        return load_locations_file(Path(LOCATIONS_FILE)), warnings

    try:
        text = download_locations(LOCATIONS_FILE)
        locations = parse_locations(text.splitlines(), LOCATIONS_FILE)

    except (OSError, HTTPException, ValueError) as error:
        reason = f"{type(error).__name__}: {error}"
        logger.error("Could not use locations from %s: %s", LOCATIONS_FILE, reason)

        try:
            cached = load_locations_file(LOCATIONS_CACHE)
        except (OSError, ValueError) as cache_error:
            raise ValueError(
                f"{reason}; no usable cached copy ({cache_error})"
            ) from error

        warning = f"Using last good locations copy: {reason}"
        logger.warning(warning)
        warnings.append(warning)
        return cached, warnings

    try:
        write_text_atomically(
            LOCATIONS_CACHE,
            text if text.endswith("\n") else text + "\n",
        )
    except OSError as error:
        logger.warning("Could not save %s: %s", LOCATIONS_CACHE.name, error)

    return locations, warnings


def log_orphaned_files(locations: list[dict[str, Any]]) -> None:
    """Warn about files of locations that are no longer in the list."""
    ids = {location["id"] for location in locations}

    try:
        entries = sorted(DATA_DIR.iterdir())
    except OSError:
        return

    for path in entries:
        name = path.name

        if name.startswith("hist.") and name.endswith(".jsonl"):
            file_id = name[len("hist."):-len(".jsonl")]
        elif name.endswith(".json") and name not in NON_LOCATION_JSON_FILES:
            file_id = name[: -len(".json")]
        else:
            continue

        if file_id not in ids:
            logger.warning("%s: not in the locations list (leftover?)", name)

# --------------------------------------------------
# Forecast download
# --------------------------------------------------

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


def build_url(location: dict[str, Any]) -> str:
    query = urlencode(
        {
            "altitude": location["altitude"],
            "lat": location["lat"],
            "lon": location["lon"],
        }
    )
    return f"{API_URL}?{query}"


_LAST_REQUEST = {"time": 0.0}


def wait_for_request_gap() -> None:
    """Keep REQUEST_GAP_SECONDS between requests (skipped locations cost nothing)."""
    wait = REQUEST_GAP_SECONDS - (time.monotonic() - _LAST_REQUEST["time"])
    if wait > 0:
        time.sleep(wait)
    _LAST_REQUEST["time"] = time.monotonic()


def get_retry_after(error: HTTPError) -> float | None:
    headers = getattr(error, "headers", None)
    value = headers.get("Retry-After") if headers is not None else None

    if not value:
        return None

    try:
        return max(0.0, float(value))
    except ValueError:
        pass

    try:
        retry_date = parsedate_to_datetime(value)
        if retry_date.tzinfo is None:
            retry_date = retry_date.replace(tzinfo=timezone.utc)
        return max(
            0.0,
            (retry_date - datetime.now(timezone.utc)).total_seconds(),
        )
    except (TypeError, ValueError, OverflowError):
        return None


def fetch_forecast(url: str, extra_headers: dict[str, str]) -> FetchResult:
    """Fetch one forecast; one retry for temporary problems, stop on a block.

    - 304: nothing changed
    - 403 or a 429 that does not clear: BlockedError (the run is stopped)
    - 5xx, timeouts, dropped connections: a single retry after RETRY_DELAY
    - everything else (404, bad JSON, too large ...): no retry
    """
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    headers.update(extra_headers)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        wait_for_request_gap()

        try:
            request = Request(url, headers=headers)

            with HTTP_OPENER.open(request, timeout=REQUEST_TIMEOUT) as response:
                validators = extract_validators(response.headers)
                payload = response.read(MAX_RESPONSE_BYTES + 1)

            if len(payload) > MAX_RESPONSE_BYTES:
                raise ValueError(
                    f"Response exceeded the {MAX_RESPONSE_BYTES}-byte limit"
                )

            try:
                decoded = json.loads(payload)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ValueError(
                    f"Response is not valid UTF-8 JSON: {error}"
                ) from error

            return FetchResult(
                FetchStatus.DOWNLOADED,
                validate_forecast(decoded),
                validators,
            )

        except HTTPError as error:
            reason = f"HTTP {error.code}"

            if error.code == 304:
                return FetchResult(
                    FetchStatus.NOT_MODIFIED,
                    validators=extract_validators(error.headers),
                )

            if error.code == 403:
                raise BlockedError(
                    "HTTP 403 Forbidden (blocked, or the User-Agent is not accepted)"
                ) from error

            if error.code == 429:
                retry_after = get_retry_after(error)
                too_long = retry_after is not None and retry_after > MAX_RETRY_WAIT
                if attempt >= MAX_ATTEMPTS or too_long:
                    raise BlockedError("HTTP 429 Too Many Requests") from error
                delay = max(float(RETRY_DELAY), retry_after or 0.0)

            elif error.code >= 500 and attempt < MAX_ATTEMPTS:
                delay = float(RETRY_DELAY)

            else:
                raise

        except (OSError, HTTPException) as error:
            # Timeouts and connection problems (HTTPError is handled above).
            reason = f"{type(error).__name__}: {error}"
            if attempt >= MAX_ATTEMPTS:
                raise
            delay = float(RETRY_DELAY)

        logger.warning("Request failed (%s); retrying in %.0f seconds", reason, delay)
        time.sleep(delay)

    raise RuntimeError("retry loop ended unexpectedly")

# --------------------------------------------------
# Forecast files
# --------------------------------------------------

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


def load_saved_forecast(path: Path) -> dict[str, Any]:
    """Read the forecast we saved earlier (used when no new download happened)."""
    with path.open("r", encoding="utf-8") as file:
        return validate_forecast(json.load(file))

# --------------------------------------------------
# History archive
# --------------------------------------------------

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
    """Refuse to append only if the data would no longer mean the same thing.

    Checked: schema, id, position and units of the fields in the new sample.
    Not checked: the display name, and which fields exist. MET adds fields to
    the complete product over time, so fields may appear or disappear; each row
    only contains the fields present at that time.
    """
    if metadata.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"{path.name}: unsupported archive schema")

    for key in ("id", "altitude"):
        if key not in metadata or metadata[key] != location[key]:
            raise ValueError(
                f"{path.name}: location metadata mismatch for {key}"
            )

    for key in ("lat", "lon"):
        saved = metadata.get(key)
        if (
            isinstance(saved, bool)
            or not isinstance(saved, (int, float))
            or round(saved, COORDINATE_DECIMALS) != location[key]
        ):
            raise ValueError(
                f"{path.name}: location metadata mismatch for {key}"
            )

    saved_units = metadata.get("units")
    if not isinstance(saved_units, dict):
        raise ValueError(f"{path.name}: invalid units metadata")

    for field in values:
        if field in saved_units and saved_units[field] != units.get(field):
            raise ValueError(
                f"{path.name}: unit of {field} changed "
                f"({saved_units[field]!r} -> {units.get(field)!r}); "
                "archive not updated"
            )

    saved_fields = metadata.get("value_fields")
    if not isinstance(saved_fields, list) or not all(
        isinstance(field, str) for field in saved_fields
    ):
        raise ValueError(f"{path.name}: invalid value_fields metadata")

    added = sorted(set(values) - set(saved_fields))
    missing = sorted(set(saved_fields) - set(values))
    if added or missing:
        logger.warning(
            "%s: fields differ from when the archive was created "
            "(new: %s, missing: %s)",
            path.name,
            ", ".join(added) or "none",
            ", ".join(missing) or "none",
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

    write_text_atomically(path, metadata_line + "\n" + sample_line + "\n")


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

# --------------------------------------------------
# One location
# --------------------------------------------------

def failure(location_id: str, stage: str, error: Exception | str) -> dict[str, str]:
    if isinstance(error, Exception):
        reason = f"{type(error).__name__}: {error}"
    else:
        reason = error
    logger.error("%s: %s failed: %s", location_id, stage, reason)
    return {"id": location_id, "stage": stage, "reason": reason}


def process_location(location: dict[str, Any]) -> list[dict[str, str]]:
    """Update one location and return its failures (empty list = all fine).

    Raises BlockedError if MET blocks us; the caller then stops the run.
    """
    location_id = location["id"]
    url = build_url(location)
    forecast_path = DATA_DIR / f"{location_id}.json"
    forecast: dict[str, Any] | None = None

    if forecast_path.exists() and expires_in_future(FETCH_STATE.get(url, {})):
        logger.info("%s: not expired yet; no request needed", location_id)
    else:
        try:
            result = fetch_forecast(url, conditional_headers(url, forecast_path))
        except (OSError, HTTPException, ValueError) as error:
            # HTTPError, URLError and TimeoutError are OSError subclasses.
            return [failure(location_id, "forecast", error)]

        if result.status is FetchStatus.NOT_MODIFIED:
            logger.info("%s: forecast not modified (HTTP 304)", location_id)
            merge_validators(url, result.validators)
        else:
            forecast = result.forecast
            if forecast is None:
                return [failure(location_id, "forecast", "no forecast in response")]

            try:
                if forecasts_are_equal(forecast_path, forecast):
                    logger.info("%s: forecast unchanged", forecast_path.name)
                else:
                    replace_forecast(forecast_path, forecast)
                    logger.info("%s: forecast updated", forecast_path.name)
            except OSError as error:
                # Do not archive data if the forecast could not be saved,
                # and do not remember validators for data we did not keep.
                return [failure(location_id, "forecast", error)]

            remember_validators(url, result.validators)

    if not location["archive"]:
        return []

    if forecast is None:
        # No new download: archive from the forecast we saved earlier.
        try:
            forecast = load_saved_forecast(forecast_path)
        except (OSError, ValueError) as error:
            # Forget the validators so the next run downloads it fresh.
            FETCH_STATE.pop(url, None)
            return [failure(location_id, "archive", f"saved forecast unreadable: {error}")]

    try:
        timestamp, values, units = get_first_sample(forecast)
        history_path = DATA_DIR / f"hist.{location_id}.jsonl"
        outcome = update_history(
            history_path,
            location,
            timestamp,
            values,
            units,
        )
        logger.info("%s: history %s", history_path.name, outcome)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return [failure(location_id, "archive", error)]

    return []

# --------------------------------------------------
# Run status
# --------------------------------------------------

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

    FETCH_STATE.clear()
    FETCH_STATE.update(load_fetch_state())

    try:
        locations, warnings = get_locations()
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

    log_orphaned_files(locations)

    failures: list[dict[str, str]] = []
    stop_reason: str | None = None
    consecutive_failures = 0

    for location in locations:
        location_id = location["id"]

        if stop_reason is not None:
            failures.append(
                {"id": location_id, "stage": "skipped", "reason": stop_reason}
            )
            continue

        try:
            location_failures = process_location(location)

        except BlockedError as error:
            stop_reason = f"run stopped: {error}"
            logger.error("%s: %s", location_id, stop_reason)
            failures.append(
                {"id": location_id, "stage": "forecast", "reason": str(error)}
            )
            continue

        except Exception as error:  # noqa: BLE001
            # One unexpected problem must not stop the other locations.
            logger.exception("%s: unexpected error", location_id)
            location_failures = [
                {
                    "id": location_id,
                    "stage": "unexpected",
                    "reason": f"{type(error).__name__}: {error}",
                }
            ]

        failures.extend(location_failures)

        if any(item["stage"] in FETCH_STAGES for item in location_failures):
            consecutive_failures += 1
        else:
            consecutive_failures = 0

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            stop_reason = (
                f"run stopped after {MAX_CONSECUTIVE_FAILURES} "
                "failed fetches in a row"
            )
            logger.error(stop_reason)

    try:
        save_fetch_state()
    except OSError as error:
        logger.error("Could not save %s: %s", STATE_FILE.name, error)

    status, failure_types = make_status(len(locations), failures)
    logger.info("Run result: %s", status)

    try:
        append_status(status, failures, failure_types, warnings)
    except OSError as error:
        logger.error("Could not append status log: %s", error)
        return 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
