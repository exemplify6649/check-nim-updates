#!/usr/bin/env python3
"""Track NVIDIA NIM catalog changes without turning transient API glitches into history.

Drop-in replacement for scripts/update_models.py.

Key differences from the original tracker:
- treats NVIDIA's 735790403 `created` value as an API placeholder, not a real model date;
- validates, retries and de-duplicates /v1/models responses;
- keeps a persistent state file so a model must be missing on multiple successful polls
  before it is recorded as removed;
- aborts suspicious mass catalog drops instead of committing destructive changes;
- refreshes metadata/state on every successful poll, even when IDs do not change;
- creates README change markers automatically when they are absent;
- distinguishes a genuinely new model from a previously removed model that reappears.

Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

API_URL = os.environ.get("NIM_MODELS_URL", "https://integrate.api.nvidia.com/v1/models")
DATA_DIR = os.environ.get("NIM_TRACKER_DATA_DIR", "data")
CSV_PATH = os.path.join(DATA_DIR, "models.csv")
JSONL_PATH = os.path.join(DATA_DIR, "changes_log.jsonl")
STATE_PATH = os.path.join(DATA_DIR, "model_state.json")
README_PATH = os.environ.get("NIM_TRACKER_README", "README.md")

MARKER_START = "<!-- CHANGES_START -->"
MARKER_END = "<!-- CHANGES_END -->"

# NVIDIA's catalog has historically returned this same value for many unrelated models.
# Keep the raw value for debugging, but do not present it as the model's creation date.
PLACEHOLDER_CREATED_VALUES = {735790403}

DEFAULT_TIMEOUT = float(os.environ.get("NIM_API_TIMEOUT", "20"))
DEFAULT_RETRIES = int(os.environ.get("NIM_API_RETRIES", "3"))
DEFAULT_REMOVE_CONFIRMATIONS = int(os.environ.get("NIM_REMOVE_CONFIRMATIONS", "2"))
DEFAULT_MIN_CATALOG_RATIO = float(os.environ.get("NIM_MIN_CATALOG_RATIO", "0.65"))

CSV_FIELDS = [
    "id",
    "owned_by",
    "created_date",
    "first_seen_utc",
    "last_seen_utc",
    "status",
    "consecutive_missing",
    "duplicate_count",
    "created",
    "object",
]


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def format_created_date(value: Any) -> str:
    """Return a real-looking UTC date, or an empty string for unknown/placeholder values."""
    ts = _to_int(value)
    if ts is None or ts <= 0 or ts in PLACEHOLDER_CREATED_VALUES:
        return ""

    try:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return ""

    # /v1/models `created` should describe a modern model. Reject obviously bogus dates.
    if dt.year < 2010:
        return ""
    return dt.strftime("%Y-%m-%d")


def display_created(value: Any) -> str:
    return format_created_date(value) or "unknown"


def fetch_models(
    api_key: str,
    api_url: str = API_URL,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> List[Dict[str, Any]]:
    """Fetch and validate NVIDIA's OpenAI-compatible model list."""
    retries = max(1, retries)
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        req = urllib.request.Request(
            api_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "Cache-Control": "no-cache",
                "User-Agent": "NIM-Tracker/2.0",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
                if resp.status != 200:
                    raise RuntimeError(f"NIM model API returned HTTP {resp.status}")

            payload = json.loads(body)
            if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
                raise RuntimeError("NIM model API returned an unexpected JSON schema")
            return payload["data"]

        except urllib.error.HTTPError as exc:
            last_error = exc
            retryable = exc.code == 429 or 500 <= exc.code <= 599
            if not retryable or attempt == retries:
                raise RuntimeError(f"NIM model API returned HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
            if attempt == retries:
                raise RuntimeError(f"Failed to fetch a valid NIM model catalog: {exc}") from exc

        time.sleep(min(2 ** (attempt - 1), 8))

    raise RuntimeError(f"Failed to fetch NIM model catalog: {last_error}")


def normalize_catalog(raw_models: Iterable[Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int], int]:
    """Validate and de-duplicate by model id.

    Returns (unique_models, duplicate_counts, invalid_row_count).
    """
    unique: Dict[str, Dict[str, Any]] = {}
    counts: Dict[str, int] = {}
    invalid = 0

    for raw in raw_models:
        if not isinstance(raw, dict):
            invalid += 1
            continue

        mid = raw.get("id")
        if not isinstance(mid, str) or not mid.strip():
            invalid += 1
            continue
        mid = mid.strip()
        counts[mid] = counts.get(mid, 0) + 1

        cleaned = {
            "id": mid,
            "owned_by": str(raw.get("owned_by") or "unknown"),
            "created": _to_int(raw.get("created")),
            "object": str(raw.get("object") or "model"),
        }

        if mid not in unique:
            unique[mid] = cleaned
        else:
            # Keep the first row, but fill any missing metadata from later duplicates.
            for key in ("owned_by", "created", "object"):
                if unique[mid].get(key) in (None, "", "unknown") and cleaned.get(key) not in (None, ""):
                    unique[mid][key] = cleaned[key]

    duplicates = {mid: count for mid, count in counts.items() if count > 1}
    return unique, duplicates, invalid


def empty_state() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "last_success_utc": "",
        "catalog": {},
        "models": {},
    }


def migrate_legacy_csv() -> Dict[str, Any]:
    """Seed state from the existing CSV so first_seen history is preserved."""
    state = empty_state()
    if not os.path.exists(CSV_PATH):
        return state

    with open(CSV_PATH, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            mid = (row.get("id") or "").strip()
            if not mid:
                continue
            created = _to_int(row.get("created"))
            first_seen = row.get("first_seen_utc") or ""
            state["models"][mid] = {
                "id": mid,
                "owned_by": row.get("owned_by") or "unknown",
                "created": created,
                "created_date": format_created_date(created),
                "object": row.get("object") or "model",
                "first_seen_utc": first_seen,
                "last_seen_utc": row.get("last_seen_utc") or first_seen,
                "missing_since_utc": "",
                "status": "listed",
                "consecutive_missing": 0,
                "duplicate_count": max(1, _to_int(row.get("duplicate_count")) or 1),
            }
    return state


def load_state() -> Tuple[Dict[str, Any], bool]:
    if not os.path.exists(STATE_PATH):
        return migrate_legacy_csv(), True

    with open(STATE_PATH, "r", encoding="utf-8") as f:
        state = json.load(f)
    if not isinstance(state, dict) or not isinstance(state.get("models"), dict):
        raise RuntimeError(f"Invalid tracker state in {STATE_PATH}")
    state.setdefault("schema_version", 1)
    state.setdefault("last_success_utc", "")
    state.setdefault("catalog", {})
    return state, False


def _metadata_snapshot(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "owned_by": row.get("owned_by") or "unknown",
        "created": _to_int(row.get("created")),
        "object": row.get("object") or "model",
    }


def compact_item(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "owned_by": row.get("owned_by", "unknown"),
        "created": display_created(row.get("created")),
    }


def apply_catalog(
    state: Dict[str, Any],
    current: Dict[str, Dict[str, Any]],
    duplicate_counts: Dict[str, int],
    now_utc: str,
    remove_confirmations: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Merge a successful catalog snapshot into persistent state."""
    remove_confirmations = max(1, remove_confirmations)
    models: Dict[str, Dict[str, Any]] = state.setdefault("models", {})
    events: Dict[str, List[Dict[str, Any]]] = {
        "added": [],
        "removed": [],
        "restored": [],
        "metadata_changed": [],
    }

    for mid, cur in current.items():
        previous = models.get(mid)
        if previous is None:
            row = {
                **cur,
                "created_date": format_created_date(cur.get("created")),
                "first_seen_utc": now_utc,
                "last_seen_utc": now_utc,
                "missing_since_utc": "",
                "status": "listed",
                "consecutive_missing": 0,
                "duplicate_count": duplicate_counts.get(mid, 1),
            }
            models[mid] = row
            events["added"].append(compact_item(row))
            continue

        old_status = previous.get("status", "listed")
        old_meta = _metadata_snapshot(previous)
        new_meta = _metadata_snapshot(cur)

        previous.update(cur)
        previous["created_date"] = format_created_date(cur.get("created"))
        previous["last_seen_utc"] = now_utc
        previous["missing_since_utc"] = ""
        previous["status"] = "listed"
        previous["consecutive_missing"] = 0
        previous["duplicate_count"] = duplicate_counts.get(mid, 1)
        previous.setdefault("first_seen_utc", now_utc)

        if old_status == "removed":
            events["restored"].append(compact_item(previous))

        changed_fields = [key for key in old_meta if old_meta[key] != new_meta[key]]
        if changed_fields:
            events["metadata_changed"].append(
                {
                    "id": mid,
                    "fields": changed_fields,
                    "before": old_meta,
                    "after": new_meta,
                }
            )

    current_ids = set(current)
    for mid, previous in list(models.items()):
        if mid in current_ids or previous.get("status") == "removed":
            continue

        miss_count = int(previous.get("consecutive_missing") or 0) + 1
        previous["consecutive_missing"] = miss_count
        if not previous.get("missing_since_utc"):
            previous["missing_since_utc"] = now_utc

        if miss_count >= remove_confirmations:
            previous["status"] = "removed"
            events["removed"].append(compact_item(previous))
        else:
            previous["status"] = "missing_pending"

    return events


def validate_catalog_size(state: Dict[str, Any], current_count: int, min_ratio: float) -> None:
    """Reject suspiciously incomplete snapshots before they can advance missing counters."""
    if min_ratio <= 0:
        return

    previous_visible = sum(
        1
        for row in state.get("models", {}).values()
        if row.get("status", "listed") in {"listed", "missing_pending"}
    )
    if previous_visible < 10:
        return

    ratio = current_count / previous_visible if previous_visible else 1.0
    if ratio < min_ratio:
        raise RuntimeError(
            f"Suspicious catalog shrink: {previous_visible} -> {current_count} "
            f"({ratio:.1%}), below NIM_MIN_CATALOG_RATIO={min_ratio:.0%}. "
            "Refusing to advance removal state. Set NIM_MIN_CATALOG_RATIO=0 to disable this guard."
        )


def _parse_seen(value: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S UTC", "%Y-%m-%d %H:%M UTC", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue
    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def visible_rows(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = [
        row.copy()
        for row in state.get("models", {}).values()
        if row.get("status", "listed") in {"listed", "missing_pending"}
    ]
    rows.sort(
        key=lambda row: (-_parse_seen(row.get("first_seen_utc", "")).timestamp(), row.get("id", ""))
    )
    return rows


def atomic_write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(content)
    os.replace(tmp, path)


def save_state(state: Dict[str, Any]) -> None:
    atomic_write_text(STATE_PATH, json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def save_csv(state: Dict[str, Any]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = CSV_PATH + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in visible_rows(state):
            out = row.copy()
            out["created_date"] = format_created_date(row.get("created"))
            writer.writerow(out)
    os.replace(tmp, CSV_PATH)


def append_change_log(now_utc: str, events: Dict[str, List[Dict[str, Any]]], catalog: Dict[str, Any]) -> None:
    if not any(events.values()):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    record = {"timestamp": now_utc, **events, "catalog": catalog}
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def render_model_list(items: List[Dict[str, Any]], emoji: str, title: str) -> str:
    if not items:
        return ""
    lines = [f"  - {emoji} **{title} ({len(items)})**:"]
    for item in items:
        created = item.get("created") or "unknown"
        lines.append(f"    - `{item['id']}` (`{item.get('owned_by', 'unknown')}` · {created})")
    return "\n".join(lines) + "\n"


def read_recent_changes(limit: int = 5) -> List[Dict[str, Any]]:
    if not os.path.exists(JSONL_PATH):
        return []
    valid: List[Dict[str, Any]] = []
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                valid.append(record)
    return list(reversed(valid[-limit:]))


def render_recent_changes(limit: int = 5) -> str:
    recent = read_recent_changes(limit)
    if not recent:
        return "*暂无模型变更记录*"

    blocks: List[str] = []
    for entry in recent:
        block = f"- **{entry.get('timestamp', 'unknown')}**\n"
        block += render_model_list(entry.get("added", []), "🟢", "新增")
        block += render_model_list(entry.get("restored", []), "🔵", "恢复")
        block += render_model_list(entry.get("removed", []), "🔴", "确认移除")
        metadata_changed = entry.get("metadata_changed", [])
        if metadata_changed:
            block += f"  - 🟡 **元数据变化 ({len(metadata_changed)})**: "
            block += ", ".join(f"`{item.get('id', '?')}`" for item in metadata_changed)
            block += "\n"
        blocks.append(block.rstrip())
    return "\n\n".join(blocks)


def update_readme(limit: int = 5) -> None:
    rendered = render_recent_changes(limit)
    section = f"{MARKER_START}\n{rendered}\n{MARKER_END}"

    if os.path.exists(README_PATH):
        with open(README_PATH, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = "# check-nim-updates\n"

    if MARKER_START in content and MARKER_END in content:
        start = content.index(MARKER_START)
        end = content.index(MARKER_END, start) + len(MARKER_END)
        updated = content[:start] + section + content[end:]
    else:
        suffix = "" if content.endswith("\n") else "\n"
        updated = content + suffix + "\n## Recent model changes\n\n" + section + "\n"

    atomic_write_text(README_PATH, updated)


def summarize_events(events: Dict[str, List[Dict[str, Any]]]) -> str:
    return ", ".join(
        [
            f"added={len(events['added'])}",
            f"restored={len(events['restored'])}",
            f"removed={len(events['removed'])}",
            f"metadata_changed={len(events['metadata_changed'])}",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track NVIDIA NIM catalog changes safely")
    parser.add_argument("--api-url", default=API_URL)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--remove-confirmations", type=int, default=DEFAULT_REMOVE_CONFIRMATIONS)
    parser.add_argument("--min-catalog-ratio", type=float, default=DEFAULT_MIN_CATALOG_RATIO)
    parser.add_argument("--dry-run", action="store_true", help="Fetch and diff, but do not write files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("ERROR: missing NVIDIA_API_KEY", file=sys.stderr)
        return 2

    now_utc = utc_now_str()
    print(f"Fetching NIM model catalog from {args.api_url} ...")
    raw_models = fetch_models(api_key, args.api_url, args.timeout, args.retries)
    current, duplicates, invalid_rows = normalize_catalog(raw_models)

    if not current:
        raise RuntimeError("NIM model catalog contains zero valid model IDs; refusing to update state")

    state, migrated = load_state()
    validate_catalog_size(state, len(current), args.min_catalog_ratio)

    events = apply_catalog(
        state,
        current,
        duplicates,
        now_utc,
        args.remove_confirmations,
    )

    catalog_meta = {
        "raw_count": len(raw_models),
        "unique_count": len(current),
        "duplicate_rows": sum(max(0, count - 1) for count in duplicates.values()),
        "duplicate_ids": sorted(duplicates),
        "invalid_rows": invalid_rows,
    }
    state["last_success_utc"] = now_utc
    state["catalog"] = catalog_meta

    print(
        f"Catalog: raw={catalog_meta['raw_count']}, unique={catalog_meta['unique_count']}, "
        f"duplicate_rows={catalog_meta['duplicate_rows']}, invalid={invalid_rows}"
    )
    if migrated:
        print("State file missing: migrated existing CSV history into data/model_state.json")
    print("Changes: " + summarize_events(events))

    pending = [row for row in state["models"].values() if row.get("status") == "missing_pending"]
    if pending:
        print(
            f"Pending removals: {len(pending)} model(s) are missing but not confirmed yet "
            f"(threshold={max(1, args.remove_confirmations)} successful polls)."
        )

    if duplicates:
        print("WARNING: duplicate model IDs returned by API: " + ", ".join(sorted(duplicates)))

    if args.dry_run:
        print("Dry run: no files written.")
        return 0

    save_state(state)
    save_csv(state)
    append_change_log(now_utc, events, catalog_meta)
    update_readme(limit=5)
    print("Update complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
