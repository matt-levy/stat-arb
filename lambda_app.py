import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

from project_paths import ROOT_DIR
from run_pipeline import main as run_pipeline_main


RUNTIME_SUBDIRECTORIES = ("data", "outputs", "reports", "logs")


def _parse_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _pipeline_args() -> List[str]:
    args: List[str] = []
    if _parse_bool("STAT_ARB_SKIP_RESEARCH"):
        args.append("--skip-research")
    if _parse_bool("STAT_ARB_SKIP_READY"):
        args.append("--skip-ready")
    if _parse_bool("STAT_ARB_SKIP_ALPACA"):
        args.append("--skip-alpaca")
    if _parse_bool("STAT_ARB_EXECUTE_TRADES"):
        args.append("--execute")
    if _parse_bool("STAT_ARB_ALLOW_STALE"):
        args.append("--allow-stale")
    return args


def _iter_runtime_files() -> Iterable[Path]:
    for subdirectory in RUNTIME_SUBDIRECTORIES:
        base_path = ROOT_DIR / subdirectory
        if not base_path.exists():
            continue
        yield from (path for path in base_path.rglob("*") if path.is_file())


def _upload_artifacts() -> Dict[str, object]:
    bucket = os.getenv("STAT_ARB_ARTIFACTS_BUCKET", "").strip()
    if not bucket:
        return {"uploaded": False, "reason": "STAT_ARB_ARTIFACTS_BUCKET not set"}

    import boto3

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = os.getenv("STAT_ARB_ARTIFACTS_PREFIX", "stat-arb").strip().strip("/")
    run_prefix = f"{prefix}/{timestamp}" if prefix else timestamp

    s3_client = boto3.client("s3")
    uploaded_keys: List[str] = []
    for path in _iter_runtime_files():
        relative_path = path.relative_to(ROOT_DIR).as_posix()
        object_key = f"{run_prefix}/{relative_path}"
        s3_client.upload_file(str(path), bucket, object_key)
        uploaded_keys.append(object_key)

    return {
        "uploaded": True,
        "bucket": bucket,
        "prefix": run_prefix,
        "file_count": len(uploaded_keys),
        "sample_keys": uploaded_keys[:10],
    }


def lambda_handler(event, context):
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        args = _pipeline_args()
        run_pipeline_main(args)
        upload_summary = _upload_artifacts()
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "Pipeline completed successfully.",
                    "started_at": started_at,
                    "runtime_root": str(ROOT_DIR),
                    "pipeline_args": args,
                    "upload_summary": upload_summary,
                    "event": event,
                },
                default=str,
            ),
        }
    except Exception as exc:  # pragma: no cover - failure path is operational
        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "message": "Pipeline failed.",
                    "started_at": started_at,
                    "runtime_root": str(ROOT_DIR),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "event": event,
                },
                default=str,
            ),
        }
