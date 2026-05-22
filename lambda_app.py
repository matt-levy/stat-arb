import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from project_paths import ROOT_DIR
from run_pipeline import main as run_pipeline_main


RUNTIME_SUBDIRECTORIES = ("data", "outputs", "reports", "logs")
LATEST_ARTIFACTS = (
    "outputs/alpaca_order_preview.csv",
    "outputs/alpaca_pair_attribution.csv",
    "outputs/live_pair_signals.csv",
    "outputs/paper_trade_ready_signals.csv",
    "logs/alpaca_execution_log.csv",
    "logs/alpaca_order_fills.csv",
    "logs/alpaca_trade_log.csv",
    "logs/alpaca_pair_lifecycle_log.csv",
)


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


def _normalize_prefix(prefix: str) -> str:
    return prefix.strip().strip("/")


def _json_bytes(payload: Dict[str, object]) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8")


def _put_json_object(s3_client, bucket: str, key: str, payload: Dict[str, object]) -> None:
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=_json_bytes(payload),
        ContentType="application/json",
    )


def _build_run_prefix(prefix: str, timestamp: str, request_id: str) -> str:
    run_id = request_id.strip() or "unknown-request"
    normalized_prefix = _normalize_prefix(prefix)
    run_suffix = f"{timestamp}-{run_id}"
    return f"{normalized_prefix}/runs/{run_suffix}" if normalized_prefix else f"runs/{run_suffix}"


def _latest_prefix(prefix: str) -> str:
    normalized_prefix = _normalize_prefix(prefix)
    return f"{normalized_prefix}/latest" if normalized_prefix else "latest"


def _iter_latest_files() -> Iterable[Path]:
    for relative_artifact in LATEST_ARTIFACTS:
        artifact_path = ROOT_DIR / relative_artifact
        if artifact_path.exists() and artifact_path.is_file():
            yield artifact_path


def _upload_artifacts(
    *,
    started_at: str,
    pipeline_args: List[str],
    event: Dict[str, object],
    context,
    status: str,
    error: Optional[str] = None,
    failure_traceback: str = "",
) -> Dict[str, object]:
    bucket = os.getenv("STAT_ARB_ARTIFACTS_BUCKET", "").strip()
    if not bucket:
        return {"uploaded": False, "reason": "STAT_ARB_ARTIFACTS_BUCKET not set"}

    import boto3

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = os.getenv("STAT_ARB_ARTIFACTS_PREFIX", "stat-arb")
    request_id = getattr(context, "aws_request_id", "") or "unknown-request"
    run_prefix = _build_run_prefix(prefix, timestamp, request_id)
    latest_prefix = _latest_prefix(prefix)

    s3_client = boto3.client("s3")
    uploaded_keys: List[str] = []
    relative_paths: List[str] = []
    for path in _iter_runtime_files():
        relative_path = path.relative_to(ROOT_DIR).as_posix()
        object_key = f"{run_prefix}/{relative_path}"
        s3_client.upload_file(str(path), bucket, object_key)
        uploaded_keys.append(object_key)
        relative_paths.append(relative_path)

    for path in _iter_latest_files():
        relative_path = path.relative_to(ROOT_DIR).as_posix()
        latest_key = f"{latest_prefix}/{relative_path}"
        s3_client.upload_file(str(path), bucket, latest_key)

    manifest = {
        "status": status,
        "started_at": started_at,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "runtime_root": str(ROOT_DIR),
        "bucket": bucket,
        "run_prefix": run_prefix,
        "latest_prefix": latest_prefix,
        "pipeline_args": pipeline_args,
        "request_id": request_id,
        "function_name": getattr(context, "function_name", os.getenv("AWS_LAMBDA_FUNCTION_NAME", "")),
        "function_version": getattr(context, "function_version", os.getenv("AWS_LAMBDA_FUNCTION_VERSION", "")),
        "invoked_function_arn": getattr(context, "invoked_function_arn", ""),
        "file_count": len(relative_paths),
        "files": relative_paths,
        "error": error or "",
        "traceback": failure_traceback,
        "event": event,
    }
    manifest_key = f"{run_prefix}/manifest.json"
    _put_json_object(s3_client, bucket, manifest_key, manifest)

    latest_manifest = {
        "status": status,
        "started_at": started_at,
        "bucket": bucket,
        "run_prefix": run_prefix,
        "manifest_key": manifest_key,
        "request_id": request_id,
        "file_count": len(relative_paths),
        "pipeline_args": pipeline_args,
        "error": error or "",
    }
    _put_json_object(s3_client, bucket, f"{_normalize_prefix(prefix)}/latest.json" if _normalize_prefix(prefix) else "latest.json", latest_manifest)
    if status == "success":
        _put_json_object(
            s3_client,
            bucket,
            f"{_normalize_prefix(prefix)}/latest-success.json" if _normalize_prefix(prefix) else "latest-success.json",
            latest_manifest,
        )
    else:
        _put_json_object(
            s3_client,
            bucket,
            f"{_normalize_prefix(prefix)}/latest-failure.json" if _normalize_prefix(prefix) else "latest-failure.json",
            latest_manifest,
        )

    return {
        "uploaded": True,
        "bucket": bucket,
        "prefix": run_prefix,
        "latest_prefix": latest_prefix,
        "manifest_key": manifest_key,
        "file_count": len(relative_paths),
        "sample_keys": uploaded_keys[:10],
    }


def lambda_handler(event, context):
    started_at = datetime.now(timezone.utc).isoformat()
    args = _pipeline_args()
    try:
        run_pipeline_main(args)
        upload_summary = _upload_artifacts(
            started_at=started_at,
            pipeline_args=args,
            event=event,
            context=context,
            status="success",
        )
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
        failure_traceback = traceback.format_exc()
        try:
            upload_summary = _upload_artifacts(
                started_at=started_at,
                pipeline_args=args,
                event=event,
                context=context,
                status="failure",
                error=str(exc),
                failure_traceback=failure_traceback,
            )
        except Exception as upload_exc:  # pragma: no cover - operational fallback
            upload_summary = {
                "uploaded": False,
                "reason": f"artifact upload failed after pipeline failure: {upload_exc}",
            }
        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "message": "Pipeline failed.",
                    "started_at": started_at,
                    "runtime_root": str(ROOT_DIR),
                    "error": str(exc),
                    "traceback": failure_traceback,
                    "pipeline_args": args,
                    "upload_summary": upload_summary,
                    "event": event,
                },
                default=str,
            ),
        }
