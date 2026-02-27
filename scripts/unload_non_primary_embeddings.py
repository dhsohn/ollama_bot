#!/usr/bin/env python3
"""Unload embedding models from non-primary Lemonade instances.

Behavior:
- Reads config/config.yaml using core.config.load_config
- Targets lemonade.instances only (primary is never touched)
- Calls /health on each instance and unloads models whose type is embedding
- Supports one-shot mode or periodic loop mode
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import AppSettings, load_config


class ApiRequestError(RuntimeError):
    """HTTP/API request failure."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class InstanceTarget:
    name: str
    host: str
    base_path: str
    api_key: str
    timeout_seconds: int

    def endpoint(self, path: str) -> str:
        host = self.host.rstrip("/")
        base_path = self.base_path.strip()
        if base_path:
            host = f"{host}/{base_path.strip('/')}"
        suffix = path if path.startswith("/") else f"/{path}"
        return f"{host}{suffix}"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(level: str, message: str) -> None:
    print(f"[{_now()}] [{level}] {message}", flush=True)


def _compact_text(raw: str, *, max_chars: int = 200) -> str:
    compact = " ".join(raw.split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars] + "..."


def _detect_windows_host_ip() -> str | None:
    env_ip = os.getenv("WINDOWS_HOST_IP", "").strip()
    if env_ip:
        return env_ip

    resolv = Path("/etc/resolv.conf")
    if not resolv.exists():
        return None

    try:
        for line in resolv.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("nameserver "):
                return line.split()[1]
    except Exception:
        return None
    return None


def _resolve_connect_url(url: str) -> tuple[str, str | None]:
    """Resolve special hostnames when local DNS cannot resolve them.

    Returns:
      - connect_url: URL used for actual network call
      - original_host_header: Host header to preserve original host routing, if rewritten
    """
    parsed = urlsplit(url)
    hostname = parsed.hostname
    if not hostname:
        return url, None

    try:
        socket.getaddrinfo(hostname, parsed.port or 80)
        return url, None
    except socket.gaierror:
        pass

    if hostname not in {"windows-host", "host.docker.internal"}:
        return url, None

    replacement_ip = _detect_windows_host_ip()
    if not replacement_ip:
        return url, None

    connect_netloc = replacement_ip
    if parsed.port is not None:
        connect_netloc = f"{replacement_ip}:{parsed.port}"

    connect_url = urlunsplit(parsed._replace(netloc=connect_netloc))
    original_host = parsed.netloc
    return connect_url, original_host


def _request_json(
    *,
    target: InstanceTarget,
    method: str,
    path: str,
    payload: dict[str, Any] | None,
    timeout_seconds: int,
) -> Any:
    original_url = target.endpoint(path)
    connect_url, host_header = _resolve_connect_url(original_url)

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if target.api_key:
        headers["Authorization"] = f"Bearer {target.api_key}"
    if host_header:
        headers["Host"] = host_header

    data: bytes | None = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    request = Request(connect_url, data=data, headers=headers, method=method)

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8", "replace")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", "replace") if exc.fp is not None else ""
        raise ApiRequestError(
            f"{method} {original_url} failed (status={exc.code}): {_compact_text(body)}",
            status_code=exc.code,
        ) from exc
    except URLError as exc:
        raise ApiRequestError(
            f"{method} {original_url} failed: {exc.reason}",
            status_code=None,
        ) from exc
    except TimeoutError as exc:
        raise ApiRequestError(
            f"{method} {original_url} timed out after {timeout_seconds}s",
            status_code=None,
        ) from exc

    if not body.strip():
        return {}

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise ApiRequestError(
            f"{method} {original_url} returned non-JSON response: {_compact_text(body)}",
            status_code=None,
        ) from exc


def _extract_loaded_embedding_models(health_payload: Any) -> list[str]:
    if not isinstance(health_payload, dict):
        return []

    all_models = health_payload.get("all_models_loaded", [])
    if not isinstance(all_models, list):
        return []

    names: list[str] = []
    seen: set[str] = set()
    for item in all_models:
        if not isinstance(item, dict):
            continue
        model_type = str(item.get("type", "")).strip().lower()
        if model_type not in {"embedding", "embeddings"}:
            continue
        model_name = (
            item.get("model_name")
            or item.get("id")
            or item.get("name")
            or item.get("model")
        )
        if isinstance(model_name, str):
            name = model_name.strip()
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    return names


def _build_targets(settings: AppSettings, selected_instances: set[str]) -> list[InstanceTarget]:
    primary_api_key = settings.lemonade.api_key
    primary_base_path = settings.lemonade.base_path

    targets: list[InstanceTarget] = []
    for instance in settings.lemonade.instances:
        key = instance.name.strip().lower()
        if selected_instances and key not in selected_instances:
            continue
        targets.append(
            InstanceTarget(
                name=instance.name,
                host=instance.host,
                base_path=instance.base_path or primary_base_path,
                api_key=instance.api_key or primary_api_key,
                timeout_seconds=instance.timeout_seconds,
            )
        )
    return targets


def run_once(
    *,
    settings: AppSettings,
    timeout_seconds: int,
    dry_run: bool,
    selected_instances: set[str],
) -> bool:
    provider = str(settings.llm_provider).strip().lower()
    if provider != "lemonade":
        log("INFO", f"skip: llm_provider={provider}")
        return True

    targets = _build_targets(settings, selected_instances)
    if not targets:
        log("INFO", "skip: no non-primary lemonade instances configured")
        return True

    ok = True
    for target in targets:
        effective_timeout = max(timeout_seconds, target.timeout_seconds)
        try:
            health = _request_json(
                target=target,
                method="GET",
                path="/health",
                payload=None,
                timeout_seconds=effective_timeout,
            )
        except ApiRequestError as exc:
            log("ERROR", f"instance={target.name} health check failed: {exc}")
            ok = False
            continue

        embedding_models = _extract_loaded_embedding_models(health)
        if not embedding_models:
            log("INFO", f"instance={target.name} embedding_loaded=0")
            continue

        for model_name in embedding_models:
            if dry_run:
                log(
                    "INFO",
                    f"instance={target.name} dry_run: would unload embedding model '{model_name}'",
                )
                continue
            try:
                result = _request_json(
                    target=target,
                    method="POST",
                    path="/unload",
                    payload={"model_name": model_name},
                    timeout_seconds=effective_timeout,
                )
                status = result.get("status") if isinstance(result, dict) else None
                log(
                    "INFO",
                    f"instance={target.name} unloaded embedding model '{model_name}'"
                    f" (status={status or 'unknown'})",
                )
            except ApiRequestError as exc:
                if exc.status_code in {404, 405}:
                    log(
                        "WARN",
                        f"instance={target.name} /unload not supported (status={exc.status_code})",
                    )
                else:
                    log("ERROR", f"instance={target.name} unload failed: {exc}")
                ok = False

    return ok


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatically unload embedding models from non-primary Lemonade instances.",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config YAML (default: config/config.yaml)",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file used by config loader (default: .env)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=0,
        help="Run interval in seconds. 0 means one-shot mode (default: 0)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=8,
        help="Minimum request timeout in seconds (default: 8)",
    )
    parser.add_argument(
        "--instance",
        action="append",
        default=[],
        help="Target instance name (can be used multiple times)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call /unload; only print what would be unloaded",
    )
    parser.add_argument(
        "--exit-on-error",
        action="store_true",
        help="In loop mode, exit immediately when a cycle fails",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.interval < 0:
        log("ERROR", "--interval must be >= 0")
        return 2
    if args.timeout < 1:
        log("ERROR", "--timeout must be >= 1")
        return 2

    selected_instances = {name.strip().lower() for name in args.instance if name.strip()}

    cycle = 0
    while True:
        cycle += 1
        try:
            settings = load_config(config_path=args.config, env_file=args.env_file)
        except Exception as exc:
            log("ERROR", f"failed to load config: {exc}")
            if args.interval > 0 and not args.exit_on_error:
                time.sleep(args.interval)
                continue
            return 1

        ok = run_once(
            settings=settings,
            timeout_seconds=args.timeout,
            dry_run=args.dry_run,
            selected_instances=selected_instances,
        )

        if args.interval == 0:
            return 0 if ok else 1

        log("INFO", f"cycle={cycle} completed (ok={ok}); next run in {args.interval}s")
        if not ok and args.exit_on_error:
            return 1
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
