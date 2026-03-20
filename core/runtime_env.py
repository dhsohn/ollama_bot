"""Runtime-environment and WSL networking helpers."""

from __future__ import annotations

import os
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def is_wsl_environment() -> bool:
    """Return whether the current runtime is WSL."""
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    proc_version = Path("/proc/version")
    try:
        text = proc_version.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False
    return "microsoft" in text or "wsl" in text


def normalize_host_token(raw: str) -> str:
    token = raw.strip()
    if not token:
        return ""
    if "://" in token:
        parsed = urlsplit(token)
        return (parsed.hostname or "").strip().lower()
    if token.startswith("[") and "]" in token:
        token = token[1:token.index("]")]
    elif token.count(":") == 1 and "." in token:
        token = token.split(":", 1)[0]
    return token.strip().lower()


def iter_wsl_bridge_candidates() -> list[str]:
    candidates: list[str] = []
    for env_key in ("WINDOWS_HOST", "WINDOWS_HOST_IP", "WSL_HOST_IP"):
        host = normalize_host_token(os.environ.get(env_key, ""))
        if host:
            candidates.append(host)

    resolv_conf = Path("/etc/resolv.conf")
    try:
        for line in resolv_conf.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines():
            stripped = line.strip()
            if not stripped.startswith("nameserver "):
                continue
            parts = stripped.split()
            if len(parts) >= 2:
                host = normalize_host_token(parts[1])
                if host:
                    candidates.append(host)
    except OSError:
        pass

    hosts_file = Path("/etc/hosts")
    try:
        for line in hosts_file.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines():
            stripped = line.split("#", 1)[0].strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            ip = normalize_host_token(parts[0])
            aliases = [normalize_host_token(alias) for alias in parts[1:]]
            if "homelab" in aliases:
                if ip:
                    candidates.append(ip)
                candidates.append("homelab")
            if "host.docker.internal" in aliases:
                if ip:
                    candidates.append(ip)
                candidates.append("host.docker.internal")
    except OSError:
        pass

    candidates.extend(("homelab", "host.docker.internal"))

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        host = normalize_host_token(candidate)
        if not host or host in {"localhost", "127.0.0.1", "::1"}:
            continue
        if host in seen:
            continue
        seen.add(host)
        deduped.append(host)
    return deduped


def can_connect_tcp(host: str, port: int, *, timeout_seconds: float = 0.35) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def resolve_wsl_loopback_host(
    *,
    url: str,
    service_name: str,
    logger: Any,
    is_wsl_environment_fn: Callable[[], bool] = is_wsl_environment,
    iter_wsl_bridge_candidates_fn: Callable[[], list[str]] = iter_wsl_bridge_candidates,
    can_connect_tcp_fn: Callable[[str, int], bool] | None = None,
) -> str:
    """Swap loopback URLs for a reachable Windows bridge host when needed on WSL."""
    tcp_probe = can_connect_tcp_fn or (lambda host, port: can_connect_tcp(host, port))

    parsed = urlsplit(url)
    host = (parsed.hostname or "").strip().lower()
    if host not in {"localhost", "127.0.0.1", "::1"}:
        return url
    if not is_wsl_environment_fn():
        return url

    port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
    if tcp_probe(host, port):
        return url

    candidates = iter_wsl_bridge_candidates_fn()
    for candidate in candidates:
        if not tcp_probe(candidate, port):
            continue

        target_host = candidate
        if ":" in target_host and not target_host.startswith("["):
            target_host = f"[{target_host}]"

        userinfo = ""
        if parsed.username:
            userinfo = parsed.username
            if parsed.password:
                userinfo += f":{parsed.password}"
            userinfo += "@"

        new_netloc = f"{userinfo}{target_host}"
        if parsed.port is not None:
            new_netloc = f"{new_netloc}:{parsed.port}"

        rewritten = urlunsplit(
            (parsed.scheme, new_netloc, parsed.path, parsed.query, parsed.fragment)
        )
        logger.warning(
            "wsl_loopback_rewritten",
            service=service_name,
            original=url,
            rewritten=rewritten,
            reason="loopback_unreachable_from_wsl",
        )
        return rewritten

    logger.warning(
        "wsl_loopback_unreachable",
        service=service_name,
        target=url,
        tried_candidates=candidates,
    )
    return url
