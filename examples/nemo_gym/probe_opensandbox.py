#!/usr/bin/env python3

"""Create one OpenSandbox sandbox and report lifecycle and execd diagnostics."""

import argparse
import asyncio
import json
import os
import ssl
import time
from datetime import timedelta
from typing import Any

import httpx
from opensandbox import Sandbox
from opensandbox.config import ConnectionConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--domain", default=os.environ.get("OPENSANDBOX_DOMAIN"))
    parser.add_argument(
        "--protocol", default=os.environ.get("OPENSANDBOX_PROTOCOL", "https")
    )
    parser.add_argument("--ca-bundle", default=os.environ.get("OPENSANDBOX_CA_BUNDLE"))
    parser.add_argument(
        "--use-server-proxy", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--create-timeout-seconds", type=int, default=720)
    parser.add_argument("--observe-seconds", type=int, default=120)
    args = parser.parse_args()
    if not args.domain:
        parser.error("--domain or OPENSANDBOX_DOMAIN is required")
    if not os.environ.get("OPENSANDBOX_API_KEY"):
        parser.error("OPENSANDBOX_API_KEY is required")
    return args


def print_json(label: str, value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    print(f"{label}={json.dumps(value, sort_keys=True, default=str)}", flush=True)


async def report_diagnostics(sandbox: Sandbox) -> None:
    for kind, getter in (
        ("events", sandbox.get_diagnostic_events),
        ("logs", sandbox.get_diagnostic_logs),
    ):
        try:
            diagnostic = await asyncio.wait_for(getter("all"), timeout=30)
            payload = diagnostic.model_dump(mode="json")
            content = payload.get("content")
            if content and len(content) > 20_000:
                payload["content"] = content[-20_000:]
                payload["probe_truncated"] = True
            print_json(f"diagnostic_{kind}", payload)
        except Exception as error:  # noqa: BLE001 - diagnostics must remain best effort
            print(
                f"diagnostic_{kind}_error={type(error).__name__}: {error}", flush=True
            )


async def probe(args: argparse.Namespace) -> None:
    api_key = os.environ["OPENSANDBOX_API_KEY"]
    ssl_context = ssl.create_default_context()
    if args.ca_bundle:
        ssl_context.load_verify_locations(cafile=args.ca_bundle)
    transport = httpx.AsyncHTTPTransport(
        verify=ssl_context,
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=0),
    )
    connection = ConnectionConfig(
        api_key=api_key,
        domain=args.domain,
        protocol=args.protocol,
        request_timeout=timedelta(seconds=120),
        use_server_proxy=args.use_server_proxy,
        headers={"OPEN-SANDBOX-API-KEY": api_key} if args.use_server_proxy else {},
        transport=transport,
        disable_metrics=True,
    )

    sandbox: Sandbox | None = None
    started = time.monotonic()
    try:
        sandbox = await asyncio.wait_for(
            Sandbox.create(
                args.image,
                timeout=timedelta(minutes=15),
                entrypoint=["tail", "-f", "/dev/null"],
                resource={"cpu": "4", "memory": "64Gi"},
                resource_requests={"cpu": "0.25", "memory": "512Mi"},
                metadata={"probe": "nemo-rl-opensandbox"},
                connection_config=connection,
                skip_health_check=True,
            ),
            timeout=args.create_timeout_seconds,
        )
        print(f"create_elapsed_seconds={time.monotonic() - started:.3f}", flush=True)
        print(f"sandbox_id={sandbox.id}", flush=True)

        deadline = time.monotonic() + args.observe_seconds
        attempt = 0
        while True:
            attempt += 1
            try:
                info = await asyncio.wait_for(sandbox.get_info(), timeout=30)
                print_json(f"info_{attempt}", info)
            except Exception as error:  # noqa: BLE001 - continue to collect other signals
                print(
                    f"info_{attempt}_error={type(error).__name__}: {error}", flush=True
                )

            health_started = time.monotonic()
            healthy = await asyncio.wait_for(sandbox.is_healthy(), timeout=30)
            print(
                f"health_{attempt}={healthy} elapsed_seconds={time.monotonic() - health_started:.3f}",
                flush=True,
            )
            if healthy or time.monotonic() >= deadline:
                break
            await asyncio.sleep(10)

        await report_diagnostics(sandbox)
        try:
            execution = await asyncio.wait_for(
                sandbox.commands.run("printf opensandbox-probe-ok"), timeout=30
            )
            print_json("command", execution)
        except Exception as error:  # noqa: BLE001 - command failure is probe output
            print(f"command_error={type(error).__name__}: {error}", flush=True)
            await report_diagnostics(sandbox)
    except Exception as error:
        print(f"probe_error={type(error).__name__}: {error}", flush=True)
        raise
    finally:
        if sandbox is not None:
            try:
                await asyncio.wait_for(sandbox.destroy(), timeout=60)
                print("cleanup=destroyed", flush=True)
            except Exception as error:  # noqa: BLE001 - report cleanup without masking probe result
                print(f"cleanup_error={type(error).__name__}: {error}", flush=True)
        await transport.aclose()


def main() -> None:
    asyncio.run(probe(parse_args()))


if __name__ == "__main__":
    main()
