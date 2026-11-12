"""Run the local gateway. Default mock mode never imports hardware SDK modules."""

import argparse
import os
import ssl
from pathlib import Path

from .gateway.server import GatewayServer
from .gateway.service import Gateway
from .gateway.upstream import Upstream
from .models import load_scene_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Dobot external execution graph gateway")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--mode", choices=("mock", "live"), default="mock")
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--scene", type=Path, default=Path(__file__).with_name("scene_config.json"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--compiler-url")
    parser.add_argument("--results-url", help="Independent provisional 3.3 receiver base URL")
    parser.add_argument("--publish-catalog-snapshot", action="store_true",
                        help="also POST /catalog/snapshots; this REPLACES the compiler's whole catalog")
    parser.add_argument("--allow-insecure-http", action="store_true", help="Explicit lab-only HTTP transport override")
    parser.add_argument("--tls-cert", type=Path)
    parser.add_argument("--tls-key", type=Path)
    args = parser.parse_args(argv)
    if args.data_dir is None:
        args.data_dir = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local/state"))) / "DobotGateway" / args.mode
    token = os.environ.get("DOBOT_GATEWAY_TOKEN")
    if args.mode == "live" and not token:
        parser.error("Live mode requires DOBOT_GATEWAY_TOKEN")
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        if not token:
            parser.error("Non-loopback binding requires DOBOT_GATEWAY_TOKEN")
        if not args.tls_cert and not args.allow_insecure_http:
            parser.error("Non-loopback binding requires TLS or the explicit lab HTTP override")
    if bool(args.tls_cert) != bool(args.tls_key):
        parser.error("Supply both --tls-cert and --tls-key")
    upstream = Upstream(args.compiler_url, os.environ.get("DOBOT_COMPILER_TOKEN"),
                        allow_http=args.allow_insecure_http) if args.compiler_url else None
    results_delivery = None
    if args.results_url:
        from .results import ResultsDelivery
        results_delivery = ResultsDelivery(args.data_dir / "results-outbox", args.results_url,
                                           token=os.environ.get("DOBOT_RESULTS_TOKEN"),
                                           allow_http=args.allow_insecure_http)
    gateway = Gateway(load_scene_config(args.scene), args.data_dir, args.mode, args.allow_live,
                      upstream, results_delivery=results_delivery,
                      publish_catalog_snapshot=args.publish_catalog_snapshot)
    server = None
    try:
        server = GatewayServer((args.host, args.port), gateway, token)
        if args.tls_cert:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(str(args.tls_cert), str(args.tls_key))
            server.socket = context.wrap_socket(server.socket, server_side=True)
        print("Dobot gateway: %s:%s mode=%s catalog_revision=%s" %
              (args.host, server.server_port, args.mode, gateway.catalog.revision), flush=True)
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        if server:
            server.server_close()
        gateway.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
