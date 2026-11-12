"""Run the normal gateway, but execute one fixed package for every execution request.

Same arguments as ``task_skills.run_gateway`` plus ``--demo-package``. Example:

  python -B -m task_skills.run_demo_gateway --mode live --allow-live --host 0.0.0.0 \
      --allow-insecure-http --data-dir ./gateway-data/live

See ``task_skills/gateway/demo_override.py`` for exactly what is swapped.
"""

import argparse
from pathlib import Path

from . import run_gateway
from .gateway.demo_override import DemoOverrideGateway

DEFAULT_PACKAGE = (Path(__file__).resolve().parents[1]
                   / "docs/execution-api/examples/runtime-chemistry-beaker-pour.json")


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--demo-package", type=Path, default=DEFAULT_PACKAGE)
    args, rest = parser.parse_known_args(argv)
    if not args.demo_package.is_file():
        raise SystemExit(f"demo package not found: {args.demo_package}")
    print(f"DEMO OVERRIDE ON: every execution request runs {args.demo_package} "
          f"regardless of the posted graph", flush=True)

    def factory(*a, **k):
        return DemoOverrideGateway(*a, demo_package_path=args.demo_package, **k)

    run_gateway.Gateway = factory  # run_gateway.main looks the name up at call time
    return run_gateway.main(rest)


if __name__ == "__main__":
    main()
