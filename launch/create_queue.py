"""Lege die W&B-Launch-Queue 'Desktop_PC' (Resource: local-process) an.

Einmalig auszufuehren – die Queue ist serverseitig und bleibt danach
bestehen. Voraussetzung: `wandb login` wurde ausgefuehrt und der eingeloggte
Account hat Schreibrechte auf der Entity.

    python launch/create_queue.py
    python launch/create_queue.py --queue Desktop_PC --entity christian-debbertin-deepfake-detection

Der Aufruf ist idempotent: existiert die Queue bereits, wird das gemeldet und
ohne Fehler beendet.
"""

import argparse
import sys

import wandb

DEFAULT_ENTITY = "christian-debbertin-deepfake-detection"
DEFAULT_QUEUE = "Desktop_PC"
RESOURCE_TYPE = "local-process"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", default=DEFAULT_QUEUE, help="Name der Run-Queue.")
    parser.add_argument("--entity", default=DEFAULT_ENTITY, help="Team/Entity der Queue.")
    args = parser.parse_args()

    api = wandb.Api()

    try:
        api.create_run_queue(name=args.queue, type=RESOURCE_TYPE, entity=args.entity)
    except Exception as exc:  # noqa: BLE001 - CLI: jede Fehlerursache verstaendlich melden
        message = str(exc).lower()
        if "already exists" in message or "duplicate" in message:
            print(f"Queue '{args.entity}/{args.queue}' existiert bereits – nichts zu tun.")
            return 0
        print(f"Anlegen der Queue fehlgeschlagen: {exc}", file=sys.stderr)
        return 1

    print(
        f"Queue '{args.entity}/{args.queue}' (Resource: {RESOURCE_TYPE}) angelegt.\n"
        "Agent starten mit:\n"
        f"  wandb launch-agent -e {args.entity} -q {args.queue} -c launch/launch-config.yaml"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
