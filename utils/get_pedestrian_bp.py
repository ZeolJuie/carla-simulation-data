#!/usr/bin/env python3
import json
import os
import sys
import time
from pathlib import Path

import carla


def wait_for_world(client, max_wait_s=240, retry_interval_s=5):
    start = time.time()
    last_error = None
    while time.time() - start < max_wait_s:
        try:
            return client.get_world()
        except Exception as exc:
            last_error = exc
            time.sleep(retry_interval_s)
    raise RuntimeError(f"CARLA not ready after {max_wait_s}s: {last_error}")


def attr_to_dict(attr):
    try:
        recommended_values = list(attr.recommended_values)
    except Exception:
        recommended_values = []

    try:
        value = attr.as_str()
    except Exception:
        value = ""

    return {
        "id": attr.id,
        "type": str(attr.type),
        "is_modifiable": bool(attr.is_modifiable),
        "value": value,
        "recommended_values": recommended_values,
    }


def make_markdown(data):
    lines = []
    lines.append(f"# Walker Blueprints")
    lines.append("")
    lines.append(f"- map: `{data['map']}`")
    lines.append(f"- count: `{len(data['walkers'])}`")
    lines.append("")

    lines.append("## Blueprint List")
    lines.append("")
    for row in data["walkers"]:
        lines.append(f"- `{row['blueprint']}`")
    lines.append("")

    lines.append("## Blueprint Attributes")
    lines.append("")
    for row in data["walkers"]:
        lines.append(f"### `{row['blueprint']}`")
        lines.append("")
        lines.append("| attribute | type | modifiable | value | recommended_values |")
        lines.append("| --- | --- | --- | --- | --- |")
        for attr in row["attributes"]:
            rec = ", ".join(attr["recommended_values"]) if attr["recommended_values"] else ""
            value = attr["value"].replace("|", "\\|")
            rec = rec.replace("|", "\\|")
            lines.append(
                f"| `{attr['id']}` | `{attr['type']}` | `{attr['is_modifiable']}` | `{value}` | `{rec}` |"
            )
        lines.append("")

    return "\n".join(lines)


def main():
    host = os.getenv("CARLA_HOST", "127.0.0.1")
    port = int(os.getenv("CARLA_PORT", "2000"))
    rpc_timeout = float(os.getenv("CARLA_TIMEOUT", "10"))
    max_wait_s = int(os.getenv("CARLA_WAIT_SECONDS", "240"))

    out_json = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/walker_blueprints.json")
    out_md = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/walker_blueprints.md")

    client = carla.Client(host, port)
    client.set_timeout(rpc_timeout)

    world = wait_for_world(client, max_wait_s=max_wait_s)

    lib = world.get_blueprint_library()
    walkers = sorted(lib.filter("walker.pedestrian.*"), key=lambda bp: bp.id)

    rows = []
    for bp in walkers:
        rows.append({
            "blueprint": bp.id,
            "attributes": [attr_to_dict(attr) for attr in bp],
        })

    data = {
        "map": world.get_map().name,
        "host": host,
        "port": port,
        "walkers": rows,
    }

    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(make_markdown(data), encoding="utf-8")

    print(f"json: {out_json}")
    print(f"md:   {out_md}")
    print(f"map:  {data['map']}")
    print(f"count: {len(rows)}")
    for row in rows:
        print(row["blueprint"])


if __name__ == "__main__":
    main()