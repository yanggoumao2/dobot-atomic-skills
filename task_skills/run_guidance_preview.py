"""Preview how guidance graphs expand into atomic flows, offline.

Each input file goes through ``Catalog.validate`` - the same call the gateway
makes on ``/execution-plans/validate`` - so the picture shows exactly what the
gateway would schedule, or the exact error it would return. Nothing here opens
a socket, reads the robot or imports the hardware SDK.

The atomic diagram puts one lane per arm (plus ``system`` for waits) and one
column per dependency depth: nodes in the same column with no edge between them
are what ``run_graph`` may run at the same time, one node per arm. Edges that
cross between arm lanes are the synchronisation points between the two arms.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path
from typing import Any

from .gateway.contracts import Catalog, ContractError
from .models import load_scene_config

LANES = ("left", "system", "right")
LANE_TITLES = {"left": "左臂 left", "system": "等待 system", "right": "右臂 right"}
SKILL_FILL = {
    "arm.move_joint": "#e8f1fb", "arm.move_linear": "#dcebfa",
    "gripper.set_position": "#fff3e0", "arm.rotate_about_pivot": "#e3f5e9",
    "arm.move_to_home": "#eeeafc", "wait": "#f4f4f4",
}
GROUP_STROKES = ("#1a5fb4", "#c0392b", "#1e8449", "#8e44ad", "#b9770e", "#117a8b")
BOX_W, BOX_H, COL_GAP, ROW_GAP, LANE_PAD, LABEL_W = 168, 58, 40, 10, 14, 96


def _read(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("guidance file must contain a JSON object")
    return document


def _depths(order: list[str], dependencies: dict[str, list[str]]) -> dict[str, int]:
    depth: dict[str, int] = {}
    for node_id in order:
        depth[node_id] = 1 + max((depth[d] for d in dependencies.get(node_id, []) if d in depth), default=-1)
    return depth


def _short_ref(ref: Any) -> str:
    parts = str(ref).split(".")
    return ".".join(parts[1:]) if len(parts) == 3 else str(ref)


def describe(node: dict[str, Any]) -> tuple[str, str]:
    """Return (skill label, one-line detail) for an expanded package node."""
    if node.get("node_type") == "wait":
        return "wait", f"等待 {node.get('duration_s', 0):g}s"
    call = node.get("skill_invocation") or {}
    skill = str(call.get("skill_id", "")).removeprefix("dobot.")
    p = call.get("parameters") or {}
    if skill in {"arm.move_joint", "arm.move_linear"}:
        start = f"{_short_ref(p['source_ref'])} → " if p.get("source_ref") else "→ "
        return skill, f"{start}{_short_ref(p.get('target_ref'))}  v={p.get('speed', '-')}"
    if skill == "gripper.set_position":
        verb = "张开" if p.get("position") == 255 else "闭合" if p.get("position") == 0 else "开度"
        return skill, f"{verb} pos={p.get('position')}" + (" 允许松开" if p.get("allow_release") else "")
    if skill == "arm.rotate_about_pivot":
        return skill, f"倾倒 tilt={p.get('tilt_deg')}° {p.get('segments', '?')}段 v={p.get('speed', '-')}"
    if skill == "arm.move_to_home":
        return skill, f"回 home  v={p.get('speed', '-')}"
    return skill, json.dumps(p, ensure_ascii=False)[:40]


def preview(path: Path, catalog: Catalog) -> dict[str, Any]:
    """Validate one guidance file and collect what the page and terminal show."""
    result: dict[str, Any] = {"file": path.name, "path": str(path)}
    try:
        guidance = _read(path)
    except (OSError, ValueError) as error:
        result["error"] = {"code": "UNREADABLE", "message": str(error), "path": ""}
        return result
    result["guidance"] = guidance
    try:
        plan = catalog.validate(guidance, check_state=False)
    except ContractError as error:
        result["error"] = error.body
        return result
    nodes = {node["node_id"]: node for node in plan.package["nodes"]}
    owner = {atom: source for source, atoms in plan.package.get("guidance_lineage", {}).items()
             for atom in atoms}
    rows = []
    for index, node_id in enumerate(plan.order, start=1):
        skill, detail = describe(nodes[node_id])
        rows.append({"index": index, "id": node_id, "arm": plan.arms[node_id], "skill": skill,
                     "detail": detail, "needs": list(plan.dependencies[node_id]),
                     "guidance": owner.get(node_id, node_id)})
    result.update(family=plan.family, rows=rows, depth=_depths(plan.order, plan.dependencies))
    return result


# ------------------------------------------------------------------ drawing

def _text(x: float, y: float, value: str, size: int = 11, weight: str = "normal",
          anchor: str = "start", fill: str = "#1d1d1f") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}" fill="{fill}">{html.escape(value)}</text>')


def _guidance_svg(guidance: dict[str, Any]) -> str:
    """Small diagram of the incoming guidance graph; tolerant of invalid input."""
    raw = [n for n in guidance.get("nodes", []) if isinstance(n, dict)]
    ids = [str(n.get("node_id")) for n in raw]
    needs: dict[str, list[str]] = {i: [] for i in ids}
    for edge in guidance.get("edges", []) if isinstance(guidance.get("edges"), list) else []:
        if isinstance(edge, dict) and edge.get("target") in needs:
            needs[edge["target"]].append(str(edge.get("source")))
    depth: dict[str, int] = {}
    for _ in ids:  # relax; a cycle simply stops refining
        for i in ids:
            depth[i] = 1 + max((depth.get(s, 0) for s in needs[i] if s in needs), default=-1)
            depth[i] = min(depth[i], len(ids))
    stack: dict[int, int] = {}
    pos = {}
    for i in ids:
        row = stack.get(depth[i], 0)
        stack[depth[i]] = row + 1
        pos[i] = (10 + depth[i] * (BOX_W + COL_GAP + 20), 10 + row * (BOX_H + 22))
    width = 20 + (max(depth.values(), default=0) + 1) * (BOX_W + COL_GAP + 20)
    height = 20 + max(stack.values(), default=1) * (BOX_H + 22)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<defs><marker id="ga" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
             'markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#555"/></marker></defs>']
    for target, sources in needs.items():
        for source in sources:
            if source in pos:
                (x1, y1), (x2, y2) = pos[source], pos[target]
                parts.append(f'<line x1="{x1 + BOX_W + 20:.1f}" y1="{y1 + BOX_H / 2:.1f}" x2="{x2 - 2:.1f}" '
                             f'y2="{y2 + BOX_H / 2:.1f}" stroke="#555" stroke-width="1.4" marker-end="url(#ga)"/>')
    for index, node in enumerate(raw):
        x, y = pos[ids[index]]
        call = node.get("skill_invocation") or {}
        label = call.get("skill_id") or node.get("node_type", "?")
        params = call.get("parameters") or ({"duration_s": node.get("duration_s")} if node.get("node_type") == "wait" else {})
        stroke = GROUP_STROKES[index % len(GROUP_STROKES)]
        parts.append(f'<rect x="{x}" y="{y}" width="{BOX_W + 20}" height="{BOX_H}" rx="8" fill="#fff" '
                     f'stroke="{stroke}" stroke-width="2.2"/>')
        parts.append(_text(x + 8, y + 17, ids[index], 12, "bold"))
        parts.append(_text(x + 8, y + 33, str(label), 11, fill="#444"))
        if "source_object" in params:
            summary = (f"{params.get('source_object')}→{params.get('target_object')} "
                       f"tilt={params.get('tilt_deg', 80)} v={params.get('motion_speed', 15)}")
        else:
            summary = ", ".join(f"{k}={v}" for k, v in params.items())
        parts.append(_text(x + 8, y + 49, summary[:34], 10, fill="#666"))
    parts.append("</svg>")
    return "".join(parts)


def _atomic_svg(result: dict[str, Any]) -> str:
    rows, depth = result["rows"], result["depth"]
    groups = list(dict.fromkeys(row["guidance"] for row in rows))
    stroke = {g: GROUP_STROKES[i % len(GROUP_STROKES)] for i, g in enumerate(groups)}
    slots: dict[tuple[str, int], int] = {}
    slot_of = {}
    for row in rows:
        key = (row["arm"], depth[row["id"]])
        slot_of[row["id"]] = slots.get(key, 0)
        slots[key] = slots.get(key, 0) + 1
    lanes = [lane for lane in LANES if any(r["arm"] == lane for r in rows)]
    lane_rows = {lane: max([n for (a, _), n in slots.items() if a == lane], default=1) for lane in lanes}
    lane_top, y = {}, 10
    for lane in lanes:
        lane_top[lane] = y
        y += lane_rows[lane] * (BOX_H + ROW_GAP) + 2 * LANE_PAD
    height = y + 10
    columns = max(depth.values()) + 1
    width = LABEL_W + columns * (BOX_W + COL_GAP) + 20
    pos = {}
    for row in rows:
        x = LABEL_W + depth[row["id"]] * (BOX_W + COL_GAP)
        top = lane_top[row["arm"]] + LANE_PAD + slot_of[row["id"]] * (BOX_H + ROW_GAP)
        pos[row["id"]] = (x, top)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<defs><marker id="aa" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
             'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#888"/></marker>'
             '<marker id="as" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
             'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#e67e22"/></marker></defs>']
    for index, lane in enumerate(lanes):
        top, h = lane_top[lane], lane_rows[lane] * (BOX_H + ROW_GAP) + 2 * LANE_PAD
        parts.append(f'<rect x="0" y="{top}" width="{width}" height="{h}" fill="{"#fafbfd" if index % 2 == 0 else "#ffffff"}" '
                     f'stroke="#e3e6ea"/>')
        parts.append(_text(10, top + h / 2 + 4, LANE_TITLES[lane], 12, "bold", fill="#555"))
    for column in range(columns):
        parts.append(_text(LABEL_W + column * (BOX_W + COL_GAP) + BOX_W / 2, height - 1,
                           f"层 {column}", 9, anchor="middle", fill="#aaa"))
    arm_of = {row["id"]: row["arm"] for row in rows}
    for row in rows:
        x2, y2 = pos[row["id"]]
        for need in row["needs"]:
            x1, y1 = pos[need]
            cross = arm_of[need] != row["arm"]
            sx, sy, ex, ey = x1 + BOX_W, y1 + BOX_H / 2, x2 - 2, y2 + BOX_H / 2
            mid = (sx + ex) / 2
            style = ('stroke="#e67e22" stroke-width="1.8" stroke-dasharray="5 3" marker-end="url(#as)"' if cross
                     else 'stroke="#888" stroke-width="1.3" marker-end="url(#aa)"')
            parts.append(f'<path d="M{sx:.1f},{sy:.1f} C{mid:.1f},{sy:.1f} {mid:.1f},{ey:.1f} {ex:.1f},{ey:.1f}" '
                         f'fill="none" {style}/>')
    for row in rows:
        x, y = pos[row["id"]]
        short = row["id"].split(".", 1)[1] if row["id"].startswith(row["guidance"] + ".") else row["id"]
        parts.append(f'<g><title>{html.escape(row["id"])}&#10;依赖: {html.escape(", ".join(row["needs"]) or "无")}</title>'
                     f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="7" '
                     f'fill="{SKILL_FILL.get(row["skill"], "#fff")}" stroke="{stroke[row["guidance"]]}" stroke-width="2"/>')
        parts.append(_text(x + 7, y + 16, f"{row['index']:02d} {short}"[:26], 11, "bold"))
        parts.append(_text(x + 7, y + 32, row["skill"], 10, fill="#555"))
        parts.append(_text(x + 7, y + 48, row["detail"][:28], 10, fill="#333") + "</g>")
    parts.append("</svg>")
    return "".join(parts)


def _section(result: dict[str, Any]) -> str:
    title = f'<h2>{html.escape(result["file"])}</h2><p class="path">{html.escape(result["path"])}</p>'
    guidance = ""
    if "guidance" in result:
        guidance = f'<h3>输入：关键实验操作执行图</h3><div class="scroll">{_guidance_svg(result["guidance"])}</div>'
    if "error" in result:
        error = result["error"]
        return (f'<section>{title}<div class="bad">网关会拒绝这张图 · <b>{html.escape(error["code"])}</b>'
                f'<br>{html.escape(error["message"])}'
                + (f' <span class="path">({html.escape(error["path"])})</span>' if error.get("path") else "")
                + f"</div>{guidance}</section>")
    rows = result["rows"]
    counts = {lane: sum(r["arm"] == lane for r in rows) for lane in LANES}
    syncs = sum(1 for r in rows for n in r["needs"] if any(x["id"] == n and x["arm"] != r["arm"] for x in rows))
    stats = (f'<div class="ok">网关接受 · {len(rows)} 个原子节点（左臂 {counts["left"]}，右臂 {counts["right"]}，'
             f'等待 {counts["system"]}）· 跨臂同步边 {syncs} 条 · 依赖层数 {max(result["depth"].values()) + 1}</div>')
    table = "".join(
        f'<tr><td>{r["index"]:02d}</td><td>{html.escape(r["id"])}</td><td>{r["arm"]}</td>'
        f'<td>{html.escape(r["skill"])}</td><td>{html.escape(r["detail"])}</td>'
        f'<td>{html.escape(", ".join(r["needs"]) or "—")}</td></tr>' for r in rows)
    return (f"<section>{title}{stats}{guidance}<h3>输出：展开后的原子操作流程</h3>"
            f'<div class="scroll">{_atomic_svg(result)}</div>'
            f"<details><summary>节点表（编号 = run_graph 的派发优先顺序）</summary><table><tr><th>#</th><th>节点</th>"
            f"<th>臂</th><th>原子</th><th>内容</th><th>依赖</th></tr>{table}</table></details></section>")


def render_html(results: list[dict[str, Any]], refresh: int | None = None) -> str:
    meta = f'<meta http-equiv="refresh" content="{refresh}">' if refresh else ""
    return f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">{meta}
<title>执行图展开预览</title><style>
body{{font-family:"Microsoft YaHei","PingFang SC",sans-serif;background:#fff;color:#1d1d1f;margin:24px;}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:17px;margin:28px 0 2px}} h3{{font-size:13px;color:#555;margin:14px 0 6px}}
.path{{color:#888;font-size:12px;margin:0}} .scroll{{overflow-x:auto;border:1px solid #eee;border-radius:6px;padding:6px}}
.ok,.bad{{padding:8px 12px;border-radius:6px;font-size:13px;margin:8px 0}} .ok{{background:#e9f7ee}} .bad{{background:#fdecec;color:#7b1d14}}
table{{border-collapse:collapse;font-size:12px;margin-top:6px}} td,th{{border:1px solid #e3e6ea;padding:3px 8px;text-align:left}}
.legend{{font-size:12px;color:#555}} .legend span{{display:inline-block;padding:1px 8px;margin-right:6px;border-radius:4px;border:1px solid #ccc}}
section{{border-top:1px solid #ddd}}
</style></head><body><h1>执行图 → 原子操作流程 预览</h1>
<p class="legend">离线预览，走的是网关同一个 <code>Catalog.validate</code>，不连机械臂。横向一列 = 一个依赖层，同列不同臂可并行；
<span style="border-color:#e67e22;color:#e67e22">橙色虚线</span>= 跨臂同步依赖；方框边框颜色 = 来自哪个上游节点。
<span style="background:#e8f1fb">关节运动</span><span style="background:#dcebfa">直线运动</span><span style="background:#fff3e0">夹爪</span>
<span style="background:#e3f5e9">倾倒</span><span style="background:#eeeafc">回 home</span><span style="background:#f4f4f4">等待</span>
生成于 {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
{"".join(_section(r) for r in results)}</body></html>"""


def print_summary(result: dict[str, Any]) -> None:
    print(f"\n== {result['file']}")
    if "error" in result:
        error = result["error"]
        print(f"   拒绝 {error['code']}: {error['message']}" + (f" ({error['path']})" if error.get("path") else ""))
        return
    for row in result["rows"]:
        print(f"   {row['index']:02d} 层{result['depth'][row['id']]:<2} {row['arm']:<6} {row['id']:<38} {row['detail']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("guidance", nargs="+", type=Path, help="guidance graph JSON file(s) or folder(s)")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("artifacts") / "guidance-preview.html")
    parser.add_argument("--scene", type=Path, default=Path(__file__).with_name("scene_config.json"))
    parser.add_argument("--watch", action="store_true", help="regenerate when an input changes; the page auto-refreshes")
    args = parser.parse_args(argv)
    files = sorted({f for p in args.guidance for f in (sorted(p.glob("*.json")) if p.is_dir() else [p])})
    if not files:
        parser.error("no guidance JSON files found")
    catalog = Catalog(load_scene_config(args.scene))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen = None
    while True:
        stamp = [f.stat().st_mtime if f.exists() else None for f in files]
        if stamp != seen:
            seen = stamp
            results = [preview(f, catalog) for f in files]
            for result in results:
                print_summary(result)
            args.out.write_text(render_html(results, refresh=2 if args.watch else None), encoding="utf-8")
            print(f"\n预览页: {args.out.resolve()}")
        if not args.watch:
            return 1 if any("error" in r for r in results) and len(files) == 1 else 0
        time.sleep(1)


if __name__ == "__main__":
    sys.exit(main())
