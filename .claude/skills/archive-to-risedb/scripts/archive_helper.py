#!/usr/bin/env python3
"""RISE 归档辅助脚本 — 生成计划、更新 README、验证结果。"""

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

BACKUP_ROOT = Path("/mnt/databackup/RISE")
README_PATH = BACKUP_ROOT / "README.md"
TODAY = date.today().strftime("%Y-%m-%d")

# ── data date extraction ──────────────────────────────────────────────

def extract_dates(paths: list[str]) -> dict[str, str]:
    """从文件名/路径提取 MMDD 日期，返回 {path: mmdd}。"""
    result = {}
    for p in paths:
        name = Path(p).name
        m = re.search(r"(?<![0-9])(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])(?![0-9])", name)
        result[p] = m.group(0) if m else "unknown"
    return result


# ── README parsing / generation ───────────────────────────────────────

def parse_readme() -> dict:
    """解析现有 README.md，返回结构化数据。"""
    if not README_PATH.exists():
        return {"categories": {"data": {}, "checkpoint": {}}, "readme_text": ""}
    text = README_PATH.read_text()
    return {"readme_text": text}


def read_current_structure() -> dict[str, list[str]]:
    """列出备份盘现有结构：{category: [MMDD, ...]}"""
    structure: dict[str, list[str]] = {}
    for cat in ("data", "checkpoint"):
        cat_path = BACKUP_ROOT / cat
        structure[cat] = sorted(
            [d.name for d in cat_path.iterdir() if d.is_dir()]
        ) if cat_path.exists() else []
    return structure


def generate_readme_section(category: str, mmdd: str, extra_info: dict) -> str:
    """生成 README 中某个条目的 markdown。"""
    robot = extra_info.get("robot", "bi_s1_follower")
    task = extra_info.get("task", "hang cloths")
    data_date = extra_info.get("data_date", mmdd)
    archive_date = extra_info.get("archive_date", TODAY)
    desc = extra_info.get("description", "")
    items_desc = extra_info.get("items_desc", "")

    label = "检查点" if category == "checkpoint" else "数据集"
    lines = [
        f"### {category}/{mmdd}/ — {desc}",
        "",
        "| 项目 | 内容 |",
        "|------|------|",
        f"| 机器人 | {robot} |",
        f"| 任务 | {task} |",
    ]
    if category == "checkpoint":
        train_data = extra_info.get("training_data", mmdd)
        lines.append(f"| 训练数据 | {train_data} |")
    lines.append(f"| 数据日期 | {data_date} |")
    lines.append(f"| 归档日期 | {archive_date} |")
    lines.append("")
    if items_desc:
        lines.append(items_desc)
        lines.append("")
    return "\n".join(lines)


# ── plan generation ───────────────────────────────────────────────────

def generate_plan(
    source_paths: list[str],
    category: str,
    data_date: str,
    keep_patterns: list[str],
    sub_path: str = "",
) -> dict:
    """生成归档计划 JSON。"""
    mmdd = data_date
    target_dir = BACKUP_ROOT / category / mmdd / sub_path if sub_path else BACKUP_ROOT / category / mmdd

    plan = {
        "category": category,
        "data_date": mmdd,
        "target_dir": str(target_dir),
        "archive_date": TODAY,
        "steps": [],
    }

    # Step: create target dir
    plan["steps"].append({"action": "mkdir", "path": str(target_dir)})

    # Step: mv each source to backup
    for src in source_paths:
        plan["steps"].append({"action": "mv", "src": src, "dst": str(target_dir / Path(src).name)})

    # Step: mv back kept items
    for kp in keep_patterns:
        kept_name = Path(kp).name if "/" in kp else kp
        plan["steps"].append({
            "action": "mv_back",
            "src": str(target_dir / kept_name),
            "dst": str(Path(kp)) if "/" in kp else str(Path(source_paths[0]).parent / kp),
        })

    return plan


# ── CLI ────────────────────────────────────────────────────────────────

def cmd_plan(args):
    """输出归档计划 JSON。"""
    plan = generate_plan(
        source_paths=args.sources,
        category=args.category,
        data_date=args.date,
        keep_patterns=args.keep,
        sub_path=args.sub_path,
    )
    print(json.dumps(plan, indent=2, ensure_ascii=False))


def cmd_dates(args):
    """检测路径中的数据日期。"""
    dates = extract_dates(args.paths)
    for p, d in dates.items():
        print(f"{d}\t{p}")


def cmd_structure(args):
    """打印备份盘当前结构。"""
    s = read_current_structure()
    for cat, mmdds in s.items():
        print(f"[{cat}]")
        for m in mmdds:
            print(f"  {m}")


def cmd_readme_section(args):
    """生成 README 条目文本。"""
    extra = {}
    if args.extra:
        extra = json.loads(args.extra)
    print(generate_readme_section(args.category, args.date, extra))


def main():
    parser = argparse.ArgumentParser(description="RISE 归档辅助脚本")
    sub = parser.add_subparsers(dest="command")

    p_plan = sub.add_parser("plan", help="生成归档计划")
    p_plan.add_argument("--sources", nargs="+", required=True)
    p_plan.add_argument("--category", required=True, choices=["data", "checkpoint"])
    p_plan.add_argument("--date", required=True, help="MMDD 数据日期")
    p_plan.add_argument("--keep", nargs="*", default=[])
    p_plan.add_argument("--sub-path", default="")

    p_dates = sub.add_parser("dates", help="提取数据日期")
    p_dates.add_argument("paths", nargs="+")

    sub.add_parser("structure", help="查看备份盘结构")

    p_sec = sub.add_parser("readme-section", help="生成 README 条目")
    p_sec.add_argument("--category", required=True, choices=["data", "checkpoint"])
    p_sec.add_argument("--date", required=True)
    p_sec.add_argument("--extra", default="{}", help="JSON extra info")

    args = parser.parse_args()

    if args.command == "plan":
        cmd_plan(args)
    elif args.command == "dates":
        cmd_dates(args)
    elif args.command == "structure":
        cmd_structure(args)
    elif args.command == "readme-section":
        cmd_readme_section(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
