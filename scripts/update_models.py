import csv
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

API_URL = "https://integrate.api.nvidia.com/v1/models"
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "models.csv")
JSONL_PATH = os.path.join(DATA_DIR, "changes_log.jsonl")
README_PATH = "README.md"

MARKER_START = "<!-- CHANGES_START -->"
MARKER_END = "<!-- CHANGES_END -->"

def fetch_models(api_key: str):
    req = urllib.request.Request(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "NIM-Model-Tracker"
        }
    )
    with urllib.request.urlopen(req) as resp:
        if resp.status != 200:
            raise RuntimeError(f"API 请求失败，HTTP 状态码: {resp.status}")
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("data", [])

def format_date_short(ts):
    """提取简短日期 YYYY-MM-DD"""
    if not ts:
        return "未知日期"
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return str(ts)

def format_datetime_utc(ts):
    """格式化完整 UTC 时间戳用于 CSV"""
    if not ts:
        return "N/A"
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return "N/A"

def load_previous_models():
    """从已有的 models.csv 加载上次存量的模型字典"""
    if not os.path.exists(CSV_PATH):
        return {}
    models = {}
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            models[row["id"]] = row
    return models

def update_csv(models):
    """全量更新 models.csv，并按 created 降序（最新在前）"""
    models.sort(key=lambda x: int(x.get("created") or 0), reverse=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    fields = ["id", "created", "created_at_utc", "owned_by", "object"]
    
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for m in models:
            writer.writerow({
                "id": m.get("id"),
                "created": m.get("created"),
                "created_at_utc": format_datetime_utc(m.get("created")),
                "owned_by": m.get("owned_by", "unknown"),
                "object": m.get("object", "model"),
            })

def build_compact_item(model_id, meta):
    """生成只包含核心三要素的紧凑 dict"""
    created_raw = meta.get("created")
    return {
        "id": model_id,
        "created": format_date_short(created_raw),
        "owned_by": meta.get("owned_by", "unknown")
    }

def append_to_jsonl(added_items, removed_items):
    """追加写入一行变动到 changes_log.jsonl"""
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    record = {
        "timestamp": now_utc,
        "added": added_items,
        "removed": removed_items
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def render_model_list(items, emoji, title):
    """紧凑排版模型列表：仅 1 个则单行展示，多个则缩进换行"""
    if not items:
        return ""
    count = len(items)
    if count == 1:
        item = items[0]
        return f"  - {emoji} **{title} (1)**: `{item['id']}` (`{item['owned_by']}` · {item['created']})\n"
    
    res = f"  - {emoji} **{title} ({count})**:\n"
    for item in items:
        res += f"    - `{item['id']}` (`{item['owned_by']}` · {item['created']})\n"
    return res

def update_readme_top5():
    """读取 jsonl 最后 5 条，渲染到 README.md"""
    if not os.path.exists(JSONL_PATH):
        return

    # 从后往前读取最后 5 行
    recent_5 = []
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        for line in reversed(lines[-5:]):
            try:
                recent_5.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    readme_blocks = []
    if not recent_5:
        readme_blocks.append("*暂无变更记录（初始数据已就绪）*")
    else:
        for entry in recent_5:
            ts = entry["timestamp"]
            added = entry.get("added", [])
            removed = entry.get("removed", [])
            
            block = f"- **{ts}**\n"
            block += render_model_list(added, "🟢", "新增")
            block += render_model_list(removed, "🔴", "移除")
            readme_blocks.append(block.rstrip())

    new_content = "\n\n".join(readme_blocks)

    if os.path.exists(README_PATH):
        with open(README_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        if MARKER_START in content and MARKER_END in content:
            before = content.split(MARKER_START)[0]
            after = content.split(MARKER_END)[1]
            updated_content = f"{before}{MARKER_START}\n{new_content}\n{MARKER_END}{after}"
            with open(README_PATH, "w", encoding="utf-8") as f:
                f.write(updated_content)

def main():
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("错误: 缺少环境变量 NVIDIA_API_KEY", file=sys.stderr)
        sys.exit(1)

    print("正在抓取 NVIDIA NIM API 模型列表...")
    models = fetch_models(api_key)
    current_map = {m["id"]: m for m in models}
    
    old_map = load_previous_models()
    is_initial_run = (len(old_map) == 0)

    if is_initial_run:
        print("首次初始化运行，写入 models.csv...")
        update_csv(models)
        update_readme_top5()
        print("初始化完成。")
        return

    # 计算差异
    current_ids = set(current_map.keys())
    old_ids = set(old_map.keys())

    added_ids = current_ids - old_ids
    removed_ids = old_ids - current_ids

    if not added_ids and not removed_ids:
        print("模型列表无任何变化，流程结束。")
        return

    # 提取简短的 {id, created, owned_by}
    added_items = [build_compact_item(mid, current_map[mid]) for mid in sorted(added_ids)]
    removed_items = [build_compact_item(mid, old_map[mid]) for mid in sorted(removed_ids)]

    print(f"检测到变动: 新增 {len(added_items)} 个，移除 {len(removed_items)} 个")

    # 1. 覆盖 models.csv（按最新时间降序）
    update_csv(models)

    # 2. 追加写入 JSONL
    append_to_jsonl(added_items, removed_items)

    # 3. 提取最近 5 次更新 README
    update_readme_top5()
    print("更新完成。")

if __name__ == "__main__":
    main()