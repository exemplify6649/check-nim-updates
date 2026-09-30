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

# 735790403 是英伟达创立日 (1993-04-26) 彩蛋默认时间戳
NVIDIA_FOUNDING_TS = 735790403

def fetch_models(api_key: str):
    req = urllib.request.Request(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "NIM-Tracker"
        }
    )
    with urllib.request.urlopen(req) as resp:
        if resp.status != 200:
            raise RuntimeError(f"API 请求失败，HTTP 状态码: {resp.status}")
        return json.loads(resp.read().decode("utf-8")).get("data", [])

def format_date_short(ts):
    """格式化为简明日期 YYYY-MM-DD"""
    if not ts:
        return "N/A"
    try:
        ts_int = int(ts)
        if ts_int == NVIDIA_FOUNDING_TS:
            return "1993-04-26"  # NVIDIA 创立日默认值
        return datetime.fromtimestamp(ts_int, tz=timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return str(ts)

def load_previous_models():
    """从本地 models.csv 加载已存模型"""
    if not os.path.exists(CSV_PATH):
        return {}
    models = {}
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            models[row["id"]] = row
    return models

def update_csv(models, old_map, now_utc_str):
    """
    保存 models.csv:
    1. 为新模型注入 first_seen_utc，旧模型保持原时间。
    2. 多级排序：先按 first_seen 倒序（最新发现的在顶端），再按 created 倒序，最后按 id 排序。
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    rows = []
    for m in models:
        mid = m["id"]
        # 如果是已知模型，保留最初捕获时间；如果是新模型，记录当前时间
        first_seen = old_map[mid]["first_seen_utc"] if mid in old_map and "first_seen_utc" in old_map[mid] else now_utc_str
        
        rows.append({
            "id": mid,
            "owned_by": m.get("owned_by", "unknown"),
            "created": m.get("created", 0),
            "created_date": format_date_short(m.get("created")),
            "first_seen_utc": first_seen,
            "object": m.get("object", "model")
        })

    # 多级降序排序：最新发现优先 -> 接口时间优先 -> ID 升序
    rows.sort(key=lambda x: (x["first_seen_utc"], int(x["created"] or 0), -ord(x["id"][0]) if x["id"] else 0), reverse=True)

    fields = ["id", "owned_by", "created_date", "first_seen_utc", "created", "object"]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def append_to_jsonl(now_utc_str, added_items, removed_items):
    """追加写入变更记录"""
    os.makedirs(DATA_DIR, exist_ok=True)
    record = {
        "timestamp": now_utc_str,
        "added": added_items,
        "removed": removed_items
    }
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def render_model_list(items, emoji, title):
    if not items:
        return ""
    if len(items) == 1:
        it = items[0]
        return f"  - {emoji} **{title} (1)**: `{it['id']}` (`{it['owned_by']}` · {it['created']})\n"
    
    res = f"  - {emoji} **{title} ({len(items)})**:\n"
    for it in items:
        res += f"    - `{it['id']}` (`{it['owned_by']}` · {it['created']})\n"
    return res

def update_readme_top5():
    """提取 JSONL 尾部 5 行渲染到 README"""
    if not os.path.exists(JSONL_PATH):
        return

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
        readme_blocks.append("*暂无模型变更记录*")
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
            updated = f"{before}{MARKER_START}\n{new_content}\n{MARKER_END}{after}"
            with open(README_PATH, "w", encoding="utf-8") as f:
                f.write(updated)

def main():
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("错误: 缺少环境变量 NVIDIA_API_KEY", file=sys.stderr)
        sys.exit(1)

    now_utc_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    print("抓取模型列表中...")
    models = fetch_models(api_key)
    current_map = {m["id"]: m for m in models}
    old_map = load_previous_models()

    # 首次初始化运行
    if not old_map:
        print(f"首次初始化：记录现存全部 {len(models)} 个模型")
        update_csv(models, {}, now_utc_str)
        update_readme_top5()
        return

    current_ids = set(current_map.keys())
    old_ids = set(old_map.keys())

    added_ids = current_ids - old_ids
    removed_ids = old_ids - current_ids

    if not added_ids and not removed_ids:
        print("模型列表无任何变化。")
        return

    print(f"检测到变动: 新增 {len(added_ids)} 个, 移除 {len(removed_ids)} 个")

    # 构建紧凑变动对象 (id, owned_by, created)
    added_items = [{
        "id": mid,
        "owned_by": current_map[mid].get("owned_by", "unknown"),
        "created": format_date_short(current_map[mid].get("created"))
    } for mid in sorted(added_ids)]

    removed_items = [{
        "id": mid,
        "owned_by": old_map[mid].get("owned_by", "unknown"),
        "created": old_map[mid].get("created_date", "未知")
    } for mid in sorted(removed_ids)]

    # 1. 更新 models.csv (保证新模型排在最前)
    update_csv(models, old_map, now_utc_str)

    # 2. 追加写入 JSONL 日志
    append_to_jsonl(now_utc_str, added_items, removed_items)

    # 3. 渲染最近 5 次变更至 README
    update_readme_top5()
    print("更新完成。")

if __name__ == "__main__":
    main()