import os
import json
import requests
import time
import config

BASE_DIR = "slack_backup"

headers = {
    "Authorization": f"Bearer {config.SLACK_TOKEN}"
}

# ---------- 工具函数 ----------

def safe_filename(name):
    for c in '\\/:*?"<>|':
        name = name.replace(c, "_")
    return name


def read_last_ts(channel_path):
    last_ts_file = os.path.join(channel_path, "last_ts.txt")
    if os.path.exists(last_ts_file):
        with open(last_ts_file, "r") as f:
            return f.read().strip()
    return "0"


def write_last_ts(channel_path, ts):
    with open(os.path.join(channel_path, "last_ts.txt"), "w") as f:
        f.write(ts)


# ---------- Slack API ----------

def get_channels():
    url = "https://slack.com/api/conversations.list"
    channels = []
    cursor = None    

    while True:
        params = {"limit": 200}
        if cursor:
            params["cursor"] = cursor

        resp = requests.get(url, headers=headers, params=params).json()
        channels.extend(resp["channels"])
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break

    return channels


def get_channel_history(channel_id, oldest_ts):
    url = "https://slack.com/api/conversations.history"
    messages = []
    cursor = None

    while True:
        params = {
            "channel": channel_id,
            "limit": 200,
            "oldest": oldest_ts
        }
        if cursor:
            params["cursor"] = cursor

        resp = requests.get(url, headers=headers, params=params).json()
        messages.extend(resp.get("messages", []))

        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break

    return messages


def get_thread_replies(channel_id, thread_ts):
    url = "https://slack.com/api/conversations.replies"
    replies = []
    cursor = None

    while True:
        params = {
            "channel": channel_id,
            "ts": thread_ts,
            "limit": 200
        }
        if cursor:
            params["cursor"] = cursor

        resp = requests.get(url, headers=headers, params=params).json()
        replies.extend(resp.get("messages", []))

        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break

    return replies


def download_file(url, path):
    resp = requests.get(url, headers=headers, stream=True)
    if resp.status_code == 200:
        with open(path, "wb") as f:
            for chunk in resp.iter_content(1024):
                f.write(chunk)
        return True
    return False


# ---------- 主流程（含线程备份） ----------

def backup_all():
    os.makedirs(BASE_DIR, exist_ok=True)

    channels = get_channels()
    print(f"Found {len(channels)} channels.")

    for ch in channels:
        cid = ch["id"]
        cname = safe_filename(ch.get("name", cid))
        print(f"\n=== Backing up #{cname} ===")

        ch_dir = os.path.join(BASE_DIR, cname)
        threads_dir = os.path.join(ch_dir, "threads")
        os.makedirs(ch_dir, exist_ok=True)
        os.makedirs(threads_dir, exist_ok=True)

        # ---------- 增量消息部分 ----------
        last_ts = read_last_ts(ch_dir)
        print(f"Last backup TS: {last_ts}")

        new_messages = get_channel_history(cid, last_ts)
        print(f"New messages: {len(new_messages)}")

        if new_messages:
            # 追加写入 messages.json
            messages_file = os.path.join(ch_dir, "messages.json")
            if os.path.exists(messages_file):
                with open(messages_file, "r", encoding="utf-8") as f:
                    old_messages = json.load(f)
            else:
                old_messages = []

            all_messages = old_messages + new_messages

            with open(messages_file, "w", encoding="utf-8") as f:
                json.dump(all_messages, f, indent=2, ensure_ascii=False)

        # ---------- 下载附件 ----------
        for msg in new_messages:
            if "files" in msg:
                for file in msg["files"]:
                    if "url_private" in file:
                        fname = safe_filename(file.get("name", file["id"]))
                        fpath = os.path.join(ch_dir, fname)

                        if os.path.exists(fpath):
                            print(f"  SKIP existing file: {fname}")
                            continue

                        print(f"  downloading: {fname}")
                        download_file(file["url_private"], fpath)
                        time.sleep(1)

        # ---------- 线程（replies）备份 ----------
        thread_count = 0

        for msg in new_messages:
            if "thread_ts" in msg:
                thread_ts = msg["thread_ts"]

                # 每个 thread 保存成 threads/<thread_ts>.json
                thread_file = os.path.join(threads_dir, f"{thread_ts}.json")

                # 增量：如果存在文件说明已备份
                if os.path.exists(thread_file):
                    print(f"  SKIP existing thread {thread_ts}")
                    continue

                print(f"  Fetching thread {thread_ts}")
                replies = get_thread_replies(cid, thread_ts)

                with open(thread_file, "w", encoding="utf-8") as f:
                    json.dump(replies, f, indent=2, ensure_ascii=False)

                thread_count += 1
                time.sleep(0.5)

        print(f"  Threads fetched: {thread_count}")

        # 更新 last_ts
        if new_messages:
            newest_ts = new_messages[0]["ts"]
            write_last_ts(ch_dir, newest_ts)
            print(f"Updated last_ts → {newest_ts}")


if __name__ == "__main__":
    backup_all()
