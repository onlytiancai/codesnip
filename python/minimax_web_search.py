import requests
import json
import os

api_key = os.environ.get("MINIMAX_API_KEY")
api_host = os.environ.get("MINIMAX_API_HOST")

# ============ web_search ============
response = requests.post(
    f"{api_host}/v1/coding_plan/search",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    },
    json={"q": "人类登月新闻"}
)
print(json.dumps(response.json(), ensure_ascii=False, indent=2))
