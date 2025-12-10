import os
import requests
from dotenv import load_dotenv

load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

def google_search(query, num_results=5):
    url = "https://google.serper.dev/search"

    payload = {
        "q": query,
        "num": num_results
    }

    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    data = response.json()

    results = []
    if "organic" in data:
        for item in data["organic"]:
            results.append({
                "title": item.get("title"),
                "snippet": item.get("snippet"),
                "url": item.get("link")
            })

    return results

print(google_search("first President of India"))
