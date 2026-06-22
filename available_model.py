import requests
import yaml


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


api_key = config["GROQ_API_KEY"]

url = "https://api.groq.com/openai/v1/models"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}


response = requests.get(url, headers=headers)

response.raise_for_status()


for model in response.json()["data"]:
    print(model["id"])