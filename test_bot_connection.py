import requests, time, datetime

url = "https://discord.com/api/webhooks/YOUR_WEBHOOK_URL"
while True:
    requests.post(url, json={"content": f"Status: OK {datetime.datetime.now()}"})
    time.sleep(30)
