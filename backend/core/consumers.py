# core/consumers.py
import json
import base64
import requests
from channels.generic.websocket import AsyncWebsocketConsumer

VIDEO_URL = 'http://video-service:8002/detect-frame'

class PolypConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def receive(self, text_data=None, bytes_data=None):
        # Expect base64-encoded JPEG from client
        payload = json.loads(text_data)
        img_b64 = payload['frame']  # e.g. "data:image/jpeg;base64,…"

        # strip header
        header, data = img_b64.split(',',1)
        resp = requests.post(VIDEO_URL, json={'frame': data})
        detections = resp.json()['detections']

        # send back detections
        await self.send(text_data=json.dumps({
            'detections': detections
        }))
