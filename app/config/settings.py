import os

from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY NOT EXISTS")

MODEL_NAME = "deepseek-flash"
MODEL_BASE_URL = "https://api.deepseek.com"

HOST = "0.0.0.0"
PORT = 8000
