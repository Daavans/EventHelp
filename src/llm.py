"""Thin wrapper around the Gemini API."""

import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()

# Aliases that always point to the newest models (both available on the free tier).
# If the main model is overloaded, we fall back to the lighter one.
MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest"]

# 429 = rate limit hit, 503 = model overloaded. Both are temporary, so worth retrying.
RETRYABLE_STATUS_CODES = {429, 503}
MAX_ATTEMPTS = 3


def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return genai.Client(api_key=api_key)


def generate(contents, config=None):
    """Call the model with retries (exponential backoff) and a fallback model."""
    client = get_client()
    for model in MODELS:
        for attempt in range(MAX_ATTEMPTS):
            try:
                return client.models.generate_content(model=model, contents=contents, config=config)
            except errors.APIError as e:
                if e.code not in RETRYABLE_STATUS_CODES:
                    raise
                wait = 2 ** (attempt + 1)
                print(f"{model} returned {e.code}, retrying in {wait}s...")
                time.sleep(wait)
    raise RuntimeError("All models are unavailable right now. Please try again later.")


def ask(prompt: str) -> str:
    """Send a single prompt to the chat model and return the reply text."""
    return generate(prompt).text
