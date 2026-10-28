"""Thin wrapper around the Gemini API."""

import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

# Aliases that always point to the newest models (both available on the free tier).
# If the main model is overloaded, we fall back to the lighter one.
CHAT_MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest"]
# Note: gemini-embedding-2 merges a list of texts into ONE vector, so we use 001,
# which returns one vector per text.
EMBEDDING_MODEL = "gemini-embedding-001"

# 429 = rate limit hit, 503 = model overloaded. Both are temporary, so worth retrying.
RETRYABLE_STATUS_CODES = {429, 503}
MAX_ATTEMPTS = 3


def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return genai.Client(api_key=api_key)


def _with_retry(call):
    """Run an API call, retrying temporary errors with exponential backoff (2s, 4s, ...)."""
    for attempt in range(MAX_ATTEMPTS):
        try:
            return call()
        except errors.APIError as e:
            if e.code not in RETRYABLE_STATUS_CODES or attempt == MAX_ATTEMPTS - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"API returned {e.code}, retrying in {wait}s...")
            time.sleep(wait)


def generate(contents, config=None):
    """Call the chat model, falling back to the next model if one stays unavailable."""
    client = get_client()
    for model in CHAT_MODELS:
        try:
            return _with_retry(
                lambda: client.models.generate_content(model=model, contents=contents, config=config)
            )
        except errors.APIError as e:
            if e.code not in RETRYABLE_STATUS_CODES:
                raise
            print(f"{model} is unavailable, trying the next model...")
    raise RuntimeError("All models are unavailable right now. Please try again later.")


def ask(prompt: str) -> str:
    """Send a single prompt to the chat model and return the reply text."""
    return generate(prompt).text


def embed(texts: list[str], task_type: str) -> list[list[float]]:
    """Turn texts into embedding vectors.

    task_type is "RETRIEVAL_DOCUMENT" for the texts we search in (the FAQ)
    and "RETRIEVAL_QUERY" for the user's question.
    """
    client = get_client()
    result = _with_retry(
        lambda: client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type),
        )
    )
    vectors = [e.values for e in result.embeddings]
    if len(vectors) != len(texts):
        raise RuntimeError(f"Expected {len(texts)} embeddings, got {len(vectors)}.")
    return vectors
