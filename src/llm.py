"""Thin wrapper around the Gemini API."""

import os
import time

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

# Aliases that always point to the newest models (both available on the free tier).
# Flash-Lite is fast and handles our tools well; Flash is the fallback.
# (On the free tier Flash is often overloaded and returns 503.)
CHAT_MODELS = ["gemini-flash-lite-latest", "gemini-flash-latest"]
# Note: gemini-embedding-2 merges a list of texts into ONE vector, so we use 001,
# which returns one vector per text.
EMBEDDING_MODEL = "gemini-embedding-001"

# 429 = rate limit hit, 500/503/504 = server error, overloaded or timed out.
# All are temporary, so worth retrying.
RETRYABLE_STATUS_CODES = {429, 500, 503, 504}
MAX_ATTEMPTS = 3
REQUEST_TIMEOUT_MS = 30_000  # give up on a single request after 30 seconds


def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))


def _is_temporary(error: Exception) -> bool:
    """True for errors that may go away if we try again."""
    if isinstance(error, errors.APIError):
        return error.code in RETRYABLE_STATUS_CODES
    # Network problems such as a dropped connection or a timeout.
    return isinstance(error, httpx.TransportError)


def _with_retry(call):
    """Run an API call, retrying temporary errors with exponential backoff (2s, 4s, ...)."""
    for attempt in range(MAX_ATTEMPTS):
        try:
            return call()
        except (errors.APIError, httpx.TransportError) as e:
            if not _is_temporary(e) or attempt == MAX_ATTEMPTS - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"Temporary error ({e}), retrying in {wait}s...")
            time.sleep(wait)


def generate(contents, config=None):
    """Call the chat model, falling back to the next model if one stays unavailable."""
    client = get_client()
    for model in CHAT_MODELS:
        try:
            return _with_retry(
                lambda: client.models.generate_content(model=model, contents=contents, config=config)
            )
        except (errors.APIError, httpx.TransportError) as e:
            if not _is_temporary(e):
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
