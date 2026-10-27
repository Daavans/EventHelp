"""Smoke test: send "Hello" to Gemini and print the reply."""

from src.llm import ask

if __name__ == "__main__":
    print(ask("Hello"))
