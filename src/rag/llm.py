import hashlib
import json
import time
from pathlib import Path

from langchain_openai import ChatOpenAI
from openai import RateLimitError

from rag.config import Settings, load_settings

Messages = list[dict[str, str]]


def cache_key(model: str, messages: Messages, temperature: float) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages, "temperature": temperature},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_file(settings: Settings, key: str) -> Path:
    return settings.cache_dir / f"{key}.json"


def _call_groq(model: str, messages: Messages, temperature: float, settings: Settings) -> str:
    client = ChatOpenAI(
        model=model,
        api_key=settings.groq_api_key,
        base_url=settings.groq_base_url,
        temperature=temperature,
        max_retries=0,
    )
    return str(client.invoke(messages).content)


def chat(
    model: str,
    messages: Messages,
    temperature: float = 0.0,
    settings: Settings | None = None,
    max_attempts: int = 5,
) -> str:
    settings = settings or load_settings()
    key = cache_key(model, messages, temperature)
    path = cache_file(settings, key)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["response"]

    for attempt in range(1, max_attempts + 1):
        try:
            text = _call_groq(model, messages, temperature, settings)
            break
        except RateLimitError:
            if attempt == max_attempts:
                raise
            wait = 10 * attempt
            print(f"rate limited, waiting {wait}s (attempt {attempt}/{max_attempts})")
            time.sleep(wait)

    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    record = {"model": model, "messages": messages, "temperature": temperature, "response": text}
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return text


if __name__ == "__main__":
    s = load_settings()
    prompt: Messages = [{"role": "user", "content": "Reply with the single word: hello"}]
    for name, model in [("generation", s.gen_model), ("judge", s.judge_model)]:
        for round_no in (1, 2):
            hit = cache_file(s, cache_key(model, prompt, 0.0)).exists()
            t0 = time.perf_counter()
            reply = chat(model, prompt, settings=s)
            ms = (time.perf_counter() - t0) * 1000
            status = "cache hit" if hit else "API call"
            print(f"{name:10s} call {round_no}: {status:9s} {ms:7.0f} ms -> {reply.strip()[:60]!r}")
