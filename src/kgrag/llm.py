"""可替换的结构化聊天客户端及 DeepSeek 兼容实现。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol
from urllib.request import Request, urlopen


class ChatClient(Protocol):
    def complete_json(self, messages: list[dict]) -> dict: ...


def _local_api_key() -> str | None:
    env_file = Path.cwd() / ".env"
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None
    for line in lines:
        name, separator, value = line.partition("=")
        if separator and name.strip() == "DEEPSEEK_API_KEY":
            value = value.strip()
            if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
                value = value[1:-1]
            return value or None
    return None


class DeepSeekChatClient:
    def __init__(
        self, *, model: str = "deepseek-flash", base_url: str = "https://api.deepseek.com",
        api_key: str | None = None, temperature: float = 0.0, max_tokens: int = 1024,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or _local_api_key()
        if not self.api_key:
            raise RuntimeError("请设置 DEEPSEEK_API_KEY")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.last_model = model
        self.last_usage: dict = {}

    def complete_json(self, messages: list[dict]) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.base_url.startswith("https://api.deepseek.com"):
            payload["thinking"] = {"type": "disabled"}
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=90) as response:
            data = json.load(response)
        choice = data["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ValueError("生成输出被截断")
        content = choice["message"].get("content")
        if not content or not content.strip():
            raise ValueError("生成接口返回空 JSON")
        self.last_model = data.get("model") or self.model
        self.last_usage = data.get("usage", {})
        result = json.loads(content)
        if not isinstance(result, dict):
            raise ValueError("生成接口未返回 JSON 对象")
        return result
