"""可替换的结构化聊天客户端及 DeepSeek 兼容实现。"""

from __future__ import annotations

import json
import os
from typing import Protocol
from urllib.request import Request, urlopen


class ChatClient(Protocol):
    def complete_json(self, messages: list[dict]) -> dict: ...


class DeepSeekChatClient:
    def __init__(
        self, *, model: str = "deepseek-chat", base_url: str = "https://api.deepseek.com",
        api_key: str | None = None, temperature: float = 0.0, max_tokens: int = 1024,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
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
