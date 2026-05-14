"""
LLM API 调用封装
支持 debug 模式和自动重试
"""

import time
import json
from typing import Any

import anthropic


class LLMDebug:
    """LLM 调用调试信息"""

    def __init__(self):
        self.requests: list[dict] = []
        self.responses: list[dict] = []
        self.duration_ms: float = 0

    def add_request(self, request: dict):
        self.requests.append(request)

    def add_response(self, response: dict):
        self.responses.append(response)

    def add_duration(self, ms: float):
        self.duration_ms = ms

    def print_debug(self):
        """打印调试信息"""
        print("\n" + "=" * 60)
        print("🔍 LLM DEBUG INFO")
        print("=" * 60)

        print(f"\n⏱️  调用耗时: {self.duration_ms:.2f}ms")

        print(f"\n📤 发送的请求 ({len(self.requests)} 个):")
        for i, req in enumerate(self.requests):
            print(f"  --- Request {i + 1} ---")
            # 打印关键信息
            if "system" in req:
                print(f"  System: {req['system'][:100]}...")

            if "messages" in req:
                for msg in req["messages"]:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        for item in content:
                            if item.get("type") == "text":
                                text = item.get("text", "")[:200]
                                print(f"  Message [{role}]: {text}...")
                    elif isinstance(content, str):
                        print(f"  Message [{role}]: {content[:200]}...")

        print(f"\n📥 LLM 返回 ({len(self.responses)} 个):")
        for i, resp in enumerate(self.responses):
            print(f"  --- Response {i + 1} ---")
            content = resp.get("content", [])
            for block in content:
                if hasattr(block, 'type'):
                    if block.type == "text":
                        text = block.text[:300] if hasattr(block, 'text') else ""
                        print(f"  Text: {text}...")
                    elif block.type == "thinking":
                        thinking = block.thinking[:200] if hasattr(block, 'thinking') else ""
                        print(f"  Thinking: {thinking}...")

        print("\n" + "=" * 60 + "\n")


class LLMCaller:
    """LLM API 调用封装"""

    def __init__(
        self,
        model: str = "MiniMax-M2.7",
        max_tokens: int = 2000,
        debug: bool = False,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.debug = debug
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = anthropic.Anthropic()
        self._debug_info: LLMDebug | None = None

    def get_debug_info(self) -> LLMDebug | None:
        return self._debug_info

    def call(self, system: str, prompt: str, retries: int | None = None) -> list[dict]:
        """
        调用 LLM API，支持自动重试

        Args:
            system: 系统提示词
            prompt: 用户输入
            retries: 重试次数，默认使用 self.max_retries

        Returns:
            message.content 列表
        """
        if retries is None:
            retries = self.max_retries

        if self.debug:
            self._debug_info = LLMDebug()

        last_error: Exception | None = None

        for attempt in range(retries + 1):
            try:
                start_time = time.time()

                request = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "system": system,
                    "messages": [{
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
                    }]
                }

                if self.debug:
                    self._debug_info.add_request(request)

                message = self.client.messages.create(**request)

                duration_ms = (time.time() - start_time) * 1000

                if self.debug:
                    self._debug_info.add_response(message.model_dump())
                    self._debug_info.add_duration(duration_ms)
                    self._debug_info.print_debug()

                return message.content

            except Exception as e:
                last_error = e
                if attempt < retries:
                    time.sleep(self.retry_delay)
                    continue
                raise last_error

        raise last_error or RuntimeError("LLM 调用失败")


def call_llm(
    system: str,
    prompt: str,
    model: str = "MiniMax-M2.7",
    max_tokens: int = 2000,
    debug: bool = False,
    max_retries: int = 3
) -> list[dict]:
    """
    快捷函数：直接调用 LLM
    """
    caller = LLMCaller(
        model=model,
        max_tokens=max_tokens,
        debug=debug,
        max_retries=max_retries
    )
    return caller.call(system, prompt)