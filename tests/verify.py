"""验证脚本：覆盖统一路由、流式输出、结构化输出、模板版本、可观测与韧性。

运行方式（在项目根目录）：
    ./.venv/bin/python tests/verify.py

脚本会真实调用 DeepSeek / MiniMax（需要 .env 中的 Key），
重试与限流通过进程内替身/临时配置验证，不会额外消耗上游配额。
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from fastapi.testclient import TestClient  # noqa: E402

import app.application.chat_app_service as chat_app_module  # noqa: E402
import app.config.settings as settings  # noqa: E402
import main  # noqa: E402
from app.adapters.base import Completion  # noqa: E402
from app.services.rate_limit_service import rate_limit_service  # noqa: E402
from app.services.retry_service import RetryService  # noqa: E402

client = TestClient(main.app)
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  |  {detail}")


def new_user_object_prompt() -> str:
    return "请返回一个用户对象的 JSON，必须包含字符串字段 name 和整数字段 age，不要输出其他文字。"


def check_routing() -> None:
    """1. 统一抽象层：同一套接口按 model 路由到两个上游。"""
    for model in ("deepseek", "minimax"):
        response = client.post(
            "/chat", json={"message": "请只回答两个字：你好", "model": model}
        )
        body = response.json()
        check(
            f"统一抽象层 / 模型路由 {model}",
            response.status_code == 200 and bool(body.get("reply")),
            f"status={response.status_code} reply={str(body.get('reply'))[:30]!r}",
        )


def check_stream() -> None:
    """2. 流式输出：SSE 逐块返回并以 [DONE] 结束。"""
    response = client.post(
        "/stream", json={"message": "从 1 数到 5，用逗号分隔", "model": "minimax"}
    )
    chunks = [
        line[len("data: ") :]
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    contents = []
    for chunk in chunks:
        if chunk == "[DONE]":
            continue
        try:
            contents.append(json.loads(chunk).get("content", ""))
        except json.JSONDecodeError:
            pass
    check(
        "流式输出 / SSE 分块",
        response.status_code == 200
        and response.headers.get("content-type", "").startswith("text/event-stream")
        and len(contents) >= 2
        and chunks[-1] == "[DONE]",
        f"chunks={len(contents)} done={chunks[-1] == '[DONE]'} text={''.join(contents)[:60]!r}",
    )
    check(
        "流式输出 / 推理块不泄漏",
        "<think>" not in "".join(contents),
        f"contains_think={'<think>' in ''.join(contents)}",
    )


def check_structured() -> None:
    """3. 结构化输出：json_object 与 json_schema（含 schema 校验）。"""
    response = client.post(
        "/chat",
        json={
            "message": new_user_object_prompt(),
            "model": "deepseek",
            "response_format": {"type": "json_object"},
        },
    )
    body = response.json()
    check(
        "结构化输出 / json_object(deepseek)",
        response.status_code == 200 and isinstance(body.get("data"), dict),
        f"status={response.status_code} data={body.get('data')}",
    )

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    response = client.post(
        "/chat",
        json={
            "message": new_user_object_prompt(),
            "model": "deepseek",
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "user", "schema": schema, "strict": True},
            },
        },
    )
    body = response.json()
    check(
        "结构化输出 / json_schema + schema 校验",
        response.status_code == 200
        and isinstance(body.get("data"), dict)
        and isinstance(body["data"].get("age"), int),
        f"status={response.status_code} data={body.get('data')}",
    )

    # MiniMax 的推理块已被剥离后，json_object 也应能稳定解析
    response = client.post(
        "/chat",
        json={
            "message": new_user_object_prompt(),
            "model": "minimax",
            "response_format": {"type": "json_object"},
        },
    )
    body = response.json()
    check(
        "结构化输出 / json_object(minimax 推理块剥离)",
        response.status_code == 200 and isinstance(body.get("data"), dict),
        f"status={response.status_code} data={body.get('data')}",
    )

    class BrokenJsonAdapter:
        """返回非 JSON 的替身，用于验证网关侧输出契约。"""

        async def complete(self, *args, **kwargs) -> Completion:
            return Completion(text="这不是 JSON")

    original = chat_app_module.get_adapter
    chat_app_module.get_adapter = lambda model_id: BrokenJsonAdapter()
    try:
        response = client.post(
            "/chat",
            json={
                "message": "随便说点什么",
                "model": "deepseek",
                "response_format": {"type": "json_object"},
            },
        )
        check(
            "结构化输出 / 非法 JSON 被拦截",
            response.status_code == 502
            and response.json().get("code") == "INVALID_MODEL_OUTPUT",
            f"status={response.status_code} body={response.json()}",
        )
    finally:
        chat_app_module.get_adapter = original


def check_prompt() -> None:
    """4. 提示词版本管理：存储、latest 解析、变量替换与错误处理。"""
    created = client.post(
        "/prompts",
        json={
            "name": "verify_demo",
            "version": "v1",
            "template": "你是${role}，请只回答：${answer}",
        },
    )
    check(
        "模板管理 / 发布版本",
        created.status_code == 201 and created.json()["version"] == "v1",
        f"status={created.status_code} body={created.json()}",
    )

    latest = client.get("/prompts/verify_demo/latest")
    check(
        "模板管理 / latest 解析",
        latest.status_code == 200 and latest.json()["is_latest"] is True,
        f"status={latest.status_code} version={latest.json().get('version')}",
    )

    response = client.post(
        "/chat",
        json={
            "message": "测试模板引用",
            "model": "deepseek",
            "prompt": {
                "name": "verify_demo",
                "version": "v1",
                "variables": {"role": "助手", "answer": "OK"},
            },
        },
    )
    check(
        "模板管理 / 版本引用 + 变量替换",
        response.status_code == 200 and bool(response.json().get("reply")),
        f"status={response.status_code} reply={str(response.json().get('reply'))[:40]!r}",
    )

    missing = client.post(
        "/chat",
        json={
            "message": "测试缺变量",
            "model": "deepseek",
            "prompt": {"name": "verify_demo", "version": "v1", "variables": {}},
        },
    )
    check(
        "模板管理 / 缺少变量返回 400",
        missing.status_code == 400
        and missing.json().get("code") == "INVALID_REQUEST",
        f"status={missing.status_code} body={missing.json()}",
    )

    unknown = client.post(
        "/chat",
        json={
            "message": "测试未知版本",
            "model": "deepseek",
            "prompt": {"name": "verify_demo", "version": "v9", "variables": {}},
        },
    )
    check(
        "模板管理 / 未知版本返回 404",
        unknown.status_code == 404,
        f"status={unknown.status_code} body={unknown.json()}",
    )


def check_metrics() -> None:
    """5. 可观测性：Token 分类、总延迟、首 Token 延迟与重试次数。"""
    metrics = client.get("/metrics").json()
    total = metrics["total"]
    by_model = metrics["by_model"]
    recent = metrics["recent"]
    stream_calls = [item for item in recent if item["stream"]]
    check(
        "可观测 / Token 分类统计",
        total["tokens"]["prompt"] > 0
        and total["tokens"]["completion"] > 0
        and total["tokens"]["total"] > 0,
        f"tokens={total['tokens']}",
    )
    check(
        "可观测 / 按模型分组",
        {"deepseek", "minimax"}.issubset(set(by_model)),
        f"models={sorted(by_model)}",
    )
    check(
        "可观测 / 首 Token 延迟",
        bool(stream_calls) and stream_calls[-1]["first_token_ms"] is not None,
        f"first_token_ms={stream_calls[-1]['first_token_ms'] if stream_calls else None}",
    )
    check(
        "可观测 / 延迟与重试计数",
        total["latency_ms"]["avg"] > 0
        and "retries" in total["calls"]
        and "retries" in recent[-1],
        f"latency_avg={total['latency_ms']['avg']} retries={total['calls'].get('retries')}",
    )


def check_retry() -> None:
    """6a. 指数退避重试：瞬时故障两次后第三次成功。"""
    service = RetryService(max_retries=3, base_delay=0.01, max_delay=0.02)
    calls = {"n": 0}

    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient")
        return "ok"

    import asyncio

    result = asyncio.run(service.run(flaky))
    check(
        "韧性 / 指数退避重试",
        result == "ok" and calls["n"] == 3,
        f"result={result} attempts={calls['n']}",
    )

    # 通过 HTTP 层验证：上游前两次失败，网关仍返回 200，并把重试次数写入指标
    class FlakyAdapter:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, *args, **kwargs) -> Completion:
            self.calls += 1
            if self.calls < 3:
                raise ConnectionError("transient upstream")
            return Completion(text="retry-ok")

    adapter = FlakyAdapter()
    original = chat_app_module.get_adapter
    chat_app_module.get_adapter = lambda model_id: adapter
    try:
        response = client.post("/chat", json={"message": "hi", "model": "deepseek"})
        metric = client.get("/metrics").json()["recent"][-1]
        check(
            "韧性 / HTTP 调用自动重试且指标留痕",
            response.status_code == 200
            and response.json()["reply"] == "retry-ok"
            and adapter.calls == 3
            and metric["retries"] == 2,
            f"status={response.status_code} attempts={adapter.calls} retries={metric['retries']}",
        )
    finally:
        chat_app_module.get_adapter = original


def check_rate_limit() -> None:
    """6b. 按模型独立限流：超限返回 429 + Retry-After。"""
    config = settings.MODEL_CONFIGS["minimax"]
    original_limit = config.rate_limit_per_minute
    rate_limit_service._hits["minimax"].clear()  # 清空历史窗口，保证可复现
    object.__setattr__(config, "rate_limit_per_minute", 1)
    try:
        first = client.post("/chat", json={"message": "限流测试", "model": "minimax"})
        second = client.post("/chat", json={"message": "限流测试", "model": "minimax"})
        check(
            "韧性 / 超限返回 429 + Retry-After",
            first.status_code == 200
            and second.status_code == 429
            and second.json().get("code") == "RATE_LIMITED"
            and second.headers.get("retry-after") is not None,
            f"first={first.status_code} second={second.status_code} "
            f"retry-after={second.headers.get('retry-after')}",
        )
    finally:
        object.__setattr__(config, "rate_limit_per_minute", original_limit)


def main_entry() -> int:
    check_routing()
    check_stream()
    check_structured()
    check_prompt()
    check_metrics()
    check_retry()
    check_rate_limit()

    failed = [name for name, ok, _ in results if not ok]
    print("\n" + "=" * 60)
    print(f"共 {len(results)} 项，通过 {len(results) - len(failed)} 项，失败 {len(failed)} 项")
    if failed:
        print("失败项：")
        for name in failed:
            print(f"  - {name}")
    print("=" * 60)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main_entry())
