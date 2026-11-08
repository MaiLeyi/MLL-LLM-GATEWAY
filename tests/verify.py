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
from app.adapters.base import Completion, TokenUsage  # noqa: E402
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
    body: dict = {}
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
    check(
        "响应字段 / usage、latency_ms、attempts 完整",
        isinstance(body.get("usage"), dict)
        and body["usage"]["total"] > 0
        and isinstance(body.get("latency_ms"), int)
        and body.get("attempts", 0) >= 1
        and bool(body.get("request_id")),
        f"usage={body.get('usage')} latency_ms={body.get('latency_ms')} attempts={body.get('attempts')}",
    )


def check_multiturn() -> None:
    """1b. 多轮消息：user/assistant 角色完整传递；message 与 messages 互斥。"""
    response = client.post(
        "/chat",
        json={
            "model": "deepseek",
            "messages": [
                {"role": "user", "content": "请记住数字 42，只回答：好"},
                {"role": "assistant", "content": "好"},
                {"role": "user", "content": "我刚才让你记住的数字是什么？只回答数字"},
            ],
        },
    )
    body = response.json()
    check(
        "多轮消息 / 多角色上下文生效",
        response.status_code == 200 and "42" in str(body.get("reply", "")),
        f"status={response.status_code} reply={str(body.get('reply'))[:40]!r}",
    )

    both = client.post(
        "/chat",
        json={
            "model": "deepseek",
            "message": "hi",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    check(
        "多轮消息 / message 与 messages 互斥",
        both.status_code == 422,
        f"status={both.status_code} body={both.json()}",
    )


def check_stream() -> None:
    """2. 流式输出：标准 SSE 事件序列（response.created / content.delta / response.completed）。"""
    response = client.post(
        "/stream", json={"message": "从 1 数到 5，用逗号分隔", "model": "minimax"}
    )
    events = []
    for line in response.text.splitlines():
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[len("data: ") :]))
            except json.JSONDecodeError:
                pass
    types = [event.get("type") for event in events]
    deltas = [
        event.get("delta", "")
        for event in events
        if event.get("type") == "content.delta"
    ]
    completed = [event for event in events if event.get("type") == "response.completed"]
    check(
        "流式输出 / 标准 SSE 事件序列",
        response.status_code == 200
        and response.headers.get("content-type", "").startswith("text/event-stream")
        and types[0] == "response.created"
        and bool(types[0] and events[0].get("request_id"))
        and len(deltas) >= 2
        and len(completed) == 1
        and completed[0].get("model") == "minimax"
        and isinstance(completed[0].get("usage"), dict),
        f"first={types[0] if types else None} deltas={len(deltas)} completed={completed}",
    )
    check(
        "流式输出 / 推理块不泄漏",
        "<think>" not in "".join(deltas),
        f"contains_think={'<think>' in ''.join(deltas)}",
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

    # 指标明细应记录解析后的模板名与具体版本，便于按模板版本归因
    metric = next(
        (
            item
            for item in client.get("/metrics").json()["recent"]
            if item["request_id"] == response.json()["request_id"]
        ),
        None,
    )
    check(
        "模板管理 / 指标记录模板版本",
        metric is not None
        and metric.get("prompt_name") == "verify_demo"
        and metric.get("prompt_version") == "v1",
        f"prompt={metric.get('prompt_name') if metric else None}@{metric.get('prompt_version') if metric else None}",
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
    check(
        "可观测 / 成本累计",
        total["cost_usd"] > 0 and "fallbacks" in total["calls"],
        f"cost_usd={total['cost_usd']} fallbacks={total['calls'].get('fallbacks')}",
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


def check_fallback_and_cost() -> None:
    """7. 主备切换与成本核算：主模型持续故障时切备用模型，并按价格表折算成本。"""
    import time as _time

    class AlwaysDownAdapter:
        """主模型替身：模拟上游持续不可用。"""

        async def complete(self, *args, **kwargs) -> Completion:
            raise ConnectionError("primary down")

    class BackupAdapter:
        """备用模型替身：返回带 usage 的固定回答。"""

        async def complete(self, *args, **kwargs) -> Completion:
            return Completion(
                text="backup-ok",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )

    adapters = {"deepseek": AlwaysDownAdapter(), "minimax": BackupAdapter()}
    original = chat_app_module.get_adapter
    chat_app_module.get_adapter = lambda model_id: adapters[model_id]
    try:
        started = _time.perf_counter()
        response = client.post("/chat", json={"message": "hi", "model": "deepseek"})
        body = response.json()
        check(
            "主备切换 / 主模型故障切到备用",
            response.status_code == 200
            and body.get("reply") == "backup-ok"
            and body.get("model") == "minimax"
            and bool(body.get("request_id"))
            and body.get("attempts") == 5  # 主模型 1+3 次重试 + 备用 1 次
            and body.get("usage", {}).get("total") == 15,
            f"status={response.status_code} model={body.get('model')} "
            f"request_id={body.get('request_id')} attempts={body.get('attempts')} "
            f"elapsed={_time.perf_counter() - started:.1f}s",
        )

        recent = client.get("/metrics").json()["recent"]
        last_two = recent[-2:]
        check(
            "主备切换 / 同一 request_id 串联两次尝试",
            last_two[0]["model"] == "deepseek"
            and last_two[0]["success"] is False
            and last_two[1]["model"] == "minimax"
            and last_two[1]["success"] is True
            and last_two[0]["request_id"]
            == last_two[1]["request_id"]
            == body["request_id"],
            f"recent={[(r['model'], r['success']) for r in last_two]}",
        )

        minimax_price = settings.MODEL_CONFIGS["minimax"]
        expected = (
            10 * minimax_price.input_price_per_million
            + 5 * minimax_price.output_price_per_million
        ) / 1_000_000
        cost = last_two[1]["cost_usd"]
        check(
            "成本核算 / 按价格表折算",
            cost is not None and abs(cost - expected) < 1e-12,
            f"cost_usd={cost} expected={expected}",
        )
    finally:
        chat_app_module.get_adapter = original


def check_timeout() -> None:
    """8. 请求级超时：合法值生效，越界值返回 422。"""
    ok = client.post(
        "/chat",
        json={
            "message": "只回答两个字：好的",
            "model": "deepseek",
            "timeout_seconds": 60,
        },
    )
    check(
        "请求超时 / 自定义超时生效",
        ok.status_code == 200 and bool(ok.json().get("reply")),
        f"status={ok.status_code}",
    )
    for bad in (0, 121):
        invalid = client.post(
            "/chat",
            json={"message": "hi", "model": "deepseek", "timeout_seconds": bad},
        )
        check(
            f"请求超时 / 越界值 {bad} 返回 422",
            invalid.status_code == 422,
            f"status={invalid.status_code}",
        )


def main_entry() -> int:
    check_routing()
    check_multiturn()
    check_stream()
    check_structured()
    check_prompt()
    check_metrics()
    check_retry()
    check_rate_limit()
    check_fallback_and_cost()
    check_timeout()

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
