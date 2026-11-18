# MLL LLM Gateway

统一的多模型调用网关：用一套 HTTP 接口接入不同供应商，按 `model` 动态路由，屏蔽鉴权、请求体与返回格式差异。

当前接入：

| model id | 协议 | 上游模型 | 能力 |
| --- | --- | --- | --- |
| `deepseek` | OpenAI Chat Completions | `deepseek-flash` | 流式、JSON 输出、Token 分类统计 |
| `minimax` | OpenAI Chat Completions | `MiniMax-M3` | 流式、JSON 输出、Token 分类统计 |

此外 `app/adapters/anthropic_adapter.py` 提供了 Anthropic Messages 协议适配器（`x-api-key` 鉴权、顶层 `system`、content block 解析），新增模型时把配置里的 `protocol` 设为 `anthropic` 即可启用。

## 功能

- **统一抽象层**：`app/adapters/` 定义统一 `LLMAdapter` 接口，`ADAPTER_REGISTRY` + 运行时模型注册表的 `protocol` 完成动态路由。
- **模型配置管理**：控制台 Models 页面以卡片展示已配置模型，支持新增、编辑、删除、设为默认，并可发起真实的上游可用性测试；配置保存到用户数据目录 `models.json`，热更新生效。
- **API Key 存储**：界面填写的 Key 写入 macOS 系统钥匙串，`models.json` 只保存非敏感字段；未写入钥匙串时回退读取 `.env`，所有接口仅返回掩码。
- **多轮消息**：`message`（单条简写）与 `messages`（多轮，含 system / user / assistant 角色）二选一，角色原样透传上游。
- **流式输出**：`POST /stream`，标准 SSE 事件序列 `response.created` → `content.delta`* → `response.completed`（含真实 usage 与延迟），异常时下发 `response.failed`；自动剥离 `<think>...</think>` 推理块。
- **结构化输出**：`response_format` 支持 `text` / `json_object` / `json_schema`；上游不支持原生 `json_schema` 时自动降级为 `json_object` 并注入 schema 说明，网关侧用 `jsonschema` 二次校验。
- **提示词版本管理**：模板按 `name@version` 存储，`latest` 解析为最新版本，支持 `${变量}` 替换；版本不可覆盖。
- **可观测性**：每次调用生成 `request_id`（主备多次尝试共享，可串联 `/metrics` 明细）；记录 prompt / completion / total / cached / reasoning Token、总延迟、首 Token 延迟、重试次数、主备切换次数与成本；`GET /metrics` 返回全局、按模型、最近调用明细。
- **主备切换**：模型配置中的 `fallback_model` 组成备用链（默认 deepseek → minimax），主模型重试耗尽后自动切换；仅上游故障（超时 / 5xx / 网络错误）触发，本网关 429 与输出契约错误不切换；流式已产出内容后不再切换。
- **成本核算**：按模型价格表（`input/output_price_per_million`）折算每次调用 `cost_usd`，并在 `/metrics` 聚合累计。
- **韧性**：统一错误码；指数退避重试（0.5s → 1s → 2s，最多重试 3 次，仅 408/429/5xx 与网络类异常）；按模型独立滑动窗口限流，超限返回 `429` + `Retry-After`。
- **请求级超时**：`timeout_seconds`（默认 30，范围 0–120）逐请求下传上游，重试与主备切换的每次尝试均生效。

## 启动

要求 Python 3.13（与当前虚拟环境一致）。

```bash
cd /path/to/mll-llm-gateway

python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# 配置上游 Key
cat > .env <<'EOF'
DEEPSEEK_API_KEY=sk-xxx
MINI_MAX_API_KEY=xxx
EOF

# 启动（默认 0.0.0.0:8000）
./.venv/bin/python main.py
# 或
./.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

启动时不再强制校验模型 Key；请求时若缺失会返回 `400 MISSING_API_KEY`，可在控制台 Models 页面填写（存入 macOS 钥匙串），或在 `.env` 中配置。

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 零依赖 Web 控制台（浏览器直接对话、看指标） |
| GET | `/models` | 可用模型 id、默认模型与卡片详情（Key 仅返回掩码） |
| POST | `/models` | 新增模型配置 |
| PUT | `/models/{id}` | 更新模型配置 |
| DELETE | `/models/{id}` | 删除模型（默认模型或被引用为备用时拒绝） |
| POST | `/models/{id}/default` | 设为默认模型 |
| POST | `/models/{id}/test` | 真实调用上游做一次最小可用性测试 |
| POST | `/chat` | 一次性返回完整回答，支持多轮 `messages`、`response_format`、`prompt` 版本引用与 `timeout_seconds` |
| POST | `/stream` | SSE 流式回答，支持 `prompt` 版本引用与 `timeout_seconds`（不支持 `response_format`） |
| POST | `/prompts` | 发布一个模板版本 |
| GET | `/prompts?name=` | 列出模板版本 |
| GET | `/prompts/{name}/{version}` | 查询模板详情，`version` 可传 `latest` |
| GET | `/metrics` | 查询 Token、延迟、首 Token 延迟与最近调用 |

### 1. 基础调用

```bash
curl -s http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"请只回答两个字：你好","model":"deepseek"}'
```

```json
{
  "reply": "你好",
  "data": null,
  "request_id": "ab207269-6e6f-4f2f-9a3d-0f3f2c1e9a11",
  "model": "deepseek",
  "usage": {"prompt": 37, "completion": 60, "total": 97, "cached": 0, "reasoning": 46},
  "latency_ms": 2550,
  "attempts": 1
}
```

`model` 为实际应答模型（发生主备切换时与请求模型不同）；`attempts` 为上游调用总次数（含重试与切换）；`request_id` 可在 `/metrics` 明细中串联本次调用。

### 2. 多轮消息与请求级超时

`messages` 支持 system / user / assistant 三种角色，与 `message` 二选一：

```bash
curl -s http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek",
    "messages": [
      {"role": "user", "content": "请记住数字 42，只回答：好"},
      {"role": "assistant", "content": "好"},
      {"role": "user", "content": "我刚才让你记住的数字是什么？只回答数字"}
    ],
    "timeout_seconds": 60
  }'
```

`timeout_seconds` 逐请求生效（默认 30，范围 0–120），越界返回 `422`。

### 3. 流式输出

```bash
curl -N -s http://127.0.0.1:8000/stream \
  -H 'Content-Type: application/json' \
  -d '{"message":"从 1 数到 5，用逗号分隔","model":"minimax"}'
```

```
data: {"type": "response.created", "request_id": "0c94744d-...", "model": "minimax"}

data: {"type": "content.delta", "delta": "1"}

data: {"type": "content.delta", "delta": ",2"}

...

data: {"type": "response.completed", "model": "minimax", "request_id": "0c94744d-...", "usage": {"prompt": 206, "completion": 59, "total": 265, "cached": 205, "reasoning": 47}, "latency_ms": 1573}
```

`response.completed` 携带真实 Token 用量与总延迟；流建立后的异常以 `{"type": "response.failed", "error": {...}}` 事件下发。

### 4. 结构化输出

JSON 对象：

```bash
curl -s http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "返回一个用户对象的 JSON，必须包含字符串 name 和整数 age",
    "model": "deepseek",
    "response_format": {"type": "json_object"}
  }'
```

```json
{"reply": "{\"name\": \"张三\", \"age\": 30}", "data": {"name": "张三", "age": 30}, "request_id": "...", "model": "deepseek", "usage": {"prompt": 80, "completion": 12, "total": 92, "cached": 0, "reasoning": 0}, "latency_ms": 1820, "attempts": 1}
```

JSON Schema（当前两个模型会降级为 json_object，但仍会按 schema 校验）：

```bash
curl -s http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "返回一个用户对象的 JSON，必须包含字符串 name 和整数 age",
    "model": "deepseek",
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "user",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
          },
          "required": ["name", "age"]
        }
      }
    }
  }'
```

模型输出非法 JSON 或不符合 schema 时返回 `502 INVALID_MODEL_OUTPUT`。

### 5. 提示词版本

发布并引用模板：

```bash
# 发布 v1
curl -s -X POST http://127.0.0.1:8000/prompts \
  -H 'Content-Type: application/json' \
  -d '{"name":"demo","version":"v1","template":"你是${role}，请只回答：${answer}"}'

# 引用 v1 调用
curl -s http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "这是一次模板引用测试",
    "model": "deepseek",
    "prompt": {
      "name": "demo",
      "version": "v1",
      "variables": {"role": "助手", "answer": "OK"}
    }
  }'

# 引用 latest
curl -s http://127.0.0.1:8000/prompts/demo/latest
```

- 版本已存在 → `400`；`latest` 不能作为发布版本号。
- 模板或版本不存在 → `404`；变量缺失 → `400`。

内置模板 `knowledge_decision@v1` 可直接引用。

### 6. 可观测数据

```bash
curl -s http://127.0.0.1:8000/metrics | python3 -m json.tool
```

返回结构：

```json
{
  "total": {
    "calls": {"total": 8, "success": 8, "failure": 0, "stream": 1, "complete": 7, "retries": 2, "fallbacks": 1},
    "tokens": {"calls_with_usage": 8, "prompt": 1013, "completion": 368, "total": 1381, "cached": 434, "reasoning": 311},
    "latency_ms": {"avg": 1661.07, "max": 5516.38},
    "first_token_ms": {"avg": 1396.74, "max": 1396.74},
    "cost_usd": 0.000711
  },
  "by_model": {"deepseek": {}, "minimax": {}},
  "recent": [
    {"timestamp": "...", "request_id": "0c94744d-...", "requested_model": "minimax",
     "prompt_name": "verify_demo", "prompt_version": "v1",
     "model": "minimax", "stream": true, "success": true,
     "latency_ms": 1573.0, "first_token_ms": 1396.74, "tokens": {}, "retries": 0,
     "error_code": null, "cost_usd": 0.000021}
  ]
}
```

`requested_model` 与 `model` 不同即发生了主备切换；`prompt_name/prompt_version` 记录引用的模板版本（`latest` 会解析为具体版本），便于按模板归因质量与成本。

每次调用同时会输出一条 `llm_call` 结构化日志，便于外部采集。

### 7. 限流

单模型每分钟限额在 `app/config/settings.py` 配置（默认 deepseek 60、minimax 30）。超限：

```json
{"code": "RATE_LIMITED", "message": "模型 minimax 请求过于频繁，限额 30 次/分钟，请 30 秒后重试"}
```

响应头带 `Retry-After`（秒）。触发限流不会引发主备切换（429 是配额治理决策，调用方应按 `Retry-After` 退避）。

### 8. 主备切换与成本核算

`app/config/settings.py` 中每个模型可配置 `fallback_model` 与价格表：

- 主模型重试耗尽后自动切换备用模型，响应的 `model` 字段反映实际应答模型，`attempts` 计入全部尝试。
- 每次调用按 `input/output_price_per_million` 折算 `cost_usd`，写入 `/metrics` 明细与聚合（含 `fallbacks` 切换次数）。
- 主备链在启动时校验：指向未配置模型会直接报错。

## 错误码

| code | HTTP | 说明 |
| --- | --- | --- |
| `INVALID_REQUEST` | 400 / 422 | 入参不合法、模板变量缺失 |
| `NOT_FOUND` | 404 | 模板或版本不存在 |
| `UNSUPPORTED_MODEL` | 400 | `model` 不在配置中 |
| `RATE_LIMITED` | 429 | 触发单模型限流 |
| `UPSTREAM_TIMEOUT` | 504 | 上游超时 |
| `UPSTREAM_ERROR` | 502 | 上游调用失败 |
| `INVALID_MODEL_OUTPUT` | 502 | 模型输出不是合法 JSON 或不符合 schema |
| `INTERNAL_ERROR` | 500 | 未处理异常 |

流式接口在响应建立后的异常会以 `{"type": "response.failed", "error": {...}}` 事件下发。

## 验证

```bash
./.venv/bin/python tests/verify.py
```

脚本会真实调用两个上游，覆盖 31 个检查点（重试 / 限流 / 主备切换通过进程内替身与临时配置验证，不额外消耗上游配额）：

1. 统一路由：`deepseek` / `minimax` 均可调用；响应字段（request_id / usage / latency_ms / attempts）完整
2. 多轮消息：多角色上下文生效；`message` 与 `messages` 互斥（422）
3. 流式：标准 SSE 事件序列（response.created / content.delta / response.completed）、真实 usage、推理块不泄漏
4. 结构化：json_object、json_schema + schema 校验、非法 JSON 拦截（502）
5. 模板：发布、latest、变量替换、指标记录模板版本、缺变量 400、未知版本 404
6. 可观测：Token 分类、按模型分组、首 Token 延迟、重试计数、成本累计
7. 韧性：指数退避重试（含 HTTP 层 + 指标留痕）、429 + Retry-After
8. 主备切换：主模型故障切备用、同一 request_id 串联两次尝试、成本按价格表折算
9. 请求超时：自定义超时生效、越界值 422

全部通过时以退出码 0 结束，失败项会列在末尾。

## 目录结构

```
app/
  adapters/        统一接口 + OpenAI / Anthropic 适配器与协议注册表
  api/v1/          chat / stream / prompts / metrics / models 路由
  application/     应用服务：校验、限流、重试、主备切换、结构化校验、指标编排
  config/          模型（含主备链与价格表）、模板与全局配置
  core/            统一错误码与异常处理器
  schemas/         Pydantic 入参/出参模型
  services/        重试、限流、指标、模板版本仓库
static/index.html  零依赖 Web 控制台（GET / 直接返回）
tests/verify.py    验收脚本
main.py            FastAPI 入口
```

## 已知限制

- 模板仓库、限流窗口、指标聚合都在进程内存中：重启丢失，多副本部署时各自独立。生产化需替换为 Redis / 数据库 / Prometheus。
- 当前两个模型都走 OpenAI 兼容协议；Anthropic 适配器已实现但未绑定到任何模型配置。
- 上游返回 `4xx`（如参数错误）统一映射为 `502 UPSTREAM_ERROR`，调用方需结合 message 判断根因。
- `/stream` 不支持 `response_format`（由 schema 的 `extra="forbid"` 在入参阶段拒绝），流式与结构化暂不能组合。
