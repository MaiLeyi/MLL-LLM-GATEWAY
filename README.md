# MLL LLM Gateway

统一的多模型调用网关：用一套 HTTP 接口接入不同供应商，按 `model` 动态路由，屏蔽鉴权、请求体与返回格式差异。

当前接入：

| model id | 协议 | 上游模型 | 能力 |
| --- | --- | --- | --- |
| `deepseek` | OpenAI Chat Completions | `deepseek-flash` | 流式、JSON 输出、Token 分类统计 |
| `minimax` | OpenAI Chat Completions | `MiniMax-M3` | 流式、JSON 输出、Token 分类统计 |

此外 `app/adapters/anthropic_adapter.py` 提供了 Anthropic Messages 协议适配器（`x-api-key` 鉴权、顶层 `system`、content block 解析），新增模型时把配置里的 `protocol` 设为 `anthropic` 即可启用。

## 功能

- **统一抽象层**：`app/adapters/` 定义统一 `LLMAdapter` 接口，`ADAPTER_REGISTRY` + `MODEL_CONFIGS[model].protocol` 完成动态路由。
- **流式输出**：`POST /stream`，SSE（`data: {...}` + `[DONE]`），并自动剥离 `<think>...</think>` 推理块。
- **结构化输出**：`response_format` 支持 `text` / `json_object` / `json_schema`；上游不支持原生 `json_schema` 时自动降级为 `json_object` 并注入 schema 说明，网关侧用 `jsonschema` 二次校验。
- **提示词版本管理**：模板按 `name@version` 存储，`latest` 解析为最新版本，支持 `${变量}` 替换；版本不可覆盖。
- **可观测性**：记录 prompt / completion / total / cached / reasoning Token、总延迟、首 Token 延迟、重试次数；`GET /metrics` 返回全局、按模型、最近调用明细。
- **韧性**：统一错误码；指数退避重试（0.5s → 1s → 2s，最多重试 3 次，仅 408/429/5xx 与网络类异常）；按模型独立滑动窗口限流，超限返回 `429` + `Retry-After`。

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

启动时会校验默认模型（`deepseek`）的 Key，缺失会直接报错，避免服务起来却无法应答。

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/chat` | 一次性返回完整回答，支持 `response_format` 与 `prompt` 版本引用 |
| POST | `/stream` | SSE 流式回答，支持 `prompt` 版本引用（不支持 `response_format`） |
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
{"reply": "你好", "data": null}
```

### 2. 流式输出

```bash
curl -N -s http://127.0.0.1:8000/stream \
  -H 'Content-Type: application/json' \
  -d '{"message":"从 1 数到 5，用逗号分隔","model":"minimax"}'
```

```
data: {"content": "1"}
data: {"content": ",2"}
...
data: [DONE]
```

### 3. 结构化输出

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
{"reply": "{\"name\": \"张三\", \"age\": 30}", "data": {"name": "张三", "age": 30}}
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

### 4. 提示词版本

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

### 5. 可观测数据

```bash
curl -s http://127.0.0.1:8000/metrics | python3 -m json.tool
```

返回结构：

```json
{
  "total": {
    "calls": {"total": 8, "success": 8, "failure": 0, "stream": 1, "complete": 7, "retries": 2},
    "tokens": {"calls_with_usage": 8, "prompt": 955, "completion": 338, "total": 1293, "cached": 562, "reasoning": 289},
    "latency_ms": {"avg": 3063.28, "max": 5516.38},
    "first_token_ms": {"avg": 1895.28, "max": 1895.28}
  },
  "by_model": {"deepseek": {}, "minimax": {}},
  "recent": [
    {"timestamp": "...", "model": "minimax", "stream": true, "success": true,
     "latency_ms": 2201.5, "first_token_ms": 1895.28, "tokens": {}, "retries": 0, "error_code": null}
  ]
}
```

每次调用同时会输出一条 `llm_call` 结构化日志，便于外部采集。

### 6. 限流

单模型每分钟限额在 `app/config/settings.py` 配置（默认 deepseek 60、minimax 30）。超限：

```json
{"code": "RATE_LIMITED", "message": "模型 minimax 请求过于频繁，限额 30 次/分钟，请 30 秒后重试"}
```

响应头带 `Retry-After`（秒）。

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

流式接口在响应建立后的异常会以 `data: {"error": {...}}` 事件下发，随后仍发送 `[DONE]`。

## 验证

```bash
./.venv/bin/python tests/verify.py
```

脚本会真实调用两个上游，并覆盖 20 个检查点：

1. 统一路由：`deepseek` / `minimax` 均可调用
2. 流式：SSE 分块、`[DONE]`、推理块不泄漏
3. 结构化：json_object、json_schema + schema 校验、非法 JSON 拦截
4. 模板：发布、latest、变量替换、缺变量 400、未知版本 404
5. 可观测：Token 分类、按模型分组、首 Token 延迟、重试计数
6. 韧性：指数退避重试（含 HTTP 层 + 指标留痕）、429 + Retry-After

全部通过时以退出码 0 结束，失败项会列在末尾。

## 目录结构

```
app/
  adapters/        统一接口 + OpenAI / Anthropic 适配器与协议注册表
  api/v1/          chat / stream / prompts / metrics 路由
  application/     应用服务：校验、限流、重试、结构化校验、指标编排
  config/          模型、模板与全局配置
  core/            统一错误码与异常处理器
  schemas/         Pydantic 入参/出参模型
  services/        重试、限流、指标、模板版本仓库
tests/verify.py    验收脚本
main.py            FastAPI 入口
```

## 已知限制

- 模板仓库、限流窗口、指标聚合都在进程内存中：重启丢失，多副本部署时各自独立。生产化需替换为 Redis / 数据库 / Prometheus。
- 当前两个模型都走 OpenAI 兼容协议；Anthropic 适配器已实现但未绑定到任何模型配置。
- 上游返回 `4xx`（如参数错误）统一映射为 `502 UPSTREAM_ERROR`，调用方需结合 message 判断根因。
- `/stream` 不支持 `response_format`（由 schema 的 `extra="forbid"` 在入参阶段拒绝），流式与结构化暂不能组合。
