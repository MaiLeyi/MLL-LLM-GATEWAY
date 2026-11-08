from typing import Final

# 版本引用别名：调用方传 latest 时解析为该模板最新发布的版本
LATEST_VERSION: Final[str] = "latest"

# 内置模板：进程启动时载入内存仓库，保证新环境开箱即用
DEFAULT_PROMPT_TEMPLATES: Final[dict[tuple[str, str], str]] = {
    ("knowledge_decision", "v1"): (
        "你是${product_name}的知识库决策器。"
        "资料不足时搜索，资料充分时结束回答。不得编造制度内容。"
    )
}
