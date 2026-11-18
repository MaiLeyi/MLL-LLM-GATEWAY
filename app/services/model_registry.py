import json
import logging
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from app.config.model_config import ModelConfig
from app.config.settings import DEFAULT_MODEL, MODEL_CONFIGS as SEED_MODEL_CONFIGS
from app.core.exceptions import (
    InternalException,
    InvalidRequestException,
    MissingApiKeyException,
    UnsupportedModelException,
)

logger = logging.getLogger(__name__)

# 模型 id 只允许字母、数字、下划线和中划线，便于作为钥匙串账户名与配置文件键
MODEL_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,49}$")
CONFIG_FILE_NAME = "models.json"
# macOS 钥匙串条目标识，账户名使用模型 id
KEYCHAIN_SERVICE = os.getenv("MLL_KEYCHAIN_SERVICE", "mll-llm-gateway")
SUPPORTED_PROTOCOLS = {"openai", "anthropic"}
SUPPORTED_STRUCTURED_MODES = {"json_schema", "json_object"}


@dataclass(frozen=True, slots=True)
class ModelDraft:
    """管理页面提交的模型配置草稿；api_key 为 None 表示不修改已有 Key。"""

    provider_model: str
    base_url: str
    protocol: str = "openai"
    structured_output_mode: str | None = None
    rate_limit_per_minute: int = 60
    fallback_model: str | None = None
    input_price_per_million: float = 0.0
    output_price_per_million: float = 0.0
    display_name: str = ""
    api_key: str | None = None


def _data_dir() -> Path:
    """模型配置文件目录：可通过 MLL_DATA_DIR 覆盖，默认使用各系统用户数据目录。"""
    override = os.getenv("MLL_DATA_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "MLL Gateway"
    if sys.platform == "win32":
        return Path(os.getenv("APPDATA") or Path.home()) / "MLL Gateway"
    return Path(os.getenv("XDG_CONFIG_HOME") or Path.home() / ".config") / "mll-gateway"


def _default_env_name(model_id: str) -> str:
    """新增模型的默认环境变量名，用于兼容手工在 .env 配置 Key 的场景。"""
    return re.sub(r"[^A-Za-z0-9]", "_", model_id).upper() + "_API_KEY"


def _mask_key(api_key: str | None) -> str | None:
    """只暴露 Key 的首尾片段，避免管理接口回显完整密钥。"""
    if not api_key:
        return None
    if len(api_key) <= 8:
        return "****"
    return f"{api_key[:3]}****{api_key[-4:]}"


def _keychain_get(account: str) -> str | None:
    """从 macOS 钥匙串读取 Key；不存在、非 mac 或调用失败都返回 None。"""
    if sys.platform != "darwin":
        return None
    try:
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-s",
                KEYCHAIN_SERVICE,
                "-a",
                account,
                "-w",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _keychain_set(account: str, api_key: str) -> None:
    """写入或更新钥匙串条目；失败时抛出业务异常，避免管理端误以为保存成功。"""
    if sys.platform != "darwin":
        raise InvalidRequestException(
            "当前系统不支持 macOS 钥匙串，请改用 .env 环境变量配置 API Key"
        )
    try:
        result = subprocess.run(
            [
                "security",
                "add-generic-password",
                "-U",
                "-s",
                KEYCHAIN_SERVICE,
                "-a",
                account,
                "-w",
                api_key,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise InternalException(f"写入系统钥匙串失败: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise InternalException(f"写入系统钥匙串失败: {detail}")


def _keychain_delete(account: str) -> None:
    """删除钥匙串条目；不存在或非 mac 时静默忽略。"""
    if sys.platform != "darwin":
        return
    try:
        subprocess.run(
            [
                "security",
                "delete-generic-password",
                "-s",
                KEYCHAIN_SERVICE,
                "-a",
                account,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        logger.warning("删除钥匙串条目失败: %s", account)


def _record(model_id: str, config: ModelConfig) -> dict:
    """把内存配置转成可持久化的非敏感记录，API Key 不写入 JSON。"""
    return {
        "id": model_id,
        "display_name": config.display_name or model_id,
        "provider_model": config.provider_model,
        "base_url": config.base_url,
        "api_key_env": config.api_key_env,
        "protocol": config.protocol,
        "structured_output_mode": config.structured_output_mode,
        "rate_limit_per_minute": config.rate_limit_per_minute,
        "fallback_model": config.fallback_model,
        "input_price_per_million": config.input_price_per_million,
        "output_price_per_million": config.output_price_per_million,
    }


def _config_from_record(model_id: str, record: dict) -> ModelConfig:
    """从磁盘记录恢复配置，对缺失字段使用安全默认值。"""
    protocol = record.get("protocol")
    structured_mode = record.get("structured_output_mode")
    fallback = record.get("fallback_model") or None
    return ModelConfig(
        provider_model=str(record["provider_model"]),
        base_url=str(record["base_url"]),
        api_key_env=str(record.get("api_key_env") or _default_env_name(model_id)),
        protocol=protocol if protocol in SUPPORTED_PROTOCOLS else "openai",
        structured_output_mode=(
            structured_mode if structured_mode in SUPPORTED_STRUCTURED_MODES else None
        ),
        rate_limit_per_minute=int(record.get("rate_limit_per_minute", 60) or 0),
        fallback_model=fallback,
        input_price_per_million=float(record.get("input_price_per_million", 0) or 0),
        output_price_per_million=float(record.get("output_price_per_million", 0) or 0),
        display_name=str(record.get("display_name") or model_id),
    )


class ModelRegistry:
    """运行时模型注册表：JSON 持久化非敏感配置，Key 走系统钥匙串并保留 .env 兜底。"""

    def __init__(self, config_path: Path | None = None) -> None:
        self._path = config_path or (_data_dir() / CONFIG_FILE_NAME)
        self._lock = threading.RLock()
        self._models: dict[str, ModelConfig] = {}
        self._default_model = DEFAULT_MODEL
        self._revision = 0
        # 钥匙串读取有子进程开销，缓存结果；写入 Key 时同步刷新
        self._secret_cache: dict[str, str | None] = {}
        self._load()

    @property
    def revision(self) -> int:
        """配置版本号，用于让适配器缓存随模型变更失效。"""
        with self._lock:
            return self._revision

    @property
    def config_path(self) -> Path:
        return self._path

    @property
    def default_model(self) -> str:
        with self._lock:
            if self._default_model in self._models:
                return self._default_model
            return next(iter(self._models), "")

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(self._models)

    def get(self, model_id: str) -> ModelConfig | None:
        with self._lock:
            return self._models.get(model_id)

    def require(self, model_id: str) -> ModelConfig:
        """取模型配置，不存在时按统一错误码返回 400。"""
        with self._lock:
            config = self._models.get(model_id)
            if config is None:
                raise UnsupportedModelException(
                    f"不支持的模型: {model_id}，可选: {sorted(self._models)}"
                )
            return config

    def fallback_chain(self, model_id: str) -> list[str]:
        """沿 fallback_model 链去重，得到依次尝试的候选列表。"""
        with self._lock:
            chain: list[str] = []
            current: str | None = model_id
            while current is not None and current in self._models and current not in chain:
                chain.append(current)
                current = self._models[current].fallback_model
            return chain

    def get_api_key(self, model_id: str) -> str:
        """Key 优先取系统钥匙串，其次取 .env 环境变量。"""
        with self._lock:
            api_key, _source = self._resolve_key_locked(model_id)
            if not api_key:
                config = self.require(model_id)
                raise MissingApiKeyException(
                    f"模型 {model_id} 未配置 API Key，请到 Models 页面填写，"
                    f"或在 .env 设置 {config.api_key_env}"
                )
            return api_key

    def describe(self, model_id: str) -> dict:
        """生成管理页面卡片数据，永不返回完整 Key。"""
        with self._lock:
            config = self.require(model_id)
            api_key, source = self._resolve_key_locked(model_id)
            return {
                "id": model_id,
                "display_name": config.display_name or model_id,
                "provider_model": config.provider_model,
                "base_url": config.base_url,
                "protocol": config.protocol,
                "structured_output_mode": config.structured_output_mode,
                "rate_limit_per_minute": config.rate_limit_per_minute,
                "fallback_model": config.fallback_model,
                "input_price_per_million": config.input_price_per_million,
                "output_price_per_million": config.output_price_per_million,
                "api_key_env": config.api_key_env,
                "has_api_key": bool(api_key),
                "api_key_masked": _mask_key(api_key),
                "api_key_source": source,
                "is_default": model_id == self.default_model,
            }

    def create(self, model_id: str, draft: ModelDraft) -> dict:
        """新增模型；id 在保存后不可修改。"""
        self._validate_model_id(model_id)
        with self._lock:
            if model_id in self._models:
                raise InvalidRequestException(f"模型已存在: {model_id}")
            config = self._config_from_draft(model_id, draft)
            candidate = {**self._models, model_id: config}
            self._validate_chain_locked(model_id, candidate)
            # 先写 Key，失败时不落模型，避免出现“配置成功但密钥失败”的中间态
            api_key = (draft.api_key or "").strip()
            if api_key:
                _keychain_set(model_id, api_key)
                self._secret_cache[model_id] = api_key
            self._models[model_id] = config
            self._bump_locked()
            self._persist_locked()
        return self.describe(model_id)

    def update(self, model_id: str, draft: ModelDraft) -> dict:
        """更新模型配置；api_key 为空表示保留原 Key。"""
        with self._lock:
            if model_id not in self._models:
                raise InvalidRequestException(f"模型不存在: {model_id}")
            config = self._config_from_draft(model_id, draft)
            candidate = {**self._models, model_id: config}
            self._validate_chain_locked(model_id, candidate)
            api_key = (draft.api_key or "").strip()
            if api_key:
                _keychain_set(model_id, api_key)
                self._secret_cache[model_id] = api_key
            self._models[model_id] = config
            self._bump_locked()
            self._persist_locked()
        return self.describe(model_id)

    def delete(self, model_id: str) -> None:
        """删除模型；默认模型或被其他模型引用为备用模型时禁止删除。"""
        with self._lock:
            if model_id not in self._models:
                raise InvalidRequestException(f"模型不存在: {model_id}")
            if len(self._models) == 1:
                raise InvalidRequestException("至少需要保留一个模型")
            if model_id == self.default_model:
                raise InvalidRequestException("默认模型不能删除，请先切换默认模型")
            referenced_by = [
                other_id
                for other_id, other in self._models.items()
                if other.fallback_model == model_id
            ]
            if referenced_by:
                raise InvalidRequestException(
                    f"模型 {model_id} 正被 {', '.join(referenced_by)} 作为备用模型引用，请先解除引用"
                )
            del self._models[model_id]
            self._secret_cache.pop(model_id, None)
            _keychain_delete(model_id)
            self._bump_locked()
            self._persist_locked()

    def set_default(self, model_id: str) -> dict:
        with self._lock:
            if model_id not in self._models:
                raise InvalidRequestException(f"模型不存在: {model_id}")
            self._default_model = model_id
            self._bump_locked()
            self._persist_locked()
        return self.describe(model_id)

    def _config_from_draft(self, model_id: str, draft: ModelDraft) -> ModelConfig:
        """校验草稿并转换为不可变配置。"""
        base_url = draft.base_url.strip().rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise InvalidRequestException("base_url 必须以 http:// 或 https:// 开头")
        if draft.protocol not in SUPPORTED_PROTOCOLS:
            raise InvalidRequestException(
                f"不支持的协议: {draft.protocol}，可选: {sorted(SUPPORTED_PROTOCOLS)}"
            )
        if (
            draft.structured_output_mode is not None
            and draft.structured_output_mode not in SUPPORTED_STRUCTURED_MODES
        ):
            raise InvalidRequestException(
                f"不支持的 structured_output_mode: {draft.structured_output_mode}"
            )
        fallback = (draft.fallback_model or "").strip() or None
        existing_env = (
            self._models[model_id].api_key_env
            if model_id in self._models
            else _default_env_name(model_id)
        )
        return ModelConfig(
            provider_model=draft.provider_model.strip(),
            base_url=base_url,
            api_key_env=existing_env,
            protocol=draft.protocol,
            structured_output_mode=draft.structured_output_mode,
            rate_limit_per_minute=draft.rate_limit_per_minute,
            fallback_model=fallback,
            input_price_per_million=draft.input_price_per_million,
            output_price_per_million=draft.output_price_per_million,
            display_name=(draft.display_name or "").strip() or model_id,
        )

    @staticmethod
    def _validate_model_id(model_id: str) -> None:
        if not MODEL_ID_PATTERN.match(model_id):
            raise InvalidRequestException(
                "模型 id 只能包含字母、数字、下划线和中划线，且以字母或数字开头（最长 50）"
            )

    def _validate_chain_locked(
        self, model_id: str, models: dict[str, ModelConfig]
    ) -> None:
        """校验备用模型存在且不会形成循环引用。"""
        fallback = models[model_id].fallback_model
        if fallback is None:
            return
        if fallback == model_id:
            raise InvalidRequestException("备用模型不能指向自身")
        if fallback not in models:
            raise InvalidRequestException(f"备用模型不存在: {fallback}")
        seen = {model_id}
        cursor: str | None = fallback
        while cursor is not None:
            if cursor in seen:
                raise InvalidRequestException("备用模型链存在循环引用")
            seen.add(cursor)
            config = models.get(cursor)
            cursor = config.fallback_model if config else None

    def _resolve_key_locked(self, model_id: str) -> tuple[str | None, str | None]:
        """返回 (Key, 来源)；来源为 keychain / env / None。"""
        config = self.require(model_id)
        if model_id not in self._secret_cache:
            self._secret_cache[model_id] = _keychain_get(model_id)
        keychain_key = self._secret_cache[model_id]
        if keychain_key:
            return keychain_key, "keychain"
        env_key = os.getenv(config.api_key_env)
        if env_key:
            return env_key, "env"
        return None, None

    def _bump_locked(self) -> None:
        self._revision += 1

    def _persist_locked(self) -> None:
        """原子写入配置文件，避免中途崩溃留下半截 JSON。"""
        payload = {
            "version": 1,
            "default_model": self.default_model,
            "models": [
                _record(model_id, config) for model_id, config in self._models.items()
            ],
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, self._path)
        except OSError as exc:
            raise InternalException(f"保存模型配置失败: {exc}") from exc

    def _load(self) -> None:
        """优先读取用户配置；首次启动或文件损坏时回退到内置种子配置。"""
        if self._path.exists():
            try:
                payload = json.loads(self._path.read_text(encoding="utf-8"))
                records = payload.get("models") or []
                models = {
                    str(record["id"]): _config_from_record(str(record["id"]), record)
                    for record in records
                }
                if not models:
                    raise ValueError("模型列表为空")
                default_model = payload.get("default_model")
                self._models = models
                self._default_model = (
                    default_model if default_model in models else next(iter(models))
                )
                for model_id in models:
                    self._validate_chain_locked(model_id, models)
                return
            except Exception as exc:  # noqa: BLE001 - 配置文件损坏时不能阻塞启动
                logger.warning("读取模型配置失败，使用内置种子配置: %s", exc)

        self._models = dict(SEED_MODEL_CONFIGS)
        self._default_model = (
            DEFAULT_MODEL if DEFAULT_MODEL in self._models else next(iter(self._models))
        )
        try:
            self._persist_locked()
        except InternalException as exc:
            logger.warning("写入初始模型配置失败: %s", exc.message)


model_registry = ModelRegistry()
