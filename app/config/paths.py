import os
import sys
from pathlib import Path


def data_dir() -> Path:
    """用户数据目录：可通过 MLL_DATA_DIR 覆盖，默认使用各系统用户数据目录。"""
    override = os.getenv("MLL_DATA_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "MLL Gateway"
    if sys.platform == "win32":
        return Path(os.getenv("APPDATA") or Path.home()) / "MLL Gateway"
    return Path(os.getenv("XDG_CONFIG_HOME") or Path.home() / ".config") / "mll-gateway"


def database_path() -> Path:
    """SQLite 数据库路径：可通过 MLL_DB_PATH 覆盖，便于测试或迁移。"""
    override = os.getenv("MLL_DB_PATH")
    if override:
        return Path(override).expanduser()
    return data_dir() / "gateway.db"
