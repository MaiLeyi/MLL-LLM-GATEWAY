"""多用户能力接入前的默认租户标识。

所有 Service / Repository 都通过参数接收 workspace_id，当前统一使用默认值；
接入登录与工作区后，只需由鉴权层把真实 id 传进来。"""

DEFAULT_USER_ID = "local-user"
DEFAULT_WORKSPACE_ID = "default"
