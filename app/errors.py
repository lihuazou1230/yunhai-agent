"""统一错误类型：接口层一律转成 `{"code": ..., "message": ...}` 的中文可读信息。

理由和前端的 `AiError` 一样——**错误信息是给用户看的**，
"500 Internal Server Error" 对排查没有帮助，对用户更是噪音。
"""

from __future__ import annotations


class AgentError(Exception):
    """业务错误基类：带 HTTP 状态码与机器可读 code。"""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, *, code: str | None = None, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        if status_code:
            self.status_code = status_code


class UnsupportedFileType(AgentError):
    status_code = 400
    code = "unsupported_file_type"


class FileTooLarge(AgentError):
    status_code = 413
    code = "file_too_large"


class EmptyDocument(AgentError):
    status_code = 400
    code = "empty_document"


class DocumentNotFound(AgentError):
    status_code = 404
    code = "document_not_found"


class JobNotFound(AgentError):
    status_code = 404
    code = "job_not_found"


class NotConfigured(AgentError):
    """后端未配 Key —— 前端据此给出「去配置」的明确引导。"""

    status_code = 503
    code = "llm_not_configured"


class LLMError(AgentError):
    status_code = 502
    code = "llm_error"


class SessionNotFound(AgentError):
    status_code = 404
    code = "session_not_found"


class RunNotFound(AgentError):
    """Agent 的待续跑状态不存在/已过期（client 工具回环用）。"""

    status_code = 404
    code = "run_not_found"
