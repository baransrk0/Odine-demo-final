"""Safe, structured application errors."""


class ApiError(Exception):
    """An expected API failure that can be rendered without internal details."""

    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
