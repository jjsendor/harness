from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from coding_harness.config import Settings


class Tracing:
    """One client per CLI process and a fresh callback handler per invocation."""

    def __init__(self, settings: Settings):
        self.client = (
            Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key.get_secret_value(),
                base_url=settings.langfuse_base_url,
            )
            if settings.tracing
            else None
        )
        self.public_key = settings.langfuse_public_key

    def handler(self) -> CallbackHandler | None:
        return CallbackHandler(public_key=self.public_key) if self.client else None

    def close(self) -> None:
        if self.client:
            self.client.shutdown()
