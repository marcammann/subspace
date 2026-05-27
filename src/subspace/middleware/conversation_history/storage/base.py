from typing import Protocol

from subspace.models.items import InputItem, OutputItem


class Storage(Protocol):
    async def save_response(
        self,
        response_id: str,
        input_items: list[InputItem],
        output_items: list[OutputItem],
    ) -> None: ...

    async def load_history(
        self,
        response_id: str,
    ) -> list[InputItem | OutputItem]: ...
