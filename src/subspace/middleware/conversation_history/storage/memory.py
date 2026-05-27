from subspace.models.items import InputItem, OutputItem


class InMemoryStorage:
    def __init__(self) -> None:
        self._responses: dict[str, tuple[list[InputItem], list[OutputItem]]] = {}

    async def save_response(
        self,
        response_id: str,
        input_items: list[InputItem],
        output_items: list[OutputItem],
    ) -> None:
        self._responses[response_id] = (input_items, output_items)

    async def load_history(
        self,
        response_id: str,
    ) -> list[InputItem | OutputItem]:
        if response_id not in self._responses:
            return []
        input_items, output_items = self._responses[response_id]
        result: list[InputItem | OutputItem] = []
        result.extend(input_items)
        result.extend(output_items)
        return result
