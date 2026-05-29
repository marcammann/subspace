from subspace.models.items import AnyItem, OutputItem


class InMemoryStorage:
    def __init__(self) -> None:
        self._responses: dict[str, tuple[list[AnyItem], list[OutputItem]]] = {}

    async def save_response(
        self,
        response_id: str,
        input_items: list[AnyItem],
        output_items: list[OutputItem],
    ) -> None:
        self._responses[response_id] = (
            [item.model_copy(deep=True) for item in input_items],
            [item.model_copy(deep=True) for item in output_items],
        )

    async def load_history(
        self,
        response_id: str,
    ) -> list[AnyItem | OutputItem]:
        if response_id not in self._responses:
            return []
        input_items, output_items = self._responses[response_id]
        result: list[AnyItem | OutputItem] = []
        result.extend(item.model_copy(deep=True) for item in input_items)
        result.extend(item.model_copy(deep=True) for item in output_items)
        return result
