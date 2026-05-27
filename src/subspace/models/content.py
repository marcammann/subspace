from typing import Annotated, Literal

from pydantic import BaseModel, Field


class InputTextContent(BaseModel):
    type: Literal["input_text"] = "input_text"
    text: str


class InputImageContent(BaseModel):
    type: Literal["input_image"] = "input_image"
    image_url: str
    detail: str | None = None


class OutputTextContent(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str


class RefusalContent(BaseModel):
    type: Literal["refusal"] = "refusal"
    refusal: str


InputContent = Annotated[
    InputTextContent | InputImageContent,
    Field(discriminator="type"),
]

OutputContent = Annotated[
    OutputTextContent | RefusalContent,
    Field(discriminator="type"),
]
