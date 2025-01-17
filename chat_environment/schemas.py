from pydantic import BaseModel, Field
from typing import Union, Dict, Any, List, Optional, Literal

class InputSchema(BaseModel):
    func_name: str
    func_input_data: Optional[Union[Dict[str, Any], List[Dict[str, Any]], str]] = None

class ChatInputMessage(BaseModel):
    content: str
    role: Literal["user", "assistant"]
    user_id: str
    timestamp: str

class ChatAutoMessage(BaseModel):
    relevance: int = Field(
        description="A score from 0-100 indicating how relevant the messages are based on context and memory. Low scores suggest hold decision."
    )
    decision: Literal["hold", "post"] = Field(
        description="Whether to post or hold the message. Use 'hold' when relevance is low or message lacks appropriate humor/context for TARS bot."
    )
    message: str = Field(
        description="The actual message content to post to Discord if decision is post"
    )

class ChatMessage(BaseModel):
    type: Literal["markdown", "text"] = Field(
        description="Format to return response in - either as markdown or plain text"
    )
    content: str = Field(
        description="The message content for TARS bot to send to Discord. Include markdown block and formatting if type is markdown."
    )

class ChatObservation(BaseModel):
    messages: List[ChatMessage]
