import base64
from pathlib import Path

from langchain_copilot import CopilotChatModel
from langchain_core.messages import HumanMessage

model = CopilotChatModel(model_name="gpt-5-mini")

image_path = Path(__file__).with_name("10_image.png")
image_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")

messages = [
    HumanMessage(
        content=[
            {"type": "text", "text": "Describe this image briefly."},
            {
                "type": "image",
                "base64": image_base64,
                "mime_type": "image/png",
            },
        ]
    )
]

response = model.invoke(messages)
print(response.content)