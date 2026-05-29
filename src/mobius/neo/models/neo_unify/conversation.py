from dataclasses import dataclass
from typing import Optional


@dataclass
class ConversationTemplate:
    """Simple conversation template for chat interactions."""

    system_message: str = ""
    roles: tuple = ("user", "assistant")
    sep: str = "</s>"
    messages: list = None

    def __post_init__(self):
        if self.messages is None:
            self.messages = []

    def get_prompt(self) -> str:
        """Build the full prompt from messages."""
        ret = self.system_message + "\n" if self.system_message else ""
        for role, content in self.messages:
            if content:
                ret += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        ret += f"<|im_start|>{self.roles[1]}\n"
        return ret

    def append_message(self, role: str, content: Optional[str]):
        """Append a new message to the conversation."""
        self.messages.append((role, content))


# Supported conversation templates
TEMPLATES = {
    "qwen": ConversationTemplate(
        system_message="You are a helpful assistant.",
        roles=("user", "assistant"),
        sep="<|im_end|>",
    ),
    "qwen_chat": ConversationTemplate(
        system_message="You are a helpful assistant.",
        roles=("user", "assistant"),
        sep="<|im_end|>",
    ),
    "default": ConversationTemplate(
        system_message="You are a helpful assistant.",
        roles=("user", "assistant"),
        sep="</s>",
    ),
}


def get_conv_template(template_name: str) -> ConversationTemplate:
    """Get a conversation template by name.

    Args:
        template_name: Name of the template to retrieve. Falls back to "default"
            if the name is not found.

    Returns:
        A ConversationTemplate instance.
    """
    if template_name is None:
        template_name = "default"
    return TEMPLATES.get(template_name, TEMPLATES["default"])


__all__ = [
    "ConversationTemplate",
    "get_conv_template",
    "TEMPLATES",
]