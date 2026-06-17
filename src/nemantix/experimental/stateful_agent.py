import collections
from typing import Any, Optional, Type

from nemantix.core.agent import Agent
from nemantix.core.exceptions import NemantixException
from nemantix.core.expertise import Expertise
from pydantic import BaseModel


class StatefulAgent(Agent):
    """
    An Agent that maintains a rolling window of the last K interactions
    and prepends this raw transcript to the user's current request.
    """

    def __init__(self, expertise: Expertise, max_history_turns: int = 16,
                 strings_max_size=1000, **kwargs):
        super().__init__(expertise=expertise, **kwargs)

        self.max_history_turns = max(1, max_history_turns)
        self.max_strings_size = int(strings_max_size)
        self.chat_history = collections.deque(maxlen=self.max_history_turns)

    def run(
        self, user_request: str, schema: Optional[Type[BaseModel]] = None, **kwargs
    ) -> tuple[Optional[NemantixException], Any]:
        augmented_request = self._build_augmented_request(user_request)

        exception, output = super().run(
            user_request=augmented_request, schema=schema, **kwargs
        )

        if exception:
            agent_response = f"Execution failed with error: {exception}"
        else:
            agent_response = self._format_output_for_history(output)

        self.chat_history.append({"user": user_request, "agent": agent_response})
        return exception, output

    def clear_history(self):
        """Allows the user to manually reset the conversation."""
        self.chat_history.clear()

    def _build_augmented_request(self, current_request: str) -> str:
        """Constructs a context-aware prompt if history exists."""
        if not self.chat_history:
            return current_request

        history_lines = ["[Previous Transcript]"]
        for turn in self.chat_history:
            history_lines.append(f"User: {turn['user']}")
            history_lines.append(f"Agent: {turn['agent']}")

        history_lines.append("\n[Current Request]")
        history_lines.append(f"User: {current_request}")

        return "\n".join(history_lines)

    def _format_output_for_history(self, output: Any) -> str:
        """
        Safely stringifies outputs to prevent massive payload injection
        into the next turn's prompt.
        """
        if output is None:
            return "<Task completed silently>"

        if isinstance(output, BaseModel):
            return output.model_dump_json()

        if isinstance(output, dict):
            return str(output)

        # TODO: handle Nemantix objects

        string_output = str(output).strip()

        # Safeguard against injecting massive text dumps into the LLM context
        if len(string_output) > self.max_strings_size:
            return string_output[:self.max_strings_size] + "... [Truncated for length]"

        return string_output
