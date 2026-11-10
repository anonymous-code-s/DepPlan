"""AutoGen agents for explicit proposal, verification and selection turns."""
import asyncio
from collections.abc import Sequence
from importlib.metadata import version

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_core import CancellationToken
from autogen_core.models import RequestUsage

from .llm import Completion, LLMClient


def orchestration_info():
    return {
        'framework': 'autogen_agentchat',
        'agentchat_version': version('autogen-agentchat'),
        'core_version': version('autogen-core'),
        'workflow': 'fixed_proposer_verifier_judge',
    }


class PlanningAgent(BaseChatAgent):
    """One role turn over a complete, explicitly constructed planning context."""

    def __init__(self, name: str, client: LLMClient, *, system: str,
                 temperature: float, max_tokens: int):
        super().__init__(name, description=f'DepPlan {name.replace("_", " ")}')
        self.client = client
        self.system = system
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.last_completion: Completion | None = None

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages(self, messages: Sequence[BaseChatMessage],
                          cancellation_token: CancellationToken) -> Response:
        self.last_completion = None
        if len(messages) != 1 or not isinstance(messages[0], TextMessage):
            raise ValueError('A planning turn requires one explicit text context')
        message = messages[0]
        role = message.metadata.get('request_role')
        if not role or not (role == self.name or role.startswith(self.name + '_')):
            raise ValueError('Request role must identify the receiving planning agent')
        if cancellation_token.is_cancelled():
            raise asyncio.CancelledError()
        request = asyncio.create_task(self.client.complete(
            role=role, system=self.system, user=message.content,
            temperature=self.temperature, max_tokens=self.max_tokens))
        cancellation_token.link_future(request)
        completion = await request
        self.last_completion = completion
        usage = completion.usage
        reported_usage = None
        if usage.prompt_tokens is not None and usage.completion_tokens is not None:
            reported_usage = RequestUsage(prompt_tokens=usage.prompt_tokens,
                                          completion_tokens=usage.completion_tokens)
        return Response(chat_message=TextMessage(
            content=completion.text, source=self.name, models_usage=reported_usage,
            metadata={'request_role': role}))

    async def complete(self, *, role: str, user: str,
                       source: str = 'coordinator') -> Completion:
        result = await self.run(task=TextMessage(
            content=user, source=source, metadata={'request_role': role}))
        if self.last_completion is None:
            raise RuntimeError('The planning agent returned no model completion')
        return Completion(result.messages[-1].content, self.last_completion.usage,
                          self.last_completion.attempts)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        self.last_completion = None
