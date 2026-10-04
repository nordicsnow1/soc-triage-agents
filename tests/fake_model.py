"""A scripted stand-in for a Strands model provider, for offline tests.

Each call to stream() plays the next step of the script, using the same
ConverseStream event format as Bedrock. It does not think; it only lets
tests check the code around the model (tool execution, recording, parsing).
"""
import json
from itertools import count

from strands.models.model import Model


def tool_step(name: str, tool_input: dict) -> dict:
    return {"tool": name, "input": tool_input}


def text_step(text: str) -> dict:
    return {"text": text}


class ScriptedModel(Model):
    def __init__(self, script: list[dict]):
        self.script = list(script)
        self.requests: list[dict] = []  # what the agent sent on each turn
        self._ids = count(1)

    def get_config(self):
        return {"model_id": "scripted-fake"}

    def update_config(self, **model_config):
        pass

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        raise NotImplementedError("the agent uses the structured-output tool, not this method")
        yield  # pragma: no cover  (makes this an async generator)

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.requests.append({"system_prompt": system_prompt,
                              "tool_names": [t["name"] for t in tool_specs or []]})
        step = self.script.pop(0) if self.script else text_step("I have nothing more to add.")

        yield {"messageStart": {"role": "assistant"}}
        if "tool" in step:
            yield {"contentBlockStart": {"start": {"toolUse": {
                "toolUseId": f"tool-{next(self._ids)}", "name": step["tool"]}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(step["input"])}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockDelta": {"delta": {"text": step["text"]}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        yield {"metadata": {"usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                            "metrics": {"latencyMs": 0}}}
