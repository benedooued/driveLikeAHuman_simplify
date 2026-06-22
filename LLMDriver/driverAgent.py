"""
DriverAgent: perceives driving environment via tools and makes decisions.
Uses Groq (free) instead of OpenAI.
OutputParser is inlined — no separate class needed.
"""

from pydoc import text
import re
import json
from rich import print
from langchain.agents import initialize_agent, AgentType
from langchain.agents.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish

from LLMDriver.prompts import (
    SYSTEM_MESSAGE_PREFIX, SYSTEM_MESSAGE_SUFFIX,
    FORMAT_INSTRUCTIONS, HUMAN_MESSAGE,
    TRAFFIC_RULES, DECISION_CAUTIONS,
)

ACTIONS_MAP = {
    'lane_left': 0, 'idle': 1, 'keep_speed': 1,
    'lane_right': 2, 'faster': 3, 'accelerate': 3,
    'slower': 4, 'decelerate': 4,
}


class _MemoryHandler(BaseCallbackHandler):
    """Lightweight callback: collects thoughts and final answer."""

    def __init__(self):
        super().__init__()
        self.steps: list[str] = []

    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        if "Thought:" in action.log and action.log.count("Thought") > 1:
            action.log = action.log.split("Thought:")[-1]
        self.steps.append(action.log.strip())

    def on_tool_end(self, output: str, **kwargs) -> None:
        if self.steps:
            self.steps[-1] += f'\nObservation: {output}\n'

    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        self.steps.append(finish.log)


class DriverAgent:
    def __init__(self, llm, tool_models: list, sce, verbose: bool = False) -> None:
        self.sce = sce
        self.llm = llm
        self._handler = _MemoryHandler()

        tools = []
        for model in tool_models:
            fn = getattr(model, 'inference')
            tools.append(Tool(name=fn.name, description=fn.description, func=fn))

        # Token buffer keeps context lean — critical for free-tier rate limits
        memory = ConversationBufferMemory(memory_key="chat_history")

        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=verbose,
            memory=None,  # memory is not used in this implementation
            agent_kwargs={
                'system_message_prefix': SYSTEM_MESSAGE_PREFIX,
                'system_message_suffix': SYSTEM_MESSAGE_SUFFIX,
                'human_message': HUMAN_MESSAGE,
                'format_instructions': FORMAT_INSTRUCTIONS,
            },
            handle_parsing_errors="Check your output and make sure it matches the format instructions!",
            max_iterations=4,
            early_stopping_method="generate",
        )

        self._last_output: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, last_decision: dict | None = None) -> dict:
        print(f'[bold]Frame {self.sce.frame}[/bold] — running driver agent...')
        self._handler.steps.clear()

        last_action = last_decision.get("action_name", "N/A") if last_decision else "N/A"
        last_expl   = last_decision.get("explanation", "N/A")  if last_decision else "N/A"

        prompt = f"""
You are the 'ego' car driving on a highway (frame: {self.sce.frame}).
Last decision: `{last_action}`. Reason: `{last_expl}`.

Current scenario:
```json
{self.sce.export2json()[:1200]}
```

Traffic rules:
{TRAFFIC_RULES}

Attention:
{DECISION_CAUTIONS}

Think step by step. Once decided, output EXACTLY:
Final Answer: {{"decision": "<action>", "explanation": "<≤20 words>"}}
STOP immediately after Final Answer.
"""
        # Récupère la réponse brute avec usage
        response = self.agent.run(prompt, callbacks=[self._handler])
        print("\n===== RESPONSE =====")
        print(response)

        print("\n===== STEPS =====")
        print(self._handler.steps)
        print("===================\n")

        thoughts   = '\n'.join(self._handler.steps[:-1])
        final_log  = self._handler.steps[-1] if self._handler.steps else ''
        print("\n===== FINAL LOG =====")
        print(final_log)
        print("=====================\n")
        parsed     = self._parse_final(final_log)
        # ==============================
# Reasoning quality metrics
# ==============================

# Hallucination detection
        hallucination = int(
            ("Invalid vehicle id" in thoughts)
                or
            ("Invalid lane id" in thoughts)
        )


# Thought-Action consistency
        thoughts_lower = thoughts.lower()

        unsafe_flag = (
            "unsafe" in thoughts_lower
            or
            "conflict" in thoughts_lower
            or
            "danger" in thoughts_lower
        )

        risky_action = parsed["action_name"] in {
            "FASTER",
            "LANE_LEFT",
            "LANE_RIGHT"
        }

        consistency_failure = int(
            unsafe_flag and risky_action
        )


        parsed["hallucination"] = hallucination
        parsed["consistency_failure"] = consistency_failure

        # ── Tokens via callback metadata ──────────────────────────────────────
        # langchain-groq expose les token counts dans llm_output après l'appel
        try:
            usage = self.llm.client.usage   # attribut exposé par Groq après appel
            parsed['_tokens_in']  = getattr(usage, 'prompt_tokens', 0)
            parsed['_tokens_out'] = getattr(usage, 'completion_tokens', 0)
        except Exception:
            parsed['_tokens_in']  = 0
            parsed['_tokens_out'] = 0

        self.sce.commit_decision(
            thoughts,
            final_log,
            json.dumps(parsed),
            parsed["format_failure"],
            parsed["hallucination"],
            parsed["consistency_failure"]
        )
        print(f'[cyan]Decision:[/cyan] {parsed}')
        return parsed

    # ------------------------------------------------------------------
    # Inline output parsing — replaces the separate OutputParser class
    # ------------------------------------------------------------------

    def _parse_final(self, text: str) -> dict:

        decision = "idle"
        explanation = "no explanation"
        format_failure = 0

        json_match = re.search(
            r'\{.*\}',
            text,
            re.DOTALL
        )

        if json_match:

            try:
                data = json.loads(json_match.group())

                decision = data.get(
                    "decision",
                    "idle"
             ).lower()

                explanation = data.get(
                "explanation",
                "no explanation"
                )

            except Exception as e:
                print(f"[red]JSON parse error:[/red] {e}")
                format_failure = 1

        else:

            format_failure = 1

            decision_match = re.search(
                r'decision\s*:\s*([A-Z_]+)',
                text,
                re.IGNORECASE
            )

            explanation_match = re.search(
                r'explanation\s*:\s*(.+)',
                text,
                re.IGNORECASE
            )

            if decision_match:
                decision = decision_match.group(1).lower()

            if explanation_match:
                explanation = explanation_match.group(1).strip()

        action_map = {
        "lane_left": 0,
        "idle": 1,
        "keep_speed": 1,
        "lane_right": 2,
        "faster": 3,
        "accelerate": 3,
        "slower": 4,
        "decelerate": 4,
    }

        action_id = action_map.get(decision, 1)

        return {
        "action_id": action_id,
        "action_name": [
            "LANE_LEFT",
            "IDLE",
            "LANE_RIGHT",
            "FASTER",
            "SLOWER"
        ][action_id],
        "explanation": explanation,
        "format_failure": format_failure,
    }