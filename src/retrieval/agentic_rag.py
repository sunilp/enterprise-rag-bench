"""Agentic RAG: LLM-driven retrieval routing and multi-step reasoning.

The LLM acts as an agent that decides the retrieval strategy:
which sources to query, whether to decompose the question,
whether to verify retrieved information, and when to stop.

Good for: complex multi-hop questions, multi-source environments.
Trade-off: highest latency, highest cost, hardest to evaluate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .naive_rag import RAGResponse


@dataclass
class AgentAction:
    """An action decided by the agent."""

    tool: str
    input: str
    reasoning: str


@dataclass
class AgentStep:
    """A step in the agent's reasoning chain."""

    action: AgentAction
    observation: str


class AgenticRAG:
    """Agent-based RAG with dynamic retrieval routing.

    The agent can:
    - Search different vector stores (by topic/source)
    - Decompose complex questions into sub-queries
    - Verify retrieved information against other sources
    - Synthesize answers from multiple retrieval rounds
    """

    SYSTEM_PROMPT = """You are a retrieval agent. Given a question, decide which
tool to use to find the answer. You have these tools:

{tool_descriptions}

Respond in JSON format:
{{"tool": "<tool_name>", "input": "<search query>", "reasoning": "<why this tool>"}}

If you have enough information to answer, respond:
{{"tool": "final_answer", "input": "<your answer>", "reasoning": "<why you're done>"}}"""

    def __init__(
        self,
        retrieval_tools: Dict[str, Any],
        llm: Any,
        max_steps: int = 5,
        top_k: int = 5,
    ):
        self.retrieval_tools = retrieval_tools
        self.llm = llm
        self.max_steps = max_steps
        self.top_k = top_k

    def _get_tool_descriptions(self) -> str:
        descriptions = []
        for name, tool in self.retrieval_tools.items():
            desc = getattr(tool, "description", f"Search {name}")
            descriptions.append(f"- {name}: {desc}")
        return "\n".join(descriptions)

    def _decide_action(
        self, question: str, history: List[AgentStep]
    ) -> AgentAction:
        """Ask the LLM what to do next."""
        context = self.SYSTEM_PROMPT.format(
            tool_descriptions=self._get_tool_descriptions()
        )

        messages = [context, f"\nQuestion: {question}"]

        for step in history:
            messages.append(
                f"\nAction: {step.action.tool}({step.action.input})"
                f"\nObservation: {step.observation[:500]}"
            )

        messages.append("\nWhat should I do next? Respond in JSON:")

        prompt = "\n".join(messages)
        response = self.llm.generate(prompt)

        try:
            parsed = json.loads(response.strip())
            return AgentAction(
                tool=parsed.get("tool", "final_answer"),
                input=parsed.get("input", ""),
                reasoning=parsed.get("reasoning", ""),
            )
        except json.JSONDecodeError:
            return AgentAction(
                tool="final_answer",
                input=response,
                reasoning="Failed to parse agent response as JSON",
            )

    def _execute_tool(self, action: AgentAction) -> str:
        """Execute a retrieval tool and return observations."""
        tool = self.retrieval_tools.get(action.tool)
        if tool is None:
            return f"Unknown tool: {action.tool}"

        try:
            results = tool.similarity_search(action.input, k=self.top_k)
            texts = [r.get("text", r.get("page_content", "")) for r in results]
            return "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(texts))
        except Exception as e:
            return f"Tool error: {e}"

    def query(self, question: str) -> RAGResponse:
        """Run the agentic RAG pipeline."""
        history: List[AgentStep] = []
        all_sources: List[dict] = []

        for step_num in range(self.max_steps):
            action = self._decide_action(question, history)

            if action.tool == "final_answer":
                return RAGResponse(
                    answer=action.input,
                    source_documents=all_sources,
                    metadata={
                        "pattern": "agentic_rag",
                        "num_steps": step_num + 1,
                        "steps": [
                            {
                                "tool": s.action.tool,
                                "input": s.action.input,
                                "reasoning": s.action.reasoning,
                            }
                            for s in history
                        ],
                    },
                )

            observation = self._execute_tool(action)
            history.append(AgentStep(action=action, observation=observation))
            all_sources.append(
                {"tool": action.tool, "query": action.input, "results": observation}
            )

        # Max steps reached — synthesize from what we have
        all_observations = "\n\n".join(s.observation for s in history)
        prompt = f"Based on the following research, answer the question.\n\nResearch:\n{all_observations}\n\nQuestion: {question}\n\nAnswer:"
        answer = self.llm.generate(prompt)

        return RAGResponse(
            answer=answer,
            source_documents=all_sources,
            metadata={
                "pattern": "agentic_rag",
                "num_steps": self.max_steps,
                "max_steps_reached": True,
            },
        )
