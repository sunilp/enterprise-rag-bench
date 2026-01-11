"""Prompt injection detection for RAG pipelines.

Detects attempts to manipulate the LLM through adversarial content
in user queries or retrieved documents. Uses a layered approach:
1. Heuristic pattern matching (fast, catches obvious attacks)
2. Structural analysis (detects role/instruction injection)
3. Optional LLM-based classification (highest accuracy)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional


class ThreatLevel(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class InjectionAnalysis:
    """Result of injection detection analysis."""
    is_injection: bool
    threat_level: ThreatLevel
    detected_patterns: List[str]
    confidence: float
    recommendation: str


# Known injection patterns (regex)
_INJECTION_PATTERNS = [
    (r"ignore\s+(all\s+)?previous\s+instructions", "instruction_override", ThreatLevel.HIGH),
    (r"ignore\s+(all\s+)?above", "instruction_override", ThreatLevel.HIGH),
    (r"disregard\s+(all\s+)?previous", "instruction_override", ThreatLevel.HIGH),
    (r"you\s+are\s+now\s+a", "role_hijack", ThreatLevel.HIGH),
    (r"act\s+as\s+(?:a|an)\s+", "role_hijack", ThreatLevel.MEDIUM),
    (r"pretend\s+(?:you\s+are|to\s+be)", "role_hijack", ThreatLevel.MEDIUM),
    (r"system\s*:\s*", "system_prompt_injection", ThreatLevel.CRITICAL),
    (r"\[INST\]|\[/INST\]|<\|im_start\|>|<\|im_end\|>", "template_injection", ThreatLevel.CRITICAL),
    (r"(?:reveal|show|display|print)\s+(?:your|the)\s+(?:system|original)\s+prompt", "prompt_extraction", ThreatLevel.HIGH),
    (r"what\s+(?:are|is)\s+your\s+(?:instructions|system\s+prompt|rules)", "prompt_extraction", ThreatLevel.MEDIUM),
    (r"do\s+not\s+(?:mention|say|tell|reveal)\s+that", "output_manipulation", ThreatLevel.MEDIUM),
    (r"translate\s+(?:the\s+)?(?:above|previous|following)\s+to", "encoding_attack", ThreatLevel.LOW),
]


class InjectionDefense:
    """Multi-layered prompt injection detection.

    Usage:
        defense = InjectionDefense()
        result = defense.analyze(user_query)
        if result.is_injection:
            # Block or sanitize the input
            ...
    """

    def __init__(self, sensitivity: float = 0.5, llm: Optional[Any] = None):
        self.sensitivity = sensitivity
        self.llm = llm
        self._compiled = [
            (re.compile(pattern, re.IGNORECASE), name, level)
            for pattern, name, level in _INJECTION_PATTERNS
        ]

    def analyze(self, text: str) -> InjectionAnalysis:
        """Analyze text for prompt injection attempts."""
        detected = []
        max_threat = ThreatLevel.NONE

        # Layer 1: Pattern matching
        for pattern, name, level in self._compiled:
            if pattern.search(text):
                detected.append(name)
                if level.value > max_threat.value:
                    max_threat = level

        # Layer 2: Structural analysis
        structural = self._structural_analysis(text)
        detected.extend(structural)
        if structural:
            if max_threat.value < ThreatLevel.MEDIUM.value:
                max_threat = ThreatLevel.MEDIUM

        confidence = min(1.0, len(detected) * 0.3 + (0.2 if max_threat.value >= ThreatLevel.HIGH.value else 0))

        is_injection = confidence >= self.sensitivity

        recommendation = "allow"
        if is_injection:
            if max_threat in (ThreatLevel.CRITICAL, ThreatLevel.HIGH):
                recommendation = "block"
            else:
                recommendation = "sanitize"

        return InjectionAnalysis(
            is_injection=is_injection,
            threat_level=max_threat,
            detected_patterns=detected,
            confidence=confidence,
            recommendation=recommendation,
        )

    def _structural_analysis(self, text: str) -> List[str]:
        """Detect structural injection patterns."""
        findings = []

        # Multiple instruction-like segments
        instruction_markers = len(re.findall(
            r"(?:you must|you should|always|never|do not|make sure)", text, re.IGNORECASE
        ))
        if instruction_markers >= 3:
            findings.append("excessive_instructions")

        # Unusually long input (potential jailbreak payload)
        if len(text) > 2000:
            findings.append("long_input")

        # Mixed languages (potential encoding attack)
        ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
        if ascii_ratio < 0.7:
            findings.append("mixed_encoding")

        return findings

    def sanitize(self, text: str) -> str:
        """Remove detected injection patterns from text."""
        sanitized = text
        for pattern, _, _ in self._compiled:
            sanitized = pattern.sub("[FILTERED]", sanitized)
        return sanitized
