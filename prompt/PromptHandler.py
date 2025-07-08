"""Handle prompt templates and variable substitution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class PromptHandler:
    template: str

    def format(self, variables: Dict[str, str]) -> str:
        text = self.template
        for k, v in variables.items():
            text = text.replace(f"{{{{{k}}}}}", v)
        return text
