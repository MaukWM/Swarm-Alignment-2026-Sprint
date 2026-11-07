"""The agents' system and task prompts, filled from the templates in templates/."""

from pathlib import Path
from string import Template

TEMPLATES = Path(__file__).parent / "templates"
NUDGE = "No tool call found. Please call a tool to continue."


def render(name: str, **values: str | int) -> str:
    """templates/<name>.md with its $placeholders filled; a missing value raises KeyError."""
    return Template((TEMPLATES / f"{name}.md").read_text().rstrip("\n")).substitute(values)


def status(best: tuple[float, int] | None) -> str:
    """The line shown after every turn: the agent's best score and which submission it was."""
    return (
        f"Your best score: {best[0]} bins (submission {best[1]}); lower is better."
        if best
        else "Your best score: none yet."
    )
