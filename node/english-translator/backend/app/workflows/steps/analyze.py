"""Analyze article step."""
from typing import Tuple, Optional

from app.services.llm_service import LLMService
from app.workflows.state import TranslationState, StepStatus


async def analyze_step(
    state: TranslationState,
    llm_service: LLMService
) -> Tuple[TranslationState, Optional[str]]:
    """
    Analyze the article content.

    Returns (updated_state, confirmation_content).
    confirmation_content is the content that needs user confirmation.
    """
    state.step_statuses["analyze"] = StepStatus.IN_PROGRESS

    try:
        # Analyze the article
        analysis = await llm_service.analyze_article(
            content=state.original_content,
            target_language=state.target_language
        )

        state.analysis = analysis
        state.file_paths["analysis"] = "01-analysis.md"
        state.step_statuses["analyze"] = StepStatus.WAITING_CONFIRMATION

        return state, analysis

    except Exception as e:
        state.step_statuses["analyze"] = StepStatus.FAILED
        raise e


async def complete_analyze(
    state: TranslationState,
    approved: bool,
    modifications: Optional[str] = None
) -> TranslationState:
    """Complete the analyze step after user confirmation."""
    if modifications:
        state.analysis = modifications

    state.step_statuses["analyze"] = StepStatus.COMPLETED
    state.confirmations_received["analyze"] = approved

    return state