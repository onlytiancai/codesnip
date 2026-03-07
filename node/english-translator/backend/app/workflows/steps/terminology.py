"""Terminology extraction step."""
from typing import Tuple, Optional

from app.services.llm_service import LLMService
from app.workflows.state import TranslationState, StepStatus


async def terminology_step(
    state: TranslationState,
    llm_service: LLMService
) -> Tuple[TranslationState, Optional[str]]:
    """
    Extract terminology from the article.
    """
    state.step_statuses["terminology"] = StepStatus.IN_PROGRESS

    try:
        # Extract terminology
        terminology = await llm_service.extract_terminology(
            content=state.original_content,
            target_language=state.target_language
        )

        state.terminology = terminology
        state.file_paths["terminology"] = "02-terminology.md"
        state.step_statuses["terminology"] = StepStatus.COMPLETED

        return state, None

    except Exception as e:
        state.step_statuses["terminology"] = StepStatus.FAILED
        raise e