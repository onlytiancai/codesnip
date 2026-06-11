"""Review step."""
from typing import Tuple, Optional

from app.services.llm_service import LLMService
from app.workflows.state import TranslationState, StepStatus


async def review_step(
    state: TranslationState,
    llm_service: LLMService
) -> Tuple[TranslationState, Optional[str]]:
    """
    Review the translation for quality.
    """
    state.step_statuses["review"] = StepStatus.IN_PROGRESS

    try:
        # Review the translation
        critique = await llm_service.review_translation(
            original=state.original_content,
            translation=state.draft,
            target_language=state.target_language
        )

        state.critique = critique
        state.file_paths["critique"] = "07-critique.md"
        state.step_statuses["review"] = StepStatus.WAITING_CONFIRMATION

        return state, critique

    except Exception as e:
        state.step_statuses["review"] = StepStatus.FAILED
        raise e


async def complete_review(
    state: TranslationState,
    approved: bool,
    modifications: Optional[str] = None
) -> TranslationState:
    """Complete the review step after user confirmation."""
    if modifications:
        state.critique = modifications

    state.step_statuses["review"] = StepStatus.COMPLETED
    state.confirmations_received["review"] = approved

    return state


async def revise_step(
    state: TranslationState,
    llm_service: LLMService
) -> Tuple[TranslationState, Optional[str]]:
    """
    Revise the translation based on review feedback.
    """
    state.step_statuses["revise"] = StepStatus.IN_PROGRESS

    try:
        # Revise based on critique
        revision = await llm_service.revise_translation(
            original=state.original_content,
            translation=state.draft,
            review=state.critique,
            target_language=state.target_language
        )

        state.revision = revision
        state.file_paths["revision"] = "08-revision.md"
        state.step_statuses["revise"] = StepStatus.COMPLETED

        return state, revision

    except Exception as e:
        state.step_statuses["revise"] = StepStatus.FAILED
        raise e


async def polish_step(
    state: TranslationState,
    llm_service: LLMService
) -> Tuple[TranslationState, Optional[str]]:
    """
    Polish the final translation.
    """
    state.step_statuses["polish"] = StepStatus.IN_PROGRESS

    try:
        # Use revision if available, otherwise use draft
        content_to_polish = state.revision or state.draft

        # Polish the translation
        polished = await llm_service.polish_translation(
            translation=content_to_polish,
            target_language=state.target_language
        )

        state.final_translation = polished
        state.file_paths["final"] = "translation.md"
        state.step_statuses["polish"] = StepStatus.COMPLETED

        return state, polished

    except Exception as e:
        state.step_statuses["polish"] = StepStatus.FAILED
        raise e