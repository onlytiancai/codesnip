"""Prompt generation step."""
from typing import Tuple, Optional

from app.services.llm_service import LLMService
from app.workflows.state import TranslationState, StepStatus


async def prompt_gen_step(
    state: TranslationState,
    llm_service: LLMService
) -> Tuple[TranslationState, Optional[str]]:
    """
    Generate translation prompt based on analysis and terminology.
    """
    state.step_statuses["prompt_gen"] = StepStatus.IN_PROGRESS

    try:
        # Generate translation prompt
        prompt = await llm_service.generate_translation_prompt(
            content=state.original_content,
            analysis=state.analysis,
            terminology=state.terminology,
            target_language=state.target_language
        )

        state.translation_prompt = prompt
        state.file_paths["prompt"] = "03-prompt.md"
        state.step_statuses["prompt_gen"] = StepStatus.WAITING_CONFIRMATION

        return state, prompt

    except Exception as e:
        state.step_statuses["prompt_gen"] = StepStatus.FAILED
        raise e


async def complete_prompt_gen(
    state: TranslationState,
    approved: bool,
    modifications: Optional[str] = None
) -> TranslationState:
    """Complete the prompt_gen step after user confirmation."""
    if modifications:
        state.translation_prompt = modifications

    state.step_statuses["prompt_gen"] = StepStatus.COMPLETED
    state.confirmations_received["prompt_gen"] = approved

    return state