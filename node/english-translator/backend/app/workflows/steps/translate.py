"""Translation step."""
from typing import Tuple, Optional, List
import asyncio

from app.services.llm_service import LLMService
from app.workflows.state import TranslationState, StepStatus
from app.workflows.steps.segment import get_segment_file_path


async def translate_step(
    state: TranslationState,
    llm_service: LLMService,
    progress_callback=None
) -> Tuple[TranslationState, Optional[str]]:
    """
    Translate all segments.
    """
    state.step_statuses["translate"] = StepStatus.IN_PROGRESS
    state.translations = []

    try:
        # Translate segments in parallel
        tasks = []
        for i, segment in enumerate(state.segments):
            task = translate_segment(
                index=i,
                segment=segment,
                llm_service=llm_service,
                translation_prompt=state.translation_prompt,
                terminology=state.terminology,
                target_language=state.target_language
            )
            tasks.append(task)

        # Run translations with progress updates
        results = []
        for i, task in enumerate(asyncio.as_completed(tasks)):
            try:
                result = await task
                results.append(result)
                state.completed_chunks = len(results)
                if progress_callback:
                    await progress_callback(
                        f"Translated {len(results)}/{state.total_chunks} chunks"
                    )
            except Exception as e:
                # Handle individual translation errors
                results.append((i, f"Error: {str(e)}", False))

        # Sort results by index and extract translations
        results.sort(key=lambda x: x[0])
        state.translations = [r[1] for r in results]

        # Set file paths
        for i in range(len(state.translations)):
            state.file_paths[f"translation_{i}"] = f"05-translations/translation_{i:03d}.md"

        state.step_statuses["translate"] = StepStatus.COMPLETED

        # Return draft (merged translations)
        draft = merge_translations(state.translations)
        state.draft = draft
        state.file_paths["draft"] = "06-draft.md"

        return state, draft

    except Exception as e:
        state.step_statuses["translate"] = StepStatus.FAILED
        raise e


async def translate_segment(
    index: int,
    segment: str,
    llm_service: LLMService,
    translation_prompt: str,
    terminology: str,
    target_language: str
) -> Tuple[int, str, bool]:
    """
    Translate a single segment.
    Returns (index, translation, success).
    """
    try:
        translation = await llm_service.translate(
            content=segment,
            translation_prompt=translation_prompt,
            terminology=terminology,
            target_language=target_language
        )
        return (index, translation, True)
    except Exception as e:
        return (index, f"Translation failed: {str(e)}", False)


def merge_translations(translations: List[str]) -> str:
    """Merge translated segments into a single document."""
    return "\n\n".join(translations)