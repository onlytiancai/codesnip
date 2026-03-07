"""Segmentation step."""
from typing import Tuple, Optional, List

from app.services.markdown_converter import MarkdownConverter
from app.workflows.state import TranslationState, StepStatus
from app.config import settings


async def segment_step(
    state: TranslationState
) -> Tuple[TranslationState, Optional[str]]:
    """
    Segment the article into chunks if needed.
    """
    state.step_statuses["segment"] = StepStatus.IN_PROGRESS

    try:
        converter = MarkdownConverter()

        # Check if segmentation is needed
        if len(state.original_content) <= settings.CHUNK_SIZE_THRESHOLD:
            # No segmentation needed
            state.segments = [state.original_content]
            state.total_chunks = 1
        else:
            # Segment the content
            chunks = converter.split_into_chunks(
                state.original_content,
                max_chunk_size=settings.MAX_CHUNK_SIZE,
                threshold=settings.CHUNK_SIZE_THRESHOLD
            )
            state.segments = [chunk for _, chunk in chunks]
            state.total_chunks = len(state.segments)

        state.step_statuses["segment"] = StepStatus.COMPLETED

        # Return summary of segmentation
        summary = f"Split into {state.total_chunks} chunk(s)"
        return state, summary

    except Exception as e:
        state.step_statuses["segment"] = StepStatus.FAILED
        raise e


def get_segment_file_path(index: int) -> str:
    """Get file path for a segment."""
    return f"04-segments/segment_{index:03d}.md"