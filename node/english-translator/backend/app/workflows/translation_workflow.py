"""Main translation workflow."""
from typing import Callable, Optional, Awaitable
import asyncio

from app.models.schemas import WorkflowStep, ProjectStatus, TranslationMode
from app.services.project_service import ProjectService
from app.services.llm_service import LLMService
from app.services.extractor import ExtractorService
from app.services.markdown_converter import MarkdownConverter
from app.workflows.state import (
    TranslationState,
    StepStatus,
    FAST_MODE_STEPS,
    NORMAL_MODE_STEPS,
    FINE_MODE_STEPS,
    CONFIRMATION_STEPS,
)
from app.workflows.steps.analyze import analyze_step, complete_analyze
from app.workflows.steps.terminology import terminology_step
from app.workflows.steps.prompt_gen import prompt_gen_step, complete_prompt_gen
from app.workflows.steps.segment import segment_step, get_segment_file_path
from app.workflows.steps.translate import translate_step
from app.workflows.steps.review import review_step, revise_step, polish_step


class TranslationWorkflow:
    """Main translation workflow orchestrator."""

    def __init__(self, project_service: ProjectService):
        self.project_service = project_service
        self.llm_service = LLMService()
        self.extractor = ExtractorService()
        self.converter = MarkdownConverter()

    async def run(
        self,
        project_id: str,
        resume: bool = False,
        progress_callback: Optional[Callable[[WorkflowStep, str, dict], Awaitable[None]]] = None,
        confirmation_callback: Optional[Callable[[WorkflowStep, str, str], Awaitable[None]]] = None
    ):
        """
        Run the translation workflow.

        Args:
            project_id: Project ID
            resume: Whether to resume from saved state
            progress_callback: Callback for progress updates
            confirmation_callback: Callback when user confirmation is needed
        """
        # Load or initialize state
        if resume:
            state = await self._load_state(project_id)
            if not state:
                raise ValueError("Cannot resume: no saved state found")
        else:
            state = await self._initialize_state(project_id)

        mode = state.mode

        # Get steps for the mode
        steps = self._get_steps_for_mode(mode)
        confirmation_steps = CONFIRMATION_STEPS.get(mode, [])

        try:
            # Update project status
            await self.project_service.update_project_status(
                project_id, ProjectStatus.IN_PROGRESS, WorkflowStep(state.current_step)
            )

            for step in steps:
                if state.step_statuses.get(step) == StepStatus.COMPLETED:
                    continue

                # Update current step
                state.current_step = step
                await self._save_state(project_id, state)

                # Execute step
                state, confirmation_content = await self._execute_step(
                    step, state, progress_callback
                )

                # Save state after each step
                await self._save_state(project_id, state)

                # Check if confirmation is needed
                if step in confirmation_steps and confirmation_content:
                    # Wait for user confirmation
                    await self.project_service.update_project_status(
                        project_id, ProjectStatus.WAITING_CONFIRMATION
                    )

                    file_path = state.file_paths.get(step.replace("_gen", ""), "")
                    if confirmation_callback:
                        await confirmation_callback(
                            WorkflowStep(step),
                            confirmation_content,
                            file_path
                        )

                    # Wait for confirmation
                    state = await self._wait_for_confirmation(
                        project_id, step, state
                    )

                    await self.project_service.update_project_status(
                        project_id, ProjectStatus.IN_PROGRESS
                    )

                # Save files
                await self._save_step_files(project_id, step, state)

            # Mark as completed
            state.current_step = "done"
            state.step_statuses["done"] = StepStatus.COMPLETED
            await self._save_state(project_id, state)

            # Save final translation
            await self._save_final_translation(project_id, state)

            await self.project_service.update_project_status(
                project_id, ProjectStatus.COMPLETED, WorkflowStep.DONE
            )

            if progress_callback:
                await progress_callback(
                    WorkflowStep.DONE,
                    "Translation completed",
                    {"final_file": state.file_paths.get("final", "translation.md")}
                )

        except Exception as e:
            await self.project_service.update_project_status(
                project_id, ProjectStatus.ERROR
            )
            raise e

    async def _initialize_state(self, project_id: str) -> TranslationState:
        """Initialize workflow state from project metadata."""
        project = await self.project_service.get_project(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")

        # Extract content from URL
        content, title, error = await self.extractor.extract_content(project.url)
        if error:
            raise ValueError(f"Failed to extract content: {error}")

        # Save original content
        await self.project_service.save_file(project_id, "original.md", content)

        # Update project title
        if title:
            await self.project_service.update_project_title(project_id, title)

        state = TranslationState(
            project_id=project_id,
            url=project.url,
            mode=project.mode.value,
            target_language=project.target_language,
            original_content=content,
            title=title,
            current_step="extract",
            step_statuses={"extract": StepStatus.COMPLETED},
            file_paths={"original": "original.md"}
        )

        return state

    async def _load_state(self, project_id: str) -> Optional[TranslationState]:
        """Load saved workflow state."""
        workflow_state = await self.project_service.get_workflow_state(project_id)
        if not workflow_state:
            return None

        # Load original content
        original = await self.project_service.get_file(project_id, "original.md")

        state = TranslationState(
            project_id=project_id,
            url=workflow_state.step_results.get("url", ""),
            mode=workflow_state.mode.value,
            target_language=workflow_state.step_results.get("target_language", "中文"),
            original_content=original or "",
            current_step=workflow_state.current_step.value,
            step_statuses=workflow_state.step_results.get("step_statuses", {}),
            analysis=await self._load_file(project_id, "01-analysis.md"),
            terminology=await self._load_file(project_id, "02-terminology.md"),
            translation_prompt=await self._load_file(project_id, "03-prompt.md"),
            draft=await self._load_file(project_id, "06-draft.md"),
            critique=await self._load_file(project_id, "07-critique.md"),
            revision=await self._load_file(project_id, "08-revision.md"),
            final_translation=await self._load_file(project_id, "translation.md"),
        )

        return state

    async def _load_file(self, project_id: str, filename: str) -> str:
        """Load file content."""
        content = await self.project_service.get_file(project_id, filename)
        return content or ""

    async def _save_state(self, project_id: str, state: TranslationState):
        """Save workflow state."""
        from app.models.schemas import WorkflowState

        workflow_state = WorkflowState(
            project_id=project_id,
            mode=TranslationMode(state.mode),
            current_step=WorkflowStep(state.current_step),
            step_results={
                "step_statuses": state.step_statuses,
                "url": state.url,
                "target_language": state.target_language,
            }
        )

        await self.project_service.update_workflow_state(project_id, workflow_state)

    async def _execute_step(
        self,
        step: str,
        state: TranslationState,
        progress_callback
    ) -> tuple:
        """Execute a single workflow step."""
        if progress_callback:
            await progress_callback(
                WorkflowStep(step),
                f"Starting {step} step",
                {}
            )

        if step == "extract":
            # Already done during initialization
            return state, None

        elif step == "analyze":
            return await analyze_step(state, self.llm_service)

        elif step == "terminology":
            return await terminology_step(state, self.llm_service)

        elif step == "prompt_gen":
            return await prompt_gen_step(state, self.llm_service)

        elif step == "segment":
            return await segment_step(state)

        elif step == "translate":
            async def progress(msg):
                if progress_callback:
                    await progress_callback(WorkflowStep.TRANSLATE, msg, {})
            return await translate_step(state, self.llm_service, progress)

        elif step == "review":
            return await review_step(state, self.llm_service)

        elif step == "revise":
            return await revise_step(state, self.llm_service)

        elif step == "polish":
            return await polish_step(state, self.llm_service)

        elif step == "done":
            return state, None

        else:
            raise ValueError(f"Unknown step: {step}")

    async def _wait_for_confirmation(
        self,
        project_id: str,
        step: str,
        state: TranslationState
    ) -> TranslationState:
        """Wait for user confirmation of a step."""
        import asyncio

        while True:
            # Check for confirmation
            workflow_state = await self.project_service.get_workflow_state(project_id)
            if workflow_state:
                confirmed = workflow_state.step_results.get(f"{step}_confirmed")
                if confirmed is not None:
                    modifications = workflow_state.step_results.get(f"{step}_modifications")

                    # Complete the step
                    if step == "analyze":
                        return await complete_analyze(state, confirmed, modifications)
                    elif step == "prompt_gen":
                        return await complete_prompt_gen(state, confirmed, modifications)
                    elif step == "review":
                        from app.workflows.steps.review import complete_review
                        return await complete_review(state, confirmed, modifications)

            # Wait a bit before checking again
            await asyncio.sleep(1)

    async def _save_step_files(self, project_id: str, step: str, state: TranslationState):
        """Save files for a completed step."""
        if step == "analyze" and state.analysis:
            await self.project_service.save_file(
                project_id, "01-analysis.md", state.analysis
            )

        elif step == "terminology" and state.terminology:
            await self.project_service.save_file(
                project_id, "02-terminology.md", state.terminology
            )

        elif step == "prompt_gen" and state.translation_prompt:
            await self.project_service.save_file(
                project_id, "03-prompt.md", state.translation_prompt
            )

        elif step == "segment":
            for i, segment in enumerate(state.segments):
                await self.project_service.save_file(
                    project_id, get_segment_file_path(i), segment
                )

        elif step == "translate":
            for i, translation in enumerate(state.translations):
                await self.project_service.save_file(
                    project_id, f"05-translations/translation_{i:03d}.md", translation
                )
            if state.draft:
                await self.project_service.save_file(
                    project_id, "06-draft.md", state.draft
                )

        elif step == "review" and state.critique:
            await self.project_service.save_file(
                project_id, "07-critique.md", state.critique
            )

        elif step == "revise" and state.revision:
            await self.project_service.save_file(
                project_id, "08-revision.md", state.revision
            )

        elif step == "polish" and state.final_translation:
            await self.project_service.save_file(
                project_id, "translation.md", state.final_translation
            )

    async def _save_final_translation(self, project_id: str, state: TranslationState):
        """Save the final translation file."""
        final = state.final_translation or state.revision or state.draft
        if final:
            await self.project_service.save_file(
                project_id, "translation.md", final
            )

    def _get_steps_for_mode(self, mode: str) -> list:
        """Get workflow steps for a translation mode."""
        if mode == "fast":
            return FAST_MODE_STEPS
        elif mode == "fine":
            return FINE_MODE_STEPS
        else:
            return NORMAL_MODE_STEPS