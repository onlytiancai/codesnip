"""LLM service for translation."""
from typing import Optional, AsyncIterator
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.models.schemas import LLMSettings


class LLMService:
    """Service for LLM interactions."""

    def __init__(self, llm_settings: Optional[LLMSettings] = None):
        self.settings = llm_settings or LLMSettings(
            provider=settings.LLM_PROVIDER,
            model=settings.OPENAI_MODEL if settings.LLM_PROVIDER == "openai" else settings.ANTHROPIC_MODEL,
            api_key=settings.OPENAI_API_KEY if settings.LLM_PROVIDER == "openai" else settings.ANTHROPIC_API_KEY,
            api_base=settings.OPENAI_API_BASE if settings.LLM_PROVIDER == "openai" else settings.ANTHROPIC_API_BASE
        )
        self._llm = None

    @property
    def llm(self):
        """Get or create LLM instance."""
        if self._llm is None:
            if self.settings.provider == "openai":
                llm_kwargs = {
                    "model": self.settings.model,
                    "api_key": self.settings.api_key,
                    "temperature": 0.3
                }
                if self.settings.api_base:
                    llm_kwargs["base_url"] = self.settings.api_base
                self._llm = ChatOpenAI(**llm_kwargs)
            elif self.settings.provider == "anthropic":
                llm_kwargs = {
                    "model": self.settings.model,
                    "api_key": self.settings.api_key,
                    "temperature": 0.3
                }
                if self.settings.api_base:
                    llm_kwargs["anthropic_api_url"] = self.settings.api_base
                self._llm = ChatAnthropic(**llm_kwargs)
            else:
                raise ValueError(f"Unsupported LLM provider: {self.settings.provider}")
        return self._llm

    async def analyze_article(self, content: str, target_language: str) -> str:
        """Analyze article content for translation."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert translation analyst. Analyze the following article and provide:
1. A brief summary of the content and main themes
2. Key terminology and specialized vocabulary (with suggested translations)
3. Cultural references or metaphors that need special attention
4. Tone and style analysis (formal/informal, technical/general audience)
5. Translation challenges and recommendations

Format your response in Markdown with clear sections."""),
            ("human", "Article to analyze:\n\n{content}\n\nTarget language: {target_language}")
        ])

        chain = prompt | self.llm
        response = await chain.ainvoke({
            "content": content,
            "target_language": target_language
        })
        return response.content

    async def extract_terminology(self, content: str, target_language: str) -> str:
        """Extract terminology from content."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a terminology extraction expert. Extract key terms from the article that:
1. Are technical or specialized terms
2. Have specific translations that need to be consistent
3. Are proper nouns (names, places, organizations)
4. Could be translated multiple ways and need standardization

Format the output as a Markdown table with columns:
| Original Term | Suggested Translation | Notes |

Only include terms that truly need attention. Don't include common terms that are straightforward to translate."""),
            ("human", "Article:\n\n{content}\n\nTarget language: {target_language}")
        ])

        chain = prompt | self.llm
        response = await chain.ainvoke({
            "content": content,
            "target_language": target_language
        })
        return response.content

    async def generate_translation_prompt(
        self,
        content: str,
        analysis: str,
        terminology: str,
        target_language: str,
        style_guide: str = ""
    ) -> str:
        """Generate translation prompt based on analysis."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a translation prompt engineering expert. Create a comprehensive translation prompt that includes:
1. Context and background from the analysis
2. Terminology guidelines
3. Style and tone requirements
4. Specific instructions for handling:
   - Metaphors and cultural references
   - Technical terms
   - Sentence structure adaptation
   - Quality standards

The prompt should be detailed enough for a translator to produce high-quality output."""),
            ("human", """Content summary:
{content_summary}

Analysis:
{analysis}

Terminology:
{terminology}

Target language: {target_language}

Style guide (optional): {style_guide}

Generate a comprehensive translation prompt:""")
        ])

        chain = prompt | self.llm
        response = await chain.ainvoke({
            "content_summary": content[:2000],
            "analysis": analysis,
            "terminology": terminology,
            "target_language": target_language,
            "style_guide": style_guide or "No specific style guide provided."
        })
        return response.content

    async def translate(
        self,
        content: str,
        translation_prompt: str,
        terminology: str,
        target_language: str
    ) -> str:
        """Translate content using provided prompt and terminology."""
        system_prompt = f"""You are an expert translator. Use the following translation guidelines:

{translation_prompt}

Terminology to follow:
{terminology}

Target language: {target_language}

Important rules:
1. Translate naturally and fluently, not word-by-word
2. Maintain the original tone and style
3. Preserve formatting (headers, lists, code blocks)
4. Keep technical terms consistent
5. Adapt metaphors appropriately for the target audience
6. Keep code, URLs, and technical identifiers unchanged"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Translate the following content:\n\n{content}")
        ]

        response = await self.llm.ainvoke(messages)
        return response.content

    async def review_translation(
        self,
        original: str,
        translation: str,
        target_language: str
    ) -> str:
        """Review and critique translation quality."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert translation reviewer. Analyze the translation and provide:
1. Overall quality assessment (score 1-10)
2. Accuracy check (any mistranslations or omissions)
3. Fluency check (awkward phrasing or unnatural language)
4. Consistency check (terminology, style, tone)
5. Specific issues with line numbers and suggestions
6. Overall recommendations

Format your response in Markdown with clear sections."""),
            ("human", """Original:
{original}

Translation:
{translation}

Target language: {target_language}

Provide your review:""")
        ])

        chain = prompt | self.llm
        response = await chain.ainvoke({
            "original": original,
            "translation": translation,
            "target_language": target_language
        })
        return response.content

    async def revise_translation(
        self,
        original: str,
        translation: str,
        review: str,
        target_language: str
    ) -> str:
        """Revise translation based on review."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert translator. Revise the translation based on the review feedback.
Address all identified issues while maintaining the strengths of the original translation."""),
            ("human", """Original:
{original}

Current Translation:
{translation}

Review Feedback:
{review}

Target language: {target_language}

Provide the revised translation:""")
        ])

        chain = prompt | self.llm
        response = await chain.ainvoke({
            "original": original,
            "translation": translation,
            "review": review,
            "target_language": target_language
        })
        return response.content

    async def polish_translation(self, translation: str, target_language: str) -> str:
        """Polish translation for better flow and style."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a language polishing expert. Improve the text for:
1. Better flow and readability
2. More natural expression
3. Appropriate style and tone
4. Consistent voice

Do not change the meaning, only improve the expression."""),
            ("human", """Text to polish:
{translation}

Target language: {target_language}

Provide the polished version:""")
        ])

        chain = prompt | self.llm
        response = await chain.ainvoke({
            "translation": translation,
            "target_language": target_language
        })
        return response.content

    async def fast_translate(self, content: str, target_language: str) -> str:
        """Quick translation without analysis."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert translator. Translate the content to {target_language}.
- Translate naturally and fluently
- Preserve formatting
- Keep code and technical terms unchanged"""),
            ("human", "{content}")
        ])

        chain = prompt | self.llm
        response = await chain.ainvoke({
            "content": content,
            "target_language": target_language
        })
        return response.content