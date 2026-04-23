import logging
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Non-diagnostic safety filter
# ---------------------------------------------------------------------------

_BANNED_EN = frozenset({
    "diagnosis", "diagnostic", "probability", "likely",
    "prediction", "classify", "classification",
})
_BANNED_FR = frozenset({
    "diagnostic", "diagnostique", "probabilite", "probable",
    "prediction", "classifier", "classification",
})
_NEGATION_PREFIXES = (
    "non-", "non ", "not a ", "not ", "never ", "no ",
    "jamais ", "pas ", "aucun ",
)

_SAFE_FALLBACK_EN = (
    "PathoIntern is a triage tool that highlights unusual patterns for pathologist review. "
    "It does not provide clinical conclusions. "
    "All clinical decisions must be made by a licensed pathologist."
)
_SAFE_FALLBACK_FR = (
    "PathoIntern est un outil de triage qui met en evidence les motifs inhabituels "
    "pour examen par un pathologiste. Il ne fournit pas de conclusions cliniques. "
    "Toutes les decisions cliniques doivent etre prises par un pathologiste agree."
)


def check_non_diagnostic(response_text: str, language: str = "en") -> tuple[bool, str]:
    """
    Scan the LLM response for banned words.

    Returns:
        (True,  original_text)  — response is safe.
        (False, safe_fallback)  — banned word found; replace entirely.

    A word is only flagged when it appears WITHOUT a negation prefix
    in the 10 characters before it (e.g. "non-diagnostic" is fine).
    """
    banned = _BANNED_FR if language == "fr" else _BANNED_EN
    lower = response_text.lower()

    for word in banned:
        pos = lower.find(word)
        while pos != -1:
            prefix = lower[max(0, pos - 10): pos]
            if not any(neg in prefix for neg in _NEGATION_PREFIXES):
                fallback = _SAFE_FALLBACK_FR if language == "fr" else _SAFE_FALLBACK_EN
                logger.warning(
                    "[PathoIntern][NON-DIAG FILTER] Blocked word %r in response. "
                    "Returning safe fallback.",
                    word,
                )
                return False, fallback
            pos = lower.find(word, pos + 1)

    return True, response_text


# ---------------------------------------------------------------------------
# System prompts — 9-rule versions (EN + FR)
# ---------------------------------------------------------------------------

_SYSTEM_EN = (
    "You are PathoIntern Assistant. Below is the provided CONTEXT. "
    "Your ONLY job is to extract the answer from the CONTEXT. "
    "If the answer is not there, say 'I am sorry, I don't have that info.' "
    "Do not add opinions. Be very brief."
)

_SYSTEM_FR = (
    "Vous êtes l'assistant PathoIntern. Voici le CONTEXTE fourni. "
    "Votre SEUL travail est d'extraire la réponse du CONTEXTE. "
    "Si la réponse n'y est pas, dites 'Je suis désolé, je n'ai pas cette information.' "
    "N'ajoutez pas d'opinions. Soyez très bref."
)

_USER_TMPL_EN = (
    "Based on the following context, answer the user's question.\n\n"
    "CONTEXT:\n{context}\n\n"
    "USER QUESTION: {query}\n\n"
    "Answer concisely using only information from the context."
)

_USER_TMPL_FR = (
    "En vous basant sur le contexte suivant, repondez a la question de l'utilisateur.\n\n"
    "CONTEXTE :\n{context}\n\n"
    "QUESTION DE L'UTILISATEUR : {query}\n\n"
    "Repondez de maniere concise en utilisant uniquement les informations du contexte."
)


# ---------------------------------------------------------------------------
# LLM singleton
# ---------------------------------------------------------------------------

class LlamaManager:
    _instance = None
    _llm = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self):
        if self._llm is not None:
            return self._llm

        try:
            from llama_cpp import Llama

            root_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            model_path = os.path.join(root_dir, "models", "llm", "LFM2.5-350M-Q4_K_M.gguf")

            if not os.path.exists(model_path):
                logger.error("[PathoIntern] Chatbot LLM not found at %s", model_path)
                raise FileNotFoundError(f"LLM not found at {model_path}")

            logger.info("[PathoIntern] Loading LLM from %s…", model_path)
            self._llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)
            logger.info("[PathoIntern] LLM loaded successfully.")
            return self._llm

        except ImportError:
            logger.error("[PathoIntern] llama-cpp-python is not installed.")
            raise

    def generate_rag_response(
        self,
        query: str,
        context: str,
        language: str = "en",
        history: list[dict] | None = None,
    ) -> str:
        """
        Generate a RAG response for ``query`` using ``context`` as grounding.

        Args:
            query:    The user's question.
            context:  Retrieved KB entries formatted as [Source N] blocks.
                      Empty string when confidence < threshold (caller handles fallback).
            language: "en" or "fr" — selects system prompt and user message template.
            history:  Prior conversation turns as [{"role": "user"|"assistant", "content": str}].
                      Last 8 items (4 exchanges) are injected between system prompt and current
                      user message so the LLM can handle follow-up questions correctly.

        Returns:
            The LLM's response, post-filtered by check_non_diagnostic().
        """
        llm = self.load_model()

        system_prompt = _SYSTEM_FR if language == "fr" else _SYSTEM_EN
        user_tmpl = _USER_TMPL_FR if language == "fr" else _USER_TMPL_EN
        user_message = user_tmpl.format(context=context, query=query)

        messages: list[dict] = [{"role": "system", "content": system_prompt}]

        # Inject up to 4 prior exchanges (8 messages) for conversational context.
        if history:
            for turn in history[-8:]:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_message})

        raw = llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.3,
            top_p=0.9,
        )["choices"][0]["message"]["content"].strip()

        _, safe_response = check_non_diagnostic(raw, language)
        return safe_response


# ---------------------------------------------------------------------------
# Whisper singleton
# ---------------------------------------------------------------------------

class WhisperManager:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self):
        if self._model is not None:
            return self._model

        try:
            import whisper

            logger.info("[PathoIntern] Loading Whisper base model…")
            self._model = whisper.load_model("base")
            logger.info("[PathoIntern] Whisper base model loaded.")
            return self._model
        except ImportError:
            logger.error("[PathoIntern] openai-whisper is not installed.")
            raise

    def transcribe_audio(self, file_path: str, language: str = "en") -> str:
        model = self.load_model()
        logger.info("[PathoIntern] Transcribing %s (language=%s)…", file_path, language)
        result = model.transcribe(file_path, language=language)
        logger.info("[PathoIntern] Transcription complete.")
        return result["text"].strip()


# Module-level singletons
llama_manager = LlamaManager()
whisper_manager = WhisperManager()
