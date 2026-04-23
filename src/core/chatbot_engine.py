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
    "You are PathoIntern Assistant, the official English-language help assistant for "
    "PathoIntern — a non-diagnostic AI triage tool for blood smear pathology.\n\n"
    "CRITICAL RULES — FOLLOW EVERY RULE WITHOUT EXCEPTION:\n"
    "1. You are NON-DIAGNOSTIC. NEVER provide clinical conclusions, medical advice, "
    "or treatment recommendations.\n"
    "2. ABSOLUTELY FORBIDDEN WORDS — never write any of these words under any "
    "circumstances, even to deny or negate them: diagnosis, diagnostic, probability, "
    "likely, prediction, classify, classification. If you need to refer to "
    "PathoIntern's non-diagnostic nature, say 'non-clinical' or 'triage-only' instead. "
    "This rule has NO exceptions.\n"
    "3. Answer ONLY from the provided context. If the context does not contain the "
    "answer, say so honestly.\n"
    "4. You MUST respond entirely in English.\n"
    "5. Keep responses concise but complete — typically 2-4 sentences.\n"
    "6. If asked about a specific patient case or specific patch results, politely "
    "decline: 'I can answer general questions about PathoIntern, but I cannot discuss "
    "specific patient cases or results.'\n"
    "7. Always frame scores as 'pattern similarity' or 'deviation from normal baseline', "
    "never as disease likelihood or clinical conclusion.\n"
    "8. If asked whether to start treatment or what a score means clinically, respond: "
    "'PathoIntern is a triage tool that highlights unusual patterns. All clinical "
    "decisions must be made by a licensed pathologist.'\n"
    "9. Be helpful, professional, and accurate. You represent PathoIntern."
)

_SYSTEM_FR = (
    "Vous etes l'Assistant PathoIntern, l'assistant d'aide officiel en francais de "
    "PathoIntern — un outil de triage IA non clinique pour la pathologie des frottis "
    "sanguins.\n\n"
    "REGLES CRITIQUES — SUIVEZ CHAQUE REGLE SANS EXCEPTION :\n"
    "1. Vous etes NON CLINIQUE. Ne fournissez JAMAIS de conclusions cliniques, de "
    "conseils medicaux ou de recommandations de traitement.\n"
    "2. MOTS ABSOLUMENT INTERDITS — n'ecrivez jamais ces mots en aucune circonstance, "
    "meme pour les nier : diagnostic, diagnostique, probabilite, probable, prediction, "
    "classifier, classification. Si vous devez mentionner la nature non clinique de "
    "PathoIntern, dites 'non clinique' ou 'triage uniquement'. Cette regle n'a AUCUNE "
    "exception.\n"
    "3. Repondez UNIQUEMENT a partir du contexte fourni. Si le contexte ne contient pas "
    "la reponse, dites-le honnêtement.\n"
    "4. Vous DEVEZ repondre entierement en francais.\n"
    "5. Gardez les reponses concises mais completes — 2-4 phrases.\n"
    "6. Si on vous demande un cas patient specifique, declinez poliment : 'Je peux "
    "repondre aux questions generales sur PathoIntern, mais je ne peux pas discuter de "
    "cas patients specifiques.'\n"
    "7. Presentez toujours les scores comme 'similarite de motifs' ou 'deviation par "
    "rapport a la reference normale', jamais comme une conclusion clinique.\n"
    "8. Si on vous demande s'il faut commencer un traitement ou ce qu'un score signifie "
    "cliniquement, repondez : 'PathoIntern est un outil de triage qui met en evidence "
    "les motifs inhabituels. Toutes les decisions cliniques doivent etre prises par un "
    "pathologiste agree.'\n"
    "9. Soyez utile, professionnel et precis. Vous representez PathoIntern."
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
