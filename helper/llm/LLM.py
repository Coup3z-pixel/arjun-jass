from typing import Type, Tuple, Optional, List, Dict, Any
from pydantic import BaseModel
from google import genai
from google.genai import types as genai_types
from openai import OpenAI as OpenAIClient, AsyncOpenAI as AsyncOpenAIClient, BadRequestError
import json, os, re

# =========================
# Models / Schemas
# =========================

class AnswerFormat(BaseModel):
    reasoning: str
    value: int

JSON_SYSTEM_MSG = {
    "role": "system",
    "content": (
        "You are playing a game where you need to make decisions. "
        "Always respond with a JSON object containing your reasoning and choice. "
        'Format: {"reasoning": "your explanation", "value": your_choice_number}. '
        "Make sure to include both reasoning and value in your response."
    )
}

# =========================
# Provider detection helpers
# =========================

def _is_openai_model(model: str) -> bool:
    if not model:
        return False
    m = model.strip()
    return (
        m.startswith("openai:") or
        m.startswith("ft:") or
        m in {
            "gpt-4o", "gpt-4.1", "gpt-4o-mini", "gpt-4.1-mini",
            "o4", "o4-mini", "gpt-4.1-nano", "gpt-4o-audio-preview",
            "chatgpt-4o-latest"
        } or
        m.startswith("gpt-") or m.startswith("o4")
    )

def _is_vertex_model(m: str) -> bool:
    if not m: return False
    m = m.strip()
    return (
        m.startswith("vertex:") or
        m.startswith("projects/") or
        m.startswith("publishers/") or
        m.startswith("gemini-") or "gemini" in m
    )

def _is_together_model(model: str) -> bool:
    if not model:
        return False
    return model.strip().startswith("together:")

def _normalize_model_for_together(model: str) -> str:
    return model[len("together:"):] if model.startswith("together:") else model

def _normalize_model_for_openai(model: str) -> str:
    return model[len("openai:"):] if model.startswith("openai:") else model

# Heuristic: some Together slugs are known/likely non-chat (base or fine-tuned refs)
def _is_together_completion_only_slug(name: str) -> bool:
    n = (name or "").lower()
    return any(tag in n for tag in [
        "-base", "reference", "mixtral-8x7b-v0.1"
    ])

# =========================
# LLM Router
# =========================

class LLM:
    """
    Unified LLM helper that routes automatically:
      - Vertex AI (Gemini & tuned endpoints) via google-genai when model looks like Vertex
      - OpenAI API when model looks like OpenAI
      - Together/OpenRouter via OpenAI-compatible APIs otherwise

    Adds: Together fallback to /completions when chat isn’t supported.
    """

    def __init__(self, model: str) -> None:
        self.model = (model or "").strip()
        self.history: List[Dict[str, Any]] = []

        if _is_openai_model(self.model):
            self.provider = "openai"
            self.oa_model = _normalize_model_for_openai(self.model)
            self.client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
            self.async_client = AsyncOpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))

        elif _is_together_model(self.model):
            self.provider = "together"
            self.oa_model = _normalize_model_for_together(self.model)
            self.client = OpenAIClient(
                base_url="https://api.together.xyz/v1",
                api_key=os.getenv("TOGETHER_API_KEY"),
            )
            self.async_client = AsyncOpenAIClient(
                base_url="https://api.together.xyz/v1",
                api_key=os.getenv("TOGETHER_API_KEY"),
            )

        elif _is_vertex_model(self.model):
            self.provider = "vertex"
            self.vertex_model = self.model[len("vertex:"):] if self.model.startswith("vertex:") else self.model
            self.client = genai.Client(
                vertexai=True,
                project=os.getenv("GOOGLE_CLOUD_PROJECT") or "buoyant-ground-472514-s0",
                location=os.getenv("GOOGLE_CLOUD_LOCATION") or "us-central1",
            )
            # google-genai handles async on same client

        else:
            self.provider = "openrouter"
            self.oa_model = self.model
            self.client = OpenAIClient(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPEN_ROUTER_API_KEY"),
            )
            self.async_client = AsyncOpenAIClient(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPEN_ROUTER_API_KEY"),
            )

    def get_model_name(self) -> str:
        """Return the original model string passed to LLM (e.g., 'together:…')."""
        return self.model

    # -------------------------
    # Session state
    # -------------------------
    def restart_model(self) -> None:
        self.history = []

    def _ensure_system_json_message(self):
        if not self.history or self.history[0].get("role") != "system":
            self.history.insert(0, JSON_SYSTEM_MSG)

    # -------------------------
    # Helpers
    # -------------------------
    def _history_to_prompt(self) -> str:
        """Render chat history into a single prompt for non-chat completions."""
        lines = []
        if self.history and self.history[0].get("role") == "system":
            lines.append(f"System: {self.history[0]['content']}".strip())
        for m in self.history:
            role = m.get("role")
            if role == "system":
                continue
            content = str(m.get("content", "")).strip()
            if not content:
                continue
            if role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant":
                lines.append(f"Assistant: {content}")
            else:
                lines.append(f"{role.capitalize()}: {content}")
        lines.append("Assistant:")
        return "\n".join(lines)

    def _parse_answerformat_from_content(self, content) -> AnswerFormat:
        if isinstance(content, list):
            content = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
        text = str(content).strip()
        if text.startswith("```"):
            lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()

        # Try strict parse
        try:
            return AnswerFormat.model_validate_json(text)
        except Exception:
            pass

        # Extract JSON objects from text
        candidates = []
        brace_stack = []
        start_idx = None
        for i, ch in enumerate(text):
            if ch == '{':
                if not brace_stack:
                    start_idx = i
                brace_stack.append('{')
            elif ch == '}':
                if brace_stack:
                    brace_stack.pop()
                    if not brace_stack and start_idx is not None:
                        candidates.append(text[start_idx:i+1])
                        start_idx = None

        def try_parse(obj: str) -> Optional[dict]:
            try:
                return json.loads(obj)
            except Exception:
                return None

        good = None
        for obj in reversed(candidates):
            data = try_parse(obj)
            if isinstance(data, dict) and "reasoning" in data and "value" in data:
                try:
                    return AnswerFormat.model_validate(data)
                except Exception:
                    good = data if good is None else good

        # Rough repair - try to extract choice from natural language
        # Look for patterns like "choose 1", "cooperate", "defect", etc.
        choice_patterns = [
            r'choose\s+(\d+)',
            r'(\d+)\s*for\s+(?:cooperate|defect)',
            r'(?:cooperate|cooperation)',
            r'(?:defect|defection)',
        ]
        
        choice_value = None
        for pattern in choice_patterns:
            match = re.search(pattern, text.lower())
            if match:
                if 'cooperate' in pattern:
                    choice_value = 1
                elif 'defect' in pattern:
                    choice_value = 2
                elif match.group(1):
                    choice_value = int(match.group(1))
                break
        
        # If no choice found, try to extract any number
        if choice_value is None:
            m3 = re.search(r'(-?\d+)', text)
            if m3:
                choice_value = int(m3.group(1))
        
        if choice_value is not None:
            return AnswerFormat.model_validate({"reasoning": text[:5000], "value": choice_value})

        raise ValueError(f"Expected JSON with keys 'reasoning' and 'value'. Got: {text[:200]}...")

    # -------------------------
    # Vertex helpers
    # -------------------------
    def _vertex_schema_answerformat(self) -> genai_types.Schema:
        return genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "reasoning": genai_types.Schema(type=genai_types.Type.STRING),
                "value": genai_types.Schema(type=genai_types.Type.INTEGER),
            },
            required=["reasoning", "value"],
        )

    def _vertex_system_instruction(self) -> str:
        if self.history and self.history[0].get("role") == "system":
            return str(self.history[0].get("content") or "")
        return (
            "You are a function that returns JSON only. "
            'Output must be: {"reasoning": string, "value": integer}. '
            "Return JSON only."
        )

    def _history_to_vertex_contents(self) -> List[Dict[str, Any]]:
        contents: List[Dict[str, Any]] = []
        for msg in self.history:
            role = msg.get("role")
            if role == "system":
                continue
            text = msg.get("content", "")
            if text is None:
                continue
            vrole = "model" if role == "assistant" else "user"
            contents.append({"role": vrole, "parts": [{"text": str(text)}]})
        return contents

    # -------------------------
    # Public API
    # -------------------------
    def _together_needs_completions(self, exc: Exception) -> bool:
        if getattr(self, "provider", "") != "together":
            return False
        msg = str(exc).lower()
        return any(k in msg for k in [
            "inputs cannot be empty", "missing inputs", "use prompt",
            "not a chat model", "invalid_request_error"
        ])

    def _use_completions_first(self) -> bool:
        return getattr(self, "provider", "") == "together" and _is_together_completion_only_slug(self.oa_model)

    def ask(self, prompt) -> Tuple[int, str]:
        self._ensure_system_json_message()
        self.history.append({"role": "user", "content": prompt})

        # ---------- Vertex ----------
        if getattr(self, "provider", "") == "vertex":
            try:
                resp = self.client.models.generate_content(
                    model=self.vertex_model,
                    contents=self._history_to_vertex_contents(),
                    config=genai_types.GenerateContentConfig(
                        system_instruction=self._vertex_system_instruction(),
                        response_mime_type="application/json",
                        response_schema=self._vertex_schema_answerformat(),
                        temperature=0,
                    ),
                )
                try:
                    part = resp.candidates[0].content.parts[0]
                    data = part.as_dict if hasattr(part, "as_dict") else json.loads(getattr(part, "text", resp.text))
                except Exception:
                    data = json.loads(resp.text)
                return (int(data["value"]), str(data["reasoning"]))
            except Exception as e:
                return (50, f"API Error (Vertex): {str(e)}")

        # ---------- OpenAI-compatible (OpenAI / Together / OpenRouter) ----------
        # If we KNOW the Together model is completions-only, go straight there
        if self._use_completions_first():
            try:
                prompt_all = self._history_to_prompt()
                resp = self.client.completions.create(
                    model=self.oa_model,
                    prompt=prompt_all,
                    max_tokens=1024,
                    temperature=0
                )
                text = resp.choices[0].text
                af = self._parse_answerformat_from_content(text)
                return (af.value, af.reasoning)
            except Exception as e:
                return (50, f"API Error (completions): {str(e)}")

        # Otherwise, try chat then fall back
        try:
            resp = self.client.chat.completions.create(
                model=self.oa_model,
                messages=self.history,
                response_format={"type": "json_object"},
                temperature=0
            )
            # Parse chat
            choice = resp.choices[0]
            parsed = getattr(choice.message, "parsed", None)
            if parsed is not None:
                try:
                    af = parsed if isinstance(parsed, AnswerFormat) else AnswerFormat.model_validate(parsed)
                except Exception:
                    af = self._parse_answerformat_from_content(choice.message.content)
            else:
                af = self._parse_answerformat_from_content(choice.message.content)
            return (af.value, af.reasoning)

        except BadRequestError as e:
            if self._together_needs_completions(e):
                try:
                    prompt_all = self._history_to_prompt()
                    resp = self.client.completions.create(
                        model=self.oa_model,
                        prompt=prompt_all,
                        max_tokens=1024,
                        temperature=0
                    )
                    text = resp.choices[0].text
                    af = self._parse_answerformat_from_content(text)
                    return (af.value, af.reasoning)
                except Exception as e2:
                    return (50, f"API Error (completions): {str(e2)}")
            # Not the Together “use prompt” case → retry chat without json mode
            try:
                resp = self.client.chat.completions.create(
                    model=self.oa_model,
                    messages=self.history,
                    temperature=0
                )
                choice = resp.choices[0]
                af = self._parse_answerformat_from_content(choice.message.content)
                return (af.value, af.reasoning)
            except Exception as e2:
                return (50, f"API Error: {str(e2)}")
        except Exception as e:
            return (50, f"API Error: {str(e)}")

    async def ask_async(self, prompt: str) -> Tuple[int, str]:
        self._ensure_system_json_message()
        self.history.append({"role": "user", "content": prompt})

        # ---------- Vertex ----------
        if getattr(self, "provider", "") == "vertex":
            try:
                resp = await self.client.models.generate_content_async(
                    model=self.vertex_model,
                    contents=self._history_to_vertex_contents(),
                    config=genai_types.GenerateContentConfig(
                        system_instruction=self._vertex_system_instruction(),
                        response_mime_type="application/json",
                        response_schema=self._vertex_schema_answerformat(),
                        temperature=0,
                    ),
                )
                try:
                    part = resp.candidates[0].content.parts[0]
                    data = part.as_dict if hasattr(part, "as_dict") else json.loads(getattr(part, "text", resp.text))
                except Exception:
                    data = json.loads(resp.text)
                return (int(data["value"]), str(data["reasoning"]))
            except Exception as e:
                return (50, f"API Error (Vertex): {str(e)}")

        # ---------- OpenAI-compatible ----------
        if self._use_completions_first():
            try:
                prompt_all = self._history_to_prompt()
                resp = await self.async_client.completions.create(
                    model=self.oa_model,
                    prompt=prompt_all,
                    max_tokens=1024,
                    temperature=0
                )
                text = resp.choices[0].text
                af = self._parse_answerformat_from_content(text)
                return (af.value, af.reasoning)
            except Exception as e:
                return (50, f"API Error (completions): {str(e)}")

        try:
            resp = await self.async_client.chat.completions.create(
                model=self.oa_model,
                messages=self.history,
                response_format={"type": "json_object"},
                temperature=0
            )
            choice = resp.choices[0]
            parsed = getattr(choice.message, "parsed", None)
            if parsed is not None:
                try:
                    af = parsed if isinstance(parsed, AnswerFormat) else AnswerFormat.model_validate(parsed)
                except Exception:
                    af = self._parse_answerformat_from_content(choice.message.content)
            else:
                af = self._parse_answerformat_from_content(choice.message.content)
            return (af.value, af.reasoning)

        except BadRequestError as e:
            if self._together_needs_completions(e):
                prompt_all = self._history_to_prompt()
                resp = await self.async_client.completions.create(
                    model=self.oa_model,
                    prompt=prompt_all,
                    max_tokens=1024,
                    temperature=0
                )
                text = resp.choices[0].text
                af = self._parse_answerformat_from_content(text)
                return (af.value, af.reasoning)
            # Not the Together “use prompt” case → retry chat without json mode
            resp = await self.async_client.chat.completions.create(
                model=self.oa_model,
                messages=self.history,
                temperature=0
            )
            choice = resp.choices[0]
            af = self._parse_answerformat_from_content(choice.message.content)
            return (af.value, af.reasoning)
