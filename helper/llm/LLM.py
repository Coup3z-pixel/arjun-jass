from typing import Type, Tuple, Optional, List, Dict, Any
from pydantic import BaseModel
from google import genai
from google.genai import types as genai_types
from openai import OpenAI as OpenAIClient, AsyncOpenAI as AsyncOpenAIClient, BadRequestError
import json, os, re

class AnswerFormat(BaseModel):
    reasoning: str
    value: int

JSON_SYSTEM_MSG = {
    "role": "system",
    "content": (
        "You are a function that returns JSON only. "
        "Output must be a single JSON object with keys: "
        '{"reasoning": string, "value": integer}. '
        "Do not include any text outside the JSON. Return JSON only."
    )
}

def _is_openai_model(model: str) -> bool:
    if not model: return False
    m = model.strip()
    return (
        m.startswith("openai:") or
        m.startswith("openai/") or
        m.startswith("ft:") or
        m.startswith("gpt-") or
        m.startswith("o4") or
        m in {
            "gpt-4o", "gpt-4.1", "gpt-4o-mini", "gpt-4.1-mini",
            "o4", "o4-mini", "gpt-4.1-nano", "gpt-4o-audio-preview",
            "chatgpt-4o-latest"
        }
    )

def _is_vertex_model(m: str) -> bool:
    if not m: return False
    m = m.strip()
    return (
        m.startswith("vertex:") or
        m.startswith("projects/") or
        m.startswith("publishers/") or
        m.startswith("gemini-") or
        "gemini" in m
    )

def _is_together_model(model: str) -> bool:
    if not model: return False
    return model.strip().startswith("together:")

def _normalize_model_for_together(model: str) -> str:
    return model[len("together:"):] if model.startswith("together:") else model

def _normalize_model_for_openai(model: str) -> str:
    if model.startswith("openai:"): return model[len("openai:"):]
    if model.startswith("openai/"): return model[len("openai/"):]
    return model

def _looks_like_chat_model(m: str) -> bool:
    m = m.lower()
    return any(k in m for k in ["-instruct", "/instruct", "-chat", "/chat", "sonar", "scout"])

def _chat_to_prompt(history: List[Dict[str, Any]]) -> str:
    sys = ""
    buf = []
    for msg in history:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if not content: continue
        if role == "system":
            sys = content
        elif role == "user":
            buf.append(f"User: {content}")
        elif role == "assistant":
            buf.append(f"Assistant: {content}")
    if sys:
        buf.insert(0, f"System: {sys}")
    buf.append("Assistant:")
    return "\n".join(buf)

class LLM:
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
            self.together_model = _normalize_model_for_together(self.model)
            # Disable native Together client for now - use OpenAI-compatible API
            self.together_client = None
            # Create OpenAI-compatible client directly
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
            # Normalize tuned model resource names: use tunedModels/<id> instead of models/<id>
            try:
                if self.vertex_model.startswith("projects/") and "/locations/" in self.vertex_model and "/models/" in self.vertex_model and "publishers/" not in self.vertex_model:
                    left, right = self.vertex_model.split("/models/", 1)
                    # Many tuned model IDs are numeric-like; swap segment to tunedModels
                    if right and "/" not in right:
                        self.vertex_model = f"{left}/tunedModels/{right}"
            except Exception:
                pass
            self.client = genai.Client(
                vertexai=True,
                project=os.getenv("GOOGLE_CLOUD_PROJECT") or "buoyant-ground-472514-s0",
                location=os.getenv("GOOGLE_CLOUD_LOCATION") or "us-central1",
            )

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

    def _ensure_together_oa_client(self):
        """Make sure OA-compatible Together client exists before fallback."""
        if self.client is None:
            self.client = OpenAIClient(
                base_url="https://api.together.xyz/v1",
                api_key=os.getenv("TOGETHER_API_KEY"),
            )
        if self.async_client is None:
            self.async_client = AsyncOpenAIClient(
                base_url="https://api.together.xyz/v1",
                api_key=os.getenv("TOGETHER_API_KEY"),
            )

    def restart_model(self) -> None:
        self.history = []

    def _ensure_system_json_message(self):
        if not self.history or self.history[0].get("role") != "system":
            self.history.insert(0, JSON_SYSTEM_MSG)

    def _current_system_text(self) -> str:
        """Return current system content or a concise default."""
        if self.history and self.history[0].get("role") == "system":
            return str(self.history[0].get("content") or "")
        return (
            "You are a function that returns JSON only. "
            'Output must be: {"reasoning": string, "value": integer}. '
            "Return JSON only."
        )

    def _last_user_text(self) -> str:
        """Return the most recent user message content (single turn)."""
        for msg in reversed(self.history):
            if msg.get("role") == "user":
                return str(msg.get("content") or "")
        return ""

    def _coerce_to_answer_format(self, answer_format: Type[BaseModel], reasoning: str, value: int) -> BaseModel:
        """Build a best-effort object for answer_format, filling required fields with defaults."""
        data: Dict[str, Any] = {
            "reasoning": str(reasoning),
            "value": int(value) if isinstance(value, (int, float, str)) and str(value).lstrip("-+").isdigit() else 1,
        }
        try:
            annotations = getattr(answer_format, "__annotations__", {}) or {}
            for field_name, field_type in annotations.items():
                if field_name in data:
                    continue
                lname = str(field_name).lower()
                # Heuristics for common fields
                if "percent" in lname:
                    data[field_name] = 50
                elif field_type in (int, "int"):
                    data[field_name] = 1
                elif field_type in (float, "float"):
                    data[field_name] = 1.0
                elif field_type in (bool, "bool"):
                    data[field_name] = True
                else:
                    data[field_name] = str(reasoning)
        except Exception:
            # If anything goes wrong, keep minimal fields
            pass
        return answer_format.model_validate(data)

    def _parse_answerformat_from_content(self, content) -> AnswerFormat:
        if isinstance(content, list):
            content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        text = str(content).strip()
        if text.startswith("```"):
            lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()
        try:
            return AnswerFormat.model_validate_json(text)
        except Exception:
            pass
        candidates, brace_stack, start_idx = [], [], None
        for i, ch in enumerate(text):
            if ch == '{':
                if not brace_stack: start_idx = i
                brace_stack.append('{')
            elif ch == '}':
                if brace_stack:
                    brace_stack.pop()
                    if not brace_stack and start_idx is not None:
                        candidates.append(text[start_idx:i+1]); start_idx = None
        def try_parse(obj: str) -> Optional[dict]:
            try: return json.loads(obj)
            except Exception: return None
        for obj in reversed(candidates):
            data = try_parse(obj)
            if isinstance(data, dict) and "reasoning" in data and "value" in data:
                return AnswerFormat.model_validate(data)
        m = re.search(r'(-?\d+)', text)
        if m:
            return AnswerFormat.model_validate({"reasoning": text[:5000], "value": int(m.group(1))})
        raise ValueError(f"Expected JSON with keys 'reasoning' and 'value'. Got: {text[:200]}...")

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

    # ---------------- Public API ----------------

    def ask(self, prompt) -> Tuple[int, str]:
        self._ensure_system_json_message()
        self.history.append({"role": "user", "content": prompt})

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

        if getattr(self, "provider", "") == "together":
            # Use OpenAI-compatible Together API
            try:
                if _looks_like_chat_model(self.together_model):
                    resp = self.client.chat.completions.create(
                        model=self.together_model,
                        messages=self.history,
                        response_format={"type": "json_object"},
                        temperature=0,
                        max_tokens=8192,
                    )
                    choice = resp.choices[0]
                    parsed = getattr(choice.message, "parsed", None)
                    if parsed is not None:
                        af = parsed if isinstance(parsed, AnswerFormat) else AnswerFormat.model_validate(parsed)
                    else:
                        af = self._parse_answerformat_from_content(choice.message.content)
                    return (af.value, af.reasoning)
                else:
                    # Base model → use text completions endpoint with single-turn prompt
                    sys_text = self._current_system_text()
                    user_text = self._last_user_text()
                    prompt_text = f"System: {sys_text}\nUser: {user_text}\n"
                    try:
                        resp = self.client.completions.create(
                            model=self.together_model,
                            prompt=prompt_text + "Assistant:",
                            temperature=0.5,
                            max_tokens=8192,
                        )
                        content = resp.choices[0].text
                    except Exception:
                        # Fallback: single-message chat if completions not available
                        resp = self.client.chat.completions.create(
                            model=self.together_model,
                            messages=[{"role": "system", "content": sys_text}, {"role": "user", "content": user_text}],
                            temperature=0.5,
                            max_tokens=8192,
                        )
                        content = resp.choices[0].message.content
                    af = self._parse_answerformat_from_content(content)
                    return (af.value, af.reasoning)
            except Exception as e:
                return (50, f"API Error (Together): {str(e)}")

        # OpenAI / OpenRouter
        try:
            resp = self.client.chat.completions.create(
                model=self.oa_model,
                messages=self.history,
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=8192
            )
        except BadRequestError:
            try:
                resp = self.client.chat.completions.create(
                    model=self.oa_model,
                    messages=self.history,
                    temperature=0,
                    max_tokens=8192
                )
            except Exception as e2:
                return (50, f"API Error: {str(e2)}")
        except Exception as e:
            return (50, f"API Error: {str(e)}")

        try:
            choice = resp.choices[0]
            parsed = getattr(choice.message, "parsed", None)
            if parsed is not None:
                af = parsed if isinstance(parsed, AnswerFormat) else AnswerFormat.model_validate(parsed)
            else:
                af = self._parse_answerformat_from_content(choice.message.content)
            return (af.value, af.reasoning)
        except Exception as e:
            return (50, f"Parse Error: {str(e)}")

    async def ask_async(self, prompt: str) -> Tuple[int, str]:
        self._ensure_system_json_message()
        self.history.append({"role": "user", "content": prompt})

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

        if getattr(self, "provider", "") == "together":
            # keep logic centralized to sync path (native SDK lacks proper async)
            v, r = self.ask(prompt)
            return (v, r)

        try:
            resp = await self.async_client.chat.completions.create(
                model=self.oa_model,
                messages=self.history,
                response_format={"type": "json_object"},
                temperature=0
            )
        except BadRequestError:
            resp = await self.async_client.chat.completions.create(
                model=self.oa_model,
                messages=self.history,
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

    def ask_with_custom_format(self, prompt, answer_format: Type[BaseModel]):
        schema = answer_format.model_json_schema()
        properties = schema.get("properties", {})
        field_desc = []
        for field_name, field_info in properties.items():
            field_type = field_info.get("type", "string")
            field_desc.append(f'"{field_name}": {field_type}')
        fields_str = ", ".join(field_desc)
        system_content = (
            f"You are a function that returns JSON only. "
            f"Output must be a single JSON object with keys: {{{fields_str}}}. "
            f"Do not include any text outside the JSON. Return JSON only."
        )

        if self.history and self.history[0].get("role") == "system":
            self.history[0] = {"role": "system", "content": system_content}
        else:
            self.history.insert(0, {"role": "system", "content": system_content})

        self.history.append({"role": "user", "content": prompt})

        if getattr(self, "provider", "") == "vertex":
            def to_vertex_schema(p_schema: dict) -> genai_types.Schema:
                if p_schema.get("type") == "object":
                    props = {}
                    for k, v in (p_schema.get("properties") or {}).items():
                        t = v.get("type")
                        if t == "string":
                            props[k] = genai_types.Schema(type=genai_types.Type.STRING)
                        elif t == "integer":
                            props[k] = genai_types.Schema(type=genai_types.Type.INTEGER)
                        elif t == "number":
                            props[k] = genai_types.Schema(type=genai_types.Type.NUMBER)
                        elif t == "boolean":
                            props[k] = genai_types.Schema(type=genai_types.Type.BOOLEAN)
                        else:
                            props[k] = genai_types.Schema(type=genai_types.Type.STRING)
                    required = p_schema.get("required", [])
                    return genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties=props,
                        required=required,
                    )
                return genai_types.Schema(type=genai_types.Type.STRING)

            v_schema = to_vertex_schema(schema)
            try:
                resp = self.client.models.generate_content(
                    model=self.vertex_model,
                    contents=self._history_to_vertex_contents(),
                    config=genai_types.GenerateContentConfig(
                        system_instruction=system_content,
                        response_mime_type="application/json",
                        response_schema=v_schema,
                        temperature=0,
                    ),
                )
                try:
                    part = resp.candidates[0].content.parts[0]
                    data = part.as_dict if hasattr(part, "as_dict") else json.loads(getattr(part, "text", resp.text))
                except Exception:
                    data = json.loads(resp.text)
                return answer_format.model_validate(data)
            except Exception as e:
                raise RuntimeError(f"Vertex error: {e}") from e

        if getattr(self, "provider", "") == "together":
            # Use OpenAI-compatible Together API
            try:
                if _looks_like_chat_model(self.together_model):
                    try:
                        resp = self.client.chat.completions.create(
                            model=self.together_model,
                            messages=self.history,
                            response_format={"type": "json_object"},
                            temperature=0,
                            max_tokens=8192
                        )
                    except BadRequestError:
                        resp = self.client.chat.completions.create(
                            model=self.together_model,
                            messages=self.history,
                            temperature=0,
                            max_tokens=8192
                        )
                    text = resp.choices[0].message.content
                else:
                    # Base model → use text completions endpoint with a flattened prompt
                    prompt_text = _chat_to_prompt(self.history)
                    try:
                        resp = self.client.completions.create(
                            model=self.together_model,
                            prompt=prompt_text,
                            temperature=0,
                            max_tokens=8192
                        )
                        text = resp.choices[0].text
                    except Exception:
                        # Fallback: single-message chat
                        resp = self.client.chat.completions.create(
                            model=self.together_model,
                            messages=[{"role": "user", "content": prompt_text}],
                            temperature=0,
                            max_tokens=8192
                        )
                        text = resp.choices[0].message.content
                if isinstance(text, list):
                    text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in text)
                if text.startswith("```"):
                    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
                    text = "\n".join(lines).strip()
                try:
                    return answer_format.model_validate_json(text)
                except Exception:
                    af = self._parse_answerformat_from_content(text)
                    return self._coerce_to_answer_format(answer_format, af.reasoning, af.value)
                    af = self._parse_answerformat_from_content(text)
                    return self._coerce_to_answer_format(answer_format, af.reasoning, af.value)
            except Exception as e:
                raise RuntimeError(f"Together error: {e}") from e

        # OpenAI / OpenRouter
        max_retries = 3
        for attempt in range(max_retries):
            try:
                try:
                    resp = self.client.chat.completions.create(
                        model=self.oa_model,
                        messages=self.history,
                        response_format={"type": "json_object"},
                        temperature=0,
                        max_tokens=8192
                    )
                except BadRequestError:
                    resp = self.client.chat.completions.create(
                        model=self.oa_model,
                        messages=self.history,
                        temperature=0,
                        max_tokens=8192
                    )
                text = resp.choices[0].message.content
                if isinstance(text, list):
                    text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in text)
                if str(text).startswith("```"):
                    lines = [ln for ln in str(text).splitlines() if not ln.strip().startswith("```")]
                    text = "\n".join(lines).strip()
                return answer_format.model_validate_json(text)
            except Exception as e:
                if attempt == max_retries - 1:
                    # soft parse fallback
                    af = self._parse_answerformat_from_content(str(text) if 'text' in locals() else str(e))
                    return self._coerce_to_answer_format(answer_format, af.reasoning, af.value)
                continue

    async def ask_with_custom_format_async(self, prompt: str, answer_format: Type[BaseModel]):
        schema = answer_format.model_json_schema()
        properties = schema.get("properties", {})
        field_desc = []
        for field_name, field_info in properties.items():
            field_type = field_info.get("type", "string")
            field_desc.append(f'"{field_name}": {field_type}')
        fields_str = ", ".join(field_desc)
        system_content = (
            f"You are a function that returns JSON only. "
            f"Output must be a single JSON object with keys: {{{fields_str}}}. "
            f"Do not include any text outside the JSON. Return JSON only."
        )

        if self.history and self.history[0].get("role") == "system":
            self.history[0] = {"role": "system", "content": system_content}
        else:
            self.history.insert(0, {"role": "system", "content": system_content})

        self.history.append({"role": "user", "content": prompt})

        if getattr(self, "provider", "") == "vertex":
            def to_vertex_schema(p_schema: dict) -> genai_types.Schema:
                if p_schema.get("type") == "object":
                    props = {}
                    for k, v in (p_schema.get("properties") or {}).items():
                        t = v.get("type")
                        if t == "string":
                            props[k] = genai_types.Schema(type=genai_types.Type.STRING)
                        elif t == "integer":
                            props[k] = genai_types.Schema(type=genai_types.Type.INTEGER)
                        elif t == "number":
                            props[k] = genai_types.Schema(type=genai_types.Type.NUMBER)
                        elif t == "boolean":
                            props[k] = genai_types.Schema(type=genai_types.Type.BOOLEAN)
                        else:
                            props[k] = genai_types.Schema(type=genai_types.Type.STRING)
                    required = p_schema.get("required", [])
                    return genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties=props,
                        required=required,
                    )
                return genai_types.Schema(type=genai_types.Type.STRING)

            v_schema = to_vertex_schema(schema)
            resp = await self.client.models.generate_content_async(
                model=self.vertex_model,
                contents=self._history_to_vertex_contents(),
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_content,
                    response_mime_type="application/json",
                    response_schema=v_schema,
                    temperature=0,
                ),
            )
            try:
                part = resp.candidates[0].content.parts[0]
                data = part.as_dict if hasattr(part, "as_dict") else json.loads(getattr(part, "text", resp.text))
            except Exception:
                data = json.loads(resp.text)
            return answer_format.model_validate(data)

        if getattr(self, "provider", "") == "together":
            v, r = self.ask(prompt)  # centralize logic
            return answer_format.model_validate({"reasoning": r, "value": v})

        # OpenAI/OpenRouter
        try:
            resp = await self.async_client.chat.completions.create(
                model=self.oa_model,
                messages=self.history,
                response_format={"type": "json_object"},
                temperature=0
            )
        except BadRequestError:
            resp = await self.async_client.chat.completions.create(
                model=self.oa_model,
                messages=self.history,
                temperature=0
            )
        content = resp.choices[0].message.content
        if isinstance(content, list):
            content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        text = str(content).strip()
        if text.startswith("```"):
            lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()
        return answer_format.model_validate_json(text)

    def get_model_name(self) -> str:
        return self.model
