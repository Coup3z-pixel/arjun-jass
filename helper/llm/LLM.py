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
        "You are a function that returns JSON only. "
        "Output must be a single JSON object with keys: "
        '{"reasoning": string, "value": integer}. '
        "Do not include any text outside the JSON. Return JSON only."
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
        m.startswith("projects/") or          # tuned endpoint passed raw
        m.startswith("publishers/") or        # rarely used
        m.startswith("gemini-") or "gemini" in m  # base models
    )

def _is_together_model(model: str) -> bool:
    if not model:
        return False
    return model.strip().startswith("together:")

def _normalize_model_for_together(model: str) -> str:
    return model[len("together:"):] if model.startswith("together:") else model

def _normalize_model_for_openai(model: str) -> str:
    return model[len("openai:"):] if model.startswith("openai:") else model

# =========================
# LLM Router
# =========================

class LLM:
    """
    Unified LLM helper that routes automatically:
      - Vertex AI (Gemini & tuned endpoints) via google-genai when model looks like Vertex
      - OpenAI API when model looks like OpenAI
      - Together/OpenRouter via OpenAI-compatible APIs otherwise
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
                api_key="2d73106bc4fa51e9fb2af9b7185f9a87dc023d524030817acfbb731265e01808",
            )
            self.async_client = AsyncOpenAIClient(
                base_url="https://api.together.xyz/v1",
                api_key="2d73106bc4fa51e9fb2af9b7185f9a87dc023d524030817acfbb731265e01808",
            )

        elif _is_vertex_model(self.model):
            self.provider = "vertex"
            # Strip optional "vertex:" prefix; pass either a base model or an endpoint resource name.
            self.vertex_model = self.model[len("vertex:"):] if self.model.startswith("vertex:") else self.model

            # Prefer env vars; override here if needed
            self.client = genai.Client(
                vertexai=True,
                project=os.getenv("GOOGLE_CLOUD_PROJECT") or "buoyant-ground-472514-s0",
                location=os.getenv("GOOGLE_CLOUD_LOCATION") or "us-central1",
            )
            # google-genai uses same client for async

        else:
            # OpenRouter fallback (OpenAI-compatible)
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

    # =========================
    # Session state
    # =========================

    def restart_model(self) -> None:
        self.history = []

    def _ensure_system_json_message(self):
        # We still record a system message in history for consistency,
        # but Vertex will read it via system_instruction instead of chat messages.
        if not self.history or self.history[0].get("role") != "system":
            self.history.insert(0, JSON_SYSTEM_MSG)

    # =========================
    # Parsing helpers (for OpenAI-style responses)
    # =========================

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

        # Attempt repairs
        for obj in reversed(candidates):
            data = try_parse(obj)
            if not isinstance(data, dict):
                continue
            reasoning = data.get("reasoning")
            value = data.get("value")
            if value is None:
                m = re.search(r'"value"\s*:\s*(-?\d+)', text)
                if m:
                    value = int(m.group(1))
                else:
                    m2 = re.search(r'(-?\d+)', text)
                    if m2:
                        value = int(m2.group(1))
            if reasoning is None:
                reasoning = text[:5000]
            if value is not None and reasoning is not None:
                return AnswerFormat.model_validate({"reasoning": str(reasoning), "value": int(value)})

        m3 = re.search(r'(-?\d+)', text)
        if m3:
            return AnswerFormat.model_validate({"reasoning": text[:5000], "value": int(m3.group(1))})

        raise ValueError(f"Expected JSON with keys 'reasoning' and 'value'. Got: {text[:200]}...")

    # =========================
    # Vertex helpers
    # =========================

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
        # Use our system prompt from history (first message), else a default.
        if self.history and self.history[0].get("role") == "system":
            return str(self.history[0].get("content") or "")
        return (
            "You are a function that returns JSON only. "
            'Output must be: {"reasoning": string, "value": integer}. '
            "Return JSON only."
        )

    def _history_to_vertex_contents(self) -> List[Dict[str, Any]]:
        """
        Convert chat.history (system/user/assistant) into Vertex 'contents'.
        Vertex uses: [{"role": "user"|"model", "parts": [{"text": "..."}]}]
        We'll skip the system message since we pass it as system_instruction.
        """
        contents: List[Dict[str, Any]] = []
        for msg in self.history:
            role = msg.get("role")
            if role == "system":
                continue
            text = msg.get("content", "")
            if text is None:
                continue
            # Map OpenAI "assistant" → Vertex "model"
            vrole = "model" if role == "assistant" else "user"
            contents.append({"role": vrole, "parts": [{"text": str(text)}]})
        return contents

    # =========================
    # Public API
    # =========================

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
                # When response_mime_type is JSON, SDK provides structured parts
                try:
                    part = resp.candidates[0].content.parts[0]
                    data = part.as_dict if hasattr(part, "as_dict") else json.loads(getattr(part, "text", resp.text))
                except Exception:
                    data = json.loads(resp.text)
                return (int(data["value"]), str(data["reasoning"]))
            except Exception as e:
                return (50, f"API Error (Vertex): {str(e)}")

        # ---------- OpenAI-compatible (OpenAI / Together / OpenRouter) ----------
        try:
            resp = self.client.chat.completions.create(
                model=self.oa_model,
                messages=self.history,
                response_format={"type": "json_object"},
                temperature=0
            )
        except BadRequestError as e:
            # Retry without JSON mode
            try:
                resp = self.client.chat.completions.create(
                    model=self.oa_model,
                    messages=self.history,
                    temperature=0
                )
            except Exception as e2:
                return (50, f"API Error: {str(e2)}")
        except Exception as e:
            return (50, f"API Error: {str(e)}")

        # Parse
        try:
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
        except Exception as e:
            return (50, f"Parse Error: {str(e)}")

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

        # ---------- OpenAI-compatible (OpenAI / Together / OpenRouter) ----------
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
        """
        Like ask(), but uses a custom Pydantic model as the enforced JSON schema.
        For Vertex, we map the schema to google-genai Schema and force JSON output.
        For OpenAI-compatible, we send a dynamic system message + json_object mode.
        """
        # Build dynamic system message describing required keys/types
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

        # Replace/insert system
        if self.history and self.history[0].get("role") == "system":
            self.history[0] = {"role": "system", "content": system_content}
        else:
            self.history.insert(0, {"role": "system", "content": system_content})

        self.history.append({"role": "user", "content": prompt})

        # ---------- Vertex path ----------
        if getattr(self, "provider", "") == "vertex":
            # Convert Pydantic schema to google-genai Schema
            def to_vertex_schema(p_schema: dict) -> genai_types.Schema:
                # Minimal mapping for objects with primitive props
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
                            # fallback to string for complex types (arrays/objects) unless extended
                            props[k] = genai_types.Schema(type=genai_types.Type.STRING)
                    required = p_schema.get("required", [])
                    return genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties=props,
                        required=required,
                    )
                # fallback
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

        # ---------- OpenAI-compatible path ----------
        max_retries = 3
        for attempt in range(max_retries):
            try:
                try:
                    resp = self.client.chat.completions.create(
                        model=self.oa_model,
                        messages=self.history,
                        response_format={"type": "json_object"},
                        temperature=0,
                        max_tokens=8192  # Increased token limit
                    )
                except BadRequestError:
                    resp = self.client.chat.completions.create(
                        model=self.oa_model,
                        messages=self.history,
                        temperature=0,
                        max_tokens=8192  # Increased token limit
                    )
                
                content = resp.choices[0].message.content
                if isinstance(content, list):
                    content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
                
                text = str(content).strip()
                if text.startswith("```"):
                    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
                    text = "\n".join(lines).strip()
                
                # Try to validate and fix JSON if needed
                try:
                    return answer_format.model_validate_json(text)
                except json.JSONDecodeError as json_err:
                    print(f"[DEBUG] JSON decode error on attempt {attempt + 1}: {json_err}")
                    print(f"[DEBUG] Raw response length: {len(text)}")
                    print(f"[DEBUG] Raw response preview: {text[:200]}...")
                    print(f"[DEBUG] Raw response ending: ...{text[-200:]}")
                    
                    # Try to fix the JSON
                    fixed_text = self._fix_malformed_json(text, json_err)
                    if fixed_text != text:
                        print(f"[DEBUG] Attempting repair with fixed JSON")
                        try:
                            return answer_format.model_validate_json(fixed_text)
                        except:
                            pass
                    
                    # If it's the last attempt, create fallback
                    if attempt == max_retries - 1:
                        print(f"[DEBUG] Creating fallback response after {max_retries} attempts")
                        return self._create_fallback_response(answer_format, text)
                    continue
                        
                except Exception as e:
                    print(f"[DEBUG] Pydantic validation error on attempt {attempt + 1}: {e}")
                    # This is likely a pydantic validation error, not JSON parsing
                    if attempt == max_retries - 1:
                        print(f"[DEBUG] Creating fallback response after {max_retries} attempts")
                        return self._create_fallback_response(answer_format, text)
                    continue
                        
            except Exception as e:
                print(f"[DEBUG] General error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    print(f"[DEBUG] All attempts failed, creating emergency fallback")
                    return self._create_emergency_fallback(answer_format)
                continue
        
        raise RuntimeError(f"Failed to get valid response after {max_retries} attempts")
    
    def _fix_malformed_json(self, text: str, json_error: json.JSONDecodeError) -> str:
        """Attempt to fix malformed JSON based on the error type."""
        try:
            # Handle EOF while parsing string
            if "EOF while parsing a string" in str(json_error):
                print(f"[DEBUG] Attempting to fix EOF parsing error at position {json_error.pos}")
                
                # Find the position of the error and try to close the JSON properly
                error_pos = json_error.pos
                
                # Look for the last complete field before the error
                text_before_error = text[:error_pos]
                
                # Try to find a reasonable place to truncate and close
                last_quote = text_before_error.rfind('"')
                if last_quote > 0:
                    # Check if this quote is part of a field name or value
                    before_quote = text_before_error[:last_quote].rstrip()
                    if before_quote.endswith(':'):
                        # This is a field value, close it properly
                        fixed = text_before_error[:last_quote] + '""}'
                    else:
                        # This might be the end of a field name, close the object
                        fixed = text_before_error[:last_quote + 1] + '}'
                    
                    # Validate the fix
                    try:
                        json.loads(fixed)
                        print(f"[DEBUG] Successfully fixed JSON by closing string")
                        return fixed
                    except Exception as fix_err:
                        print(f"[DEBUG] Fix attempt failed: {fix_err}")
                
                # Try to extract reasoning and create minimal valid JSON
                import re
                reasoning_match = re.search(r'"reasoning":\s*"([^"]*)', text)
                if reasoning_match:
                    reasoning = reasoning_match.group(1)
                    # Clean up the reasoning text
                    reasoning = reasoning.replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
                    fallback_json = f'{{"reasoning": "{reasoning}", "value": 1}}'
                    try:
                        json.loads(fallback_json)
                        print(f"[DEBUG] Created fallback JSON with extracted reasoning")
                        return fallback_json
                    except Exception as fallback_err:
                        print(f"[DEBUG] Fallback JSON failed: {fallback_err}")
            
            # Handle other JSON errors by creating minimal valid response
            print(f"[DEBUG] Creating minimal fallback JSON")
            return '{"reasoning": "JSON parsing error - using fallback", "value": 1}'
            
        except Exception as e:
            print(f"[DEBUG] Error in JSON fix attempt: {e}")
            return '{"reasoning": "JSON repair failed", "value": 1}'
    
    def _create_fallback_response(self, answer_format: Type[BaseModel], raw_text: str):
        """Create a fallback response when JSON parsing fails."""
        try:
            # Extract any reasoning text from the raw response
            import re
            reasoning_text = "Response parsing failed"
            
            # Try to extract reasoning from various patterns
            patterns = [
                r'"reasoning":\s*"([^"]*)',
                r'reasoning[:\s]*([^,}\n]*)',
                r'because\s+([^.!?\n]{10,100})',
                r'I\s+(?:choose|pick|select)\s+([^.!?\n]{10,100})'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, raw_text, re.IGNORECASE)
                if match:
                    reasoning_text = match.group(1).strip()[:200]  # Limit length
                    break
            
            # Create response based on expected format
            if hasattr(answer_format, '__annotations__'):
                fallback_data = {}
                for field_name, field_type in answer_format.__annotations__.items():
                    if field_name == 'reasoning' or 'reason' in field_name.lower():
                        fallback_data[field_name] = reasoning_text
                    elif 'percent' in field_name.lower():
                        fallback_data[field_name] = 50  # Default percentage
                    elif field_type in (int, 'int') or 'value' in field_name.lower():
                        fallback_data[field_name] = 1
                    elif field_type in (float, 'float'):
                        fallback_data[field_name] = 1.0
                    elif field_type in (bool, 'bool'):
                        fallback_data[field_name] = True
                    else:
                        fallback_data[field_name] = reasoning_text
                
                return answer_format.model_validate(fallback_data)
            
            # Generic fallback
            return answer_format.model_validate({"reasoning": reasoning_text, "value": 1})
            
        except Exception as e:
            print(f"[DEBUG] Error creating fallback: {e}")
            return self._create_emergency_fallback(answer_format)
    
    def _create_emergency_fallback(self, answer_format: Type[BaseModel]):
        """Create minimal emergency fallback when everything else fails."""
        try:
            # Try to create with minimal required fields
            fallback_data = {
                "reasoning": "Emergency fallback - JSON parsing failed",
                "value": 1
            }
            
            # Add other common fields that might be expected
            if hasattr(answer_format, '__annotations__'):
                for field_name, field_type in answer_format.__annotations__.items():
                    if field_name not in fallback_data:
                        if 'percent' in field_name.lower():
                            fallback_data[field_name] = 50  # Default percentage
                        elif field_type in (int, 'int'):
                            fallback_data[field_name] = 1
                        elif field_type in (float, 'float'):
                            fallback_data[field_name] = 1.0
                        elif field_type in (bool, 'bool'):
                            fallback_data[field_name] = True
                        elif field_type in (str, 'str'):
                            fallback_data[field_name] = "Default response"
            
            return answer_format.model_validate(fallback_data)
        except Exception as e:
            print(f"[DEBUG] Even emergency fallback failed: {e}")
            # Absolute last resort - try with just the most basic fields
            try:
                return answer_format.model_validate({
                    "reasoning": "System error - using absolute fallback",
                    "value": 1,
                    "keep_percent": 50,
                    "donate_percent": 50
                })
            except:
                # Create the most basic possible response
                return answer_format.model_validate({"reasoning": "Error", "value": 1})

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

        # ---------- Vertex ----------
        if getattr(self, "provider", "") == "vertex":
            # Map schema to google-genai
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

        # ---------- OpenAI-compatible ----------
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
