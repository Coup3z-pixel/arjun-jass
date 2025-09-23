import google.genai as genai
from google.genai import types as genai_types
import os, json

client = genai.Client(
    vertexai=True,
    project="buoyant-ground-472514-s0",
    location="us-central1",
)

MODEL = "projects/buoyant-ground-472514-s0/locations/us-central1/models/4604485316777082880"

resp = client.models.generate_content(
    model=MODEL,
    contents=[{"role":"user","parts":[{"text":"Return only JSON: {\"reasoning\":\"ok\",\"value\":7}"}]}],
    config=genai_types.GenerateContentConfig(
        system_instruction="You are a function that returns JSON only with keys reasoning (string) and value (integer).",
        response_mime_type="application/json",
        response_schema=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "reasoning": genai_types.Schema(type=genai_types.Type.STRING),
                "value": genai_types.Schema(type=genai_types.Type.INTEGER),
            },
            required=["reasoning","value"],
        ),
        temperature=0,
    ),
)

# Try to read structured part first; fall back to text
try:
    part = resp.candidates[0].content.parts[0]
    data = part.as_dict if hasattr(part, "as_dict") else json.loads(getattr(part, "text", resp.text))
except Exception:
    data = json.loads(resp.text)

print(data)  # should be a dict with {"reasoning": "...", "value": 7}
