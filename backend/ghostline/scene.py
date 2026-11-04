"""Scene understanding: keyframes -> SceneDescription (SPEC §7.2).

Backends: `mock` (templates, for synthetic/dev), `meta` (Muse Spark via Meta Model API),
`baseten` (Muse Spark Open via Baseten, OpenAI-compatible). Real calls are cached.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

from pydantic import ValidationError

from .clients.cache import cached_call, file_sha256
from .config import get_settings
from .schemas import ReferenceObject, SceneDescription

KNOWN_SIZES = """Known reference sizes (pick one the user can actually place in the plane of motion):
- US quarter: diameter 0.02426 m
- US penny: diameter 0.01905 m
- Credit card: 0.08560 x 0.05398 m (use long_side 0.0856)
- A4 paper: 0.297 x 0.210 m; US Letter: 0.2794 x 0.2159 m
- Any coloured sticker/disk the user states the size of"""

SYSTEM_PROMPT = f"""You are the vision module of Ghostline, a measurement instrument.
Given keyframes from a phone video of a simple physical experiment, return ONLY a JSON object:
{{"system_type": "pendulum"|"falling_drag"|"spring_mass"|"unknown",
  "tracked_concepts": [short noun phrases for the moving object, e.g. "orange pendulum bob"],
  "reference_object": {{"concept": noun phrase for segmentation, "known_size_m": float, "dimension": "diameter"|"width"|"height"|"long_side"}},
  "anchor_hint": string or null (e.g. "pivot at top center"),
  "gravity_direction_hint": string or null (e.g. "down"),
  "notes": short string, "confidence": 0..1}}
Concepts must be SHORT (2-3 words: colour + simple noun, e.g. "orange ball", "green circle"); the
segmentation model matches short concrete phrases best. If the user states the reference object's size,
use exactly that size.
{KNOWN_SIZES}"""

TEMPLATES = {
    "pendulum": SceneDescription(
        system_type="pendulum", tracked_concepts=["orange pendulum bob"],
        reference_object=ReferenceObject(concept="green reference disk (US quarter)", known_size_m=0.02426, dimension="diameter"),
        anchor_hint="pivot at top center", gravity_direction_hint="down",
        notes="Mock scene backend (template). Set SCENE_BACKEND=meta or baseten for Muse Spark.", confidence=0.9),
    "falling_drag": SceneDescription(
        system_type="falling_drag", tracked_concepts=["orange coffee filter"],
        reference_object=ReferenceObject(concept="green reference card (credit card)", known_size_m=0.0856, dimension="long_side"),
        anchor_hint=None, gravity_direction_hint="down",
        notes="Mock scene backend (template). Set SCENE_BACKEND=meta or baseten for Muse Spark.", confidence=0.9),
}


def _img_part(p: Path) -> dict:
    b64 = base64.b64encode(p.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def _parse(text: str) -> SceneDescription:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    start, end = text.find("{"), text.rfind("}")
    return SceneDescription.model_validate(json.loads(text[start:end + 1]))


def _openai_scene(base_url: str, api_key: str, model: str, keyframes: list[Path], hint: str | None,
                  user_note: str = "") -> SceneDescription:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    user_text = ("Describe this experiment." + (f" The user says it is a {hint} experiment." if hint else "")
                 + (f" User's note: {user_note}" if user_note else ""))
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "text", "text": user_text}] + [_img_part(k) for k in keyframes]}]
    hashes = [file_sha256(k) for k in keyframes]
    err = None
    for attempt in range(3):
        msgs = messages if err is None else messages + [
            {"role": "user", "content": f"Your previous JSON failed validation: {err}. Return corrected JSON only."}]
        req = {"system": SYSTEM_PROMPT, "user": user_text, "validation_error": err}

        def call():
            r = client.chat.completions.create(model=model, messages=msgs, temperature=0,
                                               response_format={"type": "json_object"})
            return r.choices[0].message.content

        text = cached_call(model, req, call, hashes)
        try:
            return _parse(text)
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            err = str(e)[:500]
    raise RuntimeError(f"scene model returned invalid JSON 3 times: {err}")


def describe_scene(keyframes: list[Path], system_hint: str | None = None, backend: str | None = None,
                   user_note: str = "") -> SceneDescription:
    s = get_settings()
    backend = backend or s.scene_backend
    if backend == "meta":
        return _openai_scene(s.meta_base_url, s.model_api_key, s.vision_model, keyframes, system_hint, user_note)
    if backend == "baseten":
        try:
            return _openai_scene(s.baseten_base_url, s.baseten_api_key, s.baseten_model, keyframes, system_hint, user_note)
        except Exception:
            if s.model_api_key:  # fallback to Meta API
                return _openai_scene(s.meta_base_url, s.model_api_key, s.vision_model, keyframes, system_hint, user_note)
            raise
    tpl = TEMPLATES.get(system_hint or "pendulum", TEMPLATES["pendulum"])
    return tpl.model_copy(deep=True)


def scene_backend_label(backend: str | None = None) -> str:
    s = get_settings()
    b = backend or s.scene_backend
    return {"meta": f"Muse Spark ({s.vision_model}) via Meta Model API",
            "baseten": f"Muse Spark Open ({s.baseten_model}) via Baseten",
            }.get(b, "template (mock)")
