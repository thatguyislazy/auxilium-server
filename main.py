import os
import json
import re
import time
import base64
import tempfile
import subprocess
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

import anthropic
from google import genai
from google.genai import types as gemini_types

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

app = FastAPI(title="Auxilium ASL Recognition Server")
NUM_KEYFRAMES = 12

# =============================================================
# PROMPTS
# FIX: Use single-line JSON format in prompt so the response
# is always parseable. Multiline JSON in the example caused
# the regex to fail and return UNKNOWN.
# =============================================================

ASL_PROMPT = """Role: Senior ASL Interpreter & Computer Vision Specialist.
Task: Identify the SINGLE ASL sign from {num_frames} chronological keyframes (Frame 1 = start, Frame {num_frames} = end).

PRE-ANALYSIS LOGIC:
1. Track Landmarks: Trace the dominant hand path from Frame 1 to Frame {num_frames}.
2. Spatial Anchors: Is the hand at the Chin (AGE), Nose (BORED), or Chest (BIRTHDAY)?
3. Motion Vector:
   - Linear downward from chin: AGE
   - X-axis zigzag on flat palm: ART
   - Z-axis forward arc past base hand: AFTER
   - Compound (Mouth then Morning arc): BREAKFAST
4. Hand Relationship: Is one hand a base/canvas while the other moves (ART/AFTER)?

IMPORTANT: Respond with ONLY a single-line JSON object. No markdown, no backticks, no newlines inside the JSON:
{{"prediction": "UPPERCASE_WORD", "confidence": 0.95, "explanation": "Spatial Context: [location]. Movement: [axis/path]. Distinguisher: This matches [WORD] and not [SIMILAR SIGN] because [reason].", "top3": [{{"label": "WORD1", "confidence": 0.90}}, {{"label": "WORD2", "confidence": 0.07}}, {{"label": "WORD3", "confidence": 0.03}}]}}"""

ASL_PROMPT_VIDEO = """Role: Senior ASL Interpreter & Computer Vision Specialist.
Task: Identify the SINGLE ASL sign in this video.

PRE-ANALYSIS LOGIC:
1. Track Landmarks: Trace the dominant hand path from start to end.
2. Spatial Anchors: Is the hand at the Chin (AGE), Nose (BORED), or Chest (BIRTHDAY)?
3. Motion Vector:
   - Linear downward from chin: AGE
   - X-axis zigzag on flat palm: ART
   - Z-axis forward arc past base hand: AFTER
   - Compound (Mouth then Morning arc): BREAKFAST
4. Hand Relationship: Is one hand a base/canvas while the other moves?

IMPORTANT: Respond with ONLY a single-line JSON object. No markdown, no backticks, no newlines inside the JSON:
{"prediction": "UPPERCASE_WORD", "confidence": 0.95, "explanation": "Spatial Context: [location]. Movement: [axis/path]. Distinguisher: This matches [WORD] and not [SIMILAR SIGN] because [reason].", "top3": [{"label": "WORD1", "confidence": 0.90}, {"label": "WORD2", "confidence": 0.07}, {"label": "WORD3", "confidence": 0.03}]}"""


# =============================================================
# ROUTES
# =============================================================

@app.get("/")
async def root():
    providers = []
    if claude_client: providers.append("claude")
    if gemini_client: providers.append("gemini")
    return {"status": "running", "providers": providers, "version": "2.2"}

@app.get("/health")
async def health():
    return {"status": "ok", "providers": {"claude": bool(claude_client), "gemini": bool(gemini_client)}}

@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    try:
        video_bytes = await video.read()
        print(f"\n{'='*50}")
        print(f"=== RECEIVED VIDEO: {len(video_bytes)} bytes ===")
        result = analyze_asl_video(video_bytes)
        # FIX: Always return 200 with prediction, even on partial errors
        # Only return 500 if we have zero result at all
        if result is None:
            return JSONResponse(status_code=500, content={"error": "No result"})
        return JSONResponse(content=result)
    except Exception as e:
        print(f"!!! ERROR: {e}")
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# =============================================================
# KEYFRAME EXTRACTION
# =============================================================

def extract_keyframes(video_path: str, num_frames: int = NUM_KEYFRAMES) -> list:
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 2:
            cap.release()
            return []

        indices = set(int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames))
        temp_dir = tempfile.mkdtemp()
        keyframes = []
        count = 0

        while True:
            success, frame = cap.read()
            if not success:
                break
            if count in indices:
                path = os.path.join(temp_dir, f"frame_{len(keyframes):02d}.jpg")
                cv2.imwrite(path, frame)
                keyframes.append(path)
            count += 1
            if len(keyframes) == num_frames:
                break

        cap.release()
        print(f">>> Extracted {len(keyframes)} keyframes (OpenCV)")
        return keyframes
    except Exception as e:
        print(f"!!! OpenCV error: {e}, trying ffmpeg...")
        return extract_keyframes_ffmpeg(video_path, num_frames)


def extract_keyframes_ffmpeg(video_path: str, num_frames: int = NUM_KEYFRAMES) -> list:
    temp_dir = tempfile.mkdtemp()
    try:
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-q:v", "2", f"{temp_dir}/all_%04d.jpg", "-y"],
            capture_output=True, text=True, timeout=30
        )
        all_frames = sorted(Path(temp_dir).glob("all_*.jpg"))
        if all_frames:
            step = max(1, len(all_frames) // num_frames)
            selected = all_frames[::step][:num_frames]
            print(f">>> Extracted {len(selected)} keyframes (ffmpeg)")
            return [str(f) for f in selected]
    except Exception as e:
        print(f"!!! ffmpeg error: {e}")
    return []


# =============================================================
# MAIN PIPELINE
# =============================================================

def analyze_asl_video(video_bytes: bytes) -> dict:
    keyframe_paths = []
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        temp_path = f.name

    try:
        keyframe_paths = extract_keyframes(temp_path)
        if len(keyframe_paths) < 2:
            return {"prediction": "UNKNOWN", "confidence": 0.0,
                    "explanation": "Could not extract frames from video",
                    "top3": [], "error": "Frame extraction failed"}

        print(f">>> {len(keyframe_paths)} keyframes ready")

        # 1. Claude (primary)
        if claude_client:
            print(">>> Trying Claude...")
            result = analyze_with_claude(keyframe_paths, len(video_bytes))
            if result and "prediction" in result:
                return result
            print(f">>> Claude failed: {result}")

        # 2. Gemini keyframes
        if gemini_client:
            print(">>> Trying Gemini keyframes...")
            result = analyze_with_gemini_keyframes(keyframe_paths, len(video_bytes))
            if result and "prediction" in result:
                return result
            print(f">>> Gemini keyframes failed: {result}")

            # 3. Gemini video upload (last resort)
            print(">>> Trying Gemini video upload...")
            result = analyze_with_gemini_video(temp_path, len(video_bytes))
            if result and "prediction" in result:
                return result
            print(f">>> Gemini video failed: {result}")

        return {"prediction": "UNKNOWN", "confidence": 0.0,
                "explanation": "All AI providers failed — check API keys and quota.",
                "top3": []}

    finally:
        try: os.unlink(temp_path)
        except: pass
        for p in keyframe_paths:
            try: os.unlink(p)
            except: pass
        if keyframe_paths:
            try: os.rmdir(os.path.dirname(keyframe_paths[0]))
            except: pass


# =============================================================
# CLAUDE
# =============================================================

def analyze_with_claude(frame_paths: list, video_size: int) -> dict:
    try:
        prompt_text = ASL_PROMPT.format(num_frames=len(frame_paths))
        content = []
        for i, path in enumerate(frame_paths):
            with open(path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({"type": "text", "text": f"Frame {i+1} of {len(frame_paths)}"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}})
        content.append({"type": "text", "text": prompt_text})

        models = ["claude-3-5-sonnet-20241022", "claude-haiku-3-5-20241022"]
        for model in models:
            try:
                print(f">>> Claude: trying {model}...")
                resp = claude_client.messages.create(
                    model=model, max_tokens=1024,
                    messages=[{"role": "user", "content": content}]
                )
                print(f">>> Claude: success with {model}")
                return parse_response(resp.content[0].text, f"claude/{model}", video_size, "claude-keyframes")
            except anthropic.RateLimitError:
                print(f">>> Claude: rate limited on {model}")
                time.sleep(3)
            except Exception as e:
                print(f">>> Claude: {model} error: {e}")

        return {"error": "Claude: all models failed"}
    except Exception as e:
        return {"error": f"Claude error: {e}"}


# =============================================================
# GEMINI KEYFRAMES
# =============================================================

def analyze_with_gemini_keyframes(frame_paths: list, video_size: int) -> dict:
    try:
        prompt_text = ASL_PROMPT.format(num_frames=len(frame_paths))
        parts = [prompt_text]
        for i, path in enumerate(frame_paths):
            with open(path, "rb") as f:
                img_bytes = f.read()
            parts.append(f"Frame {i+1} of {len(frame_paths)}")
            parts.append(gemini_types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))

        models = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash-lite",
        ]
        for model in models:
            try:
                print(f">>> Gemini: trying {model}...")
                resp = gemini_client.models.generate_content(model=model, contents=parts)
                print(f">>> Gemini: success with {model}")
                return parse_response(resp.text, f"gemini/{model}", video_size, "gemini-keyframes")
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    print(f">>> Gemini: rate limited on {model}, waiting 5s...")
                    time.sleep(5)
                else:
                    print(f">>> Gemini: {model} error: {err}")

        return {"error": "Gemini keyframes: all models failed"}
    except Exception as e:
        return {"error": f"Gemini keyframes error: {e}"}


# =============================================================
# GEMINI VIDEO (last resort)
# =============================================================

def analyze_with_gemini_video(video_path: str, video_size: int) -> dict:
    try:
        print(">>> Gemini: uploading raw video...")
        uploaded = gemini_client.files.upload(file=video_path)
        waited = 0
        while uploaded.state.name == "PROCESSING":
            time.sleep(2); waited += 2
            if waited > 60:
                return {"error": "Video processing timed out"}
            uploaded = gemini_client.files.get(name=uploaded.name)

        if uploaded.state.name == "FAILED":
            return {"error": "Gemini failed to process video"}

        for model in ["gemini-2.5-flash", "gemini-2.0-flash"]:
            try:
                resp = gemini_client.models.generate_content(
                    model=model, contents=[uploaded, ASL_PROMPT_VIDEO]
                )
                try: gemini_client.files.delete(name=uploaded.name)
                except: pass
                return parse_response(resp.text, f"gemini/{model}", video_size, "gemini-video")
            except Exception as e:
                print(f">>> Gemini video {model} error: {e}")
                time.sleep(3)

        return {"error": "Gemini video: all models failed"}
    except Exception as e:
        return {"error": f"Gemini video error: {e}"}


# =============================================================
# RESPONSE PARSER
# FIX: More robust parsing — handles multiline JSON, extra text,
# markdown fences, and partial responses gracefully.
# Never returns "UNKNOWN" when AI gave a real answer.
# =============================================================

def parse_response(result_text: str, used_model: str, video_size: int, method: str) -> dict:
    print(f">>> Raw response: {result_text[:300]}")

    # Step 1: Strip markdown fences
    cleaned = re.sub(r'```(?:json)?', '', result_text).strip()

    # Step 2: Try direct JSON parse first
    result = None
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Step 3: Try extracting JSON object (handles extra text around it)
    if result is None:
        # Remove newlines inside JSON to fix multiline format issue
        cleaned_single = re.sub(r'\s+', ' ', cleaned)
        match = re.search(r'\{.*\}', cleaned_single, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    # Step 4: If still no valid JSON, try to extract prediction from plain text
    if result is None:
        print(f"!!! Could not parse JSON from: {result_text[:200]}")
        # Try to find a word that looks like a sign name in the text
        word_match = re.search(r'"prediction"\s*:\s*"([A-Z]+)"', result_text)
        if word_match:
            result = {
                "prediction": word_match.group(1),
                "confidence": 0.5,
                "explanation": result_text[:200],
                "top3": []
            }
        else:
            return {"error": f"Could not parse AI response: {result_text[:100]}"}

    prediction = str(result.get("prediction", "UNKNOWN")).upper()
    confidence = min(1.0, max(0.0, float(result.get("confidence", 0.0))))
    top3 = result.get("top3", [])
    explanation = result.get("explanation", "")

    if not top3:
        top3 = [{"label": prediction, "confidence": confidence}]

    print(f">>> RESULT: {prediction} ({confidence*100:.1f}%) via {method} [{used_model}]")

    return {
        "prediction": prediction,
        "confidence": confidence,
        "top3": top3,
        "explanation": explanation,
        "debug": {"model": used_model, "method": method, "video_size": video_size}
    }
