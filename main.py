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

# --- AI Providers ---
import anthropic
from google import genai
from google.genai import types as gemini_types

# --- Config ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

app = FastAPI(title="Auxilium ASL Recognition Server")
NUM_KEYFRAMES = 8

# =============================================================
# PROMPT
# =============================================================

ASL_PROMPT = """Role: You are a Senior ASL Interpreter and Computer Vision Expert.
Task: I am showing you {num_frames} keyframes extracted in chronological order from a video of someone performing a SINGLE ASL sign. Frame 1 is the start, Frame {num_frames} is the end. Analyze carefully to identify the specific ASL sign being performed.

CRITICAL ANALYSIS STEPS:
1. Identify Anchor Points: Determine the hand's position relative to the body (e.g., Chin for AGE, Chest for BIRTHDAY, Nose for BORED).
2. Track Motion Vectors by comparing all frames:
   - Y-axis: Is there a downward tug (AGE) or an upward rise (MORNING)?
   - Z-axis: Is the hand moving toward or away from the camera (AFTER)?
   - X-axis: Is there a side-to-side zigzag or oscillation (ART)?
3. Compound Sign Check: Does the sign have two distinct parts? (e.g., EAT + MORNING = BREAKFAST).
4. Hand Relationship: Is one hand acting as a base or 'canvas' while the other moves (ART/AFTER)?

Respond ONLY with this exact JSON format (no markdown, no backticks):
{{"prediction": "WORD", "confidence": 0.95, "explanation": "Spatial Context: [location]. Movement: [axis/path]. Distinguisher: This matches [WORD] and not [SIMILAR SIGN] because of [reason].", "top3": [{{"label": "WORD1", "confidence": 0.90}}, {{"label": "WORD2", "confidence": 0.07}}, {{"label": "WORD3", "confidence": 0.03}}]}}"""

ASL_PROMPT_VIDEO = """Role: You are a Senior ASL Interpreter and Computer Vision Expert.
Task: Analyze this video carefully to identify the specific ASL sign being performed.

CRITICAL ANALYSIS STEPS:
1. Identify Anchor Points: Determine the hand's position relative to the body (e.g., Chin for AGE, Chest for BIRTHDAY, Nose for BORED).
2. Track Motion Vectors:
   - Y-axis: Is there a downward tug (AGE) or an upward rise (MORNING)?
   - Z-axis: Is the hand moving toward or away from the camera (AFTER)?
   - X-axis: Is there a side-to-side zigzag or oscillation (ART)?
3. Compound Sign Check: Does the sign have two distinct parts? (e.g., EAT + MORNING = BREAKFAST).
4. Hand Relationship: Is one hand acting as a base or 'canvas' while the other moves (ART/AFTER)?

Respond ONLY with this exact JSON format (no markdown, no backticks):
{"prediction": "WORD", "confidence": 0.95, "explanation": "Spatial Context: [location]. Movement: [axis/path]. Distinguisher: This matches [WORD] and not [SIMILAR SIGN] because of [reason].", "top3": [{"label": "WORD1", "confidence": 0.90}, {"label": "WORD2", "confidence": 0.07}, {"label": "WORD3", "confidence": 0.03}]}"""

# =============================================================
# ROUTES
# =============================================================

@app.get("/")
async def root():
    providers = []
    if claude_client: providers.append("claude")
    if gemini_client: providers.append("gemini")
    return {"status": "running", "providers": providers}

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
        if "error" in result:
            return JSONResponse(status_code=500, content=result)
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
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 2:
            cap.release()
            return []
        temp_dir = tempfile.mkdtemp()
        frames = []
        for i in range(num_frames):
            idx = int(i * (total - 1) / (num_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                path = f"{temp_dir}/frame_{i:02d}.jpg"
                cv2.imwrite(path, frame)
                frames.append(path)
        cap.release()
        print(f">>> Extracted {len(frames)} keyframes (OpenCV)")
        return frames
    except ImportError:
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
            return {"error": "Could not extract frames from video"}
        print(f">>> {len(keyframe_paths)} keyframes ready")

        # 1. Try Claude (primary)
        if claude_client:
            print(">>> Trying Claude...")
            result = analyze_with_claude(keyframe_paths, len(video_bytes))
            if result and "error" not in result:
                return result
            print(f">>> Claude failed: {result.get('error', 'unknown')}")

        # 2. Try Gemini with keyframes
        if gemini_client:
            print(">>> Trying Gemini keyframes...")
            result = analyze_with_gemini_keyframes(keyframe_paths, len(video_bytes))
            if result and "error" not in result:
                return result
            print(f">>> Gemini keyframes failed: {result.get('error', 'unknown')}")

            # 3. Try Gemini with raw video (last resort)
            print(">>> Trying Gemini video upload...")
            result = analyze_with_gemini_video(temp_path, len(video_bytes))
            if result and "error" not in result:
                return result

        return {"error": "All AI providers failed. Check API keys and quota."}

    except Exception as e:
        print(f"!!! Analysis error: {e}")
        import traceback; traceback.print_exc()
        return {"error": str(e)}
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
                img_b64 = base64.standard_b64encode(f.read()).decode("utf-8")
            content.append({"type": "text", "text": f"--- Frame {i+1} of {len(frame_paths)} ---"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}})
        content.append({"type": "text", "text": prompt_text})

        models_to_try = ["claude-sonnet-4-5-20250514", "claude-haiku-3-5-20241022"]
        for model_name in models_to_try:
            try:
                print(f">>> Claude: trying {model_name}...")
                response = claude_client.messages.create(
                    model=model_name, max_tokens=1024,
                    messages=[{"role": "user", "content": content}]
                )
                print(f">>> Claude: success with {model_name}")
                return parse_response(response.content[0].text.strip(), f"claude/{model_name}", video_size, "claude-keyframes")
            except anthropic.RateLimitError:
                print(f">>> Claude: rate limited on {model_name}")
                time.sleep(2)
            except anthropic.APIError as e:
                print(f">>> Claude: API error on {model_name}: {e}")
            except Exception as e:
                print(f">>> Claude: error on {model_name}: {e}")

        return {"error": "Claude: all models failed"}
    except Exception as e:
        return {"error": f"Claude error: {e}"}

# =============================================================
# GEMINI KEYFRAMES
# =============================================================

def analyze_with_gemini_keyframes(frame_paths: list, video_size: int) -> dict:
    try:
        prompt_text = ASL_PROMPT.format(num_frames=len(frame_paths))
        content_parts = [prompt_text]
        for i, path in enumerate(frame_paths):
            with open(path, "rb") as f:
                img_bytes = f.read()
            content_parts.append(f"\n--- Frame {i+1} of {len(frame_paths)} ---")
            content_parts.append(gemini_types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))

        models_to_try = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash-lite",
        ]
        for model_name in models_to_try:
            try:
                print(f">>> Gemini: trying {model_name}...")
                response = gemini_client.models.generate_content(model=model_name, contents=content_parts)
                print(f">>> Gemini: success with {model_name}")
                return parse_response(response.text.strip(), f"gemini/{model_name}", video_size, "gemini-keyframes")
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    print(f">>> Gemini: rate limited on {model_name}, waiting 5s...")
                    time.sleep(5)  # FIX: wait longer between rate-limited retries
                elif "503" in err or "UNAVAILABLE" in err:
                    print(f">>> Gemini: {model_name} unavailable, waiting 3s...")
                    time.sleep(3)
                else:
                    print(f">>> Gemini: {model_name} error: {err}")

        return {"error": "Gemini: all models rate limited or failed"}
    except Exception as e:
        return {"error": f"Gemini keyframes error: {e}"}

# =============================================================
# GEMINI VIDEO (last resort)
# =============================================================

def analyze_with_gemini_video(video_path: str, video_size: int) -> dict:
    try:
        print(">>> Gemini: uploading raw video...")
        uploaded_file = gemini_client.files.upload(file=video_path)
        waited = 0
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(2); waited += 2
            if waited > 60:
                return {"error": "Video processing timed out"}
            uploaded_file = gemini_client.files.get(name=uploaded_file.name)
        if uploaded_file.state.name == "FAILED":
            return {"error": "Gemini failed to process video"}

        for model_name in ["gemini-2.5-flash", "gemini-2.0-flash"]:
            try:
                response = gemini_client.models.generate_content(
                    model=model_name, contents=[uploaded_file, ASL_PROMPT_VIDEO]
                )
                try: gemini_client.files.delete(name=uploaded_file.name)
                except: pass
                return parse_response(response.text.strip(), f"gemini/{model_name}", video_size, "gemini-video")
            except Exception as e:
                print(f">>> Gemini video {model_name} error: {e}")
                time.sleep(3)

        return {"error": "Gemini video: all models failed"}
    except Exception as e:
        return {"error": f"Gemini video error: {e}"}

# =============================================================
# RESPONSE PARSER
# =============================================================

def parse_response(result_text: str, used_model: str, video_size: int, method: str) -> dict:
    if result_text.startswith("```"):
        lines = [l for l in result_text.split("\n") if not l.strip().startswith("```")]
        result_text = "\n".join(lines).strip()

    result = None
    try:
        result = json.loads(result_text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if match:
            try: result = json.loads(match.group())
            except: pass

    if not result:
        result = {"prediction": "UNKNOWN", "confidence": 0.0, "explanation": result_text, "top3": []}

    if "top3" not in result:
        result["top3"] = [{"label": result.get("prediction", "UNKNOWN"), "confidence": result.get("confidence", 0.0)}]

    prediction = result.get("prediction", "UNKNOWN").upper()
    confidence = min(1.0, max(0.0, float(result.get("confidence", 0.0))))
    print(f">>> RESULT: {prediction} ({confidence*100:.1f}%) via {method} [{used_model}]")

    return {
        "prediction": prediction,
        "confidence": confidence,
        "top3": result.get("top3", []),
        "explanation": result.get("explanation", ""),
        "debug": {"model": used_model, "method": method, "video_size": video_size}
    }
