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

from google import genai
from google.genai import types as gemini_types

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

app = FastAPI(title="Auxilium ASL Recognition Server")
NUM_KEYFRAMES = 12

# =============================================================
# PROMPTS
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
    return {
        "status": "running",
        "providers": ["gemini"] if gemini_client else [],
        "version": "3.0"
    }


@app.get("/health")
async def health():
    gemini_ok = False
    gemini_error = None
    if not GEMINI_API_KEY:
        gemini_error = "GEMINI_API_KEY is not set in environment"
    elif gemini_client:
        try:
            resp = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=["Say hello in one word"]
            )
            gemini_ok = True
        except Exception as e:
            gemini_error = str(e)
    else:
        gemini_error = "gemini_client failed to initialize"

    return {
        "status": "ok" if gemini_ok else "degraded",
        "providers": {
            "gemini": gemini_ok,
            "gemini_error": gemini_error
        }
    }


@app.get("/test-gemini")
async def test_gemini():
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY is not set in environment variables"}
    if not gemini_client:
        return {"error": "gemini_client failed to initialize — check API key"}
    try:
        resp = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=["Say hello in one word"]
        )
        return {"status": "ok", "response": resp.text}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    try:
        video_bytes = await video.read()
        print(f"\n{'='*50}")
        print(f"=== RECEIVED VIDEO: {len(video_bytes)} bytes ===")
        result = analyze_asl_video(video_bytes)
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
    if not gemini_client:
        return {
            "prediction": "UNKNOWN",
            "confidence": 0.0,
            "explanation": "GEMINI_API_KEY is not set. Please add it in Render environment variables.",
            "top3": [],
            "error": "No Gemini client"
        }

    keyframe_paths = []
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        temp_path = f.name

    try:
        keyframe_paths = extract_keyframes(temp_path)
        if len(keyframe_paths) < 2:
            return {
                "prediction": "UNKNOWN",
                "confidence": 0.0,
                "explanation": "Could not extract frames from video",
                "top3": [],
                "error": "Frame extraction failed"
            }

        print(f">>> {len(keyframe_paths)} keyframes ready")

        # 1. Try Gemini keyframes first
        print(">>> Trying Gemini keyframes...")
        result = analyze_with_gemini_keyframes(keyframe_paths, len(video_bytes))
        if result and "prediction" in result:
            return result
        print(f">>> Gemini keyframes failed: {result}")

        # 2. Fallback: Gemini video upload
        print(">>> Trying Gemini video upload...")
        result = analyze_with_gemini_video(temp_path, len(video_bytes))
        if result and "prediction" in result:
            return result
        print(f">>> Gemini video failed: {result}")

        return {
            "prediction": "UNKNOWN",
            "confidence": 0.0,
            "explanation": f"All Gemini methods failed. Last error: {result.get('error', 'unknown')}",
            "top3": []
        }

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

        # Valid Gemini models as of mid-2025
        models = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",  # reliable fallback
        ]

        for model in models:
            try:
                print(f">>> Gemini keyframes: trying {model}...")
                resp = gemini_client.models.generate_content(model=model, contents=parts)
                print(f">>> Gemini keyframes: success with {model}")
                return parse_response(resp.text, f"gemini/{model}", video_size, "gemini-keyframes")
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    print(f">>> Gemini: rate limited on {model}, waiting 10s...")
                    time.sleep(10)
                elif "404" in err or "not found" in err.lower():
                    print(f">>> Gemini: model {model} not found, trying next...")
                elif "403" in err or "API_KEY" in err or "permission" in err.lower():
                    print(f"!!! Gemini: AUTH ERROR — check GEMINI_API_KEY: {err}")
                    return {"error": f"Gemini auth failed: {err}"}
                else:
                    print(f">>> Gemini: {model} error: {err}")

        return {"error": "Gemini keyframes: all models failed"}
    except Exception as e:
        return {"error": f"Gemini keyframes error: {e}"}


# =============================================================
# GEMINI VIDEO UPLOAD (fallback)
# =============================================================

def analyze_with_gemini_video(video_path: str, video_size: int) -> dict:
    try:
        print(">>> Gemini: uploading raw video...")
        uploaded = gemini_client.files.upload(file=video_path)
        waited = 0
        while uploaded.state.name == "PROCESSING":
            time.sleep(2)
            waited += 2
            if waited > 60:
                return {"error": "Video processing timed out"}
            uploaded = gemini_client.files.get(name=uploaded.name)

        if uploaded.state.name == "FAILED":
            return {"error": "Gemini failed to process video"}

        for model in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
            try:
                print(f">>> Gemini video: trying {model}...")
                resp = gemini_client.models.generate_content(
                    model=model, contents=[uploaded, ASL_PROMPT_VIDEO]
                )
                try: gemini_client.files.delete(name=uploaded.name)
                except: pass
                print(f">>> Gemini video: success with {model}")
                return parse_response(resp.text, f"gemini/{model}", video_size, "gemini-video")
            except Exception as e:
                err = str(e)
                print(f">>> Gemini video {model} error: {err}")
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    time.sleep(10)
                elif "403" in err or "API_KEY" in err:
                    return {"error": f"Gemini auth failed: {err}"}

        return {"error": "Gemini video: all models failed"}
    except Exception as e:
        return {"error": f"Gemini video error: {e}"}


# =============================================================
# RESPONSE PARSER
# =============================================================

def parse_response(result_text: str, used_model: str, video_size: int, method: str) -> dict:
    print(f">>> Raw response: {result_text[:300]}")

    # Strip markdown fences
    cleaned = re.sub(r'```(?:json)?', '', result_text).strip()

    result = None

    # Try direct JSON parse
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON object (handles extra text around it)
    if result is None:
        cleaned_single = re.sub(r'\s+', ' ', cleaned)
        match = re.search(r'\{.*\}', cleaned_single, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    # Try extracting prediction from plain text
    if result is None:
        print(f"!!! Could not parse JSON from: {result_text[:200]}")
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
