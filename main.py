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
# Claude (primary)
import anthropic

# Gemini (fallback)
from google import genai
from google.genai import types as gemini_types

# --- Config ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Initialize clients
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# FastAPI app
app = FastAPI(title="Auxilium ASL Recognition Server")

NUM_KEYFRAMES = 8


# =============================================================
# ASL PROMPT — "Auxilium Precision" (works for both Claude & Gemini)
# =============================================================

ASL_PROMPT = """Role: You are a certified ASL interpreter and computer vision specialist.
Task: I am showing you {num_frames} keyframes extracted in chronological order from a video of someone performing a SINGLE ASL sign. Frame 1 is the start, Frame {num_frames} is the end. Analyze and identify the specific ASL sign being performed. Focus on identifying the sign accurately even if there is significant motion blur or fast movement.

Instructions: Analyze the frames and determine:

1. Initial Handshape: Describe the handshape of the dominant and non-dominant hands in Frame 1 (e.g., Flat-B, Open-A, 1-point, S-fist, C-shape, Claw-5, etc.).
2. Motion Path & Velocity: By comparing Frame 1 through Frame {num_frames}, trace the movement from start to finish. Is it linear, circular, repetitive, or a single stroke? Does the hand move forward, backward, up, down, or sideways? Is the motion fast or slow?
3. Hand Relationship: Do the hands touch? Does one slide over the other, or do they move symmetrically? Does one hand stay still while the other moves?
4. Spatial Location: Where is the sign performed relative to the head, chest, or neutral space?
5. Orientation: Which way are the palms facing throughout the movement? Does orientation change between frames?
6. Potential Confusions: List signs that look similar and explain why this specific video matches one over the other based on the motion path, palm orientation, or thumb position.

Respond with ONLY this JSON (no markdown, no backticks, no other text):
{{"prediction": "WORD", "confidence": 0.85, "explanation": "Initial handshape: [describe]. Motion: [direction/path]. Hand relationship: [describe]. This matches [WORD] because [reason], not [confused sign] because [difference].", "top3": [{{"label": "WORD1", "confidence": 0.85}}, {{"label": "WORD2", "confidence": 0.10}}, {{"label": "WORD3", "confidence": 0.05}}]}}

Rules:
- "prediction" = the English word in UPPERCASE
- "confidence" = number between 0.0 and 1.0
- "top3" = three most likely signs, confidences should roughly sum to 1.0
- In "explanation", always mention what signs could be confused and why you ruled them out"""


ASL_PROMPT_VIDEO = """Role: You are a certified ASL interpreter and computer vision specialist.
Task: Analyze the uploaded video and identify the specific ASL sign being performed. Focus on identifying the sign accurately even if there is significant motion blur or fast movement.

Instructions: Analyze the video and determine:

1. Initial Handshape: Describe the handshape at the very first frame.
2. Motion Path & Velocity: Trace the movement from start to finish. Direction? Speed?
3. Hand Relationship: Do the hands touch? Does one slide over the other?
4. Spatial Location: Where relative to head, chest, or neutral space?
5. Orientation: Which way are the palms facing?
6. Potential Confusions: List similar signs and explain why this one matches.

Respond with ONLY this JSON (no markdown, no backticks):
{"prediction": "WORD", "confidence": 0.85, "explanation": "description", "top3": [{"label": "W1", "confidence": 0.85}, {"label": "W2", "confidence": 0.10}, {"label": "W3", "confidence": 0.05}]}"""


# =============================================================
# ROUTES
# =============================================================

@app.get("/")
async def root():
    providers = []
    if claude_client:
        providers.append("claude")
    if gemini_client:
        providers.append("gemini")
    return {
        "status": "running",
        "providers": providers,
        "mode": "AI-powered ASL recognition (Claude + Gemini)",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "providers": {
        "claude": bool(claude_client),
        "gemini": bool(gemini_client),
    }}


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
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# =============================================================
# KEYFRAME EXTRACTION
# =============================================================

def extract_keyframes(video_path: str, num_frames: int = NUM_KEYFRAMES) -> list:
    """Extract evenly-spaced keyframes using OpenCV (most reliable)."""
    try:
        import cv2
    except ImportError:
        print("!!! OpenCV not available, trying ffmpeg...")
        return extract_keyframes_ffmpeg(video_path, num_frames)

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


def extract_keyframes_ffmpeg(video_path: str, num_frames: int = NUM_KEYFRAMES) -> list:
    """Fallback: Extract keyframes using ffmpeg."""
    temp_dir = tempfile.mkdtemp()
    try:
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-q:v", "2",
             f"{temp_dir}/all_%04d.jpg", "-y"],
            capture_output=True, text=True, timeout=30
        )
        all_frames = sorted(Path(temp_dir).glob("all_*.jpg"))
        if len(all_frames) > 0:
            step = max(1, len(all_frames) // num_frames)
            selected = all_frames[::step][:num_frames]
            print(f">>> Extracted {len(selected)} keyframes (ffmpeg)")
            return [str(f) for f in selected]
    except Exception as e:
        print(f"!!! ffmpeg error: {e}")
    return []


# =============================================================
# MAIN ANALYSIS PIPELINE
# =============================================================

def analyze_asl_video(video_bytes: bytes) -> dict:
    """Extract keyframes → try Claude → fallback to Gemini."""

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        temp_path = f.name

    try:
        # Extract keyframes
        keyframe_paths = extract_keyframes(temp_path)

        if len(keyframe_paths) < 2:
            return {"error": "Could not extract frames from video"}

        print(f">>> {len(keyframe_paths)} keyframes ready")

        # Strategy 1: Try Claude (primary — best at image analysis)
        if claude_client:
            print(">>> Trying Claude...")
            result = analyze_with_claude(keyframe_paths, len(video_bytes))
            if result and "error" not in result:
                return result
            print(f">>> Claude failed: {result.get('error', 'unknown')}")

        # Strategy 2: Fallback to Gemini
        if gemini_client:
            print(">>> Falling back to Gemini...")
            result = analyze_with_gemini_keyframes(keyframe_paths, len(video_bytes))
            if result and "error" not in result:
                return result
            print(f">>> Gemini keyframes failed, trying video upload...")

            # Strategy 3: Gemini with raw video (last resort)
            result = analyze_with_gemini_video(temp_path, len(video_bytes))
            if result and "error" not in result:
                return result

        return {"error": "All AI providers failed. Check API keys."}

    except Exception as e:
        print(f"!!! Analysis error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

    finally:
        try:
            os.unlink(temp_path)
        except:
            pass
        # Clean up keyframes
        for p in keyframe_paths:
            try:
                os.unlink(p)
            except:
                pass
        if keyframe_paths:
            try:
                os.rmdir(os.path.dirname(keyframe_paths[0]))
            except:
                pass


# =============================================================
# CLAUDE ANALYSIS (PRIMARY)
#
# Claude accepts multiple images in a single message.
# We send all keyframes as base64-encoded images with
# frame labels, plus the ASL analysis prompt.
# =============================================================

def analyze_with_claude(frame_paths: list, video_size: int) -> dict:
    """Send keyframes to Claude for ASL analysis."""

    try:
        prompt_text = ASL_PROMPT.format(num_frames=len(frame_paths))

        # Build content: images first, then prompt text
        content = []

        for i, path in enumerate(frame_paths):
            with open(path, "rb") as f:
                img_bytes = f.read()
            img_b64 = base64.standard_b64encode(img_bytes).decode("utf-8")

            # Add frame label
            content.append({
                "type": "text",
                "text": f"--- Frame {i+1} of {len(frame_paths)} ---"
            })
            # Add image
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": img_b64,
                }
            })

        # Add the prompt at the end
        content.append({
            "type": "text",
            "text": prompt_text,
        })

        # Try Claude models (best → cheapest)
        models_to_try = [
            "claude-sonnet-4-5-20250514",
            "claude-haiku-3-5-20241022",
        ]

        response = None
        used_model = "unknown"

        for model_name in models_to_try:
            try:
                print(f">>> Claude: trying {model_name}...")
                response = claude_client.messages.create(
                    model=model_name,
                    max_tokens=1024,
                    messages=[{
                        "role": "user",
                        "content": content,
                    }],
                )
                used_model = f"claude/{model_name}"
                print(f">>> Claude: success with {model_name}")
                break
            except anthropic.RateLimitError:
                print(f">>> Claude: rate limited on {model_name}")
                time.sleep(2)
                continue
            except anthropic.APIError as e:
                print(f">>> Claude: API error on {model_name}: {e}")
                continue
            except Exception as e:
                print(f">>> Claude: error on {model_name}: {e}")
                continue

        if response is None:
            return {"error": "Claude: all models failed"}

        result_text = response.content[0].text.strip()
        print(f">>> Claude raw response: {result_text}")

        return parse_response(result_text, used_model, video_size, "claude-keyframes")

    except Exception as e:
        print(f"!!! Claude error: {e}")
        return {"error": f"Claude error: {e}"}


# =============================================================
# GEMINI ANALYSIS (FALLBACK)
# =============================================================

def analyze_with_gemini_keyframes(frame_paths: list, video_size: int) -> dict:
    """Send keyframes to Gemini as images."""

    try:
        prompt_text = ASL_PROMPT.format(num_frames=len(frame_paths))

        content_parts = [prompt_text]
        for i, path in enumerate(frame_paths):
            with open(path, "rb") as f:
                img_bytes = f.read()
            content_parts.append(f"\n--- Frame {i+1} of {len(frame_paths)} ---")
            content_parts.append(
                gemini_types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
            )

        models_to_try = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash-lite",
        ]

        response = None
        used_model = "unknown"

        for model_name in models_to_try:
            try:
                print(f">>> Gemini: trying {model_name}...")
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=content_parts
                )
                used_model = f"gemini/{model_name}"
                print(f">>> Gemini: success with {model_name}")
                break
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    print(f">>> Gemini: rate limited on {model_name}")
                    time.sleep(3)
                    continue
                elif "503" in error_str or "UNAVAILABLE" in error_str:
                    print(f">>> Gemini: {model_name} unavailable")
                    time.sleep(2)
                    continue
                else:
                    print(f">>> Gemini: {model_name} error: {error_str}")
                    continue

        if response is None:
            return {"error": "Gemini: all models failed"}

        result_text = response.text.strip()
        print(f">>> Gemini raw response: {result_text}")

        return parse_response(result_text, used_model, video_size, "gemini-keyframes")

    except Exception as e:
        print(f"!!! Gemini keyframes error: {e}")
        return {"error": f"Gemini error: {e}"}


def analyze_with_gemini_video(video_path: str, video_size: int) -> dict:
    """Last resort: Upload raw video to Gemini."""

    try:
        print(">>> Gemini: uploading raw video...")
        uploaded_file = gemini_client.files.upload(file=video_path)

        max_wait = 60
        waited = 0
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(2)
            waited += 2
            if waited > max_wait:
                return {"error": "Video processing timed out"}
            uploaded_file = gemini_client.files.get(name=uploaded_file.name)

        if uploaded_file.state.name == "FAILED":
            return {"error": "Gemini failed to process video"}

        models_to_try = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
        ]

        response = None
        used_model = "unknown"

        for model_name in models_to_try:
            try:
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=[uploaded_file, ASL_PROMPT_VIDEO]
                )
                used_model = f"gemini/{model_name}"
                break
            except Exception:
                continue

        try:
            gemini_client.files.delete(name=uploaded_file.name)
        except:
            pass

        if response is None:
            return {"error": "Gemini video: all models failed"}

        result_text = response.text.strip()
        return parse_response(result_text, used_model, video_size, "gemini-video")

    except Exception as e:
        return {"error": f"Gemini video error: {e}"}


# =============================================================
# RESPONSE PARSER
# =============================================================

def parse_response(result_text: str, used_model: str, video_size: int, method: str) -> dict:
    """Parse AI response text into standardized result."""

    # Clean markdown fences
    if result_text.startswith("```"):
        lines = result_text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        result_text = "\n".join(lines).strip()

    # Parse JSON
    try:
        result = json.loads(result_text)
    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
            except json.JSONDecodeError:
                result = {
                    "prediction": "UNKNOWN",
                    "confidence": 0.0,
                    "explanation": result_text,
                    "top3": [{"label": "UNKNOWN", "confidence": 0.0}]
                }
        else:
            result = {
                "prediction": "UNKNOWN",
                "confidence": 0.0,
                "explanation": result_text,
                "top3": [{"label": "UNKNOWN", "confidence": 0.0}]
            }

    # Ensure required fields
    if "top3" not in result:
        result["top3"] = [{"label": result.get("prediction", "UNKNOWN"),
                           "confidence": result.get("confidence", 0.0)}]

    prediction = result.get("prediction", "UNKNOWN").upper()
    confidence = min(1.0, max(0.0, float(result.get("confidence", 0.0))))

    print(f">>> RESULT: {prediction} ({confidence*100:.1f}%) via {method}")

    return {
        "prediction": prediction,
        "confidence": confidence,
        "top3": result.get("top3", []),
        "explanation": result.get("explanation", ""),
        "debug": {
            "model": used_model,
            "method": method,
            "video_size": video_size,
        }
    }
