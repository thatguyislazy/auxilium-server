import os
import json
import re
import time
import tempfile
import subprocess
import base64
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types

# --- Config ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# FastAPI app
app = FastAPI(title="Auxilium ASL Recognition Server")

# Number of keyframes to extract from video
NUM_KEYFRAMES = 8


@app.get("/")
async def root():
    return {
        "status": "running",
        "model": "gemini-2.5-flash",
        "mode": "AI-powered ASL recognition (keyframe analysis)",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": "gemini"}


@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    """Receive a video clip, extract keyframes, send to Gemini for ASL recognition."""
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
#
# Instead of sending the raw video to Gemini (which it analyzes
# poorly for ASL), we extract N evenly-spaced keyframes and
# send them as a numbered image sequence. This lets Gemini:
#   1. See each frame clearly (no video compression artifacts)
#   2. Compare frames side-by-side to track motion
#   3. Focus on hand shape changes between frames
# =============================================================

def extract_keyframes(video_path: str, num_frames: int = NUM_KEYFRAMES) -> list:
    """Extract evenly-spaced keyframes from video using ffmpeg."""

    temp_dir = tempfile.mkdtemp()

    try:
        # Use ffmpeg to extract frames
        # -vf fps=N extracts N frames per second; instead we use select filter
        # for evenly spaced frames
        result = subprocess.run(
            [
                "ffmpeg", "-i", video_path,
                "-vf", f"select='not(mod(n\\,max(1\\,int(ceil(t_total/{num_frames})))))',setpts=N/FRAME_RATE/TB",
                "-frames:v", str(num_frames),
                "-q:v", "2",  # High quality JPEG
                f"{temp_dir}/frame_%02d.jpg",
                "-y",  # Overwrite
            ],
            capture_output=True, text=True, timeout=30
        )

        # If ffmpeg select filter fails, use simpler approach
        frames = sorted(Path(temp_dir).glob("frame_*.jpg"))

        if len(frames) < 2:
            # Fallback: use thumbnail filter
            subprocess.run(
                [
                    "ffmpeg", "-i", video_path,
                    "-vf", f"thumbnail={num_frames},setpts=N/FRAME_RATE/TB",
                    "-frames:v", str(num_frames),
                    "-q:v", "2",
                    f"{temp_dir}/frame_%02d.jpg",
                    "-y",
                ],
                capture_output=True, text=True, timeout=30
            )
            frames = sorted(Path(temp_dir).glob("frame_*.jpg"))

        if len(frames) < 2:
            # Final fallback: extract all frames and pick evenly spaced ones
            subprocess.run(
                [
                    "ffmpeg", "-i", video_path,
                    "-q:v", "2",
                    f"{temp_dir}/all_%04d.jpg",
                    "-y",
                ],
                capture_output=True, text=True, timeout=30
            )
            all_frames = sorted(Path(temp_dir).glob("all_*.jpg"))
            if len(all_frames) > 0:
                step = max(1, len(all_frames) // num_frames)
                frames = all_frames[::step][:num_frames]

        print(f">>> Extracted {len(frames)} keyframes")
        return [str(f) for f in frames]

    except Exception as e:
        print(f"!!! Frame extraction error: {e}")
        return []


def extract_keyframes_opencv(video_path: str, num_frames: int = NUM_KEYFRAMES) -> list:
    """Fallback: Extract keyframes using OpenCV if ffmpeg is unavailable."""
    try:
        import cv2
    except ImportError:
        print("!!! OpenCV not available")
        return []

    temp_dir = tempfile.mkdtemp()
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total < 2:
        cap.release()
        return []

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


# =============================================================
# ASL ANALYSIS PROMPTS
#
# Based on Gemini's own recommendation ("Auxilium Precision")
# for maximum accuracy. Two versions:
# - SEQUENCE: for keyframe image analysis (primary)
# - VIDEO: for raw video fallback
# =============================================================

ASL_PROMPT_SEQUENCE = """Role: You are a certified ASL interpreter and computer vision specialist.
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

1. Initial Handshape: Describe the handshape of the dominant and non-dominant hands at the very first frame (e.g., Flat-B, Open-A, 1-point, S-fist, C-shape, etc.).
2. Motion Path & Velocity: Trace the movement from start to finish. Is it linear, circular, repetitive, or a single stroke? Note if the motion is fast or slow. What DIRECTION does the hand move?
3. Hand Relationship: Do the hands touch? Does one slide over the other, or do they move symmetrically? Does one stay still?
4. Spatial Location: Where is the sign performed relative to the head, chest, or neutral space?
5. Orientation: Which way are the palms facing throughout the movement?
6. Potential Confusions: List signs that look similar (e.g., "After" vs "Clean", "Help" vs "Thank-you") and explain why this specific video matches one over the other based on the motion path, palm orientation, or thumb position.

Respond with ONLY this JSON (no markdown, no backticks, no other text):
{"prediction": "WORD", "confidence": 0.85, "explanation": "Initial handshape: [describe]. Motion: [direction/path]. This matches [WORD] because [reason], not [confused sign] because [difference].", "top3": [{"label": "WORD1", "confidence": 0.85}, {"label": "WORD2", "confidence": 0.10}, {"label": "WORD3", "confidence": 0.05}]}"""


def analyze_asl_video(video_bytes):
    """Extract keyframes and send to Gemini for ASL recognition."""

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        temp_path = f.name

    try:
        # =====================================================
        # STRATEGY 1: Extract keyframes and send as images
        # This is MUCH more accurate than raw video because
        # Gemini can compare individual frames side-by-side
        # =====================================================
        keyframe_paths = extract_keyframes(temp_path)

        if len(keyframe_paths) < 2:
            # Try OpenCV fallback
            keyframe_paths = extract_keyframes_opencv(temp_path)

        if len(keyframe_paths) >= 2:
            print(f">>> Using keyframe analysis ({len(keyframe_paths)} frames)")
            return analyze_with_keyframes(keyframe_paths, len(video_bytes))
        else:
            # =====================================================
            # STRATEGY 2: Fallback to raw video upload
            # Only used if frame extraction completely fails
            # =====================================================
            print(">>> Keyframe extraction failed, falling back to video upload")
            return analyze_with_video(temp_path, len(video_bytes))

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


def analyze_with_keyframes(frame_paths: list, video_size: int) -> dict:
    """Send keyframes as numbered images to Gemini."""

    try:
        # Build the content: prompt + numbered images
        prompt = ASL_PROMPT_SEQUENCE.format(num_frames=len(frame_paths))

        # Build content parts: text prompt first, then images
        content_parts = [prompt]

        for i, path in enumerate(frame_paths):
            with open(path, "rb") as img_file:
                img_bytes = img_file.read()

            # Add frame label + image
            content_parts.append(f"\n--- Frame {i+1} of {len(frame_paths)} ---")
            content_parts.append(
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
            )

        # Try models in order (best first)
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
                print(f">>> Trying {model_name} with {len(frame_paths)} keyframes...")
                response = client.models.generate_content(
                    model=model_name,
                    contents=content_parts
                )
                used_model = model_name
                print(f">>> Success with {model_name}")
                break
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    print(f">>> Rate limited on {model_name}, trying next...")
                    time.sleep(3)
                    continue
                elif "503" in error_str or "UNAVAILABLE" in error_str:
                    print(f">>> {model_name} unavailable, trying next...")
                    time.sleep(2)
                    continue
                else:
                    print(f">>> {model_name} error: {error_str}")
                    continue

        if response is None:
            print(">>> All models failed on keyframes. Waiting 30s...")
            time.sleep(30)
            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=content_parts
                )
                used_model = "gemini-2.0-flash (retry)"
            except Exception as e:
                return {"error": f"All models unavailable. ({e})"}

        return parse_response(response, used_model, video_size, "keyframes")

    finally:
        # Clean up frame files
        for path in frame_paths:
            try:
                os.unlink(path)
            except:
                pass
        # Try to remove temp directory
        if frame_paths:
            try:
                os.rmdir(os.path.dirname(frame_paths[0]))
            except:
                pass


def analyze_with_video(video_path: str, video_size: int) -> dict:
    """Fallback: Upload raw video to Gemini."""

    # Upload file
    print(">>> Uploading video to Gemini...")
    uploaded_file = client.files.upload(file=video_path)
    print(f">>> Upload complete: {uploaded_file.name}")

    # Wait for processing
    max_wait = 60
    waited = 0
    while uploaded_file.state.name == "PROCESSING":
        print(">>> Waiting for Gemini to process video...")
        time.sleep(2)
        waited += 2
        if waited > max_wait:
            return {"error": "Video processing timed out"}
        uploaded_file = client.files.get(name=uploaded_file.name)

    if uploaded_file.state.name == "FAILED":
        return {"error": "Gemini failed to process video"}

    # Try models
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
            print(f">>> Trying {model_name} with video...")
            response = client.models.generate_content(
                model=model_name,
                contents=[uploaded_file, ASL_PROMPT_VIDEO]
            )
            used_model = model_name
            break
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                time.sleep(3)
                continue
            elif "503" in error_str or "UNAVAILABLE" in error_str:
                time.sleep(2)
                continue
            else:
                continue

    if response is None:
        time.sleep(30)
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[uploaded_file, ASL_PROMPT_VIDEO]
            )
            used_model = "gemini-2.0-flash (retry)"
        except Exception as e:
            return {"error": f"All models unavailable. ({e})"}

    # Clean up
    try:
        client.files.delete(name=uploaded_file.name)
    except:
        pass

    return parse_response(response, used_model, video_size, "video")


def parse_response(response, used_model: str, video_size: int, method: str) -> dict:
    """Parse Gemini response into standardized result."""

    result_text = response.text.strip()
    print(f">>> Gemini raw response: {result_text}")

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

    print(f">>> Prediction: {prediction} ({confidence*100:.1f}%)")
    print(f">>> Explanation: {result.get('explanation', 'N/A')}")
    print(f">>> Method: {method}")

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
