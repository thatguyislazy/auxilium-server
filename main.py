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

# In-increase sa 12 para mas mahuli ang motion vectors (zigzag, compound transitions)
NUM_KEYFRAMES = 12 

# =============================================================
# OPTIMIZED PROMPTS (Precision-Weighted)
# =============================================================

ASL_PROMPT = """Role: Senior ASL Interpreter & Computer Vision Specialist.
Task: Identify the SINGLE ASL sign from {num_frames} chronological keyframes.

PRE-ANALYSIS LOGIC:
1. Track Landmarks: Trace the dominant hand path from Frame 1 to {num_frames}.
2. Spatial Anchors: Is the hand at the Chin (AGE), Nose (BORED), or Chest (BIRTHDAY)?
3. Motion Vector: 
   - Linear downward from chin: AGE.
   - X-axis zigzag on flat palm: ART.
   - Z-axis forward arc past base hand: AFTER.
   - Compound (Mouth then Morning arc): BREAKFAST.

Respond with ONLY this JSON format:
{{
  "prediction": "UPPERCASE_WORD",
  "confidence": 0.95,
  "explanation": "Trace the movement path. Explain why it is NOT a similar-looking sign.",
  "top3": [
    {{"label": "WORD1", "confidence": 0.90}},
    {{"label": "WORD2", "confidence": 0.07}},
    {{"label": "WORD3", "confidence": 0.03}}
  ]
}}"""

ASL_PROMPT_VIDEO = ASL_PROMPT.replace("{num_frames} chronological keyframes", "the uploaded video")

# =============================================================
# ROUTES
# =============================================================

@app.get("/")
async def root():
    providers = []
    if claude_client: providers.append("claude")
    if gemini_client: providers.append("gemini")
    return {"status": "running", "providers": providers, "version": "2.1-precision"}

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
        return JSONResponse(content=result)
    except Exception as e:
        print(f"!!! ERROR: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# =============================================================
# ROBUST KEYFRAME EXTRACTION
# =============================================================

def extract_keyframes(video_path: str, num_frames: int = NUM_KEYFRAMES) -> list:
    """Extracts unique keyframes using a while-loop for better frame accuracy."""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < 2:
            cap.release()
            return []

        # Determine target indices
        indices = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
        
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
        print(f">>> Extracted {len(keyframes)} precision keyframes")
        return keyframes
    except Exception as e:
        print(f"!!! OpenCV error: {e}, falling back to ffmpeg")
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

        # 1. Strategy: Claude (Primary - Best at spatial reasoning)
        if claude_client:
            print(">>> Strategy: Claude")
            result = analyze_with_claude(keyframe_paths, len(video_bytes))
            if result and "error" not in result: return result

        # 2. Strategy: Gemini Keyframes
        if gemini_client:
            print(">>> Strategy: Gemini Keyframes")
            result = analyze_with_gemini_keyframes(keyframe_paths, len(video_bytes))
            if result and "error" not in result: return result

            # 3. Strategy: Gemini Video Upload (Final Fallback)
            print(">>> Strategy: Gemini Video Upload")
            result = analyze_with_gemini_video(temp_path, len(video_bytes))
            if result and "error" not in result: return result

        return {"error": "All AI providers failed. Check keys/quota."}

    finally:
        if os.path.exists(temp_path): os.unlink(temp_path)
        for p in keyframe_paths:
            if os.path.exists(p): os.unlink(p)

# =============================================================
# MODEL WRAPPERS
# =============================================================

def analyze_with_claude(frame_paths: list, video_size: int) -> dict:
    try:
        prompt_text = ASL_PROMPT.format(num_frames=len(frame_paths))
        content = []
        for i, path in enumerate(frame_paths):
            with open(path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({"type": "text", "text": f"Frame {i+1}"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}})
        content.append({"type": "text", "text": prompt_text})

        # Try models (Primary 3.5 Sonnet is very stable)
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": content}]
        )
        return parse_response(response.content[0].text, "claude-3.5-sonnet", video_size, "claude-keyframes")
    except Exception as e:
        return {"error": str(e)}

def analyze_with_gemini_keyframes(frame_paths: list, video_size: int) -> dict:
    try:
        prompt_text = ASL_PROMPT.format(num_frames=len(frame_paths))
        parts = [prompt_text]
        for path in frame_paths:
            with open(path, "rb") as f:
                parts.append(gemini_types.Part.from_bytes(data=f.read(), mime_type="image/jpeg"))
        
        response = gemini_client.models.generate_content(model="gemini-2.0-flash", contents=parts)
        return parse_response(response.text, "gemini-2.0-flash", video_size, "gemini-keyframes")
    except Exception as e:
        return {"error": str(e)}

def analyze_with_gemini_video(video_path: str, video_size: int) -> dict:
    try:
        uploaded_file = gemini_client.files.upload(file=video_path)
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(2)
            uploaded_file = gemini_client.files.get(name=uploaded_file.name)
        
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[uploaded_file, ASL_PROMPT_VIDEO]
        )
        try: gemini_client.files.delete(name=uploaded_file.name)
        except: pass
        return parse_response(response.text, "gemini-video", video_size, "gemini-video")
    except Exception as e:
        return {"error": str(e)}

# =============================================================
# RESPONSE PARSER
# =============================================================

def parse_response(result_text: str, used_model: str, video_size: int, method: str) -> dict:
    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
    if not json_match:
        return {"error": "AI response did not contain valid JSON"}
    
    try:
        result = json.loads(json_match.group())
        prediction = result.get("prediction", "UNKNOWN").upper()
        confidence = float(result.get("confidence", 0.0))

        print(f">>> FINAL RESULT: {prediction} via {used_model}")

        return {
            "prediction": prediction,
            "confidence": confidence,
            "top3": result.get("top3", []),
            "explanation": result.get("explanation", ""),
            "debug": {"model": used_model, "method": method, "video_size": video_size}
        }
    except Exception as e:
        return {"error": f"JSON Parsing failed: {str(e)}"}
