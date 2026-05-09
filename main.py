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

# Initialize clients
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

app = FastAPI(title="Auxilium ASL Recognition Server")

# Dinaan natin sa 10 frames para mahuli ang fine details (e.g., zigzag sa ART o chin-to-chest sa BIRTHDAY)
NUM_KEYFRAMES = 10 


# =============================================================
# ASL PROMPT — "Auxilium Precision" (Optimized for Motion Vectors)
# =============================================================

ASL_PROMPT = """Role: You are a Senior ASL Interpreter and Computer Vision Expert.
Task: Analyze {num_frames} chronological keyframes to identify a SINGLE ASL sign. 

CRITICAL ANALYSIS STEPS:
1. Identify Anchor Points: Is the hand at the Chin, Forehead, Chest, or Neutral Space? (e.g., 'AGE' is at the chin).
2. Track Motion Vectors: 
   - Y-axis: Is there a downward 'tug' (AGE) or upward 'rise' (MORNING)?
   - Z-axis: Is it moving toward/away from the camera (AFTER)?
   - X-axis: Is there a side-to-side oscillation (ART)?
3. Compound Sign Detection: Check if the sequence shows two movements (e.g., 'EAT' then 'MORNING' for 'BREAKFAST').
4. Hand Relationship: Describe contact between dominant and non-dominant hands.

Respond with ONLY this JSON format:
{{
  "prediction": "UPPERCASE_WORD",
  "confidence": 0.95,
  "explanation": "Spatial Context: [location]. Movement: [axis shift]. Distinguisher: This is [WORD] and not [SIMILAR SIGN] because of [reason].",
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
    return {
        "status": "running",
        "providers": providers,
        "mode": "Auxilium Precision Logic (Claude 3.5/4.5 + Gemini 2.0/2.5)",
    }

@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    try:
        video_bytes = await video.read()
        print(f"\n{'='*50}\n=== RECEIVED VIDEO: {len(video_bytes)} bytes ===")
        result = analyze_asl_video(video_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        print(f"!!! ERROR: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# =============================================================
# KEYFRAME EXTRACTION
# =============================================================

def extract_keyframes(video_path: str, num_frames: int = NUM_KEYFRAMES) -> list:
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 2: return []

        temp_dir = tempfile.mkdtemp()
        frames = []
        for i in range(num_frames):
            idx = int(i * (total - 1) / (num_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                path = os.path.join(temp_dir, f"frame_{i:02d}.jpg")
                cv2.imwrite(path, frame)
                frames.append(path)
        cap.release()
        return frames
    except Exception as e:
        print(f"!!! Extraction error: {e}")
        return []


# =============================================================
# ANALYSIS PIPELINE
# =============================================================

def analyze_asl_video(video_bytes: bytes) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        temp_path = f.name

    try:
        keyframe_paths = extract_keyframes(temp_path)
        if len(keyframe_paths) < 2: return {"error": "Keyframe extraction failed"}

        # 1. Claude (Primary)
        if claude_client:
            print(">>> Strategy: Claude Keyframes")
            res = analyze_with_claude(keyframe_paths, len(video_bytes))
            if res and "error" not in res: return res

        # 2. Gemini Fallback
        if gemini_client:
            print(">>> Strategy: Gemini Fallback")
            res = analyze_with_gemini_keyframes(keyframe_paths, len(video_bytes))
            if res and "error" not in res: return res

        return {"error": "All AI providers failed"}
    finally:
        if os.path.exists(temp_path): os.unlink(temp_path)


def analyze_with_claude(frame_paths: list, video_size: int) -> dict:
    try:
        content = []
        for i, path in enumerate(frame_paths):
            with open(path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({"type": "text", "text": f"Frame {i+1}"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}})
        
        content.append({"type": "text", "text": ASL_PROMPT.format(num_frames=len(frame_paths))})

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
        parts = [ASL_PROMPT.format(num_frames=len(frame_paths))]
        for path in frame_paths:
            with open(path, "rb") as f:
                parts.append(gemini_types.Part.from_bytes(data=f.read(), mime_type="image/jpeg"))
        
        response = gemini_client.models.generate_content(model="gemini-2.0-flash", contents=parts)
        return parse_response(response.text, "gemini-2.0-flash", video_size, "gemini-keyframes")
    except Exception as e:
        return {"error": str(e)}


# =============================================================
# ROBUST PARSER
# =============================================================

def parse_response(result_text: str, used_model: str, video_size: int, method: str) -> dict:
    # Hanapin lang ang JSON block gamit ang Regex para hindi mag-crash sa extra text
    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
    if not json_match:
        return {"error": "No valid JSON found in AI response"}
    
    try:
        result = json.loads(json_match.group())
        prediction = result.get("prediction", "UNKNOWN").upper()
        confidence = float(result.get("confidence", 0.0))

        print(f">>> SUCCESS: {prediction} ({confidence*100:.1f}%)")

        return {
            "prediction": prediction,
            "confidence": confidence,
            "top3": result.get("top3", []),
            "explanation": result.get("explanation", ""),
            "debug": {"model": used_model, "method": method, "video_size": video_size}
        }
    except Exception as e:
        return {"error": f"Parsing failed: {str(e)}"}
