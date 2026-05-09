import os
import json
import re
import time
import tempfile
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from google import genai

# --- Config ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCezZTdc114JRS-OQmChqerXChR89c-hm4")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# FastAPI app
app = FastAPI(title="Auxilium ASL Recognition Server")


@app.get("/")
async def root():
    return {
        "status": "running",
        "model": "gemini-2.0-flash",
        "mode": "AI-powered ASL recognition (Optimized Precision)",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": "gemini"}


@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    """Receive a video clip, send to Gemini for ASL recognition."""
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


def analyze_asl_video(video_bytes):
    """Send video to Gemini for ASL recognition using the Precision Prompt."""

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        temp_path = f.name

    try:
        # Upload file to Gemini
        print(">>> Uploading video to Gemini...")
        uploaded_file = client.files.upload(file=temp_path)
        print(f">>> Upload complete: {uploaded_file.name}")

        # Wait for processing
        while uploaded_file.state.name == "PROCESSING":
            print(">>> Waiting for Gemini to process video...")
            time.sleep(2)
            uploaded_file = client.files.get(name=uploaded_file.name)

        if uploaded_file.state.name == "FAILED":
            return {"error": "Gemini failed to process video"}

        print(f">>> File state: {uploaded_file.state.name}")

        # REFINED AUXILIUM PRECISION PROMPT
        prompt = """Role: You are a Senior ASL Interpreter and Computer Vision Expert.
Task: Analyze this video carefully to identify the specific ASL sign being performed.

CRITICAL ANALYSIS STEPS:
1. Identify Anchor Points: Where is the hand relative to the body? (e.g., Chin for AGE, Chest for BIRTHDAY, Nose for BORED).
2. Track Motion Vectors: 
   - Y-axis: Is there a downward tug (AGE) or an upward rise (MORNING)?
   - Z-axis: Is the hand moving toward or away from the camera (AFTER)?
   - X-axis: Is there a side-to-side zigzag or oscillation (ART)?
3. Compound Sign Check: Does the sign have two distinct parts? (e.g., EAT + MORNING = BREAKFAST).
4. Hand Relationship: Is one hand acting as a base or 'canvas' (ART/AFTER)?

Respond ONLY with this exact JSON format (no markdown, no backticks):
{"prediction": "WORD", "confidence": 0.95, "explanation": "Spatial Context: [location]. Movement: [axis/path]. Distinguisher: This matches [WORD] and not [SIMILAR SIGN] because of [reason].", "top3": [{"label": "WORD1", "confidence": 0.90}, {"label": "WORD2", "confidence": 0.07}, {"label": "WORD3", "confidence": 0.03}]}"""

        # Try multiple models (fallback chain)
        models_to_try = [
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-2.0-flash-lite",
        ]

        response = None
        used_model = "unknown"

        for model_name in models_to_try:
            try:
                print(f">>> Trying model: {model_name}")
                response = client.models.generate_content(
                    model=model_name,
                    contents=[uploaded_file, prompt]
                )
                used_model = model_name
                print(f">>> Success with {model_name}")
                break
            except Exception as e:
                print(f"!!! {model_name} failed, trying next...")
                continue

        if response is None:
            return {"error": "All Gemini models rate limited or failed."}

        result_text = response.text.strip()
        print(f">>> Gemini raw response: {result_text}")

        # Robust JSON extraction to prevent crashes
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {
                "prediction": "UNKNOWN",
                "confidence": 0.0,
                "explanation": "Invalid JSON response from AI",
                "top3": []
            }

        prediction = result.get("prediction", "UNKNOWN").upper()
        confidence = float(result.get("confidence", 0.0))

        # Clean up uploaded file
        try:
            client.files.delete(name=uploaded_file.name)
        except:
            pass

        return {
            "prediction": prediction,
            "confidence": confidence,
            "top3": result.get("top3", []),
            "explanation": result.get("explanation", ""),
            "debug": {
                "model": used_model,
                "video_size": len(video_bytes),
            }
        }

    except Exception as e:
        print(f"!!! Gemini error: {e}")
        return {"error": str(e)}

    finally:
        try:
            os.unlink(temp_path)
        except:
            pass
