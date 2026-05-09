import os
import json
import re
import time
import tempfile
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from google import genai

# --- Config ---
# Gamitin ang bagong key mo dito
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDQU02MBmJA93TDUm2DQUHq7-BRys0Z4Z4")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# FastAPI app
app = FastAPI(title="Auxilium ASL Recognition Server")


@app.get("/")
async def root():
    return {
        "status": "running",
        "model": "gemini-2.0-flash",
        "mode": "Optimized Auxilium Recognition",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "provider": "gemini"}


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
    """Send video to Gemini for ASL recognition using Precision Prompt."""

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        temp_path = f.name

    uploaded_file = None
    try:
        # Step 1: Upload file to Gemini
        print(">>> Uploading video to Gemini...")
        uploaded_file = client.files.upload(file=temp_path)
        print(f">>> Upload complete: {uploaded_file.name}")

        # Step 2: Robust Polling for ACTIVE state
        max_retries = 20
        for i in range(max_retries):
            uploaded_file = client.files.get(name=uploaded_file.name)
            state = uploaded_file.state.name
            print(f">>> Current File State: {state}")
            
            if state == "ACTIVE":
                break
            if state == "FAILED":
                return {"error": "Gemini failed to process video"}
            time.sleep(2)
        else:
            return {"error": "Video processing timed out."}

        # Step 3: Auxilium Precision Prompt (Merged)
        prompt = """Role: You are a Senior ASL Interpreter and Computer Vision Expert.
Analyze this video carefully to identify the specific ASL sign being performed.

CRITICAL ANALYSIS:
1. Anchor Points: Is the hand at the Chin (AGE), Chest (BIRTHDAY), or Nose (BORED)?
2. Motion Vectors: Trace the Y-axis (up/down), X-axis (side-to-side for ART), and Z-axis (forward for AFTER).
3. Compound Signs: Check if the sign has dalawang parts (e.g., EAT + MORNING = BREAKFAST).

Respond ONLY with this exact JSON format (no markdown, no backticks):
{"prediction": "WORD", "confidence": 0.95, "explanation": "Spatial Context: [location]. Movement: [axis shift]. Distinguisher: This is [WORD] and not [SIMILAR SIGN] because of [reason].", "top3": [{"label": "WORD1", "confidence": 0.85}, {"label": "WORD2", "confidence": 0.10}, {"label": "WORD3", "confidence": 0.05}]}"""

        # Step 4: Try models (Fallback chain)
        models_to_try = ["gemini-2.0-flash", "gemini-1.5-flash"]
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
            return {"error": "All models failed. Check quota or API key."}

        result_text = response.text.strip()
        
        # Step 5: Robust JSON Extraction
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            raise ValueError("No valid JSON found in AI response")

        # Standardizing output for Auxilium App
        prediction = result.get("prediction", "UNKNOWN").upper()
        confidence = float(result.get("confidence", 0.0))

        print(f">>> Final Prediction: {prediction} ({confidence*100:.1f}%)")

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
        print(f"!!! Analysis error: {e}")
        return {"error": str(e)}

    finally:
        # Clean up
        if uploaded_file:
            try: client.files.delete(name=uploaded_file.name)
            except: pass
        if os.path.exists(temp_path):
            os.unlink(temp_path)
