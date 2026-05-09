import os
import json
import re
import time
import tempfile
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types as gemini_types

# --- Config ---
# Mas mainam na kunin sa Environment Variables sa Render dashboard
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCHhE34uKz9obBZxAieEh2CvcYRALnVCaA")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="Auxilium ASL Recognition Server")

# =============================================================
# ROUTES
# =============================================================

@app.get("/")
async def root():
    return {
        "status": "running",
        "model": "gemini-2.0-flash",
        "mode": "Pure Gemini ASL Recognition",
    }

@app.get("/health")
async def health():
    # Siguraduhin na ito ay nagbabalik ng 200 OK para sa app mo
    return {"status": "ok", "provider": "gemini"}

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
        return JSONResponse(status_code=500, content={"error": str(e)})

# =============================================================
# CORE ANALYSIS LOGIC
# =============================================================

def analyze_asl_video(video_bytes: bytes):
    """Upload video to Gemini and use Auxilium Precision Prompt."""

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        temp_path = f.name

    try:
        # Upload file to Gemini
        print(">>> Uploading video to Gemini...")
        uploaded_file = client.files.upload(file=temp_path)

        # Wait for processing
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(2)
            uploaded_file = client.files.get(name=uploaded_file.name)

        if uploaded_file.state.name == "FAILED":
            return {"error": "Gemini failed to process video"}

        # AUXILIUM PRECISION PROMPT
        # In-update para sa spatial and vector awareness
        prompt = """Role: You are a Senior ASL Interpreter and Computer Vision Expert.
Analyze this video carefully to identify the specific ASL sign being performed.

CRITICAL ANALYSIS:
1. Anchor Points: Is the hand at the Chin (AGE), Chest (BIRTHDAY), or Nose (BORED)?
2. Motion Vectors: Trace the Y-axis (up/down), X-axis (side-to-side for ART), and Z-axis (forward for AFTER).
3. Compound Signs: Check if the sign has two parts (e.g., EAT + MORNING = BREAKFAST).

Respond ONLY with this exact JSON format (no markdown, no backticks):
{"prediction": "WORD", "confidence": 0.95, "explanation": "Spatial Context: [location]. Movement: [axis shift]. Distinguisher: This is [WORD] and not [SIMILAR SIGN] because of [reason].", "top3": [{"label": "WORD1", "confidence": 0.85}, {"label": "WORD2", "confidence": 0.10}, {"label": "WORD3", "confidence": 0.05}]}"""

        # Model Fallback Chain (Pure Gemini)
        models_to_try = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash"]
        response = None

        for model_name in models_to_try:
            try:
                print(f">>> Trying model: {model_name}")
                response = client.models.generate_content(
                    model=model_name,
                    contents=[uploaded_file, prompt]
                )
                break
            except Exception as e:
                print(f">>> {model_name} failed, trying next...")
                continue

        if response is None:
            return {"error": "All Gemini models failed. Check quota/API key."}

        # Parse and Clean Result
        result_text = response.text.strip()
        
        # Robust JSON extraction
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            raise ValueError("No valid JSON in response")

        # Clean up
        try:
            client.files.delete(name=uploaded_file.name)
        except:
            pass

        return {
            "prediction": result.get("prediction", "UNKNOWN").upper(),
            "confidence": result.get("confidence", 0.0),
            "top3": result.get("top3", []),
            "explanation": result.get("explanation", ""),
            "debug": {"model": model_name}
        }

    except Exception as e:
        print(f"!!! Analysis error: {e}")
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
