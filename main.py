import os
import json
import re
import time
import tempfile
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from google import genai

# --- Config ---
# Ginagamit ang Key na gumagana sa iyo
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDuAo2jPiTd2PT5tpdVkR9Pg4C8zqbzo8k")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="Auxilium ASL Recognition Server")

@app.get("/")
async def root():
    return {
        "status": "running",
        "model": "gemini-2.0-flash-lite",
        "mode": "AI-powered ASL recognition (Working Logic)",
    }

@app.get("/health")
async def health():
    return {"status": "ok", "model": "gemini"}

@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    try:
        video_bytes = await video.read()
        print(f"\n{'='*50}\n=== RECEIVED VIDEO: {len(video_bytes)} bytes ===")
        result = analyze_asl_video(video_bytes)
        if "error" in result:
            return JSONResponse(status_code=500, content=result)
        return JSONResponse(content=result)
    except Exception as e:
        print(f"!!! ERROR: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

def analyze_asl_video(video_bytes):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        temp_path = f.name

    try:
        print(">>> Uploading video to Gemini...")
        uploaded_file = client.files.upload(file=temp_path)
        
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(2)
            uploaded_file = client.files.get(name=uploaded_file.name)

        if uploaded_file.state.name == "FAILED":
            return {"error": "Gemini failed to process video"}

        # Heto ang Precision Prompt na hango sa naging test natin kanina
        prompt = """Role: You are a Senior ASL Interpreter. Analyze this video and identify the sign.
        
        CRITICAL RULES:
        1. Anchor Points: Check if the hand is at the Chin (AGE), Nose (BORED), or Chest (BIRTHDAY).
        2. Motion Vectors: Track Y-axis (down/up), X-axis (oscillation for ART), and Z-axis (forward for AFTER).
        3. Compound Check: Is it two signs? (e.g., EAT + MORNING = BREAKFAST).

        Respond ONLY with this exact JSON format:
        {"prediction": "WORD", "confidence": 0.95, "explanation": "Spatial: [location]. Motion: [axis shift]. Match: [reason].", "top3": [{"label": "W1", "confidence": 0.85}, {"label": "W2", "confidence": 0.10}, {"label": "W3", "confidence": 0.05}]}"""

        models_to_try = ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash"]
        response = None
        used_model = "unknown"

        for model_name in models_to_try:
            try:
                print(f">>> Trying model: {model_name}")
                response = client.models.generate_content(model=model_name, contents=[uploaded_file, prompt])
                used_model = model_name
                break
            except Exception as e:
                print(f"!!! {model_name} failed, trying next...")
                continue

        if response is None:
            return {"error": "All models failed. Quota reached or connection issue."}

        # Safe parsing using Regex
        result_text = response.text.strip()
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            raise ValueError("No JSON found")

        # Cleanup file after analysis
        try: client.files.delete(name=uploaded_file.name)
        except: pass

        return {
            "prediction": result.get("prediction", "UNKNOWN").upper(),
            "confidence": result.get("confidence", 0.0),
            "top3": result.get("top3", []),
            "explanation": result.get("explanation", ""),
            "debug": {"model": used_model, "video_size": len(video_bytes)}
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_path): os.unlink(temp_path)
