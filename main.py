import os
import json
import re
import time
import tempfile
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from google import genai

# --- Config ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCHhE34uKz9obBZxAieEh2CvcYRALnVCaA")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# FastAPI app
app = FastAPI(title="Auxilium ASL Recognition Server")


@app.get("/")
async def root():
    return {
        "status": "running",
        "model": "gemini-2.0-flash-lite",
        "mode": "AI-powered ASL recognition",
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
    """Send video to Gemini for ASL recognition."""

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

        # ASL analysis prompt
        prompt = """You are an expert ASL (American Sign Language) interpreter.

Analyze this video carefully and identify the ASL sign being performed.

Rules:
1. Focus on hand shapes, hand movements, hand positions relative to the body, and facial expressions
2. Consider both one-handed and two-handed signs
3. Be specific - give the exact English word/phrase the sign represents
4. If you can identify the sign, respond with ONLY a JSON object (no markdown, no backticks)
5. If you cannot identify the sign clearly, still give your best guess

Respond ONLY with this exact JSON format, nothing else:
{"prediction": "WORD", "confidence": 0.85, "explanation": "brief description of the sign observed", "top3": [{"label": "WORD1", "confidence": 0.85}, {"label": "WORD2", "confidence": 0.10}, {"label": "WORD3", "confidence": 0.05}]}

Important:
- prediction should be the English word in UPPERCASE
- confidence should be between 0.0 and 1.0
- top3 should list the 3 most likely signs
- explanation should briefly describe what hand movements/shapes you see"""

        # Try multiple models (fallback chain)
        models_to_try = [
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
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
                    raise

        if response is None:
            # All models failed — wait and retry once
            print(">>> All models failed. Waiting 30 seconds...")
            time.sleep(30)
            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash-lite",
                    contents=[uploaded_file, prompt]
                )
                used_model = "gemini-2.0-flash-lite"
            except Exception as e:
                return {"error": f"All models rate limited. Try again in a minute. ({e})"}

        result_text = response.text.strip()
        print(f">>> Gemini raw response: {result_text}")

        # Clean markdown
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
                result = json.loads(json_match.group())
            else:
                result = {
                    "prediction": "UNKNOWN",
                    "confidence": 0.0,
                    "explanation": result_text,
                    "top3": [{"label": "UNKNOWN", "confidence": 0.0}]
                }

        if "top3" not in result:
            result["top3"] = [{"label": result.get("prediction", "UNKNOWN"),
                               "confidence": result.get("confidence", 0.0)}]

        prediction = result.get("prediction", "UNKNOWN").upper()
        confidence = float(result.get("confidence", 0.0))

        print(f">>> Prediction: {prediction} ({confidence*100:.1f}%)")
        print(f">>> Explanation: {result.get('explanation', 'N/A')}")

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
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

    finally:
        try:
            os.unlink(temp_path)
        except:
            pass