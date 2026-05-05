import os
import json
import re
import time
import tempfile
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from google import genai

# --- Config ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# FastAPI app
app = FastAPI(title="Auxilium ASL Recognition Server")


@app.get("/")
async def root():
    return {
        "status": "running",
        "model": "gemini-2.5-flash",
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


# =============================================================
# IMPROVED ASL PROMPT
#
# Key changes from the old prompt:
#
# 1. TEMPORAL ANALYSIS: Tells Gemini to analyze the video
#    frame-by-frame and track how hands MOVE over time.
#    This is critical for motion-heavy signs like AFTER,
#    BEFORE, AGAIN, FINISH, etc. that look similar in a
#    single frame but differ in movement direction.
#
# 2. SIGN COMPONENTS: Breaks down ASL signs into the 5
#    linguistic parameters (handshape, location, movement,
#    orientation, non-manual markers). This forces Gemini
#    to analyze each component instead of just guessing
#    from overall appearance.
#
# 3. COMMON CONFUSION PAIRS: Lists signs that look similar
#    and tells Gemini how to distinguish them. This directly
#    fixes the AFTER→ME misidentification.
#
# 4. STRICTER OUTPUT FORMAT: Reduces hallucination by being
#    very explicit about the JSON format.
# =============================================================

ASL_PROMPT = """You are a certified ASL (American Sign Language) interpreter with expertise in identifying signs from video recordings.

TASK: Watch this video carefully from start to finish and identify the ASL sign being performed.

CRITICAL ANALYSIS STEPS — follow these in order:

STEP 1 - TEMPORAL MOTION ANALYSIS (most important):
- Watch the ENTIRE video from first frame to last frame
- Track how the hands MOVE over time — direction, speed, repetition
- Note the STARTING position and ENDING position of each hand
- Movement direction is often what distinguishes similar signs

STEP 2 - HAND SHAPE ANALYSIS:
- Identify the handshape(s) used (flat hand, fist, pointed finger, etc.)
- Note if handshape changes during the sign
- Check both dominant and non-dominant hand

STEP 3 - LOCATION & CONTACT:
- Where are the hands relative to the body? (face, chest, waist, neutral space)
- Do the hands touch each other? Touch the body?
- What is the contact point?

STEP 4 - ORIENTATION:
- Which direction do the palms face? (up, down, inward, outward)
- Does palm orientation change during the sign?

STEP 5 - NON-MANUAL MARKERS:
- Check facial expression (raised eyebrows, mouth shape, head tilt)
- These can change the meaning entirely

COMMON CONFUSION PAIRS — pay special attention:
- AFTER vs ME: "AFTER" = flat hand moves FORWARD off the back of the other hand. "ME" = pointing to self/chest with index finger.
- BEFORE vs AFTER: Direction of movement is opposite
- HELP vs THANK-YOU: Both involve flat hand near chin, but movement differs
- WANT vs FREEZE: Similar handshape but different location and movement
- LIKE vs FAVORITE: Similar but FAVORITE touches chin
- UNDERSTAND vs KNOW: Location on forehead differs
- SORRY vs PLEASE: Both circular on chest, but handshape differs (fist vs flat)
- AGAIN vs REPEAT: Similar but AGAIN uses bent hand into flat palm
- HELLO vs GOODBYE: Wave direction and palm orientation differ
- YES vs NO: Fist nodding vs two fingers closing

OUTPUT FORMAT — respond with ONLY this JSON, no markdown, no backticks, no explanation outside the JSON:
{"prediction": "WORD", "confidence": 0.85, "explanation": "I observed [specific hand movements/shapes/positions]. The dominant hand [does what] while the non-dominant hand [does what]. The movement goes [direction].", "top3": [{"label": "WORD1", "confidence": 0.85}, {"label": "WORD2", "confidence": 0.10}, {"label": "WORD3", "confidence": 0.05}]}

Rules:
- "prediction" = the English word in UPPERCASE
- "confidence" = number between 0.0 and 1.0
- "explanation" = describe exactly what you SEE the hands doing (movement, shape, location)
- "top3" = three most likely signs with confidence scores that sum to ~1.0
- If unsure, lower your confidence score but still give your best guess
- Do NOT default to simple signs like "ME", "HELLO", "YES" unless you are truly confident"""


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
        max_wait = 60  # Don't wait forever
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

        print(f">>> File state: {uploaded_file.state.name}")

        # =====================================================
        # MODEL ORDER: Try best models FIRST, fall back to lite
        #
        # Old order (wrong): flash-lite → flash → 2.5-lite → 2.5
        # New order (correct): 2.5-flash → 2.0-flash → 2.5-lite → 2.0-lite
        #
        # The better models are MUCH more accurate at video
        # analysis and temporal understanding. Only fall back
        # to lite models if rate limited.
        # =====================================================
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
                print(f">>> Trying model: {model_name}")
                response = client.models.generate_content(
                    model=model_name,
                    contents=[uploaded_file, ASL_PROMPT]
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
                    continue  # Try next model instead of crashing

        if response is None:
            # All models failed — wait and retry with the most reliable one
            print(">>> All models failed. Waiting 30 seconds...")
            time.sleep(30)
            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[uploaded_file, ASL_PROMPT]
                )
                used_model = "gemini-2.0-flash (retry)"
            except Exception as e:
                return {"error": f"All models unavailable. Try again in a minute. ({e})"}

        result_text = response.text.strip()
        print(f">>> Gemini raw response: {result_text}")

        # Clean markdown fences if present
        if result_text.startswith("```"):
            lines = result_text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            result_text = "\n".join(lines).strip()

        # Parse JSON response
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
            # Try to extract JSON from mixed text
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

        # Ensure all required fields exist
        if "top3" not in result:
            result["top3"] = [{"label": result.get("prediction", "UNKNOWN"),
                               "confidence": result.get("confidence", 0.0)}]

        prediction = result.get("prediction", "UNKNOWN").upper()
        confidence = min(1.0, max(0.0, float(result.get("confidence", 0.0))))

        print(f">>> Prediction: {prediction} ({confidence*100:.1f}%)")
        print(f">>> Explanation: {result.get('explanation', 'N/A')}")
        print(f">>> Top 3: {result.get('top3', [])}")

        # Clean up uploaded file from Gemini
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
