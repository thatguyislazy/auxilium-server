import os
import json
import re
import time
import math
import tempfile
import subprocess
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from google import genai
from google.genai import types as gemini_types

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

app = FastAPI(title="Auxilium ASL Recognition Server")

NUM_KEYFRAMES   = 20   # dense frames for image analysis
NUM_STRIP_STEPS = 8    # sparse frames for motion-strip pass

# MediaPipe landmark names (indices 0–20) for readable prompt injection
MP_LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC","THUMB_MCP","THUMB_IP","THUMB_TIP",
    "INDEX_MCP","INDEX_PIP","INDEX_DIP","INDEX_TIP",
    "MIDDLE_MCP","MIDDLE_PIP","MIDDLE_DIP","MIDDLE_TIP",
    "RING_MCP","RING_PIP","RING_DIP","RING_TIP",
    "PINKY_MCP","PINKY_PIP","PINKY_DIP","PINKY_TIP",
]

# =============================================================
# PROMPTS
# =============================================================

ASL_KEYFRAME_PROMPT = """\
You are a senior ASL interpreter analyzing {num_frames} chronological keyframes \
(Frame 1 = start of sign, Frame {num_frames} = end of sign) extracted at equal \
intervals from a single continuous ASL sign.

{landmark_section}

── STEP 1 · MOTION TRAJECTORY ────────────────────────────────────────────────
Track the dominant hand through every frame and describe:
  a) Start position (body-relative: forehead / nose / chin / chest / waist / neutral space)
  b) End position
  c) Primary axis of movement (X = lateral, Y = vertical, Z = toward/away camera)
  d) Path shape (arc / straight / circular / zigzag / bouncing / twisting)
  e) Palm orientation changes (facing signer vs camera vs down vs up)
  f) Finger configuration changes (open → fist, index-out, etc.)
  g) Whether the non-dominant hand is static base, mirrors, or is absent

── STEP 2 · VELOCITY & RHYTHM ──────────────────────────────────────────────
Compare hand positions across frames to estimate:
  a) Speed profile (slow-start / constant / fast-finish / decelerate / staccato)
  b) Number of repeated movements (1 = single, 2+ = repeated)
  c) Any holds or pauses mid-sign

── STEP 3 · SIGN IDENTIFICATION ────────────────────────────────────────────
Using the trajectory and rhythm you described, identify the SINGLE ASL sign \
being performed. Consider the full ASL lexicon — do NOT restrict yourself to \
any predefined list.

Disambiguate carefully using the landmark data if provided:
  • Wrist-to-chin distance < 0.15 → sign is near face (AGE, BORED, UGLY...)
  • Index tip traces large arc → HELP, PLEASE, SORRY...
  • Repeated side-to-side on flat palm → ART / PAINT
  • Symmetrical outward arc from chest → OPEN / SHARE
  • Wrist rotation with index up → CHANGE / TURN
  • Finger extension/flexion sequence → letters vs words

── OUTPUT FORMAT ────────────────────────────────────────────────────────────
Respond with ONLY one single-line JSON object — no markdown, no backticks, \
no extra text before or after:
{{"prediction":"UPPERCASE_WORD","confidence":0.95,"trajectory":"start→path→end in 10 words","explanation":"Step-1 summary. Step-2 summary. Why this sign and not similar ones.","top3":[{{"label":"WORD1","confidence":0.90}},{{"label":"WORD2","confidence":0.07}},{{"label":"WORD3","confidence":0.03}}]}}
"""

ASL_VIDEO_PROMPT = """\
You are a senior ASL interpreter. Watch this video of a single ASL sign \
from start to finish and identify it.

{landmark_section}

Follow this reasoning process:
  1. Describe the dominant hand's start position, end position, and path shape.
  2. Note palm orientation and finger configuration at key moments.
  3. Note speed, rhythm, and any repetitions.
  4. Identify the sign — consider the full ASL lexicon, not a preset list.
  5. If landmark data is provided above, use it to confirm spatial positions.

Respond with ONLY one single-line JSON object — no markdown, no backticks:
{{"prediction":"UPPERCASE_WORD","confidence":0.95,"trajectory":"start→path→end in 10 words","explanation":"Trajectory. Landmark evidence. Disambiguation.","top3":[{{"label":"WORD1","confidence":0.90}},{{"label":"WORD2","confidence":0.07}},{{"label":"WORD3","confidence":0.03}}]}}
"""

ASL_MOTION_STRIP_PROMPT = """\
You are analyzing {num_frames} equally-spaced frames from an ASL sign video.
Describe ONLY the dominant hand's motion in ONE sentence covering:
  start-position, end-position, path-shape, palm-orientation-change, speed, repetitions.
Example: "Hand moves from chin downward in a straight arc, palm facing left, single smooth motion."
Respond with ONLY the sentence — no JSON, no extra text.
"""


# =============================================================
# ROUTES
# =============================================================

@app.get("/")
async def root():
    return {
        "status": "running",
        "providers": ["gemini"] if gemini_client else [],
        "version": "5.0-mediapipe"
    }


@app.get("/health")
async def health():
    gemini_ok = False
    gemini_error = None
    if not GEMINI_API_KEY:
        gemini_error = "GEMINI_API_KEY is not set in environment"
    elif gemini_client:
        try:
            gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=["Say hello in one word"]
            )
            gemini_ok = True
        except Exception as e:
            gemini_error = str(e)
    else:
        gemini_error = "gemini_client failed to initialize"

    # Check MediaPipe
    try:
        import mediapipe as mp
        mp_version = mp.__version__
    except Exception as e:
        mp_version = f"UNAVAILABLE: {e}"

    return {
        "status": "ok" if gemini_ok else "degraded",
        "providers": {"gemini": gemini_ok, "gemini_error": gemini_error},
        "mediapipe_version": mp_version
    }


@app.get("/test-gemini")
async def test_gemini():
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY is not set"}
    if not gemini_client:
        return {"error": "gemini_client failed to initialize"}
    try:
        resp = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=["Say hello in one word"]
        )
        return {"status": "ok", "response": resp.text}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    try:
        video_bytes = await video.read()
        print(f"\n{'='*50}")
        print(f"=== RECEIVED VIDEO: {len(video_bytes)} bytes ===")
        result = analyze_asl_video(video_bytes)
        if result is None:
            return JSONResponse(status_code=500, content={"error": "No result"})
        return JSONResponse(content=result)
    except Exception as e:
        print(f"!!! ERROR: {e}")
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# =============================================================
# PASS 0 · MEDIAPIPE LANDMARK EXTRACTION
# =============================================================

def extract_landmarks(video_path: str, max_frames: int = 16) -> list[dict]:
    """
    Run MediaPipe Hands on evenly-spaced frames and return a list of
    per-frame landmark dicts.  Each dict has:
        frame_idx  : int
        handedness : "Left" | "Right"
        landmarks  : list of {name, x, y, z}   (normalized 0-1 coords)
        derived    : {wrist_x, wrist_y, finger_spread, finger_curl_index, ...}

    Returns [] on any failure — caller treats landmarks as optional.
    """
    try:
        import cv2
        import mediapipe as mp

        mp_hands  = mp.solutions.hands
        hands_sol = mp_hands.Hands(
            static_image_mode=True,   # process each frame independently
            max_num_hands=2,
            min_detection_confidence=0.5,
            model_complexity=1        # 0=lite, 1=full — use full for accuracy
        )

        cap   = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 2:
            cap.release()
            return []

        # Pick evenly-spaced frame indices
        indices = set(
            int(i * (total - 1) / (max_frames - 1)) for i in range(max_frames)
        )

        results_list = []
        count        = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if count in indices:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands_sol.process(rgb)
                if result.multi_hand_landmarks and result.multi_handedness:
                    # Prefer the dominant / more-confident hand
                    best_hand_lm   = result.multi_hand_landmarks[0]
                    best_handedness = result.multi_handedness[0].classification[0].label

                    lm_list = []
                    for idx, lm in enumerate(best_hand_lm.landmark):
                        lm_list.append({
                            "name": MP_LANDMARK_NAMES[idx],
                            "x": round(lm.x, 4),
                            "y": round(lm.y, 4),
                            "z": round(lm.z, 4),
                        })

                    derived = _derive_hand_features(lm_list)

                    results_list.append({
                        "frame_idx":  count,
                        "handedness": best_handedness,
                        "landmarks":  lm_list,
                        "derived":    derived,
                    })
            count += 1

        cap.release()
        hands_sol.close()
        print(f">>> MediaPipe: extracted landmarks from {len(results_list)} frames")
        return results_list

    except Exception as e:
        print(f">>> MediaPipe failed (non-fatal): {e}")
        return []


def _derive_hand_features(lm_list: list) -> dict:
    """
    Compute a few scalar features from 21 landmarks that are directly
    useful for sign disambiguation.
    """
    try:
        def lm(name):
            for l in lm_list:
                if l["name"] == name:
                    return l
            return {"x": 0, "y": 0, "z": 0}

        def dist(a, b):
            return math.sqrt(
                (a["x"]-b["x"])**2 + (a["y"]-b["y"])**2 + (a["z"]-b["z"])**2
            )

        wrist       = lm("WRIST")
        index_tip   = lm("INDEX_TIP")
        middle_tip  = lm("MIDDLE_TIP")
        ring_tip    = lm("RING_TIP")
        pinky_tip   = lm("PINKY_TIP")
        thumb_tip   = lm("THUMB_TIP")
        index_mcp   = lm("INDEX_MCP")
        middle_mcp  = lm("MIDDLE_MCP")

        # Finger extension: tip-to-wrist vs mcp-to-wrist
        # >1.0 means finger is extended, <0.8 means curled
        def extension(tip_name, mcp_name):
            tip = lm(tip_name); mcp = lm(mcp_name)
            tip_d = dist(wrist, tip)
            mcp_d = dist(wrist, mcp)
            return round(tip_d / (mcp_d + 1e-6), 3)

        # Spread = average distance between adjacent fingertips
        tips = [index_tip, middle_tip, ring_tip, pinky_tip]
        spreads = [dist(tips[i], tips[i+1]) for i in range(len(tips)-1)]
        avg_spread = round(sum(spreads) / len(spreads), 4)

        # Palm normal approximation via cross product of two palm vectors
        # Positive z-component → palm facing camera; negative → facing away
        v1 = (index_mcp["x"] - wrist["x"],
              index_mcp["y"] - wrist["y"],
              index_mcp["z"] - wrist["z"])
        v2 = (middle_mcp["x"] - wrist["x"],
              middle_mcp["y"] - wrist["y"],
              middle_mcp["z"] - wrist["z"])
        palm_z = v1[0]*v2[1] - v1[1]*v2[0]   # z component of cross product

        return {
            "wrist_x":             round(wrist["x"], 3),
            "wrist_y":             round(wrist["y"], 3),
            "index_extension":     extension("INDEX_TIP",  "INDEX_MCP"),
            "middle_extension":    extension("MIDDLE_TIP", "MIDDLE_MCP"),
            "ring_extension":      extension("RING_TIP",   "RING_MCP"),
            "pinky_extension":     extension("PINKY_TIP",  "PINKY_MCP"),
            "thumb_extension":     extension("THUMB_TIP",  "THUMB_CMC"),
            "finger_spread":       avg_spread,
            "palm_facing_camera":  palm_z > 0,
            "thumb_index_dist":    round(dist(thumb_tip, index_tip), 4),
        }
    except Exception:
        return {}


def _format_landmark_section(landmarks: list[dict]) -> str:
    """
    Convert the raw landmark list into a compact, human-readable block
    that can be pasted into the Gemini prompt.
    Only key landmarks are shown per frame to keep token count low.
    """
    if not landmarks:
        return ""

    lines = ["── MEDIAPIPE HAND LANDMARK DATA ──────────────────────────────────────"]
    lines.append(
        "Coordinates are normalized (0-1): x=left→right, y=top→bottom, z=depth.\n"
        "Extension ratio >1.0 = finger extended; <0.8 = curled."
    )

    key_lms = {"WRIST", "THUMB_TIP", "INDEX_TIP", "MIDDLE_TIP", "RING_TIP", "PINKY_TIP"}

    for i, frame_data in enumerate(landmarks):
        d = frame_data.get("derived", {})
        lm_subset = [
            l for l in frame_data["landmarks"] if l["name"] in key_lms
        ]
        coord_str = "  ".join(
            f"{l['name']}({l['x']},{l['y']})" for l in lm_subset
        )
        ext_str = (
            f"ext=[I:{d.get('index_extension','?')} "
            f"M:{d.get('middle_extension','?')} "
            f"R:{d.get('ring_extension','?')} "
            f"P:{d.get('pinky_extension','?')} "
            f"T:{d.get('thumb_extension','?')}]"
        )
        palm_str = "palm→cam" if d.get("palm_facing_camera") else "palm→away"
        spread_str = f"spread:{d.get('finger_spread','?')}"
        hand_str = frame_data.get("handedness", "?")

        lines.append(
            f"F{i+1:02d} [{hand_str}] {coord_str} | {ext_str} | {palm_str} | {spread_str}"
        )

    lines.append("──────────────────────────────────────────────────────────────────\n")
    return "\n".join(lines)


def _summarize_landmark_motion(landmarks: list[dict]) -> str:
    """
    Produce a compact human-readable summary of the overall hand trajectory
    derived purely from landmark math (no vision model needed).
    Injected alongside the visual data to help Gemini.
    """
    if len(landmarks) < 2:
        return ""

    try:
        first = landmarks[0]["derived"]
        last  = landmarks[-1]["derived"]

        # Wrist travel direction
        dx = last["wrist_x"] - first["wrist_x"]
        dy = last["wrist_y"] - first["wrist_y"]

        h_dir = "rightward" if dx > 0.05 else ("leftward" if dx < -0.05 else "laterally stable")
        v_dir = "downward"  if dy > 0.05 else ("upward"   if dy < -0.05 else "vertically stable")

        # Finger state across frames
        ext_vals = [f["derived"].get("index_extension", 1.0) for f in landmarks]
        avg_ext  = sum(ext_vals) / len(ext_vals)
        finger_state = "extended" if avg_ext > 1.0 else ("curled" if avg_ext < 0.8 else "mid-flex")

        # Palm orientation consistency
        palm_states = [f["derived"].get("palm_facing_camera", True) for f in landmarks]
        palm_flips  = sum(1 for i in range(1, len(palm_states)) if palm_states[i] != palm_states[i-1])
        palm_desc   = (
            "palm orientation flips (wrist rotation present)" if palm_flips >= 2
            else ("palm facing camera throughout" if palm_states[0] else "palm facing away throughout")
        )

        # Finger spread changes (open → closed or vice versa)
        spread_start = landmarks[0]["derived"].get("finger_spread", 0)
        spread_end   = landmarks[-1]["derived"].get("finger_spread", 0)
        spread_change = spread_end - spread_start
        spread_desc = (
            "fingers close during sign" if spread_change < -0.03
            else ("fingers open during sign" if spread_change > 0.03 else "finger spread stable")
        )

        return (
            f"Landmark summary: wrist moves {h_dir} and {v_dir}; "
            f"index finger mostly {finger_state}; {palm_desc}; {spread_desc}."
        )
    except Exception:
        return ""


# =============================================================
# KEYFRAME EXTRACTION + ANNOTATION
# =============================================================

def extract_keyframes(video_path: str, num_frames: int = NUM_KEYFRAMES) -> list:
    try:
        import cv2
        cap   = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 2:
            cap.release()
            return []

        indices   = set(
            int(i * (total - 1) / (num_frames - 1)) for i in range(num_frames)
        )
        temp_dir  = tempfile.mkdtemp()
        keyframes, count = [], 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if count in indices:
                path = os.path.join(temp_dir, f"frame_{len(keyframes):02d}.jpg")
                cv2.imwrite(path, frame)
                keyframes.append(path)
            count += 1
            if len(keyframes) == num_frames:
                break

        cap.release()
        keyframes = _annotate_frames(keyframes)
        print(f">>> Extracted {len(keyframes)} keyframes (OpenCV)")
        return keyframes

    except Exception as e:
        print(f"!!! OpenCV error: {e}, trying ffmpeg...")
        return extract_keyframes_ffmpeg(video_path, num_frames)


def extract_keyframes_ffmpeg(video_path: str, num_frames: int = NUM_KEYFRAMES) -> list:
    temp_dir = tempfile.mkdtemp()
    try:
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-q:v", "2",
             f"{temp_dir}/all_%04d.jpg", "-y"],
            capture_output=True, text=True, timeout=30
        )
        all_frames = sorted(Path(temp_dir).glob("all_*.jpg"))
        if all_frames:
            step     = max(1, len(all_frames) // num_frames)
            selected = all_frames[::step][:num_frames]
            selected = _annotate_frames([str(f) for f in selected])
            print(f">>> Extracted {len(selected)} keyframes (ffmpeg)")
            return selected
    except Exception as e:
        print(f"!!! ffmpeg error: {e}")
    return []


def _annotate_frames(frame_paths: list) -> list:
    """Burn a frame-number badge into each image for temporal anchoring."""
    try:
        import cv2
        n = len(frame_paths)
        for i, path in enumerate(frame_paths):
            img = cv2.imread(path)
            if img is None:
                continue
            label      = f"F{i+1}/{n}"
            h, w       = img.shape[:2]
            font_scale = max(0.5, w / 640)
            thickness  = max(1, int(font_scale * 1.5))
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            cv2.rectangle(img, (4, 4), (tw + 12, th + 14), (255, 255, 255), -1)
            cv2.putText(
                img, label, (8, th + 8),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), thickness
            )
            cv2.imwrite(path, img)
        return frame_paths
    except Exception:
        return frame_paths


# =============================================================
# MAIN PIPELINE  (Pass 0 → 1 → 2 → 3)
# =============================================================

def analyze_asl_video(video_bytes: bytes) -> dict:
    if not gemini_client:
        return {
            "prediction":  "UNKNOWN",
            "confidence":  0.0,
            "explanation": "GEMINI_API_KEY is not set.",
            "top3":        [],
            "error":       "No Gemini client"
        }

    keyframe_paths = []
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        temp_path = f.name

    try:
        # ── Pass 0: MediaPipe landmark extraction ────────────────────────────
        print(">>> Pass 0: MediaPipe landmark extraction...")
        landmarks = extract_landmarks(temp_path)
        landmark_section = _format_landmark_section(landmarks)
        landmark_summary = _summarize_landmark_motion(landmarks)
        if landmark_summary:
            print(f">>> Landmark summary: {landmark_summary}")

        # ── Extract keyframes for visual passes ──────────────────────────────
        keyframe_paths = extract_keyframes(temp_path, NUM_KEYFRAMES)
        if len(keyframe_paths) < 2:
            return {
                "prediction":  "UNKNOWN",
                "confidence":  0.0,
                "explanation": "Could not extract frames from video.",
                "top3":        [],
                "error":       "Frame extraction failed"
            }

        print(f">>> {len(keyframe_paths)} keyframes ready")

        # ── Pass 1: motion-strip description (cheap) ─────────────────────────
        motion_context = _get_motion_description(keyframe_paths)
        if landmark_summary:
            # Prepend landmark-derived summary — it's more precise than vision
            motion_context = f"{landmark_summary} {motion_context}".strip()
        print(f">>> Motion context: {motion_context}")

        # ── Pass 2: full keyframe analysis with landmarks + motion context ───
        result = analyze_with_gemini_keyframes(
            keyframe_paths, len(video_bytes), motion_context, landmark_section
        )
        if result and "prediction" in result:
            result["landmark_frames"] = len(landmarks)
            return result
        print(f">>> Gemini keyframes failed: {result}")

        # ── Pass 3 fallback: native video upload ─────────────────────────────
        print(">>> Pass 3: Gemini video upload fallback...")
        result = analyze_with_gemini_video(
            temp_path, len(video_bytes), motion_context, landmark_section
        )
        if result and "prediction" in result:
            result["landmark_frames"] = len(landmarks)
            return result
        print(f">>> Gemini video failed: {result}")

        return {
            "prediction":  "UNKNOWN",
            "confidence":  0.0,
            "explanation": f"All methods failed. Last: {result.get('error', 'unknown')}",
            "top3":        []
        }

    finally:
        try: os.unlink(temp_path)
        except: pass
        for p in keyframe_paths:
            try: os.unlink(p)
            except: pass
        if keyframe_paths:
            try: os.rmdir(os.path.dirname(keyframe_paths[0]))
            except: pass


# =============================================================
# PASS 1 · MOTION-STRIP DESCRIPTION
# =============================================================

def _get_motion_description(frame_paths: list) -> str:
    try:
        step        = max(1, len(frame_paths) // NUM_STRIP_STEPS)
        strip_frames = frame_paths[::step][:NUM_STRIP_STEPS]
        prompt_text  = ASL_MOTION_STRIP_PROMPT.format(num_frames=len(strip_frames))

        parts = [prompt_text]
        for path in strip_frames:
            with open(path, "rb") as f:
                img_bytes = f.read()
            parts.append(
                gemini_types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
            )

        resp = gemini_client.models.generate_content(
            model="gemini-2.0-flash", contents=parts
        )
        return resp.text.strip()
    except Exception as e:
        print(f">>> Motion strip failed (non-fatal): {e}")
        return ""


# =============================================================
# PASS 2 · GEMINI KEYFRAMES (main analysis)
# =============================================================

def analyze_with_gemini_keyframes(
    frame_paths:      list,
    video_size:       int,
    motion_context:   str = "",
    landmark_section: str = "",
) -> dict:
    try:
        base_prompt = ASL_KEYFRAME_PROMPT.format(
            num_frames=len(frame_paths),
            landmark_section=landmark_section
        )

        if motion_context:
            base_prompt = (
                f"── PRE-COMPUTED MOTION DESCRIPTION ──────────────────────────\n"
                f"{motion_context}\n"
                f"Use this as a starting hypothesis, but verify against the frames.\n\n"
            ) + base_prompt

        parts = [base_prompt]
        for i, path in enumerate(frame_paths):
            with open(path, "rb") as f:
                img_bytes = f.read()
            parts.append(f"Frame {i+1} of {len(frame_paths)}:")
            parts.append(
                gemini_types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
            )

        models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

        for model in models:
            try:
                print(f">>> Gemini keyframes: trying {model}...")
                resp = gemini_client.models.generate_content(
                    model=model,
                    contents=parts,
                    config=gemini_types.GenerateContentConfig(temperature=0.2)
                )
                print(f">>> Gemini keyframes: success with {model}")
                return parse_response(
                    resp.text, f"gemini/{model}", video_size, "gemini-keyframes"
                )
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    print(f">>> Rate limited on {model}, waiting 10s...")
                    time.sleep(10)
                elif "404" in err or "not found" in err.lower():
                    print(f">>> Model {model} not found, skipping...")
                elif "403" in err or "API_KEY" in err or "permission" in err.lower():
                    print(f"!!! AUTH ERROR: {err}")
                    return {"error": f"Gemini auth failed: {err}"}
                else:
                    print(f">>> {model} error: {err}")

        return {"error": "Gemini keyframes: all models failed"}
    except Exception as e:
        return {"error": f"Gemini keyframes error: {e}"}


# =============================================================
# PASS 3 · GEMINI VIDEO UPLOAD (fallback)
# =============================================================

def analyze_with_gemini_video(
    video_path:       str,
    video_size:       int,
    motion_context:   str = "",
    landmark_section: str = "",
) -> dict:
    try:
        print(">>> Gemini: uploading raw video...")
        uploaded = gemini_client.files.upload(file=video_path)
        waited   = 0
        while uploaded.state.name == "PROCESSING":
            time.sleep(2)
            waited += 2
            if waited > 60:
                return {"error": "Video processing timed out"}
            uploaded = gemini_client.files.get(name=uploaded.name)

        if uploaded.state.name == "FAILED":
            return {"error": "Gemini failed to process video"}

        prompt = ASL_VIDEO_PROMPT.format(landmark_section=landmark_section)
        if motion_context:
            prompt = (
                f"Pre-computed motion description: {motion_context}\n"
                f"Use as a hypothesis.\n\n"
            ) + prompt

        for model in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
            try:
                print(f">>> Gemini video: trying {model}...")
                resp = gemini_client.models.generate_content(
                    model=model,
                    contents=[uploaded, prompt],
                    config=gemini_types.GenerateContentConfig(temperature=0.2)
                )
                try:
                    gemini_client.files.delete(name=uploaded.name)
                except Exception:
                    pass
                print(f">>> Gemini video: success with {model}")
                return parse_response(
                    resp.text, f"gemini/{model}", video_size, "gemini-video"
                )
            except Exception as e:
                err = str(e)
                print(f">>> Gemini video {model} error: {err}")
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    time.sleep(10)
                elif "403" in err or "API_KEY" in err:
                    return {"error": f"Gemini auth failed: {err}"}

        return {"error": "Gemini video: all models failed"}
    except Exception as e:
        return {"error": f"Gemini video error: {e}"}


# =============================================================
# RESPONSE PARSER
# =============================================================

def parse_response(
    result_text: str, used_model: str, video_size: int, method: str
) -> dict:
    print(f">>> Raw response: {result_text[:400]}")

    cleaned = re.sub(r'```(?:json)?', '', result_text).strip()
    result  = None

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    if result is None:
        cleaned_single = re.sub(r'\s+', ' ', cleaned)
        match = re.search(r'\{.*\}', cleaned_single, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    if result is None:
        print(f"!!! Could not parse JSON from: {result_text[:200]}")
        word_match = re.search(r'"prediction"\s*:\s*"([A-Z]+)"', result_text)
        if word_match:
            result = {
                "prediction":  word_match.group(1),
                "confidence":  0.5,
                "explanation": result_text[:200],
                "top3":        []
            }
        else:
            return {"error": f"Could not parse AI response: {result_text[:100]}"}

    prediction  = str(result.get("prediction", "UNKNOWN")).upper()
    confidence  = min(1.0, max(0.0, float(result.get("confidence", 0.0))))
    top3        = result.get("top3", [])
    explanation = result.get("explanation", "")
    trajectory  = result.get("trajectory", "")

    if not top3:
        top3 = [{"label": prediction, "confidence": confidence}]

    print(f">>> RESULT: {prediction} ({confidence*100:.1f}%) via {method} [{used_model}]")
    if trajectory:
        print(f">>> Trajectory: {trajectory}")

    return {
        "prediction":  prediction,
        "confidence":  confidence,
        "top3":        top3,
        "trajectory":  trajectory,
        "explanation": explanation,
        "debug": {
            "model":      used_model,
            "method":     method,
            "video_size": video_size
        }
    }
