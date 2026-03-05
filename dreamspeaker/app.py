import os
import logging
import datetime
import tempfile
import threading
from pathlib import Path

from flask import Flask, request, jsonify

from dreamspeaker.util import load_keys
from dreamspeaker.transcriber import DreamTranscriber
from dreamspeaker.claude_dream import ClaudeDreamCorrection
from dreamspeaker.notion_poster import NotionDreamPoster

# ──── Colored logging ────

class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG:    "\033[36m",   # cyan
        logging.INFO:     "\033[32m",   # green
        logging.WARNING:  "\033[33m",   # yellow
        logging.ERROR:    "\033[31m",   # red
        logging.CRITICAL: "\033[1;31m", # bold red
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        record.msg = f"{color}{record.msg}{self.RESET}"
        return super().format(record)


logger = logging.getLogger("dreamspeaker")
logger.setLevel(logging.DEBUG)

console = logging.StreamHandler()
console.setFormatter(ColorFormatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(console)

# ──── Setup ────

project_root = Path(__file__).parent.parent
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)

NOTION_TOKEN, DATABASE_ID, CLAUDE_KEY, API_KEY = load_keys(project_root / "keys.env")

transcriber = DreamTranscriber()  # lazy-loads model on first transcription, unloads after 5 min

claude = ClaudeDreamCorrection(api_key=CLAUDE_KEY)
notion = NotionDreamPoster(notion_token=NOTION_TOKEN, database_id=DATABASE_ID)

# ──── Flask app ────

app = Flask(__name__)


def _file_logger(log_file):
    """Returns a log function that writes to both the file logger and the console logger."""
    def log(msg):
        logger.info(msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    return log


def process_dream(audio_path, log_file):
    """Background pipeline: transcribe → Claude → Notion → cleanup."""
    log = _file_logger(log_file)
    try:
        # 1. Transcribe
        log("Transcribing audio...")
        transcription = transcriber.transcribe(audio_path, log=log)
        log(f"Transcription ({len(transcription)} chars): {transcription[:200]}...")

        if not transcription.strip():
            log("Empty transcription, skipping.")
            return

        # 2. Claude
        log("Sending to Claude for formatting...")
        result = claude.correct_and_title(transcription, log=log)
        title = result["dream_title"]
        text = result["dream_text"]
        emoji = result["dream_emoji"]
        log(f"Title: {emoji} {title}")

        # 3. Notion
        log("Posting to Notion...")
        notion.post_dream(title, text, transcription, emoji, log=log)

        log("Pipeline complete!")
    except Exception as e:
        log(f"ERROR in pipeline: {e}")
    finally:
        # Cleanup temp audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)


@app.route("/upload", methods=["POST"])
def upload():
    # Auth check
    key = request.headers.get("X-API-Key")
    if key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    # Get audio file
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio = request.files["audio"]
    if audio.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save to temp file
    suffix = Path(audio.filename).suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(log_dir))
    audio.save(tmp.name)
    tmp.close()

    # Log file for this dream
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"dream_{timestamp}.txt"
    with open(log_file, "w") as f:
        f.write(f"Dream session started at {timestamp}\n")

    logger.info(f"Received audio: {audio.filename} → {tmp.name}")

    # Spawn background thread
    thread = threading.Thread(target=process_dream, args=(tmp.name, log_file))
    thread.start()

    return jsonify({"status": "ok", "message": "Audio received, processing started"}), 200


def run():
    app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    run()
