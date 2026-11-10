import os
import sys
import time
from collections import defaultdict
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS

# Ensure current directory is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

import engine

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1 MB max payload to prevent DoS

# In-memory sliding window rate limiter (max 60 requests/minute per client IP)
RATE_LIMIT_WINDOW = 60
MAX_REQUESTS_PER_WINDOW = 60
request_history = defaultdict(list)

def is_rate_limited(client_ip):
    now = time.time()
    history = request_history[client_ip]
    # Prune timestamps older than window
    request_history[client_ip] = [t for t in history if now - t < RATE_LIMIT_WINDOW]
    if len(request_history[client_ip]) >= MAX_REQUESTS_PER_WINDOW:
        return True
    request_history[client_ip].append(now)
    return False

# Security Headers Middleware
@app.after_request
def apply_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "font-src 'self' https://fonts.gstatic.com;"
    )
    return response

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/examples", methods=["GET"])
def get_examples():
    """Return the 4 reference EEG examples (2 Seizure, 2 Non-Seizure)."""
    return jsonify({
        "status": "success",
        "count": len(engine.REFERENCE_EXAMPLES),
        "examples": engine.REFERENCE_EXAMPLES
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    """Secure endpoint for EEG sequence seizure detection."""
    client_ip = request.remote_addr or "127.0.0.1"
    if is_rate_limited(client_ip):
        return jsonify({"error": "Rate limit exceeded. Please wait a moment."}), 429

    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json."}), 400

    try:
        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({"error": "Invalid or empty JSON body."}), 400

        raw_sequence = payload.get("sequence")
        if raw_sequence is None:
            return jsonify({"error": "Missing 'sequence' field in JSON payload."}), 400

        # Validate structure and sanitize
        if not isinstance(raw_sequence, (list, tuple)):
            return jsonify({"error": "'sequence' must be a numeric list or 2D array."}), 400

        # Recursively sanitize and convert to numpy float32
        import numpy as np
        try:
            arr = np.array(raw_sequence, dtype=np.float32)
        except (ValueError, TypeError):
            return jsonify({"error": "All elements in 'sequence' must be numeric (floats or ints)."}), 400

        # Sanitize non-finite values (NaN, Inf)
        if not np.all(np.isfinite(arr)):
            return jsonify({"error": "Sequence contains invalid values (NaN or Infinity)."}), 400

        # Sanity check magnitude to prevent extreme numeric overflow
        if np.any(np.abs(arr) > 10000.0):
            return jsonify({"error": "Input signal amplitude exceeds safety threshold (|value| > 10,000)."}), 400

        # Run inference
        result = engine.run_inference(arr)
        result["status"] = "success"
        return jsonify(result)

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        # Secure logging without leaking system internals
        print(f"[Error in /api/predict]: {e}")
        return jsonify({"error": "Internal processing error occurred while evaluating sequence."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting TinyML Seizure Detection Web Server on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
