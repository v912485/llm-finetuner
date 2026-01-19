from flask import Flask, jsonify, request
from flask_cors import CORS
from routes import model_routes, training_routes, dataset_routes, settings_routes
from config.settings import setup_logging
import os
from utils.auth import is_request_authenticated, is_admin_configured
import re

app = Flask(__name__)

def _parse_allowed_origins():
    raw = os.environ.get("ALLOWED_ORIGINS", "")
    if not raw:
        return [
            re.compile(r"^http://localhost(:\d+)?$"),
            re.compile(r"^http://127\.0\.0\.1(:\d+)?$"),
        ]
    return [o.strip() for o in raw.split(",") if o.strip()]


# Configure CORS (LAN-safe default: localhost only unless ALLOWED_ORIGINS is set)
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": _parse_allowed_origins(),
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "supports_credentials": False,
        }
    },
)

# Set up logging
logger = setup_logging()

# Require auth for all API routes (except health and preflight)
@app.before_request
def _require_admin_token():
    if request.method == "OPTIONS":
        return None
    if request.path == "/api/settings/admin_token":
        if request.method == "GET":
            return None
        if request.method == "POST" and not is_admin_configured():
            return None
    if request.path == "/api/settings/huggingface_token":
        if request.method == "GET":
            return None
        if request.method == "POST" and not is_admin_configured():
            return None
    if request.path == "/api/health":
        return None
    if not request.path.startswith("/api/"):
        return None
    if is_request_authenticated():
        return None
    return jsonify({"status": "error", "message": "Unauthorized"}), 401

# Register routes
app.register_blueprint(model_routes.bp)
app.register_blueprint(training_routes.bp)
app.register_blueprint(dataset_routes.bp)
app.register_blueprint(settings_routes.bp)

# Add global error handler
@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify({
        "status": "error",
        "message": "An unexpected error occurred"
    }), 500

# Add health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "success",
        "message": "Service is running"
    })

if __name__ == '__main__':
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 5000))
    
    # Get debug setting from environment
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    
    # Run the app
    app.run(debug=debug, host="0.0.0.0", port=port) 
