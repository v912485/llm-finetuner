from flask import Flask, jsonify
from flask_cors import CORS
from routes import model_routes, training_routes, dataset_routes, settings_routes
from config.settings import setup_logging
import os

app = Flask(__name__)

# Configure CORS with more restrictive settings
CORS(app, resources={
    r"/api/*": {
        "origins": os.environ.get("ALLOWED_ORIGINS", "*"),
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# Set up logging
logger = setup_logging()

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
