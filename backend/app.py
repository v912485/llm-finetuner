from flask import Flask
from flask_cors import CORS
from routes import model_routes, training_routes, dataset_routes, settings_routes
from config.settings import setup_logging

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
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

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0") 
