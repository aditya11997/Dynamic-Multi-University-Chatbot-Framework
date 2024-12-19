from flask import Flask
from flask_cors import CORS
from routes.routes import routes  # Import the Blueprint
from src.config import BotConfig
import os

app = Flask(__name__, static_folder='static', static_url_path='/')
CORS(app, resources={r"/*": {"origins": "*"}})

# Register the Blueprint
app.register_blueprint(routes)

if __name__ == "__main__":
    config = BotConfig()
    app.run(host="0.0.0.0", port=os.getenv("PORT", 5000), debug=True)
