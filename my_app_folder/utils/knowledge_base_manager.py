from flask import Flask, request, jsonify
from src.data_load import KnowledgeBaseManager  # Assuming your class is in data_load.py
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

@app.route('/register-university', methods=['POST'])
def register_university():
    try:
        data = request.json
        university_name = data.get("university_name")
        sitemap_url = data.get("sitemap_url")

        if not university_name or not sitemap_url:
            return jsonify({"error": "University name and sitemap URL are required"}), 400

        manager = KnowledgeBaseManager(university_name, sitemap_url)

        # Fetch and process URLs
        urls = manager.fetch_urls_from_sitemap()
        if not urls:
            return jsonify({"error": "Failed to fetch URLs from sitemap"}), 400

        manager.process_urls_in_batches(urls)

        return jsonify({"message": f"{university_name} has been successfully registered!"}), 200
    except Exception as e:
        logging.error(f"Error registering university: {e}")
        return jsonify({"error": "An error occurred during registration. Please try again later."}), 500

if __name__ == "__main__":
    app.run(debug=True)
