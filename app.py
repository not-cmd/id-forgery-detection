from flask import Flask, jsonify
from id_forgery_app import create_app

# This file is used by Render.com to find the Flask app
app = create_app()

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002) 