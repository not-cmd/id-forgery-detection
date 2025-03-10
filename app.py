from id_forgery_app import app

# This file is used by Render.com to find the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000) 