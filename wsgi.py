from main import app  # Adjust 'main' to the actual module name where your Flask 'app' is defined

if __name__ == "__main__":
    from waitress import serve
    serve(app, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
