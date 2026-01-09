# Procfile for Render and other PaaS platforms (Heroku, Railway, etc.)
# 
# This file defines how to run the application in production

# Main web process - runs the Flask app with Gunicorn
web: cd backend && gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120 --access-logfile - --error-logfile -

# Optional: Background worker process (uncomment if needed)
# worker: python scripts/background_worker.py
