"""
WSGI Configuration for PythonAnywhere Deployment.

This file is used by PythonAnywhere to serve the Flask application.
Copy the contents of this file to your PythonAnywhere WSGI configuration file.

Instructions:
1. Go to Web tab on PythonAnywhere
2. Click on the WSGI configuration file link
3. Replace the contents with this file's contents
4. Update 'yourusername' to your actual PythonAnywhere username
5. Reload the web app
"""

import sys
import os
from pathlib import Path

# =============================================================================
# CONFIGURATION - Update these paths with your PythonAnywhere username
# =============================================================================

# Replace 'yourusername' with your actual PythonAnywhere username
PYTHONANYWHERE_USERNAME = 'yourusername'

# Project paths
PROJECT_ROOT = f'/home/{PYTHONANYWHERE_USERNAME}/Real-Time-Sales-Analytics-Dashboard'
BACKEND_DIR = f'{PROJECT_ROOT}/backend'

# =============================================================================
# PATH SETUP
# =============================================================================

# Add project directories to Python path
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================

# Set production environment variables
os.environ.setdefault('FLASK_ENV', 'production')
os.environ.setdefault('SECRET_KEY', 'change-this-to-a-secure-random-string-in-production')

# Disable debug mode in production
os.environ.setdefault('DEBUG', 'False')

# =============================================================================
# LOGGING SETUP (Optional - helps with debugging)
# =============================================================================

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{PROJECT_ROOT}/logs/pythonanywhere.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Starting PythonAnywhere WSGI application...")

# =============================================================================
# FLASK APPLICATION IMPORT
# =============================================================================

try:
    # Import the Flask application
    from app import create_app
    
    # Create the application instance for WSGI
    application = create_app()
    
    # Configure for production
    application.config['DEBUG'] = False
    application.config['TESTING'] = False
    
    logger.info("Flask application loaded successfully!")
    
except Exception as e:
    logger.error(f"Failed to load Flask application: {e}")
    import traceback
    traceback.print_exc()
    raise

# =============================================================================
# NOTES FOR PYTHONANYWHERE
# =============================================================================
"""
STATIC FILES CONFIGURATION:
In the Web tab, add these static file mappings:

URL                     Directory
/static/               /home/yourusername/Real-Time-Sales-Analytics-Dashboard/frontend/static

VIRTUAL ENVIRONMENT:
Set the virtualenv path to:
/home/yourusername/.virtualenvs/salesenv

SCHEDULED TASKS (for background jobs):
Since PythonAnywhere free tier doesn't support always-on tasks,
use the Scheduled Tasks feature for periodic jobs like:
- Data refresh
- Model retraining
- Insights generation

Create a script like 'scheduled_tasks.py' and schedule it to run periodically.
"""
