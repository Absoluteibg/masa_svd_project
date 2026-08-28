"""
run_api.py  —  Start the MASA-SVD FastAPI server.

Usage:  python scripts/run_api.py
Then open browser at:  http://localhost:8000/docs

The /docs page shows the full interactive Swagger UI.
You can test /verify directly from the browser — great for viva demo.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import uvicorn
from src.config import config

if __name__ == "__main__":
    print(f"\n[API] Starting MASA-SVD server on http://{config.API_HOST}:{config.API_PORT}")
    print(f"[API] Interactive docs → http://localhost:{config.API_PORT}/docs\n")
    uvicorn.run(
        "src.api.server:app",
        host    = config.API_HOST,
        port    = config.API_PORT,
        reload  = False,
        log_level = "info",
    )
    