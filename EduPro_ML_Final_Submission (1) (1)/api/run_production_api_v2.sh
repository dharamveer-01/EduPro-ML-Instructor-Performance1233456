
#!/bin/bash

export EDUPRO_BASE_DIR="/content/edupro_ml_system"
export EDUPRO_API_KEY="EDUPRO-DEMO-API-KEY-2026"

cd /content/edupro_ml_system/api

uvicorn production_api_v2:app \
    --host 0.0.0.0 \
    --port 8000
