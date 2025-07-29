import os
from fastapi import FastAPI
from src.api_v1 import router as api_v1_router

app = FastAPI()
app.include_router(api_v1_router)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}...")
    uvicorn.run("src.main:app", host="0.0.0.0", port=port, reload=True)
