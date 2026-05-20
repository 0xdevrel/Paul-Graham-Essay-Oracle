import asyncio
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import threading
from pathlib import Path

from app.database import init_db, get_db_connection, get_ingestion_state, set_ingestion_state
from app.scraper import scrape_and_save_essays
from app.embedder import create_embeddings_for_all_essays
from app.search import ask_chatbot

app = FastAPI(title="Paul Graham Essay Chatbot", description="A vector-search RAG chatbot for PG essays.")

# Initialize database on startup
@app.on_event("startup")
def startup_event():
    init_db()
    
    # Reset stale ingestion status to idle on startup
    state = get_ingestion_state()
    if state.get("status") in ["starting", "scraping", "embedding"]:
        print("Stale ingestion status detected on startup. Resetting status to 'idle'.")
        set_ingestion_state("idle", "0", "")

# Mount the static files directory
STATIC_DIR = Path(__file__).resolve().parent / "static"
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Endpoint to serve index.html directly at root
@app.get("/")
def read_root():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend assets not found.")
    return FileResponse(index_file)

# Request/Response models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    improved_query: str
    sources: list[dict]

# Background Ingestion thread worker
def pipeline_worker():
    """Runs the ingestion pipeline synchronously inside a background thread."""
    try:
        # 1. Run Scraping
        asyncio.run(scrape_and_save_essays())
        # 2. Run Embedding
        create_embeddings_for_all_essays()
    except Exception as e:
        print(f"Background ingestion failed: {e}")
        set_ingestion_state("error", "0", str(e))

@app.post("/api/ingest")
def trigger_ingestion(background_tasks: BackgroundTasks):
    """Triggers the scraping and vector embedding ingestion in the background."""
    state = get_ingestion_state()
    if state["status"] in ["scraping", "embedding"]:
        return {"status": "running", "message": "Ingestion pipeline is already running."}
        
    # Reset status
    set_ingestion_state("starting", "0", "")
    
    # We spawn a thread to run the ingestion in the background safely
    thread = threading.Thread(target=pipeline_worker)
    thread.daemon = True
    thread.start()
    
    return {"status": "started", "message": "Paul Graham essay ingestion triggered successfully in the background."}

@app.get("/api/status")
def get_status():
    """Fetches the current indexing statistics and ingestion progress."""
    state = get_ingestion_state()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Fetch database counts
    cursor.execute("SELECT COUNT(*) as count FROM essays;")
    essay_count = cursor.fetchone()["count"]
    
    cursor.execute("SELECT COUNT(*) as count FROM chunks;")
    chunk_count = cursor.fetchone()["count"]
    
    conn.close()
    
    return {
        "status": state["status"],
        "progress": int(state["progress"]),
        "error": state["error"],
        "total_essays": essay_count,
        "total_chunks": chunk_count
    }

@app.get("/api/essays")
def list_essays():
    """Returns a list of all ingested essays with titles and URLs."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT title, url FROM essays ORDER BY title ASC;")
    rows = cursor.fetchall()
    conn.close()
    
    essays = [{"title": r["title"], "url": r["url"]} for r in rows]
    return {"essays": essays}

@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Answers a question grounded strictly in Paul Graham's essays."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    # Check if database has content
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM chunks;")
    chunk_count = cursor.fetchone()["count"]
    conn.close()
    
    if chunk_count == 0:
        return ChatResponse(
            answer="The knowledge base is currently empty. Please trigger 'Data Ingestion' from the sidebar to scrape and index Paul Graham's essays first!",
            improved_query=request.message,
            sources=[]
        )
        
    try:
        response = ask_chatbot(request.message)
        return ChatResponse(
            answer=response["answer"],
            improved_query=response["improved_query"],
            sources=response["sources"]
        )
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while running RAG pipeline: {str(e)}")

# Mount static folder (for styles/images/js)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
