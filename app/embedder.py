import re
import json
import logging
import time
from google import genai
from app.config import EMBEDDING_MODEL, EMBEDDING_DIMENSION, CHUNK_SIZE, CHUNK_OVERLAP, get_api_key
from app.database import get_db_connection, set_ingestion_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embedder")

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Splits raw essay text into semantic overlapping passages aligned perfectly
    on complete sentence boundaries. Prevents sentence-slicing.
    """
    paragraphs = text.split("\n\n")
    sentences = []
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Split paragraph into sentences
        raw_sentences = re.split(r'(?<=[.!?])\s+', para)
        for s in raw_sentences:
            s = s.strip()
            if not s:
                continue
            
            # If a single sentence exceeds the chunk_size, split it at word boundaries
            if len(s) > chunk_size:
                words = s.split(" ")
                current_part = ""
                for w in words:
                    if len(current_part) + len(w) + 1 <= chunk_size:
                        current_part += (" " if current_part else "") + w
                    else:
                        if current_part:
                            sentences.append(current_part)
                        current_part = w
                if current_part:
                    sentences.append(current_part)
            else:
                sentences.append(s)
                
    chunks = []
    i = 0
    num_sentences = len(sentences)
    
    while i < num_sentences:
        chunk_sentences = []
        chunk_len = 0
        j = i
        
        # Add sentences until we exceed the chunk size
        while j < num_sentences:
            s = sentences[j]
            added_len = len(s) + (1 if chunk_len > 0 else 0)
            if chunk_len + added_len <= chunk_size:
                chunk_sentences.append(s)
                chunk_len += added_len
                j += 1
            else:
                break
                
        # Handle case where a sentence couldn't be added to a new chunk
        if not chunk_sentences:
            chunk_sentences.append(sentences[i])
            j = i + 1
            
        chunks.append(" ".join(chunk_sentences))
        
        if j == num_sentences:
            break
            
        # Determine the start of the next chunk with sentence-aligned overlap
        next_start = j
        overlap_len = 0
        k = j - 1
        
        while k >= i:
            s = sentences[k]
            added_overlap = len(s) + (1 if overlap_len > 0 else 0)
            if overlap_len + added_overlap <= overlap:
                overlap_len += added_overlap
                next_start = k
                k -= 1
            else:
                break
                
        # Ensure forward progress
        if next_start == j:
            next_start = j
            
        i = next_start
        
    return chunks

def generate_embeddings_batch(client: genai.Client, texts: list[str]) -> list[list[float]]:
    """Calls Google Gemini Embeddings API in a single batched request.
    Handles rate-limits (429 RESOURCE_EXHAUSTED) with robust retry backoff.
    """
    max_retries = 10
    base_delay = 10.0
    for attempt in range(max_retries):
        try:
            # gemini-embedding-2 with configured output dimension
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=[[t] for t in texts],
                config={'output_dimensionality': EMBEDDING_DIMENSION}
            )
            
            # Parse the embeddings
            embeddings = []
            for emb in response.embeddings:
                embeddings.append(emb.values)
            
            # Small delay between successful batches to be courteous to limits
            time.sleep(1.5)
            return embeddings
            
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                # Exponential backoff: 10s, 20s, 40s, 80s...
                delay = base_delay * (2.0 ** attempt)
                logger.warning(f"Gemini Embeddings API rate limit hit (429/Resource Exhausted). Retrying batch in {delay:.2f} seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
            else:
                logger.error(f"Error calling Embeddings API: {e}")
                raise e
                
    raise Exception("Max retries exceeded for generating embeddings due to Gemini API rate limits. Please try again in a few minutes.")

def create_embeddings_for_all_essays(force_rebuild: bool = False):
    """Fetches scraped essays, chunks them, generates vector embeddings, and saves them to SQLite.
    Supports incremental (resumable) indexing by default unless force_rebuild is True.
    """
    try:
        # Ensure API key is configured
        api_key = get_api_key()
        client = genai.Client(api_key=api_key)
        
        set_ingestion_state("embedding", "55", "")
        logger.info("Initializing vector indexing...")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        from app.database import VEC_EXTENSION_AVAILABLE, serialize_vector
        
        if force_rebuild:
            logger.info("force_rebuild is True. Clearing all existing vector chunks database tables...")
            cursor.execute("DELETE FROM chunks;")
            if VEC_EXTENSION_AVAILABLE:
                cursor.execute("DELETE FROM vec_chunks;")
            cursor.execute("DELETE FROM fts_chunks;")
            conn.commit()
            embedded_essay_ids = set()
        else:
            # Incremental: verify that each already embedded essay has all its chunks intact
            cursor.execute("SELECT essay_id, COUNT(*) as c FROM chunks GROUP BY essay_id;")
            db_counts = {row["essay_id"]: row["c"] for row in cursor.fetchall()}
            
            # Fetch all scraped essays to verify counts
            cursor.execute("SELECT id, title, content FROM essays;")
            essays = cursor.fetchall()
            
            embedded_essay_ids = set()
            partially_embedded_ids = set()
            
            for essay in essays:
                essay_id = essay["id"]
                if essay_id in db_counts:
                    # Calculate expected chunks
                    expected_count = len(chunk_text(essay["content"]))
                    actual_count = db_counts[essay_id]
                    if actual_count == expected_count:
                        embedded_essay_ids.add(essay_id)
                    else:
                        partially_embedded_ids.add(essay_id)
                        logger.info(f"Essay '{essay['title']}' was partially embedded ({actual_count}/{expected_count} chunks). Preparing to re-embed.")
            
            # Delete partial chunks to avoid duplicates
            if partially_embedded_ids:
                placeholder = ",".join("?" for _ in partially_embedded_ids)
                cursor.execute(f"DELETE FROM chunks WHERE essay_id IN ({placeholder});", tuple(partially_embedded_ids))
                if VEC_EXTENSION_AVAILABLE:
                    cursor.execute("DELETE FROM vec_chunks WHERE rowid NOT IN (SELECT id FROM chunks);")
                cursor.execute("DELETE FROM fts_chunks WHERE rowid NOT IN (SELECT id FROM chunks);")
                conn.commit()
                logger.info(f"Cleared partial database chunks for {len(partially_embedded_ids)} essays.")
                
            logger.info(f"Incremental mode: {len(embedded_essay_ids)} essays fully embedded. Skipping them.")
        
        # Retrieve all scraped essays (if not already fetched)
        if 'essays' not in locals():
            cursor.execute("SELECT id, title, content FROM essays;")
            essays = cursor.fetchall()
            
        total_essays = len(essays)
        if total_essays == 0:
            raise Exception("No essays found in database to embed. Please scrape essays first.")
            
        logger.info(f"Retrieved {total_essays} essays in total.")
        
        # Filter essays to embed
        essays_to_embed = [e for e in essays if e["id"] not in embedded_essay_ids]
        num_to_embed = len(essays_to_embed)
        
        if num_to_embed == 0:
            logger.info("All essays are already embedded! Vector database is fully up to date.")
            set_ingestion_state("done", "100", "")
            conn.close()
            return
            
        logger.info(f"{num_to_embed} essays need to be embedded. Commencing chunking...")
        
        # Create chunks only for the essays we need to embed
        all_chunks_to_embed = [] # list of dicts: {'essay_id', 'chunk_idx', 'content'}
        for essay in essays_to_embed:
            essay_chunks = chunk_text(essay["content"])
            for idx, text in enumerate(essay_chunks):
                word_count = len(text.split())
                all_chunks_to_embed.append({
                    "essay_id": essay["id"],
                    "chunk_index": idx,
                    "content": text,
                    "char_count": len(text),
                    "word_count": word_count
                })
                
        total_chunks = len(all_chunks_to_embed)
        logger.info(f"Total chunks to generate: {total_chunks}")
        
        # Batch Embed & Save Chunks
        # Embed in a smaller, safer batch size (20) to stay safely within TPM quotas
        batch_size = 20
        chunks_inserted = 0
        
        for i in range(0, total_chunks, batch_size):
            batch = all_chunks_to_embed[i:i + batch_size]
            batch_texts = [item["content"] for item in batch]
            
            logger.info(f"Embedding chunk batch {i} to {i + len(batch)} of {total_chunks}...")
            embeddings = generate_embeddings_batch(client, batch_texts)
            
            # Insert chunks and vectors in a single transaction
            for idx, item in enumerate(batch):
                embedding = embeddings[idx]
                
                # Serialize the embedding vector into a binary blob
                embedding_blob = serialize_vector(embedding)
                
                # First insert chunk metadata and binary vector to standard table
                cursor.execute("""
                INSERT INTO chunks (essay_id, chunk_index, content, char_count, word_count, embedding)
                VALUES (?, ?, ?, ?, ?, ?);
                """, (item["essay_id"], item["chunk_index"], item["content"], item["char_count"], item["word_count"], embedding_blob))
                
                chunk_id = cursor.lastrowid
                
                # If sqlite-vec is available, also load it into the virtual matching table
                if VEC_EXTENSION_AVAILABLE:
                    embedding_json = json.dumps(embedding)
                    cursor.execute("""
                    INSERT INTO vec_chunks (rowid, embedding)
                    VALUES (?, ?);
                    """, (chunk_id, embedding_json))
                    
                # Insert chunk into FTS5 text table for hybrid search
                cursor.execute("""
                INSERT INTO fts_chunks (rowid, chunk_id, content)
                VALUES (?, ?, ?);
                """, (chunk_id, chunk_id, item["content"]))
                
            conn.commit()
            chunks_inserted += len(batch)
            
            # Update ingestion state (55% to 95% for embedding phase)
            progress_val = int(55 + (chunks_inserted / total_chunks) * 40)
            set_ingestion_state("embedding", str(progress_val), "")
            
        conn.close()
        logger.info(f"Vector database built successfully! Loaded {chunks_inserted} new chunks.")
        set_ingestion_state("done", "100", "")
        
    except Exception as e:
        logger.error(f"Embedder encountered a critical error: {e}")
        set_ingestion_state("error", "0", str(e))
        raise e

