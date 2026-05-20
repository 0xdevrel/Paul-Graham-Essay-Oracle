import re
import json
import logging
from google import genai
from google.genai import types
from app.config import EMBEDDING_MODEL, EMBEDDING_DIMENSION, LLM_MODEL, get_api_key
from app.database import get_db_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("search")

def clean_fts_query(query: str) -> str:
    """Cleans a query string to make it safe for SQLite FTS5 MATCH syntax,
    preventing syntax error crashes from arbitrary user inputs or punctuation.
    """
    # Keep alphanumeric characters and whitespace, remove punctuation
    cleaned = re.sub(r'[^\w\s]', ' ', query)
    words = cleaned.strip().split()
    # Double-quote each word for FTS5 syntax safety, joining them with OR
    safe_words = [f'"{w}"' for w in words if w]
    return " OR ".join(safe_words)

def route_and_expand_query(client: genai.Client, query: str) -> dict:
    """Classifies user intent and generates an improved query in a single-pass JSON-structured
    LLM roundtrip to shave off latency.
    
    Returns a dict with:
    - 'category': one of 'greeting', 'essay_query', or 'out_of_scope'
    - 'reason': brief 1-sentence reason
    - 'improved_query': rewritten query (meaningful if category is 'essay_query')
    """
    prompt = f"""Analyze the user's input to classify its intent AND generate a search-optimized query for a vector and full-text database of Paul Graham's essays.

Categories:
1. "greeting": Conversational greetings, introductions, or generic chit-chat (e.g., "hello", "hi", "hey", "who are you", "what is your name", "how's it going").
2. "essay_query": Explicit or implicit questions about startups, essay topics, entrepreneurship, technology, programming, career, hacking, writing, doing great work, or Paul Graham himself.
3. "out_of_scope": Specific factual or analytical questions completely unrelated to Paul Graham's writings, startups, or essays (e.g., "What is the capital of France?", "Write a python script to reverse a list", "How do I bake chocolate chip cookies?").

For "essay_query", you must also generate a search-optimized rewrite.
Paul Graham frequently writes about topics like:
- "doing things that don't scale", "making something people want", "ramen profitable"
- "the cofounder relationship", "organic growth", "giving startup ideas"
- "hacker culture", "lisp", "wealth creation", "writing online", "doing great work"

The improved query should be a plain-text paragraph or a combined string that blends the user's original intent with these style-specific concepts and keywords to maximize lexical and semantic retrieval similarity. Do not use markdown or bullet points.

User input: "{query}"

You MUST respond with a JSON object containing three fields:
- "category": "greeting", "essay_query", or "out_of_scope"
- "reason": "A brief explanation of the classification."
- "improved_query": "The rewritten, search-optimized query string, or null if the category is greeting/out_of_scope."
"""
    try:
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        data = json.loads(response.text.strip())
        logger.info(f"Route and expand query for '{query}': {data}")
        return data
    except Exception as e:
        logger.warning(f"Failed to route and expand query '{query}' (defaulting to essay_query): {e}")
        return {
            "category": "essay_query",
            "reason": str(e),
            "improved_query": query
        }

def search_vector_chunks(client: genai.Client, query: str, limit: int = 6) -> list[dict]:
    """Generates embedding for the query, retrieves candidates using native vector/fallback search
    and SQLite FTS5 lexical search, then merges them using Reciprocal Rank Fusion (RRF).
    """
    # 1. Generate query embedding
    try:
        emb_response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=query,
            config={'output_dimensionality': EMBEDDING_DIMENSION}
        )
        query_vector = emb_response.embeddings[0].values
    except Exception as e:
        logger.error(f"Failed to generate query embedding: {e}")
        raise e
        
    from app.database import VEC_EXTENSION_AVAILABLE, get_db_connection, deserialize_vector
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 2. Dense Vector Search (semantic, top 20 candidates)
    vector_results = []
    if VEC_EXTENSION_AVAILABLE:
        try:
            logger.info("Using native sqlite-vec extension for MATCH vector search (top 20).")
            query_vector_json = json.dumps(query_vector)
            cursor.execute("""
                SELECT 
                    c.id as chunk_id,
                    c.content as content,
                    c.chunk_index as chunk_index,
                    e.title as essay_title,
                    e.url as essay_url,
                    v.distance as distance
                FROM vec_chunks v
                JOIN chunks c ON v.rowid = c.id
                JOIN essays e ON c.essay_id = e.id
                WHERE v.embedding MATCH ?
                ORDER BY v.distance ASC
                LIMIT ?;
            """, (query_vector_json, 20))
            for r in cursor.fetchall():
                vector_results.append({
                    "chunk_id": r["chunk_id"],
                    "content": r["content"],
                    "chunk_index": r["chunk_index"],
                    "essay_title": r["essay_title"],
                    "essay_url": r["essay_url"],
                    "distance": r["distance"]
                })
        except Exception as e:
            logger.error(f"sqlite-vec MATCH query failed: {e}")
    
    # Fallback to pure-Python vector search if vector_results is empty
    if not vector_results:
        logger.info("sqlite-vec not loaded or failed. Executing fallback pure-Python dot-product vector scan (top 20).")
        try:
            cursor.execute("""
                SELECT 
                    c.id as chunk_id,
                    c.content as content,
                    c.chunk_index as chunk_index,
                    c.embedding as embedding_blob,
                    e.title as essay_title,
                    e.url as essay_url
                FROM chunks c
                JOIN essays e ON c.essay_id = e.id;
            """)
            all_chunks = cursor.fetchall()
            
            candidates = []
            for r in all_chunks:
                chunk_vector = deserialize_vector(r["embedding_blob"])
                similarity = sum(x * y for x, y in zip(chunk_vector, query_vector))
                distance = 1.0 - similarity
                candidates.append({
                    "chunk_id": r["chunk_id"],
                    "content": r["content"],
                    "chunk_index": r["chunk_index"],
                    "essay_title": r["essay_title"],
                    "essay_url": r["essay_url"],
                    "distance": distance
                })
            candidates.sort(key=lambda x: x["distance"])
            vector_results = candidates[:20]
        except Exception as e:
            logger.error(f"Fallback vector scan failed: {e}")
            
    # 3. FTS5 Lexical Search (top 20 candidates)
    fts_results = []
    cleaned_query = clean_fts_query(query)
    if cleaned_query:
        logger.info(f"Using SQLite FTS5 for lexical search (top 20) with query: {cleaned_query}")
        try:
            cursor.execute("""
                SELECT 
                    c.id as chunk_id,
                    c.content as content,
                    c.chunk_index as chunk_index,
                    e.title as essay_title,
                    e.url as essay_url,
                    bm25(fts_chunks) as score
                FROM fts_chunks f
                JOIN chunks c ON f.rowid = c.id
                JOIN essays e ON c.essay_id = e.id
                WHERE fts_chunks MATCH ?
                ORDER BY score ASC
                LIMIT ?;
            """, (cleaned_query, 20))
            for r in cursor.fetchall():
                fts_results.append({
                    "chunk_id": r["chunk_id"],
                    "content": r["content"],
                    "chunk_index": r["chunk_index"],
                    "essay_title": r["essay_title"],
                    "essay_url": r["essay_url"],
                    "score": r["score"]
                })
        except Exception as e:
            logger.warning(f"FTS5 MATCH failed: {e}")
            
    conn.close()
    
    # 4. Merge candidates using Reciprocal Rank Fusion (RRF)
    rrf_scores = {}
    chunk_details = {}
    
    # Process vector results
    for rank_idx, doc in enumerate(vector_results):
        rank = rank_idx + 1
        doc_id = doc["chunk_id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (60.0 + rank))
        chunk_details[doc_id] = doc
        
    # Process FTS5 results
    for rank_idx, doc in enumerate(fts_results):
        rank = rank_idx + 1
        doc_id = doc["chunk_id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (60.0 + rank))
        chunk_details[doc_id] = doc
        
    # Sort candidates by combined RRF score descending
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    # Retrieve top candidates up to limit
    fused_results = []
    for doc_id in sorted_doc_ids[:limit]:
        doc = chunk_details[doc_id]
        doc.pop("distance", None)
        doc.pop("score", None)
        fused_results.append(doc)
        
    logger.info(f"RRF Hybrid Search merged {len(vector_results)} dense and {len(fts_results)} lexical chunks into {len(fused_results)} final results.")
    return fused_results

def generate_rag_answer(client: genai.Client, raw_query: str, chunks: list[dict]) -> dict:
    """Takes the retrieved chunks and synthesizes a comprehensive response strictly grounded in them."""
    if not chunks:
        return {
            "answer": "I'm sorry, but I cannot find an answer to that in Paul Graham's essays. (No relevant documents retrieved)",
            "sources": []
        }
        
    # Format context for prompt
    context_blocks = []
    sources_map = {}
    
    for idx, chunk in enumerate(chunks):
        source_idx = idx + 1
        sources_map[source_idx] = {
            "title": chunk["essay_title"],
            "url": chunk["essay_url"]
        }
        context_blocks.append(f"--- SOURCE [{source_idx}]: {chunk['essay_title']} ---\n{chunk['content']}")
        
    context_str = "\n\n".join(context_blocks)
    
    prompt = f"""You are an expert chatbot designed to answer questions ONLY using Paul Graham's essays.
You are provided with a set of relevant passages (sources) from Paul Graham's essays. Your goal is to synthesize a detailed, highly accurate, and engaging answer to the user's question.

CRITICAL RULES:
1. Ground your answer strictly in the provided sources. Do not make assumptions, use outside knowledge, or extrapolate.
2. If the provided sources do not contain enough information to answer the question, or if the question is unrelated to Paul Graham's writings, you MUST respond exactly: "I'm sorry, but I cannot find an answer to that in Paul Graham's essays." Do not explain why, just output that exact sentence.
3. Use inline numeric citations (e.g. [1], [2]) throughout your response to show exactly where the information came from.
4. Do not include a "Sources" list or bibliography at the end of your answer content. We will display the sources separately in the UI.

Provided Essay Passages:
{context_str}

User's Question: {raw_query}

Answer:
"""
    try:
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt
        )
        answer = response.text.strip()
        
        # Determine unique cited sources
        cited_sources = []
        for idx in sources_map:
            citation_str = f"[{idx}]"
            if citation_str in answer:
                item = sources_map[idx]
                if item not in cited_sources:
                    cited_sources.append(item)
                    
        # If no citations were used, but the answer says it can't find it, ensure we empty the sources list
        if "cannot find an answer to that" in answer.lower():
            cited_sources = []
            
        return {
            "answer": answer,
            "sources": cited_sources
        }
        
    except Exception as e:
        logger.error(f"Error during RAG generation: {e}")
        raise e

def generate_greeting_response(client: genai.Client, raw_query: str) -> str:
    """Generates a warm, professional, premium greeting introducing the Paul Graham Essay Assistant."""
    prompt = f"""You are a premium, intelligent AI assistant designed exclusively to help users explore the essays of Paul Graham.
The user sent a conversational greeting or generic inquiry: "{raw_query}"

Provide a warm, engaging, and professional response. 
Briefly introduce yourself as the PG Essay AI assistant. Inform them that you are ready to answer any questions about startups, entrepreneurship, hacking, doing great work, Lisp, or general wisdom from Paul Graham's writings.
Mention a few example topics they could ask about, such as:
- "How do you get startup ideas?"
- "What does it mean to do great work?"
- "Why should founders do things that don't scale?"

Keep the response concise, elegant, and beautifully formatted in markdown.
"""
    try:
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        logger.warning(f"Failed to generate custom greeting, returning default: {e}")
        return "Hello! I am your Paul Graham Essay AI assistant. I can answer questions grounded strictly in Paul Graham's essays. How can I help you explore his ideas today?"

def generate_out_of_scope_response(client: genai.Client, raw_query: str) -> str:
    """Generates a polite refusal indicating that the question is outside the scope of Paul Graham's essays."""
    prompt = f"""You are an AI assistant designed exclusively to answer questions about Paul Graham's essays.
The user asked a question that is out of scope or unrelated to Paul Graham's essays: "{raw_query}"

Write a polite, elegant response explaining that your knowledge base is strictly limited to Paul Graham's essays. Explain that you cannot answer this specific question but would be happy to help them explore startup ideas, hacker culture, doing great work, or other topics discussed in Paul's writing.
Keep it brief and professional.
"""
    try:
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        logger.warning(f"Failed to generate custom out-of-scope response, returning default: {e}")
        return "I'm sorry, but I cannot find an answer to that in Paul Graham's essays. I am specialized in answering questions about startups, hacking, and essays written by Paul Graham."

def ask_chatbot(raw_query: str) -> dict:
    """Executes full RAG workflow: expand query, fetch matching vectors, and generate grounded answer."""
    api_key = get_api_key()
    client = genai.Client(api_key=api_key)
    
    # 1. Single-pass query routing and expansion
    routing_data = route_and_expand_query(client, raw_query)
    category = routing_data.get("category", "essay_query")
    improved = routing_data.get("improved_query") or raw_query
    
    if category == "greeting":
        answer = generate_greeting_response(client, raw_query)
        return {
            "answer": answer,
            "sources": [],
            "improved_query": "[Bypassed Search - Greeting]"
        }
        
    elif category == "out_of_scope":
        answer = generate_out_of_scope_response(client, raw_query)
        return {
            "answer": answer,
            "sources": [],
            "improved_query": "[Bypassed Search - Out of Scope]"
        }
    
    # 2. Hybrid search (Dense vector + FTS5) with RRF
    retrieved_chunks = search_vector_chunks(client, improved)
    
    # 3. Grounded generation
    rag_response = generate_rag_answer(client, raw_query, retrieved_chunks)
    rag_response["improved_query"] = improved
    
    return rag_response
