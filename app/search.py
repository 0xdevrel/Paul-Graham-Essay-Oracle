import json
import logging
from google import genai
from google.genai import types
from app.config import EMBEDDING_MODEL, LLM_MODEL, get_api_key
from app.database import get_db_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("search")

def improve_query(client: genai.Client, raw_query: str) -> str:
    """Uses gemini-2.5-flash to rewrite and expand the query with terms 
    characteristic of Paul Graham's essays to maximize vector matching.
    """
    prompt = f"""You are an AI assistant designed to optimize search queries for a vector database of Paul Graham's essays.
Your task is to analyze the user's question, clarify their intent, and produce a list of search keywords, phrases, and concepts that are highly characteristic of Paul Graham's writing style and vocabulary.

Paul Graham frequently writes about topics like:
- "doing things that don't scale", "making something people want", "ramen profitable"
- "the cofounder relationship", "organic growth", "giving startup ideas"
- "hacker culture", "lisp", "wealth creation", "writing online"

User Question: "{raw_query}"

Provide a single, search-optimized search query. The output should be a plain-text paragraph or a combined string that blends the user's original intent with these style-specific concepts to maximize retrieval similarity. Do not use any markdown tags, bullet points, introduction, or formatting. Output only the search-optimized query string.
"""
    try:
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt
        )
        improved = response.text.strip()
        logger.info(f"Raw query: '{raw_query}' -> Improved query: '{improved}'")
        return improved
    except Exception as e:
        logger.warning(f"Failed to improve query (using raw instead): {e}")
        return raw_query

def search_vector_chunks(client: genai.Client, query: str, limit: int = 6) -> list[dict]:
    """Generates embedding for the query and retrieves the closest chunks.
    Uses sqlite-vec MATCH virtual table search if available, otherwise executes
    a pure-Python dot-product scan over all chunks in the database.
    """
    # 1. Generate query embedding
    try:
        emb_response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=query
        )
        query_vector = emb_response.embeddings[0].values
    except Exception as e:
        logger.error(f"Failed to generate query embedding: {e}")
        raise e
        
    from app.database import VEC_EXTENSION_AVAILABLE, get_db_connection, deserialize_vector
    
    if VEC_EXTENSION_AVAILABLE:
        logger.info("Using native sqlite-vec extension for MATCH vector search.")
        conn = get_db_connection()
        cursor = conn.cursor()
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
            ORDER BY v.distance
            LIMIT ?;
        """, (query_vector_json, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for r in rows:
            results.append({
                "chunk_id": r["chunk_id"],
                "content": r["content"],
                "chunk_index": r["chunk_index"],
                "essay_title": r["essay_title"],
                "essay_url": r["essay_url"],
                "distance": r["distance"]
            })
        return results
    else:
        logger.info("sqlite-vec not loaded. Executing fallback pure-Python dot-product vector scan.")
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Pull all chunks with their serialized vectors
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
        
        rows = cursor.fetchall()
        conn.close()
        
        candidates = []
        for r in rows:
            # Deserialize raw float bytes back to floats
            chunk_vector = deserialize_vector(r["embedding_blob"])
            
            # Compute dot product (since vectors from Gemini are normalized, this equals cosine similarity)
            similarity = sum(x * y for x, y in zip(chunk_vector, query_vector))
            
            # Convert similarity to distance format (smaller distance is a better match)
            distance = 1.0 - similarity
            
            candidates.append({
                "chunk_id": r["chunk_id"],
                "content": r["content"],
                "chunk_index": r["chunk_index"],
                "essay_title": r["essay_title"],
                "essay_url": r["essay_url"],
                "distance": distance
            })
            
        # Sort ascending by distance (closest matches first)
        candidates.sort(key=lambda x: x["distance"])
        return candidates[:limit]

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
        # We parse the text to see which citations [1], [2], etc. were actually used
        cited_sources = []
        for idx in sources_map:
            citation_str = f"[{idx}]"
            if citation_str in answer:
                # Add citation details
                item = sources_map[idx]
                # Avoid duplicates in final list
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

def classify_query(client: genai.Client, query: str) -> dict:
    """Classifies the user query into one of three categories:
    - 'greeting': conversational greetings (e.g. hello, hi, hey, how are you, who are you)
    - 'essay_query': questions related to startup ideas, doing great work, Paul Graham, tech, essays, etc.
    - 'out_of_scope': queries completely unrelated to Paul Graham's topics or writing (e.g. capital of France, math, general coding, recipe)
    
    Returns a dict with 'category' and 'reason'.
    """
    prompt = f"""Analyze the user's input and classify it into one of these three categories:
1. "greeting": Conversational greetings, introductions, or generic chit-chat (e.g., "hello", "hi", "hey", "who are you", "what is your name", "how's it going").
2. "essay_query": Explicit or implicit questions about startups, essay topics, entrepreneurship, technology, programming, career, hacking, writing, doing great work, or Paul Graham himself.
3. "out_of_scope": Specific factual or analytical questions completely unrelated to Paul Graham's writings, startups, or essays (e.g., "What is the capital of France?", "Write a python script to reverse a list", "How do I bake chocolate chip cookies?").

User input: "{query}"

You MUST respond with a JSON object containing two fields:
- "category": one of "greeting", "essay_query", or "out_of_scope"
- "reason": a brief 1-sentence explanation of why it was classified this way.
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
        logger.info(f"Query classification for '{query}': {data}")
        return data
    except Exception as e:
        logger.warning(f"Failed to classify query '{query}' (defaulting to 'essay_query'): {e}")
        return {"category": "essay_query", "reason": str(e)}

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
    
    # 1. Classify query intent to route or filter
    classification = classify_query(client, raw_query)
    category = classification.get("category", "essay_query")
    
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
    
    # 3. Standard RAG flow for essay_query
    # Expand query to match PG's narrative terminology
    improved = improve_query(client, raw_query)
    
    # Perform vector search in SQLite using sqlite-vec
    retrieved_chunks = search_vector_chunks(client, improved)
    
    # Generate answer grounded in the sources
    rag_response = generate_rag_answer(client, raw_query, retrieved_chunks)
    
    # Add improved query to response for UI visualization
    rag_response["improved_query"] = improved
    return rag_response
