import os
import sys
from dotenv import load_dotenv

# Load env file
load_dotenv()

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.search import route_and_expand_query, search_vector_chunks, ask_chatbot
from google import genai
from app.config import get_api_key

def main():
    api_key = get_api_key()
    client = genai.Client(api_key=api_key)
    
    query = "how to get wealthy"
    print(f"--- ROUTING AND EXPANDING QUERY: {query} ---")
    routing_data = route_and_expand_query(client, query)
    print("Routing result:", routing_data)
    
    category = routing_data.get("category", "essay_query")
    improved = routing_data.get("improved_query") or query
    
    print(f"\n--- SEARCHING FOR: {improved} ---")
    chunks = search_vector_chunks(client, improved, limit=6)
    print(f"Retrieved {len(chunks)} chunks:")
    for idx, c in enumerate(chunks):
        print(f"[{idx+1}] Title: {c['essay_title']}, URL: {c['essay_url']}")
        print(f"Content: {c['content'][:150]}...")
        print("-" * 40)
        
    print("\n--- GENERATING ANSWER ---")
    response = ask_chatbot(query)
    print("Answer:")
    print(response["answer"])
    print("\nSources:")
    print(response["sources"])

if __name__ == "__main__":
    main()
