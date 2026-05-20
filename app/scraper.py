import asyncio
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import logging
from app.database import get_db_connection, set_ingestion_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scraper")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

async def fetch_page(client: httpx.AsyncClient, url: str) -> str:
    """Fetches HTML content of a URL."""
    try:
        response = await client.get(url, headers=HEADERS, timeout=15.0)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        raise e

def extract_essay_links(index_html: str, base_url: str) -> list[dict]:
    """Parses index page to find all essay links and their titles."""
    soup = BeautifulSoup(index_html, "html.parser")
    links = []
    
    # Paul Graham's site links are typically inside table cells
    # We find all <a> tags whose href matches essay styles
    for a in soup.find_all("a"):
        href = a.get("href", "")
        # Filter out external links, indices, and administrative pages
        if not href or href.startswith("http") or href.startswith("https") or "/" in href:
            continue
        if href in ["index.html", "articles.html", "rss.html", "index.html", "search.html", "sub.html"]:
            continue
        if not href.endswith(".html"):
            continue
            
        title = a.get_text(strip=True)
        # Skip empty link texts or very short labels
        if not title or len(title) < 3 or title.lower() in ["index", "home", "essays", "rss", "contact", "about"]:
            continue
            
        full_url = urljoin(base_url, href)
        links.append({
            "title": title,
            "url": full_url,
            "filename": href
        })
        
    # Remove duplicates while preserving order
    seen = set()
    unique_links = []
    for link in links:
        if link["url"] not in seen:
            seen.add(link["url"])
            unique_links.append(link)
            
    return unique_links

def clean_essay_content(essay_html: str) -> tuple[str, str]:
    """Cleans up raw HTML to extract the clean essay title and text content."""
    soup = BeautifulSoup(essay_html, "html.parser")
    
    # 1. Extract Title
    title = ""
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)
    
    # 2. Extract and Clean Text
    # Paul Graham's essays are usually wrapped inside a large table, often in font tags
    # Let's find the main font tag if possible or default to body
    content_area = soup.find("body") or soup
    
    # Remove script and style elements
    for element in content_area(["script", "style", "nav", "footer"]):
        element.decompose()
        
    # Extract text with newlines
    raw_text = content_area.get_text(separator="\n")
    
    # Clean up formatting
    lines = []
    for line in raw_text.splitlines():
        cleaned_line = line.strip()
        if not cleaned_line:
            continue
        # Skip site headers/footers
        if cleaned_line in ["Home", "Essays", "RSS", "Search", "Index", "Contact"]:
            continue
        # Skip tiny lines of navigation delimiters
        if cleaned_line == "|" or cleaned_line == "•":
            continue
        lines.append(cleaned_line)
        
    # Join and format paragraphs
    # Try to group text back into paragraphs
    cleaned_content = "\n\n".join(lines)
    
    # Fallback title if title wasn't found or was generic
    if not title or title.lower() in ["essay", "paul graham"]:
        # Try using the first line
        if lines:
            title = lines[0]
        else:
            title = "Untitled Essay"
            
    return title, cleaned_content

async def scrape_and_save_essays():
    """Crawl, extract, clean, and store all essays into the SQLite database."""
    base_url = "https://paulgraham.com/"
    index_url = urljoin(base_url, "articles.html")
    
    try:
        set_ingestion_state("scraping", "5", "")
        logger.info("Starting scraping process...")
        
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            # 1. Fetch Index Page
            index_html = await fetch_page(client, index_url)
            essay_links = extract_essay_links(index_html, base_url)
            
            total_essays = len(essay_links)
            logger.info(f"Found {total_essays} essays to scrape.")
            
            if total_essays == 0:
                raise Exception("No essays found on Paul Graham's index page. Site structure may have changed.")
                
            # Connect to DB to save
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Semaphores and rate-limiting
            sem = asyncio.Semaphore(4) # Limit concurrent HTTP requests
            
            async def process_essay(essay_info, idx):
                async with sem:
                    url = essay_info["url"]
                    try:
                        # Fetch the essay
                        html = await fetch_page(client, url)
                        # Clean up HTML and extract text
                        title, content = clean_essay_content(html)
                        
                        # Use title from page, fallback to list title
                        final_title = title if len(title) > 5 else essay_info["title"]
                        
                        # Save to db
                        cursor.execute(
                            "INSERT OR REPLACE INTO essays (title, url, content) VALUES (?, ?, ?);",
                            (final_title, url, content)
                        )
                        conn.commit()
                        
                        # Calculate progress (5% to 50% for scraping)
                        progress_val = int(5 + (idx / total_essays) * 45)
                        set_ingestion_state("scraping", str(progress_val), "")
                        logger.info(f"[{idx}/{total_essays}] Scraped: {final_title}")
                        
                    except Exception as e:
                        logger.error(f"Failed to scrape {url}: {e}")
                    
                    # Sleep slightly between requests to be polite
                    await asyncio.sleep(0.2)
            
            # Run all tasks
            tasks = [process_essay(link, i+1) for i, link in enumerate(essay_links)]
            await asyncio.gather(*tasks)
            
            conn.close()
            logger.info("Scraping completed successfully.")
            set_ingestion_state("scraping_done", "50", "")
            
    except Exception as e:
        logger.error(f"Scraper encountered a critical error: {e}")
        set_ingestion_state("error", "0", str(e))
        raise e
