import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import time

def is_valid_url(url):
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)

def get_domain(url):
    return urlparse(url).netloc

def extract_links(html, base_url, domain):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for tag in soup.find_all("a", href=True):
        href = tag.get("href")
        joined = urljoin(base_url, href)
        if is_valid_url(joined) and get_domain(joined) == domain:
            links.add(joined)
    return links

def crawl_10_urls(seed_url, debug=False):
    """
    Recursively crawl starting from seed_url until collecting 10 unique URLs

    Args:
        seed_url (str): Starting URL for the crawl
        debug (bool): Print debug information

    Returns:
        list: List of up to 10 unique URLs discovered during crawling
    """
    visited = set()
    to_visit = [seed_url]
    domain = get_domain(seed_url)
    
    if debug:
        print(f"Starting crawl for domain: {domain}")
        print(f"Seed URL: {seed_url}")

    # Headers to mimic a real browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    while to_visit and len(visited) < 10:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue
            
        if debug:
            print(f"\nTrying to visit: {current_url}")
            
        try:
            resp = requests.get(current_url, timeout=10, headers=headers)
            if debug:
                print(f"Response status: {resp.status_code}")
                
            if resp.status_code != 200:
                if debug:
                    print(f"Skipping due to status code: {resp.status_code}")
                continue
                
            visited.add(current_url)
            if debug:
                print(f"Added to visited. Total visited: {len(visited)}")
                
            if len(visited) >= 10:
                break
                
            links = extract_links(resp.text, current_url, domain)
            if debug:
                print(f"Found {len(links)} links on this page")
                
            # Add new links to visit queue
            new_links_added = 0
            for link in links:
                if link not in visited and link not in to_visit:
                    to_visit.append(link)
                    new_links_added += 1
                    if len(visited) + len(to_visit) >= 50:  # Prevent queue from getting too large
                        break
                        
            if debug:
                print(f"Added {new_links_added} new links to queue")
                print(f"Queue size: {len(to_visit)}")
                
            # Add small delay to be respectful
            time.sleep(0.5)
            
        except Exception as e:
            if debug:
                print(f"Error accessing {current_url}: {str(e)}")
            continue
            
    if debug:
        print(f"\nCrawling complete. Total URLs found: {len(visited)}")
        
    return list(visited)

# # Test with debug enabled
# test_url = "https://www.bbc.com/sport"
# print("Testing with debug enabled:")
# test = crawl_10_urls(test_url, debug=True)
# print(f"\nFinal result: {test}")
# print(f"Total URLs found: {len(test)}")

# # Also test with a simpler site if BBC doesn't work
# print("\n" + "="*50)
# print("Testing with a simpler site:")
# simple_test = crawl_10_urls("https://httpbin.org/links/5/0", debug=True)
# print(f"\nSimple test result: {simple_test}")
# print(f"Total URLs found: {len(simple_test)}")