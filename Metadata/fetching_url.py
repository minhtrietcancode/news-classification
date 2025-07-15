import requests
from bs4 import BeautifulSoup

def fetch_raw_text(url):
    """
    Simple function to fetch raw text from any URL
    """
    try:
        # Add https if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Fetch the page
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML and extract raw text
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get raw text
        raw_text = soup.get_text()
        
        # Clean up extra whitespace
        lines = (line.strip() for line in raw_text.splitlines())
        text = '\n'.join(line for line in lines if line)
        
        return text
        
    except Exception as e:
        return f"Error: {e}"

# Usage
if __name__ == "__main__":
    # Your test URL
    test_url = "https://www.geeksforgeeks.org/computer-vision/backpropagation-in-convolutional-neural-networks/"
    
    # You can also use it with any other URL
    # test_url = "https://example.com"
    
    print(f"Fetching raw text from: {test_url}")
    print("=" * 50)
    
    raw_text = fetch_raw_text(test_url)
    print(raw_text)