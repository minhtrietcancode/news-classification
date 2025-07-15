geekforgeek_seed_url = "https://www.geeksforgeeks.org/machine-learning/machine-learning/"
medium_seed_url = "https://medium.com/tag/programming"
tech_seed_url = "https://medium.com/tag/technology"
ai_seed_url = "https://medium.com/tag/artificial-intelligence"

seed_urls = [
    geekforgeek_seed_url,
    medium_seed_url,
    tech_seed_url,
    ai_seed_url
]

# Import the defined crawling function
from crawler import crawl_10_urls

# Open file to write all URLs
with open('technology.txt', 'w') as f:
    # Now loops through each seed url and performing the search, then save to a file  
    for url in seed_urls:
        print(f"Crawling: {url}")
        crawled_urls = crawl_10_urls(url)
        
        # Write each URL to the file
        for crawled_url in crawled_urls:
            f.write(crawled_url + '\n')
        
        print(f"Added {len(crawled_urls)} URLs to technology.txt")

print("All URLs saved to technology.txt")