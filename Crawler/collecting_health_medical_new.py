url1 = "https://www.healthline.com/health-news"
url2 = "https://www.healthline.com/nutrition"
url3 = "https://www.healthline.com/sleep"
url4 = "https://www.healthline.com/cancer-care"


seed_urls = [
    url1,
    url2,
    url3,
    url4,
]

# Import the defined crawling function
from crawler import crawl_10_urls

# Open file to write all URLs
with open('health_medical.txt', 'w') as f:
    # Now loops through each seed url and performing the search, then save to a file  
    for url in seed_urls:
        print(f"Crawling: {url}")
        crawled_urls = crawl_10_urls(url)
        
        # Write each URL to the file
        for crawled_url in crawled_urls:
            f.write(crawled_url + '\n')
        
        print(f"Added {len(crawled_urls)} URLs to health_medical.txt")

print("All URLs saved to health_medical.txt")