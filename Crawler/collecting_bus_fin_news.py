url1 = "https://www.cnbc.com/finance/"
url2 = "https://www.cnbc.com/cnbc-pro-chart-investing/"
url3 = "https://www.cnbc.com/investing/"
url4 = "https://www.cnbc.com/financial-advisors/"

seed_urls = [
   url1,
   url2,
   url3,
   url4,
]

# Import the defined crawling function
from crawler import crawl_10_urls

# Open file to write all URLs
with open('bus_fin.txt', 'w') as f:
    # Now loops through each seed url and performing the search, then save to a file  
    for url in seed_urls:
        print(f"Crawling: {url}")
        crawled_urls = crawl_10_urls(url)
        
        # Write each URL to the file
        for crawled_url in crawled_urls:
            f.write(crawled_url + '\n')
        
        print(f"Added {len(crawled_urls)} URLs to bus_fin.txt")

print("All URLs saved to bus_fin.txt")