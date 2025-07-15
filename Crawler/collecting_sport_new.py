bbc_sport_seed_url = "https://www.bbc.com/sport"
espn_seed_url = "https://www.espn.com/"
sky_sport_seed_url = "https://www.skysports.com/"
cnn_sport_seed_url = "https://edition.cnn.com/sport"

seed_urls = [
    bbc_sport_seed_url,
    espn_seed_url,
    sky_sport_seed_url,
    cnn_sport_seed_url
]

# Import the defined crawling function
from crawler import crawl_10_urls

# Open file to write all URLs
with open('sport.txt', 'w') as f:
    # Now loops through each seed url and performing the search, then save to a file  
    for url in seed_urls:
        print(f"Crawling: {url}")
        crawled_urls = crawl_10_urls(url)
        
        # Write each URL to the file
        for crawled_url in crawled_urls:
            f.write(crawled_url + '\n')
        
        print(f"Added {len(crawled_urls)} URLs to sport.txt")

print("All URLs saved to sport.txt")