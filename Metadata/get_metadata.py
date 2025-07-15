import os
import json
import time
from pathlib import Path
from fetching_url import fetch_raw_text

class URLContentProcessor:
    def __init__(self, url_directory="URL", output_file="metadata.json"):
        self.url_directory = url_directory
        self.output_file = output_file
        
    def read_urls_from_files(self):
        """Read all URLs from .txt files in the URL directory"""
        url_data = {}
        
        try:
            # Check if URL directory exists
            if not os.path.exists(self.url_directory):
                print(f"❌ Directory '{self.url_directory}' not found!")
                return url_data
            
            # Get all .txt files from URL directory
            txt_files = [f for f in os.listdir(self.url_directory) if f.endswith('.txt')]
            
            if not txt_files:
                print(f"❌ No .txt files found in '{self.url_directory}' directory!")
                return url_data
                
            print(f"📁 Found {len(txt_files)} .txt files: {txt_files}")
            
            for filename in txt_files:
                file_path = os.path.join(self.url_directory, filename)
                category = os.path.splitext(filename)[0]  # Remove .txt extension
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        urls = []
                        for line_num, line in enumerate(file, 1):
                            line = line.strip()
                            if line and not line.startswith('#'):  # Skip empty lines and comments
                                # Clean up the URL (remove quotes if present)
                                url = line.strip('"\'')
                                if url:
                                    urls.append(url)
                        
                        print(f"📄 {filename}: Found {len(urls)} URLs")
                        
                        # Store URLs with their category
                        for url in urls:
                            url_data[url] = {"category": category, "content": None}
                            
                except Exception as e:
                    print(f"❌ Error reading {filename}: {e}")
                    
        except Exception as e:
            print(f"❌ Error accessing directory: {e}")
            
        return url_data
    
    def fetch_content(self, url):
        """Fetch raw text content from a URL using the imported function"""
        try:
            print(f"🔄 Fetching: {url}")
            
            # Use the imported fetch_raw_text function
            content = fetch_raw_text(url)
            
            if content.startswith("Error:"):
                print(f"❌ {content}")
                return content
            else:
                print(f"✅ Successfully fetched content ({len(content)} characters)")
                return content
                
        except Exception as e:
            print(f"❌ Error processing {url}: {e}")
            return f"Error: Processing error - {str(e)}"
    
    def process_all_urls(self):
        """Process all URLs and return structured data"""
        print("🚀 Starting URL processing...")
        print("=" * 60)
        
        # Read URLs from files
        url_data = self.read_urls_from_files()
        
        if not url_data:
            print("❌ No URLs found to process!")
            return {}
        
        print(f"\n📊 Total URLs to process: {len(url_data)}")
        print("=" * 60)
        
        # Process each URL
        processed_count = 0
        for url, data in url_data.items():
            print(f"\n[{processed_count + 1}/{len(url_data)}] Processing URL from '{data['category']}.txt'")
            
            # Fetch content
            content = self.fetch_content(url)
            data["content"] = content
            
            processed_count += 1
            
            # Small delay to be respectful to servers
            time.sleep(0.5)
        
        print(f"\n✅ Completed processing {processed_count} URLs")
        return url_data
    
    def save_to_json(self, data):
        """Save processed data to JSON file"""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Data saved to '{self.output_file}'")
            print(f"📊 File size: {os.path.getsize(self.output_file)} bytes")
            
        except Exception as e:
            print(f"❌ Error saving to JSON: {e}")
    
    def run(self):
        """Main execution method"""
        print("🔥 URL Content Processor Started")
        print("=" * 60)
        
        # Process all URLs
        processed_data = self.process_all_urls()
        
        if processed_data:
            # Save to JSON
            print("\n" + "=" * 60)
            self.save_to_json(processed_data)
            
            # Print summary
            print("\n📈 Summary:")
            print("-" * 30)
            categories = {}
            for url, data in processed_data.items():
                category = data["category"]
                categories[category] = categories.get(category, 0) + 1
            
            for category, count in categories.items():
                print(f"{category}: {count} URLs")
            
            print(f"\nTotal: {len(processed_data)} URLs processed")
        else:
            print("❌ No data to save!")

def main():
    """Main function"""
    processor = URLContentProcessor()
    processor.run()

if __name__ == "__main__":
    main()