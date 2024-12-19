import os  # Used for file and directory operations.
import pickle  # Used for saving and loading data.
import requests  # Used for HTTP requests to fetch sitemap and content.
from urllib.parse import urlparse  # used for URL validation and joining.
from bs4 import BeautifulSoup  # Used for parsing XML sitemap.
from dotenv import load_dotenv  # Used for loading environment variables.
import logging  # Used for logging information and errors.
from langchain.schema import Document
from langchain_community.document_loaders import UnstructuredURLLoader  # Used to load data from URLs.


class KnowledgeBaseManager:
    def __init__(self, university_name, sitemap_url, output_dir="data"):
        """
        Initialize KnowledgeBaseManager for a specific university.

        Args:
            university_name (str): The name of the university.
            sitemap_url (str): URL of the university's sitemap.
            output_dir (str): Directory to store data files.
        """
        load_dotenv()
        self.university_name = university_name.lower().replace(" ", "_")
        self.university_name = university_name.lower().replace("-", "_")
        self.sitemap_url = sitemap_url
        self.output_dir = os.path.abspath(output_dir)
        self.data_file = os.path.join(self.output_dir, f"{self.university_name}_data.pkl")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        logging.basicConfig(level=logging.INFO)
        logging.info(f"Initialized KnowledgeBaseManager for {university_name}")

    def is_valid_url(self, url):
        """Check if a URL is valid."""
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)
    
    def fetch_urls_from_html(self, html_content, base_url):
        """Extract URLs from an HTML document."""
        soup = BeautifulSoup(html_content, "html.parser")
        urls = []
        for link in soup.find_all("a", href=True):
            url = link["href"]
            if not urlparse(url).netloc:  # If URL is relative, make it absolute
                url = requests.compat.urljoin(base_url, url)
            if self.is_valid_url(url):
                urls.append(url)
        return urls

    def fetch_urls_from_sitemap_or_html(self, visited=None):
        logging.info(f"Requesting URL: {self.sitemap_url}")
        """Fetch URLs from the sitemap."""
        if visited is None:
            visited = set()

        # Avoid processing the same sitemap multiple times
        if self.sitemap_url in visited:
            logging.warning(f"Sitemap {self.sitemap_url} has already been visited. Skipping.")
            return []

        visited.add(self.sitemap_url)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        }
        try:
            response = requests.get(self.sitemap_url, headers=headers, allow_redirects=True)
            logging.info(f"Final URL after redirects: {response.url}")
            logging.info(f"Response Status Code: {response.status_code}")
            response.raise_for_status()
            if not response:
                logging.error("Failed to fetch sitemap.")
                return []
            content_type = response.headers.get("Content-Type", "").lower()
            logging.info(content_type)
            if "xml" in content_type:
                soup = BeautifulSoup(response.content, "xml")
                urls = []

                # Check for nested sitemaps
                nested_sitemaps = [sitemap.loc.text for sitemap in soup.find_all("sitemap") if sitemap.loc]
                if nested_sitemaps:
                    logging.info(f'Found {len(nested_sitemaps)} nested sitemaps.')
                    for nested_sitemap in nested_sitemaps:
                        logging.info(f'Processing nested sitemap: {nested_sitemap}')
                        if self.is_valid_url(nested_sitemap):
                            urls.extend(KnowledgeBaseManager(self.university_name, nested_sitemap).fetch_urls_from_sitemap_or_html(visited))

                # Process <loc> tags in the current sitemap (handle direct URLs)
                loc_tags = [loc.text for loc in soup.find_all("loc") if self.is_valid_url(loc.text)]
                urls.extend(loc_tags)

                logging.info(f"Fetched {len(loc_tags)} URLs from current sitemap {self.sitemap_url}.")
                return urls
            elif "html" in content_type:
                # Treat as HTML document
                urls = self.fetch_urls_from_html(response.content, self.sitemap_url)
                logging.info(f"Fetched {len(urls)} URLs from HTML page.")
                return urls
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch sitemap: {str(e)}")
            return []

    def process_urls_in_batches(self, urls, batch_size=100):
        """
        Process URLs in batches to load and save data.

        Args:
            urls (list): List of URLs to process.
            batch_size (int): Number of URLs to process in each batch.
        """
        valid_urls = [url for url in urls if self.is_valid_url(url) and not url.endswith(".xml")]
        total_urls = len(valid_urls)
        logging.info(f"Total valid URLs: {total_urls}")

        for i in range(0, total_urls, batch_size):
            batch = valid_urls[i:i + batch_size]
            logging.info(f"Processing batch {i // batch_size + 1}: {len(batch)} URLs")
            try:
                self._load_and_save_batch(batch)
            except Exception as e:
                logging.error(f"Error processing batch {i // batch_size + 1}: {str(e)}")

    def _load_and_save_batch(self, urls):
        """Load data from URLs and save to the pickle file."""
        try:
            # Filter out non-text URLs
            text_urls = [url for url in urls if not url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.pdf'))]
            logging.info(f"Processing {len(text_urls)} text URLs (skipping images and PDFs).")

            loader = UnstructuredURLLoader(urls=urls)
            new_data = loader.load()

            # Load existing data if the file exists
            if os.path.exists(self.data_file):
                try:
                    with open(self.data_file, 'rb') as f:
                        existing_data = pickle.load(f)
                    logging.info(f"Loaded existing data with {len(existing_data)} documents.")
                except EOFError:
                    logging.warning(f"{self.data_file} is empty. Starting with a new dataset.")
                    existing_data = []
            else:
                existing_data = []

            # Append new data
            updated_data = existing_data + new_data

            # Save updated data
            with open(self.data_file, 'wb') as f:
                pickle.dump(updated_data, f)
            logging.info(f"Data saved successfully. Total documents: {len(updated_data)}")
        except Exception as e:
            logging.error(f"Failed to load and save data: {str(e)}")

    def update_existing_data(self, additional_urls, batch_size=100):
        """
        Update existing pickle file with new data.

        Args:
            additional_urls (list): List of new URLs to add.
            batch_size (int): Number of URLs to process in each batch.
        """
        valid_urls = [url for url in additional_urls if self.is_valid_url(url) and not url.endswith(".xml")]
        total_urls = len(valid_urls)
        logging.info(f"Valid URLs to process: {total_urls}")

        for i in range(0, total_urls, batch_size):
            batch = valid_urls[i:i + batch_size]
            logging.info(f"Processing batch {i // batch_size + 1}: {len(batch)} URLs")
            try:
                self._load_and_save_batch(batch)
            except Exception as e:
                logging.error(f"Error processing batch {i // batch_size + 1}: {str(e)}")

    def load_data(self):
        """Load data from the pickle file."""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'rb') as f:
                    data = pickle.load(f)
                logging.info(f"Loaded {len(data)} documents from {self.data_file}.")
                return data
            except Exception as e:
                logging.error(f"Failed to load data: {str(e)}")
                return []
        else:
            logging.warning(f"No data file found at {self.data_file}.")
            return []
    
    def process_manual_data(self, manual_data):
        """
        Process manual data input and save it to the pickle file.

        Args:
            manual_data (str): Multiline string where each line is treated as a separate document.

        Returns:
            bool: True if data is saved successfully, False otherwise.
        """
        if not manual_data:
            return []

        documents = [
            Document(page_content=text.strip(), metadata={"source": "user-input"})
            for text in manual_data.split("\n") if text.strip()
        ]
        logging.info(f"Processed {len(documents)} documents from textual input.")
        return documents

if __name__ == "__main__":
    # Example usage for Pace University
    pace_manager = KnowledgeBaseManager(
        university_name="Pace University",
        sitemap_url="https://www.yu.edu/sitemap.xml"
    )
    urls = pace_manager.fetch_urls_from_sitemap_or_html()
    pace_manager.process_urls_in_batches(urls, batch_size=50)
