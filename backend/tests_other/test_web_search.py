"""
Web Search Testing Tool
Tests each search method individually to see how they work
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import json
import time
from xml.etree import ElementTree as ET


class SearchTester:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def test_wikipedia_opensearch(self, query: str):
        """
        Test Wikipedia OpenSearch API
        Best for: Quick searches, definitions, general knowledge
        """
        print("\n" + "="*80)
        print("🔍 TEST 1: Wikipedia OpenSearch API")
        print("="*80)
        print(f"Query: {query}")
        print(f"URL: https://en.wikipedia.org/w/api.php")
        print("\nHow it works:")
        print("- Uses Wikipedia's official OpenSearch API")
        print("- Returns article titles, descriptions, and URLs")
        print("- Fast and reliable, never blocks\n")
        
        try:
            api_url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'opensearch',
                'search': query,
                'limit': 3,
                'namespace': 0,
                'format': 'json'
            }
            
            print(f"Request params: {json.dumps(params, indent=2)}\n")
            
            response = self.session.get(api_url, params=params, timeout=10)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"\nRaw Response Structure:")
                print(f"- Array[0]: Query term")
                print(f"- Array[1]: Article titles")
                print(f"- Array[2]: Descriptions")
                print(f"- Array[3]: URLs\n")
                
                if len(data) >= 4:
                    titles = data[1]
                    descriptions = data[2]
                    urls = data[3]
                    
                    print(f"✅ Found {len(titles)} results:\n")
                    for i, (title, desc, url) in enumerate(zip(titles, descriptions, urls), 1):
                        print(f"Result {i}:")
                        print(f"  Title: {title}")
                        print(f"  Description: {desc[:200]}...")
                        print(f"  URL: {url}\n")
                    
                    return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def test_wikipedia_textextracts(self, query: str):
        """
        Test Wikipedia TextExtracts API
        Best for: Getting full article content/summaries
        """
        print("\n" + "="*80)
        print("🔍 TEST 2: Wikipedia TextExtracts API")
        print("="*80)
        print(f"Query: {query}")
        print("\nHow it works:")
        print("- Gets full article text/extract from Wikipedia")
        print("- Returns introduction paragraphs")
        print("- Great for detailed information\n")
        
        try:
            api_url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'prop': 'extracts|info',
                'exintro': True,
                'explaintext': True,
                'inprop': 'url',
                'titles': query,
                'redirects': 1
            }
            
            print(f"Request params: {json.dumps(params, indent=2)}\n")
            
            response = self.session.get(api_url, params=params, timeout=10)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                pages = data.get('query', {}).get('pages', {})
                
                print(f"\n✅ Found {len(pages)} page(s):\n")
                for page_id, page_data in pages.items():
                    if page_id != '-1':
                        title = page_data.get('title', 'Unknown')
                        extract = page_data.get('extract', '')
                        url = page_data.get('fullurl', '')
                        
                        print(f"Title: {title}")
                        print(f"URL: {url}")
                        print(f"Extract (first 500 chars):")
                        print(f"{extract[:500]}...\n")
                        return True
                    else:
                        print("❌ Page not found")
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    # def test_crossref_api(self, query: str):
    #     """
    #     Test CrossRef API
    #     Best for: Academic papers, DOIs, citations
    #     """
    #     print("\n" + "=" * 80)
    #     print("🔍 TEST 3: CrossRef API")
    #     print("=" * 80)
    #     print(f"Query: {query}")
    #     print("\nHow it works:")
    #     print("- Searches academic metadata from CrossRef")
    #     print("- Returns titles, authors, abstracts, DOIs")
    #     print("- Good for research/academic queries\n")

    #     try:
    #         api_url = "https://api.crossref.org/works"
    #         params = {
    #             'query': query,
    #             'rows': 3,
    #             'select': 'title,author,abstract,DOI,URL,published'
    #         }

    #         response = self.session.get(api_url, params=params, timeout=10)

    #         if response.status_code != 200:
    #             print(f"❌ HTTP Error: {response.status_code}")
    #             return False

    #         data = response.json()
    #         items = data.get('message', {}).get('items', [])

    #         print(f"✓ Found {len(items)} results via CrossRef\n")

    #         results = []

    #         for i, item in enumerate(items, start=1):
    #             title_list = item.get('title', [])
    #             title = title_list[0] if title_list else "Untitled Paper"

    #             abstract = item.get('abstract', '')
    #             snippet = abstract[:300] if abstract else "No abstract available."

    #             doi = item.get('DOI', '')
    #             url = item.get('URL', f"https://doi.org/{doi}" if doi else "")

    #             # Year
    #             published = item.get('published', {}).get('date-parts', [[None]])[0]
    #             year = published[0] if published else "Unknown"

    #             # ----- PRINT RESULTS -----
    #             print(f"Result {i}:")
    #             print(f"  Title: {title}")
    #             print(f"  Year: {year}")
    #             print(f"  DOI URL: {url}")
    #             print(f"  Snippet: {snippet}")
    #             print("-" * 60)

    #             results.append({
    #                 'title': title,
    #                 'url': url,
    #                 'year': year,
    #                 'snippet': snippet,
    #                 'source': 'CrossRef'
    #             })

    #         return results

    #     except Exception as e:
    #         print(f"❌ Error: {e}")
    #         return False


    def test_crossref_api(self, query: str):
        print("\n" + "="*80)
        print("🔍 TEST 3: DuckDuckGo Fallback Search")
        print("="*80)
        print(f"Query: {query}")
        print("URL: https://duckduckgo.com/html\n")

        try:
            url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")

            results = []

            # DuckDuckGo can use either of these container classes:
            containers = soup.select("div.result, div.web-result")[:5]

            for c in containers:
                # Try BOTH possible title locations
                title_tag = c.select_one("a.result__a")
                if not title_tag:
                    continue

                # Try BOTH possible snippet structures
                snippet_tag = c.select_one("div.result__snippet, a.result__snippet")
                snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

                results.append({
                    "title": title_tag.get_text(strip=True),
                    "url": title_tag.get("href", ""),
                    "snippet": snippet,
                    "source": "DuckDuckGo"
                })

            if results:
                print(f"✓ Found {len(results)} results via DuckDuckGo:\n")
                for r in results[:3]:
                    print(f"Title: {r['title']}")
                    print(f"URL: {r['url']}")
                    print(f"Snippet: {r['snippet']}\n")
            else:
                print("❌ No results — DuckDuckGo returned empty or rate-limited HTML.")

            return results

        except Exception as e:
            print("DuckDuckGo error:", e)
            return []

    def test_semantic_scholar(self, query: str):
        """
        Test Semantic Scholar API
        Best for: Scientific research, academic papers
        """
        print("\n" + "="*80)
        print("🔍 TEST 4: Semantic Scholar API")
        print("="*80)
        print(f"Query: {query}")
        print(f"URL: https://api.semanticscholar.org/")
        print("\nHow it works:")
        print("- Searches academic papers and research")
        print("- Returns abstracts, authors, citations")
        print("- Great for scientific/medical queries\n")
        
        try:
            api_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                'query': query,
                'limit': 3,
                'fields': 'title,abstract,url,year,authors,citationCount'
            }
            
            print(f"Request params: {json.dumps(params, indent=2)}\n")
            
            response = self.session.get(api_url, params=params, timeout=10)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                papers = data.get('data', [])
                
                print(f"\n✅ Found {len(papers)} papers:\n")
                for i, paper in enumerate(papers, 1):
                    print(f"Paper {i}:")
                    print(f"  Title: {paper.get('title', 'N/A')}")
                    print(f"  Year: {paper.get('year', 'N/A')}")
                    
                    authors = paper.get('authors', [])
                    author_names = ', '.join([a.get('name', '') for a in authors[:3]])
                    print(f"  Authors: {author_names}")
                    print(f"  Citations: {paper.get('citationCount', 0)}")
                    
                    abstract = paper.get('abstract', '')
                    if abstract:
                        print(f"  Abstract: {abstract[:300]}...")
                    print(f"  URL: {paper.get('url', 'N/A')}\n")
                
                return len(papers) > 0
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def test_arxiv(self, query: str):
        """
        Test arXiv API
        Best for: Physics, math, CS, and other technical topics
        """
        print("\n" + "="*80)
        print("🔍 TEST 5: arXiv API")
        print("="*80)
        print(f"Query: {query}")
        print(f"URL: http://export.arxiv.org/api/")
        print("\nHow it works:")
        print("- Searches preprint papers on arXiv")
        print("- Returns title, abstract, authors")
        print("- Excellent for technical/scientific content\n")
        
        try:
            api_url = "http://export.arxiv.org/api/query"
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': 3,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            print(f"Request params: {json.dumps(params, indent=2)}\n")
            
            response = self.session.get(api_url, params=params, timeout=10)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                ns = {'atom': 'http://www.w3.org/2005/Atom'}
                
                entries = root.findall('atom:entry', ns)
                print(f"\n✅ Found {len(entries)} papers:\n")
                
                for i, entry in enumerate(entries, 1):
                    title = entry.find('atom:title', ns)
                    summary = entry.find('atom:summary', ns)
                    published = entry.find('atom:published', ns)
                    link = entry.find('atom:id', ns)
                    
                    authors = entry.findall('atom:author', ns)
                    author_names = []
                    for author in authors[:3]:
                        name = author.find('atom:name', ns)
                        if name is not None:
                            author_names.append(name.text)
                    
                    print(f"Paper {i}:")
                    if title is not None:
                        print(f"  Title: {title.text.strip()}")
                    if published is not None:
                        print(f"  Published: {published.text[:10]}")
                    if author_names:
                        print(f"  Authors: {', '.join(author_names)}")
                    if summary is not None:
                        print(f"  Summary: {summary.text.strip()[:300]}...")
                    if link is not None:
                        print(f"  URL: {link.text}\n")
                
                return len(entries) > 0
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def test_all_methods(self, query: str):
        """
        Test all search methods with a single query
        """
        print("\n" + "🌟"*40)
        print(f"TESTING ALL SEARCH METHODS")
        print(f"Query: '{query}'")
        print("🌟"*40)
        
        results = {
            'Wikipedia OpenSearch': self.test_wikipedia_opensearch(query),
            'Wikipedia TextExtracts': self.test_wikipedia_textextracts(query),
            'Semantic Scholar': self.test_semantic_scholar(query),
            'arXiv': self.test_arxiv(query),  
            'crossRef': self.test_crossref_api(query)
        }
        
        # Summary
        print("\n" + "="*80)
        print("📊 RESULTS SUMMARY")
        print("="*80)
        print("\nRELIABILITY TIERS:")
        print("="*80)
        print("TIER 1 (99%+ Success) - Always try first:")
        for method in ['Wikipedia OpenSearch', 'Wikipedia TextExtracts']:
            if method in results:
                status = "✅ SUCCESS" if results[method] else "❌ FAILED"
                print(f"  {method:30s} {status}")
        
        print("\nTIER 2 (80-90% Success) - Good academic sources:")
        for method in ['Semantic Scholar', 'arXiv']:
            if method in results:
                status = "✅ SUCCESS" if results[method] else "❌ FAILED"
                print(f"  {method:30s} {status}")
        
        successful = sum(1 for v in results.values() if v)
        print(f"\n✅ {successful}/{len(results)} methods returned results")
        
        if successful == 0:
            print("\n⚠️ WARNING: All methods failed for this query")
            print("This might indicate:")
            print("  - Query too specific/recent")
            print("  - Network issues")
            print("  - Rate limiting")
        elif successful < len(results) / 2:
            print(f"\n⚠️ NOTE: Only {successful} methods worked")
            print("This is normal for very specific queries")
        else:
            print(f"\n✅ Good coverage: {successful} methods found results")
        
        print("="*80)


def main():
    tester = SearchTester()
    
    # Test queries - try different types
    test_queries = [
        "Air Quality Index Delhi",           # Current events
        "Python programming",                # General knowledge
        "machine learning algorithms",       # Technical/academic
        "climate change",                    # Scientific topic
        "Albert Einstein relativity"         # Historical/scientific
    ]
    
    print("\n" + "🚀"*40)
    print("WEB SEARCH TESTING TOOL")
    print("This tool tests each search API individually")
    print("🚀"*40)
    
    # Interactive mode
    while True:
        print("\n" + "-"*80)
        print("OPTIONS:")
        print("1. Test with predefined queries")
        print("2. Test with custom query")
        print("3. Test specific method")
        print("4. Exit")
        print("-"*80)
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            print("\n📋 Testing with predefined queries...")
            for query in test_queries:
                tester.test_all_methods(query)
                time.sleep(2)  # Be respectful to APIs
            break
            
        elif choice == '2':
            query = input("\nEnter your search query: ").strip()
            if query:
                tester.test_all_methods(query)
            else:
                print("❌ Empty query")
                
        elif choice == '3':
            print("\nAvailable methods:")
            print("1. Wikipedia OpenSearch")
            print("2. Wikipedia TextExtracts")
            print("3. CrossRef API")
            print("4. Semantic Scholar")
            print("5. arXiv")
            
            method_choice = input("Choose method (1-5): ").strip()
            query = input("Enter search query: ").strip()
            
            if query:
                if method_choice == '1':
                    tester.test_wikipedia_opensearch(query)
                elif method_choice == '2':
                    tester.test_wikipedia_textextracts(query)
                elif method_choice == '3':
                    tester.test_crossref_api(query)
                elif method_choice == '4':
                    tester.test_semantic_scholar(query)
                elif method_choice == '5':
                    tester.test_arxiv(query)
                else:
                    print("❌ Invalid method choice")
            else:
                print("❌ Empty query")
                
        elif choice == '4':
            print("\n👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main()