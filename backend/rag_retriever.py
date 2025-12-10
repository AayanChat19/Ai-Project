import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import json
import os
import requests
from bs4 import BeautifulSoup
import time
from dotenv import load_dotenv

load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")


class WebSearcher:
    """Enhanced web search with source reliability ranking"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        
        # Source reliability scores (higher = more reliable)
        self.reliability_scores = {
            'wikipedia': 0.95,
            'google': 0.90,
            'semantic_scholar': 0.85,
            'arxiv': 0.85,
            'crossref': 0.80,
            'default': 0.70
        }
    
    def search_wikipedia(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search Wikipedia - HIGHEST RELIABILITY"""
        try:
            api_url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'opensearch',
                'search': query,
                'limit': max_results,
                'namespace': 0,
                'format': 'json'
            }
            
            response = self.session.get(api_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                if len(data) >= 4:
                    titles = data[1]
                    descriptions = data[2]
                    urls = data[3]
                    
                    for title, desc, url in zip(titles, descriptions, urls):
                        if desc:
                            results.append({
                                'title': title,
                                'url': url,
                                'snippet': desc,
                                'source': 'Wikipedia',
                                'reliability': self.reliability_scores['wikipedia']
                            })
                
                if results:
                    print(f"✓ Wikipedia: {len(results)} results")
                    return results
                    
            # Fallback to TextExtracts
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
            
            response = self.session.get(api_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                pages = data.get('query', {}).get('pages', {})
                
                for page_id, page_data in pages.items():
                    if page_id != '-1':
                        extract = page_data.get('extract', '')
                        if extract:
                            snippet = extract[:500] + ('...' if len(extract) > 500 else '')
                            results.append({
                                'title': page_data.get('title', 'Wikipedia Article'),
                                'url': page_data.get('fullurl', ''),
                                'snippet': snippet,
                                'source': 'Wikipedia',
                                'reliability': self.reliability_scores['wikipedia']
                            })
                
                if results:
                    print(f"✓ Wikipedia (extracts): {len(results)} results")
                return results
                    
        except Exception as e:
            print(f"Wikipedia error: {e}")
        
        return []
    
    def search_google_search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search Google via Serper API - HIGH RELIABILITY"""
        if not SERPER_API_KEY:
            print("⚠ No Serper API key configured")
            return []
            
        try:
            url = "https://google.serper.dev/search"
            payload = {"q": query, "num": max_results}
            headers = {
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json"
            }

            response = self.session.post(url, json=payload, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                results = []

                if "organic" in data:
                    for item in data["organic"]:
                        snippet = item.get("snippet", "")
                        if len(snippet) > 400:
                            snippet = snippet[:400] + "..."

                        results.append({
                            "title": item.get("title", "No title"),
                            "url": item.get("link", ""),
                            "snippet": snippet,
                            "source": "Google",
                            "reliability": self.reliability_scores['google']
                        })

                if results:
                    print(f"✓ Google: {len(results)} results")
                return results

        except Exception as e:
            print(f"Google Search error: {e}")

        return []
    
    def search_semantic_scholar(self, query: str, max_results: int = 3) -> List[Dict]:
        """Search Semantic Scholar - ACADEMIC RELIABILITY"""
        try:
            api_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                'query': query,
                'limit': max_results,
                'fields': 'title,abstract,url,year,authors'
            }
            
            response = self.session.get(api_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for paper in data.get('data', []):
                    abstract = paper.get('abstract', '')
                    if abstract:
                        snippet = abstract[:400] + ('...' if len(abstract) > 400 else '')
                        year = paper.get('year', '')
                        title = paper.get('title', 'Research Paper')
                        if year:
                            title = f"{title} ({year})"
                        
                        results.append({
                            'title': title,
                            'url': paper.get('url', ''),
                            'snippet': snippet,
                            'source': 'Semantic Scholar',
                            'reliability': self.reliability_scores['semantic_scholar']
                        })
                
                if results:
                    print(f"✓ Semantic Scholar: {len(results)} results")
                return results
                
        except Exception as e:
            print(f"Semantic Scholar error: {e}")
        
        return []
    
    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Multi-source search with reliability-based prioritization
        Priority: Wikipedia > Google > Academic sources
        """
        all_results = []
        
        # Tier 1: Wikipedia (highest reliability)
        print("🔍 Searching Wikipedia...")
        wiki_results = self.search_wikipedia(query, max_results)
        all_results.extend(wiki_results)
        
        # Tier 2: Google (if available and needed)
        if len(all_results) < max_results:
            print("🔍 Searching Google...")
            google_results = self.search_google_search(query, max_results - len(all_results))
            all_results.extend(google_results)
        
        # Tier 3: Academic sources (for scientific queries)
        if len(all_results) < max_results and any(
            word in query.lower() for word in ['research', 'study', 'scientific', 'paper']
        ):
            print("🔍 Searching Semantic Scholar...")
            academic_results = self.search_semantic_scholar(query, 2)
            all_results.extend(academic_results)
        
        # Sort by reliability score
        all_results.sort(key=lambda x: x.get('reliability', 0.5), reverse=True)
        
        if not all_results:
            print("⚠ No web results found")
        else:
            print(f"✓ Total web results: {len(all_results)}")
        
        return all_results[:max_results]


class EnhancedRAGRetriever:
    """
    Enhanced RAG system with:
    - Claim-level verification support
    - Evidence quality scoring
    - Multi-query retrieval
    - Web search fallback with reliability ranking
    """
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 relevance_threshold: float = 0.7,
                 enable_web_search: bool = True):
        print(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        self.dimension = self.embedder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.metadata = []
        
        self.relevance_threshold = relevance_threshold
        self.enable_web_search = enable_web_search
        self.web_searcher = WebSearcher() if enable_web_search else None
        
        print(f"RAG Retriever initialized (Web: {'ON' if enable_web_search else 'OFF'})")
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """Add documents to local knowledge base"""
        if not documents:
            return
        
        print(f"Adding {len(documents)} documents to index...")
        embeddings = self.embedder.encode(documents, convert_to_numpy=True)
        self.index.add(embeddings.astype('float32'))
        self.documents.extend(documents)
        
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{"doc_id": i} for i in range(len(documents))])
        
        print(f"Total documents: {len(self.documents)}")
    
    def _calculate_relevance_score(self, distance: float) -> float:
        """Convert L2 distance to relevance score (0-1)"""
        scale = 2.0
        return np.exp(-distance / scale)
    
    def retrieve_local(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve from local knowledge base"""
        if len(self.documents) == 0:
            return []
        
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        k = min(k, len(self.documents))
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                distance = float(distances[0][i])
                relevance = self._calculate_relevance_score(distance)
                
                # Local sources get high base reliability
                metadata = self.metadata[idx] if idx < len(self.metadata) else {}
                reliability = metadata.get('reliability', 'high')
                reliability_score = 0.90 if reliability == 'high' else 0.75
                
                results.append({
                    "document": self.documents[idx],
                    "distance": distance,
                    "relevance": relevance,
                    "metadata": metadata,
                    "rank": i + 1,
                    "source_type": "local",
                    "reliability": reliability_score
                })
        
        return results
    
    def retrieve_web(self, query: str, max_results: int = 3) -> List[Dict]:
        """Retrieve evidence from web with reliability scores"""
        if not self.enable_web_search or not self.web_searcher:
            return []
        
        print(f"\n{'='*60}")
        print(f"🌐 Web search: {query}")
        print('='*60)
        
        search_results = self.web_searcher.search(query, max_results=max_results)
        
        if not search_results:
            return []
        
        web_evidence = []
        for i, result in enumerate(search_results):
            web_evidence.append({
                "document": result['snippet'],
                "distance": 0.0,
                "relevance": 0.85,  # High relevance for search results
                "metadata": {
                    "source": result['title'],
                    "url": result['url'],
                    "topic": "Web Search",
                    "reliability": result.get('reliability', 0.70)
                },
                "rank": i + 1,
                "source_type": "web",
                "reliability": result.get('reliability', 0.70)
            })
        
        print(f"✓ Retrieved {len(web_evidence)} web sources")
        return web_evidence
    
    def retrieve(self, query: str, k: int = 5, force_web: bool = False) -> List[Dict]:
        """
        Intelligent retrieval with evidence quality ranking
        
        Returns evidence sorted by:
        1. Source reliability (local > Wikipedia > Google > others)
        2. Relevance score
        """
        # Get local results
        local_results = self.retrieve_local(query, k=k)
        
        # Check local quality
        has_good_local_results = False
        if local_results:
            max_local_relevance = max(r['relevance'] for r in local_results)
            has_good_local_results = max_local_relevance >= self.relevance_threshold
            
            print(f"\n📊 Local: Max relevance {max_local_relevance:.3f} | Threshold {self.relevance_threshold:.3f}")
        
        # Decide on web search
        use_web = force_web or not has_good_local_results
        
        if use_web and self.enable_web_search:
            print("🔄 Triggering web search...")
            web_results = self.retrieve_web(query, max_results=3)
            
            if web_results:
                combined = local_results + web_results
                
                # Sort by reliability first, then relevance
                combined.sort(
                    key=lambda x: (x.get('reliability', 0.5), x.get('relevance', 0.0)),
                    reverse=True
                )
                
                return combined[:k]
        
        return local_results
    
    def save_index(self, path: str):
        """Save index to disk"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        faiss.write_index(self.index, f"{path}.faiss")
        
        with open(f"{path}.json", 'w') as f:
            json.dump({
                "documents": self.documents,
                "metadata": self.metadata,
                "relevance_threshold": self.relevance_threshold
            }, f)
        
        print(f"Index saved to {path}")
    
    def load_index(self, path: str):
        """Load index from disk"""
        self.index = faiss.read_index(f"{path}.faiss")
        
        with open(f"{path}.json", 'r') as f:
            data = json.load(f)
            self.documents = data["documents"]
            self.metadata = data["metadata"]
            if "relevance_threshold" in data:
                self.relevance_threshold = data["relevance_threshold"]
        
        print(f"Index loaded. Total documents: {len(self.documents)}")


class KnowledgeBase:
    """Sample knowledge base with reliability metadata"""
    
    @staticmethod
    def get_sample_documents() -> List[str]:
        return [
            # Astronomy & Space
            "The Earth orbits the Sun in approximately 365.25 days, which is why we have leap years every four years.",
            "The Moon is Earth's only natural satellite and is approximately 384,400 km away from Earth.",
            "Mars is the fourth planet from the Sun and is often called the Red Planet due to iron oxide on its surface.",
            "The speed of light in vacuum is approximately 299,792,458 meters per second (3 x 10^8 m/s).",
            "The Sun is a medium-sized star composed primarily of hydrogen and helium.",
            
            # Chemistry & Physics
            "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level under standard atmospheric pressure.",
            "The periodic table was created by Dmitri Mendeleev in 1869 and organizes elements by atomic number.",
            "Gold has the chemical symbol Au, derived from the Latin word 'aurum'.",
            "Oxygen makes up about 21% of Earth's atmosphere by volume.",
            "The law of conservation of energy states that energy cannot be created or destroyed, only transformed.",
            
            # Biology & Medicine
            "The human body has 206 bones in adulthood, though babies are born with around 270 bones that fuse together.",
            "DNA stands for deoxyribonucleic acid and contains genetic instructions for all living organisms.",
            "The human brain weighs approximately 1.4 kilograms (3 pounds) and contains about 86 billion neurons.",
            "Photosynthesis is the process by which plants convert sunlight, water, and CO2 into glucose and oxygen.",
            "The heart pumps approximately 5 liters of blood per minute in an average adult at rest.",
            "Vaccines work by training the immune system to recognize and fight specific pathogens without causing disease.",
            
            # Computer Science
            "Python is a high-level programming language created by Guido van Rossum and first released in 1991.",
            "HTML stands for HyperText Markup Language and is the standard language for creating web pages.",
            "Binary code uses only two digits, 0 and 1, to represent all data in computers.",
            "The first computer programmer was Ada Lovelace, who wrote an algorithm in the 1840s.",
            
            # Geography
            "The Pacific Ocean is the largest and deepest ocean, covering about 63 million square miles.",
            "The capital of France is Paris, which has a population of over 2 million in the city proper.",
            "Mount Everest is the highest mountain above sea level at 8,849 meters (29,032 feet).",
            "The Amazon River is the second longest river in the world and has the largest drainage basin.",
            "The Sahara Desert is the largest hot desert in the world, covering much of North Africa.",
            
            # History
            "The United States Declaration of Independence was signed on July 4, 1776.",
            "World War II ended in 1945 after lasting six years from 1939 to 1945.",
            "The Great Wall of China was built over many centuries, with major construction during the Ming Dynasty.",
            "The printing press was invented by Johannes Gutenberg around 1440.",
            
            # Mathematics
            "Pi (π) is approximately 3.14159 and represents the ratio of a circle's circumference to its diameter.",
            "The Pythagorean theorem states that a² + b² = c² in a right triangle.",
            "Zero was invented as a number in ancient India and later adopted by Arab mathematicians.",
            
            # Common Misconceptions
            "Lightning can strike the same place twice - tall buildings are frequently struck multiple times.",
            "Humans use all parts of their brain, not just 10%. Brain imaging shows activity throughout the entire brain.",
            "The Great Wall of China is not visible from space with the naked eye, contrary to popular belief.",
            "Goldfish have memories lasting at least several months, not just three seconds.",
            "Shaving does not make hair grow back thicker or darker - this is a common myth.",
            "Going outside with wet hair in cold weather does not cause colds - viruses do.",
            "Cracking knuckles does not cause arthritis according to scientific studies.",
            "Swallowed gum does not stay in your stomach for seven years - it passes through normally.",
            
            # Technology & AI
            "The World Wide Web was invented by Tim Berners-Lee in 1989 at CERN.",
            "Artificial Intelligence refers to computer systems that can perform tasks normally requiring human intelligence.",
            "Smartphones typically use lithium-ion batteries for their high energy density and rechargeability.",
            "Large Language Models (LLMs) like GPT and Claude are trained on vast amounts of text data to generate human-like responses.",
            "Machine learning is a subset of AI where computers learn from data without being explicitly programmed.",
            "Neural networks are computing systems inspired by biological neural networks in animal brains.",
            "The first iPhone was released by Apple in 2007, revolutionizing the smartphone industry.",
            "Quantum computers use quantum bits (qubits) that can exist in multiple states simultaneously.",
            "Blockchain is a distributed ledger technology that powers cryptocurrencies like Bitcoin.",
            "5G networks offer faster speeds and lower latency compared to 4G LTE networks.",

            # More History
            "The Renaissance was a period of cultural rebirth in Europe from the 14th to 17th century.",
            "The fall of the Berlin Wall occurred on November 9, 1989, symbolizing the end of the Cold War.",
            "The French Revolution began in 1789 and led to major changes in French and European society.",
            "The Industrial Revolution started in Britain in the late 18th century and transformed manufacturing.",
            "The Roman Empire reached its greatest extent under Emperor Trajan around 117 AD.",
            "Christopher Columbus reached the Americas in 1492, though Vikings had arrived centuries earlier.",
            "The Apollo 11 mission landed humans on the Moon on July 20, 1969.",
            "The Black Death pandemic in the 14th century killed an estimated 75-200 million people in Eurasia.",

            # Climate & Environment
            "Climate change is primarily caused by greenhouse gas emissions from human activities.",
            "The ozone layer protects Earth from harmful ultraviolet radiation from the Sun.",
            "Deforestation in the Amazon rainforest contributes significantly to global carbon emissions.",
            "The greenhouse effect is a natural process that warms Earth's surface to support life.",
            "Rising global temperatures are causing polar ice caps and glaciers to melt at accelerating rates.",
            "Ocean acidification is caused by increased absorption of atmospheric carbon dioxide.",
            "Renewable energy sources include solar, wind, hydroelectric, and geothermal power.",
            "The Paris Agreement aims to limit global temperature increase to well below 2°C above pre-industrial levels.",

            # Psychology & Behavior
            "The placebo effect demonstrates the power of belief in producing real physiological changes.",
            "Cognitive bias refers to systematic patterns of deviation from rationality in judgment.",
            "Working memory can typically hold about 7 plus or minus 2 items at once.",
            "Neuroplasticity is the brain's ability to reorganize itself by forming new neural connections.",
            "Sleep is crucial for memory consolidation and learning.",
            "The fight-or-flight response is an automatic physiological reaction to perceived threats.",
            "Confirmation bias is the tendency to search for information that confirms existing beliefs.",

            # Health & Nutrition
            "The human body requires essential amino acids that cannot be synthesized internally.",
            "Vitamin D is produced in the skin when exposed to sunlight and is crucial for bone health.",
            "Regular physical exercise reduces the risk of heart disease, diabetes, and many other conditions.",
            "The gut microbiome contains trillions of microorganisms that influence digestion and immunity.",
            "Antibiotics are effective against bacteria but do not work against viral infections.",
            "Proper hydration is essential for bodily functions; adults need about 2-3 liters of water daily.",
            "Trans fats are artificially created fats that increase bad cholesterol and heart disease risk.",
            "The Mediterranean diet is associated with reduced risk of heart disease and longer life expectancy.",

            # Economics & Business
            "GDP (Gross Domestic Product) measures the total value of goods and services produced in a country.",
            "Inflation is the rate at which the general level of prices for goods and services rises.",
            "Supply and demand are fundamental economic concepts that determine market prices.",
            "The stock market allows companies to raise capital by selling shares to investors.",
            "Cryptocurrency is a digital or virtual currency secured by cryptography.",
            "The Federal Reserve is the central banking system of the United States, established in 1913.",
            "Compound interest is interest calculated on both the initial principal and accumulated interest.",

            # Literature & Arts
            "William Shakespeare wrote approximately 37 plays and 154 sonnets during his lifetime.",
            "The Mona Lisa by Leonardo da Vinci is one of the most famous paintings in the world.",
            "The printing revolution enabled mass production of books and spread of knowledge in the 15th century.",
            "Classical music refers to Western art music from roughly 1750 to 1820, including Mozart and Beethoven.",
            "The Sistine Chapel ceiling was painted by Michelangelo between 1508 and 1512.",
            "Poetry often uses meter, rhyme, and figurative language to convey emotions and ideas.",

            # Sports & Athletics
            "The Olympic Games originated in ancient Greece around 776 BC and were revived in 1896.",
            "A marathon is 42.195 kilometers (26.2 miles) based on the Athens to Marathon distance.",
            "The FIFA World Cup is held every four years and is the most-watched sporting event globally.",
            "The four tennis Grand Slam tournaments are Australian Open, French Open, Wimbledon, and US Open.",
            "Human reaction time to visual stimuli averages around 200-250 milliseconds.",

            # More Science
            "Evolution by natural selection was proposed by Charles Darwin in 'On the Origin of Species' (1859).",
            "Atoms are composed of protons, neutrons, and electrons, with electrons orbiting the nucleus.",
            "The theory of relativity by Albert Einstein revolutionized understanding of space, time, and gravity.",
            "Antibiotics were discovered by Alexander Fleming in 1928 with the discovery of penicillin.",
            "The Hubble Space Telescope has been observing the universe since its launch in 1990.",
            "CRISPR is a gene-editing technology that allows precise modifications to DNA sequences.",
            "Black holes are regions of spacetime where gravity is so strong that nothing can escape.",
            "Plate tectonics explains how Earth's crust is divided into plates that move and interact.",
            "Radiocarbon dating is used to determine the age of organic materials up to about 50,000 years old.",

            # Language & Communication
            "English is the most widely spoken language globally when combining native and second-language speakers.",
            "Mandarin Chinese has the most native speakers of any language in the world.",
            "Body language and non-verbal cues often communicate more than spoken words.",
            "The alphabet used for English originated from the Phoenician alphabet around 1050 BC.",

            # More Common Facts
            "Coffee is the second most traded commodity in the world after crude oil.",
            "The human eye can distinguish approximately 10 million different colors.",
            "Honey never spoils and has been found edible in ancient Egyptian tombs after thousands of years.",
            "Octopuses have three hearts and blue blood due to copper-based hemocyanin.",
            "A day on Venus is longer than a year on Venus due to its slow rotation.",
            "The average person walks the equivalent of five times around the world in a lifetime.",
            "Sound travels at approximately 343 meters per second in air at room temperature.",
            "The human nose can detect over 1 trillion different scents.",
        ]
    
    @staticmethod
    def get_sample_metadata() -> List[Dict]:
        sources = [
            {"topic": "Astronomy", "source": "NASA Educational Resources", "url": "https://science.nasa.gov/solar-system/", "reliability": "high"},
            {"topic": "Astronomy", "source": "Space.com", "url": "https://www.space.com/18175-moon-distance-from-earth.html", "reliability": "high"},
            {"topic": "Astronomy", "source": "NASA Mars Exploration", "url": "https://mars.nasa.gov/all-about-mars/facts/", "reliability": "high"},
            {"topic": "Physics", "source": "Physics Textbook", "url": "https://en.wikipedia.org/wiki/Speed_of_light", "reliability": "high"},
            {"topic": "Astronomy", "source": "NASA Solar System", "url": "https://science.nasa.gov/sun/", "reliability": "high"},
            
            {"topic": "Chemistry", "source": "Chemistry Handbook", "url": "https://en.wikipedia.org/wiki/Properties_of_water", "reliability": "high"},
            {"topic": "Chemistry", "source": "Royal Society of Chemistry", "url": "https://www.rsc.org/periodic-table/", "reliability": "high"},
            {"topic": "Chemistry", "source": "Periodic Table", "url": "https://ptable.com/#Properties/Series", "reliability": "high"},
            {"topic": "Atmospheric Science", "source": "NOAA", "url": "https://www.noaa.gov/education/resource-collections/atmosphere", "reliability": "high"},
            {"topic": "Physics", "source": "Physics Principles", "url": "https://en.wikipedia.org/wiki/Conservation_of_energy", "reliability": "high"},
            
            {"topic": "Biology", "source": "Gray's Anatomy", "url": "https://www.britannica.com/science/human-skeletal-system", "reliability": "high"},
            {"topic": "Molecular Biology", "source": "NCBI Genetics", "url": "https://www.genome.gov/genetics-glossary/Deoxyribonucleic-Acid", "reliability": "high"},
            {"topic": "Neuroscience", "source": "Brain Research Institute", "url": "https://www.britannica.com/science/human-brain", "reliability": "high"},
            {"topic": "Botany", "source": "Plant Biology", "url": "https://www.britannica.com/science/photosynthesis", "reliability": "high"},
            {"topic": "Cardiology", "source": "American Heart Association", "url": "https://www.heart.org/en/health-topics/high-blood-pressure/the-facts-about-high-blood-pressure", "reliability": "high"},
            {"topic": "Immunology", "source": "CDC Vaccines", "url": "https://www.cdc.gov/vaccines/vac-gen/howvpd.htm", "reliability": "high"},
            
            {"topic": "Computer Science", "source": "Python.org", "url": "https://www.python.org/about/", "reliability": "high"},
            {"topic": "Web Development", "source": "W3C Standards", "url": "https://www.w3.org/standards/webdesign/htmlcss", "reliability": "high"},
            {"topic": "Computer Science", "source": "CS Fundamentals", "url": "https://en.wikipedia.org/wiki/Binary_number", "reliability": "high"},
            {"topic": "History of Computing", "source": "Computer History Museum", "url": "https://www.computerhistory.org/babbage/adalovelace/", "reliability": "high"},
            
            {"topic": "Oceanography", "source": "NOAA Ocean Facts", "url": "https://oceanservice.noaa.gov/facts/pacific.html", "reliability": "high"},
            {"topic": "Geography", "source": "National Geographic", "url": "https://www.nationalgeographic.com/travel/article/paris-travel-guide", "reliability": "high"},
            {"topic": "Geography", "source": "USGS", "url": "https://www.usgs.gov/programs/VHP/mount-everest", "reliability": "high"},
            {"topic": "Geography", "source": "World Rivers", "url": "https://www.britannica.com/place/Amazon-River", "reliability": "high"},
            {"topic": "Geography", "source": "Desert Research", "url": "https://www.worldatlas.com/deserts/the-sahara-desert.html", "reliability": "high"},
            
            {"topic": "History", "source": "Library of Congress", "url": "https://www.loc.gov/exhibits/declara/declara1.html", "reliability": "high"},
            {"topic": "History", "source": "World War II Museum", "url": "https://www.nationalww2museum.org/war/articles/end-world-war-ii", "reliability": "high"},
            {"topic": "History", "source": "UNESCO World Heritage", "url": "https://whc.unesco.org/en/list/438/", "reliability": "high"},
            {"topic": "History", "source": "Gutenberg Museum", "url": "https://www.gutenberg.org/about/", "reliability": "high"},
            
            {"topic": "Mathematics", "source": "Math Encyclopedia", "url": "https://www.piday.org/learn-about-pi/", "reliability": "high"},
            {"topic": "Mathematics", "source": "Geometry Textbook", "url": "https://www.mathsisfun.com/pythagoras.html", "reliability": "high"},
            {"topic": "Mathematics", "source": "History of Math", "url": "https://www.britannica.com/science/zero-mathematics", "reliability": "high"},
            
            {"topic": "Meteorology", "source": "National Weather Service", "url": "https://www.weather.gov/safety/lightning-myths", "reliability": "high"},
            {"topic": "Neuroscience", "source": "Scientific American", "url": "https://www.scientificamerican.com/article/do-people-only-use-10-percent-of-their-brains/", "reliability": "high"},
            {"topic": "Engineering", "source": "NASA", "url": "https://www.nasa.gov/general/is-the-great-wall-of-china-visible-from-space/", "reliability": "high"},
            {"topic": "Zoology", "source": "Marine Biology", "url": "https://www.sciencedirect.com/science/article/abs/pii/S0003347207004454", "reliability": "high"},
            {"topic": "Biology", "source": "Medical Research", "url": "https://www.mayoclinic.org/healthy-lifestyle/adult-health/expert-answers/hair-removal/faq-20058427", "reliability": "high"},
            {"topic": "Health", "source": "Mayo Clinic", "url": "https://www.mayoclinic.org/diseases-conditions/common-cold/expert-answers/cold-weather/faq-20057966", "reliability": "high"},
            {"topic": "Health", "source": "Johns Hopkins", "url": "https://www.hopkinsmedicine.org/health/wellness-and-prevention/myth-buster-does-knuckle-cracking-cause-arthritis", "reliability": "high"},
            {"topic": "Health", "source": "Medical Research", "url": "https://www.dukehealth.org/blog/swallowed-gum-dont-worry", "reliability": "high"},
            
            {"topic": "Technology", "source": "CERN Archives", "url": "https://home.cern/science/computing/birth-web", "reliability": "high"},
            {"topic": "Computer Science", "source": "AI Research", "url": "https://en.wikipedia.org/wiki/Artificial_intelligence", "reliability": "high"},
            {"topic": "Technology", "source": "IEEE Battery Tech", "url": "https://batteryuniversity.com/article/bu-204-how-do-lithium-batteries-work", "reliability": "high"},

            # New metadata for additional documents
            {"topic": "AI/ML", "source": "OpenAI Research", "url": "https://openai.com/research", "reliability": "high"},
            {"topic": "AI/ML", "source": "Stanford AI Lab", "url": "https://ai.stanford.edu/", "reliability": "high"},
            {"topic": "AI/ML", "source": "Deep Learning Textbook", "url": "https://www.deeplearningbook.org/", "reliability": "high"},
            {"topic": "Technology", "source": "Apple Archives", "url": "https://www.apple.com/iphone/", "reliability": "high"},
            {"topic": "Quantum Computing", "source": "IBM Quantum", "url": "https://www.ibm.com/quantum", "reliability": "high"},
            {"topic": "Technology", "source": "Blockchain.com", "url": "https://www.blockchain.com/learning-portal/blockchain-basics", "reliability": "high"},
            {"topic": "Technology", "source": "Qualcomm 5G", "url": "https://www.qualcomm.com/5g/what-is-5g", "reliability": "high"},

            {"topic": "History", "source": "Britannica", "url": "https://www.britannica.com/event/Renaissance", "reliability": "high"},
            {"topic": "History", "source": "History Channel", "url": "https://www.history.com/topics/cold-war/berlin-wall", "reliability": "high"},
            {"topic": "History", "source": "French History", "url": "https://www.britannica.com/event/French-Revolution", "reliability": "high"},
            {"topic": "History", "source": "Industrial Revolution", "url": "https://www.britannica.com/event/Industrial-Revolution", "reliability": "high"},
            {"topic": "History", "source": "Roman History", "url": "https://www.britannica.com/place/Roman-Empire", "reliability": "high"},
            {"topic": "History", "source": "Historical Records", "url": "https://www.britannica.com/biography/Christopher-Columbus", "reliability": "high"},
            {"topic": "History", "source": "NASA History", "url": "https://www.nasa.gov/mission_pages/apollo/apollo-11.html", "reliability": "high"},
            {"topic": "History", "source": "Pandemic Research", "url": "https://www.britannica.com/event/Black-Death", "reliability": "high"},

            {"topic": "Climate Science", "source": "IPCC Reports", "url": "https://www.ipcc.ch/", "reliability": "high"},
            {"topic": "Atmospheric Science", "source": "NASA Ozone Watch", "url": "https://ozonewatch.gsfc.nasa.gov/", "reliability": "high"},
            {"topic": "Environmental Science", "source": "Amazon Conservation", "url": "https://www.worldwildlife.org/places/amazon", "reliability": "high"},
            {"topic": "Climate Science", "source": "NOAA Climate", "url": "https://www.climate.gov/", "reliability": "high"},
            {"topic": "Climate Science", "source": "NASA Climate", "url": "https://climate.nasa.gov/", "reliability": "high"},
            {"topic": "Marine Biology", "source": "NOAA Ocean Acidification", "url": "https://www.noaa.gov/education/resource-collections/ocean-coasts/ocean-acidification", "reliability": "high"},
            {"topic": "Energy", "source": "Renewable Energy World", "url": "https://www.energy.gov/eere/renewable-energy", "reliability": "high"},
            {"topic": "Climate Policy", "source": "UN Climate Change", "url": "https://unfccc.int/process-and-meetings/the-paris-agreement", "reliability": "high"},

            {"topic": "Psychology", "source": "Psychology Today", "url": "https://www.psychologytoday.com/us/basics/placebo-effect", "reliability": "high"},
            {"topic": "Psychology", "source": "Cognitive Science", "url": "https://www.simplypsychology.org/cognitive-bias.html", "reliability": "high"},
            {"topic": "Neuroscience", "source": "Memory Research", "url": "https://www.simplypsychology.org/short-term-memory.html", "reliability": "high"},
            {"topic": "Neuroscience", "source": "Brain Plasticity", "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3222570/", "reliability": "high"},
            {"topic": "Neuroscience", "source": "Sleep Research", "url": "https://www.sleepfoundation.org/how-sleep-works/why-do-we-need-sleep", "reliability": "high"},
            {"topic": "Psychology", "source": "Behavioral Science", "url": "https://www.simplypsychology.org/fight-flight.html", "reliability": "high"},
            {"topic": "Psychology", "source": "Cognitive Bias", "url": "https://www.britannica.com/science/confirmation-bias", "reliability": "high"},

            {"topic": "Nutrition", "source": "NIH Nutrition", "url": "https://www.ncbi.nlm.nih.gov/books/NBK557845/", "reliability": "high"},
            {"topic": "Health", "source": "Harvard Health", "url": "https://www.health.harvard.edu/staying-healthy/vitamin-d-and-your-health", "reliability": "high"},
            {"topic": "Health", "source": "WHO Physical Activity", "url": "https://www.who.int/news-room/fact-sheets/detail/physical-activity", "reliability": "high"},
            {"topic": "Microbiology", "source": "NIH Microbiome", "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4425030/", "reliability": "high"},
            {"topic": "Medicine", "source": "CDC Antibiotic Use", "url": "https://www.cdc.gov/antibiotic-use/", "reliability": "high"},
            {"topic": "Health", "source": "Mayo Clinic Hydration", "url": "https://www.mayoclinic.org/healthy-lifestyle/nutrition-and-healthy-eating/in-depth/water/art-20044256", "reliability": "high"},
            {"topic": "Nutrition", "source": "Harvard Nutrition Source", "url": "https://www.hsph.harvard.edu/nutritionsource/what-should-you-eat/fats-and-cholesterol/types-of-fat/trans-fats/", "reliability": "high"},
            {"topic": "Nutrition", "source": "Mediterranean Diet Research", "url": "https://www.hsph.harvard.edu/nutritionsource/healthy-weight/diet-reviews/mediterranean-diet/", "reliability": "high"},

            {"topic": "Economics", "source": "IMF", "url": "https://www.imf.org/external/pubs/ft/fandd/basics/gdp.htm", "reliability": "high"},
            {"topic": "Economics", "source": "Federal Reserve Education", "url": "https://www.federalreserveeducation.org/about-the-fed/structure-and-functions/inflation", "reliability": "high"},
            {"topic": "Economics", "source": "Economics Textbook", "url": "https://www.britannica.com/topic/supply-and-demand", "reliability": "high"},
            {"topic": "Finance", "source": "SEC Investor Education", "url": "https://www.investor.gov/introduction-investing/investing-basics/glossary/stock", "reliability": "high"},
            {"topic": "Finance", "source": "Blockchain Research", "url": "https://www.investopedia.com/terms/c/cryptocurrency.asp", "reliability": "high"},
            {"topic": "Economics", "source": "Federal Reserve History", "url": "https://www.federalreserve.gov/aboutthefed/structure-federal-reserve-system.htm", "reliability": "high"},
            {"topic": "Finance", "source": "Investment Education", "url": "https://www.investopedia.com/terms/c/compoundinterest.asp", "reliability": "high"},

            {"topic": "Literature", "source": "Folger Shakespeare Library", "url": "https://www.folger.edu/shakespeare", "reliability": "high"},
            {"topic": "Art", "source": "Louvre Museum", "url": "https://www.louvre.fr/en/oeuvre-notices/mona-lisa-portrait-lisa-gherardini-wife-francesco-del-giocondo", "reliability": "high"},
            {"topic": "History", "source": "Gutenberg Museum", "url": "https://www.britannica.com/technology/printing-press", "reliability": "high"},
            {"topic": "Music", "source": "Classical Music History", "url": "https://www.britannica.com/art/Classical-music-era", "reliability": "high"},
            {"topic": "Art", "source": "Vatican Museums", "url": "https://www.museivaticani.va/content/museivaticani/en/collezioni/musei/cappella-sistina.html", "reliability": "high"},
            {"topic": "Literature", "source": "Poetry Foundation", "url": "https://www.poetryfoundation.org/learn/glossary-terms", "reliability": "high"},

            {"topic": "Sports", "source": "Olympic.org", "url": "https://olympics.com/en/olympic-games/ancient-olympic-games", "reliability": "high"},
            {"topic": "Sports", "source": "Marathon History", "url": "https://www.britannica.com/sports/marathon-race", "reliability": "high"},
            {"topic": "Sports", "source": "FIFA", "url": "https://www.fifa.com/tournaments/mens/worldcup", "reliability": "high"},
            {"topic": "Sports", "source": "Tennis Grand Slams", "url": "https://www.britannica.com/sports/Grand-Slam-tennis", "reliability": "high"},
            {"topic": "Biology", "source": "Reaction Time Research", "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4456887/", "reliability": "high"},

            {"topic": "Biology", "source": "Evolution Research", "url": "https://www.britannica.com/science/evolution-scientific-theory", "reliability": "high"},
            {"topic": "Physics", "source": "Physics Textbook", "url": "https://www.britannica.com/science/atom", "reliability": "high"},
            {"topic": "Physics", "source": "Einstein Archives", "url": "https://www.britannica.com/science/theory-of-relativity", "reliability": "high"},
            {"topic": "Medicine", "source": "Medical History", "url": "https://www.britannica.com/science/penicillin", "reliability": "high"},
            {"topic": "Astronomy", "source": "Hubble Space Telescope", "url": "https://hubblesite.org/", "reliability": "high"},
            {"topic": "Biotechnology", "source": "CRISPR Research", "url": "https://www.broadinstitute.org/what-broad/areas-focus/project-spotlight/questions-and-answers-about-crispr", "reliability": "high"},
            {"topic": "Astrophysics", "source": "NASA Black Holes", "url": "https://www.nasa.gov/black-holes", "reliability": "high"},
            {"topic": "Geology", "source": "USGS Plate Tectonics", "url": "https://www.usgs.gov/programs/earthquake-hazards/plate-tectonics", "reliability": "high"},
            {"topic": "Archaeology", "source": "Radiocarbon Dating", "url": "https://www.britannica.com/science/radiocarbon-dating", "reliability": "high"},

            {"topic": "Linguistics", "source": "Language Statistics", "url": "https://www.ethnologue.com/insights/ethnologue200/", "reliability": "high"},
            {"topic": "Linguistics", "source": "Language Research", "url": "https://www.britannica.com/topic/Chinese-languages", "reliability": "high"},
            {"topic": "Psychology", "source": "Communication Research", "url": "https://www.verywellmind.com/types-of-nonverbal-communication-2795397", "reliability": "high"},
            {"topic": "History", "source": "Alphabet History", "url": "https://www.britannica.com/topic/alphabet", "reliability": "high"},

            {"topic": "Economics", "source": "Commodity Trading", "url": "https://www.investopedia.com/articles/investing/100615/4-countries-produce-most-coffee.asp", "reliability": "high"},
            {"topic": "Biology", "source": "Vision Research", "url": "https://www.nih.gov/news-events/nih-research-matters/how-many-colors-can-we-see", "reliability": "high"},
            {"topic": "Food Science", "source": "Honey Research", "url": "https://www.smithsonianmag.com/science-nature/the-science-behind-honeys-eternal-shelf-life-1218690/", "reliability": "high"},
            {"topic": "Marine Biology", "source": "Octopus Research", "url": "https://oceana.org/marine-life/octopus/", "reliability": "high"},
            {"topic": "Astronomy", "source": "Planetary Science", "url": "https://solarsystem.nasa.gov/planets/venus/overview/", "reliability": "high"},
            {"topic": "Health", "source": "Fitness Research", "url": "https://www.arthritis.org/health-wellness/healthy-living/physical-activity/walking/walk-around-world", "reliability": "high"},
            {"topic": "Physics", "source": "Acoustics", "url": "https://www.britannica.com/science/sound-physics", "reliability": "high"},
            {"topic": "Biology", "source": "Olfaction Research", "url": "https://www.science.org/doi/10.1126/science.1249168", "reliability": "high"},
        ]
        return sources


def create_enhanced_knowledge_base(enable_web_search: bool = True, 
                                   relevance_threshold: float = 0.7) -> EnhancedRAGRetriever:
    """Create RAG retriever with sample knowledge base"""
    retriever = EnhancedRAGRetriever(
        enable_web_search=enable_web_search,
        relevance_threshold=relevance_threshold
    )
    
    docs = KnowledgeBase.get_sample_documents()
    metadata = KnowledgeBase.get_sample_metadata()
    retriever.add_documents(docs, metadata)
    
    return retriever


if __name__ == "__main__":
    print("Testing Enhanced RAG with Reliability Ranking...")
    print("="*80)
    
    retriever = create_enhanced_knowledge_base(
        enable_web_search=True,
        relevance_threshold=0.7
    )
    
    test_queries = [
        "How long does Earth orbit the Sun?",
        "What are the latest AI developments in 2024?",
        "Who won the Nobel Prize in Physics 2024?",
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print('='*80)
        
        results = retriever.retrieve(query, k=3)
        
        for result in results:
            icon = "📚" if result['source_type'] == 'local' else "🌐"
            reliability = result.get('reliability', 0.5)
            print(f"\n{icon} Rank {result['rank']} | Relevance: {result['relevance']:.3f} | Reliability: {reliability:.2f}")
            print(f"Document: {result['document'][:200]}...")
            print(f"Source: {result['metadata'].get('source', 'Unknown')}")
        
        time.sleep(2)