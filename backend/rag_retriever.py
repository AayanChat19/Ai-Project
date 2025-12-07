import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import json
import os

class RAGRetriever:
    """
    RAG system using FAISS for retrieving relevant evidence
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        self.dimension = self.embedder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.metadata = []
        print(f"RAG Retriever initialized with dimension: {self.dimension}")
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
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
        
        print(f"Total documents in index: {len(self.documents)}")
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        if len(self.documents) == 0:
            return []
        
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        k = min(k, len(self.documents))
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append({
                    "document": self.documents[idx],
                    "score": float(distances[0][i]),
                    "metadata": self.metadata[idx] if idx < len(self.metadata) else {},
                    "rank": i + 1
                })
        
        return results
    
    def save_index(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        faiss.write_index(self.index, f"{path}.faiss")
        
        with open(f"{path}.json", 'w') as f:
            json.dump({
                "documents": self.documents,
                "metadata": self.metadata
            }, f)
        
        print(f"Index saved to {path}")
    
    def load_index(self, path: str):
        self.index = faiss.read_index(f"{path}.faiss")
        
        with open(f"{path}.json", 'r') as f:
            data = json.load(f)
            self.documents = data["documents"]
            self.metadata = data["metadata"]
        
        print(f"Index loaded from {path}. Total documents: {len(self.documents)}")


class KnowledgeBase:
    """Enhanced knowledge base with sources"""
    
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
            
            # Technology
            "The World Wide Web was invented by Tim Berners-Lee in 1989 at CERN.",
            "Artificial Intelligence refers to computer systems that can perform tasks normally requiring human intelligence.",
            "Smartphones typically use lithium-ion batteries for their high energy density and rechargeability.",
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
        ]
        return sources


def create_default_knowledge_base() -> RAGRetriever:
    """Create RAG retriever with enhanced knowledge base"""
    retriever = RAGRetriever()
    docs = KnowledgeBase.get_sample_documents()
    metadata = KnowledgeBase.get_sample_metadata()
    retriever.add_documents(docs, metadata)
    return retriever


if __name__ == "__main__":
    print("Testing Enhanced RAG Retriever...")
    retriever = create_default_knowledge_base()
    
    test_queries = [
        "How long does Earth take to orbit the Sun?",
        "What is the speed of light?",
        "Can you see the Great Wall from space?",
        "Do vaccines cause autism?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        results = retriever.retrieve(query, k=3)
        for result in results:
            print(f"\nRank {result['rank']} (Distance: {result['score']:.4f})")
            print(f"Document: {result['document']}")
            print(f"Source: {result['metadata'].get('source', 'Unknown')}")
            print(f"Topic: {result['metadata'].get('topic', 'Unknown')}")