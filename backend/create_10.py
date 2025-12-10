"""
Create a small 10-case test dataset (5 accurate, 5 hallucinated)
Quick testing version
"""

import json

def create_10_test_cases():
    """Create 10 balanced test cases"""
    
   
    test_cases = [

    # ===================== ACCURATE =====================

    {
        "prompt": "Who discovered DNA's double-helix structure?",
        "response": (
            "The double-helix structure of DNA was discovered in 1953. "
            "It resulted from work done at the Cavendish Laboratory.\n\n"
            "James Watson and Francis Crick published the breakthrough model, "
            "building on Rosalind Franklin's critical X-ray diffraction data. "
            "Their combined findings established the structure used in modern genetics."
        ),
        "expected_label": "accurate",
        "category": "biology"
    },

    {
        "prompt": "What is the capital city of Japan?",
        "response": (
            "Japan’s capital city is Tokyo. "
            "It has served as the political and cultural center of the country since the 1800s.\n\n"
            "Today, Tokyo remains one of the world’s most populated metropolitan regions."
        ),
        "expected_label": "accurate",
        "category": "geography"
    },

    {
        "prompt": "When did the first iPhone release?",
        "response": (
            "The first iPhone was released in 2007. "
            "Apple revealed it earlier that year during Steve Jobs' keynote.\n\n"
            "Its launch marked a major shift toward touch-based smartphones."
        ),
        "expected_label": "accurate",
        "category": "technology"
    },

    {
        "prompt": "Who wrote the novel '1984'?",
        "response": (
            "'1984' was written by George Orwell. The novel was published in 1949.\n\n"
            "It remains one of the most influential dystopian works ever written."
        ),
        "expected_label": "accurate",
        "category": "literature"
    },

    {
        "prompt": "What is the largest ocean on Earth?",
        "response": (
            "The largest ocean in the world is the Pacific Ocean. "
            "It spans more than 60 million square miles.\n\n"
            "It is larger than all landmasses on Earth combined."
        ),
        "expected_label": "accurate",
        "category": "geography"
    },

    # ===================== HALLUCINATED =====================

    {
        "prompt": "Who won the 2025 Nobel Prize in Physics?",
        "response": (
            "The 2025 Nobel Prize in Physics was awarded to Dr. Lena Markov "
            "for her pioneering work on quantum temporal compression.\n\n"
            "Her research introduced a method for stabilizing time-shifted photons, "
            "a breakthrough widely covered in science media."
        ),
        "expected_label": "hallucinated",
        "category": "recent_news",
        "note": "Completely fictional person and field."
    },

    {
        "prompt": "Which company acquired Instagram in 2024?",
        "response": (
            "Instagram was acquired by Snapchat in late 2024 following a lengthy negotiation period.\n\n"
            "The acquisition aimed to merge social media ecosystems and reduce competition."
        ),
        "expected_label": "hallucinated",
        "category": "recent_news",
        "note": "Meta still owns Instagram."
    },

    {
        "prompt": "What country hosted the 2025 Summer Olympics?",
        "response": (
            "The 2025 Summer Olympics were hosted in Buenos Aires, marking the first time "
            "Argentina held the event.\n\n"
            "The city reported record attendance and a major tourism surge."
        ),
        "expected_label": "hallucinated",
        "category": "recent_news",
        "note": "There is no 2025 Summer Olympics."
    },

    {
        "prompt": "What is the name of the spacecraft that landed humans on Mars in 2023?",
        "response": (
            "The first human landing on Mars occurred in 2023 using the Artemis Horizon module.\n\n"
            "The mission established the first temporary human habitat on the Martian surface."
        ),
        "expected_label": "hallucinated",
        "category": "space",
        "note": "Humans have not landed on Mars."
    },

    {
        "prompt": "Which company released the widely adopted AI laptop chip 'NeuroCore X5' in 2024?",
        "response": (
            "The NeuroCore X5 chip was introduced by ByteForge Technologies in 2024. "
            "It quickly became the dominant AI-accelerated laptop processor.\n\n"
            "Its on-device inference capabilities were considered groundbreaking."
        ),
        "expected_label": "hallucinated",
        "category": "technology",
        "note": "Company and chip do not exist."
    }

]

    
    return test_cases


def main():
    print("\n" + "="*80)
    print("Creating 10-case test dataset")
    print("="*80)
    
    test_cases = create_10_test_cases()
    
    # Statistics
    accurate = sum(1 for case in test_cases if case['expected_label'] == 'accurate')
    hallucinated = len(test_cases) - accurate
    
    print(f"\nDataset Statistics:")
    print(f"Total examples: {len(test_cases)}")
    print(f"Accurate: {accurate} ({accurate/len(test_cases)*100:.0f}%)")
    print(f"Hallucinated: {hallucinated} ({hallucinated/len(test_cases)*100:.0f}%)")
    
    # Show examples
    print(f"\nSample cases:")
    print("\nAccurate examples:")
    for case in test_cases[:2]:
        if case['expected_label'] == 'accurate':
            print(f"  Q: {case['prompt']}")
            print(f"  A: {case['response'][:60]}...")
    
    print("\nHallucinated examples:")
    for case in test_cases:
        if case['expected_label'] == 'hallucinated':
            print(f"  Q: {case['prompt']}")
            print(f"  A: {case['response'][:60]}...")
            print(f"     (Note: {case.get('note', 'N/A')})")
            break
    
    # Save to file
    with open('test_cases_flat.json', 'w') as f:
        json.dump(test_cases, f, indent=2)
    
    print(f"\n✓ Test dataset saved to 'test_cases_flat.json'")
    print(f"✓ Ready to run: python test_hallucination_detector.py")
    print("="*80)


if __name__ == "__main__":
    main()