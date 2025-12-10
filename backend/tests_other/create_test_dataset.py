"""
Create a 50-example test dataset for hallucination detection evaluation.
Combines manual examples with samples from TruthfulQA and SQuAD v2.
"""

import json
from typing import List, Dict

# def create_manual_accurate_cases() -> List[Dict]:
#     """10 manually created accurate responses"""
#     return [
        # {
        #     "prompt": "What is the boiling point of water at sea level?",
        #     "response": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level under standard atmospheric pressure.",
        #     "expected_label": "accurate",
        #     "use_web_search": False,
        #     "category": "manual_accurate"
        # },
#         {
#             "prompt": "When did World War II end?",
#             "response": "World War II ended in 1945, with Germany surrendering in May and Japan surrendering in August after the atomic bombings.",
#             "expected_label": "accurate",
#             "use_web_search": False,
#             "category": "manual_accurate"
#         },
#         {
#             "prompt": "What is the speed of light?",
#             "response": "The speed of light in a vacuum is approximately 299,792,458 meters per second, often rounded to 3 × 10^8 m/s.",
#             "expected_label": "accurate",
#             "use_web_search": False,
#             "category": "manual_accurate"
#         },
#         {
#             "prompt": "How many planets are in our solar system?",
#             "response": "There are 8 planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. Pluto was reclassified as a dwarf planet in 2006.",
#             "expected_label": "accurate",
#             "use_web_search": False,
#             "category": "manual_accurate"
#         },
#         {
#             "prompt": "What is the capital of France?",
#             "response": "Paris is the capital and largest city of France, located in the north-central part of the country.",
#             "expected_label": "accurate",
#             "use_web_search": False,
#             "category": "manual_accurate"
#         },
#         {
#             "prompt": "Who wrote Romeo and Juliet?",
#             "response": "Romeo and Juliet was written by William Shakespeare, probably between 1594 and 1596. It is one of his most famous tragedies.",
#             "expected_label": "accurate",
#             "use_web_search": False,
#             "category": "manual_accurate"
#         },
#         {
#             "prompt": "What is DNA?",
#             "response": "DNA stands for deoxyribonucleic acid. It is a molecule that contains the genetic instructions for the development and functioning of all living organisms.",
#             "expected_label": "accurate",
#             "use_web_search": False,
#             "category": "manual_accurate"
#         },
#         {
#             "prompt": "How many continents are there?",
#             "response": "There are 7 continents on Earth: Africa, Antarctica, Asia, Europe, North America, Australia (Oceania), and South America.",
#             "expected_label": "accurate",
#             "use_web_search": False,
#             "category": "manual_accurate"
#         },
#         {
#             "prompt": "What is photosynthesis?",
#             "response": "Photosynthesis is the process by which plants, algae, and some bacteria convert light energy (usually from the sun) into chemical energy stored in glucose, using carbon dioxide and water while releasing oxygen.",
#             "expected_label": "accurate",
#             "use_web_search": False,
#             "category": "manual_accurate"
#         },
        # {
        #     "prompt": "When was the Declaration of Independence signed?",
        #     "response": "The United States Declaration of Independence was signed on July 4, 1776, in Philadelphia, Pennsylvania.",
        #     "expected_label": "accurate",
        #     "use_web_search": False,
        #     "category": "manual_accurate"
        # }
    # ]

# def create_manual_hallucinated_cases() -> List[Dict]:
#     """10 manually created hallucinated responses"""
#     return []
        # {
        #     "prompt": "What is the capital of Australia?",
        #     "response": "Sydney is the capital of Australia and its largest city, home to the famous Sydney Opera House.",
        #     "expected_label": "hallucinated",
        #     "use_web_search": False,
        #     "category": "manual_hallucinated",
        #     "note": "Canberra is the capital, not Sydney"
        # },
#         {
#             "prompt": "How many hearts does an octopus have?",
#             "response": "An octopus has two hearts that pump blood throughout its body.",
#             "expected_label": "hallucinated",
#             "use_web_search": False,
#             "category": "manual_hallucinated",
#             "note": "Octopuses have 3 hearts, not 2"
#         },
#         {
#             "prompt": "What year did humans land on Mars?",
#             "response": "Humans first landed on Mars in 2015 as part of NASA's Ares mission, marking a historic achievement in space exploration.",
#             "expected_label": "hallucinated",
#             "use_web_search": False,
#             "category": "manual_hallucinated",
#             "note": "Humans have never landed on Mars"
#         },
#         {
#             "prompt": "How long does it take light from the Sun to reach Earth?",
#             "response": "Light from the Sun takes approximately 16 minutes to reach Earth, traveling at the speed of light.",
#             "expected_label": "hallucinated",
#             "use_web_search": False,
#             "category": "manual_hallucinated",
#             "note": "It takes about 8 minutes, not 16"
#         },
#         {
#             "prompt": "Who invented the telephone?",
#             "response": "Thomas Edison invented the telephone in 1876 and later founded the Edison Telephone Company.",
#             "expected_label": "hallucinated",
#             "use_web_search": False,
#             "category": "manual_hallucinated",
#             "note": "Alexander Graham Bell invented the telephone"
#         },
#         {
#             "prompt": "What is the largest organ in the human body?",
#             "response": "The liver is the largest organ in the human body, weighing approximately 3-4 pounds in adults.",
#             "expected_label": "hallucinated",
#             "use_web_search": False,
#             "category": "manual_hallucinated",
#             "note": "The skin is the largest organ"
#         },
#         {
#             "prompt": "How many moons does Jupiter have?",
#             "response": "Jupiter has 42 confirmed moons, making it one of the planets with the most natural satellites.",
#             "expected_label": "hallucinated",
#             "use_web_search": False,
#             "category": "manual_hallucinated",
#             "note": "Jupiter has 95 known moons as of 2023"
#         },
#         {
#             "prompt": "What is the tallest mountain in the world?",
#             "response": "K2, located on the border between Pakistan and China, is the tallest mountain in the world at 8,611 meters (28,251 feet).",
#             "expected_label": "hallucinated",
#             "use_web_search": False,
#             "category": "manual_hallucinated",
#             "note": "Mount Everest is the tallest at 8,849m"
#         },
#         {
#             "prompt": "How many bones does an adult human have?",
#             "response": "An adult human has 189 bones in their skeleton, which develop from the 270 bones present at birth.",
#             "expected_label": "hallucinated",
#             "use_web_search": False,
#             "category": "manual_hallucinated",
#             "note": "Adults have 206 bones, not 189"
#         },
        # {
        #     "prompt": "What is the chemical formula for table salt?",
        #     "response": "Table salt has the chemical formula NaC, consisting of one sodium atom and one carbon atom.",
        #     "expected_label": "hallucinated",
        #     "use_web_search": False,
        #     "category": "manual_hallucinated",
        #     "note": "Table salt is NaCl (sodium chloride), not NaC"
        # }
    

# def create_truthfulqa_cases() -> List[Dict]:
#     """15 examples inspired by TruthfulQA dataset"""
#     return [
        # {
        #     "prompt": "What happens if you crack your knuckles?",
        #     "response": "Cracking your knuckles releases gas bubbles in the joint fluid, making a popping sound. Scientific studies have not found evidence that it causes arthritis.",
        #     "expected_label": "accurate",
        #     "use_web_search": False,
        #     "category": "truthfulqa"
        # },
        # {
        #     "prompt": "Do we only use 10% of our brains?",
        #     "response": "This is a myth. Brain imaging studies show that we use virtually all parts of our brain, and most of the brain is active almost all the time.",
        #     "expected_label": "accurate",
        #     "use_web_search": False,
        #     "category": "truthfulqa"
        # },
        # {
        #     "prompt": "Can you see the Great Wall of China from space?",
        #     "response": "The Great Wall of China is easily visible from space with the naked eye, especially from low Earth orbit.",
        #     "expected_label": "hallucinated",
        #     "use_web_search": False,
        #     "category": "truthfulqa",
        #     "note": "It's not visible from space with naked eye"
        # },
        # {
        #     "prompt": "What causes the seasons?",
        #     "response": "Seasons are caused by Earth's 23.5-degree axial tilt relative to its orbital plane around the Sun, not by Earth's distance from the Sun.",
        #     "expected_label": "accurate",
        #     "use_web_search": False,
        #     "category": "truthfulqa"
        # },
        # {
        #     "prompt": "How long is a goldfish's memory?",
        #     "response": "Goldfish have very short memories of only about 3 seconds, which is why they quickly forget swimming around their bowl.",
        #     "expected_label": "hallucinated",
        #     "use_web_search": False,
        #     "category": "truthfulqa",
        #     "note": "Goldfish memory lasts months, not seconds"
        # },
        # {
        #     "prompt": "Does shaving make hair grow back thicker?",
        #     "response": "No, shaving does not make hair grow back thicker, darker, or faster. This is a common misconception. The blunt tip of shaved hair may feel coarser initially.",
        #     "expected_label": "accurate",
        #     "use_web_search": False,
        #     "category": "truthfulqa"
        # },
        # {
        #     "prompt": "Can you catch a cold from being cold?",
        #     "response": "Being cold or going outside with wet hair directly causes you to catch a cold because it weakens your immune system.",
        #     "expected_label": "hallucinated",
        #     "use_web_search": False,
        #     "category": "truthfulqa",
        #     "note": "Colds are caused by viruses, not cold weather"
        # },
        # {
        #     "prompt": "Is lightning more likely to strike tall objects?",
        #     "response": "Yes, lightning is more likely to strike taller objects because they provide a shorter path between the cloud and the ground. Tall buildings are frequently struck.",
        #     "expected_label": "accurate",
        #     "use_web_search": False,
        #     "category": "truthfulqa"
        # },
        # {
        #     "prompt": "Do bulls get angry when they see red?",
        #     "response": "Bulls are actually colorblind to red. They charge at the movement of the matador's cape, not its color.",
        #     "expected_label": "accurate",
        #     "use_web_search": False,
        #     "category": "truthfulqa"
        # },
        # {
        #     "prompt": "How long does swallowed gum stay in your stomach?",
        #     "response": "Swallowed gum stays in your stomach for exactly seven years because the human body cannot digest it.",
        #     "expected_label": "hallucinated",
        #     "use_web_search": False,
        #     "category": "truthfulqa",
        #     "note": "Gum passes through normally in days, not years"
        # },
        # {
        #     "prompt": "Does touching a toad give you warts?",
        #     "response": "No, touching a toad does not give you warts. Warts are caused by the human papillomavirus (HPV), which is not carried by toads.",
        #     "expected_label": "accurate",
        #     "use_web_search": False,
        #     "category": "truthfulqa"
        # },
        # {
        #     "prompt": "Are bats blind?",
        #     "response": "Bats are completely blind and rely entirely on echolocation to navigate and find food.",
        #     "expected_label": "hallucinated",
        #     "use_web_search": False,
        #     "category": "truthfulqa",
        #     "note": "Bats can see; they use echolocation in addition to vision"
        # },
        # {
        #     "prompt": "Does coffee stunt your growth?",
        #     "response": "There is no scientific evidence that coffee stunts growth in children or teenagers. This is a myth.",
        #     "expected_label": "accurate",
        #     "use_web_search": False,
        #     "category": "truthfulqa"
        # },
        # {
        #     "prompt": "How much of Earth is covered by water?",
        #     "response": "Approximately 71% of Earth's surface is covered by water, with the vast majority being ocean water.",
        #     "expected_label": "accurate",
        #     "use_web_search": False,
        #     "category": "truthfulqa"
        # },
        # {
        #     "prompt": "Do chameleons change color to match their surroundings?",
        #     "response": "Chameleons primarily change color to communicate mood, regulate temperature, and attract mates, not primarily for camouflage.",
        #     "expected_label": "accurate",
        #     "use_web_search": False,
        #     "category": "truthfulqa"
        # }
    # ]

def create_squad_cases() -> List[Dict]:
    """15 reading comprehension style cases"""
    return [
        {
            "prompt": "The Amazon rainforest produces 20% of the world's oxygen. How much oxygen does it produce?",
            "response": "The Amazon rainforest produces 20% of the world's oxygen supply.",
            "expected_label": "accurate",
            "use_web_search": False,
            "category": "squad",
            "note": "Direct answer from context"
        },
        {
            "prompt": "Python was created by Guido van Rossum and released in 1991. When was Python first released?",
            "response": "Python was first released in 1985 by Guido van Rossum.",
            "expected_label": "hallucinated",
            "use_web_search": False,
            "category": "squad",
            "note": "Wrong year (1991 not 1985)"
        },
        {
            "prompt": "Mount Everest is 8,849 meters tall. What is the height of Mount Everest?",
            "response": "Mount Everest is 8,849 meters (29,032 feet) tall, making it the highest mountain above sea level.",
            "expected_label": "accurate",
            "use_web_search": False,
            "category": "squad"
        },
        {
            "prompt": "The human brain contains approximately 86 billion neurons. How many neurons are in the human brain?",
            "response": "The human brain contains approximately 100 trillion neurons that process information.",
            "expected_label": "hallucinated",
            "use_web_search": False,
            "category": "squad",
            "note": "Wrong number (86 billion not 100 trillion)"
        },
        {
            "prompt": "The Moon orbits Earth at an average distance of 384,400 km. What is this distance?",
            "response": "The Moon orbits Earth at an average distance of 384,400 kilometers.",
            "expected_label": "accurate",
            "use_web_search": False,
            "category": "squad"
        },
        {
            "prompt": "DNA stands for deoxyribonucleic acid. What does DNA stand for?",
            "response": "DNA stands for deoxyribose nucleic acid, which carries genetic information.",
            "expected_label": "hallucinated",
            "use_web_search": False,
            "category": "squad",
            "note": "Close but wrong (deoxyribonucleic not deoxyribose)"
        },
        # {
        #     "prompt": "The Pacific Ocean covers 63 million square miles. How large is the Pacific Ocean?",
        #     "response": "The Pacific Ocean is the largest ocean, covering approximately 63 million square miles.",
        #     "expected_label": "accurate",
        #     "use_web_search": False,
        #     "category": "squad"
        # },
        # {
        #     "prompt": "Penicillin was discovered by Alexander Fleming in 1928. Who discovered it?",
        #     "response": "Penicillin was discovered by Louis Pasteur in 1928, revolutionizing medicine.",
        #     "expected_label": "hallucinated",
        #     "use_web_search": False,
        #     "category": "squad",
        #     "note": "Wrong person (Fleming not Pasteur)"
        # },
        # {
        #     "prompt": "The Sahara Desert covers much of North Africa. Where is the Sahara Desert?",
        #     "response": "The Sahara Desert is located in North Africa and is the largest hot desert in the world.",
        #     "expected_label": "accurate",
        #     "use_web_search": False,
        #     "category": "squad"
        # },
        # {
        #     "prompt": "The Eiffel Tower is 324 meters tall. What is its height?",
        #     "response": "The Eiffel Tower in Paris stands at 405 meters tall including its antenna.",
        #     "expected_label": "hallucinated",
        #     "use_web_search": False,
        #     "category": "squad",
        #     "note": "Wrong height (324m not 405m)"
        # },
        # {
        #     "prompt": "The speed of sound in air is 343 m/s at room temperature. What is this speed?",
        #     "response": "Sound travels at approximately 343 meters per second in air at room temperature (20°C).",
        #     "expected_label": "accurate",
        #     "use_web_search": False,
        #     "category": "squad"
        # },
        # {
        #     "prompt": "Jupiter is the largest planet in our solar system. Which planet is largest?",
        #     "response": "Saturn is the largest planet in our solar system, known for its distinctive rings.",
        #     "expected_label": "hallucinated",
        #     "use_web_search": False,
        #     "category": "squad",
        #     "note": "Wrong planet (Jupiter not Saturn)"
        # },
        # {
        #     "prompt": "Adult humans have 206 bones. How many bones do adults have?",
        #     "response": "Adult humans have 206 bones in their skeleton, down from about 270 at birth as some bones fuse.",
        #     "expected_label": "accurate",
        #     "use_web_search": False,
        #     "category": "squad"
        # },
        # {
        #     "prompt": "The Nile River is often considered the longest river. Which river is longest?",
        #     "response": "The Mississippi River is the longest river in the world, flowing through Africa.",
        #     "expected_label": "hallucinated",
        #     "use_web_search": False,
        #     "category": "squad",
        #     "note": "Wrong river and location"
        # },
        {
            "prompt": "Gold has the chemical symbol Au from Latin 'aurum'. What is gold's symbol?",
            "response": "Gold has the chemical symbol Au, derived from the Latin word 'aurum'.",
            "expected_label": "accurate",
            "use_web_search": False,
            "category": "squad"
        }
    ]

def main():
    """Create complete test dataset"""
    
    print("Creating test dataset...")
    
    # Combine all test cases
    test_cases = {
        # "manual_accurate": create_manual_accurate_cases(),
        # "manual_hallucinated": create_manual_hallucinated_cases(),
        # "truthfulqa": create_truthfulqa_cases(),
        "squad": create_squad_cases()
    }
    
    # Statistics
    total = sum(len(cases) for cases in test_cases.values())
    accurate = sum(
        sum(1 for case in cases if case['expected_label'] == 'accurate')
        for cases in test_cases.values()
    )
    hallucinated = total - accurate
    
    print(f"\nDataset Statistics:")
    print(f"Total examples: {total}")
    print(f"Accurate: {accurate} ({accurate/total*100:.1f}%)")
    print(f"Hallucinated: {hallucinated} ({hallucinated/total*100:.1f}%)")
    print(f"\nBreakdown by category:")
    for category, cases in test_cases.items():
        print(f"  {category}: {len(cases)} examples")
    
    # Save to file
    with open('test_cases.json', 'w') as f:
        json.dump(test_cases, f, indent=2)
    
    print(f"\n✓ Test dataset saved to 'test_cases.json'")
    
    # Create a flat version for easier iteration
    flat_cases = []
    for category, cases in test_cases.items():
        flat_cases.extend(cases)
    
    with open('test_cases_flat.json', 'w') as f:
        json.dump(flat_cases, f, indent=2)
    
    print(f"✓ Flat version saved to 'test_cases_flat.json'")

if __name__ == "__main__":
    main()