import regex as re

from datasets import Dataset, DatasetDict
from parameters import class_names

# prepare mappings
id2label = {0: 'O'}
for i, class_name in enumerate(class_names, start=1):
    id2label[i] = 'B-' + class_name.capitalize()

label2id = {v: i for i, v in id2label.items()}

train_data = [
    {"id": "0", "tokens": ["Look", "at", "that", "antelope", "running"], "ner_tags": [0, 0, 0, 11, 0]},
    {"id": "1", "tokens": ["A", "badger", "is", "digging", "a", "hole"], "ner_tags": [0, 1, 0, 0, 0, 0]},
    {"id": "2", "tokens": ["I", "think", "that", "is", "a", "bat"], "ner_tags": [0, 0, 0, 0, 0, 2]},
    {"id": "3", "tokens": ["The", "bear", "is", "climbing", "the", "tree"], "ner_tags": [0, 3, 0, 0, 0, 0]},
    {"id": "4", "tokens": ["Watch", "out", "for", "the", "bee"], "ner_tags": [0, 0, 0, 0, 4]},
    {"id": "5", "tokens": ["I", "saw", "a", "shiny", "beetle", "today"], "ner_tags": [0, 0, 0, 0, 5, 0]},
    {"id": "6", "tokens": ["The", "bison", "herd", "is", "huge"], "ner_tags": [0, 6, 0, 0, 0]},
    {"id": "7", "tokens": ["Could", "that", "be", "a", "wild", "boar"], "ner_tags": [0, 0, 0, 0, 0, 7]},
    {"id": "8", "tokens": ["I", "love", "watching", "butterflies", "in", "spring"], "ner_tags": [0, 0, 0, 8, 0, 0]},
    {"id": "9", "tokens": ["My", "cat", "is", "sleeping", "again"], "ner_tags": [0, 9, 0, 0, 0]},
    {"id": "10", "tokens": ["That", "caterpillar", "will", "turn", "into", "a", "butterfly"], "ner_tags": [0, 10, 0, 0, 0, 0, 8]},
    {"id": "11", "tokens": ["Is", "that", "an", "antelope", "in", "the", "distance"], "ner_tags": [0, 0, 0, 11, 0, 0, 0]},
    {"id": "12", "tokens": ["A", "badger", "just", "crossed", "the", "road"], "ner_tags": [0, 1, 0, 0, 0, 0]},
    {"id": "13", "tokens": ["I", "heard", "a", "bat", "flying", "nearby"], "ner_tags": [0, 0, 0, 2, 0, 0]},
    {"id": "14", "tokens": ["The", "bear", "left", "paw", "prints"], "ner_tags": [0, 3, 0, 0, 0]},
    {"id": "15", "tokens": ["Be", "careful", "of", "that", "bee"], "ner_tags": [0, 0, 0, 0, 4]},
    {"id": "16", "tokens": ["This", "beetle", "looks", "so", "colorful"], "ner_tags": [0, 5, 0, 0, 0]},
    {"id": "17", "tokens": ["The", "bison", "moved", "slowly", "across", "the", "field"], "ner_tags": [0, 6, 0, 0, 0, 0, 0]},
    {"id": "18", "tokens": ["Did", "you", "see", "that", "boar", "near", "the", "trees"], "ner_tags": [0, 0, 0, 0, 7, 0, 0, 0]},
    {"id": "19", "tokens": ["A", "butterfly", "landed", "on", "my", "hand"], "ner_tags": [0, 8, 0, 0, 0, 0]},
    {"id": "20", "tokens": ["The", "caterpillar", "is", "crawling", "on", "the", "leaf"], "ner_tags": [0, 10, 0, 0, 0, 0, 0]}
]

validation_data = [
    {"id": "1", "tokens": ["Is", "that", "an", "antelope"], "ner_tags": [0, 0, 0, 11]},
    {"id": "2", "tokens": ["A", "badger", "is", "here"], "ner_tags": [0, 1, 0, 0]},
    {"id": "3", "tokens": ["I", "see", "a", "bat", "here"], "ner_tags": [0, 0, 0, 2, 0]},
    {"id": "4", "tokens": ["The", "bear", "is", "here"], "ner_tags": [0, 3, 0, 0]},
    {"id": "5", "tokens": ["It", "looks", "like", "a", "bee"], "ner_tags": [0, 0, 0, 0, 4]},
    {"id": "6", "tokens": ["This", "beetle", "looks", "nice"], "ner_tags": [0, 5, 0, 0]},
    {"id": "7", "tokens": ["The", "bison", "in", "the", "picture"], "ner_tags": [0, 6, 0, 0, 0]},
    {"id": "8", "tokens": ["Did", "you", "see", "that", "boar"], "ner_tags": [0, 0, 0, 0, 7]},
    {"id": "9", "tokens": ["A", "butterfly", "captured", "here"], "ner_tags": [0, 8, 0, 0]},
    {"id": "10", "tokens": ["The", "caterpillar", "is", "photographed", "here"], "ner_tags": [0, 10, 0, 0, 0]}
]

test_data = [
    {"id": "0", "tokens": ["The", "antelope", "grazed", "on", "the", "grass"], "ner_tags": [0, 11, 0, 0, 0, 0]},
    {"id": "1", "tokens": ["A", "badger", "dug", "under", "the", "fence"], "ner_tags": [0, 1, 0, 0, 0, 0]},
    {"id": "2", "tokens": ["The", "bat", "flapped", "its", "wings", "in", "the", "dark"], "ner_tags": [0, 2, 0, 0, 0, 0, 0, 0]},
    {"id": "3", "tokens": ["A", "bear", "was", "wandering", "through", "the", "forest"], "ner_tags": [0, 3, 0, 0, 0, 0, 0]},
    {"id": "4", "tokens": ["A", "bee", "buzzed", "around", "the", "flowers"], "ner_tags": [0, 4, 0, 0, 0, 0]},
    {"id": "5", "tokens": ["The", "beetle", "climbed", "up", "the", "tree"], "ner_tags": [0, 5, 0, 0, 0, 0]},
    {"id": "6", "tokens": ["The", "bison", "roamed", "the", "open", "plains"], "ner_tags": [0, 6, 0, 0, 0, 0]},
    {"id": "7", "tokens": ["The", "boar", "ran", "into", "the", "woods"], "ner_tags": [0, 7, 0, 0, 0, 0]},
    {"id": "8", "tokens": ["A", "butterfly", "fluttered", "by", "the", "pond"], "ner_tags": [0, 8, 0, 0, 0, 0]},
    {"id": "9", "tokens": ["The", "cat", "sat", "on", "the", "windowsill"], "ner_tags": [0, 9, 0, 0, 0, 0]},
    {"id": "10", "tokens": ["A", "caterpillar", "was", "eating", "a", "leaf"], "ner_tags": [0, 10, 0, 0, 0, 0]},
]

def split_text_with_punctuation(sentence: str) -> list:
    """
    Use regular expression to split on spaces but keep punctuation as separate tokens
    """
    tokens = re.findall(r'\w+|[?!.]', sentence)
    return tokens


def get_template_data(id2label: dict) -> dict:
    """
    Prepare data based on sentence templates. Can be used to enrich test dataset.
    """
    data = []
    animals = [x[2:].lower() for x in id2label.values() if x != 'O']

    for idx, animal in enumerate(animals):
        sentence_templates = [
            f"I see {animal} here",
            f"There's {animal} in the picture",
            f"Do we have {animal} here?",
            f"Is there {animal}?",
            f"This picture has image of {animal}",
            f"Nice picture. It is {animal}, right?",
            f"Is it {animal}?",
            f"{animal} looks cool here",
            f"Funny to see {animal} here",
            f"Look, what a beautiful {animal}!",
            f"{animal} looks very cool here, right?",
            f"I like this picture. It has my favorite animal - {animal}.",
            f"Is that {animal}?",
            f"Looks like {animal} to me, right?",
            f"It has this funny looking {animal}",
            f"There's {animal} in this pic",
            f"Does it has {animal}?",
            f"Picture has {animal}",
            f"There is {animal} in the picture",
            f"It is {animal}",
            f"See this {animal}?",
            f"We have here rare kind of animal - {animal}",
        ]

        # Process each sentence
        for sentence in sentence_templates:
            tokens = split_text_with_punctuation(sentence)
            ner_tags = [0] * len(tokens)  # Initialize ner_tags with 'O'

            # Find the position of the animal and assign appropriate NER tag
            ner_tags[tokens.index(animal)] = idx + 1  # Using 1-based index from id2label

            # Create data dictionary for this sentence
            data.append({
                "id": str(idx),
                "tokens": tokens,
                "ner_tags": ner_tags
            })

    return data


def prepare_dataset() -> DatasetDict:
    """
    Prepare dataset that contains train, validation and test datasets
    """
    data = get_template_data(id2label)
    new_train_data = train_data + data

    # convert to datasets
    train_dataset = Dataset.from_list(new_train_data)
    validation_dataset = Dataset.from_list(validation_data)
    test_dataset = Dataset.from_list(test_data)

    # Combine into DatasetDict
    dataset = DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset
    })

    return dataset
