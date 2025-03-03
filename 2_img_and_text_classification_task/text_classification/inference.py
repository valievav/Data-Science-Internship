import os

from transformers import pipeline

from parameters import ner_model_name

os.environ["WANDB_DISABLED"] = "true"

# Load the pipeline
classifier = pipeline("ner", model=ner_model_name, tokenizer=ner_model_name)
classifier.model.eval()

# Inference
text = "There's a badger in this picture"
results = classifier(text)
print(results)
