import os

import evaluate
import numpy as np
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification

from prepare_dataset import prepare_dataset, id2label, label2id
from parameters import ner_model_name, pretrained_model_name

os.environ["WANDB_DISABLED"] = "true"


def tokenize_and_align_labels(examples: list) -> list:
    """
    Tokenize input and assign label -100 for certain cases:
    - start-end tokens [CLS] and [SEP]
    - sub-tokens of a word (1st will be labeled, next ones will be set to -100)
    """
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p: list) -> dict:
    """
    Compute metrics: precision, recall, f1, accuracy
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

dataset = prepare_dataset()

label_list = list(id2label.values())
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

seqeval = evaluate.load("seqeval")

model = AutoModelForTokenClassification.from_pretrained(
    pretrained_model_name, num_labels=len(label_list), id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir=ner_model_name,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model(ner_model_name)
