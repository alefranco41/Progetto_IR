import torch
import gc
from datasets import load_dataset, load_metric, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
import numpy as np
from huggingface_hub import login
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re


login() #token: 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

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

nltk.download('stopwords')
nltk.download('punkt')
porter = PorterStemmer()  # Inizializza il Porter Stemmer

stop_words = set(stopwords.words('english'))  # Inizializza l'insieme delle stopwords in lingua inglese

def preprocess_function(examples):
    review_titles = examples["Review_Title"]
    if isinstance(review_titles, list):
        review_titles = [str(title) for title in review_titles]  # Convert all elements to strings
    else:
        review_titles = [str(review_titles)]

    inputs = tokenizer(review_titles, truncation=True)
    labels = [0 if rating < 4 else 1 for rating in examples["Rating"]]
    inputs["labels"] = labels
    return inputs



def compute_metrics(eval_pred):
   load_accuracy = load_metric("accuracy")
   load_f1 = load_metric("f1")
  
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   f1 = load_f1.compute(predictions=predictions, references=labels, average="macro")["f1"]
   return {"accuracy": accuracy, "f1": f1}

class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss


if torch.cuda.is_available():
    print("GPU disponibile. PyTorch utilizzerà la GPU per l'addestramento.")
else:
    print("GPU non disponibile. PyTorch utilizzerà la CPU per l'addestramento.")
    
car_dataset = load_dataset("florentgbelidji/car-reviews")

num_samples = len(car_dataset["train"])
split_index = int(0.9 * num_samples)  # 90% del dataset per l'addestramento, 10% per il test

train_dataset = Dataset.from_dict({"Review_Title": car_dataset["train"]["Review_Title"][:split_index], "Rating": car_dataset["train"]["Rating"][:split_index]})
test_dataset = Dataset.from_dict({"Review_Title": car_dataset["train"]["Review_Title"][split_index:], "Rating": car_dataset["train"]["Rating"][split_index:]})


tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
repo_name = "car_sentiment"


training_args = TrainingArguments(
   output_dir=repo_name,
   learning_rate=2e-5,
   per_device_train_batch_size=8,
   per_device_eval_batch_size=8,
   num_train_epochs=2,
   weight_decay=0.01,
   save_strategy="epoch",
   push_to_hub=True,
)
 
trainer = MyTrainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_train,
   eval_dataset=tokenized_test,
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics,
)

torch.cuda.empty_cache()


trainer.train()
print(trainer.evaluate())
trainer.push_to_hub()



