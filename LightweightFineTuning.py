#!/usr/bin/env python
# coding: utf-8

# # Lightweight Fine-Tuning Project

# PEFT technique: I used LoRA (Low-Rank) configuration. I evaluated the BERT model initially and then fine-tuned it on a subset of the BBC News (SetFit/bbc-news) dataset from Hugging Face. The fine-tuning process adapts the model more closely to the task.
# 
# Model: The base model I used is "bert-base-uncased". This model is utilized for both the initial evaluation and the PEFT process (training and evaluation.
# 
# Evaluation approach: The evaluation is performed using the Trainer class from the Hugging Face's Transformers library. The evaluation strategy is set to "epoch," meaning an evaluation is oerformed after each training epoch. The evaluation metric is accuracy.
# 
# Fine-tuning dataset: The fine-tuning dataset is the BBC News (SetFit/bbc-news) dataset from Hugging Face. To expedite the process, I used a subset of 1000 samples from the dataset. The dataset is pre-proceesed using the "bert-base-uncased" tokenizer.

# ## Loading and Evaluating a Foundation Model
# 
# Load a pre-trained Hugging Face model and evaluate its performance prior to fine-tuning.

# In[1]:


get_ipython().system('pip install -q "datasets==2.15.0"')
get_ipython().system('pip install datasets transformers')
get_ipython().system('pip install tabulate')


# In[2]:


from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from tabulate import tabulate
import numpy as np
import pandas as pd


# In[3]:


# load the dataset
dataset = load_dataset("SetFit/bbc-news")

# access the train and test splits - splits where already available from the dataset
train_split = dataset["train"]
test_split = dataset["test"]

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Had to use a subset beacause the full dataset is too large to train and evaluate on Workspace
# Number of samples you want to use from the dataset
num_samples = 1000

# Create a smaller subset of the train and test datasets
small_train_split = train_split.select(range(num_samples))
small_test_split = test_split.select(range(num_samples))

# Tokenize the smaller datasets
small_tokenized_train = small_train_split.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)
small_tokenized_test = small_test_split.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)

# Define id2label and label2id
id2label = {0: "tech", 1: "business", 2: "sport", 3: "entertainment", 4: "politics"}
label2id = {"tech": 0, "business": 1, "sport": 2, "entertainment": 3, "politics": 4}

# load model for sequence classification and define label mapping based on categories
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5,
       id2label = id2label,
       label2id = label2id)

# compute metrics based on the actual labels and the model predictions
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}

# define training arguments for evaluation
training_args = TrainingArguments(
output_dir = "./bbc_news_results",
per_device_eval_batch_size = 10,
evaluation_strategy = "epoch",
save_strategy="epoch",
load_best_model_at_end=True)


# In[4]:


# trainer for pre fine-tuning evaluation
trainer = Trainer(
model=model,
args=training_args,
train_dataset=small_tokenized_train,
eval_dataset=small_tokenized_test,
data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
compute_metrics=compute_metrics)


# In[5]:


# evaluate before fine-tuning
pre_eval_results = trainer.evaluate()


# In[6]:


# create a table for a more readable result
# referenced a website to create the table
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.from_dict.html
df_results = pd.DataFrame.from_dict(pre_eval_results, orient="index")

# referenced a webiste to display results for a nice table format
#https://pypi.org/project/tabulate/
formatted_df = tabulate(df_results, tablefmt="presto")

print(f"Pre-Fine-Tuning Evaluation Results:\n \n{formatted_df}")


# In[7]:


# create a dataset for visual review with the text, predictions, and labels
visual_review = small_tokenized_test.select([0, 5, 34, 85, 107, 268, 436])
results = trainer.predict(visual_review)

# find a resource for this whole thing
mapped_label = {0: "tech", 1: "business", 2: "sport", 3: "entertainment", 4: "politics"}

df = pd.DataFrame({
    "text": [item["text"] for item in visual_review],
    "predictions": [mapped_label[p] for p in results.predictions.argmax(axis=1)],
    "true labels": [mapped_label[l] for l in results.label_ids],
})

pd.set_option("display.max_colwidth", 80)
df


# ## Performing Parameter-Efficient Fine-Tuning
# 
# Create a PEFT model from the loaded model, run a training loop, and save the PEFT model weights.

# In[8]:


from peft import LoraConfig, get_peft_model, TaskType


# In[9]:


# initialize LoraConfig and loaded model
# Referenced a website to get the LoRA Configuration
# https://www.kaggle.com/code/anthonynam/lora-fine-tuning-with-distilbert-7-prompts-v4
config = LoraConfig(
    r=8,
    task_type=TaskType.SEQ_CLS
)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5,
        id2label = id2label,
        label2id = label2id)

# create PEFT model
lora_model = get_peft_model(model, config)

# unfreeze model parameters
for param in lora_model.base_model.parameters():
    param.requires_grad = True
    
# define arguments for training and evaluation
# used this website for information on logging_dir and logging_strategy
# https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments
training_args = TrainingArguments(
output_dir = "./peft_results",
logging_dir = "./peft_logs",
learning_rate = 2e-5,
per_device_train_batch_size = 10,
per_device_eval_batch_size = 10,
evaluation_strategy = "epoch",
save_strategy = "epoch",
logging_strategy = "epoch",
num_train_epochs = 5,
weight_decay=0.01,
load_best_model_at_end = True)


# In[10]:


# trainer for the Peft model with smaller datasets
lora_trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=small_tokenized_train.rename_column('label', 'labels'),
    eval_dataset=small_tokenized_test.rename_column('label', 'labels'),
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics)


# In[11]:


# train the model
lora_trainer.train()


# In[12]:


# save the PEFT model weights
lora_model.save_pretrained("bert-lora")


# ## Performing Inference with a PEFT Model
# 
# Load the saved PEFT model weights and evaluate the performance of the trained PEFT model.

# In[13]:


from peft import AutoPeftModelForSequenceClassification


# In[14]:


# load the saved PEFT model
lora_model = AutoPeftModelForSequenceClassification.from_pretrained("bert-lora", num_labels=5)


# In[15]:


# evaluate the PEFT model
post_eval_results = lora_trainer.evaluate()


# In[16]:


df_results = pd.DataFrame.from_dict(post_eval_results, orient="index")

formatted_df = tabulate(df_results, tablefmt="plain")

print(f"Post-Fine-Tuning Evaluation Results:\n{formatted_df}")


# In[17]:


# create a dataset for visual review with the text, predictions, and labels
visual_review = small_tokenized_test.select([0, 5, 34, 85, 107, 268, 436])
results = lora_trainer.predict(visual_review)

df = pd.DataFrame({
    "text": [item["text"] for item in visual_review],
    "predictions": results.predictions.argmax(axis=1),
    "label": results.label_ids,
})

# show all the cells
pd.set_option("display.max_colwidth", 80)
df


# In[20]:


# create dataframe for accuracy comparison
comparison_dict = {
    "Metrics": ["Accuracy"],
    "Pre-fine-tuning results": [pre_eval_results["eval_accuracy"]],
    "Post-fine-tuning results": [post_eval_results["eval_accuracy"]],
}

#create a table for a more readable result
df_results = pd.DataFrame.from_dict(comparison_dict, orient="index")

formatted_df = tabulate(df_results, tablefmt="plain")

print(f"Comparison of Pre-Fine-Tuning and Post-Fine-Tuning Evaluation Results:\n\n{formatted_df}")


# The PEFT model had a much higher accuracy than the model before fine-tuning. The model drastically improved from having an 20.2% accuracy before fine-tuning to having a 98% accuracy after fine-tuning.
