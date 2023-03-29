from .base import BaseModel

import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from transformers import Trainer, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, EarlyStoppingCallback


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        inputs["labels"] = inputs["labels"].type(torch.LongTensor).to("cuda:0")
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]).cuda())
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

class BERTBase(BaseModel):
    def __init__(self, model_path = None, model_args={}, suffix = ''):

        super(BERTBase, self).__init__()
        self.training_args = TrainingArguments(output_dir = '../checkpoints')

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        if model_path is None:
            self.model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels = 2)

    def load(self, model_path, suffix = ''):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path + '/' + self.name + suffix, num_labels = 2).to('cuda:0')

    def save(self, model_path, suffix = ''):
        self.model.save_pretrained(model_path + '/' + self.name + suffix)

    def predict(self, df):
        input_texts = self.extract_features(df)
        predictions = np.zeros((len(input_texts), 2))

        for k, text in enumerate(input_texts):
            patent_bert_inputs = self.tokenizer(text, padding = True, truncation = True, max_length = 512, return_tensors = "pt").to('cuda:0')
            patent_bert_outputs = self.model(**patent_bert_inputs)

            patent_bert_output = patent_bert_outputs[0]
            predictions[k, :] = patent_bert_output.cpu().detach().numpy()[0]

        return predictions[:, 1]

    def train(self, df_train, df_eval, labels_train, labels_eval):
        train_texts = self.extract_features(df_train)
        eval_texts = self.extract_features(df_eval)

        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True, max_length=512)
        eval_encodings = self.tokenizer(eval_texts, truncation=True, padding=True, max_length=512)

        train_dataset = BaseDataset(train_encodings, labels_train)
        eval_dataset = BaseDataset(eval_encodings, labels_eval)

        trainer = WeightedTrainer(
            model = self.model,
            args = self.training_args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            compute_metrics = BERTBase.compute_metrics,
            callbacks = [EarlyStoppingCallback(early_stopping_patience = 3)]
        )

        trainer.train()

    @staticmethod
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average = 'binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def extract_features(self, df_train, df_eval):
        return [], []


class BERTExample(BERTBase):
    def __init__(self, model_path = None, model_args={}, suffix = ''):
        super(BERTExample, self).__init__(model_path, suffix)

        self.name = 'BERT_Example'
        self.description = 'BERT with basic text'

        if model_path is not None:
            self.load(model_path, suffix)

        self.training_args = TrainingArguments(
            output_dir = './checkpoints',
            num_train_epochs = 26,
            per_device_train_batch_size = 4,
            per_device_eval_batch_size = 4,
            warmup_steps = 100,
            weight_decay = 0.001,
            logging_dir = './logs',
            evaluation_strategy = "steps",
            save_strategy = "steps",
            eval_steps = 100,
            gradient_accumulation_steps = 4,
            logging_steps = 10000,
            save_steps = 100,
            learning_rate = 2e-5,
            load_best_model_at_end = True,
            metric_for_best_model = 'f1',
        )

    def extract_features(self, df):
        return df['texts'].tolist()


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
