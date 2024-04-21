import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForQuestionAnswering, Trainer, TrainingArguments, TrainerCallback
import deepspeed
import argparse
import json
from tqdm import tqdm
import numpy as np
import subprocess
import time

transformers.logging.set_verbosity_error()  # Only show errors, not warnings



class SquadDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.tokenizer = tokenizer
        contexts, questions, answers = self.read_squad(path)
        answers = self.add_end_idx(contexts, answers)
        self.encodings = tokenizer(contexts, questions, truncation=True, padding='max_length', max_length=384)
        self.encodings = self.update_start_end_positions(self.encodings, answers)

    def read_squad(self, path):
        with open(path, 'r') as f:  # Ensure it's 'r' for reading text files
            squad_dict = json.load(f)
        contexts, questions, answers = [], [], []
        for group in squad_dict['data']:
            for paragraph in group['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    question = qa['question']
                    for answer in qa['answers']:
                        contexts.append(context)
                        questions.append(question)
                        answers.append(answer)
        return contexts, questions, answers

    def add_end_idx(self, contexts, answers):
        for answer, context in zip(answers, contexts):
            gold_text = answer['text']
            start_idx = answer['answer_start']
            end_idx = start_idx + len(gold_text)
            answer['answer_end'] = end_idx if context[start_idx:end_idx] == gold_text else start_idx + context[start_idx:].find(gold_text) + len(gold_text)
        return answers

    def update_start_end_positions(self, encodings, answers):
        start_positions, end_positions = [], []
        for i, answer in enumerate(answers):
            start_positions.append(encodings.char_to_token(i, answer['answer_start']))
            end_positions.append(encodings.char_to_token(i, answer['answer_end'] - 1))
            if start_positions[-1] is None:
                start_positions[-1] = self.tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = self.tokenizer.model_max_length
        encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
        return encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])


class MetricsCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.losses = []
        self.throughputs = []
        self.stats_efficiency = []
        self.gpu_utils = []
        self.last_log_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if not state.log_history:
            return  # If log_history is empty, just return and wait for next call

        current_time = time.time()
        batch_duration = current_time - self.last_log_time
        self.last_log_time = current_time

        if 'loss' in state.log_history[-1]:
            self.losses.append(state.log_history[-1]['loss'])

        num_samples = args.per_device_train_batch_size * torch.distributed.get_world_size()
        if batch_duration > 0:
            throughput = num_samples / batch_duration  # samples per second
            self.throughputs.append(throughput)

        if len(self.losses) > 1:
            loss_improvement = self.losses[-2] - self.losses[-1]
            stat_efficiency = abs(loss_improvement / num_samples)  # absolute efficiency per sample
            self.stats_efficiency.append(stat_efficiency)

        gpu_util = self.get_average_gpu_utilization()
        self.gpu_utils.append(gpu_util)

    def on_train_end(self, args, state, control, **kwargs):
        # Ensure calculations for final metrics do not rely on uninitialized values
        total_runtime = time.time() - self.last_log_time
        avg_loss = np.mean(self.losses) if self.losses else 0
        avg_throughput = np.mean(self.throughputs) if self.throughputs else 0
        avg_gpu_utilization = np.mean(self.gpu_utils) if self.gpu_utils else 0
        avg_stat_efficiency = np.mean(self.stats_efficiency) if self.stats_efficiency else 0

        metrics = {
            'total_runtime': total_runtime,
            'average_loss': avg_loss,
            'average_throughput': avg_throughput,
            'average_gpu_utilization': avg_gpu_utilization,
            'average_statistical_efficiency': avg_stat_efficiency,
            'losses': self.losses,
            'throughputs': self.throughputs,
            'statistical_efficiency': self.stats_efficiency,
            'gpu_utilization': self.gpu_utils
        }
        with open('training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        print("Training metrics saved.")

    def get_average_gpu_utilization(self):
        try:
            # Call nvidia-smi to get GPU utilization
            result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True)
            output = result.stdout.strip()

            # Extract GPU utilization percentages
            gpu_utils = [int(x) for x in output.split('\n')]

            # Calculate the average GPU utilization
            avg_gpu_util = np.mean(gpu_utils) if gpu_utils else 0
    #         avg_gpu_util = np.max(gpu_utils) if gpu_utils else 0
            return avg_gpu_util
        except Exception as e:
            print(f"Error fetching GPU utilization: {e}. Defaulting to 100%.")
            return 100.0  # Default to 100% if there's an error

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
    model.to(device)  # Ensure model is on the correct device
    
    dataset = SquadDataset('./dataset/train-v2.0.json', tokenizer)

    training_args = TrainingArguments(
            output_dir='./results',                # Directory for saving output files.
            num_train_epochs=5,                    # Total number of training epochs.
            per_device_train_batch_size=16,        # This will be dynamically handled by DeepSpeed; ensure it matches expected GPU capacities.
            gradient_accumulation_steps=1,         # Matches the DeepSpeed setting.
            learning_rate=3e-5,                    # Learning rate settings should be consistent with optimizer settings in DeepSpeed config.
            warmup_steps=500,                      # Warm-up steps used in learning rate scheduler.
            weight_decay=0.01,                     # Weight decay to match the optimizer settings in DeepSpeed.
            adam_epsilon=1e-6,                     # Epsilon for the Adam optimizer to prevent any division by zero in the optimizer.
            fp16=True,                             # Enable mixed precision training, matching DeepSpeed's FP16 setting.
            deepspeed='ds_config.json',            # Path to the DeepSpeed configuration file.
            logging_dir='./logs',                  # Directory for storing logs (optional but useful).
            logging_steps=50,                      # Log every 50 steps.
            save_steps=1000,                       # Save the model every 1000 steps.
            evaluation_strategy="steps",           # Evaluation strategy to use.
            eval_steps=500                         # Evaluate the model every 500 steps.
        )




    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[MetricsCallback()]
    )

    trainer.train()

if __name__ == "__main__":
    main()