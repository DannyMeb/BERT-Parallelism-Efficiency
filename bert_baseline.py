import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# subprocess.run(["pip", "install", "transformers"])
import transformers
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizerFast, BertForQuestionAnswering
from torch.optim import AdamW 
import json
import time
import numpy as np
import subprocess

transformers.logging.set_verbosity_error()  # Only show errors, not warnings


def get_average_gpu_utilization():
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

# Define the SquadDataset class


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer):
        self.tokenizer = tokenizer
        contexts, questions, answers = self.read_squad(path)
        answers = self.add_end_idx(contexts, answers)
        self.encodings = tokenizer(contexts, questions, truncation=True, padding=True)
        self.encodings = self.update_start_end_positions(self.encodings, answers, tokenizer)

    def read_squad(self, path):
        with open(path, 'rb') as f:
            squad = json.load(f)
        contexts = []
        questions = []
        answers = []
        for group in squad['data']:
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
            gold_text = answer["text"]
            start_idx = answer["answer_start"]
            end_idx = start_idx + len(gold_text)

            if context[start_idx:end_idx] == gold_text:
                answer["answer_end"] = end_idx
            else:
                answer["answer_end"] = start_idx + context[start_idx:].find(gold_text) + len(gold_text)
        return answers

    def update_start_end_positions(self, encodings, answers, tokenizer):
        start_positions = []
        end_positions = []
        for i in range(len(answers)):
            start_positions.append(encodings.char_to_token(i, answers[i]["answer_start"]))
            end_positions.append(encodings.char_to_token(i, answers[i]["answer_end"] - 1))
            if start_positions[-1] is None or end_positions[-1] is None:
                start_positions[-1] = tokenizer.model_max_length
                end_positions[-1] = tokenizer.model_max_length
        encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
        return encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

class BertFinetuner:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.setup()

        if rank == 0 and not os.path.exists("metrics"):
            os.makedirs("metrics")

        self.device = torch.device(f'cuda:{rank}')
        self.model_name = 'bert-base-uncased'
        self.squad_path = './dataset/train-v2.0.json'
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name)
        dataset = SquadDataset(self.squad_path, self.tokenizer)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, _ = random_split(dataset, [train_size, val_size])

        self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        self.train_loader = DataLoader(self.train_dataset, batch_size=16, sampler=self.train_sampler)

        self.model = BertForQuestionAnswering.from_pretrained(self.model_name).to(self.device)
        self.model = DDP(self.model, device_ids=[rank])

    def setup(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size)

    @staticmethod
    def cleanup():
        dist.destroy_process_group()

    
    @staticmethod
    def get_gpu_utilization():
        try:
            # Call nvidia-smi to get GPU utilization
            result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True)
            output = result.stdout.strip()

            # Extract GPU utilization percentages
            gpu_utils = [int(x) for x in output.split('\n')]

            # Calculate the average GPU utilization
            avg_gpu_util = np.mean(gpu_utils) if gpu_utils else 0
            return avg_gpu_util
        except Exception as e:
            print(f"Error fetching GPU utilization: {e}. Defaulting to 100%.")
            return 100.0  # Default to 100% if there's an error

    def train(self):
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        total_steps = len(self.train_loader) * 5  # Total steps for 5 epochs
        current_step = 0
        
        throughputs, gpu_utils, losses = [], [], []
        training_start_time = time.time()
        
        self.model.train()
        for epoch in range(5):
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(self.train_loader):
                batch_start_time = time.time()
                
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                start_positions = batch['start_positions'].to(self.device)
                end_positions = batch['end_positions'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask,
                                     start_positions=start_positions, end_positions=end_positions)
                loss = outputs.loss
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                
                current_step += 1
                batch_end_time = time.time()
                batch_duration = batch_end_time - batch_start_time
                batch_throughput = len(input_ids) / batch_duration
                
                
                gpu_utilization = self.get_gpu_utilization()
                gpu_utils.append(gpu_utilization)
                losses.append(loss.item())
                throughputs.append(batch_throughput)
                # Update progress every N steps
                if self.rank == 0 and batch_idx % 50 == 0:  # Update every 50 batches
                    elapsed_time = time.time() - training_start_time
                    progress = (current_step / total_steps) * 100
                    print(f"Epoch {epoch+1}: {progress:.2f}% | {current_step}/{total_steps} [Elapsed: {elapsed_time:.2f} sec, Loss: {loss.item():.2f}, Throughput: {batch_throughput:.2f} items/sec, GPU Utlization: {gpu_utilization}]")

            if self.rank == 0:
                avg_epoch_loss = epoch_loss / len(self.train_loader)
                print(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.2f}")

        training_end_time = time.time()
        runtime = training_end_time - training_start_time
        avg_throughput = np.mean(throughputs)
        avg_loss=np.mean(losses)
        final_loss = losses[-1]
        avg_gpu_utilization = np.mean(gpu_utils) if gpu_utils else 0
        print("✨✨✨ Training Completed Successfully! ✨✨✨")
        print(f'Training time: {runtime}')
        print(f'Final loss: {final_loss}')
        print(f'Average Loss: {avg_loss}')
        print(f'Average Throughput : {avg_throughput}')
        print(f'Average GPU Utilization: {avg_gpu_utilization}')
        
        if self.rank == 0:
            metrics_filename = f"metrics/detailed_baseline_{self.world_size}GPU_metricsFinal5EpochsAverage.json"
            metrics = {
                "runtime": runtime,
                "average loss": avg_loss,
                "average throughput": avg_throughput,
                "average gpu utilization": avg_gpu_utilization,
                "throughputs": throughputs,
                "gpu_utilizations": gpu_utils,
                "loss" : losses
            }
            with open(metrics_filename, "w") as f:
                json.dump(metrics, f, indent=4)
        self.cleanup()


def main(rank, world_size):
    finetuner = BertFinetuner(rank, world_size)
    finetuner.train()

if __name__ == "__main__":
    world_size = 4
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)