# coding=utf-8
import argparse
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import pandas as pd

logger = logging.getLogger(__name__)

class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, idx, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.idx = str(idx)
        self.label = label

def convert_examples_to_features(js, tokenizer, args):
    code = ' '.join(js['func'].split())
    encoding = tokenizer(
        code, truncation=True, padding='max_length', max_length=args.block_size, return_tensors='pt'
    )
    return InputFeatures(
        input_ids=encoding['input_ids'].squeeze(),
        attention_mask=encoding['attention_mask'].squeeze(),
        idx=js['idx'],
        label=js['target']
    )

class TextDataset(Dataset):
    def __init__(self, examples, tokenizer, args):
        self.examples = []
        for js in examples:
            self.examples.append(convert_examples_to_features(js, tokenizer, args))
        for idx, example in enumerate(self.examples[:3]):
            logger.info("*** Example ***")
            logger.info(f"idx: {idx}")
            logger.info(f"label: {example.label}")
            logger.info(f"input_ids: {' '.join(map(str, example.input_ids))}")
            logger.info(f"attention_mask: {' '.join(map(str, example.attention_mask))}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (
            torch.tensor(self.examples[i].input_ids, dtype=torch.long),
            torch.tensor(self.examples[i].attention_mask, dtype=torch.long),
            torch.tensor(self.examples[i].label, dtype=torch.long)
        )

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train(args, train_dataset, model, tokenizer):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4, pin_memory=True
    )
    args.max_steps = args.epoch * len(train_dataloader)
    args.save_steps = len(train_dataloader) // 2  # Save twice per epoch
    args.warmup_steps = len(train_dataloader) // 10  # 10% warmup
    args.logging_steps = len(train_dataloader) // 4  # Log 4 times per epoch
    args.num_train_epochs = args.epoch
    model.to(args.device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, total_steps=args.max_steps)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per GPU = {args.per_gpu_train_batch_size}")
    logger.info(f"  Total train batch size = {args.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_steps}")

    global_step = args.start_step
    tr_loss, logging_loss, train_loss = 0.0, 0.0, 0.0
    best_f1 = 0.0
    early_stopping_counter = 0
    best_loss = None

    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            inputs, attention_mask, labels = batch
            inputs, attention_mask, labels = inputs.to(args.device), attention_mask.to(args.device), labels.to(args.device)
            model.train()
            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description(f"epoch {idx} loss {avg_loss}")

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    if args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer, eval_when_training=True)
                        for key, value in results.items():
                            logger.info(f"  {key} = {value:.4f}")
                        if results['eval_f1'] > best_f1:
                            best_f1 = results['eval_f1']
                            logger.info("  " + "*" * 20)
                            logger.info(f"  Best F1: {round(best_f1, 4)}")
                            logger.info("  " + "*" * 20)
                            output_dir = os.path.join(args.output_dir, 'checkpoint-best-f1')
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = model.module if hasattr(model, 'module') else model
                            torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.bin'))
                            logger.info(f"Saving model checkpoint to {output_dir}")

        avg_loss = train_loss / tr_num
        if args.early_stopping_patience is not None:
            if best_loss is None or avg_loss < best_loss - args.min_loss_delta:
                best_loss = avg_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= args.early_stopping_patience:
                    logger.info("Early stopping")
                    break

def evaluate(args, model, tokenizer, eval_when_training=False):
    eval_dataset = TextDataset(args.eval_data, tokenizer, args)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)

    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {args.eval_batch_size}")
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        inputs, attention_mask, label = batch
        inputs, attention_mask, label = inputs.to(args.device), attention_mask.to(args.device), label.to(args.device)
        with torch.no_grad():
            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=label)
            lm_loss = outputs.loss
            logit = outputs.logits
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = np.argmax(logits, axis=1)
    
    # Calculate metrics
    eval_acc = np.mean(labels == preds)
    eval_precision = precision_score(labels, preds, average='macro', zero_division=0)
    eval_recall = recall_score(labels, preds, average='macro', zero_division=0)
    eval_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    eval_loss = eval_loss / nb_eval_steps

    # Print confusion matrix components for debugging
    tp = np.sum((labels == 1) & (preds == 1))
    fp = np.sum((labels == 0) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))
    tn = np.sum((labels == 0) & (preds == 0))
    logger.info(f"True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}, True Negatives: {tn}")

    result = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "eval_precision": eval_precision,
        "eval_recall": eval_recall,
        "eval_f1": eval_f1
    }
    return result

def test(args, model, tokenizer):
    eval_dataset = TextDataset(args.test_data, tokenizer, args)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    logger.info("***** Running Test *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {args.eval_batch_size}")
    model.eval()
    logits = []
    labels = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        inputs, attention_mask, label = batch
        inputs, attention_mask = inputs.to(args.device), attention_mask.to(args.device)
        with torch.no_grad():
            outputs = model(input_ids=inputs, attention_mask=attention_mask)
            logit = outputs.logits
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())

    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = np.argmax(logits, axis=1)
    with open(os.path.join(args.output_dir, "predictions.txt"), 'w') as f:
        for example, pred in zip(eval_dataset.examples, preds):
            f.write(f"{example.idx}\t{pred}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default="../datasets/Devign/devign.json", type=str, required=True)
    parser.add_argument("--output_dir", default="./output", type=str, required=True)
    parser.add_argument("--model_type", default="roberta", type=str)
    parser.add_argument("--model_name_or_path", default="microsoft/codebert-base", type=str)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--evaluate_during_training", action='store_true')
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=16, type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--block_size", default=512, type=int)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--min_loss_delta", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.per_gpu_train_batch_size = args.train_batch_size // max(1, args.n_gpu)
    args.per_gpu_eval_batch_size = args.eval_batch_size // max(1, args.n_gpu)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    set_seed(args.seed)

    # Load and split dataset
    with open(args.data_file, 'r') as f:
        data = json.load(f)
        
    random.shuffle(data)
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=args.seed)
    dev_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=args.seed)
    args.train_data = train_data
    args.eval_data = dev_data
    args.test_data = test_data

    # Log class distribution
    train_labels = [item['target'] for item in train_data]
    dev_labels = [item['target'] for item in dev_data]
    test_labels = [item['target'] for item in test_data]
    logger.info(f"Train class distribution: {np.bincount(train_labels)}")
    logger.info(f"Dev class distribution: {np.bincount(dev_labels)}")
    logger.info(f"Test class distribution: {np.bincount(test_labels)}")

    # Load model and tokenizer
    config = RobertaConfig.from_pretrained(args.model_name_or_path, num_labels=2)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    model.to(args.device)

    if args.do_train:
        train_dataset = TextDataset(args.train_data, tokenizer, args)
        train(args, train_dataset, model, tokenizer)

    if args.do_eval:
        output_dir = os.path.join(args.output_dir, 'checkpoint-best-f1/model.bin')
        if os.path.exists(output_dir):
            model.load_state_dict(torch.load(output_dir))
            model.to(args.device)
            result = evaluate(args, model, tokenizer)
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info(f"  {key} = {result[key]:.4f}")
        else:
            logger.warning("No checkpoint found for evaluation")

    if args.do_test:
        output_dir = os.path.join(args.output_dir, 'checkpoint-best-f1/model.bin')
        if os.path.exists(output_dir):
            model.load_state_dict(torch.load(output_dir))
            model.to(args.device)
            test(args, model, tokenizer)
        else:
            logger.warning("No checkpoint found for testing")

if __name__ == "__main__":
    main()