"""
Preprocess the RedPajama sample dataset for training.

Tokenize the text, filter out examples that compress too poorly or are mostly
numbers, and chunk into equal-sized pieces.

Sorts the chunks by the average token probability in the style of "CRAMMING: TRAINING
A LANGUAGE MODEL ON A SINGLE GPU IN ONE DAY" (https://arxiv.org/abs/2212.14034). Also
filters out chunks with fewer than 10 unique tokens.
"""

from collections import Counter

import datasets
import torch
import tqdm
import transformers

NUM_PROC = 64
MAX_COMPRESSION_RATIO = 0.3
MAX_NUMBER_FRAC = 0.1
CHUNK_SIZE = 4096

tokenizer = transformers.AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")


def do_tokenize(examples):
    """Tokenize text and compute the compression ratio."""
    tokenized = tokenizer(examples["text"], return_tensors="pt")
    return {
        **tokenized,
        "compression_ratio": tokenized.input_ids.numel() / len(examples["text"]),
    }


def number_frac(examples):
    """Estimate the fraction of words that are numbers."""
    divisor = 0
    count = 0
    for word in examples["text"].split():
        for chunk in word.split(","):
            if all(c in "+-0123456789e." for c in chunk):
                count += 1
            divisor += 1
    return count / max(1, divisor)


def chunkinate(ds, chunk_size):
    """Chop up examples into equal-sized chunks for optimal mouthfeel."""
    current_chunk = {
        "input_ids": [],
        "attention_mask": [],
    }
    for example in tqdm.tqdm(ds, total=len(ds)):
        ids_in = example["input_ids"][0]
        mask_in = example["attention_mask"][0]
        if ids_in[-1] != tokenizer.eos_token_id:
            ids_in.append(tokenizer.eos_token_id)
            mask_in.append(1)
        while len(ids_in) > 1:
            free_space = chunk_size - len(current_chunk["input_ids"])
            if free_space < 1:
                yield current_chunk
                current_chunk = {
                    "input_ids": [],
                    "attention_mask": [],
                }
                free_space = chunk_size
            current_chunk["input_ids"].extend(ids_in[:free_space])
            current_chunk["attention_mask"].extend(mask_in[:free_space])
            ids_in = [tokenizer.bos_token_id] + ids_in[free_space:]
            mask_in = [1] + mask_in[free_space:]
    if current_chunk["input_ids"]:
        # pad the last chunk
        padding_size = chunk_size - len(current_chunk["input_ids"])
        if padding_size > 0:
            current_chunk["input_ids"].extend([tokenizer.pad_token_id] * padding_size)
            current_chunk["attention_mask"].extend([0] * padding_size)
        yield current_chunk


ds = datasets.load_dataset("togethercomputer/RedPajama-Data-1T-Sample")["train"]
ds_p = ds.map(do_tokenize, num_proc=NUM_PROC).filter(
    lambda ex: ex["compression_ratio"] <= MAX_COMPRESSION_RATIO, num_proc=NUM_PROC
)

token_counts = Counter()
for example in tqdm.tqdm(ds_p, total=len(ds_p)):
    assert len(example["input_ids"]) == 1
    token_counts.update(example["input_ids"][0])

tok_probs = torch.tensor(
    [token_counts.get(idx, 0) for idx in range(tokenizer.vocab_size)]
) / sum(token_counts.values())


def example_avg_token_prob(example):
    return torch.mean(tok_probs[example["input_ids"][0][:2048]])


ds_pp = ds_p.map(
    lambda ex: {"avg_token_prob": example_avg_token_prob(ex)}, num_proc=NUM_PROC
)

ds_ppp = ds_pp.filter(lambda ex: number_frac(ex) < MAX_NUMBER_FRAC, num_proc=NUM_PROC)
ds_pppp = ds_ppp.sort(["avg_token_prob"], reverse=True)


def unique_tokens(example):
    return len(set(example["input_ids"]))


chunked = datasets.Dataset.from_generator(lambda: chunkinate(ds_pppp, CHUNK_SIZE))
chunked = chunked.filter(lambda ex: unique_tokens(ex) > 10, num_proc=NUM_PROC)
chunked.save_to_disk(f"redpajama_1t_{CHUNK_SIZE}")
