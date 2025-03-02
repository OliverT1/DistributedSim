import torch
import argparse
import numpy as np
import math
import os
from datasets import load_dataset, Dataset, DatasetDict, load_dataset_builder, concatenate_datasets

def generate_char_vocab():
    """
    Generates a fixed character vocabulary and returns two mappings:
    char -> int, int -> char, and also the special end-of-sequence token id.
    """
    vocab = ' !$&\',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\n'
    char_int = {char: i for i, char in enumerate(vocab)}
    int_char = {i: char for i, char in enumerate(vocab)}

    # Define a special end-of-sequence token.
    eos_token = '<EOS>'
    char_int[eos_token] = len(char_int)
    eos_token_id = char_int[eos_token]
    return char_int, eos_token_id

def build_dataset(dataset, block_size=1024, char=False, start_pc=0.0, end_pc=1.0):
    """
    Loads and preprocesses the dataset with caching, using either a custom character-level tokenizer
    or the GPT2 tokenizer. It uses only a fraction of the full dataset, as controlled by dataset_proportion,
    then splits that fraction into training and validation portions (with val_ratio held out as validation).
    When rank and world_size are provided, the training portion is sharded among nodes.

    Args:
        dataset: a string identifier ("shakespeare", "wikitext", or "owt")
        block_size: the sequence block size.
        char (bool): If True, use character-level tokenization; otherwise, use GPT-2 tokenization.
        start_pc (float): The start percentage of the dataset to use.
        end_pc (float): The end percentage of the dataset to use.
    """
    # Decide cache locations based on tokenization mode and rank.
    if char:
        cache_dir = os.path.join("cache", f"{dataset}_char")
    else:
        cache_dir = os.path.join("cache", dataset)
    os.makedirs(cache_dir, exist_ok=True)

    data_cache_file = os.path.join(cache_dir, f"data_block{block_size}_{start_pc}_{end_pc}.pt")

    # if os.path.exists(data_cache_file):
    #     print(f"Loading cached dataset from {data_cache_file}")
    #     cached_data = torch.load(data_cache_file)
    #     return cached_data["train"], cached_data["val"], cached_data["vocab_size"]

    print(f"Loading dataset: {dataset} {'(char-level)' if char else '(GPT2 tokenization)'} start%: {start_pc} end%: {end_pc}")
    
    # Determine the dataset identifier and mapping function.
    if dataset == "shakespeare":
        dataset_id = "Trelis/tiny-shakespeare"
        mapping_fn = lambda x: {'text': x['Text']}
        load_config = {}
    elif dataset == "wikitext":
        dataset_id = "wikitext"
        config = "wikitext-2-raw-v1"
        mapping_fn = lambda x: {'text': x['text']}
        load_config = {"name": config}
    elif dataset == "owt":
        dataset_id = "Skylion007/openwebtext"
        mapping_fn = lambda x: x  # Assume openwebtext already has a 'text' field.
        load_config = {"trust_remote_code": True}
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Use the dataset builder to obtain the total number of records.
    builder = load_dataset_builder(dataset_id, **load_config)
    if dataset == "wikitext" or dataset == "shakespeare":
        total_records = builder.info.splits["train"].num_examples + builder.info.splits["test"].num_examples
    else:
        total_records = builder.info.splits["train"].num_examples

    print(f"Total records to import: {total_records}")
    
    # Calculate the number of records to use and how to split them.
    start_record = int(total_records * start_pc)
    end_record = int(total_records * end_pc)

    used_records = end_record - start_record
    print(f"Using {used_records} records: {start_record} to {end_record}")

    if dataset == 'wikitext' or dataset == 'shakespeare':
        # Small enough dataset that we can load the whole thing in.
        dataset = load_dataset(dataset_id, **load_config)
        dataset = concatenate_datasets([dataset['train'], dataset['test']])

        dataset = dataset.map(mapping_fn, remove_columns=dataset.column_names)

        dataset = dataset.select(range(start_record, end_record))
    else:
        # Large dataset - we need to slice instead of loading the whole thing.
        dataset = load_dataset(dataset_id, split=f"train[{start_record}:{end_record}]", **load_config)

    ## Initialize the tokenizer.
    if char:
        char_int, eos_token_id = generate_char_vocab()
        vocab_size = len(char_int)
        def tokenize(example):
            text = example['text']
            if isinstance(text, str):
                return {'tokenized': [char_int[c] for c in text]}
            elif isinstance(text, list):
                return {'tokenized': [[char_int[c] for c in t] for t in text]}
            else:
                raise Exception("Unknown type")
    else:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        vocab_size = tokenizer.vocab_size
        eos_token_id = tokenizer.eos_token_id
        def tokenize(example):
            return {'tokenized': tokenizer(example['text'], truncation=True, max_length=block_size)['input_ids']}

    ## Tokenize the dataset.
    dataset = dataset.map(
        tokenize,
        num_proc=os.cpu_count(),
        batched=True
    )

    # Convert tokenized lists to blocks with fixed block size
    def convert_to_blocks(examples):
        # Flatten all ids and add EOS tokens
        all_ids = np.concatenate([np.array(ids + [eos_token_id]) for ids in examples["tokenized"] if ids])
        num_blocks = len(all_ids) // block_size
        if num_blocks == 0:
            return {"ids": torch.tensor([])}
        all_ids = all_ids[: num_blocks * block_size]
        data_2d = all_ids.reshape(-1, block_size)
        return {"ids": data_2d}

    dataset_processed = dataset.map(
        convert_to_blocks,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=os.cpu_count()
    )

    dataset_processed.set_format(type='numpy', columns=['ids'])

    data = dataset_processed["ids"]

    print(f"Dataset size: {data.shape}")

    np.save(data_cache_file, data)

    # cache_data = {
    #     "data": data,
    #     "vocab_size": vocab_size,
    # }
    # # Optionally cache the processed data:
    # torch.save(cache_data, data_cache_file)

    return data, vocab_size

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--dataset", type=str, default="shakespeare",
                        help="Dataset: shakespeare, wikitext, code, or owt")
    parser.add_argument("--char", action="store_true",
                        help="Enable character-level tokenization")
    parser.add_argument("--start_pc", type=float, default=0.0,
                        help="Proportion of the dataset to use (0 to 1)")
    parser.add_argument("--end_pc", type=float, default=1.0,
                        help="Fraction of the used dataset to reserve for validation")
    args = parser.parse_args()

    data, vocab_size = build_dataset(args.dataset, 
                                                       args.block_size, 
                                                       char=args.char,
                                                       start_pc=args.start_pc,
                                                       end_pc=args.end_pc)
    
    print(f"Finished importing dataset: {args.dataset} {'(char-level)' if args.char else '(GPT2 tokenization)'} start%: {args.start_pc} end%: {args.end_pc}")

if __name__ == "__main__":
    main()
