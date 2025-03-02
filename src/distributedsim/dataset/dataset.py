import torch
import numpy as np
import boto3
import io
import os
from tqdm import tqdm

from .build_dataset import build_dataset
from .gpt_dataset import GPTTrainDataset

def count_files_in_s3_folder(bucket_name, folder_prefix, s3_client):
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=folder_prefix)

    file_count = sum(1 for page in pages for _ in page.get('Contents', []))
    
    return file_count

def load_chunk(chunk_id, s3_client):
    cache_location = f'cache/s3/owt/'
    if not os.path.exists(cache_location):
        os.makedirs(cache_location, exist_ok=True)

    cache_file = f'{cache_location}/chunk_{chunk_id}.npy'
    if os.path.exists(cache_file):
        return np.load(cache_file)
    else:
        s3_client.download_file(Bucket='exo-datasets', Key=f'owt/chunk_{chunk_id}.npy', Filename=cache_file)
        return np.load(cache_file)

def load_data(start_pc, end_pc):
    s3_client = boto3.client('s3')

    chunk_count = count_files_in_s3_folder('exo-datasets', 'owt/', s3_client)

    chunk_ids = np.arange(chunk_count)
    chunk_ids = chunk_ids[int(start_pc * chunk_count):int(end_pc * chunk_count)]
    print(f' importing {len(chunk_ids)} chunks')
    data = []
    for chunk_id in tqdm(chunk_ids):
        data.append(load_chunk(chunk_id, s3_client))
    return np.concatenate(data)


def get_dataset(dataset, start_pc, end_pc, block_size=1024, char=False):
    if dataset != 'owt':
        data, vocab_size = build_dataset(dataset, block_size, char, start_pc, end_pc)
    else:
        # For OWT, pull from S3
        data = load_data(start_pc, end_pc)
        vocab_size = 50257

    return data, vocab_size
