import torch
from transformers import AutoTokenizer, AutoModel
import pickle
from tqdm import tqdm


def setup_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    else:
        print("Using single GPU or CPU")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return tokenizer, model, device


def preprocess_code(code_snippet, tokenizer):
    tokens = tokenizer(code_snippet, padding=True, truncation=True, max_length=512, return_tensors="pt")
    return tokens


def get_code_embedding(inputs, model, device):
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    embedding = torch.mean(last_hidden_state, dim=1)
    return embedding.cpu().numpy()


# def process_chunk(chunk, tokenizer, model, device):
#     key_embeddings = []
#     for key, snippet in chunk:
#         inputs = preprocess_code(snippet, tokenizer)
#         embedding = get_code_embedding(inputs, model, device)
#         key_embeddings.append((key, embedding))
#     return key_embeddings

def process_code_snippets(code_snippets, path2idx, tokenizer, model, device, batch_size=32):
    func_id2dcmp_embeds = {}
    model.eval()

    # Create a list of (key, value) pairs
    items = list(code_snippets.items())

    for i in tqdm(range(0, len(items), batch_size), desc="Processing batches"):
        batch_items = items[i:i + batch_size]
        keys, snippets = zip(*batch_items)

        inputs = tokenizer(list(snippets), padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden_state = outputs.last_hidden_state
        embeddings = torch.mean(last_hidden_state, dim=1)

        for key, embed in zip(keys, embeddings.cpu().numpy()):
            func_id2dcmp_embeds[path2idx[key]] = embed

    return func_id2dcmp_embeds


# def process_code_snippets_parallel(code_snippets, num_workers, path2idx):
#     chunk_size = len(code_snippets) // num_workers
#
#     chunks = [(key, value) for key, value in code_snippets.items()]
#     chunks = [chunks[i:i + chunk_size] for i in range(0, len(chunks), chunk_size)]
#
#     func_id2dcmp_embeds = {}
#     with ProcessPoolExecutor(max_workers=num_workers) as executor:
#         futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
#         for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
#             results = future.result()
#             for key, embed in results:
#                 func_id2dcmp_embeds[path2idx[key]] = embed
#
#     return func_id2dcmp_embeds


def save_embeddings(embeddings, filename):
    with open(filename, 'wb') as f:
        pickle.dump(embeddings, f)


def load_embeddings(filename):
    with open(filename, 'rb') as f:
        id2vec = pickle.load(f)
    return id2vec

def read_code_snippets(directory):
    snippets = {}
    with open(directory, 'rb') as fr:
        raw_data = pickle.load(fr)
    for data in raw_data:
        snippets[data['func_ident']] = data['func_dcmp']
    return snippets


def run_dcmp_vec(cached_file_path, path2idx, saved_path, batch_size=256):
    tokenizer, model, device = setup_model_and_tokenizer()

    # Read code snippets from a directory
    code_snippets = read_code_snippets(cached_file_path)
    func_id2dcmp_embeds = process_code_snippets(code_snippets, path2idx, tokenizer, model, device, batch_size)

    # Save embeddings
    save_embeddings(func_id2dcmp_embeds, saved_path)
    # print(embeddings)
    # print(len(embeddings.keys()))
    # print(f"Processed {len(embeddings)} code snippets")
    # print(f"Each embedding has shape: {embeddings[list(embeddings.keys())[0]].shape}")
    print('Building dcmp vectors done!')


# if __name__ == '__main__':
#     # # Later, you can load the embeddings like this:
#     # loaded_embeddings = load_embeddings('./embeds/glove.6B.100d.txt')
#     # print(loaded_embeddings)
#     run_dcmp_vec('')