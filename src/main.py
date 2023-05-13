import torch
from tokenizer import bert_tokenizer
from model import Transformer
from train import batch_training

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_HEADS = 6
EMBED_DIM = NUM_HEADS * 128
BLOCK_SIZE = 64
NUM_LAYERS = 6

LEARNING_RATE = 3e-4

if __name__ == "__main__":
    input_filepath = "data/wikitext-2.txt"
    data = open(input_filepath, encoding="utf-8").read()

    wrapped_tokenizer = bert_tokenizer(input_filepath)
    vocab_size = wrapped_tokenizer.vocab_size

    tokens = wrapped_tokenizer.tokenize(data)
    token_indices = wrapped_tokenizer.convert_tokens_to_ids(tokens)
    token_indices = torch.tensor(token_indices, dtype=torch.long)

    # train-test-split
    # - Training set: 90% for big text dataset
    # - Testing set: remaining 10%
    n = int(0.9 * len(token_indices))
    train_indices = token_indices[:n]
    test_indices = token_indices[n:]

    # start creating transformer model for GPT
    # device: CUDA if NVIDIA GPU and CUDA driver being installed,
    #         else the computation will take place in CPU
    model = Transformer(
        embed_dim=EMBED_DIM,
        n_layers=NUM_LAYERS,
        n_heads=NUM_HEADS,
        block_size=BLOCK_SIZE,
        vocab_size=vocab_size
    )
    print(model.eval())
    model = model.to(DEVICE)

    batch_training(
        model=model,
        max_iteration=2000,
        block_size=BLOCK_SIZE,
        learning_rate=LEARNING_RATE,
        train_indices=train_indices,
        test_indices=test_indices,
        device=DEVICE
    )