import torch

def get_batch(data: list[str], block_size: int, batch_size: int, device):
    """
    This is a simple function to create batches of data.
    GPUs allow for parallel processing we can feed multiple chunks at once
    so that's why we would need batches - how many independant sequences
    will we process in parallel.
    Arguments:
    - data: list[str]: data to take batch from
    - block_size (int): size of the text that is proccessed at once
    - batch_size (int): number of sequences to process in parallel
    Returns:
    - x, y: a tuple with token sequence and token target
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # we stack batch_size rows of sentences
    # so x and y are the matrices with rows_num=batch_size
    # and col_num=block_size
    x = torch.stack([data[i : i + block_size] for i in ix])
    # y is x shifted one position right - because we predict
    # word in y having all the previous words as context
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(
    data: list[str],
    model: torch.nn.Module,
    device,
    block_size: int,
    batch_size: int,
    eval_iters: int = 10,
):
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data=data, block_size=block_size, batch_size=batch_size, device=device)
        logits, loss = model.forward(X, Y)
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out


def batch_training(
        model: torch.nn.Module,
        max_iteration: int, 
        block_size: int,
        learning_rate: float,
        train_indices: torch.Tensor, 
        test_indices: torch.Tensor,
        device
    ):
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(max_iteration):
        # every EVAL_INTER evaluate the loss on train and val sets
        if step % 200 == 0 or step == max_iteration - 1:
            loss_train = estimate_loss(
                data=train_indices, model=model, block_size=block_size, batch_size=32, device=device
            )
            loss_val = estimate_loss(
                data=test_indices, model=model, block_size=block_size, batch_size=32, device=device
            )
            print("step {:10} | train loss {:6.4f} | val loss {:6.4f}".format(step, loss_train, loss_val))

        # sample a batch of data
        xb, yb = get_batch(data=train_indices, block_size=block_size, batch_size=32, device=device)
        logits, loss = model.forward(xb, yb)

        # zero_grad() method sets the gradients of all parameters in the optimizer to zero
        optimizer.zero_grad(set_to_none=True)

        # backward() method on the loss variable calculates the gradients 
        # of the loss with respect to the model's parameters.
        loss.backward()

        # step() method on the optimizer updates the model's parameters 
        # using the calculated gradients, in order to minimize the loss.
        optimizer.step()