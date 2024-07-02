import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
import os
from unet_main import UNet
from unet_utils import AttU_Net
from data import SpectrogramDataset
from torch.utils.tensorboard import SummaryWriter
import math
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset

def random_split(dataset, lengths,
                 generator=default_generator):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
    ...   ).manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]

# def early_stopping(train_loss, validation_loss, min_delta, tolerance):
#     counter = 0
#     if (validation_loss - train_loss) > min_delta:
#         counter +=1
#         if counter >= tolerance:
#           return True

if __name__ == "__main__":
    learning_rate = 1e-4
    batch_size = 25 #30
    epochs = 400 #10
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "ecg")

    model_save_path = os.path.join(current_dir, "models", "unet_8.pth")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = SpectrogramDataset(data_path)

    random_gen = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [0.9, 0.1], generator = random_gen) #0.9 0.1

    train_dataloader = DataLoader(dataset = train_dataset,
                                  batch_size = batch_size,
                                  shuffle = True)
    val_dataloader = DataLoader(dataset = val_dataset,
                                  batch_size = batch_size,
                                  shuffle = True)

    model = UNet(in_channels = 4).to(device)
    # model = AttU_Net().to(device)
    pretrained_path = False
    # pretrained_path = "./models/unet_5.pth" #path model
    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path))
    optimizer = optim.AdamW(model.parameters(), lr = learning_rate)
    criterion = nn.MSELoss()
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()

    show_epoch = 1
    history = {"train_loss": [], "val_loss":[]}
    best_loss = float('inf')
    early_stopping_counter = 0
    patience = 5
    for epoch in tqdm(range(epochs), desc = f"Total Epochs: {epochs}"):
        model.train()
        train_running_loss = 0
        for idx, img_and_target in enumerate(tqdm(train_dataloader, desc = f"Epoch {show_epoch} of {epochs}")):
            img = img_and_target[0].float().to(device)
            target = img_and_target[1].float().to(device)

            pred = model(img)
            # print(pred.shape)

            loss = criterion(pred, target)
            train_running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = train_running_loss / (idx + 1)
        history["train_loss"].append(train_loss)
        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for idx, img_and_target in enumerate(tqdm(val_dataloader)):
                img = img_and_target[0].float().to(device)
                target = img_and_target[1].float().to(device)

                pred = model(img)

                loss = criterion(pred, target)
                val_running_loss += loss.item()

            val_loss = train_running_loss / (idx + 1)
            if val_loss < best_loss:
                best_loss = val_loss
            else:
                early_stopping_counter += 1
            
            if early_stopping_counter > patience:
                break
            history["val_loss"].append(val_loss)


        print()
        print(f"\nEpoch {show_epoch} Summary:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print()
        show_epoch += 1
        # if early_stopping(train_loss, val_loss, min_delta=10, tolerance = 20):
        #     print("We are at epoch:", epoch)
        #     break
        writer.flush()
    torch.save(model.state_dict(), model_save_path)
    print(history)