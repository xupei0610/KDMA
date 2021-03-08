import os
import time
import numpy
import torch
from rl import utils

import config
import data
from models.networks import ExpertNetwork

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, default=None)
parser.add_argument("--device", type=str, default=None)
settings = parser.parse_args()

train_data_files = [
    "data/students03.csv",
]
val_data_files= train_data_files

# data cleansing
invalid_agent_ids = [
3,   5,  13,  14,  15,  17,  20,  21,  24,  27,  29,  36,  39,  40,  42,  44,  45,  46,  49,  50,
54,  55,  59,  60,  61,  71,  73,  74,  75,  81,  82,  92,  97,  98,
102, 103, 120, 122, 124, 125, 141, 144, 154, 159, 168, 178, 198,
202, 203, 206, 211, 216, 218, 219, 220, 221, 222, 223, 224, 225, 228, 229, 231, 232, 237, 243, 244, 247, 248, 250,
251, 255, 258, 262, 265, 273, 275, 277, 278, 290,
301, 305, 306, 311, 315, 316, 319, 320, 322, 325, 326, 328, 329, 330, 331, 333, 334, 335, 336, 339, 340, 342, 344, 345, 346, 349, 350,
353, 354, 357, 359, 361, 362, 370, 374, 375, 376, 377, 378, 383, 390, 391, 392,
408, 409, 410, 411, 412, 413, 418, 420, 421, 431, 433,
]
valid_agent_ids = [[i for i in range(434) if i not in invalid_agent_ids]]

def main():
    from torch.utils.tensorboard import SummaryWriter
    if settings.log_dir is None:
        writer = None
    else:
        writer = SummaryWriter(log_dir=settings.log_dir)

    if settings.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = settings.device

    utils.seed(0)
    model = ExpertNetwork(agent_dim=4, neighbor_dim=4, out_dim=2)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.EXPERT_LR)

    if settings.log_dir is not None and os.path.exists(os.path.join(settings.log_dir, "ckpt")):
        ckpt = torch.load(os.path.join(settings.log_dir, "ckpt"))
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        epoch = ckpt["epoch"]
    else:
        epoch = 0

    dataset = data.as_pytorch_dataset(
        neighbor_radius=config.NEIGHBORHOOD_RADIUS,
        interpolation=True,
        data_files=train_data_files,
        agent_ids=valid_agent_ids,
        transform=[data.FlipTransform(2), data.RotationTransform(3)],
        device=device,
        seed=1
    )
    train_dataset = dataset
    val_dataset = data.as_pytorch_dataset(
        neighbor_radius=config.NEIGHBORHOOD_RADIUS,
        interpolation=False,
        agent_ids=valid_agent_ids,
        data_files=val_data_files,
        device=device
    )

    train_iterator = torch.utils.data.DataLoader(train_dataset, collate_fn=dataset.batch_collator,
            batch_size=config.EXPERT_BATCH_SIZE)
    val_iterator = torch.utils.data.DataLoader(val_dataset, collate_fn=dataset.batch_collator,
            batch_size=len(val_dataset))

    print(model)

    criterion = lambda y_, y: (y - y_).square().sum(1)
    max_epochs = epoch+config.EXPERT_EPOCHS
    for epoch in range(epoch, max_epochs):
        training_loss, val_loss, val_error = 0., 0., 0.
        model.train()
        start_time = time.time()
        for neighbors, agent, out in train_iterator:
            out_ = model(agent, neighbors)
            loss = criterion(out_, out)
            training_loss += loss.sum().item()
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            for neighbors, agent, out in val_iterator:
                out_ = model(agent, neighbors)
                loss = criterion(out_, out)
                val_loss += loss.sum().item()

        duration = time.time() - start_time
        step = epoch*len(train_dataset)
        training_loss /= len(train_dataset)
        val_loss /= len(val_dataset)
        if writer:
            writer.add_scalar("sl_loss/training_loss", training_loss, step)
            writer.add_scalar("sl_loss/test_loss", val_loss, step)
        print("Epoch: {}/{} ###################################################### - {:.3f}s".format(epoch+1, max_epochs, duration))
        print("Training Loss: {:.6f}".format(training_loss))
        print("Testing Loss: {:.6f}".format(val_loss))

    if settings.log_dir is not None:
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, os.path.join(settings.log_dir, "ckpt"))
    writer.close()

if __name__ == "__main__":
    main()

