from .planner import Planner, save_model 
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms

def train(args):
    from os import path

    # Initialize model, optimizer, and tensorboard logger
    model = Planner()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train_logger, valid_logger = None, None

    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    # Load your dataset
    train_data = load_data()

    for epoch in range(args.num_epochs):
        model.train()  # Set the model to training mode

        for batch_idx, (image, target) in enumerate(train_data):
            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            output = model(image)

            # Compute the loss (you might need to define your own loss function)
            loss = F.mse_loss(output, target)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            # Log information and visualize
            global_step = epoch * len(train_data) + batch_idx
            if train_logger is not None and global_step % args.log_interval == 0:
                log(train_logger, image, target, output, global_step)

        # Save the model after each epoch
        save_model(model)


def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments 
    parser.add_argument('-n', '--num_epochs', type=int, default=20)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor(), ToHeatmap(2)])')
    parser.add_argument('-w', '--size-weight', type=float, default=0.01)

    args = parser.parse_args()
    train(args)
