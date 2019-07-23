#!/usr/bin/env python3

#
ARGS = None
DEVICES = {False: 'cpu', True: 'cuda'}
# DEVICE_ID = 'cpu' # set to 'cpu' for cpu, 'cuda' / 'cuda:0' or similar for gpu.
LOG_DIR = 'checkpoints'
VISUALIZE = False # set True to visualize input, prediction and the output from the last batch
DATA_ROOT = ''
FOLDERS = {'train': 'train', 'validation': 'val', 'test': 'test'}

# Standard libraries
from copy import deepcopy
import time

import arg_helper
from srcnn_utils import *

# Function to read dataset
def get_loaders(batch_size, device, **kwargs):
    load_train = kwargs.get('load_train', False) 
    load_test = kwargs.get('load_test', False)
    loaders = {}
    if load_train:
        # indices = range(100) # TODO: For sanity check
        train_set = ImageFolder(root=os.path.join(DATA_ROOT, FOLDERS['train']), device=device)
        # train_set = torch.utils.data.Subset(train_set, indices) # TODO: For sanity check
        loaders['train'] = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        val_set = ImageFolder(root=os.path.join(DATA_ROOT, FOLDERS['validation']), device=device)
        # val_set = torch.utils.data.Subset(val_set, indices) # TODO: For sanity check
        loaders['val'] = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    if load_test:
        test_set = ImageFolder(root=os.path.join(DATA_ROOT, FOLDERS['test']), device=device)
        loaders['test'] = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return loaders

def get_pad_count(n, s, f):
    padding = ((n*s)-s-n+f) / 2
    assert padding.is_integer()
    return int(padding)

# SRCNN
class SRNET(nn.Module):
    def __init__(self):
        super(SRNET, self).__init__() #TODO: Search super(with parameters)

        self.layers = []

        # Adding layers
        self.layers.append(nn.Conv2d(in_channels=1, out_channels=ARGS.kernelcounts[0], kernel_size=ARGS.kernelsizes[0], bias=True))
        for i in range(1, ARGS.convlayers):
            self.layers.append(nn.Conv2d(in_channels=ARGS.kernelcounts[i-1], out_channels=ARGS.kernelcounts[i], kernel_size=ARGS.kernelsizes[i], bias=True))
              
        # Adding ReLUs
        for i in ARGS.relupositions:
            self.layers.insert(i, nn.ReLU())

        # Place layers in a sequence
        self.layers = nn.Sequential(*self.layers)
        print(self.layers)

    def forward(self, image):   
        pred = self.layers(image)
        return pred

class AverageMeter(object):
  '''A handy class from the PyTorch ImageNet tutorial''' 
  def __init__(self):
    self.reset()

  def reset(self):
    self.loss, self.avg, self.loss_sum, self.loss_count = 0, 0, 0, 0

  def update(self, loss, n=1):
    self.loss = loss
    self.loss_sum += loss * n
    self.loss_count += n
    self.avg = self.loss_sum / self.loss_count

# Training for one epoch
def train(train_loader, net, device, get_mse_loss, optimizer, epoch):
    net.train()
    
    train_loss = AverageMeter() # Keep training loss for the entire epoch
    psnr_val = AverageMeter()
    # iter_loss = AverageMeter() # Keep training loss for each n iteration

    print_freq = 100 # Printing frequency in terms of iterations for net loss
    inputs = None
    preds = None
    targets = None

    # For each batch (iteration)
    for iteri, (inputs, targets) in enumerate(train_loader, 1):        
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad() # Clear gradients
        preds = net(inputs) # Forward propagation
        mse_loss = get_mse_loss(preds, targets) # get_mse_loss(y_hat, ground_truth) #TODO: loss calculation only for central pixel (check the paper)
        psnr_val.update(compute_psnr(mse_loss.item())) #TODO per image PSNR calculation
        mse_loss.backward() # Backpropagation
        optimizer.step()
        # iter_loss.update(mse_loss.item()) # TODO: check
        train_loss.update(mse_loss.item()) # TODO: check

        # Print every print_freq mini-batches
        # if (not iteri % print_freq):
        #     print('- Training: [E: %d, I: %3d] Loss: %.4f' % (epoch, iteri, iter_loss.avg))
        #     iter_loss.reset()

        # if (iteri==0) and VISUALIZE: 
        #     visualize_batch(inputs, preds, targets)

    # Visualize results periodically
    draw_freq = 10 # TODO: argparse
    if (draw_freq == 1) or (epoch % draw_freq == 0):
        visualize_batch(inputs, preds, targets, os.path.join(LOG_DIR, 'train_e{}.png'.format(epoch)))

    # Return the average loss of all batches in this epoch
    return train_loss.avg, psnr_val.avg

# Testing
def test(test_loader, net, device, get_mse_loss):
    net.eval()

    with torch.no_grad():
        test_loss = AverageMeter()
        psnr_val = AverageMeter()
        inputs = None
        preds = None
        targets = None

        for iteri, (inputs, targets) in enumerate(test_loader, 1):
            inputs, targets = inputs.to(device), targets.to(device)

            preds = net(inputs)
            mse_loss = get_mse_loss(preds, targets)
            psnr_val.update(compute_psnr(mse_loss.item())) #TODO per image PSNR calculation
            test_loss.update(mse_loss.item())

    return test_loss.avg, psnr_val.avg

def get_current_config():
    global ARGS
    """Return a string indicating current parameter configuration"""
    config = vars(ARGS)
    message = "\nRunning with the following parameter settings:\n"
    separator = "-" * (len(message)-2) + "\n"
    lines = ""
    for item, key in config.items():
        lines += "- {}: {}\n".format(item, key)
    return (message + separator + lines + separator)

def show_current_config():
    """Print current parameter configuration"""
    print(get_current_config())

def main():
    global ARGS
    torch.multiprocessing.set_start_method('spawn', force=True)
    ARGS = arg_helper.arg_handler()
    # If required args are parsed properly
    if ARGS:
        show_current_config()
        # Construct network
        torch.manual_seed(5)
        device = torch.device(DEVICES[ARGS.gpu])
        print('Device: ' + str(device))
        net = SRNET().to(device=device)

        # Mean Squared Error
        get_mse_loss = nn.MSELoss()
        # Optimizer: Stochastic Gradient Descend with initial learning rate
        optimizer = optim.SGD(net.parameters(), lr=0.001) #TODO: change lr per layer (check the paper)
        # Learning rate scheduler
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, 
                                                        #  patience=lr_patience, min_lr=min_lr, verbose=True)
        # Determine pipeline execution
        run_full = (ARGS.pipe == "full")
        run_train = run_full or (ARGS.pipe == "train")
        run_test = run_full or (ARGS.pipe == "test")

        # Get loaders as dict
        # loaders = get_loaders(batch_size, device, load_train=run_train, load_test=run_test)

        # Training mode
        if (run_train):
            print("Training started.")
            # Initialization
            # train_loader = loaders['train']
            # val_loader = loaders['val']
            try:
                MAX_EPOCH = 101
                for epoch in range(1, MAX_EPOCH):
                    # Train over full dataset (1 epoch)
                    train_loss = train(train_loader, net, device, get_mse_loss, optimizer, epoch)
                         
            except KeyboardInterrupt:
                print("\nKeyboard interrupt, stoping execution...\n")
                
            finally:
                pass
                # print('Training finished!')
                # print('Saving training data...')
                # draw_train_val_plots(train_losses, val_losses, path=LOG_DIR, show=False)
                # draw_accuracy_plot(accuracies, len(train_losses), path=LOG_DIR, show=False)
                # stats = {
                #     'train_losses': train_losses,
                #     'val_losses': val_losses,
                #     'accuracies': accuracies,
                #     'total_epoch': epoch,
                #     'max_accuracy': np.max(accuracies),
                #     'min_accuracy': np.min(accuracies),
                #     'max_loss': np.max(val_losses),
                #     'min_loss': np.min(val_losses),
                #     'best_accuracy_epoch': np.argmax(accuracies) + 1,
                #     'best_loss_epoch': (np.argmin(val_losses) + 1) * val_freq,
                # }
                # save_stats("stats.txt", stats, path=LOG_DIR)
                # print('Saved.')
        # Testing mode
        elif (run_test):
            print("Testing started.")
            # test_loader = loaders['test']
            test()

if __name__ == "__main__":
    main()