#!/usr/bin/env python3
# Standard libraries
from copy import deepcopy
import time

import arg_helper_train
from srcnn import SRNET
from srcnn_utils import *

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
    # PyTorch's own optimization for training
    net.train()
    
    train_loss = AverageMeter() # Keep training loss for the entire epoch
    psnr_val = AverageMeter()

    # For each batch (iteration)
    for iteri, (inputs, targets, paths) in enumerate(train_loader, 1):  
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad() # Clear gradients
        preds = net(inputs) # Forward propagation
        rmse_loss = torch.sqrt(get_mse_loss(preds, targets)) # get_mse_loss(y_hat, ground_truth)
        loss = rmse_loss.item()
        psnr_val.update(compute_psnr(loss))
        rmse_loss.backward() # Backpropagation
        optimizer.step()
        train_loss.update(loss) # TODO: check

        console_log("Train", epoch, iteri, loss, psnr_val.loss)
        for i in range(inputs.size()[0]):
            save_visualized_image_trio("Train", epoch, iteri, loss, psnr_val.loss, inputs[i], preds[i], targets[i], paths[i]) # Assuming SGD with batch size = 1; therefore use inputs[0], preds[0], targets[0], paths[0]

    # Return the average loss of all batches in this epoch
    return train_loss.avg, psnr_val.avg

# Validation
def validation(validation_loader, net, device, get_mse_loss, epoch):
    net.eval()

    with torch.no_grad():
        validation_loss = AverageMeter()
        psnr_val = AverageMeter()

        for iteri, (inputs, targets, paths) in enumerate(validation_loader, 1):
            inputs, targets = inputs.to(device), targets.to(device)
            preds = net(inputs)
            rmse_loss = torch.sqrt(get_mse_loss(preds, targets)) # get_mse_loss(y_hat, ground_truth)
            loss = rmse_loss.item()
            psnr_val.update(compute_psnr(loss))
            validation_loss.update(loss)

            console_log("Validation", epoch, iteri, loss, psnr_val.loss)
            save_visualized_image_trio("Validation", epoch, iteri, loss, psnr_val.loss, inputs[0], preds[0], targets[0], paths[0]) # Assuming SGD with batch size = 1; therefore use inputs[0], preds[0], targets[0], paths[0]

    return validation_loss.avg, psnr_val.avg

def main():
    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.manual_seed(globals.SEED) # seed is used for pseudo random initialization of network parameters (weights and biases)
    globals.ARGS = arg_helper_train.arg_handler()

    # If required args are parsed properly
    if globals.ARGS:
        # Construct network
        net, get_mse_loss, optimizer, last_epoch = None, None, None, 1
        # Checking if any checkpoint provided
        checkpoint_path = globals.ARGS.checkpoint
        if not checkpoint_path:
            net = SRNET()
            # Mean Squared Error
            get_mse_loss = nn.MSELoss()

            # Assume N convolutional layers with bias, params_list: [W0, B0, W1, B1,... WN, BN]
            params_list = list(net.parameters())

            # Before the loop, groups: []
            groups = []
            for i in range(0, len(params_list), 2): # 0, 2... N
                # Convolutional layer-k: (Wk, Bk), params_list[i]: Wk, params_list[i+1]: Bk
                params = {}
                weight = params_list[i]
                bias = params_list[i+1]
                weight_bias_pair = (weight, bias)
                # Get LR for each layer
                lr = globals.ARGS.learnrates[i//2]
                params['params'] = weight_bias_pair
                params["lr"] = lr
                groups.append(params)
            # After the loop, groups: [{'params': (W0, B0), 'lr': lr0},... 
            #                          {'params': (WN, BN), 'lr': lrN}]

            # Optimizer: Stochastic Gradient Descend with initial learning rate
            optimizer = optim.SGD(groups, lr=1e-05) # Default lr (if unspecified) is 1e-05

        else:
            try:
                net, optimizer, get_mse_loss, last_epoch, _ = load_checkpoint()
                last_epoch += 1 # Continue from next epoch
            except FileNotFoundError:
                print("Checkpoint could not be found under {}!".format(checkpoint_path))
                exit(1)

        # Constructing output folder
        output_folder_path = construct_output_folder("train")

        # Prints parameter settings (user-provided input or default values) given from the console
        show_current_config("train")

        # Saves parameter settings to a file under output_folder_path
        write_current_config(output_folder_path)
        
        # Device: CPU or CUDA
        device = torch.device(globals.DEVICES[globals.ARGS.nogpu])
        print('Device: ' + str(device))
        net.to(device=device)

        # Get loaders as dict
        loaders = get_train_loaders()
        print("Images loaded.")
        
        # Initialization
        train_loader = loaders['train']
        validation_loader = loaders['validation']
        csv_line_template = "{},{},{},{}\n"
        print("Training started.")

        try:
            with open(output_folder_path + "train_val_loss_psnr.csv", "w+") as file:
                file.write(csv_line_template.format("train_loss", "val_loss", "train_psnr", "val_psnr"))
                MAX_EPOCH = 101
                for epoch in range(last_epoch, MAX_EPOCH):
                    # Train over full dataset (1 epoch)
                    train_loss, train_psnr = train(train_loader, net, device, get_mse_loss, optimizer, epoch)
                    print('* Training loss for current epoch: %.4f' % train_loss)
                    print('* Training PSNR for current epoch: %.4f' % train_psnr)

                    # Validation over validation set
                    val_loss, val_psnr = validation(validation_loader, net, device, get_mse_loss, epoch)
                    print('* Validation loss for current epoch: %.4f' % val_loss)
                    print('* Validation PSNR for current epoch: %.4f' % val_psnr)

                    file.write(csv_line_template.format(train_loss, val_loss, train_psnr, val_psnr))
                    save_checkpoint(net, optimizer, get_mse_loss, epoch)
                    print()

        except KeyboardInterrupt:
            print("\nKeyboard interrupt, stoping execution...\n")
            
        finally:
            print('Training finished!')

if __name__ == "__main__":
    globals.initialize()
    main()