#!/usr/bin/env python3
# Standard libraries
from copy import deepcopy
import time

import arg_helper
from srcnn_utils import *

# SRCNN
class SRNET(nn.Module):
    def __init__(self):
        # Base class initialization
        super(SRNET, self).__init__() #TODO: Search super (with parameters)

        self.layers = []
        calculate_padding = lambda f: int(arg_helper.get_pad_count(f))

        # Adding layers
        # Adding first convolutional layer
        kernel_size = globals.ARGS.kernelsizes[0]
        self.layers.append(nn.Conv2d(in_channels=1, out_channels=globals.ARGS.kernelcounts[0], 
                                     kernel_size=kernel_size, padding=calculate_padding(kernel_size), bias=True))

        # Adding remaining convolutional layers
        for i in range(1, globals.ARGS.convlayers):
            kernel_size = globals.ARGS.kernelsizes[i]
            self.layers.append(nn.Conv2d(in_channels=globals.ARGS.kernelcounts[i-1], out_channels=globals.ARGS.kernelcounts[i], 
                                         kernel_size=kernel_size, padding=calculate_padding(kernel_size), bias=True))
              
        # Adding ReLUs
        for i in globals.ARGS.relupositions:
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
    # PyTorch's own optimization for training
    net.train()
    
    train_loss = AverageMeter() # Keep training loss for the entire epoch
    psnr_val = AverageMeter()

    print_message_frequency = 500 # print_message_frequency = k => draw every kth iteration
    draw_image_frequency = 5 # e.g. draw_image_frequency = k => draw every kth epoch
    inputs = None
    preds = None
    targets = None

    # For each batch (iteration)
    for iteri, (inputs, targets, paths) in enumerate(train_loader, 1):        
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad() # Clear gradients
        preds = net(inputs) # Forward propagation
        rmse_loss = torch.sqrt(get_mse_loss(preds, targets)) # get_mse_loss(y_hat, ground_truth)
        psnr_val.update(compute_psnr(rmse_loss.item()))
        rmse_loss.backward() # Backpropagation
        optimizer.step()
        train_loss.update(rmse_loss.item()) # TODO: check

        # Print every print_message_frequency mini-batches
        if (not iteri % print_message_frequency):
            print('- Training: [Epoch: %d, Iteration: %3d] Loss: %.4f PSNR: %.4f' % (epoch, iteri, rmse_loss.item(), psnr_val.loss))

        # Visualize and save image results (input, pred, target) periodically according to frequency value (in epochs)
        if (draw_image_frequency == 1) or (epoch % draw_image_frequency == 0):
            if not globals.SAVED_TRAIN_PIC:
                globals.SAVED_TRAIN_PIC = paths[0]
            if globals.SAVED_TRAIN_PIC == paths[0]:
                image_name = globals.ARGS.outputfolder + "train_" + str(epoch)
                visualize_trio(inputs[0], preds[0], targets[0], paths[0], image_name)

    # Return the average loss of all batches in this epoch
    return train_loss.avg, psnr_val.avg

# Testing or Validation
def test(test_loader, net, device, get_mse_loss, epoch):
    net.eval()

    with torch.no_grad():
        test_loss = AverageMeter()
        psnr_val = AverageMeter()
        inputs = None
        preds = None
        targets = None
        print_message_frequency = 500 # print_message_frequency = k => draw every kth iteration
        draw_image_frequency = 1 # e.g. draw_image_frequency = k => draw every kth epoch

        for iteri, (inputs, targets, paths) in enumerate(test_loader, 1): # Ignoring 3rd element of tuple which is path of the image(s)
            inputs, targets = inputs.to(device), targets.to(device)
            preds = net(inputs)
            rmse_loss = torch.sqrt(get_mse_loss(preds, targets)) # get_mse_loss(y_hat, ground_truth)
            psnr_val.update(compute_psnr(rmse_loss.item()))
            test_loss.update(rmse_loss.item())

            # Print every print_message_frequency mini-batches
            if (not iteri % print_message_frequency):
                print('- Validation: [Epoch: %d, Iteration: %3d] Loss: %.4f PSNR: %.4f' % (epoch, iteri, rmse_loss.item(), psnr_val.loss))

            # Visualize and save image results (input, pred, target) periodically according to frequency value (in epochs)
            if (draw_image_frequency == 1) or (epoch % draw_image_frequency == 0):
                if not globals.SAVED_VAL_PIC:
                    globals.SAVED_VAL_PIC = paths[0]
                if globals.SAVED_VAL_PIC == paths[0]:
                    image_name = globals.ARGS.outputfolder + "val_" + str(epoch)
                    visualize_trio(inputs[0], preds[0], targets[0], paths[0], image_name)

    return test_loss.avg, psnr_val.avg

def main():
    torch.multiprocessing.set_start_method('spawn', force=True)
    globals.ARGS = arg_helper.arg_handler()

    # If required args are parsed properly
    if globals.ARGS:
        output_root = globals.OUTPUT_ROOT # outputs/
        output_folder_name = globals.ARGS.outputfolder # outputs/output_folder_name

        # Check if outputs directory exists, if not create the directory
        if not os.path.exists(output_root):
            os.mkdir(output_root)

        # Check if output folder name is provided from console, if not create a default name
        if not output_folder_name:
            output_folder_name = create_output_folder_name()

        # Output folder path
        subfolder = output_root + output_folder_name + "/"
        globals.ARGS.outputfolder = subfolder

        # Check if output_folder_name exists, if not create one, otherwise abort
        if not os.path.exists(subfolder):
            os.mkdir(subfolder)
        else:
            print("Directory already exists: {}".format(subfolder))
            exit(1)

        # Prints parameter settings (user-provided input or default values) given from the console
        show_current_config()

        # Saves parameter settings to a file under subfolder directory
        #write_current_config(subfolder)

        # Construct network
        torch.manual_seed(5) # seed is used for pseudo random initialization of network parameters (weights and biases)
        device = torch.device(globals.DEVICES[globals.ARGS.nogpu])
        print('Device: ' + str(device))
        net = SRNET().to(device=device)

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

        # for group in optimizer.param_groups:
        #     print(group["params"][0].size()) # weight
        #     print(group["params"][1].size()) # bias
        #     print()

        # Learning rate scheduler
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, 
                                                        #  patience=lr_patience, min_lr=min_lr, verbose=True)
        # Determine pipeline execution
        pipe = globals.ARGS.pipe
        run_full = (pipe == "full")
        run_train = run_full or (pipe == "train")
        run_test = run_full or (pipe == "test")

        # Get loaders as dict
        loaders = get_loaders(device, load_train=run_train, load_test=run_test)
        print("Images are loaded.")
        # Training mode
        if (run_train):
            print("Training started.")
            # Initialization
            train_loader = loaders['train']
            val_loader = loaders['validation']
            csv_line_template = "{},{},{},{}\n"

            try:
                with open(subfolder + "train_val_loss_psnr.csv", "w+") as file:
                    file.write(csv_line_template.format("train_loss", "val_loss", "train_psnr", "val_psnr"))
                    MAX_EPOCH = 101
                    for epoch in range(1, MAX_EPOCH):
                        # Train over full dataset (1 epoch)
                        train_loss, train_psnr = train(train_loader, net, device, get_mse_loss, optimizer, epoch)
                        print()
                        print('* Training loss for current epoch: %.4f' % train_loss)
                        print('* Training PSNR for current epoch: %.4f' % train_psnr)

                        # Validation over validation set
                        val_loss, val_psnr = test(val_loader, net, device, get_mse_loss, epoch)
                        print('* Validation loss for current epoch: %.4f' % val_loss)
                        print('* Validation PSNR for current epoch: %.4f' % val_psnr)

                        file.write(csv_line_template.format(train_loss, val_loss, train_psnr, val_psnr))
                        save_checkpoint(net, globals.ARGS.outputfolder, epoch)

            except KeyboardInterrupt:
                print("\nKeyboard interrupt, stoping execution...\n")
                
            finally:
                pass
                # print('Training finished!')
                # print('Saving training data...')
                # draw_train_val_plots(train_losses, val_losses, path=globals.LOG_DIR, show=False)
                # draw_accuracy_plot(accuracies, len(train_losses), path=globals.LOG_DIR, show=False)
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
                # save_stats("stats.txt", stats, path=globals.LOG_DIR)
                # print('Saved.')
        # Testing mode
        elif (run_test):
            print("Testing started.")
            # test_loader = loaders['test']
            test() # TODO: Give params

if __name__ == "__main__":
    globals.initialize()
    main()