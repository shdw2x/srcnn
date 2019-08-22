import globals

import arg_helper_test
from srcnn import SRNET
from srcnn_utils import *

# Testing
def test(test_loader, net, device):
    net.eval()
    output_path = globals.ARGS.outputfolder
    with torch.no_grad():
        for i, (y, cb, cr, paths) in enumerate(test_loader, 1):
            y = y.to(device)
            y_pred, cb, cr, input_path = net(y).cpu()[0], cb[0], cr[0], paths[0]

            path_no_extension, extension = os.path.splitext(input_path)
            filename = os.path.basename(path_no_extension)
            save_output_image(output_path + filename + "_test" + extension, y_pred, cb, cr)

def main():
    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.manual_seed(globals.SEED) # seed is used for pseudo random initialization of network parameters
    globals.ARGS = arg_helper_test.arg_handler()

    # If required args are parsed properly
    if globals.ARGS:
        # Load network from checkpoint file specified with flag
        checkpoint_path = globals.ARGS.checkpoint
        net = None
        try:
            net, *_, scale_factor = load_checkpoint()
            globals.ARGS.scalefactor = scale_factor
        except FileNotFoundError:
            print("Checkpoint could not be found under {}!".format(checkpoint_path))
            exit(1)

        # Constructing output folder
        output_folder_path = construct_output_folder("test")

        # Prints parameter settings (user-provided input or default values) given from the console
        show_current_config("test")
            
        # Device: CPU or CUDA
        device = torch.device(globals.DEVICES[globals.ARGS.nogpu])
        print('Device: ' + str(device))
        net.to(device=device)

        # Get loaders as dict
        loader = get_test_loader()
        print("Images loaded.")

        test_loader = loader['test']

        print("Testing started.")
        test(test_loader, net, device)
        print('Testing finished!')

if __name__ == "__main__":
    globals.initialize()
    main()