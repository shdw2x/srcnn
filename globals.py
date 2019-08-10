def initialize():
    global ARGS, DEVICES, VISUALIZE, DATA_ROOT, OUTPUT_ROOT, TITLE, SAVED_PICS, PRINT_MESSAGE_FREQUENCY, DRAW_IMAGE_FREQUENCY
    ARGS = None
    DEVICES = {False: 'cuda', True: 'cpu'}
    # DEVICE_ID = 'cpu' # set to 'cpu' for cpu, 'cuda' / 'cuda:0' or similar for gpu.
    VISUALIZE = False # set True to visualize input, prediction and the output from the last batch
    DATA_ROOT = './dataset/'
    OUTPUT_ROOT = './outputs/'
    TITLE = ["input", "pred", "output"]
    SAVED_PICS = {
        "Train": "",
        "Validation": "./dataset/validation\\images\\butterfly_GT.bmp"
    }
    PRINT_MESSAGE_FREQUENCY = 1000 # PRINT_MESSAGE_FREQUENCY = k => print message every kth iteration
    DRAW_IMAGE_FREQUENCY = 1 # e.g. DRAW_IMAGE_FREQUENCY = k => draw image trio every kth epoch

