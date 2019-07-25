def initialize():
    global ARGS, DEVICES, LOG_DIR, VISUALIZE, DATA_ROOT
    ARGS = None
    DEVICES = {False: 'cpu', True: 'cuda'}
    # DEVICE_ID = 'cpu' # set to 'cpu' for cpu, 'cuda' / 'cuda:0' or similar for gpu.
    LOG_DIR = 'checkpoints'
    VISUALIZE = False # set True to visualize input, prediction and the output from the last batch
    DATA_ROOT = './dataset/'