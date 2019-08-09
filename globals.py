def initialize():
    global ARGS, DEVICES, VISUALIZE, DATA_ROOT, OUTPUT_ROOT, TITLE, SAVED_TRAIN_PIC, SAVED_VAL_PIC
    ARGS = None
    DEVICES = {False: 'cuda', True: 'cpu'}
    # DEVICE_ID = 'cpu' # set to 'cpu' for cpu, 'cuda' / 'cuda:0' or similar for gpu.
    VISUALIZE = False # set True to visualize input, prediction and the output from the last batch
    DATA_ROOT = './dataset/'
    OUTPUT_ROOT = './outputs/'
    TITLE = ["input", "pred", "output"]
    SAVED_TRAIN_PIC = ""
    SAVED_VAL_PIC = "./dataset/validation\\images\\butterfly_GT.bmp" 