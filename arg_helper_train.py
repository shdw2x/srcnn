from arg_help_formatter import *
from srcnn_utils import get_pad_count

def convert_positive_int(value):
    try:
        value = int(value)
        assert (value > 0)
    except Exception:
        raise argparse.ArgumentTypeError("Positive integer is expected but got: {}".format(value))
    return value

def convert_positive_float(value):
    try:
        value = float(value)
        assert (value > 0)
    except Exception:
        raise argparse.ArgumentTypeError("Positive float is expected but got: {}".format(value))
    return value

def group_arg_list_size_check(args):

    len_layer = args.convlayers
    len_ks = len(args.kernelsizes)
    len_kc = len(args.kernelcounts)
    len_rp = len(args.relupositions)
    len_lr = len(args.learnrates)
    
    try:
        assert len_layer == len_ks, "Layer count ({}) does not match with number of elements in the kernel sizes list ({})".format(len_layer, len_ks)
        assert len_layer == len_kc, "Layer count ({}) does not match with number of elements in the kernel counts list ({})".format(len_layer, len_kc)
        assert len_layer >= len_rp, "Layer count ({}) is smaller than the number of relu positions given ({})".format(len_layer, len_rp)
        assert len_layer == len_lr, "Layer count ({}) does not match with learning rates ({})".format(len_layer, len_lr)

        # Checking each kernel size for padding
        for ks in args.kernelsizes:
            padding = get_pad_count(ks)
            assert padding.is_integer(), "Invalid kernel size ({}) for proper calculation of padding ({})".format(ks, padding)

        # Sort ReLU Positions
        relu_positions = args.relupositions = sorted(args.relupositions)

        # Detect duplicate ReLU layer positions if any
        for i in range(len(relu_positions)-1):
            assert relu_positions[i] not in relu_positions[i+1:], "Duplicate positions given for ReLU layers"

        # Adjusting ReLU Layer positions according to Convolutional Layers
        # Throws error if invalid ReLU position given 
        for i in range(len_rp):
            assert relu_positions[i] != len_layer, "ReLU layer ({}) cannot be the last layer of the neural network".format(relu_positions[i])
            assert relu_positions[i] < len_layer, "Invalid position for ReLU layer ({})".format(relu_positions[i])
            relu_positions[i] += i

    except AssertionError as e:
        print(e)
        exit(1)

# Handles cmd args
def arg_handler():
    parser = argparse.ArgumentParser(description='Image Super Resolution with PyTorch: [Train]', 
                                     formatter_class=ArgHelpFormatter, 
                                     add_help=False)
    # Optional flags
    parser.add_argument("-h", "--help", help="Help message", action="store_true")
    parser.add_argument("-ng", "--nogpu", help="Use GPU", default=False, action="store_true")

    # Required flags
    enable_exec = ("-h" not in sys.argv)
    group = parser.add_argument_group(title='required arguments')

    group.add_argument("-sf", "--scalefactor",
                       help="Specify scale factor for upsampling (default: 2)",
                       type=convert_positive_int,
                       metavar="FACTOR",
                       default=2)
    
    group.add_argument("-bs", "--batchsize",  
                       help="Specify batch size (default: 1)", 
                       type=convert_positive_int, 
                       metavar="BATCH",
                       default=1)
    
    group.add_argument("-cl", "--convlayers",  
                       help="Specify number of convolutional layers (default: 3)", 
                       type=convert_positive_int, 
                       metavar="CONV", 
                       default=3)
    
    group.add_argument("-ks", "--kernelsizes", 
                       nargs='+',
                       type=convert_positive_int,
                       help="Specify kernel sizes for each convolutional layer (default: 9 1 5...)",
                       metavar="KSIZES", 
                       default=[9, 1, 5])
    
    group.add_argument("-kc", "--kernelcounts", 
                       nargs='+',
                       type=convert_positive_int,
                       help="Specify number of kernels for each convolutional layer (default: 64 32 1...)", 
                       metavar="KCOUNTS", 
                       default=[64, 32, 1])
    
    group.add_argument("-rp", "--relupositions", 
                       nargs='+',
                       type=convert_positive_int,  
                       help="Specify after which convolutional layer ReLU takes place (default: 1 2 3...)", 
                       metavar="POS", 
                       default=[1, 2])
    
    group.add_argument("-lr", "--learnrates", 
                       nargs='+',
                       type=convert_positive_float,
                       help="Specify learning rate (default: 0.0001 0.0001 0.00001...)", 
                       metavar="LR",
                       default=[1e-04, 1e-04, 1e-05])

    group.add_argument("-of", "--outputfolder", 
                       type=str,
                       help="Specify output folder name (default: '')", 
                       metavar="FOLDER",
                       default="")

    group.add_argument("-c", "--checkpoint", 
                       type=str,
                       help="Specify checkpoint path (default: '')", 
                       metavar="PATH",
                       default="")

    args = parser.parse_args()

    # Print help if -h is used
    if args.help:
        parser.print_help()
        return

    group_arg_list_size_check(args)
    return args