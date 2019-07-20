import argparse
import sys

# Disable multiple occurences of the same flag
class UniqueStore(argparse.Action):
    def __call__(self, parser, namespace, values, option_string):
        if getattr(namespace, self.dest, self.default) is not None:
            parser.error(option_string + " appears several times.")
        setattr(namespace, self.dest, values)

# Custom format for arg Help print
class CustomFormatter(argparse.HelpFormatter):
    def __init__(self,
                 prog,
                 indent_increment=2,
                 max_help_position=100, # Modified
                 width=100):
        super().__init__(prog, indent_increment, max_help_position, width)

    def _format_action_invocation(self, action):
        if not action.option_strings:
            metavar, = self._metavar_formatter(action, action.dest)(1)
            return metavar
        else:
            parts = []
            if action.nargs == 0:
                parts.extend(action.option_strings)
            else:
                default = action.dest.upper()
                args_string = self._format_args(action, default)
                for option_string in action.option_strings:
                    parts.append('%s' % option_string)
                parts[-1] += ' %s' % args_string
            return ', '.join(parts)

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
        assert len_layer == len_ks, "Layer count ({}) does not match with kernel sizes ({})".format(len_layer, len_ks)
        assert len_layer == len_kc, "Layer count ({}) does not match with kernel counts ({})".format(len_layer, len_kc)
        assert len_layer == len_rp, "Layer count ({}) does not match with relu positions ({})".format(len_layer, len_rp)
        assert len_layer == len_lr, "Layer count ({}) does not match with learning rates ({})".format(len_layer, len_lr)
    except AssertionError as e:
        print(e)
        exit(1)

# Handles cmd args
def arg_handler():
    parser = argparse.ArgumentParser(description='Image Super Resolution with PyTorch', 
                                     formatter_class=CustomFormatter, 
                                     add_help=False)
    # Optional flags
    parser.add_argument("-h", "--help", help="Help message", action="store_true")

    # Required flags
    enable_exec = ("-h" not in sys.argv)
    group = parser.add_argument_group(title='required arguments')
    
    group.add_argument("-p", "--pipe",  help="Specify pipeline execution mode", type=str,
                       choices=['train', 'test', 'full'], required=enable_exec, action=UniqueStore)
    
    group.add_argument("-bs", "--batchsize",  help="Specify batch size (e.g. 16)", type=convert_positive_int, 
                       metavar="BATCH", required=enable_exec)
    
    group.add_argument("-cl", "--convlayers",  
                       help="Specify number of convolutional layers (e.g. 1, 2, 4)", 
                       type=convert_positive_int, metavar="CONV", default=3, required=enable_exec)
    
    group.add_argument("-ks", "--kernelsizes", nargs='+',
                       type=convert_positive_int,
                       help="Specify kernel sizes for each convolutional layer (e.g. [1, 2, 3...])",
                       metavar="KSIZES", default=[9, 1, 5], required=enable_exec)
    
    group.add_argument("-kc", "--kernelcounts", nargs='+',
                       type=convert_positive_int,
                       help="Specify number of kernels for each convolutional layer (e.g. [2, 4, 8...])", 
                       metavar="KCOUNTS", default=[64, 32, 3], required=enable_exec) #TODO: Think about last channel due to grayscale
    
    group.add_argument("-rp", "--relupositions", nargs='+',
                       type=convert_positive_int,  
                       help="Specify after which convolutional layer ReLU takes place (e.g. 3, [1, 3, 5])", 
                       metavar="POS", default=[2], required=enable_exec)
    
    group.add_argument("-lr", "--learnrates", nargs='+',
                       type=convert_positive_float,
                       help="Specify learning rate (e.g. [0.0001, 0.1])", 
                       metavar="LR", required=enable_exec)

    args = parser.parse_args()

    group_arg_list_size_check(args)

    # Print help if -h is used
    if args.help:
        parser.print_help()
        return

    return args