from arg_help_formatter import *

def arg_handler():
    parser = argparse.ArgumentParser(description='Image Super Resolution with PyTorch: [Test]', 
                                     formatter_class=ArgHelpFormatter, 
                                     add_help=False)
    # Optional flags
    parser.add_argument("-h", "--help", help="Help message", action="store_true")
    parser.add_argument("-ng", "--nogpu", help="Use GPU", default=False, action="store_true")

    # Required flags
    enable_exec = ("-h" not in sys.argv)
    group = parser.add_argument_group(title='required arguments')

    group.add_argument("-of", "--outputfolder", 
                       type=str,
                       help="Specify output folder name (default: '')", 
                       metavar="FOLDER",
                       default="")

    group.add_argument("-c", "--checkpoint", 
                       type=str,
                       help="Specify checkpoint path", 
                       metavar="PATH",
                       default="")

    args = parser.parse_args()

    # Print help if -h is used
    if args.help:
        parser.print_help()
        return

    return args