from datetime import datetime
import torch
from utils import get_model_attribute


class Args:
    """
    Program configuration
    """

    def __init__(self):
        # Device
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        # Logging / temp
        self.clean_tensorboard = False
        self.clean_temp = False
        self.log_tensorboard = True
        # Print per-batch metrics (recon and total loss)
        self.print_batch_stats = True

        # Algorithm Version
        # GraphRNN | DFScodeRNN (GraphGen) | DGMG (Deep GMG)
        self.note = 'DFScodeRNN'

        # Dataset selection
        self.graph_type = 'amazon'   # CHANGE to 'enzymes' for the second run
        self.num_graphs = None  # None => use entire dataset

        # Output formats
        self.produce_graphs = True
        self.produce_min_dfscodes = True
        self.produce_min_dfscode_tensors = True

        # GraphRNN specific (if used)
        self.max_prev_node = None

        # Specific to GraphRNN
        # Model parameters
        self.hidden_size_node_level_rnn = 128  # hidden size for node level RNN
        self.embedding_size_node_level_rnn = 64  # the size for node level RNN input
        self.embedding_size_node_output = 64  # the size of node output embedding
        self.hidden_size_edge_level_rnn = 16  # hidden size for edge level RNN
        self.embedding_size_edge_level_rnn = 8  # the size for edge level RNN input
        self.embedding_size_edge_output = 8  # the size of edge output embedding

        # Model hyperparameters: DFScodeRNN/GraphGen
        self.hidden_size_dfscode_rnn        = 256
        self.embedding_size_dfscode_rnn     = 92
        self.embedding_size_timestamp_output = 512
        self.embedding_size_vertex_output   = 512
        self.embedding_size_edge_output     = 512
        self.dfscode_rnn_dropout            = 0.2
        self.rnn_type                       = 'LSTM'  # LSTM | GRU
        self.num_layers                     = 4

        # Specific to DGMG
        self.feat_size = 128
        self.hops      = 2
        self.dropout   = 0.2

        # Loss settings
        self.loss_type = 'BCE'     # 'BCE' or 'NLL'
        self.weights   = False     # whether to weight reconstruction

        # ** Hierarchy-Penalty Hyperparameters **
        # margin m for hinge (child must be = m deeper than parent)
        self.alpha = 0.25          # matches DeCoGen's tuned alpha, for a fair comparison
        self.lambda_hier = 5       # matches DeCoGen's tuned lambda_hier (per earlier correction)
        self.hyperparameter_search = True  # Enable grid search if desired

        # ** Direct-Supervision Variant (Exp 1c) **
        # When True, replaces the soft margin/ranking hierarchy loss with
        # direct MSE supervision: predicted_depth(child) = predicted_depth(parent) + 1.
        # See _hierarchy_penalty_direct_supervision in train.py.
        self.direct_supervision = True   # NEW

        # Training config
        self.batch_size        = 32
        self.num_workers       = 0
        self.epochs            = 5000
        self.lr                = 0.003
        self.gamma             = 0.3
        self.milestones        = [100, 200, 400, 800]
        self.gradient_clipping = True

        # Early stopping
        self.early_stop = True       # enable or disable early stopping
        self.patience   = 100     # validation checks without improvement before stopping
        self.threshold  = 0.000     # minimum relative improvement (0.05%) to reset patience

        # Output paths
        self.dir_input        = ''
        self.model_save_path  = self.dir_input + 'model_save/'
        self.tensorboard_path = self.dir_input + 'tensorboard/'
        self.dataset_path     = self.dir_input + 'datasets/'
        self.temp_path        = self.dir_input + 'tmp/'

        # Model save / validate
        self.save_model      = True
        self.epochs_save     = 1
        self.epochs_validate = 1

        # When to print batch stats relative to saved epochs
        # By default, print every time we save the model
        self.print_on_saved_epoch = True

        # Timestamp
        self.time = '{0:%Y-%m-%d %H:%M:%S}'.format(datetime.now())

        # Filenames
        self.fname = f"{self.note}_{self.graph_type}_directsup"   # tagged so checkpoints are distinguishable
        self.current_model_save_path    = (
            f"{self.model_save_path}{self.fname}_{self.time}/"
        )
        self.current_dataset_path       = None
        self.current_processed_dataset_path = None
        self.current_min_dfscode_path   = None
        self.current_temp_path          = (
            f"{self.temp_path}{self.fname}_{self.time}/"
        )

        # Model loading
        self.load_model      = False
        self.load_model_path = ''
        self.load_device     = torch.device('cuda:0')
        self.epochs_end      = 10000

    def update_args(self):
        if self.load_model:
            args = get_model_attribute(
                'saved_args', self.load_model_path, self.load_device)
            # preserve runtime overrides
            args.device            = self.load_device
            args.load_model        = True
            args.load_model_path   = self.load_model_path
            args.epochs            = self.epochs_end

            args.clean_tensorboard = False
            args.clean_temp        = False

            args.produce_graphs            = False
            args.produce_min_dfscodes      = False
            args.produce_min_dfscode_tensors = False

            # preserve early stopping settings
            args.early_stop = self.early_stop
            args.patience   = self.patience
            args.threshold  = self.threshold

            return args

        return self