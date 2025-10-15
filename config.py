import os
import torch
import sys


class POPAR_3D_Org:

    server = "sol"  # server = lab | bridges2 | agave | sol
    debug_mode = False
    space_x = 1.5
    space_y = 1.5
    space_z = 2.0
    a_min = -175
    a_max = 250
    adaptive_shuffle_update_epoch = 5
    adaptive_shuffle_update_step = 0.05
    def __init__(self, args):
        self.image_size = args.image_size
        self.patch_size = args.patch_size
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_steps = args.num_steps
        self.epochs = args.epochs
        self.eval_num = args.eval_num
        self.task = args.task
        self.losses = args.losses
        self.dataset = args.dataset


        self.backbone = args.backbone
        self.continue_pretrain = args.continue_pretrain

        self.method = "popar3d_Org_" + self.server + "_" +self.backbone+"_" + "_".join(self.task) + "_" + "_".join(self.losses) + "_" + "+".join(self.dataset) + "_i_" + str(self.image_size) + "_p_" + str(self.patch_size)
        if self.continue_pretrain:
            self.method+= "_continue_pretrain"


        self.grad_clip = args.grad_clip
        self.opt = args.opt
        self.lrdecay = args.lrdecay
        self.decay = args.decay
        self.momentum = args.momentum
        self.lr = args.lr
        self.lr_schedule = args.lr_schedule
        self.max_grad_norm = args.max_grad_norm
        self.warmup_steps = args.warmup_steps

        self.weight = args.weight



        if self.server == "agave" or self.server == "sol":
            self.model_path = os.path.join("/data/jliang12/jpang12/POPAR_3D/pretrained_weight",self.method)


        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path,  exist_ok=True)
        logs_path = os.path.join(self.model_path, "Logs")
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)

        if os.path.exists(os.path.join(logs_path, "log.txt")):
            self.log_writter = open(os.path.join(logs_path, "log.txt"), 'a')
        else:
            self.log_writter = open(os.path.join(logs_path, "log.txt"), 'w')


        self.graph_path = os.path.join(logs_path, "graph_path")
        if not os.path.exists(self.graph_path):
            os.makedirs(self.graph_path)

        if self.debug_mode:
            self.log_writter = sys.stdout
            self.log_writter2 = sys.stdout
            self.eval_num = 5


    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:", file=self.log_writter)
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)), file=self.log_writter)
        print("\n", file=self.log_writter)
