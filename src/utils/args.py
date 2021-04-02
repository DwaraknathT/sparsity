import argparse
import os

parser = argparse.ArgumentParser(description="Ensemble Training")
# Model Parameters
parser.add_argument("--model", default="vgg19_bn", type=str, help="Model to use")
parser.add_argument("--run_name", default=None, type=str, help="Name of this run")
parser.add_argument("--model_type", default="dense", type=str, help="Dense/sparse")
parser.add_argument("--seed", default=1, type=int, help="random seed")
parser.add_argument("--mode", default="train", type=str, help="In eval mode or not")
parser.add_argument("--load_model", default=False, type=bool, help="Load prev weights")
parser.add_argument("--resume", default=False, type=bool, help="Resume training")

# Training params
parser.add_argument("--dataset", default="cifar10", type=str, help="Dataset to use")
parser.add_argument("--num_classes", default=10, type=int, help="Number of classes")
parser.add_argument("--optim", default="sgd", type=str, help="Optimizer to use")
parser.add_argument("--lr", default=0.05, type=float, help="Initial LR")
parser.add_argument("--lr_schedule", default="swa", type=str, help="LR scheduler")
parser.add_argument("--lr_cycle", default="full", type=str, help="Full or half cycle")
parser.add_argument("--up_step", default=5000, type=int, help="Cyclic lr step size")
parser.add_argument("--down_step", default=5000, type=int, help="Cyclic lr step size")
parser.add_argument(
    "--milestones",
    default=[25000, 50000, 75000, 90000],
    type=list,
    help="Multi step lr",
)

parser.add_argument("--epochs", default=200, type=int, help="No of epochs")
parser.add_argument("--clip", default=1, type=int, help="Gradient clipping")
parser.add_argument("--steps", default=None, type=int, help="No of steps")
parser.add_argument("--steps_per_epoch", default=500, type=int, help="No of steps")
parser.add_argument("--eval_step", default=1000, type=int, help="Eval every this steps")
parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
parser.add_argument(
    "--output_dir",
    type=str,
    help="Output directory for storing ckpts. Default is in runs/hparams",
)
parser.add_argument(
    "--use_colab", type=bool, default=False, help="Use Google colaboratory"
)

# sparsity params
parser.add_argument("--ramping", type=bool, default=False, help="Use ramping sparsity")
parser.add_argument("--snip", type=bool, default=False, help="Use ramping sparsity")
parser.add_argument(
    "--compute_aware",
    type=bool,
    default=False,
    help="Use compute aware snip condition",
)
parser.add_argument("--snip_batch", type=int, default=1, help="No of batches for snip")
parser.add_argument(
    "--carry_mask", type=bool, default=False, help="Carry mask in ramping pruning"
)
parser.add_argument(
    "--initial_sparsity", default=0.0, type=float, help="Initial sparsity"
)
parser.add_argument("--final_sparsity", default=0.75, type=float, help="Final sparsity")
parser.add_argument("--start_step", default=0, type=float, help="Pruning start step")
parser.add_argument(
    "--end_step", default=0.2, type=float, help="Stop pruning at this step"
)
parser.add_argument("--prune_freq", default=100, type=int, help="Prune every x steps")
parser.add_argument(
    "--global_prune", default=False, type=bool, help="Layer wise pruning"
)
parser.add_argument(
    "--prune_type", default="weight", type=str, help="Weight or unit pruning."
)
parser.add_argument(
    "--ramp_type",
    default="linear",
    type=str,
    help="Ramp type: linear, full_cycle, half_cycle",
)
parser.add_argument("--ramp_cycle_step", default=None, type=int, help="Ramp cycle step")

# mask params
parser.add_argument(
    "--union_mask", default=False, type=bool, help="Take the union of masks"
)

# Attack params
parser.add_argument(
    "--attack", default="fgsm", type=str, help="Adversarial attack to use"
)

# Transformer arguments
parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--d_inner_hid", type=int, default=2048)
parser.add_argument("--d_k", type=int, default=512)
parser.add_argument("--d_v", type=int, default=512)

parser.add_argument("--n_head", type=int, default=8)
parser.add_argument("--n_layers", type=int, default=6)
parser.add_argument("--n_warmup_steps", type=int, default=4000)
parser.add_argument("--lr_mul", type=float, default=1.0)
parser.add_argument("--embs_share_weight", action="store_true")
parser.add_argument("--proj_share_weight", action="store_true")
parser.add_argument("--scale_emb_or_prj", type=str, default="prj")

parser.add_argument("--use_tb", action="store_true")
parser.add_argument("--save_mode", type=str, choices=["all", "best"], default="best")
parser.add_argument("--label_smoothing", action="store_true")


def get_args():
    args = parser.parse_args()
    if args.run_name is None:
        raise ValueError("Ernter a run name")
    if args.model_type == "dense":
        args.output_dir = "{}/{}/lr_{}/{}".format(
            args.dataset, args.model, args.lr_schedule, args.run_name
        )
    else:
        args.output_dir = "{}/{}/lr_{}/{}/{}_{}_{}/{}/{}".format(
            args.dataset,
            args.model,
            args.lr_schedule,
            args.final_sparsity,
            args.start_step,
            args.end_step,
            args.prune_freq,
            args.ramp_type,
            args.run_name,
        )
    if args.use_colab:
        from google.colab import drive

        drive.mount("/content/gdrive")
        colab_str = "/content/gdrive/My Drive/sparsity/"
        OUTPUT_DIR = "{}/{}".format(colab_str, args.output_dir)
        if not os.path.isdir(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        args.output_dir = OUTPUT_DIR
    else:
        output_dir = "runs/{}".format(args.output_dir)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        args.output_dir = "runs/{}".format(args.output_dir)

    return args
