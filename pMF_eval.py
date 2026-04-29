import argparse
import os
import sys

# # Reuse model code and utilities from the pMF directory
# _PMF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
# sys.path.insert(0, _PMF_DIR)

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

from huggingface_hub import hf_hub_download

import pMF.utils.torch_dist_util as dist
import pMF.utils.torch_util as tu
from pMF.pmf import pixelMeanFlow
from pMF.evaluate import run_evaluate


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', type=str, required=True)
    parser.add_argument('--hf-repo-id', type=str, required=True)
    parser.add_argument('--hf-filename', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--img-size', type=int, required=True)
    parser.add_argument('--fid-ref', type=str, required=True)
    parser.add_argument('--num-samples', default=50000, type=int)
    parser.add_argument('--gen-bsz', type=int, default=64)
    parser.add_argument('--sample-seed', default=42, type=int)
    parser.add_argument('--num-sampling-steps', default=1, type=int)
    parser.add_argument('--cfg-omega', default=7.5, type=float)
    parser.add_argument('--interval-min', default=0.1, type=float)
    parser.add_argument('--interval-max', default=0.8, type=float)
    parser.add_argument('--save-samples', action='store_true')
    return parser


def main(args):
    dist.initialize()

    if dist.process_index() == 0:
        os.makedirs(args.workdir, exist_ok=True)

    tu.seed(0)

    ckpt_path = hf_hub_download(repo_id=args.hf_repo_id, filename=args.hf_filename)

    model = pixelMeanFlow(args.model, img_size=args.img_size)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    model = tu.device_put(model)

    run_evaluate(
        model,
        args.workdir,
        fid_ref=args.fid_ref,
        num_samples=args.num_samples,
        device_batch_size=args.gen_bsz,
        initial_seed=args.sample_seed,
        keep_samples=args.save_samples,
        num_steps=args.num_sampling_steps,
        omega=args.cfg_omega,
        t_min=args.interval_min,
        t_max=args.interval_max,
    )


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)