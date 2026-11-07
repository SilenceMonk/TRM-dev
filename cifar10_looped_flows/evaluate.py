"""EMA PNG export and optional fixed torch-fidelity Inception FID against official train50k."""
import argparse
import json
from pathlib import Path
import time
import torch
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from model import LoopedFlow, sample
from data import ROOT


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--num-images',type=int,default=5000)
    p.add_argument('--flow-steps',type=int,default=32,choices=[8,16,32,64])
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--seed',type=int,default=12345)
    p.add_argument('--reset-state',action='store_true')
    p.add_argument('--fid',action='store_true')
    p.add_argument('--compile',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(3)
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    images_dir=output/'generated'
    images_dir.mkdir(exist_ok=True)
    if list(images_dir.glob('*.png')):
        raise FileExistsError('Use a fresh output directory to avoid mixing samples')
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    net=LoopedFlow().cuda().eval()
    net.load_state_dict(checkpoint['ema'], strict=True)
    if args.compile:net.core.compile(dynamic=False)
    rng=torch.Generator(device='cuda').manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    with torch.inference_mode():
        for start in range(0,args.num_images,args.batch_size):
            b=min(args.batch_size,args.num_images-start)
            noise=torch.randn(b,3,32,32,device='cuda',generator=rng)
            images=sample(net,noise,args.flow_steps,args.reset_state).clamp(-1,1).add(1).div(2)
            for j,image in enumerate(images):save_image(image,images_dir/f'{start+j:06d}.png')
    torch.cuda.synchronize()
    elapsed=time.monotonic()-started
    report=dict(vars(args),checkpoint_step=checkpoint['step'],parameters=sum(p.numel() for p in net.parameters()),
                R=3,m=6,seconds_including_png_io_and_compile=elapsed,
                seconds_per_1000_including_png_io_and_compile=elapsed/args.num_images*1000,
                peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,
                postprocessing='clamp [-1,1], (x+1)/2, nearest uint8 via torchvision.save_image',
                reference='official train50000; model trained on fixed train45000')
    if args.fid:
        import importlib.metadata
        import torch_fidelity
        real=ROOT/'data/cifar10/fid_reference_train50000'
        real.mkdir(exist_ok=True)
        ds=CIFAR10(str(ROOT/'data/cifar10'),train=True,download=False)
        if len(list(real.glob('*.png'))) != 50000:
            for i,image in enumerate(ds.data):
                # Exact uint8 source: no resizing or augmentation.
                from PIL import Image
                Image.fromarray(image).save(real/f'{i:06d}.png')
        report['torch_fidelity_version']=importlib.metadata.version('torch-fidelity')
        report['fid']=torch_fidelity.calculate_metrics(input1=str(images_dir),input2=str(real),cuda=True,
                                                       fid=True,isc=False,kid=False,verbose=True,
                                                       input2_cache_name='cifar10_official_train50000_uint8_v1',
                                                       cache_root=str(ROOT/'data/cifar10/fid_cache'))
    (output/'metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
