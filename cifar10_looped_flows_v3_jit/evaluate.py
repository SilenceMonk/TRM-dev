"""Model/EMA PNG export and optional fixed torch-fidelity Inception FID against official train50k."""
import argparse
import json
import hashlib
from pathlib import Path
import time
import torch
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from model import TRMJiT, sample
from jit_head import OUTPUT_HEAD_DESCRIPTION
from data import ROOT


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--weights',choices=('model','ema'),default='ema')
    p.add_argument('--num-images',type=int,default=5000)
    p.add_argument('--flow-steps',type=int,default=32,choices=[8,16,32,64])
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--seed',type=int,default=12345)
    p.add_argument('--fid',action='store_true')
    p.add_argument('--compile',action='store_true')
    args=p.parse_args()
    if args.num_images < 2 or args.batch_size < 1:
        p.error('num-images must be >=2 and batch-size must be positive')
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = True
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    images_dir=output/'generated'
    images_dir.mkdir(exist_ok=True)
    if list(images_dir.glob('*.png')):
        raise FileExistsError('Use a fresh output directory to avoid mixing samples')
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if checkpoint['config']['output_head'] != OUTPUT_HEAD_DESCRIPTION:
        raise ValueError('Checkpoint output head does not match gamma-only head; use its original code revision')
    net=TRMJiT(Nsup=checkpoint['config']['Nsup']).cuda().eval()
    net.load_state_dict(checkpoint[args.weights], strict=True)
    if args.compile:net.core.compile(dynamic=False)
    rng=torch.Generator(device='cuda').manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    def progress(phase, generated):
        value=dict(phase=phase, generated=generated, total=args.num_images, weights=args.weights,
                   checkpoint_step=checkpoint['step'], seconds=time.monotonic()-started)
        temp=output/'progress.tmp'
        temp.write_text(json.dumps(value,indent=2)); temp.replace(output/'progress.json')
        print(json.dumps(value),flush=True)
    progress('generating',0)
    with torch.inference_mode():
        for start in range(0,args.num_images,args.batch_size):
            b=min(args.batch_size,args.num_images-start)
            noise=torch.randn(b,3,32,32,device='cuda',generator=rng)
            images=sample(net,noise,args.flow_steps,t_eps=checkpoint['config']['t_eps']).clamp(-1,1).add(1).div(2)
            for j,image in enumerate(images):save_image(image,images_dir/f'{start+j:06d}.png')
            progress('generating',start+b)
    torch.cuda.synchronize()
    elapsed=time.monotonic()-started
    report=dict(vars(args),checkpoint_step=checkpoint['step'],parameters=sum(p.numel() for p in net.parameters()),
                checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                Nsup=net.Nsup,H_cycles=3,L_cycles=6,t_eps=checkpoint['config']['t_eps'],seconds_including_png_io_and_compile=elapsed,
                seconds_per_1000_including_png_io_and_compile=elapsed/args.num_images*1000,
                peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,
                postprocessing='clamp [-1,1], (x+1)/2, nearest uint8 via torchvision.save_image',
                reference='official train50000; model trained on fixed train45000')
    (output/'generation_metrics.json').write_text(json.dumps(report,indent=2))
    if args.fid:
        progress('preparing_fid',args.num_images)
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
        progress('calculating_fid',args.num_images)
        report['fid_protocol']=f'FID-{args.num_images} generated vs official train50000; no resizing of PNG inputs'
        report['fid']=torch_fidelity.calculate_metrics(input1=str(images_dir),input2=str(real),cuda=True,
                                                       fid=True,isc=False,kid=False,verbose=True,batch_size=64,
                                                       input2_cache_name='cifar10_official_train50000_uint8_v1',
                                                       cache_root=str(ROOT/'data/cifar10/fid_cache'))
    (output/'metrics.json').write_text(json.dumps(report,indent=2))
    progress('complete',args.num_images)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
