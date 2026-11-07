"""Explicit old-to-refine migration; never overwrites the source experiment."""
import argparse
import copy
import hashlib
from pathlib import Path
import torch
from model import LoopedFlow
from train import atomic_json, atomic_save


def migrate(checkpoint, source):
    result = copy.deepcopy(checkpoint)
    template = LoopedFlow()
    new_names = ['refine.weight', 'refine.bias']
    expected_old = set(template.state_dict()) - set(new_names)
    for key in ('model', 'ema'):
        if set(result[key]) != expected_old:
            raise ValueError(f'{key}: expected exactly the original architecture')
        result[key].update({name: template.state_dict()[name].clone() for name in new_names})
        template.load_state_dict(result[key], strict=True)
    # Existing parameter registration order is unchanged; refine is appended.
    old_names = [name for name, _ in template.named_parameters() if name not in new_names]
    groups = result['optimizer']['param_groups']
    if len(groups) != 1 or len(groups[0]['params']) != len(old_names):
        raise ValueError('Expected original single AdamATan2 parameter group')
    ids = groups[0]['params']
    if ids != list(range(len(old_names))):
        raise ValueError('Unexpected optimizer parameter IDs; refusing positional migration')
    groups[0]['params'] = ids + [len(ids), len(ids)+1]
    # No moments or step are fabricated for new parameters. AdamATan2 lazily
    # initializes them on their first update; every existing state is preserved.
    parent = dict(path=str(Path(source).resolve()), step=result['step'],
                  sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest())
    result['config'].update(parameters=sum(p.numel() for p in template.parameters()),
                            output_head='unpatchify + zero-init residual Conv2d(3,3,3,padding=1)',
                            refine_start_step=result['step'],
                            parent_checkpoint=parent)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    run = Path(args.run).resolve()
    run.mkdir(parents=True, exist_ok=False)
    original = torch.load(args.source, map_location='cpu', weights_only=False)
    checkpoint = migrate(original, args.source)
    checkpoint['config']['run'] = str(run)
    atomic_save(run / 'latest.pt', checkpoint)
    atomic_json(run / 'config.json', checkpoint['config'])
    atomic_json(run / 'status.json', dict(step=checkpoint['step'], complete=False,
                phase='prepared_stopped', added_parameters=84,
                parent_checkpoint=checkpoint['config']['parent_checkpoint']))
    print(f'Prepared {run} at step {checkpoint["step"]}; training was not started.')


if __name__ == '__main__':
    main()
