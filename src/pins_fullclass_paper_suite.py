#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


METHOD_ALIASES = {
    'lipschitz_forgetting': 'paper_faithful_jit',
    'baseline': 'paper_faithful_bsln',
    'retrain': 'repo_fallback_rtrn',
    'zsmgm': 'tuned_zsmgm',
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Run the Pins full-class suite for JiT, BSLN, RTRN, and tuned ZS-MGM.'
    )
    parser.add_argument('-weight_path', type=str, required=True,
                        help='path to the pretrained checkpoint used by all methods')
    parser.add_argument('-results_dir', type=str, default=None,
                        help='directory for per-run JSON outputs and suite summary')
    parser.add_argument('-data_root', type=str, default=None,
                        help='optional dataset root override passed through to the runner')
    parser.add_argument('-device', type=str, choices=['cpu', 'cuda', 'mps'], default=None,
                        help='device override passed through to the runner')
    parser.add_argument('-gpu', action='store_true', default=False,
                        help='pass -gpu to the runner')
    parser.add_argument('-cuda_visible_devices', type=str, default=None,
                        help='optional CUDA_VISIBLE_DEVICES value for each run')
    parser.add_argument('-wandb_mode', type=str,
                        choices=['disabled', 'offline', 'online'], default='disabled',
                        help='WANDB_MODE to use for all suite runs')
    parser.add_argument('-dataset', type=str, default='PinsFaceRecognition',
                        choices=['PinsFaceRecognition'], help='dataset to run')
    parser.add_argument('-net', type=str, default='VGG16',
                        choices=['VGG16'], help='network to run')
    parser.add_argument('-classes', type=int, default=105,
                        help='number of classes for the Pins dataset')
    parser.add_argument('-forget_class', type=str, action='append', default=None,
                        help='forget class to run; defaults to class 1. Can be passed multiple times.')
    parser.add_argument('-start_seed', type=int, default=0,
                        help='first seed index used for multi-seed methods')
    parser.add_argument('-jit_seeds', type=int, default=10,
                        help='number of seeds for paper-faithful JiT')
    parser.add_argument('-bsln_seeds', type=int, default=1,
                        help='number of seeds for the paper-faithful baseline')
    parser.add_argument('-rtrn_seeds', type=int, default=10,
                        help='number of seeds for repo-fallback retrain')
    parser.add_argument('-zsmgm_seeds', type=int, default=10,
                        help='number of seeds for tuned ZS-MGM')
    parser.add_argument('-batch_size', type=int, default=128,
                        help='batch size passed through to the runner')
    parser.add_argument('-zsmgm_config_path', type=str, default=None,
                        help='path to the tuned ZS-MGM config JSON; defaults to the canonical tuned config')
    parser.add_argument('-rerun_existing', action='store_true', default=False,
                        help='rerun methods even when the expected result JSON already exists')
    parser.add_argument('-dry_run', action='store_true', default=False,
                        help='print the planned commands without executing them')
    return parser.parse_args(argv)


def resolve_repo_root():
    return Path(__file__).resolve().parents[1]


def make_repo_local_path(path, repo_root):
    path = Path(path)
    if path.is_absolute():
        return path
    return repo_root / path


def make_display_path(path, repo_root):
    path = Path(path)
    if not path.is_absolute():
        return path
    try:
        return path.relative_to(repo_root)
    except ValueError:
        return path


def relative_to_repo(path, repo_root):
    return str(make_display_path(path, repo_root))


def resolve_results_dir(args, repo_root):
    if args.results_dir is not None:
        return Path(args.results_dir)
    return Path('results') / 'pins_fullclass_paper_suite'


def resolve_forget_classes(args):
    return args.forget_class if args.forget_class else ['1']


def resolve_data_root(args, repo_root):
    if args.data_root is not None:
        return Path(args.data_root)

    candidates = [
        Path('105_classes_pins_dataset'),
        Path('../datasets/105_classes_pins_dataset'),
        Path('../Unlearning/datasets/105_classes_pins_dataset'),
    ]
    for candidate in candidates:
        if make_repo_local_path(candidate, repo_root).exists():
            return candidate
    return None


def resolve_zsmgm_config(args, repo_root):
    if args.zsmgm_config_path is not None:
        config_path = Path(args.zsmgm_config_path)
    else:
        config_path = (
            Path('results')
            / 'tuning_zsmgm'
            / 'zsmgm_full_class_PinsFaceRecognition_VGG16_forget-1_seed-0'
            / 'best_config.json'
        )
    if not make_repo_local_path(config_path, repo_root).exists():
        raise FileNotFoundError(
            'No tuned ZS-MGM config was found. Pass -zsmgm_config_path or run tune_full_class_zsmgm.py first.'
        )
    return config_path


def iter_plan(args, repo_root, results_dir, zsmgm_config_path, data_root):
    runner = Path('src') / 'forget_full_class_main.py'
    base_command = [
        sys.executable,
        str(runner),
        '-net', args.net,
        '-weight_path', args.weight_path,
        '-dataset', args.dataset,
        '-classes', str(args.classes),
        '-results_dir', str(results_dir),
        '-b', str(args.batch_size),
    ]
    if args.gpu:
        base_command.append('-gpu')
    if args.device is not None:
        base_command.extend(['-device', args.device])
    if data_root is not None:
        base_command.extend(['-data_root', str(data_root)])

    method_specs = [
        ('lipschitz_forgetting', args.jit_seeds, []),
        ('baseline', args.bsln_seeds, []),
        ('retrain', args.rtrn_seeds, []),
        ('zsmgm', args.zsmgm_seeds, ['-zsmgm_config_path', str(zsmgm_config_path)]),
    ]

    for forget_class in resolve_forget_classes(args):
        for method_name, seed_count, extra_args in method_specs:
            for seed in range(args.start_seed, args.start_seed + seed_count):
                result_path = (
                    results_dir
                    / f'full_class_{args.dataset}_{args.net}_{method_name}_forget-{forget_class}_seed-{seed}.json'
                )
                result_fs_path = make_repo_local_path(result_path, repo_root)
                command = list(base_command)
                command.extend([
                    '-method', method_name,
                    '-forget_class', str(forget_class),
                    '-seed', str(seed),
                ])
                command.extend(extra_args)
                yield {
                    'method': method_name,
                    'suite_label': METHOD_ALIASES[method_name],
                    'forget_class': str(forget_class),
                    'seed': int(seed),
                    'command': command,
                    'result_path': result_path,
                    'result_fs_path': result_fs_path,
                }


def build_env(args):
    env = os.environ.copy()
    env['WANDB_MODE'] = args.wandb_mode
    if args.cuda_visible_devices is not None:
        env['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    if args.device == 'mps':
        env.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    return env


def execute_plan(plan_entry, repo_root, env):
    print(' '.join(plan_entry['command']))
    completed = subprocess.run(
        plan_entry['command'],
        cwd=repo_root,
        env=env,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Run failed for {plan_entry['suite_label']} forget={plan_entry['forget_class']} seed={plan_entry['seed']}"
        )


def write_manifest(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)


def main(argv=None):
    args = parse_args(argv)
    repo_root = resolve_repo_root()
    results_dir = resolve_results_dir(args, repo_root)
    results_dir_fs = make_repo_local_path(results_dir, repo_root)
    results_dir_fs.mkdir(parents=True, exist_ok=True)
    zsmgm_config_path = resolve_zsmgm_config(args, repo_root)
    data_root = resolve_data_root(args, repo_root)
    env = build_env(args)

    plan = list(iter_plan(args, repo_root, results_dir, zsmgm_config_path, data_root))
    manifest_entries = []
    executed = 0
    skipped = 0
    for entry in plan:
        status = 'pending'
        if entry['result_fs_path'].exists() and not args.rerun_existing:
            status = 'skipped_existing'
            skipped += 1
        elif args.dry_run:
            status = 'dry_run'
            print(' '.join(entry['command']))
        else:
            execute_plan(entry, repo_root, env)
            status = 'executed'
            executed += 1
        manifest_entries.append({
            'suite_label': entry['suite_label'],
            'method': entry['method'],
            'forget_class': entry['forget_class'],
            'seed': entry['seed'],
            'result_path': relative_to_repo(entry['result_path'], repo_root),
            'status': status,
            'command': entry['command'],
        })

    manifest = {
        'repo_root': str(repo_root),
        'dataset': args.dataset,
        'net': args.net,
        'classes': args.classes,
        'forget_classes': resolve_forget_classes(args),
        'weight_path': args.weight_path,
        'data_root': None if data_root is None else relative_to_repo(data_root, repo_root),
        'results_dir': relative_to_repo(results_dir, repo_root),
        'wandb_mode': args.wandb_mode,
        'device': args.device,
        'gpu': args.gpu,
        'cuda_visible_devices': args.cuda_visible_devices,
        'zsmgm_config_path': relative_to_repo(zsmgm_config_path, repo_root),
        'executed_runs': executed,
        'skipped_runs': skipped,
        'dry_run': args.dry_run,
        'entries': manifest_entries,
    }
    write_manifest(results_dir_fs / 'paper_suite_manifest.json', manifest)

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    from summarize_full_class_results import create_summary

    summary = create_summary(
        results_dir=results_dir_fs,
        methods=list(METHOD_ALIASES),
        method_labels=METHOD_ALIASES,
        dataset=args.dataset,
        net=args.net,
        forget_classes=resolve_forget_classes(args),
        min_samples=2,
    )
    summary_path = results_dir_fs / 'paper_suite_summary.json'
    write_manifest(summary_path, summary)
    print(json.dumps(summary, indent=2))
    print(f'saved manifest to {results_dir_fs / "paper_suite_manifest.json"}')
    print(f'saved summary to {summary_path}')


if __name__ == '__main__':
    main()