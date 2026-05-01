#!/bin/python3.8

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import optuna


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Tune Pins full-class ZS-MGM hyperparameters with Optuna.'
    )
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weight_path', type=str, required=True, help='path to model weights')
    parser.add_argument('-dataset', type=str, required=True, nargs='?',
                        choices=['PinsFaceRecognition'],
                        help='dataset to tune on')
    parser.add_argument('-classes', type=int, required=True, help='number of classes')
    parser.add_argument('-forget_class', type=str, required=True, nargs='?',
                        help='class to forget')
    parser.add_argument('-device', type=str, choices=['cpu', 'cuda', 'mps'], default=None,
                        help='device override passed to the runner')
    parser.add_argument('-gpu', action='store_true', default=False, help='pass -gpu through to the runner')
    parser.add_argument('-data_root', type=str, default=None, help='dataset root override')
    parser.add_argument('-b', type=int, default=128, help='batch size for the runner')
    parser.add_argument('-seed', type=int, default=0, help='seed passed to the runner')
    parser.add_argument('-n_trials', type=int, default=20, help='number of Optuna trials to run')
    parser.add_argument('-timeout', type=int, default=None, help='optional Optuna timeout in seconds')
    parser.add_argument('-sampler_seed', type=int, default=0, help='seed for the Optuna sampler')
    parser.add_argument('-study_name', type=str, default=None, help='optional Optuna study name')
    parser.add_argument('-study_dir', type=str, default=None,
                        help='directory to store trial outputs and best_config.json')
    parser.add_argument('-storage', type=str, default=None,
                        help='optional Optuna storage URL, e.g. sqlite:///results/tuning_zsmgm/study.db')
    parser.add_argument('-trial_wandb_mode', type=str,
                        choices=['disabled', 'offline', 'online'], default='disabled',
                        help='WANDB_MODE used for each trial run')
    parser.add_argument('-retain_weight', type=float, default=1.0,
                        help='weight on retain accuracy loss in the tuning objective')
    parser.add_argument('-test_weight', type=float, default=1.0,
                        help='weight on test accuracy loss in the tuning objective')
    parser.add_argument('-mia_weight', type=float, default=1.0,
                        help='weight on the MIA term in the tuning objective')
    parser.add_argument('-df_weight', type=float, default=1.0,
                        help='weight on the forget-set accuracy term in the tuning objective')
    parser.add_argument('-mia_target', type=float, default=None,
                        help='optional MIA target; when set, tune absolute gap to this value instead of minimizing raw MIA')
    parser.add_argument('-df_target', type=float, default=None,
                        help='optional forget-set accuracy target in percentage points; when set, tune absolute gap to this value instead of minimizing raw forget accuracy')
    parser.add_argument('-retain_floor', type=float, default=None,
                        help='optional minimum retain accuracy; violations add a penalty')
    parser.add_argument('-test_floor', type=float, default=None,
                        help='optional minimum test accuracy; violations add a penalty')
    parser.add_argument('-constraint_penalty', type=float, default=5.0,
                        help='penalty multiplier for retain/test floor violations')
    return parser.parse_args(argv)


def default_study_name(args):
    return (
        f'zsmgm_full_class_{args.dataset}_{args.net}_'
        f'forget-{args.forget_class}_seed-{args.seed}'
    )


def resolve_repo_root():
    return Path(__file__).resolve().parents[1]


def resolve_study_dir(args, repo_root):
    if args.study_dir is not None:
        return Path(args.study_dir)
    return repo_root / 'results' / 'tuning_zsmgm' / default_study_name(args)


def trial_result_filename(args):
    return (
        f'full_class_{args.dataset}_{args.net}_zsmgm_'
        f'forget-{args.forget_class}_seed-{args.seed}.json'
    )


def format_float(value):
    return f'{value:.12g}'


def suggest_zsmgm_params(trial):
    return {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'epsilon': trial.suggest_float('epsilon', 1e-3, 1e-1, log=True),
        'lambda_manifold': trial.suggest_float('lambda_manifold', 1e-2, 1e1, log=True),
        'k_neighbors': trial.suggest_categorical('k_neighbors', [1, 3, 5, 10]),
        'pgd_steps': trial.suggest_categorical('pgd_steps', [3, 5, 10, 20]),
        'pgd_alpha': 1.0 / 255.0,
    }


def build_objective_score(results, args):
    raw_mia = max(0.0, float(results['MIA']))
    raw_df = max(0.0, float(results['df']))
    components = {
        'retain_gap': max(0.0, 1.0 - (float(results['RetainTestAcc']) / 100.0)),
        'test_gap': max(0.0, 1.0 - (float(results['TestAcc']) / 100.0)),
        'mia': raw_mia,
        'df': raw_df,
        'mia_gap': raw_mia if args.mia_target is None else abs(raw_mia - float(args.mia_target)),
        'df_gap': (
            max(0.0, raw_df / 100.0)
            if args.df_target is None
            else abs(raw_df - float(args.df_target)) / 100.0
        ),
    }
    penalties = {
        'retain_floor_penalty': 0.0,
        'test_floor_penalty': 0.0,
    }

    if args.retain_floor is not None:
        penalties['retain_floor_penalty'] = (
            max(0.0, (float(args.retain_floor) - float(results['RetainTestAcc'])) / 100.0)
            * args.constraint_penalty
        )
    if args.test_floor is not None:
        penalties['test_floor_penalty'] = (
            max(0.0, (float(args.test_floor) - float(results['TestAcc'])) / 100.0)
            * args.constraint_penalty
        )

    score = (
        args.retain_weight * components['retain_gap']
        + args.test_weight * components['test_gap']
        + args.mia_weight * components['mia_gap']
        + args.df_weight * components['df_gap']
        + penalties['retain_floor_penalty']
        + penalties['test_floor_penalty']
    )
    return score, components, penalties


def build_trial_command(args, params, trial_results_dir, repo_root):
    runner_path = repo_root / 'src' / 'forget_full_class_main.py'
    command = [
        sys.executable,
        str(runner_path),
        '-net', args.net,
        '-weight_path', args.weight_path,
        '-dataset', args.dataset,
        '-classes', str(args.classes),
        '-method', 'zsmgm',
        '-forget_class', str(args.forget_class),
        '-results_dir', str(trial_results_dir),
        '-b', str(args.b),
        '-seed', str(args.seed),
        '-zsmgm_learning_rate', format_float(params['learning_rate']),
        '-zsmgm_epsilon', format_float(params['epsilon']),
        '-zsmgm_lambda_manifold', format_float(params['lambda_manifold']),
        '-zsmgm_k_neighbors', str(int(params['k_neighbors'])),
        '-zsmgm_pgd_steps', str(int(params['pgd_steps'])),
        '-zsmgm_pgd_alpha', format_float(params['pgd_alpha']),
    ]
    if args.gpu:
        command.append('-gpu')
    if args.device is not None:
        command.extend(['-device', args.device])
    if args.data_root is not None:
        command.extend(['-data_root', args.data_root])
    return command


def run_trial_command(command, trial_dir, repo_root, args):
    env = os.environ.copy()
    env['WANDB_MODE'] = args.trial_wandb_mode
    if args.device == 'mps':
        env.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - started

    stdout_path = trial_dir / 'runner_stdout.txt'
    stderr_path = trial_dir / 'runner_stderr.txt'
    stdout_path.write_text(completed.stdout, encoding='utf-8')
    stderr_path.write_text(completed.stderr, encoding='utf-8')

    return completed, elapsed, stdout_path, stderr_path


def _json_safe_number(value):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def serialize_trial(trial):
    return {
        'number': trial.number,
        'state': str(trial.state),
        'value': _json_safe_number(trial.value),
        'params': dict(trial.params),
        'user_attrs': dict(trial.user_attrs),
    }


def main(argv=None):
    args = parse_args(argv)
    repo_root = resolve_repo_root()
    study_name = args.study_name or default_study_name(args)
    study_dir = resolve_study_dir(args, repo_root)
    trials_dir = study_dir / 'trials'
    trials_dir.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    study = optuna.create_study(
        direction='minimize',
        study_name=study_name,
        sampler=sampler,
        storage=args.storage,
        load_if_exists=bool(args.storage),
    )

    def objective(trial):
        params = suggest_zsmgm_params(trial)
        trial_dir = trials_dir / f'trial_{trial.number:04d}'
        trial_dir.mkdir(parents=True, exist_ok=True)
        command = build_trial_command(args, params, trial_dir, repo_root)

        print(
            f'trial={trial.number} start '
            f'lr={params["learning_rate"]:.6g} '
            f'eps={params["epsilon"]:.6g} '
            f'lambda={params["lambda_manifold"]:.6g} '
            f'k={params["k_neighbors"]} '
            f'pgd_steps={params["pgd_steps"]}'
        )

        completed, elapsed, stdout_path, stderr_path = run_trial_command(
            command, trial_dir, repo_root, args
        )
        result_path = trial_dir / trial_result_filename(args)

        trial.set_user_attr('trial_dir', str(trial_dir))
        trial.set_user_attr('command', command)
        trial.set_user_attr('runner_returncode', completed.returncode)
        trial.set_user_attr('stdout_path', str(stdout_path))
        trial.set_user_attr('stderr_path', str(stderr_path))
        trial.set_user_attr('elapsed_seconds', elapsed)

        if completed.returncode != 0:
            trial.set_user_attr('status', 'failed')
            print(f'trial={trial.number} failed rc={completed.returncode} elapsed={elapsed:.2f}s')
            return float('inf')

        if not result_path.is_file():
            trial.set_user_attr('status', 'missing_results')
            print(f'trial={trial.number} missing results file {result_path}')
            return float('inf')

        with open(result_path, 'r', encoding='utf-8') as result_file:
            results = json.load(result_file)

        score, components, penalties = build_objective_score(results, args)
        trial.set_user_attr('status', 'completed')
        trial.set_user_attr('results', results)
        trial.set_user_attr('score_components', components)
        trial.set_user_attr('score_penalties', penalties)

        print(
            f'trial={trial.number} complete score={score:.6f} '
            f'retain={results["RetainTestAcc"]:.4f} '
            f'test={results["TestAcc"]:.4f} '
            f'mia={results["MIA"]:.5f} '
            f'df={results["df"]:.4f} '
            f'elapsed={elapsed:.2f}s'
        )
        return score

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    successful_trials = [
        trial for trial in study.trials
        if trial.value is not None and math.isfinite(float(trial.value))
    ]
    if not successful_trials:
        raise RuntimeError('No successful tuning trials completed.')

    best_trial = min(successful_trials, key=lambda trial: float(trial.value))
    best_config = dict(best_trial.params)
    best_config['pgd_alpha'] = 1.0 / 255.0

    best_config_path = study_dir / 'best_config.json'
    with open(best_config_path, 'w', encoding='utf-8') as config_file:
        json.dump(best_config, config_file, indent=2)

    trial_records_path = study_dir / 'trial_records.json'
    with open(trial_records_path, 'w', encoding='utf-8') as trial_file:
        json.dump([serialize_trial(trial) for trial in study.trials], trial_file, indent=2)

    summary = {
        'study_name': study.study_name,
        'best_trial_number': best_trial.number,
        'best_value': float(best_trial.value),
        'best_params': best_config,
        'best_results': best_trial.user_attrs.get('results'),
        'weights': {
            'retain_weight': args.retain_weight,
            'test_weight': args.test_weight,
            'mia_weight': args.mia_weight,
            'df_weight': args.df_weight,
        },
        'targets': {
            'mia_target': args.mia_target,
            'df_target': args.df_target,
        },
        'floors': {
            'retain_floor': args.retain_floor,
            'test_floor': args.test_floor,
            'constraint_penalty': args.constraint_penalty,
        },
        'trial_wandb_mode': args.trial_wandb_mode,
        'n_trials': len(study.trials),
        'study_dir': str(study_dir),
        'best_config_path': str(best_config_path),
    }
    summary_path = study_dir / 'study_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as summary_file:
        json.dump(summary, summary_file, indent=2)

    print(json.dumps(summary, indent=2))
    print(f'saved best config to {best_config_path}')
    print(f'saved trial records to {trial_records_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())