#!/usr/bin/env python3

import argparse
import itertools
import json
import math
from pathlib import Path

try:
    import numpy as np
    from scipy import stats
except ModuleNotFoundError as exc:
    raise SystemExit(
        'summarize_full_class_results.py requires numpy and scipy. '
        'Install the repo requirements before running this summary.'
    ) from exc


METRICS = ('TestAcc', 'RetainTestAcc', 'MIA', 'df')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Summarize full-class result JSON files and compute Student t-tests.'
    )
    parser.add_argument('-results_dir', type=str, required=True,
                        help='directory containing full_class_*.json files')
    parser.add_argument('-dataset', type=str, default=None,
                        help='optional dataset filter')
    parser.add_argument('-net', type=str, default=None,
                        help='optional model filter')
    parser.add_argument('-forget_class', type=str, action='append', default=None,
                        help='optional forget-class filter; can be passed multiple times')
    parser.add_argument('-method', type=str, action='append', required=True,
                        help='method to include; can be passed multiple times')
    parser.add_argument('-method_label', type=str, action='append', default=[],
                        help='optional method label mapping in the form actual=label')
    parser.add_argument('-output_path', type=str, default=None,
                        help='optional output path; defaults to <results_dir>/full_class_summary.json')
    parser.add_argument('-min_samples', type=int, default=2,
                        help='minimum number of matched samples needed for a t-test')
    return parser.parse_args(argv)


def parse_method_labels(raw_items):
    labels = {}
    for raw_item in raw_items:
        if '=' not in raw_item:
            raise ValueError(
                f'method_label must use actual=label format, received {raw_item!r}.'
            )
        actual, label = raw_item.split('=', 1)
        actual = actual.strip()
        label = label.strip()
        if not actual or not label:
            raise ValueError(
                f'method_label must use actual=label format, received {raw_item!r}.'
            )
        labels[actual] = label
    return labels


def default_output_path(results_dir):
    return Path(results_dir) / 'full_class_summary.json'


def iter_result_paths(results_dir):
    return sorted(Path(results_dir).glob('full_class_*.json'))


def sample_key(record):
    return (str(record['forget_class']), int(record['seed']))


def class_key(record):
    return str(record['forget_class'])


def load_records(results_dir, dataset=None, net=None, forget_classes=None, methods=None):
    selected_methods = set(methods)
    selected_forget_classes = None if not forget_classes else {str(item) for item in forget_classes}
    records = []
    for path in iter_result_paths(results_dir):
        with path.open('r', encoding='utf-8') as handle:
            record = json.load(handle)
        if record.get('method') not in selected_methods:
            continue
        if dataset is not None and record.get('dataset') != dataset:
            continue
        if net is not None and record.get('net') != net:
            continue
        if selected_forget_classes is not None and str(record.get('forget_class')) not in selected_forget_classes:
            continue
        record['path'] = str(path)
        records.append(record)
    return records


def build_method_maps(records):
    by_method = {}
    by_method_and_key = {}
    by_method_and_class = {}

    for record in records:
        method = record['method']
        by_method.setdefault(method, []).append(record)

        method_by_key = by_method_and_key.setdefault(method, {})
        key = sample_key(record)
        if key in method_by_key:
            raise ValueError(
                f'Duplicate results for method={method!r}, forget_class={key[0]!r}, seed={key[1]!r}.'
            )
        method_by_key[key] = record

        method_by_class = by_method_and_class.setdefault(method, {})
        method_by_class.setdefault(class_key(record), []).append(record)

    return by_method, by_method_and_key, by_method_and_class


def summarize_metric(values):
    array = np.asarray(values, dtype=float)
    summary = {
        'count': int(array.size),
        'mean': float(np.mean(array)),
        'min': float(np.min(array)),
        'max': float(np.max(array)),
    }
    if array.size >= 2:
        summary['std'] = float(np.std(array, ddof=1))
        summary['stderr'] = float(summary['std'] / math.sqrt(array.size))
    else:
        summary['std'] = None
        summary['stderr'] = None
    return summary


def summarize_method(records):
    metric_summary = {}
    for metric in METRICS:
        metric_summary[metric] = summarize_metric([record[metric] for record in records])
    return {
        'count': len(records),
        'seeds': sorted({int(record['seed']) for record in records}),
        'forget_classes': sorted({str(record['forget_class']) for record in records}),
        'metrics': metric_summary,
        'files': sorted(record['path'] for record in records),
    }


def is_fixed_reference(records_by_class):
    if not records_by_class:
        return False
    return all(len(class_records) == 1 for class_records in records_by_class.values())


def _constant_difference_result(test_type, n, mean_a, mean_b, mean_difference, note):
    return {
        'status': 'constant_difference',
        'test_type': test_type,
        'n': int(n),
        'mean_a': float(mean_a),
        'mean_b': float(mean_b),
        'mean_difference': float(mean_difference),
        't_statistic': None,
        'p_value': None,
        'note': note,
    }


def _insufficient_result(test_type, n, note):
    return {
        'status': 'insufficient_samples',
        'test_type': test_type,
        'n': int(n),
        't_statistic': None,
        'p_value': None,
        'note': note,
    }


def paired_ttest(metric, method_a, method_b, records_a_by_key, records_b_by_key, min_samples):
    common_keys = sorted(set(records_a_by_key).intersection(records_b_by_key))
    if len(common_keys) < min_samples:
        return _insufficient_result(
            test_type='paired',
            n=len(common_keys),
            note=f'Need at least {min_samples} matched (forget_class, seed) pairs for a paired t-test.',
        )

    values_a = np.asarray([float(records_a_by_key[key][metric]) for key in common_keys], dtype=float)
    values_b = np.asarray([float(records_b_by_key[key][metric]) for key in common_keys], dtype=float)
    differences = values_a - values_b
    if np.allclose(differences, differences[0]):
        return _constant_difference_result(
            test_type='paired',
            n=len(common_keys),
            mean_a=np.mean(values_a),
            mean_b=np.mean(values_b),
            mean_difference=np.mean(differences),
            note='All paired differences are identical; the t-statistic is undefined.',
        )

    test_result = stats.ttest_rel(values_a, values_b)
    return {
        'status': 'ok',
        'test_type': 'paired',
        'n': len(common_keys),
        'mean_a': float(np.mean(values_a)),
        'mean_b': float(np.mean(values_b)),
        'mean_difference': float(np.mean(differences)),
        't_statistic': float(test_result.statistic),
        'p_value': float(test_result.pvalue),
        'matched_pairs': len(common_keys),
        'method_a': method_a,
        'method_b': method_b,
    }


def one_sample_reference_ttest(metric, method_a, method_b, reference_records_by_class,
                               sampled_records, mean_difference_sign, min_samples):
    deltas = []
    sample_values = []
    reference_values = []
    matched_count = 0

    for record in sampled_records:
        reference_record = reference_records_by_class.get(class_key(record))
        if reference_record is None:
            continue
        sample_value = float(record[metric])
        reference_value = float(reference_record[metric])
        deltas.append((sample_value - reference_value) * mean_difference_sign)
        sample_values.append(sample_value)
        reference_values.append(reference_value)
        matched_count += 1

    if matched_count < min_samples:
        return _insufficient_result(
            test_type='one_sample_vs_reference',
            n=matched_count,
            note=f'Need at least {min_samples} samples against a fixed reference for a one-sample t-test.',
        )

    delta_array = np.asarray(deltas, dtype=float)
    if np.allclose(delta_array, delta_array[0]):
        return _constant_difference_result(
            test_type='one_sample_vs_reference',
            n=matched_count,
            mean_a=np.mean(sample_values if mean_difference_sign == 1 else reference_values),
            mean_b=np.mean(reference_values if mean_difference_sign == 1 else sample_values),
            mean_difference=np.mean(delta_array),
            note='All sample-minus-reference differences are identical; the t-statistic is undefined.',
        )

    test_result = stats.ttest_1samp(delta_array, popmean=0.0)
    mean_a = np.mean(sample_values if mean_difference_sign == 1 else reference_values)
    mean_b = np.mean(reference_values if mean_difference_sign == 1 else sample_values)
    return {
        'status': 'ok',
        'test_type': 'one_sample_vs_reference',
        'n': matched_count,
        'mean_a': float(mean_a),
        'mean_b': float(mean_b),
        'mean_difference': float(np.mean(delta_array)),
        't_statistic': float(test_result.statistic),
        'p_value': float(test_result.pvalue),
        'method_a': method_a,
        'method_b': method_b,
    }


def compare_metric(metric, method_a, method_b, method_records, method_records_by_key,
                   method_records_by_class, min_samples):
    paired_result = paired_ttest(
        metric=metric,
        method_a=method_a,
        method_b=method_b,
        records_a_by_key=method_records_by_key[method_a],
        records_b_by_key=method_records_by_key[method_b],
        min_samples=min_samples,
    )
    if paired_result['status'] == 'ok' or paired_result['status'] == 'constant_difference':
        return paired_result

    method_a_by_class = method_records_by_class[method_a]
    method_b_by_class = method_records_by_class[method_b]
    method_a_is_reference = is_fixed_reference(method_a_by_class)
    method_b_is_reference = is_fixed_reference(method_b_by_class)

    if method_a_is_reference and len(method_records[method_b]) >= min_samples:
        reference_records_by_class = {
            current_class: class_records[0]
            for current_class, class_records in method_a_by_class.items()
        }
        return one_sample_reference_ttest(
            metric=metric,
            method_a=method_a,
            method_b=method_b,
            reference_records_by_class=reference_records_by_class,
            sampled_records=method_records[method_b],
            mean_difference_sign=-1,
            min_samples=min_samples,
        )

    if method_b_is_reference and len(method_records[method_a]) >= min_samples:
        reference_records_by_class = {
            current_class: class_records[0]
            for current_class, class_records in method_b_by_class.items()
        }
        return one_sample_reference_ttest(
            metric=metric,
            method_a=method_a,
            method_b=method_b,
            reference_records_by_class=reference_records_by_class,
            sampled_records=method_records[method_a],
            mean_difference_sign=1,
            min_samples=min_samples,
        )

    return paired_result


def create_summary(results_dir, methods, method_labels=None, dataset=None, net=None,
                   forget_classes=None, min_samples=2):
    method_labels = method_labels or {}
    records = load_records(
        results_dir=results_dir,
        dataset=dataset,
        net=net,
        forget_classes=forget_classes,
        methods=methods,
    )
    by_method, by_method_key, by_method_class = build_method_maps(records)

    missing_methods = [method for method in methods if method not in by_method]

    method_summaries = {}
    for method in methods:
        records_for_method = by_method.get(method, [])
        method_summaries[method] = {
            'label': method_labels.get(method, method),
            'summary': summarize_method(records_for_method) if records_for_method else None,
        }

    comparisons = []
    present_methods = [method for method in methods if method in by_method]
    for method_a, method_b in itertools.combinations(present_methods, 2):
        metric_results = {}
        for metric in METRICS:
            metric_results[metric] = compare_metric(
                metric=metric,
                method_a=method_a,
                method_b=method_b,
                method_records=by_method,
                method_records_by_key=by_method_key,
                method_records_by_class=by_method_class,
                min_samples=min_samples,
            )
        comparisons.append({
            'method_a': method_a,
            'label_a': method_labels.get(method_a, method_a),
            'method_b': method_b,
            'label_b': method_labels.get(method_b, method_b),
            'metrics': metric_results,
        })

    return {
        'results_dir': str(Path(results_dir)),
        'filters': {
            'dataset': dataset,
            'net': net,
            'forget_classes': None if not forget_classes else [str(item) for item in forget_classes],
            'methods': list(methods),
            'min_samples': int(min_samples),
        },
        'method_labels': method_labels,
        'files_considered': len(records),
        'missing_methods': missing_methods,
        'methods': method_summaries,
        'pairwise_tests': comparisons,
    }


def main(argv=None):
    args = parse_args(argv)
    labels = parse_method_labels(args.method_label)
    summary = create_summary(
        results_dir=args.results_dir,
        methods=args.method,
        method_labels=labels,
        dataset=args.dataset,
        net=args.net,
        forget_classes=args.forget_class,
        min_samples=args.min_samples,
    )
    output_path = Path(args.output_path) if args.output_path is not None else default_output_path(args.results_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f'saved summary to {output_path}')


if __name__ == '__main__':
    main()