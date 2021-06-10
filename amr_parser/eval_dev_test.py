# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-09-18 12:44
import argparse
import math
import os
import warnings
from glob import glob

from amr_parser.match import match
from elit.metrics.amr.smatch_eval import smatch_eval, SmatchScores
from elit.metrics.f1 import F1_
from elit.metrics.mtl import MetricDict
from elit.utils.io_util import run_cmd, load_pickle, save_pickle
from elit.utils.log_util import flash, cprint
from elit.utils.time_util import CountdownTimer
from amr_parser.work import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('load_path', type=str, help='The folder containing checkpoints')
    parser.add_argument('--version', type=str, help='AMR data version', default='2.0')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    load_path = args.load_path
    version = args.version
    device = args.device
    history_path = f'{load_path}/eval.pkl'
    try:
        history = load_pickle(history_path)
    except FileNotFoundError:
        history = dict()
    # for k, v in history.items():
    #     print(f'{k} {v}')
    devs = sorted([x for x in glob(f'{load_path}/epoch*_batch*') if x.endswith('9')])
    if not devs:
        flash('[yellow]No checkpoints.[/yellow]\n')
        return
    devs = [x for x in devs if f'{x}_dev_out.pred' not in history]
    timer = CountdownTimer(len(devs))
    for dev in devs:
        pred_path = f'{dev}_dev_out.pred'
        if not os.path.isfile(f'{load_path}/{pred_path}'):
            print(dev)
            predict(dev, f'data/AMR/amr_{version}/dev.txt.features.preproc', device=device, output_suffix='_dev_out')
            match(pred_path, f'{dev}_dev_out')
        scores = post_process_then_run_eval_script(pred_path, version=version)
        timer.log(f'{os.path.basename(dev)} {scores}', newline=True)
        history[pred_path] = scores
        save_pickle(history, history_path)
    # print(sorted([(k, v.score) for (k, v) in history.items() if k.endswith('_dev_out.pred')], key=lambda x: x[1], reverse=True))
    best_dev, best_score = max([(k, v) for (k, v) in history.items() if k.endswith('_dev_out.pred')
                                # and int(k.split('batch')[-1].split('_')[0]) < 50000
                                and not math.isnan(v)
                                ],
                               key=lambda x: (x[1], x[0]))
    print(f'Best epoch {best_dev} {best_score}')
    checkpoint = best_dev[:-len('_dev_out.pred')]
    # steps = int(checkpoint.split('batch')[-1])
    # if steps < 50000:
    #     return
    test_pred = f'{checkpoint}_test_out.pred'
    finegrained = ['Smatch',
                   'Unlabeled',
                   'No WSD',
                   'Concepts',
                   'SRL',
                   'Reentrancies',
                   'Negations',
                   'Named Ent.',
                   'Wikification']
    if test_pred in history and isinstance(history[test_pred], MetricDict) and not any(
            math.isnan(history[test_pred][x]) for x in finegrained):
        test_score = history[test_pred]
    else:
        flash(f'Running prediction {checkpoint} on testset [blink][yellow]...[/yellow][/blink]')
        predict(checkpoint, f'data/AMR/amr_{version}/test.txt.features.preproc', device=device)
        test_score = post_process_then_run_eval_script(test_pred, False, False, version=version)
        history[test_pred] = test_score
    if not isinstance(test_score, F1_):
        cprint(f'Official score on testset: [red]{test_score.score.f:.1%}[/red]')
        print('\t'.join(f'{test_score[k].score * 100:.1f}' for k in finegrained))
    save_pickle(history, history_path)


def post_process_then_run_eval_script(path, use_fast=True, dev=True, version='2.0'):
    try:
        flash('Running post process [blink][yellow]...[/yellow][/blink]')
        run_cmd(f'sh postprocess_{version}.sh {path}')
        scores: SmatchScores = smatch_eval(f'{path}.post',
                                           f'data/AMR/amr_{version}/{"dev" if dev else "test"}.txt',
                                           use_fast=use_fast)
    except Exception as e:
        warnings.warn(f'Failed to parse the output of smatch script due to {e}')
        scores = F1_(float("nan"), float("nan"), float("nan"))
    return scores


if __name__ == '__main__':
    main()
