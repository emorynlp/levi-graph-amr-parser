# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-09-18 12:44
import argparse

from amr_parser.work import predict
from elit.metrics.amr.smatch_eval import smatch_eval, SmatchScores
from elit.utils.io_util import run_cmd
from elit.utils.log_util import flash, cprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str, help='The checkpoint')
    parser.add_argument('--version', type=str, help='AMR data version', default='2.0')
    args = parser.parse_args()
    version = args.version
    device = 0
    checkpoint = args.checkpoint
    test_pred = f'{checkpoint}_test_out.pred'
    flash(f'Running prediction {checkpoint} on testset [blink][yellow]...[/yellow][/blink]')
    predict(checkpoint, f'data/AMR/amr_{version}/test.txt.features.preproc', device=device)
    test_score = eval_checkpoint(test_pred, False, False, version)
    cprint(f'Official score on testset: [red]{test_score.score:.1%}[/red]')
    print(test_score)


def eval_checkpoint(path, use_fast=True, dev=True, version='2.0'):
    run_cmd(f'sh postprocess_{version}.sh {path}')
    scores: SmatchScores = smatch_eval(f'{path}.post',
                                       f'/home/hhe43/amr_gs/data/AMR/amr_{version}/{"dev" if dev else "test"}.txt',
                                       use_fast=use_fast)
    return scores


if __name__ == '__main__':
    main()
