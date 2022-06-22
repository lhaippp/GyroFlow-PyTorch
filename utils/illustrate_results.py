import os
import json
import glob
import time
from prettytable import PrettyTable


def monitor():
    total_table = []
    for dataset in ['test']:
        table = PrettyTable(['exp_name', 'exp_id', 'backbone', 'tau', 'best_epe', 'regular_best_epe'],
                            sortby='best_epe',
                            header_style='upper',
                            valign='m',
                            title='{} Result'.format(dataset),
                            reversesort=True)

        valid_dirs = [
            "experiment_sigmoid_different_map_tau", "experiment_gumbel_softmax_map_tau", "experiment_sigmoid_different_map_tau_9k_data"
        ]
        exp_dirs = glob.glob('./experiments/*/exp_*')
        exp_dirs += glob.glob('/data/fusion_gyro_optical_flow/deep_gyro_of_fusion/work/SGF_diff_position_ablation/experiments/*/exp_*')
        exp_dirs += glob.glob(
            '/data/fusion_gyro_optical_flow/deep_gyro_of_fusion/work/20220421.replaceInitFlowWithGyroField/experiments/*/exp_*')
        exp_dirs += glob.glob('/data/fusion_gyro_optical_flow/deep_gyro_of_fusion/work/20220617.GyroFlow.hard.map/experiments/*/exp_*')
        exp_dirs += glob.glob(
            "/data/fusion_gyro_optical_flow/deep_gyro_of_fusion/work/20220617.GyroFlow.gumbel.softmax/experiments/*/exp_*")
        exp_dirs = [i for i in exp_dirs if i.split('/')[-2] in valid_dirs]

        for exp_dir in exp_dirs:
            if dataset == "test":
                params_json_path = os.path.join(exp_dir, 'params.json')
                results_best_json_path = os.path.join(exp_dir, 'test_metrics_best.json')
                # results_last_json_path = os.path.join(exp_dir, 'test_metrics_latest.json')
                # logs_txt_path = os.path.join(exp_dir, 'log.txt')
                if not os.path.exists(params_json_path) or not os.path.exists(results_best_json_path):
                    continue

                params = json.load(open(params_json_path, 'r'))
                best_results = json.load(open(results_best_json_path, 'r'))
                # last_results = json.load(open(results_last_json_path, 'r'))
                # exp info
                exp_name = exp_dir.split('/')[-2]
                exp_id = exp_dir.split('_')[-1]
                backbone = params["model_name"]
                tau = params["map_tau"]
                # model = params['model']
                # results
                if 'nn_epe' in best_results:
                    best_epe = '{:>8.4f}'.format(best_results['nn_epe'])
                    # last_epe = '{:>8.4f}'.format(last_results['nn_epe'])
                elif 'ar_epe' in best_results:
                    best_epe = '{:>8.4f}'.format(best_results['ar_epe'])
                    # last_epe = '{:>8.4f}'.format(last_results['ar_epe'])

                regular_best_epe = '{:>8.4f}'.format(best_results['ours_RE_None'])
                cur_row = [exp_name, exp_id, backbone, tau, best_epe, regular_best_epe]
                table.add_row(cur_row)

        print(table)
        total_table.append(str(table))


def run(interval):
    while True:
        monitor()
        time.sleep(interval)


if __name__ == '__main__':
    interval = 10 * 60
    run(interval)
