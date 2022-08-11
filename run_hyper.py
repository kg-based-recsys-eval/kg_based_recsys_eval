# -*- coding: utf-8 -*-
# @Time   : 2020/7/24 15:57
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : run_hyper.py
# UPDATE:
# @Time   : 2020/8/20 21:17, 2020/8/29
# @Author : Zihan Lin, Yupeng Hou
# @Email  : linzihan.super@foxmail.com, houyupeng@ruc.edu.cn

import argparse

from recbole.trainer import HyperTuning
from recbole.quick_start import objective_function

from recbole.config import Config # NEW
from recbole.utils import init_logger, set_color # NEW
import logging # NEW
from logging import getLogger # NEW

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', type=str, default=None, help='fixed config files')
    parser.add_argument('--params_file', type=str, default=None, help='parameters file')
    parser.add_argument('--output_file', type=str, default='hyper_example.result', help='output file')
    args, _ = parser.parse_known_args()

    # plz set algo='exhaustive' to use exhaustive search, in this case, max_evals is auto set
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None

    config = Config(config_file_list=config_file_list) # NEW
    init_logger(config) # NEW
    logger = getLogger()  # NEW

    hp = HyperTuning(objective_function, algo='exhaustive',
                     params_file=args.params_file, fixed_config_file_list=config_file_list)
    hp.run()
    hp.export_result(output_file=args.output_file)

    logger.info(set_color('best params: ', 'yellow') + f': {hp.best_params}')  # NEW
    logger.info(set_color('best result:', 'yellow') + f': {hp.params2result[hp.params2str(hp.best_params)]}')  # NEW

    print('best params: ', hp.best_params)
    print('best result: ')
    print(hp.params2result[hp.params2str(hp.best_params)])


if __name__ == '__main__':
    main()


# config = Config(config_dict=config_dict, config_file_list=config_file_list)
# # logger initialization
# init_logger(config) # NEW
# logger = getLogger()  # NEW
# dataset = create_dataset(config)
# logger.info(dataset)  # NEW
# train_data, valid_data, test_data = data_preparation(config, dataset)
# init_seed(config['seed'], config['reproducibility'])
# model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
# logger.info(model)  # NEW
# trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
# best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, saved=saved)
# test_result = trainer.evaluate(test_data, load_best_model=saved)

# logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')  # NEW
# logger.info(set_color('test result', 'yellow') + f': {test_result}')  # NEW