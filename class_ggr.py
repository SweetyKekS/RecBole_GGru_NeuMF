import recbole
import pandas as pd
import torch
import numpy as np
import os
import random
from pandas.core.common import random_state
import logging
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import NeuMF
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from recbole.quick_start import run_recbole, load_data_and_model
from recbole.utils.case_study import full_sort_topk


class GGR_NeuMF:
    def __init__(self, path, top_k = 10, pretrain = False, epochs = 10):
        self.website = pd.read_csv(path)
        self.website['userId'] = self.website['userId'][self.website['userId'] > 0]
        # print(website.head())
        self.website_new = self.website.copy()
        # print(website_new.head(20))
        self.epochs = epochs
        self.website_new = self.website_new.drop(columns=['message', 'streamer', 'user', 'streamStarted', 'game', 'event'])
        self.website_new = self.website_new.dropna()
        self.temp_result = 0
        self.users_list = self.website_new['userId'].unique().astype('int')
        self.stream_list = self.website_new['streamId'].unique().astype('int')
        self.stream_list = self.stream_list[self.stream_list >= 0]
        global users_stream_dict
        users_stream_dict = self.pair2dict()

        if not os.path.exists('website_wr.csv'):
            print('Файл "website_wr.csv" не существует.')
            self.website_new['rating'] = self.website_new.apply(self.rating, axis=1)
            self.website_new.loc[self.website_new['rating'] > 100, 'rating'] = 5
            self.website_new.loc[self.website_new['rating'] > 50, 'rating'] = 4
            self.website_new.loc[self.website_new['rating'] > 10, 'rating'] = 3
            self.website_new.loc[self.website_new['rating'] > 6, 'rating'] = 2
            self.website_new.loc[self.website_new['rating'] > 5, 'rating'] = 1
            self.website_new.to_csv('website_wr.csv', index= False)
            print('website_wr.csv сохранён.')
        self.website_new = pd.read_csv('website_wr.csv')
        # print(self.website_new)

        if not os.path.exists('recbox_data/NeuMF/NeuMF.inter'):
            print('Файл "NeuMF.inter" не существует.')
            self.temp = self.website_new[['userId', 'streamId', 'timestamp', 'rating']].rename(
                columns={'userId': 'user_id:token', 'streamId': 'item_id:token', 'rating': 'rating:float',
                         'timestamp': 'timestamp:float'})
            self.temp.to_csv('recbox_data/NeuMF/NeuMF.inter', index=False, sep='\t')
            print('Файл "NeuMF.inter" сохранён.')

        if not pretrain:
            self.parameter_dict = {
                'data_path': 'recbox_data',
                'USER_ID_FIELD': 'user_id',
                'train_batch_size': 256,
                'eval_batch_size': 128,
                'ITEM_ID_FIELD': 'item_id',
                'RATING_FIELD': 'rating',
                'TIME_FIELD': 'timestamp',
                'NEG_PREFIX': 'neg_',
                'show_progress': True,
                'checkpoint_dir': 'saved/',
                'threshold':{'rating': 3},
                'user_inter_num_interval': "[2,inf)",
                'item_inter_num_interval': "[2,inf)",
                'load_col': {'inter': ['user_id', 'item_id', 'timestamp', 'rating']},
                'unused_col': {'inter': ['timestamp']},
                'neg_sampling': None,
                'epochs': epochs,
                'valid_metric': 'NDCG@10',
                'eval_args': {
                    'split': {'RS': [7.5, 1.5, 1]},
                    'group_by': 'user',
                    'order': 'TO',
                    'mode': 'full'}
            }
            # Используем модель NeuMF, модель общей рекомендации, обучаются на данных неявной обратной связи и оцениваются в рамках задачи рекомендации top-n.
            # Все модели на основе коллаборативных фильтров (CF) относятся к этому классу.
            self.config = Config(model='NeuMF', dataset='NeuMF', config_dict=self.parameter_dict)

            init_seed(self.config['seed'], self.config['reproducibility'])

            init_logger(self.config)
            self.logger = getLogger()
            #
            # self.c_handler = logging.StreamHandler()
            # self.c_handler.setLevel(logging.INFO)
            # self.logger.addHandler(self.c_handler)
            #
            # self.logger.info(self.config)
            self.dataset = create_dataset(self.config)
            # self.logger.info(self.dataset)
            self.train_data, self.valid_data, self.test_data = data_preparation(self.config, self.dataset)
            self.model = NeuMF(self.config, self.train_data.dataset).to(self.config['device'])
            # self.logger.info(self.model)

            # trainer loading and initialization
            self.trainer = Trainer(self.config, self.model)

            # model training
            self.best_valid_score, self.best_valid_result = self.trainer.fit(self.train_data)

        else:
            self.config, self.model, self.dataset, self.train_data, self.valid_data, self.test_data = load_data_and_model(
                model_file='saved/NeuMF-Sep-30-2022_15-56-44.pth', )

        self.external_user_ids = self.dataset.id2token(
            self.dataset.uid_field, list(range(self.dataset.user_num)))[1:]
        self.topk_items = []
        for internal_user_id in list(range(self.dataset.user_num))[1:]:
            try:
                _, self.topk_iid_list = full_sort_topk([internal_user_id], self.model,
                                                       self.test_data, k=top_k, device=self.config['device'])
            except TypeError:
                pass
            try:
                self.last_topk_iid_list = self.topk_iid_list[-1]
            except IndexError:
                pass
            self.external_item_list = self.dataset.id2token(self.dataset.iid_field, self.last_topk_iid_list.cpu()).tolist()
            self.topk_items.append(self.external_item_list)
        self.external_item_str = [' '.join(x) for x in self.topk_items]
        self.result = pd.DataFrame(self.external_user_ids, columns=['user_id'])
        self.result['prediction'] = self.external_item_str
        self.temp_result = self.result.copy()
        self.temp_result['user_id'] = self.temp_result['user_id'].astype('float')

    def load_and_train(self,epochs=10):
        config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
            model_file='saved/NeuMF-Sep-30-2022_15-56-44.pth', )
        self.trainer = Trainer(config, model)
        self.trainer.epochs = epochs
        print(self.trainer.valid_metric)
        self.best_valid_score, self.best_valid_result = self.trainer.fit(train_data)



    def show_prediction(self, num = 20):
        return self.result.head(num)

    @staticmethod
    def rating(seq):
        userid2 = seq['userId']
        streamid2 = seq['streamId']
        for i in users_stream_dict[userid2]:
            if streamid2 == i[0]:
                return i[1]

    def pair2dict(self):
        # Создадим словарь пользватель: [(стример: количество раз встречалась пара пользователь/стример в файле)]
        users_stream_dict = {}
        for i in self.users_list:
            temp = self.website_new['streamId'][self.website_new['userId'] == i].value_counts()
            streamid = temp.index.astype('int')
            rat = temp.values
            temp_zip = list(zip(streamid, rat))
            users_stream_dict[i] = temp_zip
        return users_stream_dict

    def pred_to_list(self, user_id):
        temp_result = self.result.copy()
        temp_result['user_id'] = temp_result['user_id'].astype('float')
        return list(temp_result.loc[temp_result['user_id'] == user_id]['prediction'].values[0].split(' '))

    def to_csv(self):
        return self.result.to_csv(f'NeuMF_{self.epochs}_epochs.csv', index = False)

    def to_dict(self):
        result_dict = self.temp_result.copy()
        result_dict.set_index('user_id', inplace=True)
        result_dict = result_dict.to_dict('index')
        return result_dict

    def check_result_by_content(self, user_id):
        num_ids = self.temp_result[self.temp_result['user_id'] == user_id].index.values[0]
        user_res = float(self.result['user_id'].values[num_ids])
        print('Какие игры смотрел пользователь: \n', self.website['game'][self.website['userId'] == user_res].value_counts())
        for stream in self.result['prediction'].values[num_ids].split(' '):
            print("\t", 'StreamId: ', stream, '\t Game:',
                  self.website['game'][self.website['streamId'] == float(stream)].unique() )
