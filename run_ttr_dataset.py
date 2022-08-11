import numpy as np
import pandas as pd
import string
import random
import logging
from logging import getLogger
from colorama import init

log_colors_config = {
    'DEBUG': 'cyan',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red',
}

d_path = '../dataset/ttr/'

# read data
avk_hackathon_data_test_transactions = pd.read_csv(d_path + 'avk_hackathon_data_test_transactions.csv')
avk_hackathon_data_train_transactions = pd.read_csv(d_path + 'avk_hackathon_data_train_transactions.csv')

cols = ['party_rk', #'account_rk', 'financial_account_type_cd',
       'transaction_dttm', #'transaction_type_desc',
       'transaction_amt_rur', #'merchant_rk', 'merchant_type',
       'merchant_group_rk', 'category']

avk_hackathon_data_transactions = pd.concat([avk_hackathon_data_train_transactions, avk_hackathon_data_test_transactions])[cols]
del avk_hackathon_data_test_transactions
del avk_hackathon_data_train_transactions

avk_hackathon_data_transactions.dropna(subset=['merchant_group_rk'], inplace=True)

avk_hackathon_data_transactions['category'].fillna('UNK', inplace=True)
avk_hackathon_data_transactions.to_csv(d_path + 'avk_hackathon_data_transactions.csv', index=False)

# user_fts = pd.read_csv(d_path + 'avk_hackathon_data_party_x_socdem.csv')    

# def make_user_file():

#     user_fts['age'] = user_fts.age.astype(int)
#     user_fts['gender_cd'] = user_fts['gender_cd'].fillna('UNK')
#     user_fts['marital_status_desc'] = user_fts.marital_status_desc.fillna('UNK')
#     user_fts.columns = ['user_id:token', 'gender:token', 'age:token', 'marital_status_desc:token', 'children_cnt:token', 'region_flg:token']
#     print(f'shape user_fts: {user_fts.shape}')
#     logger.info(f'shape user_fts: {user_fts.shape}')

#     # save results
#     user_fts.to_csv(d_path + 'ttr.user', index=False)


def make_item_and_link_file():

    # DISTINCT merchant_group - most pop category
    df_user_agr = avk_hackathon_data_transactions.groupby(['merchant_group_rk', 'category'])['party_rk'].count().reset_index().sort_values(
                                                            by=['merchant_group_rk', 'party_rk'], ascending=False
                                                          ).reset_index(drop=True)
    
    # encoding categories
    categories = list(df_user_agr.category.unique())
    print(f'n categories: {len(categories)}')
    logger.info(f'n categories: {len(categories)}')
    categories_mapping = dict(zip(categories, list(range(len(categories)))))
    df_user_agr['category_id'] = df_user_agr['category'].map(categories_mapping)

    # graph
    df_item_category_full = df_user_agr[['merchant_group_rk', 'category_id']].drop_duplicates()
    df_item_category_full.columns = ['item_id:token', 'category_id:token']
    df_item_category_full.to_csv(d_path + 'item_category_full.csv', index=False)


    df_user_agr.drop_duplicates(subset=['merchant_group_rk'], keep='first', inplace=True)
    print(f'shape df_transact_agr: {df_user_agr.shape}')
    logger.info(f'n categories: {len(categories)}')

    # merchant_group - Avg(transaction_amt_rur)
    df_transact_agr = avk_hackathon_data_transactions.groupby(['merchant_group_rk'])['transaction_amt_rur'].mean().reset_index()
    cut_labels_trs = np.arange(1, 21)
    df_transact_agr['transaction_bin_number'] = pd.qcut(df_transact_agr['transaction_amt_rur'], q=20, labels=cut_labels_trs)
    print(*sorted(pd.qcut(df_transact_agr['transaction_amt_rur'], q=20).unique()))
    # print(f'shape df_transact_agr: {df_transact_agr.shape}')
    # logger.info(f'shape df_transact_agr: {df_transact_agr.shape}')

    item_fts = df_user_agr.sort_values('merchant_group_rk').reset_index(drop=True) # merchant_group_rk - category - party_rk
    # + transaction_bin_number
    item_fts = item_fts.set_index('merchant_group_rk').join(df_transact_agr.set_index('merchant_group_rk')[['transaction_bin_number']]).reset_index()
    item_fts = item_fts[['merchant_group_rk', 'category_id', 'transaction_bin_number']]
    item_fts['merchant_group_rk'] = item_fts.merchant_group_rk.astype(int)
    item_fts.columns = ['item_id:token', 'category_id:token', 'transaction_bin_number:token']
    print(f'shape .item: {item_fts.shape}')
    logger.info(f'shape .item: {item_fts.shape}')

    # # save results
    item_fts.to_csv(d_path + 'ttr.item', index=False)

    # LINK FILES
    df_item_entities = pd.DataFrame(sorted(item_fts['item_id:token']), columns=['item_id:token'])
    df_item_entities['entity_id:token'] = df_item_entities['item_id:token'].apply(lambda x: 'i.' + ''.join(random.choices(string.ascii_lowercase+ string.digits, k = 5)))
    print(f'shape .link: {df_item_entities.shape}')
    logger.info(f'shape .link: {df_item_entities.shape}')

    # save results
    df_item_entities.to_csv(d_path + 'ttr.link', index=False)

    # category link
    df_category_entities = pd.DataFrame(sorted(item_fts['category_id:token'].drop_duplicates()), columns=['id:token'])
    df_category_entities['entity_id:token'] = df_category_entities['id:token'].apply(lambda x: 'c.' + ''.join(random.choices(string.ascii_lowercase+ string.digits, k = 5)))
    print(f'shape .category_link: {df_category_entities.shape}')
    logger.info(f'shape .category_link: {df_category_entities.shape}')
    df_category_entities.to_csv(d_path + 'category_link.kg', index=False)

    # transaction link
    df_transact_entities = pd.DataFrame(sorted(item_fts['transaction_bin_number:token'].drop_duplicates()), columns=['id:token'])
    df_transact_entities['entity_id:token'] = df_transact_entities['id:token'].apply(lambda x: 't.' + ''.join(random.choices(string.ascii_lowercase+ string.digits, k = 5)))
    print(f'shape .transact_link: {df_transact_entities.shape}')
    logger.info(f'shape .transact_link: {df_transact_entities.shape}')
    df_transact_entities.to_csv(d_path + 'transact_link.kg', index=False)


def make_inter_file():
    avk_hackathon_data_transactions['timestamp:float'] = pd.to_datetime(avk_hackathon_data_transactions.transaction_dttm).astype(int)/ 10**9
    
    df_inter = avk_hackathon_data_transactions[['party_rk', 'merchant_group_rk', 'timestamp:float']]
    df_inter = df_inter.sort_values('timestamp:float').drop_duplicates(subset=['party_rk', 'merchant_group_rk'], keep='first')
    print(f'shape .df_inter before filtering: {df_inter.shape}')

    inter_stats = df_inter.groupby('party_rk')[['merchant_group_rk']].count().reset_index()
    
    df_inter = df_inter[df_inter['party_rk'].map(dict(zip(inter_stats['party_rk'], inter_stats['merchant_group_rk'] >= 10)))]

    df_inter['merchant_group_rk'] = df_inter['merchant_group_rk'].astype(int)
    df_inter['rating'] = 1
    df_inter = df_inter[['party_rk', 'merchant_group_rk', 'rating', 'timestamp:float']]
    df_inter.columns = ['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float']
    print(f'shape .df_inter: {df_inter.shape}')
    logger.info(f'shape .df_inter: {df_inter.shape}')

    # save results
    df_inter.to_csv(d_path + 'ttr.inter', index=False)

    # LINK FILES
    df_item_entities = pd.DataFrame(sorted(df_inter['item_id:token']), columns=['item_id:token'])
    df_item_entities['entity_id:token'] = df_item_entities['item_id:token'].apply(lambda x: 'i.' + ''.join(random.choices(string.ascii_lowercase+ string.digits, k = 5)))
    print(f'shape .link: {df_item_entities.shape}')
    logger.info(f'shape .link: {df_item_entities.shape}')

    # save results
    df_item_entities.to_csv(d_path + 'ttr.link', index=False)


def make_kg_file():
    df_item_entities = pd.read_csv(d_path + 'ttr.link')
    df_category_entities = pd.read_csv(d_path + 'category_link.kg')
    df_transact_entities = pd.read_csv(d_path + 'transact_link.kg')

    inter_df = pd.read_csv(d_path + 'ttr.inter')
    item_df = pd.read_csv(d_path + 'ttr.item')
    item_category_full = pd.read_csv(d_path + 'item_category_full.csv')

    item_entities_link_dict = dict(zip(df_item_entities['item_id:token'], df_item_entities['entity_id:token']))
    category_entities_link_dict = dict(zip(df_category_entities['id:token'], df_category_entities['entity_id:token']))
    transact_entities_link_dict = dict(zip(df_transact_entities['id:token'], df_transact_entities['entity_id:token']))

    # cobyu
    df_cobyu = inter_df.set_index(['user_id:token', 'timestamp:float'])[['item_id:token']].join(
               inter_df.set_index(['user_id:token', 'timestamp:float'])[['item_id:token']],
               rsuffix='_also_buy').reset_index()
    df_cobyu = df_cobyu[df_cobyu['item_id:token'] != df_cobyu['item_id:token_also_buy']]
    df_cobyu = df_cobyu[['item_id:token', 'item_id:token_also_buy']].drop_duplicates()
    
    # item.item.also_byu
    df_kg_cobuy = pd.DataFrame()
    df_kg_cobuy['head_id:token'] = df_cobyu['item_id:token'].map(item_entities_link_dict)
    df_kg_cobuy['relation_id:token'] = 'item.item.also_byu'
    df_kg_cobuy['tail_id:token'] = df_cobyu['item_id:token_also_buy'].map(item_entities_link_dict)
    print(f'shape .df_kg_cobuy: {df_kg_cobuy.shape}')
    logger.info(f'shape .df_kg_cobuy: {df_kg_cobuy.shape}')

    # item.transact_bin.item_in_transact_bin
    df_kg_transact = pd.DataFrame()
    df_kg_transact['head_id:token'] = item_df['item_id:token'].map(item_entities_link_dict)
    df_kg_transact['relation_id:token'] = 'item.transact_bin.item_transact_bin'
    df_kg_transact['tail_id:token'] = item_df['transaction_bin_number:token'].map(transact_entities_link_dict)
    print(f'shape .df_kg_transact: {df_kg_transact.shape}')
    logger.info(f'shape .df_kg_transact: {df_kg_transact.shape}')

    # item.category.item_in_catogory
    df_kg_cat = pd.DataFrame()
    df_kg_cat['head_id:token'] = item_category_full['item_id:token'].map(item_entities_link_dict)
    df_kg_cat['relation_id:token'] = 'item.category.item_in_catogory'
    df_kg_cat['tail_id:token'] = item_category_full['category_id:token'].map(category_entities_link_dict)
    print(f'shape .df_kg_cat: {df_kg_cat.shape}')
    logger.info(f'shape .df_kg_cat: {df_kg_cat.shape}')

    df_kg_full = pd.concat([df_kg_cobuy, df_kg_transact, df_kg_cat])
    df_kg_full.to_csv(d_path + 'ttr.kg', index=False)
    print(f'shape .df_kg_full: {df_kg_full.shape}')
    logger.info(f'shape .df_kg_full: {df_kg_full.shape}')


if __name__ == '__main__':
    logging.basicConfig(filename='../dataset/ttr/make_ttr_dataset.log')
    logger = getLogger()
    # make_user_file()
    make_item_and_link_file()
    make_inter_file()
    make_kg_file()


