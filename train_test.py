import lightgbm as lgb
import csv
import pandas as pd
import time
import json
import pkuseg
import jieba
import math
# -*- coding: utf-8 -*-
from joblib import load, dump
from tqdm import tqdm
import re
import numpy as np
from features_ents import feature_ents
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.metrics import roc_auc_score, f1_score
import gc
import multiprocessing
from sklearn.metrics import precision_score, recall_score

class Train():
    def __init__(self):
        self.train_data_path = "data/coreEntityEmotion_train.txt"
        self.train_idf_path = "./data/train_idf.txt"

    def read_idf(self):
        idf = {}
        #if model == "train":
        with open(self.train_idf_path, 'r', encoding = 'utf-8') as f:
            for i in f.readlines():
                if len(i.strip().split()) == 2:
                    v = i.strip().split()
                    idf[v[0]] = float(v[1])
        return idf

    def train_get_feature_each_group(self, news_zuhe):
        news_list = news_zuhe.to_dict('recoreds')
        #print(news_list)
        fea_ents = feature_ents()
        buf = []
        #idf = self.read_idf()
        for news in tqdm(news_list):
            feature = fea_ents.combine_features(news, "train")
            buf.append(feature)
        return buf
        
    def train_get_feature(self, process_number):
        train_data = open(self.train_data_path, "r", encoding='utf-8')
        train_data_list = []
        ############################## 以下部分count为测试 #################################
        count = 0
        actual_coreentity = 0
        for line in train_data.readlines():
            news1 = json.loads(line.strip())
            for entity in news1['coreEntityEmotions']:
                actual_coreentity += 1
            train_data_list.append(news1)
            ####################### 测试 ###########################
            #count += 1
            #if count == 2:
                #break
            ####################### 结束测试 ########################
        res = []
        ########################## 开始多进程分组 ##############################
        allData = pd.DataFrame([news for news in train_data_list], index = [i for i in range(len(train_data_list))])
        allData['idx'] = allData.index.values
        indexs = [i for i in range(0,len(train_data_list))]
        ########################## 分割每个进程的数量 ###########################
        def div_list(ls,n):
            if not isinstance(ls,list) or not isinstance(n,int):
                return []
            ls_len = len(ls)
            if n<=0 or 0==ls_len:
                return []
            if n > ls_len:
                return []
            elif n == ls_len:
                return [[i] for i in ls]
            else:
                j = int(ls_len/n)
                k = int(ls_len%n)
                ls_return = []
                for i in range(0,(n-1)*j,j):
                    ls_return.append(ls[i:i+j])
                #算上末尾的j+k
                ls_return.append(ls[(n-1)*j:])
                return ls_return
        indexs = div_list(indexs, process_number)
        pool = multiprocessing.Pool(process_number)
        redata = pool.map(self.train_get_feature_each_group, [allData.loc[i] for i in indexs])
        ########################## 整合多进程结果 ##############################
        for news_feature_group in tqdm(redata):
            for each_news_feature in news_feature_group:
                res.append(each_news_feature)
        print(len(res))
        ########################## 把特征写到csv文件里 ##########################
        train_df = pd.concat(res, axis=0, join='outer').reset_index(drop=True)
        train_df.to_csv('./models/train_df.csv', index=False)
        print("多少主要实体： ", actual_coreentity)
        in_coreentity = train_df.loc[:,'label'].sum()
        print("实际有多少分词被主要实体发现：", in_coreentity)
        print("比例： ", in_coreentity / actual_coreentity)

class Train_Test():
    def __init__(self):
        self.a = 1

    def prepare(self, train_df, test_df):
        train_df1 = train_df
        test_df1 = test_df
        total = [train_df1, test_df1]
        total_df = pd.concat(total)
        enc = LabelEncoder().fit(total_df.cixing)

        test_df['cixing_enc'] = enc.transform(test_df.cixing)
        train_df['cixing_enc'] = enc.transform(train_df.cixing)
        
        ## 统计
        counter = Counter(test_df.tags.values)

        freq = train_df.tags.apply(lambda x: counter[x]).reset_index(drop=True)
        train_df['tag_freq'] = freq
        test_df['tag_freq'] = test_df.tags.apply(lambda x: counter[x]).reset_index(drop=True)
        '''
        test_df['score'] = np.load('./models/lgb-Copy1.joblib')
        positive_counter = Counter(test_df[test_df.score >= 0.3333333333].tags.values)

        train_df['positive_tag_freq'] =  train_df.tags.apply(lambda x: positive_counter[x]).reset_index(drop=True)
        test_df['positive_tag_freq'] =  test_df.tags.apply(lambda x: positive_counter[x]).reset_index(drop=True)
        '''
        train_df.to_csv('./models/train_df_prepare.csv', index=False)
        test_df.to_csv('./models/test_df_prepare.csv', index=False)
        
        return train_df, test_df
    
    def tongji(self, train_df, test_df):
        a = train_df.id.unique()
        b = test_df.id.unique()
        print(len(a),"\n")
        print(len(b),"\n")

    def evaluate_5_fold(self, train_df, test_df, cols, model):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        y_test = 0
        oof_train = np.zeros((train_df.shape[0],))
        for i, (train_index, val_index) in enumerate(kf.split(train_df[cols])):
            X_train, y_train = train_df.loc[train_index, cols], train_df.label.values[train_index]
            X_val, y_val = train_df.loc[val_index, cols], train_df.label.values[val_index]
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval = lgb.Dataset(X_val, y_val,reference=lgb_train)
            print('开始训练......')
            params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'auc', 'binary_logloss'},
            'learning_rate': 0.01,
            'num_leaves': 38,
            'min_data_in_leaf': 170,
            'bagging_fraction': 0.85,
            'bagging_freq': 1,
            'seed':42,
            'num_threads':-1,
            }
            gbm = lgb.train(params,lgb_train,num_boost_round=40000,valid_sets=lgb_eval,early_stopping_rounds=50,verbose_eval=False,)
            dump(gbm, "models/"+"gbm_"+ str(i) +".joblib")
            y_pred = gbm.predict(X_val)
            if model == "test":
                y_test = gbm.predict(test_df.loc[:, cols])
            oof_train[val_index] = y_pred
        auc = roc_auc_score(train_df.label.values, oof_train)
        print('5 Fold auc:', auc)
        gc.collect()
        dump(auc, "models/"+"auc"+".joblib")
        dump(oof_train, "models/"+"oof_train"+".joblib")
        dump(y_test, "models/"+"y_test"+".joblib")
        return auc, oof_train, y_test

class Test():
    def __init__(self):
        self.test_file = 'data/coreEntityEmotion_test_stage1.txt'
        self.test_idf_path = "./data/test_idf.txt"

    def read_idf(self):
        idf = {}
        with open(self.test_idf_path, 'r', encoding = 'utf-8') as f:
            for i in f.readlines():
                if len(i.strip().split()) == 2:
                    v = i.strip().split()
                    idf[v[0]] = float(v[1])
        return idf

    def test_get_feature_each_group(self, news_zuhe):
        news_list = news_zuhe.to_dict('recoreds')
        fea_ents = feature_ents()
        #idf = self.read_idf()
        buf = []
        for news in tqdm(news_list):
            feature = fea_ents.combine_features(news, "test")
            buf.append(feature)
        return buf

    def test_get_feature(self, process_number):
        test_data = open(self.test_file, "r", encoding='utf-8')
        test_data_list = []
        ############################## 以下部分count为测试 #################################
        count = 0
        for line in test_data.readlines():
            news1 = json.loads(line.strip())
            test_data_list.append(news1)
            ####################### 测试 ###########################
            #count += 1
            #if count == 2:
                #break
            ####################### 结束测试 ########################
        res = []
        ########################## 开始多进程分组 ##############################
        allData = pd.DataFrame([news for news in test_data_list], index = [i for i in range(len(test_data_list))])
        allData['idx'] = allData.index.values
        indexs = [i for i in range(0,len(test_data_list))]
        ########################## 分割每个进程的数量 ###########################
        def div_list(ls,n):
            if not isinstance(ls,list) or not isinstance(n,int):
                return []
            ls_len = len(ls)
            if n<=0 or 0==ls_len:
                return []
            if n > ls_len:
                return []
            elif n == ls_len:
                return [[i] for i in ls]
            else:
                j = int(ls_len/n)
                k = int(ls_len%n)
                ls_return = []
                for i in range(0,(n-1)*j,j):
                    ls_return.append(ls[i:i+j])
                #算上末尾的j+k
                ls_return.append(ls[(n-1)*j:])
                return ls_return
        indexs = div_list(indexs, process_number)
        pool = multiprocessing.Pool(process_number)
        redata = pool.map(self.test_get_feature_each_group, [allData.loc[i] for i in indexs])
        ########################## 整合多进程结果 ##############################
        for news_feature_group in tqdm(redata):
            for each_news_feature in news_feature_group:
                res.append(each_news_feature)
        print(len(res))
        ########################## 把特征写到csv文件里 ##########################
        test_df = pd.concat(res, axis=0,join='outer').reset_index(drop=True)
        test_df.to_csv('./models/test_df.csv', index=False)

def get_keywords(x):
    score = x.score.values
    tags = x.tags.values
    ret = pd.Series()
    ret['id'] = x['id'].values[0]
    ####### 阈值0.2,0.3都会有问题 #########
    yuzhi = 0.25
    def check_substring(x, check_list):
        for each_word in check_list:
            if x in each_word or each_word in x:
                return True
        return False
    if len(tags) == 0:
        ret['label1'] = ''
        ret['label2'] = ''
        ret['label3'] = ''
        ret['score1'] = ''
        ret['score2'] = ''
        ret['score3'] = ''
    elif len(tags) == 1:
        ret['label1'] = tags[0]
        ret['label2'] = ''
        ret['label3'] = ''
        ret['score1'] = score[0]
        ret['score2'] = ''
        ret['score3'] = ''
    elif len(tags) == 2:
        ret['label1'] = tags[0]
        if score[1] >= yuzhi and tags[1] not in tags[0] and tags[0] not in tags[1]:
            ret['label2'] = tags[1]
        else:
            ret['label2'] = ''
        ret['label3'] = ''
        
        ret['score1'] = score[0]
        if score[1] >= yuzhi and tags[1] not in tags[0] and tags[0] not in tags[1]:
            ret['score2'] = score[1]
        else:
            ret['score2'] = ''
        ret['score3'] = ''
    else:
        sort = np.argsort(score)[::-1]
        ret['label1'] = tags[sort[0]]
        ret['label2'] = ''
        ret['label3'] = ''
        count = 2 #从第二个开始
        temp = []
        temp.append(tags[sort[0]])
        second_score = 0
        for i in range(1, len(sort)):
            if score[sort[i]] < yuzhi or count == 4:
                break
            if score[sort[i]] >= yuzhi and (not check_substring(tags[sort[i]],temp)):
                ret['label{}'.format(count)] = tags[sort[i]]
                count += 1
                '''
                if count == 2:
                    if score[sort[i]]*2 >= score[sort[0]]:
                        ret['label{}'.format(count)] = tags[sort[i]]
                        second_score = score[sort[i]]
                        count += 1
                    elif score[sort[i]]*2 < score[sort[0]]:
                        count = 4
                elif count == 3:
                    if score[sort[i]] + score[sort[0]] >= 2*second_score:
                        ret['label{}'.format(count)] = tags[sort[i]]
                        count += 1
                    else:
                        count = 4
                 '''
            temp.append(tags[sort[i]])
        ## 这部分主要是sort，不影响最后输出txt
        ret['score1'] = score[sort[0]]
        ret['score2'] = ''
        ret['score3'] = ''
        count = 2
        temp = []
        temp.append(tags[sort[0]])
        for i in range(1, len(sort)):
            if score[sort[i]] < yuzhi or count == 4:
                break
            if score[sort[i]] >= yuzhi and (not check_substring(tags[sort[i]],temp)):
                #ret['label2'] = tags[sort[1]]
                ret['score{}'.format(count)] = score[sort[i]]
                count += 1
            temp.append(tags[sort[i]])
    return ret

## 后处理 之前写得 没什么用 主要是置换逗号
def postprocessing(x):
    x['label1'] = x['label1'].replace(',', '，')
    x['label2'] = x['label2'].replace(',', '，')
    x['label3'] = x['label3'].replace(',', '，')
    return x

class Get_idf():
    def __init__(self):
        self.train_data_path = "data/coreEntityEmotion_train.txt"
        self.test_data_path = "data/coreEntityEmotion_test_stage1.txt"
        #self.train_idf_path = "./data/train_idf.txt"
        #self.test_idf_path = "./data/test_idf.txt"
    
    def get_idf_each_process_news(self, news_zuhe):
        news_list = news_zuhe.to_dict('recoreds')
        buf = []
        for news in tqdm(news_list):
            fea_ents = feature_ents()
            news = fea_ents.process_sentence(news)
            text = 30*(news['title']+'。') + 3*(news['first_sentence']+'。') + 1*(news['other_sentence']+'。')+\
            3*(news['last_sentence']+'。')
            jieba_tags = jieba.analyse.extract_tags(sentence=text, topK=40, allowPOS=('r','m','d', 'p', 'q', 'ad', 'u', 'f'))
            #print(jieba_tags)
            buf.append(jieba_tags)
        return buf
    
    def docs(self, w, D):
        c = 0
        for d in D:
            if w in d:
                c = c + 1
        return c

    def save(self, idf_dict, path):
        #print (idf_dict)
        f = open(path, "w", encoding = 'utf-8')
        #f.truncate()
        # write_list = []
        for key in idf_dict.keys():
            # write_list.append(str(key)+" "+str(idf_dict[key]))
            f.write(str(key) + " " + str(idf_dict[key]) + "\n")
        f.close()

    def get_idf_all_news(self, model, process_number):
        if model == "train":
            train_data = open(self.train_data_path, "r", encoding='utf-8')
        elif model == "test":
            train_data = open(self.test_data_path, "r", encoding='utf-8')
        train_data_list = []
        ############################## 以下部分count为测试 #################################
        count = 0
        for line in train_data.readlines():
            news1 = json.loads(line.strip())
            train_data_list.append(news1)
            ####################### 测试 ###########################
            #count += 1
            #if count == 2:
                #break
            ####################### 结束测试 ########################
        ########################## 开始多进程分组 ##############################
        allData = pd.DataFrame([news for news in train_data_list], index = [i for i in range(len(train_data_list))])
        indexs = [i for i in range(0,len(train_data_list))]
        ########################## 分割每个进程的数量 ###########################
        def div_list(ls,n):
            if not isinstance(ls,list) or not isinstance(n,int):
                return []
            ls_len = len(ls)
            if n<=0 or 0==ls_len:
                return []
            if n > ls_len:
                return []
            elif n == ls_len:
                return [[i] for i in ls]
            else:
                j = int(ls_len/n)
                k = int(ls_len%n)
                ls_return = []
                for i in range(0,(n-1)*j,j):
                    ls_return.append(ls[i:i+j])
                #算上末尾的j+k
                ls_return.append(ls[(n-1)*j:])
                return ls_return
        indexs = div_list(indexs, process_number)
        pool = multiprocessing.Pool(process_number)
        redata = pool.map(self.get_idf_each_process_news, [allData.loc[i] for i in indexs])
        ########################## 整合多进程结果 ##############################
        D = []  # 所有分词后文档
        W = set() #所有词的set
        for each_process in tqdm(redata):
            for each_news in each_process:
                #print(each_news)
                D.append(each_news)
                W = W | set(each_news)
        #计算idf
        idf_dict = {}
        n = len(D)
        #print(D)
        #print(len(D))
        #print(W)
        #idf = log(n / docs(w, D))
        for w in list(W):
            idf = math.log(n * 1.0 / self.docs(w, D))
            idf_dict[w] = idf
        ########################## 最后写入文件  ###############################
        if model == "train":
            path = "./data/train_idf.txt"
        elif model == "test":
            path = "./data/test_idf.txt"
        self.save(idf_dict, path)

if __name__ == "__main__":
    
    ## 搜狗+百度词典 深蓝词典转换
    jieba.load_userdict('./字典/明星.txt')
    jieba.load_userdict('./字典/实体名词.txt')
    jieba.load_userdict('./字典/歌手.txt')
    jieba.load_userdict('./字典/动漫.txt')
    jieba.load_userdict('./字典/电影.txt')
    jieba.load_userdict('./字典/电视剧.txt')
    jieba.load_userdict('./字典/流行歌.txt')
    jieba.load_userdict('./字典/创造101.txt')
    jieba.load_userdict('./字典/百度明星.txt')
    jieba.load_userdict('./字典/美食.txt')
    jieba.load_userdict('./字典/FIFA.txt')
    jieba.load_userdict('./字典/NBA.txt')
    jieba.load_userdict('./字典/网络流行新词.txt')
    jieba.load_userdict('./字典/显卡.txt')
    ## 爬取漫漫看网站和百度热点上面的词条
    jieba.load_userdict('./字典/漫漫看_明星.txt')
    jieba.load_userdict('./字典/百度热点人物+手机+软件.txt')
    jieba.load_userdict('./字典/自定义词典.txt')
    ## 实体名词抽取之后的结果 有一定的人工过滤 
    ## origin_zimu 这个只是把英文的组织名过滤出来
    jieba.load_userdict('./字典/person.txt')
    jieba.load_userdict('./字典/origin_zimu.txt')
    ## 第一个是所有《》里面出现的实体名词
    ## 后者是本地测试集的关键词加上了 
    jieba.load_userdict('./字典/出现的作品名字.txt')
    jieba.load_userdict('./字典/val_keywords.txt')
    jieba.load_userdict('./data/nerDict.txt')
    ## 停用词合集
    jieba.analyse.set_stop_words('./data/stopwords.txt')
   
    process_number = 50
    
    '''
    print("\nGet idf now...\n")
    get_idf = Get_idf()
    get_idf.get_idf_all_news("train", process_number)
    get_idf.get_idf_all_news("test", process_number)
    print("Complete !!! Get idf ...\n")
    '''
    print("Get feature now...\n")
    train = Train()
    train.train_get_feature(process_number)
    test = Test()
    test.test_get_feature(process_number)
    print("Complete !!! Get all feature...\n")
    
    print("Loading Trained And Tested CSV Data...\n")
    train_df = pd.read_csv('./models/train_df.csv')
    test_df = pd.read_csv('./models/test_df.csv')
    print("Complete !!! Loading Trained And Tested CSV Data...\n")
    
    train_test = Train_Test()
    
    print("Preprocessing...\n")
    #train_df, test_df = train_test.prepare(train_df, test_df)
    #train_test.tongji(train_df, test_df)
    train_df, test_df = train_test.prepare(train_df)
    train_test.tongji(train_df)
    print("Preprocessing... Done\n")
    
    print("Loading PreProcessed Trained And Tested CSV Data...\n")
    train_df = pd.read_csv('./models/train_df_prepare.csv')
    test_df = pd.read_csv('./models/test_df_prepare.csv')
    print("Complete !!! Loading PreProcessed Trained And Tested CSV Data...\n")
    
    cols = [col for col in train_df.columns if col not in ['tags', 'label', 'cixing', 'id', 'cixing_z_bili']]
    auc, lgb_oof_train, lgb_sub = train_test.evaluate_5_fold(train_df, test_df, cols, "test")
    
    auc = load("./models/auc.joblib")
    lgb_oof_train = load("./models/oof_train.joblib")
    lgb_sub = load("./models/y_test.joblib")
    
    print("Output...\n")
    test_df['score'] = lgb_sub
    id_ = test_df.id.unique()
    #print(test_df['id'])
    sub = pd.DataFrame()
    sub['id'] = id_
    sub = test_df.groupby(by='id', sort = False)
    #print(sub['id'])
    sub = sub.apply(get_keywords)
    #print(sub['id'])
    sub.fillna('', inplace = True)
    #sub = test_df.groupby('id').apply(get_keywords)

    sub.fillna('', inplace=True)
    sub = sub.apply(postprocessing, axis=1)
    sub.to_csv('sub.csv', index=False)

    
    sub = pd.read_csv('sub.csv')
    sub.fillna('', inplace = True)
    process_num = "1"
    res_file = open("./results/result_"+str(process_num)+".txt",'w', newline='', encoding='utf-8')
    write = csv.writer(res_file, delimiter='"')
    #print("length_sub_index", len(sub.index))
    for indexs in sub.index:
        #print(sub.loc[indexs])
        ents = []
        emos = []
        if sub.loc[indexs, 'label1'] == '':
            pass
        else:
            ents.append(sub.loc[indexs, 'label1'])
        if sub.loc[indexs, 'label2'] == '':
            pass
        else:
            ents.append(sub.loc[indexs, 'label2'])
        if sub.loc[indexs, 'label3'] == '':
            pass
        else:
            ents.append(sub.loc[indexs, 'label3'])

        for ent in ents:
            emos.append("POS")
            
        row = ['{}\t{}\t{}'.format(sub.loc[indexs,'id'], ','.join(ents), ','.join(emos))]
        write.writerow(row)
    print('Done.\n')
    
