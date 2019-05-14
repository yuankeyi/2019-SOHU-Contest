import jieba
import jieba.analyse
from joblib import dump,load
from tqdm import tqdm
import time
import json
import re
import numpy as np
import csv
from scipy import stats
from scipy.stats import skew, kurtosis
from collections import Counter
import pandas as pd
import math
# -*- coding: utf-8 -*-
import pkuseg
from gensim.models import Doc2Vec
from gensim.models import Word2Vec
from sklearn.metrics import precision_score, recall_score, f1_score

class feature_ents():
    def __init__(self):
        self.ner_dict_path = "../models/nerDict.txt"
        self.train_file_path = "./data/coreEntityEmotion_train.txt"
        #self.ner = ner()
        self.jieba = jieba

    # 针对每条news的title和content做操作,把一些奇奇怪怪的字符直接先去掉
    def clean_sentence(self, news):
        # 先针对title
        news['title'] = ''.join(filter(lambda ch: ch not in ' \t◆#%', news['title']))
        news['title'] = re.sub('&amp;', '', news['title'])
        news['title'] = re.sub('&quot;', '', news['title'])
        news['title'] = re.sub('&#34;', '', news['title'])
        news['title'] = re.sub('&nbsp;', '', news['title'])
        news['title'] = re.sub('&gt;', '', news['title'])
        news['title'] = re.sub('&lt;', '', news['title'])
        news['title'] = re.sub('hr/;', '', news['title'])
        news['title'] = re.sub(re.compile('······'), '', news['title'])
        # 再针对content
        news['content'] = ''.join(filter(lambda ch: ch not in ' \t◆#%', news['content']))
        news['content'] = re.sub('&amp;', '', news['content'])
        news['content'] = re.sub('&quot;', '', news['content'])
        news['content'] = re.sub('&#34;', '', news['content'])
        news['content'] = re.sub('&nbsp;', '', news['content'])
        news['content'] = re.sub('&gt;', '', news['content'])
        news['content'] = re.sub('&lt;', '', news['content'])
        news['content'] = re.sub('hr/;', '', news['content'])
        news['content'] = re.sub(re.compile('······'), '', news['content'])
        # 最后返回news这个dictionary
        #print(news['title'])
        #print(news['content'])
        return news
    
    # 提取出每一个新闻的首句，末句。剩下的即为中间句
    def process_sentence(self, news):
        if news['content'] == '' or len(news['content']) < 2:
            news['first_sentence'] = ''
            news['other_sentence'] = ''
            news['last_sentence'] = ''
            return news
        
        sentence_delimiters = re.compile(u'[。？！；!?]')
        sentences = [i for i in sentence_delimiters.split(news['content']) if i != '']
        num_sentence = len(sentences)
        #print(num_sentence)
        if num_sentence == 1:
            news['first_sentence'] = sentences[0]
            news['other_sentence'] = ''
            news['last_sentence'] = sentences[0]
        elif num_sentence == 2:
            news['first_sentence'] = sentences[0]
            news['other_sentence'] = ''
            news['last_sentence'] = sentences[-1]
        else:
            news['first_sentence'] = sentences[0]
            news['other_sentence'] = ''.join(sentences[1:-1])
            news['last_sentence'] = sentences[-1]
        #print(news)
        return news
    
    # 返回num_sentences, num_words, new_tags(目标20个分词), new_weight, new_cixing, 
    # len_tags, words, text
    def load_sentence(self, news):
        #jieba.load_userdict('./data/nerDict.txt')
        # 按标题、首句、其他句和末句按不同权重组合在一起，并先计算tfidf值。
        all_docs_reg = re.findall(r"《(.+?)》",news['first_sentence'])
        first_sentence_reg = ' '.join(all_docs_reg)
        text = 25*(news['title']+'。') + 5*(news['first_sentence']+'。') + 1*(news['other_sentence']+'。')+\
            3*(news['last_sentence']+'。')+8*first_sentence_reg
        #jieba_tags = jieba.analyse.extract_tags(sentence = text, topK= 20, allowPOS=('r','m','d','p','ad','u','f','l'),withWeight = True, withFlag = True)
        #jieba_tags = jieba.analyse.extract_tags(sentence = text, topK= 40, allowPOS = ('x', 'nz', 'l', 'n', 'v', 'ns', 'j', 'a', 'vn', 'nr', 'eng', 'nrt','t', 'z', 'i', 'b', 'o', 'nt', 'vd', 'c', 's', 'mq', 'rz','e', 'y', 'an', 'rr'),withWeight = True, withFlag = True)
        #print("jieba", jieba_tags)
        jieba_tags = jieba.analyse.extract_tags(sentence=text, topK=40, allowPOS=('r','m','d', 'p', 'q', 'ad', 'u', 'f'),withWeight=True,withFlag=True)
        #print("another jieba_tags", jieba_tags)
        # tags存放分句，cixing存放词性，weight存放对应的tfidf值
        tags = []
        cixing = []
        weight = []
        for partical in jieba_tags:
            tags.append(partical[0].word)
            cixing.append(partical[0].flag)
            weight.append(partical[1])
        #print(tags)
        #print(cixing)
        #print(weight)

        # 对有权重的句子进行分句处理
        sentence_delimiters = re.compile(u'[。？！；!?]')
        sentences = [i for i in sentence_delimiters.split(text) if i != '']
        num_sentences = len(sentences)

        # 把text中的每一句进行分词，统计加权总词数，以及把分词加进去
        words = []
        num_words = 0
        for each_sentence in sentences:
            cut = jieba.lcut(each_sentence)
            words.append(cut)
            num_words += len(cut)
        #print(words)
        #print(num_words)

        # 对分词可以进行再次筛选，筛选条件可人工设定
        new_tags = []
        new_cixing = []
        new_weight = []
        len_tags = [] #存的是分词长度
        for i in range(len(tags)):
            #print(tags[i])
            if tags[i].isdigit() and tags[i] not in ['985', '211']:
                continue
            if ',' in tags[i]:
                continue
            if '’' in tags[i]:
                continue
            new_tags.append(tags[i])
            new_weight.append(weight[i])
            new_cixing.append(cixing[i])
            len_tags.append(len(tags[i]))
        #print(new_tags)
        #print(new_cixing)
        #print(new_weight)
        #print(len_tags)
        return num_sentences, num_words, new_tags, new_weight, new_cixing,\
             len_tags, words, text
    
    # 0. 去读idf

    # 1. 检查是不是在title
    def in_title(self, news, new_tags):
        occur_in_title = np.zeros(len(new_tags))
        for i in range(len(new_tags)):
            if new_tags[i] in news['title']:
                occur_in_title[i] = 1
            else:
                occur_in_title[i] = 0
        return occur_in_title

    # 2. 检查是不是在首句
    def in_first_sentence(self, news, new_tags):
        occur_in_first_sentence = np.zeros(len(new_tags))
        for i in range(len(new_tags)):
            if new_tags[i] in news['first_sentence']:
                occur_in_first_sentence[i] = 1
            else:
                occur_in_first_sentence[i] = 0
        return occur_in_first_sentence

    # 3. 检查是不是在末句
    def in_last_sentence(self, news, new_tags):
        occur_in_last_sentence = np.zeros(len(new_tags))
        for i in range(len(new_tags)):
            if new_tags[i] in news['last_sentence']:
                occur_in_last_sentence[i] = 1
            else:
                occur_in_last_sentence[i] = 0
        return occur_in_last_sentence
    
    # 4. 检查是不是在中间
    def in_other_sentence(self, news, new_tags):
        occur_in_other_sentence = np.zeros(len(new_tags))
        for i in range(len(new_tags)):
            if new_tags[i] in news['other_sentence']:
                occur_in_other_sentence[i] = 1
            else:
                occur_in_other_sentence[i] = 0
        return occur_in_other_sentence

    # 5. 共现矩阵及相关统计特征。例如均值、方差、偏度等。得到新特征后贪心验证只保留以下三个
    # 返回的是var_gongxian, kurt_gongxian, diff_min_gongxian,ske
    def common_matrix(self, new_tags, words):
        num_tags = len(new_tags)
        arr = np.zeros((num_tags, num_tags))
        for i in range(num_tags):
            for j in range(i+1, num_tags):
                count = 0
                for each_word in words:
                    if new_tags[i] in each_word and new_tags[j] in each_word:
                        count += 1
                arr [i,j] = count 
                arr [j,i] = count
        
        ske = stats.skew(arr)
        var_gongxian = np.zeros(len(new_tags))
        kurt_gongxian = np.zeros(len(new_tags))
        diff_min_gongxian = np.zeros(len(new_tags))
        for i in range(num_tags):
            var_gongxian[i] = np.var(arr[i])
            kurt_gongxian[i] = stats.kurtosis(arr[i])
            diff_sim = np.diff(arr[i])
            if len(diff_sim) > 0:
                diff_min_gongxian[i] = np.min(diff_sim)
            else:
                diff_min_gongxian[i] = 0
        return var_gongxian, kurt_gongxian, diff_min_gongxian, ske

    # 6. 计算text_rank
    def cal_text_rank(self, new_tags, text):
        #textrank_tags = dict(jieba.analyse.textrank(sentence=text, allowPOS = ('x', 'nz', 'l', 'n', 'v', 'ns', 'j', 'a', 'vn', 'nr', 'eng', 'nrt','t', 'z', 'i', 'b', 'o', 'nt', 'vd', 'c', 's', 'mq', 'rz','e', 'y', 'an', 'rr'), withWeight=True))
        textrank_tags = dict(jieba.analyse.textrank(sentence=text, allowPOS=('r','m','d', 'p', 'q', 'ad', 'u', 'f'), withWeight=True))
        text_rank = []
        for tag in new_tags:
            if tag in textrank_tags:
                text_rank.append(textrank_tags[tag])
            else:
                text_rank.append(0)
        #print(text_rank)
        return text_rank
    
    # 7. 词频
    def text_frequency(self, new_tags, words):
        all_words = np.concatenate(words).tolist()
        tf = []
        for tag in new_tags:
            tf.append(all_words.count(tag))
        #print(tf)
        tf = np.array(tf)
        return tf
    
    # 8. 头词频，文章前四分之一的词频
    def head_text_frequency(self, new_tags, words):
        hf = []
        head = len(words) // 4 + 1
        head_words = np.concatenate(words[:head]).tolist()
        for tag in new_tags:
            hf.append(head_words.count(tag))
        #print(hf)
        return hf, head_words
    
    # 9. 是否包含数字
    def has_number(self, new_tags):
        has_number = []
        for tag in new_tags:
            if bool(re.search(r'\d', tag)) == True:
                has_number.append(1)
            else:
                has_number.append(0)
        #print(has_number)
        return has_number
    
    # 10. 是否包含字母
    def has_english(self, new_tags):
        has_english = []
        for tag in new_tags:
            if bool(re.search(r'[a-zA-Z]', tag)) == True:
                has_english.append(1)
            else:
                has_english.append(0)
        #print(has_english)
        return has_english
    
    # 11. 是否为电视作品名称（待补）
    def in_TV(self, new_tags, TV):
        is_TV = []
        for tag in new_tags:
            if tag in TV:
                is_TV.append(1)
            else:
                is_TV.append(0)
        return is_TV
    
    # 12. idf：永轩联机跑出的逆词频（待补）
    def cal_idf(self, idf, new_tags):
        v_idf = []
        for tag in new_tags:
            v_idf.append(idf.get(tag,0))
            #print(tag, idf.get(tag,0))
        return v_idf

    # 13. 计算文本相似度
    def text_similar(self, news, new_tags):
        doc2vec_model = Doc2Vec.load('./vecmodels/doc2vec.model')
        word2vec_model = Word2Vec.load('./vecmodels/word2vec.model')
        wv = word2vec_model.wv
        def Cosine(vec1, vec2):
            npvec1, npvec2 = np.array(vec1), np.array(vec2)
            if (math.sqrt((npvec1**2).sum()) * math.sqrt((npvec2**2).sum())) == 0:
                return 0.0
            else:
                return npvec1.dot(npvec2)/(math.sqrt((npvec1**2).sum()) * math.sqrt((npvec2**2).sum()))
        def Euclidean(vec1, vec2):
            npvec1, npvec2 = np.array(vec1), np.array(vec2)
            return math.sqrt(((npvec1-npvec2)**2).sum())
        ## 后面加大窗口和迭代又算了一次word2vec 模型 主要是用来算候选关键词之间的相似度
        #word2vec_model_256 = Word2Vec.load('word2vec_iter10_sh_1_hs_1_win_10.model')

        default = np.zeros(100)
        doc_vec = doc2vec_model.docvecs.vectors_docs[news['idx']]
        sim = []
        sim_euc = []
        for tag in new_tags:
            if tag in wv:
                #print(tag, "in wv")
                sim.append(Cosine(wv[tag], doc_vec))
                sim_euc.append(Euclidean(wv[tag], doc_vec))
            else:
                #print(tag, "not in wv")
                sim.append(Cosine(default, doc_vec))
                sim_euc.append(Euclidean(default, doc_vec))
        return sim, sim_euc

    # 14. 关键词所在句子长度的最大，最小，平均值
    def tag_sentence_length(self, new_tags, words):
        mean_tag_sentence_length = np.zeros(len(new_tags))
        max_tag_sentence_length = np.zeros(len(new_tags))
        min_tag_sentence_length = np.zeros(len(new_tags))
        for i in range(len(new_tags)):
            temp = []
            for each_word in words:
                #print(each_word)
                if new_tags[i] in each_word:
                    temp.append(len(each_word))
            if len(temp) > 0:
                mean_tag_sentence_length[i] = np.mean(temp)
                max_tag_sentence_length[i] = np.max(temp)
                min_tag_sentence_length[i] = np.min(temp)
        #print(mean_tag_sentence_length, max_tag_sentence_length, min_tag_sentence_length)
        return mean_tag_sentence_length, max_tag_sentence_length, min_tag_sentence_length

    # 15. 关键词所在位置，记录为列表，然后算统计特征
    def tag_position(self, new_tags, words):
        all_words = np.concatenate(words).tolist()
        min_pos = [np.NaN for _ in range(len(new_tags))]
        diff_min_pos_bili = [np.NaN for _ in range(len(new_tags))]
        diff_kurt_pos_bili = [np.NaN for _ in range(len(new_tags))]
        for i in range (len(new_tags)):
            pos = [a for a in range (len(all_words)) if all_words[a] == new_tags[i]]
            pos_bili = np.array(pos) / len(all_words)

            if len(pos) > 0:
                min_pos[i] = np.min(pos)
                diff_pos = np.diff(pos)
                diff_pos_bili = np.diff(pos_bili)

                if len(diff_pos) > 0:
                    diff_min_pos_bili[i] = np.min(diff_pos_bili)
                    diff_kurt_pos_bili[i] = stats.kurtosis(diff_pos_bili)
                else:
                    diff_min_pos_bili[i] = np.NaN
                    diff_kurt_pos_bili[i] = np.NaN
            
            else:
                min_pos[i] = np.NaN
                diff_min_pos_bili[i] = np.NaN
                diff_kurt_pos_bili[i] = np.NaN
        #print(min_pos, diff_min_pos_bili, diff_kurt_pos_bili)
        return min_pos, diff_min_pos_bili, diff_kurt_pos_bili
    
    # 16. 关键词所在句子特征，记录为列表，然后算特征
    def tag_sentence_position(self, new_tags, words):
        all_words = np.concatenate(words).tolist()
        diff_max_min_sen_pos = [np.NaN for _ in range(len(new_tags))]
        diff_var_sen_pos_bili = [np.NaN for _ in range(len(new_tags))]
        for i in range(len(new_tags)):
            pos = [a for a in range(len(words)) if new_tags[i] in words[a]]
            pos_bili = np.array(pos)/len(all_words)

            if len(pos) > 0:
                diff_pos = np.diff(pos)
                diff_pos_bili = np.diff(pos_bili)

                if len(diff_pos) > 0:
                    diff_max_min_sen_pos[i] = np.max(diff_pos) - np.min(diff_pos)
                    diff_var_sen_pos_bili[i] = np.var(diff_pos_bili)
                else:
                    diff_max_min_sen_pos[i] = np.NaN
                    diff_var_sen_pos_bili[i] = np.NaN
            else:
                diff_max_min_sen_pos[i] = np.NaN
                diff_var_sen_pos_bili[i] = np.NaN
        #print(diff_max_min_sen_pos, diff_var_sen_pos_bili)
        return diff_max_min_sen_pos, diff_var_sen_pos_bili

    # 17. 候选关键词之间的相似度 word2vec gensim 窗口默认 迭代默认 向量长度100
    # sim_tags_arr: 相似度矩阵
    def candidate_key_similar(self, new_tags):
        #doc2vec_model = Doc2Vec.load('./vecmodels/doc2vec.model')
        word2vec_model = Word2Vec.load('./vecmodels/word2vec.model')
        wv = word2vec_model.wv
        sim_tags_arr = np.zeros((len(new_tags), len(new_tags)))
        for i in range(len(new_tags)):
            for j in range(i+1, len(new_tags)):
                if new_tags[i] in wv and new_tags[j] in wv:
                    sim_tags_arr[i,j] = word2vec_model.similarity(new_tags[i], new_tags[j])
                    sim_tags_arr[j,i] = sim_tags_arr[i,j]
        mean_sim_tags = np.zeros(len(new_tags))
        diff_mean_sim_tags = np.zeros(len(new_tags))
        for i in range(len(new_tags)):
            mean_sim_tags[i] = np.mean(sim_tags_arr[i])
            diff_sim = np.diff(sim_tags_arr[i])
            if len(diff_sim) > 0:
                diff_mean_sim_tags[i] = np.mean(diff_sim)
        return mean_sim_tags, diff_mean_sim_tags

    # 18. 候选关键词word2vec gensim 窗口10 迭代10 向量长度256 （待补）
    def candidate_key_similar_256(self, new_tags):
        word2vec_model_256 = Word2Vec.load('./vecmodels/word2vec_iter10_sh_1_hs_1_win_10.model')
        sim_tags_arr_255 = np.zeros((len(new_tags), len(new_tags)))
        for i in range(len(new_tags)):
            for j in range(len(new_tags)):
                if new_tags[i] in word2vec_model_256 and new_tags[j] in word2vec_model_256:
                    sim_tags_arr_255[i, j] = word2vec_model_256.similarity(new_tags[i], new_tags[j])
                    sim_tags_arr_255[j, i] = sim_tags_arr_255[i, j]
        kurt_sim_tags_256 = np.zeros(len(new_tags))
        diff_max_min_sim_tags_256 = np.zeros(len(new_tags))       
        for i in range(len(new_tags)):
            kurt_sim_tags_256[i] = stats.kurtosis(sim_tags_arr_255[i])
            diff_sim = np.diff(sim_tags_arr_255[i])
            if len(diff_sim) > 0:
                diff_max_min_sim_tags_256[i] = np.max(diff_sim) - np.min(diff_sim)
        return kurt_sim_tags_256, diff_max_min_sim_tags_256

    # 19. label 训练集打标签
    def label(self, new_tags, model, news):
        label = []
        if model == "train":
            temp = []
            for coreEntity in news['coreEntityEmotions']:
                temp.append(coreEntity['entity'])
            for tag in new_tags:
                changed = False
                for i in range(len(temp)): 
                    #print(tag, temp[i])
                    if tag == temp[i]:
                        label.append(1)
                        #print(1111111111111111,"\n")
                        changed = True
                        break
                    #print(temp[-1],"\n")
                    #print(coreEntity == temp[-1],"\n")
                    if (i == len(temp) - 1 and changed == False):
                        label.append(0)
                        #print(2222222222222222,"\n")
            #print(label)
            return label                  
        elif model == "test":
            return None
    
    # 20. 关键词词性比例（下面有）
    '''
    def cixing_bili(self, new_cixing):
        cixing_counter = Counter(new_cixing)
        cixing_each_num = {}
        cixing_bili = {}
        for c in ['x', 'nz', 'l', 'n', 'v', 'ns', 'j', 'a', 'vn', 'nr', 'eng', 'nrt',
                  't', 'z', 'i', 'b', 'o', 'nt', 'vd', 'c', 's', 'mq', 'rz',
                  'e', 'y', 'an', 'rr']:
            cixing_each_num['cixing_{}_num'.format(c)] = cixing_counter[c]
            if len(new_cixing)==0:
                pass
            else:
                cixing_bili['cixing_{}_bili'.format(c)] = cixing_counter[c] / len(new_cixing)
        #print(cixing_each_num, cixing_bili)
        return cixing_each_num, cixing_bili
    '''

    # 把特征接到一起
    #def combine_features(self, news, model, idf):
    def combine_features(self, news, model):
        ######################### 调用函数获得特征 ##############################
        news = self.clean_sentence(news)
        news = self.process_sentence(news)
        TV = []
        with open('./字典/出现的作品名字.txt', 'r', encoding='utf-8') as f:
            for word in f.readlines():
                TV.append(word.strip())
        num_sentences, num_words, new_tags, new_weight, new_cixing, \
            len_tags, words, text= self.load_sentence(news)
        #print(num_sentences, num_words, new_tags, new_weight, new_cixing, len_tags)
        in_title = self.in_title(news, new_tags)
        in_first_sentence = self.in_first_sentence(news, new_tags)
        in_last_sentence = self.in_last_sentence(news, new_tags)
        in_other_sentence = self.in_other_sentence(news, new_tags)
        var_gongxian, kurt_gongxian, diff_min_gongxian, ske =\
            self.common_matrix(new_tags, words)
        text_rank = self.cal_text_rank(new_tags, text)
        text_frequency = self.text_frequency(new_tags, words)
        head_frequency , head_words = self.head_text_frequency(new_tags, words)
        has_number = self.has_number(new_tags)
        has_english = self.has_english(new_tags)
        is_TV = self.in_TV(new_tags, TV)
        mean_tag_sentence_length, max_tag_sentence_length, min_tag_sentence_length =\
            self.tag_sentence_length(new_tags, words)
        #v_idf = self.cal_idf(idf, new_tags)
        min_pos, diff_min_pos_bili, diff_kurt_pos_bili =\
            self.tag_position(new_tags, words)
        sim, sim_euc = self.text_similar(news,new_tags)
        diff_max_min_sen_pos, diff_var_sen_pos_bili =\
            self.tag_sentence_position(new_tags, words)
        mean_sim_tags, diff_mean_sim_tags = self.candidate_key_similar(new_tags)
        kurt_sim_tags_256, diff_max_min_sim_tags_256 = self.candidate_key_similar_256(new_tags)
        label = self.label(new_tags, model, news)
        #cixing_each_num, cixing_bili = self.cixing_bili(new_cixing)

        ######################### 特征整合阶段 ##############################
        feature = pd.DataFrame()
        feature['id'] = [news['newsId'] for _ in range (len(new_tags))]
        feature['tags'] = new_tags
        feature['cixing'] = new_cixing
        feature['tfidf'] = new_weight
        feature['ske'] = ske
        feature['occur_in_title'] = in_title
        feature['occur_in_first_sentence'] = in_first_sentence
        feature['occur_in_last_sentence'] = in_last_sentence
        feature['occur_in_other_sentence'] = in_other_sentence
        feature['len_tags'] = len_tags
        feature['num_tags'] = len(new_tags)
        feature['num_words'] = num_words
        feature['num_sen'] = num_sentences
        #feature['classes] = news['classes']
        feature['len_text'] = len(news['title']+news['content'])
        feature['text_rank'] = text_rank
        feature['word_count'] = text_frequency
        feature['tf'] = text_frequency / num_words
        feature['head_word_count'] = head_frequency
        feature['hf'] = np.array(head_frequency) / len(head_words)
        feature['pr'] = text_frequency / text_frequency.sum()
        feature['has_num'] = has_number
        feature['has_english'] = has_english
        feature['is_TV'] = is_TV
        #feature['idf'] = v_idf
        feature['sim'] = sim
        feature['sim_euc'] = sim_euc
        feature['mean_l2'] = mean_tag_sentence_length
        feature['max_l2'] = max_tag_sentence_length
        feature['min_l2'] = min_tag_sentence_length
        feature['min_pos'] = min_pos
        feature['diff_min_pos_bili'] = diff_min_pos_bili
        feature['diff_kurt_pos_bili'] = diff_kurt_pos_bili
        feature['diff_max_min_sen_pos'] = diff_max_min_sen_pos
        feature['diff_var_sen_pos_bili'] = diff_var_sen_pos_bili
        feature['mean_sim_tags'] = mean_sim_tags
        feature['diff_mean_sim_tags'] = diff_mean_sim_tags
        feature['kurt_sim_tags_256'] = kurt_sim_tags_256
        feature['diff_max_min_sim_tags_256'] = diff_max_min_sim_tags_256
        feature['var_gongxian'] = var_gongxian
        feature['kurt_gongxian'] = kurt_gongxian
        feature['diff_min_gongxian'] = diff_min_gongxian

        ########################## 词性特征整合 ############################
        cixing_counter = Counter(new_cixing)
        for c in ['x', 'nz', 'l', 'n', 'v', 'ns', 'j', 'a', 'vn', 'nr', 'eng', 'nrt',
                  't', 'z', 'i', 'b', 'o', 'nt', 'vd', 'c', 's', 'mq', 'rz',
                  'e', 'y', 'an', 'rr']:
            feature['cixing_{}_num'.format(c)] = cixing_counter[c]
            if len(new_cixing)==0:
                pass
            else:
                feature['cixing_{}_bili'.format(c)] = cixing_counter[c] / len(new_cixing)

        ########################## 训练时需要调用label ######################
        if model == 'train':
            feature['label'] = label

        ########################## 结束并返回 ##############################
        return feature

    
  
