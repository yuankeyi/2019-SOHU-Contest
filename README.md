
# 2019-SOHU-Contest
### 一、比赛简介

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;第三届搜狐校园算法大赛开赛！ 2019年4月8日，第三届搜狐校园内容识别算法大赛正式开赛，同期面向参赛选手开放竞赛结果提交。搜狐携手清华计算机系共同发起本届大赛，面向全球范围内的全日制在校生，旨在通过提供业务场景、真实数据、专家指导，选拔和培养有志于自然语言处理领域的算法研究、应用探索的青年才俊，共同探索更多可能、开启无限未来。  大赛页面地址：<https://biendata.com/competition/sohu2019/>  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;本次比赛的主题是提取文章主题，并判断文章对主题的情绪。我们生活在一个信息爆炸的世界，每天能接触到不同的新闻文章，体裁也多种多样，包括新闻快讯、广告软文、深度分析、事件评论，以及重要人物采访等等。每天新产生的信息量已经极大地超过了读者能够接受的极限。所以，如果有一种机器模型，可以自动提取出文章的主题，并且判断出文章对这些主题内容的情感倾向，就可以极大地提高阅读和消化信息的效率。  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;具体来说，参赛选手需要根据给定的文章，提取出文章中最重要的三个主题（也就是实体）。所谓实体，意思是人、物、地区、机构、团体、企业、行业等事物。和一般的实体抽取竞赛任务不同的是，本次比赛还要求选手判断文章对主题实体的情感倾向（包括积极、中立和消极三种情绪）。  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;本次比赛的数据来自搜狐智能媒体研发中心。搜狐智能媒体研发中心，是搜狐的核心用户产品及智能技术研发部门。部门依托平台化和智能化的技术能力，在内容领域不断深耕，以提升用户体验为核心目标，不断推陈出新，改良现有产品，探索新形式。初赛将发布8万条数据，其中训练集预计将有4万条数据，每条数据中的文章都经过人工标注。参赛选手需要利用训练集的数据和标签开发自己的模型，并在测试集上评测自己的模型。

### 二、参赛情况

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在本次比赛中，评测方式为：实体词的 F1-score 以及实体情绪的F1-score 组成，每个样本计算micro F1-Score，然后取所有样本分数的平均值。具体评分细则可查看：<https://biendata.com/competition/sohu2019/evaluation/>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;本次比赛一共吸引共计约900多支队伍，而我们的最终成绩处在**第16位**。主要原因是，我们认为其情绪不如直接随机设成POS来得更好。
- 总分：0.466622166278914
- 实体分数：0.5965043422739756
- 情感分数：0.3367399902838523

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们在本次比赛中采取如下策略：
- 实体部分：我们通过两个模型，一个是LGBM的决策树的结果，另一个是深度学习的结果。通过每个模型，输出每条新闻所有的可能的实体和对应的score，在两个模型中都取出来并求平均。将平均分从高到低排列，取平均分前三个的即可，且分数大于0.25的即可。

- 情感部分：直接全部设成POS分数最高。

### 三、文件介绍

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;本篇github涉及内容为LGBM的决策树以及相关的特征工程。下面我们对文件夹以及文件进行简单介绍。

1. `data/`:该文件夹存放存放40000条训练新闻集与40000条测试新闻集。具体如下：
- `coreEntityEmotion_sample_submission_v2.txt`：主办方告诉你如何提交规定格式的`.txt`文件
- `coreEntityEmotion_train.txt`：40000条训练新闻集。格式如下：
```
news = {"newsId":"0xabcd", "coreEntity":[{"entity":"TOM", "emotion":"NEG"}, {"entity":"BOB", "emotion":"NEG"}], "title": "It is good", "content": "xxxxxxxx"}
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;由于文件太大并没有放在这里，下载链接：<https://pan.baidu.com/s/1L3tHoIBPsEvRXC0puV90wg>， 提取码：gm0g 
- `coreEntityEmotion_test_stage1.txt`：40000条测试新闻集。格式如下：
```
news = {"newsId":"0xabcd", "title": "It is good", "content": "xxxxxxxx"}
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;由于文件太大并没有放在这里，下载链接：<https://pan.baidu.com/s/1d7A3JR1IxiGyk6ORq30_ng>，提取码：6d03 
- `nerDict.txt`：主办方提供的分词词典
- `stopwords.txt`：停用词词典

2. `jieba/`:修改过的jieba源码库。其中`allowPos`的源码已经修改成**不允许**出现的词性，请注意！

3. `models/`:存放LGBM决策树跑出的结果，缓存到`.joblib`中。以及有2个含目标分词的以及其特征值的`.csv`文件

4. `results`：存放结果

5. `vecmodels`: 存放了3样东西：100维的Doc2Vec，100维的Word2Vec，256维的Word2Vec。具体看文件名很好分辨

6. `字典`：存放用户的各种自定义分词词典

7. `sub.csc`:存的是每条新闻对应的分词，以及分词所对应的score值

8. `feature_ents.py`：特征工程。根据每条新闻，去对应计算特征值。其中，计算的特征有：
```
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
    feature['cixing_{}_num'.format(c)] = cixing_counter[c]
    feature['cixing_{}_bili'.format(c)] = cixing_counter[c] / len(new_cixing)
    ########################## 训练时需要调用label ######################
    feature['label'] = label
    ########################## 结束并返回 ##############################
    return feature
```

9. `train_test.py`：运行主程序，其中会调用`feature_ents.py`。参数说明：
- `process_number = 50`:我使用了python的多进程来完成本次训练。由于数据量和运算量巨大，因此我用了50个进程来完成。你们可以根据实际需要进行调整。
- `Get_idf()`:这是一个算实际idf的值的类，但比赛中发现不如不加比较好。
- `train.train_get_feature(process_number)`: 训练数据的feature获得过程。
- `test.test_get_feature(process_number)`: 测试数据的feature获得过程。
- `train_test.prepare(train_df, test_df)`: 对训练和测试数据进行处理。
- `train_test.evaluate_5_fold`: 验证集。在这里，数据集被随机分成了5份。

### 四、代码运行

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;首先，请确保您的电脑安装了（没有标注版本号的即为最新版即可，使用`pip`安装即可）：
- `python3.5`
- `pandas`
- `jieba`
- `joblib`
- `tqdm`
- `sklearn`
- `gc`
- `multiprocessing`
- `re`
- `gensim`
- `scipy`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;其次，请确保在开头输入 `# -*- coding: utf-8 -*-`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;最后，在windows或linux下的命令行输入`python train_test.py`即可。

### 五、说明

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;本代码版权为作者所有，仅可用于大家学习交流，不可用于商业用途。违者必究。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如有疑问，可以Email作者进行询问。