# encoding:utf8
# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')
from gensim import models, corpora, similarities
from matplotlib import pyplot
import scipy as sp
import numpy as np
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
import pandas as pd
from jpype import *
import jieba
def nb():
    f1 = open('stopword.txt')
    stop_words = f1.read().splitlines()
    f1.close()
    tfidf = TfidfVectorizer(binary=False, decode_error='ignore', stop_words=stop_words)
    f2 = open('question_lb.csv')
    print('stop_words', stop_words)
    corpus = [line.strip().split(',')[0] for line in open('question_lb.csv',encoding='utf8').readlines()[:102]]
    print(corpus)
    labels = [line.strip().split(',')[1] for line in open('question_lb.csv',encoding='utf8').readlines()[:102]]
    print(labels)
    test_corpus = [line.strip() for line in open('question_lb.csv',encoding='utf8').readlines()[102:]]
    print(test_corpus)
    tfidf_train = tfidf.fit_transform(corpus)
    tfidf_test = tfidf.transform(test_corpus)# ValueError dimension mismatch·
    print(tfidf_train, 'tfidf_test\n',tfidf_test)
    # labels = [line.strip().split(',')[1] for line in open('question_lb.csv', encoding='utf8').readlines()[:102]]
    # print(labels)
    lb=list(set(labels))
    # print(lb)
    dict_lb1={}
    dict_lb2={}
    for i,j in enumerate(lb):
        dict_lb1[j]=i
        dict_lb2[i]=j
    y_labels=[dict_lb1[j] for j in labels]
    print(y_labels)
    clf = MultinomialNB().fit(tfidf_train, y_labels)

    p = clf.predict(tfidf_test)
    res=[dict_lb2[i] for i in p]

    print(labels)
    print(res)
    print(p)
    for i in res:
        print(i)

def kmeans():
    f1 = open('stopword.txt')
    # stop_words = f1.read().splitlines()
    # print('stop_words', stop_words)

    stop_words = [line.strip() for line in f1.readlines()]
    stop_words.extend(['刘老师', '黄老师', '老师', '您好', '你好', '\\t', '\\', 't'])
    f1.close()
    tfidf = TfidfVectorizer(binary=False, decode_error='ignore', stop_words=stop_words)
    f2 = open('question_lb.csv')
    print('stop_words', stop_words)
    corpus = [line.strip() for line in open('question_lb.csv', encoding='gb18030').readlines()]
    print(corpus)
    # test_corpus = [line.strip() for line in open('question_lb.csv', encoding='utf8').readlines()[1000:]]
    # print(test_corpus)
    tfidf_train = tfidf.fit_transform(corpus)
    # tfidf_test = tfidf.transform(test_corpus)  # ValueError dimension mismatch·
    print(tfidf_train, 'tfidf_test\n')#, tfidf_test)
    # 调用kmeans类
    from sklearn.cluster import KMeans
    clf = KMeans(n_clusters=9).fit(tfidf_train)

    # 9个中心
    print(clf.cluster_centers_)

    # 每个样本所属的簇
    print(clf.labels_)
    pd.DataFrame(clf.labels_).to_csv('labels.csv')
    # 用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
    print(clf.inertia_)

def lstm():
    # from gensim.models import Word2Vec
    from gensim.models import word2vec
    import jieba
    # 引入日志配置
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    from gensim.models.keyedvectors import KeyedVectors

    f1 = open('stop_words.txt')
    stop_words =[line.strip() for line in  f1]
    stop_words.extend(['刘老师', '黄老师', '老师', '您好', '你好', '\\t', '\\', 't','高考'])
    print(stop_words)
    f1.close()

    corpus = []
    for line in open('question_lb.csv', encoding='gb18030').readlines():
        line = [i for i in jieba.cut(line.strip()) if i not in stop_words]
        corpus.append(line)
    print(corpus)
    model = word2vec.Word2Vec(corpus )
    model.save('model.model')
    model=word2vec.KeyedVectors.load('model.model')
    vector = np.array([model[word] for word in (model.wv.vocab)])
    print(vector)
    doc_vector=[]
    for doc in corpus :
        l=[0]*len(vector[0])
        l_count=0
        for word in doc :
            if word in model.wv.vocab:
                l+=model[word]
                l_count+=1
        doc_vector.append(np.array(l)/l_count)
    doc_vector_df=pd.DataFrame(doc_vector)
    doc_vector_df.fillna(0,inplace=True)
    print('doc_vector_df_0\n', doc_vector_df)
    doc_vector=np.array(doc_vector_df)
    print(doc_vector)
    print(doc_vector.shape)#(5773, 100)

    from numpy import NaN
    # # 调用kmeans类
    from sklearn.cluster import KMeans
    clf = KMeans(n_clusters=9).fit(doc_vector)
    # 9个中心:9*100
    print(clf.cluster_centers_)

    # 每个样本所属的簇
    print(clf.labels_)
    pd.DataFrame(clf.labels_).to_csv('labels.csv')
    print(len(clf.labels_))

    # 用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
    print(clf.inertia_)

def xgb_filter(q):
    #使用hanlp自定义字典分词
    startJVM(getDefaultJVMPath(),"-Djava.class.path=F:\hanlp\hanlp-1.2.8.jar;F:\hanlp", "-Xms1g", "-Xmx1g")
    HanLP = JClass('com.hankcs.hanlp.HanLP')
    def generalSetting(enable=None):
        Config = JClass('com.hankcs.hanlp.HanLP$Config')
        if enable == False:
            Config.ShowTermNature = False
        else:
            Config.ShowTermNature = True
    generalSetting(enable=False)
    q = HanLP.segment(q)

    #一般停用词
    f1 = open('stop_words.txt')
    stop_words = [line.strip() for line in f1]
    stop_words.extend(['刘老师', '黄老师', '老师', '您好', '你好', '\\t', '\\', 't', '高考',' '])
    #加入高考代名词的停用词：
    stop_words0 = stop_words;
    stop_words0.extend(['sch', 'pro', 'cla', 'bat', 'maj', 'pnt', 'lx', 'typ', 'rk', 'lsy'])
    print(stop_words)

    path='G:/MY/gaokao/web/myweb/myDic/'
    major=[line.strip() for line in open(path+'/专业.txt',encoding='gb18030')]
    point=[line.strip() for line in open(path+'分数线.txt',encoding='gb18030')]
    batch=[line.strip() for line in open(path+'批次.txt',encoding='gb18030')]
    rank=[line.strip() for line in open(path+'排名.txt',encoding='gb18030')]
    province=[line.strip() for line in open(path+'生源地.txt',encoding='gb18030')]
    classy=[line.strip() for line in open(path+'科类.txt',encoding='gb18030')]
    lishuyu=[line.strip() for line in open(path+'隶属于.txt',encoding='gb18030')]
    school=[line.strip() for line in open(path+'高校全称及简称.txt',encoding='gb18030')]
    leixing=[line.strip() for line in open(path+'高校属性.txt',encoding='gb18030')]
    school_type=[line.strip() for line in open(path+'高校类型.txt',encoding='gb18030')]
    q_list=[]
    #把分完词之后的高考领域的词转化成代名词'maj'，'sch'。。。
    print(q)
    for i in q:
        # type(i)# <class 'jpype._jclass.com.hankcs.hanlp.seg.common.Term'>
        if str(i) in major:q_list.append('maj')
        elif str(i) in point:q_list.append('pnt')
        elif str(i) in batch:q_list.append('bat')
        elif str(i) in rank:q_list.append('rk')
        elif str(i) in province:q_list.append('pro')
        elif str(i) in classy:q_list.append('cla')
        elif str(i) in lishuyu:q_list.append('lsy')
        elif str(i) in school:q_list.append('sch')
        elif str(i) in leixing:q_list.append('lx')
        elif str(i) in school_type:q_list.append('typ')
        else:q_list.append(str(i))
    # 去一般停用词：
    q1 = [i for i in q_list if i not in stop_words]
    print(q_list)

    q0 = [i for i in q_list if i not in stop_words0]
    return q0

def xgb_tfidf():
    f1 = open('stopword.txt')
    # stop_words = f1.read().splitlines()
    # print('stop_words', stop_words)

    stop_words = [line.strip() for line in f1.readlines()]
    stop_words.extend(['刘老师', '黄老师', '老师', '您好', '你好', '\\t', '\\', 't','请问'])
    f1.close()
    tfidf = TfidfVectorizer(binary=False, decode_error='ignore', stop_words=stop_words)
    f2 = open('question_lb.csv',encoding='gb18030').readlines()
    print('stop_words', stop_words)
    corpus = [line.strip().split(',')[0] for line in f2[:148]]
    print(corpus)
    test_corpus = [line.strip().split(',')[0] for line in f2[148:]]
    print(test_corpus)
    tfidf_train = tfidf.fit_transform(corpus)
    tfidf_test = tfidf.transform(test_corpus)  # ValueError dimension mismatch·
    print(tfidf_train, 'tfidf_test\n')  # , tfidf_test)
    labels = [line.strip().split(',')[1] for line in  f2 ]
    print(labels)
    lb = list(set(labels))
    # print(lb)
    dict_lb1 = {}
    dict_lb2 = {}
    for i, j in enumerate(lb):
        dict_lb1[j] = i
        dict_lb2[i] = j
    y_labels = [dict_lb1[j] for j in labels]
    print(y_labels)

    train_x, train_y, test_x, test_y = tfidf_train, y_labels[:148], tfidf_test, y_labels[148:]
    return (train_x, train_y, test_x, test_y,dict_lb1,dict_lb2)

def xgb_word2vec(q):
    from gensim.models import word2vec
    import jieba
    # 引入日志配置
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    from gensim.models.keyedvectors import KeyedVectors

    f1 = open('stop_words.txt')

    stop_words = [line.strip() for line in f1]
    stop_words.extend(['刘老师', '黄老师', '老师', '您好', '你好', '\\t', '\\', 't', '高考'])
    # stop_words.extend(['sch', 'pro', 'cla', 'bat', 'maj', 'pnt', 'lx', 'typ', 'rk', 'lsy'])
    print(stop_words)
    f1.close()
    f2 = open('question_lb.csv',encoding='gb18030').readlines()

    corpus = []
    for line in f2:
        line = [i for i in jieba.cut(line.strip()) if i not in stop_words]
        # line=xgb_filter(line)#去掉专业、学校等词
        corpus.append(line)
    q = [i for i in jieba.cut(q.strip()) if i not in stop_words]
    corpus.append(q)

    print('corpus\n',corpus)
    model = word2vec.Word2Vec(corpus)
    model.save('model.model')
    model = word2vec.KeyedVectors.load('model.model')

    vector = np.array([model[word] for word in (model.wv.vocab)])
    print(vector)
    doc_vector = []
    for doc in corpus:
        l = [0] * len(vector[0])
        l_count = 0
        for word in doc:
            if word in model.wv.vocab:
                l += model[word]
                l_count += 1
        doc_vector.append(np.array(l) / l_count)
    doc_vector_df = pd.DataFrame(doc_vector)
    doc_vector_df.fillna(0, inplace=True)
    # print('doc_vector_df_0\n', doc_vector_df)
    doc_vector = np.array(doc_vector_df)
    print('doc_vector\n',doc_vector)
    labels = [line.strip().split(',')[1] for line in f2]
    print(labels)
    lb = list(set(labels))

    # print(lb)
    dict_lb1 = {}
    dict_lb2 = {}
    for i, j in enumerate(lb):
        dict_lb1[j] = i
        dict_lb2[i] = j
    y_labels = np.array([dict_lb1[j] for j in labels])
    print(y_labels)

    train_x, train_y, test_x, test_y=doc_vector[:148],y_labels[:148],doc_vector[148:-1],y_labels[148:]
    pred_y=doc_vector[-1:]
    print(corpus[148:],'\n',labels[148:])
    print(doc_vector[148:],'\n',y_labels[148:])
    return (train_x,train_y,test_x,test_y,pred_y,dict_lb1,dict_lb2)

def xgb(train_x, train_y, test_x, test_y,pred_y,dict_lb1,dict_lb2,filename):
    import xgboost as xgb
    print(train_x, train_y, test_x, test_y,dict_lb1,dict_lb2)
    xg_train = xgb.DMatrix(train_x, label=train_y)
    xg_test = xgb.DMatrix(test_x, label=test_y)
    xg_pred=xgb.DMatrix(pred_y)
    # setup parameters for xgboost
    param = {}
    # use softmax multi-class classification
    param['objective'] = 'multi:softmax'
    # scale weight of positive examples
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    # param['nthread'] = 4
    param['num_class'] = 7

    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    num_round = 5
    print('111i am here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    bst = xgb.train(param, xg_train, num_round, watchlist);
    bst.save_model('{}.model'.format(filename))
    print('2222i am here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    # get prediction
    pred = bst.predict(xg_test);
    print(pred)

#predict

    import xgboost as xgb
    # xg_question=xgb.DMatrix(doc_vector)
    # bst=xgb.Booster()#{'nthread': 4}
    # bst.load_model('xgb_word2vec.model')
    pred_ = bst.predict(xg_pred);
    print(pred_)
    category=dict_lb2[pred_[0]]
    return category

if __name__=="__main__":
    # kmeans()
    # lstm()
    # train_x, train_y, test_x, test_y, dict_lb1, dict_lb2 = xgb_tfidf()
    # xgb(train_x, train_y, test_x, test_y, dict_lb1, dict_lb2,filename='xgb_tfidf')
    train_x, train_y, test_x, test_y ,pred_y,dict_lb1, dict_lb2 = xgb_word2vec()
    xgb(train_x, train_y, test_x, test_y, pred_y,dict_lb1, dict_lb2,filename='xgb_word2vec')

    # def py2_word2vec():
    #     # import word2vec
    #     corpus_seg=open('corpus_seg.txt','w+')
    #     for line in open('question_lb.csv').readlines():#, encoding='gb18030'
    #         line=[i for i in jieba.cut(line.strip().decode('gb18030','ignore')) if i not in stop_words]
    #         line=' '.join(line)
    #         corpus_seg.write(line+'\n')
    #     corpus_seg.close()
    #     word2vec.word2vec('corpus_seg.txt','corpusWord2Vec.bin',verbose=True)
    #     model = word2vec.load('corpusWord2Vec.bin')
    #     print(model.vectors)
    #     vector=model.vectors

    # from keras.layers import
    # from keras.models import Model

    # inpE = Input((10, 5))  # here, you don't define the batch size
    # outE = LSTM(units=20, return_sequences=False, )
    # encoder = Model(inpE, outE)
    # inpD = Input((20,))
    # outD = Reshape((10, 2))
    # outD1 = LSTM(5, return_sequences=True,
    # alternativeOut=LSTM(50, return_sequences=False, )
    # alternativeOut = Reshape((10, 5))

    #
    # decoder = Model(inpD, outD1)
    # alternativeDecoder = Model(inpD, alternativeOut)
    # encoderPredictions = encoder.predict(data)
    # import gensim
    # model_3 = gensim.models.KeyedVectors.load_word2vec_format("corpus.model.txt", binary=True)
    # print(model_3)
    # import pandas as pd
    # for line in open("corpus.model.txt",encoding='utf8',errors='ignore').readlines():
    #     print(line)

    #     data=pd.read_csv("corpus.model.txt",sep=)
    #     print(data)
    #     data=np.array(data)
    #     print(data)

  # print('predicting, classification error=%f' % (sum(int(pred[i]) != test_y[i] for i in range(len(test_y)))/float(len(test_y))))
    #
    # # do the same thing again, but output probabilities
    # param['objective'] = 'multi:softprob'
    # bst = xgb.train(param, xg_train, num_round, watchlist);
    # # Note: this convention has been changed since xgboost-unity
    # # get prediction, this is in 1D array, need reshape to (ndata, nclass)
    # print('i am here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # yprob = bst.predict(xg_test)
    # print(yprob)
    # yprob=yprob.reshape(np.array(test_y).shape[0], 8)
    # ylabel = np.argmax(np.array(yprob), axis=0)  # return the index of the biggest pro
    # ylabel=[i.index(max(i)) for i in yprob.tolist()]
    # print(ylabel)
    # print( 'predicting, classification error=%f' % (sum(int(ylabel[i]) != test_y[i] for i in range(len(test_y))) /  float(len(test_y))))
    #
    # res = [dict_lb2[i] for i in ylabel]
    # print(ylabel)
    # print(res)
    # 调用kmeans类