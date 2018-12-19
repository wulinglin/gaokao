import pymysql,jieba,json
from gensim.models import word2vec
import numpy as np
import pandas as pd
from django.db import models
from myweb.models import *
from django.http import JsonResponse# from myweb import recommend
from gensim.models import word2vec
import jieba,os
# 引入日志配置
import logging
import pickle,jieba
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim.models.keyedvectors import KeyedVectors
# global info
info={}
roads_search = {'历年高校专业录取线查询':{'school':'高校名称','province':'生源地（省份）','classy':'科类（文/理科）','major':'专业名称'},
                    '历年高校录取线查询':{'school': '高校名称', 'province': '生源地（省份）', 'classy': '科类（文/理科）'},
                    '历年国家线查询':{'province': '生源地（省份）', 'classy': '科类（文/理科）','batch':'批次'},#不要批次
                    '心仪高校的专业排名':{'school': '高校名称', 'major': '专业名称'},#rank:排名？
                    '心仪专业的全国排名':{'major': '专业名称'},# 'rank': ''
                    '心仪高校的全国排名':{'school': '高校名称'},
                    '心仪高校的基本信息':{'school': '高校名称'}
                    }
search_dict = {'1': '历年国家线查询', '2': '历年高校录取线查询', '3': '历年高校专业录取线查询',
     '4':'心仪高校的专业排名', '5':'心仪专业的全国排名', '6' :'心仪高校的全国排名', '7': '心仪高校的基本信息'}
roads_recommend={
            '推荐学校': {'province':'生源地（省份）','classy':'科类（文/理科）','point':'分数','want_province': '想去的省份'},
             '推荐专业': {},
             '学校及专业推荐': {'province':'生源地（省份）','classy':'科类（文/理科）','point':'分数',
                         'isMajor':'是否有专业目标(若有则输入<专业名称>、无则输入<无>）'},#y/n
             '是否能被录取': {'school':'高校名称','province':'生源地（省份）','classy':'科类（文/理科）','point':'分数'},#专业可有可无
             '专业前景': {'major': '专业名称'},
             '复读': { },
             }
def xgb_word2vec(q):
    f1 = open('G:/MY/gaokao/web/myweb/stop_words.txt')
    stop_words = [line.strip() for line in f1.readlines()]
    print(stop_words)
    stop_words.extend(['刘老师', '黄老师', '老师', '您好', '你好', '\\t', '\\', 't', '高考'])
    # stop_words.extend(['sch', 'pro', 'cla', 'bat', 'maj', 'pnt', 'lx', 'typ', 'rk', 'lsy'])
    print(stop_words)
    f1.close()
    f2 = open('G:/MY/gaokao/web/myweb/question_lb.csv',encoding='gb18030').readlines()
    corpus = []
    for line in f2:
        line = [i for i in jieba.cut(line.strip()) if i not in stop_words]
        # line=xgb_filter(line)#去掉专业、学校等词
        corpus.append(line)
    q = [i for i in jieba.cut(q.strip()) if i not in stop_words]
    corpus.append(q)

    # print('corpus<br>',corpus)
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
    # print('doc_vector_df_0<br>', doc_vector_df)
    doc_vector = np.array(doc_vector_df)
    # print('doc_vector<br>',doc_vector)
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
    print(corpus[148:],'<br>',labels[148:])
    print(doc_vector[148:],'<br>',y_labels[148:])
    return (train_x,train_y,test_x,test_y,pred_y,dict_lb1,dict_lb2)

def xgbModel(train_x, train_y, test_x, test_y,pred_y,dict_lb1,dict_lb2,filename):
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

    print("开始训练模型")

    bst = xgb.train(param, xg_train, num_round, watchlist);
    print("模型训练完毕，开始保存模型")
    bst.save_model('{}.model'.format(filename))
    print("保存完毕")
    print('2222 i am here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # dumps(object) 返回一个字符串，它包含一个 pickle 格式的对象,dump(object, file) 将对象写到文件，这个文件可以是实际的物理文件，但也可以是任何类似于文件的对象，这个对象具有 write() 方法，可以接受单个的字符串参数；
    pickle.dump(dict_lb1,open( "dict_lb1.pickle", "wb" ))
    pickle.dump(dict_lb2,open("dict_lb2.pickle", "wb"))
    # get prediction
    pred = bst.predict(xg_test);
    print(pred)

#predict
    pred_ = bst.predict(xg_pred);
    print(pred_)
    category=dict_lb2[pred_[0]]
    info['category']=category
    return category

def getJieba(comment,custom=True):
    if custom==True:
        jieba.load_userdict('G:/MY/gaokao/web/myweb/myDic/高考_new.txt')
        seg_comment=list(jieba.cut(comment))
    if custom==False:
        seg_comment=list(jieba.cut(comment))
    return seg_comment

def stopWords():
    #一般停用词
    f1 = open('G:/MY/gaokao/web/myweb/stop_words.txt')
    stop_words1 = [line.strip() for line in f1]
    stop_words1.extend(['刘老师', '黄老师', '老师', '您好', '你好', '\\t', '\\', 't', '高考','请问',' '])
    #加入高考代名词的停用词：
    stop_words0 = stop_words1.copy();#stop_words0 = stop_words1会使两个list完全相同
    stop_words0.extend(['sch', 'pro', 'cla', 'bat', 'maj', 'pnt', 'lx', 'typ', 'rk', 'lsy'])
    return  stop_words1,stop_words0
# django里面的import和一般的python文件不一样么

def getPronoun(q):#代名词:q是分词去停用词之后的
    path='G:/MY/gaokao/web/myweb/myDic/'
    # path = os.getcwd()+'\myDic\\'
    major = [line.strip() for line in open(path + '专业.txt', encoding='gb18030')]
    point = [line.strip() for line in open(path + '分数线.txt', encoding='gb18030')]
    batch = [line.strip() for line in open(path + '批次.txt', encoding='gb18030')]
    rank = [line.strip() for line in open(path + '排名.txt', encoding='gb18030')]
    province = [line.strip() for line in open(path + '生源地.txt', encoding='gb18030')]
    classy = [line.strip() for line in open(path + '科类.txt', encoding='gb18030')]
    lishuyu = [line.strip() for line in open(path + '隶属于.txt', encoding='gb18030')]
    school = [line.strip() for line in open(path + '高校全称及简称.txt', encoding='gb18030')]
    leixing = [line.strip() for line in open(path + '高校属性.txt', encoding='gb18030')]
    school_type = [line.strip() for line in open(path + '高校类型.txt', encoding='gb18030')]
    q_list = []
    # 把分完词之后的高考领域的词转化成代名词'maj'，'sch'。。。
    print(q)
    info={}
    province_=[]
    for i in q:
        # type(i)# <class 'jpype._jclass.com.hankcs.hanlp.seg.common.Term'>
        if str(i) in major:
            info['major']=str(i)
            info['ismajor'] = str(i)
            q_list.append('maj')

        elif str(i).isdigit() and len(str(i)) == 3 and str(i) != 211 and str(i) != 985:
            info['point']= int(str(i))
            q_list.append('pnt')
        elif str(i) in batch:
            info['batch']=str(i)
            q_list.append('bat')
        elif str(i) in rank:
            info['rank'] = str(i)
            q_list.append('rk')
        elif str(i) in province:
            province_.append(str(i))
            info['province'] = province_#因为有want_province，所以是列表！
            q_list.append('pro')
        elif str(i) in classy:
            info['classy'] = str(i)
            q_list.append('cla')
        elif str(i) in lishuyu:
            info['lishuyu'] = str(i)
            q_list.append('lsy')
        elif str(i) in school:
            info['school'] = str(i)
            q_list.append('sch')
        elif str(i) in leixing:
            info['leixing'] = str(i)
            q_list.append('lx')
        elif str(i) in school_type:
            info['school_type'] = str(i)
            q_list.append('typ')
        else:
            q_list.append(str(i))

    stop_words1,stop_words0=stopWords()
    # 去一般停用词：
    q1 = [i for i in q_list if i not in stop_words1]
    # 去含代名词的停用词：
    q0 = [i for i in q_list if i not in stop_words0]
    return q1, q0,info

def question(first=None,second=None,noUnderstand=None,search=None,ismajor=None,province=None):
    if first:
        answer='初次见面，我是高小智。有什么可以帮助您呢？'
    elif second:
        answer ='亲爱的亲，我们又见面了，我是高小智。还需要我为您做什么呢？'
    elif noUnderstand:
        answer ='我不是很明白你的意思呢，可以重新表达你的意思么？'
    elif search:
        answer='''我猜您是想咨询如下信息：<br>1.历年国家线查询<br>2.历年高校录取线查询<br>3.历年高校专业录取线查询<br>
                       4.心仪高校的专业排名<br>5.心仪专业的全国排名<br>6.心仪高校的全国排名<br>7.心仪高校的基本信息<br>
                       请选择您是想请选择您想要查询的信息代号（如：1）：'''
    elif ismajor:
        answer='''美国波士顿大学教授弗兰克·帕森斯提出人与职业相匹配是职业选择的关键。他认为，每个人都有自己独特的人格模式，
                         包括能力倾向、兴趣、价值观和人格等包括能力倾向、兴趣、价值观和人格等。这些人格特质都是可以通过心理测量工
                         具来加以评量，并能考察工作因素匹配人格。这样就使得专业规划真正成为一门科学。<br>基于此，美国心理学家布里格斯
                         和迈尔斯母女提出了格斯类型指标（MBTI），该类型指标能充分结合人的性格和兴趣，做出专业和职业推荐。
                         该指标现已成为许多公司招聘人才的重要参考，具有巨大意义。<br>
                         如果您已确定了解了MBTI基本规则，回复“y”进入点击进入MBTI专业测试：
                         '''
    elif province:
        answer='您是否有想去的目标省份呢？'
    else:
        answer ='还有什么可以帮助您呢？'
    return answer

#明白了用户需求以后，看用户是否缺失必要信息
def isLack(roads,category,info):
    print(roads,category,info)
    lack = []
    print(roads[category].keys())
    for r in roads[category].keys():  # roads[category]是字典{'school': '高校名字',...}
        if r not in info.keys():
            print('{}缺乏'.format(r))
            lack.append(r)
    if len(lack)>0:
        return True
    else:return False

#让用户填充缺失的信息
def getLack(roads,category,info):
    print(roads,category,info)
    lack = []
    for r in roads[category].keys():  # roads[category]是字典{'school': '高校名字',...}
        if r not in info.keys():
            lack.append(roads[category][r])
    input_lack=''
    print(lack)
    for i in range(len(lack)):#['province', 'classy']
        if i == 0:
            input_lack = input_lack + lack[i]
        else:
            input_lack = input_lack + '，' + lack[i]
    print(input_lack)
    # '请规范补充您的以下信息以便更精准的信息反馈哦：{}'.format(input_lack)
    question_lack='您的{}是什么呢'.format(input_lack)
    return question_lack

def lack(roads,category,info):
    while isLack(roads,category,info):
        question_lack = getLack(roads, category, info)  # stop_words1, stop_words0 = stopWords()
        seg_lack = getJieba(question_lack.strip(), custom=True)  # 分词
        if 'ismajor' in roads[category].keys():
            if '无' in seg_lack:
                info['ismajor']='无'
        if 'want_province' in roads[category].keys():
            if '无' in seg_lack:
                info['want_province']='无'
        q1, q0, info = getPronoun(seg_lack)
        lack(roads,category,info)
    return info

#历史数据查询
def searchInfo(category_search,info):

    if category_search == '历年国家线查询':#1.历年国家线查询
        # info=lack(roads_search, category_search, info)
        print('hhh')
        # data = nation_line.objects.all()
        # data=nation_line.objects.filter(province__in=['四川'],classy='文科',batch='第一批')
        data=nation_line.objects.filter(province__in=info['province'],classy=info['classy'])#字段信息从info提取#,batch=info['batch']
        print(data)
    elif category_search == '历年高校录取线查询':#2.历年高校录取线查询
        data=pro_univer_point.objects.filter(school=info['school'],province=info['province'],classy=info['classy'])
    elif category_search == '历年高校专业录取线查询':#3.历年高校专业录取线查询
        data = univer_major_line.objects.filter(school=info['shool'], province=info['province'],
                                                       classy=info['classy'], major=info['major'])
    elif category_search == '心仪高校的专业排名':#4.心仪高校的专业排名
        data = univer_major_line.objects.filter(school=info['shool'], major=info['major'])#??
    elif category_search == '心仪专业的全国排名':#5.心仪专业的全国排名
        data = univer_major_line.objects.filter(major=info['major'],)#??
    elif category_search == '心仪高校的全国排名':#6.心仪高校的全国排名
        data =univer_major_line.objects.filter(school=info['school'], )#??
    elif category_search == '心仪高校的基本信息':#7.心仪高校的基本信息
        data =univer_info.objects.filter(school=info['school'], )
    # else:answer = question(noUnderstand=True)
    print(data)
    data_df=pd.DataFrame.from_records(data.values())
    print(data_df)
    data_dict=data_df.to_dict(orient='records')#records是[{},{}]格式#index是包含index的{{}，{}}
    print(data_dict)
    #[{'id': 93, 'batch': '第一批', 'province': '四川', 'classy': '文科', 'year': 2017, 'line': 537}, {'id': 94, 'batch':
    data_str_outer=''#问答系统是span,span是文本不能很好的嵌入像table这样的结构，所以只好嵌入文本
    for index,each in enumerate(data_dict):
        c = 0
        data_str_inner = ""
        for k,v in each.items():
            if k!='id' :
                if c!=len(each)-2:
                    data_str_inner+=str(v)+'&nbsp;'*(6-len(str(v)))#乘以负的会为空！
                if c==len(each)-2:#还去掉了id
                     data_str_inner += str(v)
            c += 1
        data_str_inner+='<br>'
        data_str_outer+=data_str_inner
    print(data_str_outer)
    return data_str_outer

def theRecommend(category,info):
    if roads_recommend[category]=='推荐学校':
        return JsonResponse({'answer':'请问您是否有想去的目标省份呢？若有则输入<省份名称>、无则输入<无>'})
        q_wantProvince=input('请问您是否有想去的目标省份呢？若有则输入<省份名称>、无则输入<无>')
        seg_wantProvince = getJieba(q_wantProvince.strip())
        path = 'E:\web\myweb\myDic'
        province = [line.strip() for line in open(path + '生源地.txt', encoding='gb18030')]
        want_province=[]
        for i in seg_wantProvince:
            if str(i) in province:
                want_province.append(str(i))
        info['want_province']=want_province
        info=lack(roads_recommend,category,info)
        recommend_school(info)#?????????????????????????

    elif roads_recommend[category]=='推荐专业':
        # info=lack(roads_recommend,category,info)
        q=question(ismajor=True)
        if q=='y':
            return 'http:......' #?????????????????????????
        else:
            aqSystem(q)

    elif roads_recommend[category] == '学校及专业推荐':
        info=lack(roads_recommend,category,info)
        if info['ismajor']=='无':
            #您 的分数适合的学校有：
            # recommend_school()
            #关于专业可以跳转到专业测试链接
            return 'http....'
        else:
            print('niuzitong')
            # recommend_major&school()
    elif roads_recommend[category]=='是否能被录取':
        info = lack(roads_recommend, category, info)
        if 'major' not in info.keys():
            print('niuzitong')
            # recommend_school()
        else:
            print('niuzitong')

            # recommend_major&school()
    elif roads_recommend[category] == '专业前景':
        info = lack(roads_recommend, category, info)
        return 'sql....'
    elif roads_recommend[category]=='复读':
        info = lack(roads_recommend, category, info)
        return '一些需要考虑的现实问题'
    else:
        q = question(noUnderstand=True)
        aqSystem(q)
    print(info)

def questionCategory(q):#导出word2vec和xgb模型判断类别  #不能把参数取作和函数名一样的
    import os
    import xgboost as xgb
    stop_words1, stop_words0 = stopWords()
    if os.path.exists('model.model'):
        model=word2vec.KeyedVectors.load('model.model')
        q_ = [i for i in jieba.cut(q.strip()) if i not in stop_words1]
        vector = np.array([model[word] for word in (model.wv.vocab)])  # 词向量#"word '查' not in vocabulary"
        print(vector)  # [[],[],...]5*100
        doc_vector = []  # 文档向量
        l = [0] * len(vector[0])
        l_count = 0
        for word in q_:
            if word in model.wv.vocab:
                l += model[word]
                l_count += 1
        doc_vector.append(np.array(l) / l_count)
        xg_question = xgb.DMatrix(doc_vector)
        bst = xgb.Booster()  # {'nthread': 4}
        bst.load_model('xgb_word2vec.model')
        pred = bst.predict(xg_question);
        dict_lb2=pickle.load(open('G:\MY\gaokao\web\myweb\dict_lb2.pickle','rb'))
        print('dict_lb2',dict_lb2)
        category=dict_lb2[pred[0]]
        print('category',category)
    else:
        train_x, train_y, test_x, test_y, pred_y, dict_lb1, dict_lb2 = xgb_word2vec(q)
        category=xgbModel(train_x, train_y, test_x, test_y, pred_y, dict_lb1, dict_lb2, filename='xgb_word2vec')
    return category

# from jpype import *
# def getHanLP(comment,enable=None):
#     print("0",comment)
#     if not isJVMStarted():##特别重要  如果多方启动JAVA 虚拟机，会造成线程阻塞
#         startJVM(getDefaultJVMPath(), "-Djava.class.path=F:\hanlp\hanlp-1.2.8.jar;F:\hanlp", "-Xms1g", "-Xmx1g")
#     HanLP = JClass('com.hankcs.hanlp.HanLP')
#     Config = JClass('com.hankcs.hanlp.HanLP$Config')
#     if enable == False:
#         Config.ShowTermNature = False
#     else:
#         Config.ShowTermNature = True
#     seg_comment = HanLP.segment(comment)
#     # shutdownJVM()
#     return seg_comment


# def aqSystem(q):
#     print('---',)
#     print(info)      #'search': 'category':  'category0':'province': 'classy':
#     if 'category' not in info.keys() and 'search' not in info.keys():
#         print('000')
#         category, info1 = questionCategory(q)
#         info.update(info1)
#         info['category'] = category
#     category = '历史数据查询'
#     info['category'] = category#TODO 删
#     # if request.session['info']:
#     #     info=request.session['info']
#     if info['category'] == '历史数据查询' and 'search' not in info.keys():
#         search = question(search=True)
#         info['select'] = 'y'
#         print('info', info)
#         # request.session['info']=info
#         return info,search
#     elif info['category'] =='历史数据查询' and 'select' in info.keys():
#         print('111')
#         if 'lack' not in info.keys():
#             category0=search_dict[q]
#             info['category0']=category0
#             print(search_dict, category0)
#             # print(isLack(roads_search, category0, info))
#         elif 'lack'  in info.keys():
#             seg_lack = getJieba(q.strip(),)  # 分词#q可能是选项，也可能是lack补充
#             # if 'ismajor' in roads[category].keys():
#             #     if '无' in seg_lack:
#             #         info['ismajor'] = '无'
#             q1, q0, info = getPronoun(seg_lack)
#             # lack(roads, category, info)
#         while isLack(roads_search, info['category0'], info):
#             print('222')
#             question_lack = getLack(roads_search, category0, info)  # stop_words1, stop_words0 = stopWords()
#             info['lack'] = 'y'
#             print('333')
#             return info,question_lack
#
#
#         # info1 = searchInfo(q, category, info)  # 用户输入的q就是想要查询的内容：search=q
#         # print('info<br>', info)
#         # info.update(info1)
#         # return info,info
#     else:
#         print('else')
#         # return theRecommend(category,info)