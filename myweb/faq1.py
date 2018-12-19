from jpype import *
import pymysql,jieba,json
from gensim.models import word2vec
import numpy as np
import pandas as pd
from django.db import models
from django.http import JsonResponse# from myweb import recommend

# from . import  models
startJVM(getDefaultJVMPath(), "-Djava.class.path=F:\hanlp\hanlp-1.2.8.jar;F:\hanlp", "-Xms1g", "-Xmx1g")
# 启动JVM，Linux需替换分号;为冒号:

def getHanLP(comment,enable=None):
    HanLP = JClass('com.hankcs.hanlp.HanLP')
    Config = JClass('com.hankcs.hanlp.HanLP$Config')
    if enable== False:
        Config.ShowTermNature = False
    else:
        Config.ShowTermNature = True
    seg_comment=HanLP.segment(comment)
    return seg_comment

def stopWords():
    #一般停用词
    f1 = open('stop_words.txt')
    stop_words1 = [line.strip() for line in f1]
    stop_words1.extend(['刘老师', '黄老师', '老师', '您好', '你好', '\\t', '\\', 't', '高考','请问',' '])
    #加入高考代名词的停用词：
    stop_words0 = stop_words1.copy();#stop_words0 = stop_words1会使两个list完全相同
    stop_words0.extend(['sch', 'pro', 'cla', 'bat', 'maj', 'pnt', 'lx', 'typ', 'rk', 'lsy'])
    return  stop_words1,stop_words0
# django里面的import和一般的python文件不一样么
def getPronoun(q):#代名词:q是分词去停用词之后的
    path = 'G:/MY/gaokao/web/myweb/myDic/'
    major = [line.strip() for line in open(path + '/专业.txt', encoding='gb18030')]
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
            info['province'] = str(i)
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

def question(first=None,second=None,noUnderstand=None,search=None,ismajor=None):
    answer={}
    if first:
        answer['answer'] ='初次见面，我是高小智。有什么可以帮助您呢？'
    elif second:
        answer['answer'] ='亲爱的亲，我们又见面了，我是高小智。还需要我为您做什么呢？'
    elif noUnderstand:
        answer['answer'] ='我不是很明白你的意思呢，可以重新表达你的意思么？'
    elif search:
        answer['answer'] ='''我猜您是想咨询如下信息：\n1.历年国家线查询\n2.历年高校录取线查询\n3.历年高校专业录取线查询\n
                       4.心仪高校的专业排名\n5.心仪专业的全国排名\n6.心仪高校的全国排名\n7.心仪高校的基本信息\n
                       请选择您是想请选择您想要查询的信息代号（如：1）：'''
    elif ismajor:
        answer['answer']='''美国波士顿大学教授弗兰克·帕森斯提出人与职业相匹配是职业选择的关键。他认为，每个人都有自己独特的人格模式，
                         包括能力倾向、兴趣、价值观和人格等包括能力倾向、兴趣、价值观和人格等。这些人格特质都是可以通过心理测量工
                         具来加以评量，并能考察工作因素匹配人格。这样就使得专业规划真正成为一门科学。\n基于此，美国心理学家布里格斯
                         和迈尔斯母女提出了格斯类型指标（MBTI），该类型指标能充分结合人的性格和兴趣，做出专业和职业推荐。
                         该指标现已成为许多公司招聘人才的重要参考，具有巨大意义。\n
                         如果您已确定了解了MBTI基本规则，回复“y”进入点击进入MBTI专业测试：
                         '''
    else:
        answer['answer'] ='还有什么可以帮助您呢？'
    return JsonResponse(answer)

#明白了用户需求以后，看用户是否缺失必要信息
def isLack(roads,category,info):
    lack = []
    for r in roads[category].keys():  # roads[category]是字典{'school': '高校名字',...}
        if r not in info.keys():
            lack.append(r)
    if len(lack)>0:
        return True
    else:return False

#让用户填充缺失的信息
def getLack(roads,category,info):
    lack = []
    for r in roads[category].keys():  # roads[category]是字典{'school': '高校名字',...}
        if r not in info.keys():
            lack.append(r)
    input_lack=''
    for i in len(lack):
        if i==0:
            input_lack+=lack[i]
        elif i==len(lack)-1:
            input_lack+=lack[i]
        else:
            input_lack='，'+lack[i]
    question_lack='请规范补充您的以下信息以便更精准的信息反馈哦：{}'.format(input_lack)
    return JsonResponse({'answer':question_lack})

def lack(roads,category,info):
    while isLack(roads,category,info):
        question_lack = getLack(roads, category, info)  # stop_words1, stop_words0 = stopWords()
        seg_lack = getHanLP(question_lack.strip(), enable=False)  # 分词
        if 'ismajor' in roads[category].keys():
            if '无' in seg_lack:
                info['ismajor']='无'
        q1, q0, info = getPronoun(seg_lack)
        lack(roads,category,info)
    return info

#历史数据查询
def searchInfo(search,category,info):
    roads_search = {'历年高校专业录取线查询':{'school':'高校名称','province':'生源地（省份）','classy':'科类（文/理科）','major':'专业名称'},
                    '历年高校录取线查询':{'school': '高校名称', 'province': '生源地（省份）', 'classy': '科类（文/理科）'},
                    '历年国家线查询':{'province': '生源地（省份）', 'classy': '科类（文/理科）'},#不要批次
                    '心仪高校的专业排名':{'school': '高校名称', 'major': '专业名称'},#rank:排名？
                    '心仪专业的全国排名':{'major': '专业名称'},# 'rank': ''
                    '心仪高校的全国排名':{'school': '高校名称'},
                    '心仪高校的基本信息':{'school': '高校名称'}
                    }
    if search == '1':#1.历年国家线查询
        info=lack(roads_search, category, info)
        data=models.nation_line.objects.filter(province=info['province'],classy=info['classy'])#字段信息从info提取
    elif search == '2':#2.历年高校录取线查询
        info = lack(roads_search, category, info)
        data=models.pro_univer_point.objects.filter(school=info['school'],province=info['province'],classy=info['classy'])
    elif search == '3':#3.历年高校专业录取线查询
        info = lack(roads_search, category, info)
        data = models.univer_major_line.objects.filter(school=info['shool'], province=info['province'],
                                                       classy=info['classy'], major=info['major'])

    elif search == '4':#4.心仪高校的专业排名
        info = lack(roads_search, category, info)
        data = models.univer_major_line.objects.filter(school=info['shool'], major=info['major'])#??
    elif search == '5':#5.心仪专业的全国排名
        info = lack(roads_search, category, info)
        data = models.univer_major_line.objects.filter(major=info['major'],)#??
    elif search == '6':#6.心仪高校的全国排名
        info = lack(roads_search, category, info)
        data =models.univer_major_line.objects.filter(school=info['school'], )#??
    elif search == '7':#7.心仪高校的基本信息
        info = lack(roads_search, category, info)
        data =models.univer_info.objects.filter(school=info['school'], )
    else:
        answer = question(noUnderstand=True)
    data_df=pd.DataFrame.from_records(data.values())
    data_dict=data_df.to_dict(orient='records')
    # answer_json=json.dumps(answer)
    return JsonResponse({'answer':data_dict})
    # aqSystem(q)

def theRecommend(category,info):
    roads_recommend={
            '推荐学校': {'province':'生源地（省份）','classy':'科类（文/理科）','point':'分数','want_province': '想去的省份'},
             '推荐专业': {},
             '学校及专业推荐': {'province':'生源地（省份）','classy':'科类（文/理科）','point':'分数',
                         'isMajor':'是否有专业目标(若有则输入<专业名称>、无则输入<无>）'},#y/n
             '是否能被录取': {'school':'高校名称','province':'生源地（省份）','classy':'科类（文/理科）','point':'分数'},#专业可有可无
             '专业前景': {'major': '专业名称'},
             '复读': { },
             }
    if roads_recommend[category]=='推荐学校':
        return JsonResponse({'answer':'请问您是否有想去的目标省份呢？若有则输入<省份名称>、无则输入<无>'})
        q_wantProvince=input('请问您是否有想去的目标省份呢？若有则输入<省份名称>、无则输入<无>')
        seg_wantProvince = getHanLP(q_wantProvince.strip(), enable=False)
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
            recommend_school()
            #关于专业可以跳转到专业测试链接
            return 'http....'
        else:
            recommend_major&school()
    elif roads_recommend[category]=='是否能被录取':
        info = lack(roads_recommend, category, info)
        if 'major' not in info.keys():
            recommend_school()
        else:
            recommend_major&school()
    elif roads_recommend[category] == '专业前景':
        info = lack(roads_recommend, category, info)
        return 'sql....'
    elif roads_recommend[category]=='复读':
        info = lack(roads_recommend, category, info)
        return '一些需要考虑的现实问题'
    else:
        q = question(noUnderstand=True)
        aqSystem(q)

def aqSystem(q):
    # value = request.GET["value"]
    #------------------以下这俩字典只要重新跑word2vec 和xgb的模型就会改变，所以要实时更正-------------------------
    dict_lb1={'学校及专业推荐': 4,'推荐学校': 3,'推荐专业': 1, '历史数据查询': 2,      '专业前景': 0, '是否能被录取': 5, '复读': 6}
    dict_lb2={0: '专业前景', 1: '推荐专业', 2: '历史数据查询', 3: '推荐学校', 4: '学校及专业推荐', 5: '是否能被录取', 6: '复读'}
    #-----------------------------------------------------------------------------------------------------------
    pred,info=questionCategory(q)
    print(pred,'\n',info)
    category=dict_lb2[pred]

    #历史数据查询
    print(category)
    if category=='历史数据查询':
        search=question(search=True)
        info=searchInfo(search, category, info)
        print('info\n',info)
        return info
    else:
        print('else')
        return theRecommend(category,info)

    # return
# def aq(request):
#     aqSystem(value)
#     ## TODO 写算法
#     return JsonResponse({"anwser":anwser})



def questionCategory(q):#导出word2vec和xgb模型判断类别  #不能把参数取作和函数名一样的
    print(q)
    #使用hanlp自定义字典分词
    model = word2vec.KeyedVectors.load('model.model')
    #把词向量转成成文本向量
    stop_words1, stop_words0=stopWords()
    seg_question=getHanLP(q.strip(),enable=False)#分词
    print(seg_question)
    q1,q0,info=getPronoun(seg_question)#q1含代名词,q0不含
    print(q1,q0,info)
    # seg_question=[str(word) for word in seg_question if word not in stop_words1]#去停用词
    q_ = [i for i in jieba.cut(q.strip())if i not in stop_words1]
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
    # vector = np.array([model[str(word)] for word in q1 if word in model.wv.vocab])#词向量#"word '查' not in vocabulary"
    # print(vector)#[[],[],...]5*100
    # doc_vector = []#文档向量
    # l = [0] * len(vector[0])
    # l_count=0
    # for v in vector:
    #     l+=v
    #     l_count += 1
    # doc_vector.append(np.array(l) / l_count)
    doc_vector_df = pd.DataFrame(doc_vector)
    doc_vector_df.fillna(0, inplace=True)
    # print('doc_vector_df_0\n', doc_vector_df)
    doc_vector = np.array(doc_vector_df)
    print('doc_vector\n', doc_vector)

    import xgboost as xgb
    xg_question=xgb.DMatrix(doc_vector)
    bst=xgb.Booster()#{'nthread': 4}
    bst.load_model('xgb_word2vec.model')
    pred = bst.predict(xg_question);
    return pred[0],info

if __name__=="__main__":
    aqSystem('发挥失常要复读么')#这个怎么不准呢
    # model = word2vec.KeyedVectors.load('model.model')
    # vector = np.array([model[word] for word in (model.wv.vocab)])
    # print(vector)
    # aqSystem()


    # a,b,c,d,e,g,h,i,j=range(9)#去掉了f
    # N=[
    #     {b,i,h,g,j},{a,c},{e,d},{c,e,g},{},{a},{a},{a},{a}
    # ]#N(v) 代表的是 v 的邻居节点集；

    # params = {
    #     'enableCustomDic': True,
    #     'enablePOSTagging': False,
    #     'convertMode': '2tw'
    # }
    # roads = {'高校专业录取线': [school0, province0, classy0, major0, line0],
    #          '高校录取线': [school0, province0, classy0, line0],
    #          '国家线': [province0, classy0, line0],
    #          '某高校专业排名': [school0, major0, rank0],
    #          '总体专业排名': [major0, rank0],
    #          '总体高校排名': [rank0],
    #          # '某高校排名':[school0,rank0],
    #          '某高校基本信息': [[school0, rank0], [school0, lishuyu], [school0, leixing0], [school0, school_type0], ]
    #          }