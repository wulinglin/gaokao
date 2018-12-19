#该部分是学校相似度部分
from django.shortcuts import render,HttpResponse
from . import models
import numpy as np
import pandas as pd
pd.set_option('expand_frame_repr',False)#不允许换行#https://www.cnblogs.com/yesuuu/p/6100714.html
pd.set_option('display.max_columns', 120)
np.set_printoptions(threshold=120,)#'nan'#通过set_printoptions来强制NumPy打印所有数据。
#  http://blog.csdn.net/GZHermit/article/details/72716619
import logging,json
logger=logging.getLogger('sourceDns.webdns.views')
from pandas import Series,DataFrame
from .gm import Identification_Algorithm,GM_Model,Return,GM_gbnn
import math,pymysql
from .bpnn import *
from . import  models
from collections import defaultdict
from gensim import corpora,models,similarities
from .dynamicCritiquing import *

def similarity(request):
    province = request.GET.get('province')
    school = request.GET.get('school')
    classy=request.GET.get('classy')
    sql_univer_info='''
    select b.school,
		b.major	
from myweb_univer_info a	
left join 
(select school,group_concat(major_new,',',classes,',',catalogue Separator ',') as major from myweb_univer_major_line_lb  where year =2015 and school is not null  group by school ) b 
on a.school=b.school; 
'''
    sql_univer_line='''
        select c.school_a,c.school_b,c.province,c.classy,c.batch,count(*) cnt,abs(round(avg(d_d_a),2))  from 
        (select a.school school_a,b.school school_b,a.province,a.classy,a.year,a.batch,a.d_a d_a_a,b.d_a d_a_b,(a.d_a-b.d_a)d_d_a	
        from myweb_pro_univer_nation_line a
        left join myweb_pro_univer_nation_line b
        on  a.province =b.province  and a.classy=b.classy and a.year=b.year and a.batch=b.batch
        where a.school ='{}' and a.province='{}' and a.classy='{}' and b.school !='{}' )c
        group by c.school_a,c.school_b,c.province,c.classy,c.batch;
    '''.format(school,province,classy,school)#province是否加入？加入可能会运行速度更快，但是也需要用户提供更多的信息才可,且可能有更多的局限数据！
    sql_major_ratio='select * from major_ratio'
    sql_school_major_ratio='''
    select school,
	sum(case catalogue when '哲学类' then  d.cnt/d.cnt_all else 0 end ) as cat_1,
	sum(case catalogue when '预防医学类' then  d.cnt/d.cnt_all else 0 end ) as cat_2,
	sum(case catalogue when '药学类' then  d.cnt/d.cnt_all else 0 end ) as cat_3,
	sum(case catalogue when '生物科学类' then  d.cnt/d.cnt_all else 0 end ) as cat_4,
	sum(case catalogue when '临床医学与医学技术类' then  d.cnt/d.cnt_all else 0 end ) as cat_5,
	sum(case catalogue when '口腔医学类' then  d.cnt/d.cnt_all else 0 end ) as cat_6,
	sum(case catalogue when '基础医学类' then  d.cnt/d.cnt_all else 0 end ) as cat_7,
	sum(case catalogue when '化工与制药类' then  d.cnt/d.cnt_all else 0 end ) as cat_8,
	sum(case catalogue when '法医学类' then  d.cnt/d.cnt_all else 0 end ) as cat_9,
	sum(case catalogue when '法学类' then  d.cnt/d.cnt_all else 0 end ) as cat_10,
	sum(case catalogue when '动物医学类' then  d.cnt/d.cnt_all else 0 end ) as cat_11,
	sum(case catalogue when '电气信息类' then  d.cnt/d.cnt_all else 0 end ) as cat_12,
	sum(case catalogue when '护理学类' then  d.cnt/d.cnt_all else 0 end ) as cat_13,
	sum(case catalogue when '艺术类' then  d.cnt/d.cnt_all else 0 end ) as cat_14,
	sum(case catalogue when '中国语言文学类' then  d.cnt/d.cnt_all else 0 end ) as cat_15,
	sum(case catalogue when '新闻传播学类' then  d.cnt/d.cnt_all else 0 end ) as cat_16,
	sum(case catalogue when '外国语言文学类' then  d.cnt/d.cnt_all else 0 end ) as cat_17,
	sum(case catalogue when '设计类' then  d.cnt/d.cnt_all else 0 end ) as cat_18,
	sum(case catalogue when '历史学类' then  d.cnt/d.cnt_all else 0 end ) as cat_19,
	sum(case catalogue when '交通运输类' then  d.cnt/d.cnt_all else 0 end ) as cat_20,
	sum(case catalogue when '工商管理类' then  d.cnt/d.cnt_all else 0 end ) as cat_21,
	sum(case catalogue when '植物生产类' then  d.cnt/d.cnt_all else 0 end ) as cat_22,
	sum(case catalogue when '水产类' then  d.cnt/d.cnt_all else 0 end ) as cat_23,
	sum(case catalogue when '森林资源类' then  d.cnt/d.cnt_all else 0 end ) as cat_24,
	sum(case catalogue when '环境与安全类' then  d.cnt/d.cnt_all else 0 end ) as cat_25,
	sum(case catalogue when '环境生态类' then  d.cnt/d.cnt_all else 0 end ) as cat_26,
	sum(case catalogue when '动物生产类' then  d.cnt/d.cnt_all else 0 end ) as cat_27,
	sum(case catalogue when '草业科学类' then  d.cnt/d.cnt_all else 0 end ) as cat_28,
	sum(case catalogue when '电子信息科学类' then  d.cnt/d.cnt_all else 0 end ) as cat_29,
	sum(case catalogue when '数学类' then  d.cnt/d.cnt_all else 0 end ) as cat_30,
	sum(case catalogue when '物理学类' then  d.cnt/d.cnt_all else 0 end ) as cat_31,
	sum(case catalogue when '地球物理学类' then  d.cnt/d.cnt_all else 0 end ) as cat_32,
	sum(case catalogue when '材料科学类' then  d.cnt/d.cnt_all else 0 end ) as cat_33,
	sum(case catalogue when '心理学类' then  d.cnt/d.cnt_all else 0 end ) as cat_34,
	sum(case catalogue when '统计学类' then  d.cnt/d.cnt_all else 0 end ) as cat_35,
	sum(case catalogue when '力学类' then  d.cnt/d.cnt_all else 0 end ) as cat_36,
	sum(case catalogue when '环境科学类' then  d.cnt/d.cnt_all else 0 end ) as cat_37,
	sum(case catalogue when '化学类' then  d.cnt/d.cnt_all else 0 end ) as cat_38,
	sum(case catalogue when '海洋科学类' then  d.cnt/d.cnt_all else 0 end ) as cat_39,
	sum(case catalogue when '地质学类' then  d.cnt/d.cnt_all else 0 end ) as cat_40,
	sum(case catalogue when '地理科学类' then  d.cnt/d.cnt_all else 0 end ) as cat_41,
	sum(case catalogue when '大气科学类' then  d.cnt/d.cnt_all else 0 end ) as cat_42,
	sum(case catalogue when '政治学类' then  d.cnt/d.cnt_all else 0 end ) as cat_43,
	sum(case catalogue when '武器类' then  d.cnt/d.cnt_all else 0 end ) as cat_44,
	sum(case catalogue when '土建类' then  d.cnt/d.cnt_all else 0 end ) as cat_45,
	sum(case catalogue when '航空航天类' then  d.cnt/d.cnt_all else 0 end ) as cat_46,
	sum(case catalogue when '公安学类' then  d.cnt/d.cnt_all else 0 end ) as cat_47,
	sum(case catalogue when '公安技术类' then  d.cnt/d.cnt_all else 0 end ) as cat_48,
	sum(case catalogue when '测绘类' then  d.cnt/d.cnt_all else 0 end ) as cat_49,
	sum(case catalogue when '材料类' then  d.cnt/d.cnt_all else 0 end ) as cat_50,
	sum(case catalogue when '经济学类' then  d.cnt/d.cnt_all else 0 end ) as cat_51,
	sum(case catalogue when '职业技术教育类' then  d.cnt/d.cnt_all else 0 end ) as cat_52,
	sum(case catalogue when '体育学类' then  d.cnt/d.cnt_all else 0 end ) as cat_53,
	sum(case catalogue when '教育学类' then  d.cnt/d.cnt_all else 0 end ) as cat_54,
	sum(case catalogue when '职业技术教育' then  d.cnt/d.cnt_all else 0 end ) as cat_55,
	sum(case catalogue when '水利类' then  d.cnt/d.cnt_all else 0 end ) as cat_56,
	sum(case catalogue when '轻工纺织食品类' then  d.cnt/d.cnt_all else 0 end ) as cat_57,
	sum(case catalogue when '农业经济管理类' then  d.cnt/d.cnt_all else 0 end ) as cat_58,
	sum(case catalogue when '旅游类' then  d.cnt/d.cnt_all else 0 end ) as cat_59,
	sum(case catalogue when '机械类' then  d.cnt/d.cnt_all else 0 end ) as cat_60,
	sum(case catalogue when '管理科学与工程类' then  d.cnt/d.cnt_all else 0 end ) as cat_61,
	sum(case catalogue when '公共管理类' then  d.cnt/d.cnt_all else 0 end ) as cat_62,
	sum(case catalogue when '农业工程类' then  d.cnt/d.cnt_all else 0 end ) as cat_63,
	sum(case catalogue when '海洋工程类' then  d.cnt/d.cnt_all else 0 end ) as cat_64,
	sum(case catalogue when '自动化类' then  d.cnt/d.cnt_all else 0 end ) as cat_65,
	sum(case catalogue when '能源动力类' then  d.cnt/d.cnt_all else 0 end ) as cat_66,
	sum(case catalogue when '机械类' then  d.cnt/d.cnt_all else 0 end ) as cat_67,
	sum(case catalogue when '地矿类' then  d.cnt/d.cnt_all else 0 end ) as cat_68,
	sum(case catalogue when '食品科学与工程类' then  d.cnt/d.cnt_all else 0 end ) as cat_69,
	sum(case catalogue when '仪器仪表类' then  d.cnt/d.cnt_all else 0 end ) as cat_70,
	sum(case catalogue when '生物工程类' then  d.cnt/d.cnt_all else 0 end ) as cat_71,
	sum(case catalogue when '林业工程类' then  d.cnt/d.cnt_all else 0 end ) as cat_72,
	sum(case catalogue when '核工程类' then  d.cnt/d.cnt_all else 0 end ) as cat_73,
	sum(case catalogue when '工程力学类' then  d.cnt/d.cnt_all else 0 end ) as cat_74,
	sum(case catalogue when '社会学类' then  d.cnt/d.cnt_all else 0 end ) as cat_75
from 
(select a.school,a.catalogue,a.cnt,b.cnt_all from 
(select school,catalogue,count(*) cnt   from 	myweb_univer_major_line_lb c where  c.year in (2015,2016,2017)  group by school ,catalogue  ) a
left join (select school,count(*) cnt_all from 	myweb_univer_major_line_lb group by school) b
on a.school=b.school where a.school ='{}') d group by d.school;
    '''.format(school)
    sql_univer_info3='''
    select a.school,a.leixing_int,a.province,a.provincial_capital,a.development,a.school_type,
		case 	when lishuyu in ('-----') then 1
				when lishuyu in ('教育部') then 3 
				when lishuyu in ('上海市教育委员会','中华全国妇女联合会','中华全国总工会','中国共产主义青年团中央','中国地震局',
	 						'中国民用航空总局','中国科学院', '交通运输部', '公安部', '北京市教育委员会', '卫生部', '司法部',
							 '国务院侨务办公室', '国家体育总局','国家安全生产监督管理局', '国家民族事务委员会', '外交部', '天津市教育委员会', 
							 '工业与信息化部', '新疆生产建设兵团', '重庆市教育委员会') then 3
				when lishuyu in ('云南省教育厅','内蒙古自治区教育厅','吉林省教育厅','四川省教育厅','宁夏回族自治区教育厅','安徽省教育厅',
					 '山东省教育厅','山西省教育厅','广东省教育厅','广西壮族自治区教育厅','新疆维吾尔自治区教育厅','江苏省教育厅',
					 '江西省教育厅','河北省教育厅','河南省教育厅','浙江省教育厅','海南省教育厅','湖北省教育厅','湖南省教育厅',
					 '甘肃省教育厅','福建省教育厅','西藏自治区教育厅','贵州省教育厅','辽宁省教育厅','陕西省教育厅','青海省教育厅',
					 '黑龙江省教育厅') then 2
		   end lishuyu_int,
		case when province in ('山东','江苏','安徽','浙江','福建','上海') then '华东'  
				when  province in ('广东','广西','海南') then '华南'
				when  province in ('湖北','湖南','河南','江西') then '华中'
				when  province in ('北京','天津','河北','山西','内蒙古') then '华北'
				when  province in ('宁夏','新疆','青海','陕西','甘肃') then '西北'
				when  province in ('四川','云南','贵州','西藏','重庆') then '西南'
				when  province in ('辽宁','吉林','黑龙江') then '东北'
				end province_int	
		from myweb_univer_info a
		'''
    connect=pymysql.Connect(host='172.19.235.29',user='root',passwd='mingming',port=3306,
                            db='gaokao_py3',charset='utf8')
    cursor=connect.cursor()
    cursor.execute(sql_univer_info)
    univer_info_t=cursor.fetchall()#这里只得到元组类型
    cursor.execute(sql_univer_line)
    univer_line_t=cursor.fetchall()
    cursor.execute(sql_major_ratio)
    major_ratio=cursor.fetchall()
    cursor.execute(sql_school_major_ratio)
    school_major_ratio=cursor.fetchall()
    cursor.execute(sql_univer_info3)
    univer_info3_t = cursor.fetchall()
    cursor.close()

    univer_line_df=pd.DataFrame.from_records(list(univer_line_t))
    print(sql_univer_line,univer_line_df)
    univer_line_df=univer_line_df.groupby([0,1,2,3,], as_index=False)[4,6].agg({5: 'max'})
    univer_line_df=(univer_line_df[5]).sort_values(by=[6],ascending=True)#多重索引，[5]是一个包含0.1.2.3的dataframe#sort_values vs sort_index
    univer_line_df=univer_line_df[univer_line_df[6]<=13]
    univer_line_df=univer_line_df.copy()#A value is trying to be set on a copy of a slice from a DataFrame.
    univer_line_df['score']=(univer_line_df[6].max()-univer_line_df[6])/(univer_line_df[6].max() -univer_line_df[6].min())
    print(univer_line_df)#34875
    res_rank=pd.Series(univer_line_df['score'].values,index=univer_line_df[1].values)
    print('res_rank\n',res_rank)
    school_candidates=list(univer_line_df[1].values)#numpy.ndarray:iterrows里面是

    # def major_sim(major_ratio,school_major_ratio,school_candidates):
    major_ratio=pd.DataFrame.from_records(list(major_ratio))
    school_major_ratio=pd.DataFrame.from_records(list(school_major_ratio))
    major_ratio=major_ratio[major_ratio[0].isin(school_candidates)]
    print(major_ratio,'\n--------------\n',school_major_ratio)
    res_dict={}
    res_dict[school]=1
    for index,value in major_ratio.iterrows():#iteritems():行，iterrows():行
        # print(school_major_ratio.ix[0,1:],'\n-------------',value[1:],major_ratio.ix[index,1:],type(value[0]))
        score=np.sum(abs(school_major_ratio.ix[0,1:]-value[1:]))
        name=value[0]
        res_dict[name]=score
        # sorted(res_dict.items(), key=lambda d: d[1], reverse=True
    print(res_dict)
    res_major=pd.Series(res_dict)#list
    res_major.fillna(0,inplace=True)
    print('res_major\n',res_major)
    # res=res_major+res_rank
    print('res\n',res_major.sort_values(ascending=False))

    univer_info3_df = pd.DataFrame.from_records(list(univer_info3_t))
    c0=univer_info3_df[univer_info3_df[0]==school]#school_candidates可以是array也可以是list
    c1=univer_info3_df[univer_info3_df[0].isin(school_candidates)]
    ci=c0.append(c1)
    #index=range(1,len(df)+1)
    ci = pd.DataFrame(np.array(ci), index=ci[0].values)
    ci[8] = res_major
    ci[8].fillna(0,inplace=True)
    print('ci','\n',ci)
    ci = pd.DataFrame(np.array(ci), index=range(len(ci)))
    print(ci)
    # print(len([school]+school_candidates),request.session['q0'])
    r,sc=dynamicCritiquing(school,q=None, ci=ci, k=8, sigma=None)
    print(sc)
    school_candidates=[school] + school_candidates
    request.session['q0'] = school_candidates
    request.session['school'] = school#r?school?
    request.session['r0'] = r
    request.session['ci'] = json.loads(ci.to_json(orient='split'))["data"]
    return render(request, 'sim_dynamicCri.html', {'r': r, 'sc': sc})

def dynamicCritiquing(school,q, ci, k, sigma):
    if q == None:
        r = school
    else:
        r = itemRecommend(q, ci)
    sc = compoundCritiques(r, ci, k, sigma)
    # q = userReview(r, ci, cc)
    return r, sc

def dynamic_res(request):
    r0=request.session['r0'] #上一次推荐那个r
    r=request.GET.get('r')
    ci=request.session['ci']
    sc = request.GET.get('sc')
    q0 = request.session['q0'];
    school = request.session['school']  # r?school?
    if r:
        return HttpResponse(r)
    else:#sc!='':有新评价则重新产生动态评价
        ci_df0 = pd.DataFrame(np.array(ci))
        ci_df2=ci_df0[ci_df0[0].isin(q0)]
        ci = pd.DataFrame(np.array(ci_df2), index=range(len(ci_df2)))

        r, sc = dynamicCritiquing(school,q=sc, ci=ci, k=8, sigma=None)
        request.session['r0'] = r
        print(r0)
        print(q0)
        try:
            if r0!=school:
                q0.remove(r0);request.session['q0'] = q0
        except:
            pass
        return render(request, 'sim_dynamicCri.html', {'r': r, 'sc': sc})

        # sql_univer_info3 = '''
        #     select a.school,a.leixing,a.province,a.provincial_capital,a.development,a.school_type,
        #         case 	when lishuyu in ('-----') then 0
        #                 when lishuyu in ('教育部') then 1
        #                 when lishuyu in ('上海市教育委员会','中华全国妇女联合会','中华全国总工会','中国共产主义青年团中央','中国地震局',
        #                             '中国民用航空总局','中国科学院', '交通运输部', '公安部', '北京市教育委员会', '卫生部', '司法部',
        #                              '国务院侨务办公室', '国家体育总局','国家安全生产监督管理局', '国家民族事务委员会', '外交部', '天津市教育委员会',
        #                              '工业与信息化部', '新疆生产建设兵团', '重庆市教育委员会') then 2
        #                 when lishuyu in ('云南省教育厅','内蒙古自治区教育厅','吉林省教育厅','四川省教育厅','宁夏回族自治区教育厅','安徽省教育厅',
        #                      '山东省教育厅','山西省教育厅','广东省教育厅','广西壮族自治区教育厅','新疆维吾尔自治区教育厅','江苏省教育厅',
        #                      '江西省教育厅','河北省教育厅','河南省教育厅','浙江省教育厅','海南省教育厅','湖北省教育厅','湖南省教育厅',
        #                      '甘肃省教育厅','福建省教育厅','西藏自治区教育厅','贵州省教育厅','辽宁省教育厅','陕西省教育厅','青海省教育厅','黑龙江省教育厅') then 3
        #         end lishuyu_int,
        #         case when province in ('山东','江苏','安徽','浙江','福建','上海') then 1
        #                 when  province in ('广东','广西','海南') then 2
        #                 when  province in ('湖北','湖南','河南','江西') then 3
        #                 when  province in ('北京','天津','河北','山西','内蒙古') then 4
        #                 when  province in ('宁夏','新疆','青海','陕西','甘肃') then 5
        #                 when  province in ('四川','云南','贵州','西藏','重庆') then 6
        #                 when  province in ('辽宁','吉林','黑龙江') then 7
        #                 end province_int
        #         from myweb_univer_info a
        #         '''
        # connect = pymysql.Connect(host='172.19.235.29', user='root', passwd='mingming', port=3306,
        #                           db='gaokao_py3', charset='utf8')
        # cursor = connect.cursor()
        # cursor.execute(sql_univer_info3)
        # univer_info3_t=cursor.fetchall()
        # cursor.close()
        # univer_info3_df = pd.DataFrame.from_records(list(univer_info3_t))
        # c0=univer_info3_df[univer_info3_df[0]==school]
        # c1= univer_info3_df[univer_info3_df[0].isin(q0)]
        # ci=c0.append(c1)

        # def userReview(r, ci, cc):
        #     def critique(r, cc):
        #         return render('.html', {'r': r, 'cc': cc})
        #         cc = request.GET.get('cc')
        #         return cc
        #
        #     q = critique(r, cc)
        #
        #     ci = ci[q]
        #     return ci


    # univer_info3_df=pd.DataFrame.from_records(list(univer_info3_t))
    # print(univer_info3_df)

    # ei=univer_info3_df[univer_info3_df[0]==school]
    # ci=major_ratio[univer_info3_df[0].isin(school_candidates)]
    # # cp=


 #    data=pd.DataFrame.from_records(list(univer_info_t))##7大学名字：a.academician,a.doctor,a.master；a.provincial_capital,a.development,is_doctor,is_master,leixing_int
 # #重点方法：data.columns.difference(['0'])
 #        #没有字段名的时候用data.ix[:,0]；.difference([0]
 #    data_list = (data[data.ix[:, 0].isin(school_candidates)][data.columns.difference([0])].to_dict(orient='split'))['data']
 #    # data_list=(data[data.ix[:,0]!=school][data.columns.difference([0])].to_dict(orient='split'))['data']
 #    # py不可以对json使用键来提取的，to_json得到的是一个str;所以需先转化成字典
 #    print(data_list)
 #    data_list=[each[0].split(',') for each in data_list  if each[0] is not None]#6；+和append区别！
 #    data_school=(data[data.ix[:,0]!=school].ix[:,0]).values
 #    print(data_list)
 #    print(data_school)
 #    school_test=(data[data.ix[:,0]==school][data.columns.difference([0])].to_dict(orient='split'))['data']
 #    school_list=[each[0].split(',') for each in school_test  if each[0] is not None][0]
 #    print(school_list)
 #    texts=data_list
 #    # # # 忽略掉：2.计算词频# 选择频率大于1的词
 #    # 3.创建字典（单词与编号之间的映射）
 #    dictionary=corpora.Dictionary(texts)# Dictionary(12 unique tokens: ['time', 'computer', 'graph', 'minors', 'trees']...)
 #    # 打印字典，key为单词，value为单词的编号
 #    # print(dictionary.token2id)# {'human': 0, 'interface': 1, 'computer': 2,  ...}
 #    # 4.将要比较的文档转换为向量（词袋表示方法）
 #    # 将文档分词并使用doc2bow方法对每个不同单词的词频进行了统计，并将单词转换为其编号，然后以稀疏向量的形式返回结果
 #    new_vec = dictionary.doc2bow(school_list)## [[(0, 1), (2, 1)]
 #    # 5.建立语料库： 将每一篇文档转换为向量
 #    corpus = [dictionary.doc2bow(text) for text in texts]
 #    # 6.初始化一个tfidf模型,可以用它来转换向量（词袋整数计数）表示方法为新的表示方法（Tfidf 实数权重）
 #    tfidf = models.TfidfModel(corpus)
 #    # 将整个语料库转为tfidf表示方法
 #    corpus_tfidf = tfidf[corpus]
 #    print(corpus_tfidf)
 #    # 7.创建索引
 #    index = similarities.MatrixSimilarity(corpus_tfidf)#for i in :[ 0.01074398  0.00836747  0.00905864 ...,  0.2464096   0.23184729  1.]
 #    # 8.相似度计算
 #    new_vec_tfidf = tfidf[new_vec]  # 将要比较文档转换为tfidf表示方法[(0, 0.7071067811865476), (2, 0.7071067811865476)]
 #    # 计算要比较的文档与语料库中每篇文档的相似度
 #    sims = index[new_vec_tfidf]
 #    print(sims)#'numpy.ndarray':[ 0.81649655  0.31412902  0.
 #    # np.sort(sims,axis=1)#按每行中元素从小到大排序axis 1 is out of bounds for array of dimension 1
 #    sims_index= np.argsort(-sims)  # 求a从小到大排序的坐标:np.argsort(-x) #按降序排列
 #    print(sims_index)
 #    print(sims[sims_index])# 按求出来的坐标顺序排序
 #    res=data_school[sims_index]
 #    print(res[:10])
 #    return HttpResponse('hh')

