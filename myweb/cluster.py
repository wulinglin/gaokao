#该部分是学校聚类部分
from django.shortcuts import render,HttpResponse
from . import models
import numpy as np
import pandas as pd
pd.set_option('expand_frame_repr',False)#不允许换行#https://www.cnblogs.com/yesuuu/p/6100714.html
pd.set_option('display.max_columns', 120)
np.set_printoptions(threshold=120,)#'nan'#通过set_printoptions来强制NumPy打印所有数据。
#  http://blog.csdn.net/GZHermit/article/details/72716619

import logging
logger=logging.getLogger('sourceDns.webdns.views')
from pandas import Series,DataFrame
from .gm import Identification_Algorithm,GM_Model,Return,GM_gbnn
import math
from .bpnn import *

import sys,pymysql
from .fcm_cluster import *
def cluster(request):
    # 三引号的形式用来输入多行文本，也就是说在三引号之间输入的内容将被原样保留，之中的单号和双引号不用转义，其中的不可见字符比如 / n和 / t都会被保留，这样的好处是你可以替换一些多行的文本。
#     sql='''
# select a.leixing,a.lishuyu,a.province,a.city,a.school_type,
# 		case 	when lishuyu in ('-----') then '其它'
# 				when lishuyu in ('教育部') then '教育部'
# 				when lishuyu in ('上海市教育委员会','中华全国妇女联合会','中华全国总工会','中国共产主义青年团中央','中国地震局',
# 	 						'中国民用航空总局','中国科学院', '交通运输部', '公安部', '北京市教育委员会', '卫生部', '司法部',
# 							 '国务院侨务办公室', '国家体育总局','国家安全生产监督管理局', '国家民族事务委员会', '外交部', '天津市教育委员会',
# 							 '工业与信息化部', '新疆生产建设兵团', '重庆市教育委员会') then '其它中央部委'
# 				when lishuyu in ('云南省教育厅','内蒙古自治区教育厅','吉林省教育厅','四川省教育厅','宁夏回族自治区教育厅','安徽省教育厅',
# 					 '山东省教育厅','山西省教育厅','广东省教育厅','广西壮族自治区教育厅','新疆维吾尔自治区教育厅','江苏省教育厅',
# 					 '江西省教育厅','河北省教育厅','河南省教育厅','浙江省教育厅','海南省教育厅','湖北省教育厅','湖南省教育厅',
# 					 '甘肃省教育厅','福建省教育厅','西藏自治区教育厅','贵州省教育厅','辽宁省教育厅','陕西省教育厅','青海省教育厅','黑龙江省教育厅') then '地方所属'
# 		end lishuyu_int,
# 		case when province in ('山东','江苏','安徽','浙江','福建','上海') then '华东'
# 				when  province in ('广东','广西','海南') then '华南'
# 				when  province in ('湖北','湖南','河南','江西') then '华中'
# 				when  province in ('北京','天津','河北','山西','内蒙古') then '华北'
# 				when  province in ('宁夏','新疆','青海','陕西','甘肃') then '西北'
# 				when  province in ('四川','云南','贵州','西藏','重庆') then '西南'
# 				when  province in ('辽宁','吉林','黑龙江') then '东北'
# 				end province_int,
# 		a.academician,a.doctor,a.master,a.provincial_capital,a.development,a.is_doctor,a.is_master,a.leixing_int,b.*
# from myweb_univer_info a
# left join
# (select school,
# 		 sum(case classes when '哲学' then 1 else null end ) as cl_phi,
# 		 sum(case classes when '经济学' then 1  else null end ) as cl_eco,
# 		 sum(case classes when '法学' then 1 else null end ) as cl_law,
# 		 sum(case classes when '教育学' then 1 else null  end ) as cl_edu,
# 		 sum(case classes when '文学' then 1 else null  end ) as cl_art,
# 		 sum(case classes when '历史学' then 1 else null  end ) as cl_his,
# 		 sum(case classes when '理学' then 1  else null end ) as cl_sci,
# 		 sum(case classes when '工学' then 1 else null  end ) as cl_engi,
# 		 sum(case classes when '农学' then 1  else null end ) as cl_agri,
# 		 sum(case classes when '医学' then 1  else null end ) as cl_medi,
# 		 sum(case classes when '管理学' then 1  else null end ) as cl_mana,
# 		 sum(case classes when '军事学' then 1  else null end ) as cl_mili,
# 		 sum(case catalogue when '哲学类' then 1 else null end ) as cat_1,
# 		sum(case catalogue when '预防医学类' then 1 else null end ) as cat_2,
# 		sum(case catalogue when '药学类' then 1 else null end ) as cat_3,
# 		sum(case catalogue when '生物科学类' then 1 else null end ) as cat_4,
# 		sum(case catalogue when '临床医学与医学技术类' then 1 else null end ) as cat_5,
# 		sum(case catalogue when '口腔医学类' then 1 else null end ) as cat_6,
# 		sum(case catalogue when '基础医学类' then 1 else null end ) as cat_7,
# 		sum(case catalogue when '化工与制药类' then 1 else null end ) as cat_8,
# 		sum(case catalogue when '法医学类' then 1 else null end ) as cat_9,
# 		sum(case catalogue when '法学类' then 1 else null end ) as cat_10,
# 		sum(case catalogue when '动物医学类' then 1 else null end ) as cat_11,
# 		sum(case catalogue when '电气信息类' then 1 else null end ) as cat_12,
# 		sum(case catalogue when '护理学类' then 1 else null end ) as cat_13,
# 		sum(case catalogue when '艺术类' then 1 else null end ) as cat_14,
# 		sum(case catalogue when '中国语言文学类' then 1 else null end ) as cat_15,
# 		sum(case catalogue when '新闻传播学类' then 1 else null end ) as cat_16,
# 		sum(case catalogue when '外国语言文学类' then 1 else null end ) as cat_17,
# 		sum(case catalogue when '设计类' then 1 else null end ) as cat_18,
# 		sum(case catalogue when '历史学类' then 1 else null end ) as cat_19,
# 		sum(case catalogue when '交通运输类' then 1 else null end ) as cat_20,
# 		sum(case catalogue when '工商管理类' then 1 else null end ) as cat_21,
# 		sum(case catalogue when '植物生产类' then 1 else null end ) as cat_22,
# 		sum(case catalogue when '水产类' then 1 else null end ) as cat_23,
# 		sum(case catalogue when '森林资源类' then 1 else null end ) as cat_24,
# 		sum(case catalogue when '环境与安全类' then 1 else null end ) as cat_25,
# 		sum(case catalogue when '环境生态类' then 1 else null end ) as cat_26,
# 		sum(case catalogue when '动物生产类' then 1 else null end ) as cat_27,
# 		sum(case catalogue when '草业科学类' then 1 else null end ) as cat_28,
# 		sum(case catalogue when '电子信息科学类' then 1 else null end ) as cat_29,
# 		sum(case catalogue when '数学类' then 1 else null end ) as cat_30,
# 		sum(case catalogue when '物理学类' then 1 else null end ) as cat_31,
# 		sum(case catalogue when '地球物理学类' then 1 else null end ) as cat_32,
# 		sum(case catalogue when '材料科学类' then 1 else null end ) as cat_33,
# 		sum(case catalogue when '心理学类' then 1 else null end ) as cat_34,
# 		sum(case catalogue when '统计学类' then 1 else null end ) as cat_35,
# 		sum(case catalogue when '力学类' then 1 else null end ) as cat_36,
# 		sum(case catalogue when '环境科学类' then 1 else null end ) as cat_37,
# 		sum(case catalogue when '化学类' then 1 else null end ) as cat_38,
# 		sum(case catalogue when '海洋科学类' then 1 else null end ) as cat_39,
# 		sum(case catalogue when '地质学类' then 1 else null end ) as cat_40,
# 		sum(case catalogue when '地理科学类' then 1 else null end ) as cat_41,
# 		sum(case catalogue when '大气科学类' then 1 else null end ) as cat_42,
# 		sum(case catalogue when '政治学类' then 1 else null end ) as cat_43,
# 		sum(case catalogue when '武器类' then 1 else null end ) as cat_44,
# 		sum(case catalogue when '土建类' then 1 else null end ) as cat_45,
# 		sum(case catalogue when '航空航天类' then 1 else null end ) as cat_46,
# 		sum(case catalogue when '公安学类' then 1 else null end ) as cat_47,
# 		sum(case catalogue when '公安技术类' then 1 else null end ) as cat_48,
# 		sum(case catalogue when '测绘类' then 1 else null end ) as cat_49,
# 		sum(case catalogue when '材料类' then 1 else null end ) as cat_50,
# 		sum(case catalogue when '经济学类' then 1 else null end ) as cat_51,
# 		sum(case catalogue when '职业技术教育类' then 1 else null end ) as cat_52,
# 		sum(case catalogue when '体育学类' then 1 else null end ) as cat_53,
# 		sum(case catalogue when '教育学类' then 1 else null end ) as cat_54,
# 		sum(case catalogue when '职业技术教育' then 1 else null end ) as cat_55,
# 		sum(case catalogue when '水利类' then 1 else null end ) as cat_56,
# 		sum(case catalogue when '轻工纺织食品类' then 1 else null end ) as cat_57,
# 		sum(case catalogue when '农业经济管理类' then 1 else null end ) as cat_58,
# 		sum(case catalogue when '旅游类' then 1 else null end ) as cat_59,
# 		sum(case catalogue when '机械类' then 1 else null end ) as cat_60,
# 		sum(case catalogue when '管理科学与工程类' then 1 else null end ) as cat_61,
# 		sum(case catalogue when '公共管理类' then 1 else null end ) as cat_62,
# 		sum(case catalogue when '农业工程类' then 1 else null end ) as cat_63,
# 		sum(case catalogue when '海洋工程类' then 1 else null end ) as cat_64,
# 		sum(case catalogue when '自动化类' then 1 else null end ) as cat_65,#!!!!!!!!
# 		sum(case catalogue when '能源动力类' then 1 else null end ) as cat_66,
# 		sum(case catalogue when '机械类' then 1 else null end ) as cat_67,
# 		sum(case catalogue when '地矿类' then 1 else null end ) as cat_68,
# 		sum(case catalogue when '食品科学与工程类' then 1 else null end ) as cat_69,
# 		sum(case catalogue when '仪器仪表类' then 1 else null end ) as cat_70,
# 		sum(case catalogue when '生物工程类' then 1 else null end ) as cat_71,
# 		sum(case catalogue when '林业工程类' then 1 else null end ) as cat_72,
# 		sum(case catalogue when '核工程类' then 1 else null end ) as cat_73,
# 		sum(case catalogue when '工程力学类' then 1 else null end ) as cat_74,
# 		sum(case catalogue when '社会学类' then 1 else null end ) as cat_75
# 		 from myweb_univer_major_line_lb where year =2015 group by school ) b
# 		 on a.school=b.school;
#     '''

    connect=pymysql.Connect(host='172.19.235.29',port=3306,user='root',passwd='mingming',
                            db='gaokao_py3',charset='utf8')#'latin-1' codec can't encode characters in position 0-1
    cursor=connect.cursor()
    cursor.execute(sql)
    #python对MySQL进行数据的插入、更新和删除之后需要commit，数据库才会真的有数据操作;但是查询不需要！
    data_t=cursor.fetchall()
    connect.close()
    data=pd.DataFrame.from_records(list(data_t))#data_t是元组。0\1\2\3\14
    cols=list([0,4,5,6,10,11]);cols.extend(list(range(16,103)))
    data=data[cols]
    cols_1=[10,11];cols_1.extend(list(range(16,103)))
    print(data[cols_1])
    data.fillna(0,inplace=True)
    data[cols_1] = data[cols_1].astype(np.int)
    data=pd.get_dummies(data)#虽然删去了两列但列名依旧不变
    #get_dummies(data[[0,4,5,6]]）是不对的，结果只有28列；get_dummies(data)也不对的，结果3374列（数字列也能推荐？？也许是格式是str？？全部转化成int）
    data=data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    print(data)
    data.to_csv('data.csv')
    from sklearn.decomposition import PCA
    data=pd.read_csv('data.csv')
    data=np.array(data, dtype=np.int)
    pca=PCA(n_components=6)
    pca.fit(data)
    data_pca=pca.transform(data)
    print('data_pca\n',data_pca)
    # print(len(data_pca[0]))#14
    # print(pca.components_)#主成分数组
    # print(pca.explained_variance_ratio_)#每一组主成分能够解释的比例
    # print(pca.n_components_)#14
    points=data_pca
    from sklearn.cluster import KMeans
    clf= KMeans(n_clusters=10, random_state=1)
    clf.fit(data_pca)
    res=clf.labels_
    print(res)
    r=pd.DataFrame({'聚类类别':res})
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    tsne=TSNE(n_components=2, random_state=0)
    data_pca=pd.DataFrame(data_pca)
    tsne.fit_transform(data_pca)
    print(tsne.embedding_)
    x_tsne =  pd.DataFrame(tsne.embedding_, index=data_pca.index)
    print(x_tsne)
    plt.scatter(x_tsne[:,0],x_tsne[:,1], cmap='prism', alpha=0.4)
    # d = tsne[r[u'聚类类别'] == 0]#d=data[model.labels_==0]  k-means
    # plt.plot(d[0], d[1], 'r.')
    # d = tsne[r[u'聚类类别'] == 1]
    # plt.plot(d[0], d[1], 'go')
    # d = tsne[r[u'聚类类别'] == 2]
    # plt.plot(d[0], d[1], 'b*')
    # d = tsne[r[u'聚类类别'] == 3]#d=data[model.labels_==0]  k-means
    # plt.plot(d[0], d[1], 'r.')
    # d = tsne[r[u'聚类类别'] == 4]
    # plt.plot(d[0], d[1], 'royalblue')
    # d = tsne[r[u'聚类类别'] == 5]
    # plt.plot(d[0], d[1], 'lime')
    # d = tsne[r[u'聚类类别'] == 6]#d=data[model.labels_==0]  k-means
    # plt.plot(d[0], d[1], 'dodgerblue')
    # d = tsne[r[u'聚类类别'] == 7]
    # plt.plot(d[0], d[1], 'coral')
    # d = tsne[r[u'聚类类别'] == 8]
    # plt.plot(d[0], d[1], 'sage')
    # d = tsne[r[u'聚类类别'] == 9]#d=data[model.labels_==0]  k-means
    # plt.plot(d[0], d[1], 'orange')
    # plt.show()

    # # # 给定同维向量数据集合points,数目为n,将其聚为C类（在矩阵U里面体现），m为权重值,u为初始匹配度矩阵（n*C，和为1）,采用闵式距离算法,其参数为p,迭代终止条件为终止值e（取值范围(0，1））及终止轮次。
    # # points=np.array(data,dtype=np.int)#[1 4 21 ..., 0 Decimal('15') 0]:decimal模块用于十进制数学计算'decimal.Decimal' and 'float'
    # p=2;m=2;e=0.1;terminateturn=1000
    # u0=np.random.rand(len(points),20)#np.random.random(3) vs  np.random.rand(3,5)
    # u = np.array([x / np.sum(x, axis=0) for x in u0])  # for x in u0呈现的是行，所以用行axis=0即可
    # # 其中p是一个变参数。当p=1时，就是曼哈顿距离;当p=2时，就是欧氏距离;当p→∞时，就是切比雪夫距离;闵氏距离可以表示一类的距离。
    # print('u\n',u)
    # print('points\n',points)
    # fcm_res=alg_fcm(points,u,m,p,e,terminateturn)
    # print('res1\n',fcm_res[1],'res0\n',fcm_res[0])
    # # pd.DataFrame(fcm_res[1]).to_csv('res1.csv');pd.DataFrame(fcm_res[0]).to_csv('res2.csv')

    # for i in fcm_res[0]:
    #     for j in fcm_res[0]:
    #         print(pearson(i,j))

    return HttpResponse('test')

# from sklearn.manifold import TSNE
    # import matplotlib.pyplot as plt
    # tsne=TSNE(n_components=2, random_state=0)
    # tsne.fit_transform(data_zs)
    # print(tsne.embedding_)
    # x_tsne =  pd.DataFrame(tsne.embedding_, index=data_zs.index)
    # print(x_tsne)
    # plt.scatter(x_tsne[:,0],x_tsne[:,1], cmap='prism', alpha=0.4)
    # # d = tsne[r[u'聚类类别'] == 0]#d=data[model.labels_==0]  k-means
    # # plt.plot(d[0], d[1], 'r.')
    # # d = tsne[r[u'聚类类别'] == 1]
    # # plt.plot(d[0], d[1], 'go')
    # # d = tsne[r[u'聚类类别'] == 2]
    # # plt.plot(d[0], d[1], 'b*')
    # # plt.show()
