from django.conf.urls import url, include
from . import views,views_predict,cluster,views_faq
import sys
sys.path.append('G:\MY\gaokao\web\web')
from web import view
from myweb.views_faq import *
from . import views_sim
#此myweb下的urls的urlpatterns承接web下的urls的urlpatterns
urlpatterns=[
    url(r'^$',view.index,{'html':'index.html'}),
    #推荐首页：学校+专业+学校及专业推荐
    url(r'^recommend/$', view.index,{'html':'recommend.html'}),
    #推荐学校：
    url(r'^recommend/school/info/$',view.index,{'html':'recom_sch.html'}),
    url(r'^recommend/school/pro$',views.recommend_getInfo,{'html':'recom_pro.html'}),
    url(r'^recommend/school/res/$',views.recommend_school),
    #推荐专业：
    url(r'^recommend/major/$',view.index,{'html':'recom_major.html'}),
    url(r'recommend/major/mbti/$',view.index,{'html':'recom_major_mbti.html'}),
    url(r'recommend/major/catalogue/$', views.major_cat),
    url(r'recommend/major/major/$', views.major_maj),
    #推荐学校以及专业
    url(r'^recommend/info/$', view.index, {'html': 'recommend_info.html'}),
    url(r'recommend/isMajor/$', views.recommend_isMajor),#这里的/很重要，不能丢；只有一个网页的时候可丢？#没有^也可以？
    # url(r'^recommend/pro/$',views.recommend_pro),
    url(r'^recommend/res/$', views.recommend_res),
    #历史数据查询
    url(r'search/$', view.search, ),#{'html': 'search.html'}
    url(r'search/schoolInfo/$',view.schoolInfo),#少一个/都不行
    url(r'search/nationLine/$', view.nationLine),
    url(r'search/majorLine/$', view.majorLine),
    url(r'search/schoolLine/$', view.schoolLine),
        #在schoolline当页加一个学校相似度，但是这块不再是动态评价:使用sim函数
        # url(r'search/schoolLine/similarity$', view.schoolLine),
    #FAQ客服在线
    url(r'faq/$', view.index, {'html': 'faq.html'}),
    url(r'faq/aq$',views_faq.aq ),
    #这里明明是import文件没有问题！但是models.nation_line就有问题，难道是models有两个有歧义？

#--------------------------------------------------------------
# 此部分只为写论文，不用于网站
    #学校动态相似度计算:
    url(r'sim/$', view.index, {'html': 'sim.html'}),
    url(r'sim/similarity$',views_sim.similarity),
    url(r'sim/dynamic$', views_sim.dynamic_res),

    #预测国家线
    url(r'^predict/$',view.index,{'html':'predict.html'}),
    url(r'predict/line/$',views_predict.predict_line),
    # url(r'predict/major/$',views_predict.predict_major_line),#专业分数线缺失数据太多，无法做预测
    url(r'predict/school/$',views_predict.predict_school_line),
    #预测专业分数线
#--------------------------------------------------------------------
]