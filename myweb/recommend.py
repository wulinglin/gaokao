from django.shortcuts import render,render_to_response
from . import models
import json
from django.shortcuts import render,HttpResponse
#把推荐部分放在myweb下的views文件里。查询部分在web下的views的文件里
global point2
import datetime,pymysql
from django.db.models import Case, CharField, Value, When
import pandas as pd
pd.set_option('expand_frame_repr',False)
from django.views.decorators.cache import cache_page
import logging
logger = logging.getLogger('sourceDns.webdns.views')  # 刚才在setting.py中配置的logger
import numpy as np
from numpy import NaN
from .fcm_cluster import *
try:
    mysql = pymysql.connect('172.19.235.29', 'root', 'xiaoming')
except Exception as  e:
    logger.error(e)  # 直接将错误写入到日志文件
import math
@cache_page(60*15)
# "-----------------------------------------------school--------------------------------------------------------"

def recommend_school(info):
    province=info['province']
    classy=info['classy']
    point=info['point']
    want_province=info['want_province']
    if province in ['上海', '浙江','海南','山东','西藏','新疆']:
        batches17 = ['本科', '第三批']#'第一批','第二批',
        batches_all=['本科', '第二批','第三批']
        response = '<p>上海、浙江已实行新高考，暂不提供上海、浙江的推荐。</p>'
        return HttpResponse(response)
#西藏：重点本科：少数民族325分，汉族440分。普通本科：少数民族285分，汉族355分。专科(高职)：少数民族245分，汉族315分。
    elif province=='西藏':#！不知道怎么推荐，因为学校-省份分数只有第一批、第二批、第三批，不知道是少数民族还是汉族
        if nation=='是':batches17= ['普通本科（少数民族）']#'重点本科（少数民族）','专科（少数民族）'
        else : batches17=['普通本科（汉族）']#'重点本科（汉族）','专科（汉族）'
        batches_all=['普通本科','第三批','第二批']
        response = '<p>西藏暂不提供推荐</p>'
        return HttpResponse(response)
    elif province == '新疆':#！需要补新疆的数据。不知道怎么推荐，因为学校-省份分数只有第一批、第二批、第三批，不知道是少数民族还是汉族
        if nation=='是':batches17= ['三本汉']#'重点本科（少数民族）','专科（少数民族）'
        else : batches17=['']
#其它：依旧实行一二三批，但是有的只有一二批。
    else:
        batches17=['第二批','第三批','本科', '第三批']#'第一批',艺术类本科，体育教育，专科提前批，专科，
        line17_sql= models.nation_line.objects.filter(province=province, year=2017,#datetime.datetime.now().year,
                                                       classy=classy,batch__in=batches17)
        lines17=[]
        for i in line17_sql:lines17.append(int(i.line))
        line17=min(lines17)
        sub_point=user_point-line17
        print(line17,user_point)
        if sub_point < 0:  # 如果该用户的分数没有达到二本分数线，暂不推荐。
            line17_sql = models.nation_line.objects.filter(province=province, year=datetime.datetime.now().year,classy=classy,batch__icontains='专科')
            response = '<p>本系统暂不提供本科以外的推荐。</p>'
            return HttpResponse(response)
        else:
            if want_province == ['无']:
               want_province = ['北京', '天津', '辽宁', '吉林', '黑龙江', '上海', '江苏', '浙江', '安徽', '福建', '山东', '湖北', '湖南',
                                 '广东', '重庆', '四川', '陕西', '甘肃', '河北', '山西', '内蒙古', '河南', '海南', '广西', '贵州', '云南',
                                 '西藏', '青海', '宁夏', '新疆', '江西', '香港', '澳门', '台湾']
            else: pass
            # 构造候选集：选择生源地、年份（？？选哪些年份呢）、科类、专业符合要求，(学校最低分)=<分数差值<=学校最高分+10
            candidate_school_0=models.univer_info.objects.filter(province__in=want_province)
            candidate_school_1=[]#学校候选集candidate_school_1：就是选出学生想去的省份的学校
            [candidate_school_1.append(each.school) for each in candidate_school_0]
            print(candidate_school_1)
            # 提取国家线这个表
            pro_univer_nation_line_t=models.pro_univer_nation_line.objects.filter(
                                        province=province, school__in=candidate_school_1,classy=classy)
            pro_univer_nation_line_df=pd.DataFrame.from_records(pro_univer_nation_line_t.values())
            # 必须先转换成浮点类型才能median（）；astype(int)不能转换NaN哦
            print(pro_univer_nation_line_df)
            pro_univer_nation_line_df[['d_hl', 'd_h', 'd_l', 'd_a', 'd_ha', 'd_la']] = pro_univer_nation_line_df[
                ['d_hl', 'd_h', 'd_l', 'd_a', 'd_ha', 'd_la']].astype(np.float)
            #使用中位数填充空缺值
            # d_l居然有小于0的异常值，全部剔除！！#去掉groupby后的 'batch'
            grouped = pro_univer_nation_line_df[(pro_univer_nation_line_df['d_l']>=0)&(pro_univer_nation_line_df['d_a']>=0)].groupby\
                    (['school', 'province', 'classy'], as_index=False)[['d_hl', 'd_h', 'd_l', 'd_a', 'd_ha', 'd_la']].median()

            candidate_school_dict={}
            for index,row in grouped.iterrows():#values返回numpy类型
                if np.isnan(row['d_l'])==False and np.isnan(row['d_h'])==False:
                    if row['d_l'] <= sub_point < row['d_h']:
                        if row['school'] not in candidate_school_dict:
                            print('最低分：'+str(row['d_l'])+'最高分：'+str(row['d_h'])+str(row['school'])+'：在最高分与最低分之间！')
                            candidate_school_dict[row['school']]=0
                elif np.isnan(row['d_l'])==False and np.isnan(row['d_a'])==False:
                    if row['d_l'] <= sub_point < row['d_a']+15:
                        if row['school'] not in candidate_school_dict:
                            print('最低分：' + str(row['d_l']) + '平均分：' +str(row['d_a'])+ str(row['school']) + '：在最低分与平均分+15之间！')
                            candidate_school_dict[row['school']]=0
                elif np.isnan(row['d_l'])==False and np.isnan(row['d_a'])==True and  np.isnan(row['d_h'])==True:
                    if row['d_l'] <= sub_point<row['d_l']+30:
                        if row['school'] not in candidate_school_dict:
                            print('最低分：' + str(row['d_l']) + '.' + str(row['school']) + '：在最低分与最低分+30之间！')
                            candidate_school_dict[row['school']]=0
                elif np.isnan(row['d_l']) ==True and np.isnan(row['d_a']) == False and np.isnan(row['d_h']) == True:
                    if row['d_a']-5 <= sub_point<row['d_a']+5:
                        if row['school'] not in candidate_school_dict:
                            print('平均分：' + str(row['d_a']) + '.' + str(row['school']) + '：在平均分+-5之间！')
                            candidate_school_dict[row['school']]=0
                else:pass

            sql='''
                select school,
            sum(case  year when '2010' then d_a else  null end)as  '2010',
            sum(case year when  '2011' then d_a else  null  end)as  '2011',
            sum(case  year when '2012' then d_a else null end  )as  '2012',
            sum(case  year when '2013' then d_a else null end  )as  '2013',
            sum(case year  when '2014' then d_a else null end  )as  '2014',
            sum(case year  when '2015' then d_a else null end ) as  '2015',
            sum(case year  when '2016' then d_a else null end ) as  '2016'
            from myweb_pro_univer_nation_line a where a.classy='理科' and a.province='四川' and a.school in {} group by a.school'''\
            .format(tuple(candidate_school_dict.keys()))
            print(candidate_school_dict)
            connect=pymysql.Connect(host='172.19.235.29',user='root',passwd='mingming',port=3306,
                                    db='gaokao_py3',charset='utf8')
            cursor=connect.cursor()
            cursor.execute(sql)
            data_t=cursor.fetchall()
            cursor.close()

            data_df=pd.DataFrame.from_records(list(data_t))
            print(data_df)
            list_fillna = []
            for k, v in data_df.iterrows():
                v0 = v[:1]
                v1= v[1:]
                v1.fillna(v1.mean(), inplace=True)
                v = v0.append(v1)
                list_fillna.append(v.values)
            data_fillna=pd.DataFrame(list_fillna)
            print(data_fillna.ix[:,1:])
            data_fillna.dropna(how='any',inplace=True)
            # print(data_fillna)
            points=np.array(data_fillna.ix[:,1:],dtype=np.int)
            # # 给定同维向量数据集合points,数目为n,将其聚为C类（在矩阵U里面体现），m为权重值,u为初始匹配度矩阵（n*C，和为1）,采用闵式距离算法,其参数为p,迭代终止条件为终止值e（取值范围(0，1））及终止轮次。
            # points=np.array(data,dtype=np.int)#[1 4 21 ..., 0 Decimal('15') 0]:decimal模块用于十进制数学计算'decimal.Decimal' and 'float'
            p=2;m=2;e=0.1;terminateturn=1000
            u0=np.random.rand(len(points),3)#np.random.random(3) vs  np.random.rand(3,5)
            u = np.array([x / np.sum(x, axis=0) for x in u0])  # for x in u0呈现的是行，所以用行axis=0即可
            # 其中p是一个变参数。当p=1时，就是曼哈顿距离;当p=2时，就是欧氏距离;当p→∞时，就是切比雪夫距离;闵氏距离可以表示一类的距离。
            print('u\n',u)
            print('points\n',points)
            fcm_res=alg_fcm(points,u,m,p,e,terminateturn)#u2, k, centroids
            print('res1\n',fcm_res[1],'\nres0\n',fcm_res[0])
            print('res2\n',fcm_res[2])
            centroids = [each.mean() for each in fcm_res[2]]
            centroids_sort = pd.Series(centroids).sort_values()
            centroids_dict = {}
            print(centroids_sort)
            for i, j, k in zip(['保', '稳', '冲'], centroids_sort.values, centroids_sort.index):
                print(i,j,k)
                centroids_dict[k] =i
            print(centroids_dict)

            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            labels=[each.index(max(each)) for each in fcm_res[0]]#list.index(max(each))找出最大值对应的索引index
            print('labels\n',labels)
            labels_bwc=[centroids_dict[i] for i in labels]
            print('labels_bwc\n',labels_bwc)
            candidate_school_2=data_df[0].values
            print(candidate_school_2)

            school_dict = {'development': { 1:6,2:5,3:4,4:3,5:2,6:1}, 'provincial_capital': {0: 0, 1:4},
                           'is_doctor': {0: 0, 1: 1},'is_master': {0: 0, 1: 1},'leixing_int': {0: 0, 1: 3, 2: 5}};
            candidate_school_t=models.univer_info.objects.filter(school__in=candidate_school_2)
            print('candidate_school_t\n',pd.DataFrame.from_records(candidate_school_t.values()))
            for each in candidate_school_t:
                # print(school_dict['development'][int(each.development)])#明明each.development就已经是<class 'int'>了，却还需要int()一下，
                #地理比重0.43，师资比重0.57
                candidate_school_dict[each.school]+=(school_dict['development'][int(each.development)]+school_dict['provincial_capital'][int(each.provincial_capital)])/10*0.43\
                                                    +(school_dict['leixing_int'][int(each.leixing_int)]+school_dict['is_doctor'][int(each.is_doctor)] \
                                                    + school_dict['is_master'][int(each.is_master)])/10*0.57
            # 把candidate_school_dict转化成[(),,..()]list套元组，这样就可以大小排序了，字典是不能排序的，总是乱序
            #不能groupby之后再sort_values，可以先sort_values再groupby
            candidate_school_list=[(k, v) for k, v in candidate_school_dict.items()]
            candidate_school_list=sorted(candidate_school_list,key=lambda d:d[1],reverse=True)#先排序，不要最后才排序
            print(candidate_school_list)
            bwc_dict = {}
            b, w, c = [], [], []
            for k,j,v in zip([each[0] for each in candidate_school_list ],[each[1] for each in candidate_school_list ],labels_bwc):
                if v == '保':b.append((k, j))
                if v == '稳':w.append((k, j))
                if v == '冲':c.append((k, j))
            bwc_dict['bao'] = b;
            bwc_dict['wen'] = w;
            bwc_dict['chong'] = c;  # {'a': ('保', 0.544),...}
            bwc_json = json.dumps(bwc_dict, ensure_ascii=False)  # {"保": [["安徽工业大学", 0.1], ...]}
            print('bwc_dict\n', bwc_dict)

            import matplotlib
            matplotlib.rcParams['font.sans-serif'] = ['SimHei']
            matplotlib.rcParams['font.family'] = 'sans-serif'
            X_tsne = TSNE(learning_rate=100, n_components=2, init='pca', random_state=0).fit_transform(points)
            print('1\n',X_tsne[:, 0])
            print('2\n',X_tsne[:, 1])
            plt.show()
            plt.savefig("tsne.png")
            return HttpResponse(bwc_json)
            # return render(request,'recom_sch_res.html',{'candidate_school':candidate_school})
    return render_to_response('recom_sch.html',{'error':error})

# "------------------------------------------------major---------------------------------------------------------"
def major_cat(request):
    error=False
    temp = '';answer_list ,answer = [],[]
    for key in ['t%s' % i for i in range(1, 28)]:
        temp0 = request.GET.get(key, '')
        temp += temp0
    # if len(temp)<27:
    #     error=True
    for i, j in enumerate(list(temp)):
        if i < 7:
            if j == 'a':answer_list.append('E')
            else:answer_list.append('I')
        elif i < 14:
            if j == 'a':answer_list.append('N')
            else:answer_list.append('S')
        elif i < 21:
            if j == 'a':answer_list.append('F')
            else:answer_list.append('T')
        else:
            if j == 'a':answer_list.append('J')
            else:answer_list.append('P')
    if answer_list.count('E') > answer_list.count('I'):answer.append('E')
    else:answer.append('I')
    if answer_list.count('N') > answer_list.count('S'):answer.append('N')
    else:answer.append('S')
    if answer_list.count('F') > answer_list.count('T'):answer.append('F')
    else:answer.append('T')
    if answer_list.count('J') > answer_list.count('P'):answer.append('J')
    else:answer.append('P')
    mbti_answer = ''.join(answer)
    print(mbti_answer)
    request.session['mbti'] = mbti_answer
    major16_lb_t=models.major16_lb.objects.filter(mbti__icontains=str(mbti_answer))
    print(pd.DataFrame.from_records(major16_lb_t.values()))
    catalogue=list(set([each.catalogue for each in major16_lb_t]))#这里有空值是因为catalogue有一部分是空的
    catalogue_str='、'.join(catalogue).strip()
    print(catalogue,catalogue_str)
    mbti_explanation=models.mbti_chractor.objects.values('explanation').filter(mbti=mbti_answer)
    print(mbti_explanation)
    res_dict={'catalogue':catalogue,'mbti_explanation':list(mbti_explanation)[0]['explanation'],'mbti_answer':mbti_answer}
    print(res_dict)
    res_json=json.dumps(res_dict,ensure_ascii=False)
    return render(request,'recom_major_cat.html',{'res_json':res_json,'catalogue':catalogue,})
    # return render_to_response('recom_major_mbti.html',{'error':error})

def major_maj(request):
    error=False
    lingyu = request.GET.getlist('lingyu')
    if len(lingyu)==0:
        error=True
    else:
        lingyu = request.GET.getlist('lingyu')
        mbti_answer=request.session['mbti']
        print(mbti_answer,lingyu)
        major16_lb_t = models.major16_lb.objects.filter(mbti__icontains=str(mbti_answer),catalogue__in=lingyu)
        major_recommend = [each.major for each in major16_lb_t]
        print(major_recommend)
        request.session['major_recommend'] = major_recommend
        major_dict={'major_recommend':major_recommend}
        major_json=json.dumps(major_dict,ensure_ascii=False)
        major_str = '、'.join(major_recommend).strip()
        return render(request, 'recom_major_maj.html', {'major_recommend': major_recommend, 'major_json': major_json})
        # return render(request, 'recom_major_maj.html',{'major_recommend':major_recommend,'major_str':major_str})
    return render_to_response('recom_major_cat.html',{'error':error})

# ________________________________________________________major&school_____________________________________________________
def recommend_isMajor(request):
    error = False
    # if'province'in request.GET and 'classy'in request.GET and 'point'in request.GET and 'add_point'in request.GET and 'nation'in request.GET :
    want_province = request.GET.getlist('want_province')
    if (request.GET.get('add_point') == '' and request.GET.get('point') == '') or want_province == []:
        error = True
    else:
        if request.GET.get('add_point') == '':
            user_point = int(request.GET.get('point'))
        else:
            user_point = int(request.GET.get('add_point'))
        province = request.GET.get('province');
        nation = request.GET.get('nation');
        classy = request.GET.get('classy');
        print(province, nation, classy)

        request.session['user_point']=user_point
        request.session['province']=province
        request.session['nation']=nation
        request.session['classy']=classy
        request.session['want_province']=want_province

        major16_lb_t = models.major16_lb.objects.all()
        major16_lb_df=pd.DataFrame.from_records(major16_lb_t.values())
        lb_dict = {}
        for index,row in major16_lb_df.iterrows():
            if str(row['classes']) in lb_dict:
                if row['catalogue'] not in lb_dict[row['classes']]:
                    lb_dict[row['classes']].append(row['catalogue'])
                else:
                    continue
            else:lb_dict.setdefault(row['classes'], []).append(row['catalogue'])
        return render(request, 'recommend_isMajor.html',{'lb_dict':lb_dict})
    return render_to_response('recommend_info.html',{'error':error})

def recommend_res(request):
    error=False
    catalogue = request.GET.getlist('catalogue')
    want_major = request.GET.getlist('want_major')
    if len(catalogue)<1 and len(want_major)<1:
        error=True
    else:
        request.session['catalogue'] = catalogue;
        request.session['want_major'] = want_major
        if request.session['add_point'] == '':user_point = int(request.session['point'])
        else:user_point = int(request.session['add_point'])
        province = request.session['province'];
        nation = request.session['nation'];
        classy= request.session['classy'];
        catalogue=request.session['catalogue'];
        want_major=request.session['want_major']
        want_province = request.session['want_province']
        # 都是本科。上海、浙江、海南、山东，合并本科。实行新高考，暂不推荐。浙江分了一二三段，其它都是本科# 本科 自主招生 艺术类本科 体育类本科 国家专项计划
        if province in ['上海', '浙江', '海南', '山东']:
            batches17 = ['本科', '第三批']  # '第一批','第二批',
            batches_all = ['本科', '第二批', '第三批']
            response = '<p>上海、浙江已实行新高考，暂不提供上海、浙江的推荐。</p>'
            return HttpResponse(response)
        elif province == '西藏':  # ！不知道怎么推荐，因为学校-省份分数只有第一批、第二批、第三批，不知道是少数民族还是汉族
            if nation == '是':
                batches17 = ['普通本科（少数民族）']  # '重点本科（少数民族）','专科（少数民族）'
            else:
                batches17 = ['普通本科（汉族）']  # '重点本科（汉族）','专科（汉族）'
            batches_all = ['普通本科', '第三批', '第二批']
            response = '<p>西藏暂不提供推荐</p>'
            return HttpResponse(response)
        elif province == '新疆':  # ！需要补新疆的数据。不知道怎么推荐，因为学校-省份分数只有第一批、第二批、第三批，不知道是少数民族还是汉族
            if nation == '是':
                batches17 = ['三本汉']  # '重点本科（少数民族）','专科（少数民族）'
            else:
                batches17 = ['']
            # 其它：依旧实行一二三批，但是有的只有一二批。
        else:
            batches17 = ['第二批', '第三批']  # '第一批',艺术类本科，体育教育，专科提前批，专科，
            line17_sql = models.nation_line.objects.filter(province=province, year=2017,#datetime.datetime.now().year,
                                                           classy=classy, batch__in=batches17)
            lines17 = []
            for i in line17_sql: lines17.append(int(i.line))
            line17 = min(lines17)
            sub_point = user_point - line17
            print(line17, user_point)
            if sub_point < 0:  # 如果该用户的分数没有达到二本分数线，暂不推荐。
                line17_sql = models.nation_line.objects.filter(province=province, year=datetime.datetime.now().year,
                                                               classy=classy, batch__icontains='专科')
                response = '<p>本系统暂不提供本科以外的推荐。</p>'
                return HttpResponse(response)
            else:
                if want_province == ['全国']:
                    want_province = ['北京', '天津', '辽宁', '吉林', '黑龙江', '上海', '江苏', '浙江', '安徽', '福建', '山东', '湖北', '湖南',
                                     '广东', '重庆', '四川', '陕西', '甘肃', '河北', '山西', '内蒙古', '河南', '海南', '广西', '贵州', '云南',
                                     '西藏', '青海', '宁夏', '新疆', '江西', '香港', '澳门', '台湾']
                # 学校候选集
                candidate_school_0 = models.univer_info.objects.filter(province__in=want_province)
                candidate_school_1 = []
                [candidate_school_1.append(each.school) for each in candidate_school_0]
                print(len(catalogue),len(want_major))
                print(catalogue,want_major,classy,province,want_province)
                if len(catalogue)>0 and ''.join(want_major)=='':
                    univer_major_line_lb_t=models.univer_major_line_lb.objects.filter(province=province,school__in=candidate_school_1,
                                                                                  classy=classy,catalogue__in=catalogue)
                    univer_major_line_lb_df=pd.DataFrame.from_records(univer_major_line_lb_t.values())
                    univer_major_line_lb_df[['a_l', 'h_l', 'h_a']] = univer_major_line_lb_df[['a_l', 'h_l', 'h_a']].astype(np.float)
                    # d_l居然有小于0的异常值，即分数线没达到本科的，未作剔除！！
                    grouped =pd.pivot_table(univer_major_line_lb_df,index=['school', 'province', 'classy', 'classes', 'catalogue'],
                                            values=['a_l','h_l','h_a'],aggfunc={'a_l':np.min,'h_l':np.max,'h_a':np.mean})#index,columns,h_a是（最高-均值）的均值
                    grouped['max_min'] = grouped['h_l']-grouped['a_l']
                    print(grouped.head())
                    candidate_school_catalogue={}
                    for index,row in grouped.iterrows():#school province classy classes catalogue // a_l h_a h_l max_min
                        #如果均值-最高值都全：
                        if np.isnan(row['a_l']) == False and np.isnan(row['h_l']) == False:
                            if row['a_l'] <= sub_point < row['h_l']:
                                if (index[0],index[4]) not in candidate_school_catalogue:
                                    print(index[0],index[4],'  均值分：' + str(row['a_l']) + '最高分：' + str(row['h_l']) + '：在最高分与均值分之间！')
                                    candidate_school_catalogue[(index[0],index[4])] = 0
                        # 如果均值全-最高值不全：
                        elif np.isnan(row['a_l']) == False and np.isnan(row['h_l'])== True:
                            if row['a_l'] <= sub_point < row['a_l'] + 10:#+30
                                if ((index[0],index[4])) not in candidate_school_catalogue:
                                    print(index[0],index[4],'均值分：' + str(row['a_l']) + '.' + '：在均值分与最低分+10之间！')
                                    candidate_school_catalogue[(index[0],index[4])] = 0
                        # 如果均值不全-最高值全：
                        elif np.isnan(row['a_l']) == True and np.isnan(row['h_l']) == False:
                            if row['h_l'] - 5 <= sub_point < row['h_l'] + 5:
                                if (index[0],index[4]) not in candidate_school_catalogue:
                                    print(index[0],index[4],'平均分：' + str(row['h_l']) + '.' + '：在平均分+-5之间！')
                                    candidate_school_catalogue[(index[0],index[4])] = 0
                        # 如果均值-最高值都不全：
                        elif np.isnan(row['a_l']) == True and np.isnan(row['h_l']) == True:
                            pass

                    candidate_school_t = models.univer_info.objects.filter(school__in=[key[0] for key in candidate_school_catalogue.keys()])
                    candidate_school_df=pd.DataFrame.from_records(candidate_school_t.values())
                    print(candidate_school_catalogue)
                    print(candidate_school_df)
                    for index,row in candidate_school_df.iterrows():
                        for key,values in candidate_school_catalogue.items():
                            if key[0]==row['school']:
                                if (np.isnan(row['school_rank'])==True) or (row['school_rank']==0) or (row['school_rank']!='0'):
                                    candidate_school_catalogue[key] += float(row['development']) + float(row['leixing_int'] * 2) +\
                                                                       float(row['provincial_capital']) + float(row['is_doctor']) + float(row['is_master'])
                                else:
                                    candidate_school_catalogue[key]+=float(row['development'])+float(row['leixing_int']*2)+\
                                                                        float(row['provincial_capital'])+float(row['is_doctor'])\
                                                                    +float(row['is_master'])+math.log(1000/float(row['school_rank']))

                    candidate_school_catalogue_sort=sorted(candidate_school_catalogue.items(),key=lambda d:d[1],reverse=True )[:10]
                    candidate=[each[0] for each in candidate_school_catalogue_sort]
                    # print(candidate_school_catalogue)
                    # print(candidate_school_catalogue_sort)
                    candidate_dict={}
                    candidate_dict['candidate']=candidate
                    return HttpResponse(json.dumps(candidate_dict,ensure_ascii=False))
                    # return    render(request,'recommend_res.html',{'candidate':candidate})
        # return render(request,'recommend_res.html',{'candidate':candidate})
    return render_to_response('recommend_isMajor.html', {'error': error})

