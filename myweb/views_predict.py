#该部分是分数线预测部分
from django.shortcuts import render,HttpResponse
from . import models
import numpy as np
import pandas as pd

pd.set_option('expand_frame_repr',False)
import logging
logger=logging.getLogger('sourceDns.webdns.views')
from pandas import Series,DataFrame
from .gm import Identification_Algorithm,GM_Model,Return,GM_gbnn,GM_ratio_bpnn
import math,json
from .bpnn import *

def predict_line(request):#school major province  batch  classy catalogue classes
    #初始化原始数据
    classy=request.GET.get('classy')
    province=request.GET.get('province')
    batch=request.GET.get('batch')
    #地区有少数民族和汉族之分,暂不推荐
    if province in ['西藏','新疆']:
        return HttpResponse('由于{}地区有少数民族和汉族之分，情况复杂。本系统暂不提供{}的预测！'.format(province))
    #一二三本：
    elif province in ['吉林','宁夏','甘肃','黑龙江','湖南','陕西','青海','云南']:
        # 2011, 不处理,不处理,不处理,2012-2014第三批A\B,...(2012)
        years=list(range(2011,2018,1))
        if province=='湖南':
            if batch=='第三批' and province=='湖南': #2012-2014第三批A\B
                nation_line_t0=models.nation_line.objects.filter(classy=classy,province=province,batch__in=['第三批A','第三批B','第三批'],year__in=years)
                nation_line_df0=pd.DataFrame.from_records(nation_line_t0.values())
                nation_line_df1=nation_line_df0.groupby(['classy','province','year'],as_index=False)[['line']].min()
                print(nation_line_df1)
                nation_line_df=nation_line_df1
            else:
                nation_line_t = models.nation_line.objects.filter(classy=classy,province=province,batch=batch,year__in=years)
        elif province in ['陕西','青海','云南']:#汉与少数从2012开始不分，故从2012预测
            years=list(range(2012,2018,1))
            nation_line_t=models.nation_line.objects.filter(classy=classy,province=province,batch=batch,year__in=years)
        else:#'吉林','宁夏','甘肃','黑龙江':2011
            nation_line_t = models.nation_line.objects.filter(classy=classy,province=province,batch=batch,year__in=years)
    #本科:
    elif province in ['上海','山东','浙江','海南']:#从本科最低分取
        if province in ['上海','浙江']:
            return HttpResponse('因{}刚实行文理不分科不久，数据不足，本系统暂时无法提供预测！'.format(province))
        else:nation_line_t = models.nation_line_lowest.objects.filter(classy=classy,province=province)
    #一二本
    else:
        if province in ['内蒙古']: #2013起
            nation_line_t = models.nation_line.objects.filter(classy=classy, province=province,year__in=list(range(2013,2018,1)))
        else:#'湖北','江苏','辽宁','天津':2010年前批次不规范
            nation_line_t = models.nation_line.objects.filter(classy=classy, province=province,batch=batch,year__in=list(range(2010,2018,1)))
    if province =='湖南' and batch=='第三批':
        pass
    else:
        nation_line_df=pd.DataFrame.from_records(nation_line_t.values())
    print(nation_line_df)
    data = np.array(nation_line_df['line'])
    tmp0=np.array(nation_line_df['year'])
    # date = pd.period_range('2000', '2005', freq='A-DEC')##？？？？？？？？？？？
    X=pd.Series(data,index=tmp0).sort_index()
    print(X)
    #如果某一年有缺失值，则取最近几年的连续数值：
    d = {};first = True;index = X.index;values = X.values
    for i in range(len(X)):  # ,x.index,x.values):
        if i > 0:
            if index[i] == index[i-1]+ 1:
                if first:
                    d = {}
                    d[index[i-1]] = values[i- 1]
                first = False
                d[index[i]] = values[i]
            else:
                first = True
    #所剩连续数值必须大于4年才能继续做灰色预测：
    X0 = pd.Series(d).sort_index()
    print(X0)
    if len(X0)<4:
        return HttpResponse("数据连续年份少于4年，不能做灰色预测！")
    else:
        # #归一化:为啥归一化还通不过检验？！！
        # X0_stadard=X.apply(lambda x:(x-min(X))/(max(X)-min(X)))
        #级比检验
        a_start=math.e**(-2/(len(X0)+1));a_end=math.e**(2/(len(X0)+1))
        X0_list=X0.values
        X0_lamda=[X0_list[i-1]/X0_list[i] for i,j in enumerate(X0_list) if i >0]
        print(X0_lamda,a_start,a_end)
        for each in X0_lamda:
            if each<a_start or each>a_end:
                print(each,a_start,a_end)
                return HttpResponse('不能通过灰色预测的级比检验，无法对{}{}{}的分数线进行预测。'.format(province,classy,batch))
            else:
                #累加
                X1=X0.cumsum()
                print ('原始数据累加为:',X1)
                #辨识算法:计算a,b:'numpy.ndarray'
                a_all=[]
                for each in [[0.6,0.4],[0.5,0.5],[0.4,0.6],[0.3,0.7]]:
                    a = Identification_Algorithm(X0,each[0],each[1])#算式中会有X0到累加X1的计算，所以放入X0即可
                    # 检验|a|是否小于2:适用范围是与发展系数。相关的
                    if abs(a[0]) > 2:
                        return HttpResponse('发展系数不能通过检验，无法对{}{}{}的分数线进行预测。'.format(province, classy, batch))
                    a_all.append(a)
                print ('a矩阵为:')
                print (a)

                #GM(1,1)模型
                tmp=np.array([i+1 for i in  range(len(X0)+1)])#tmp = np.array([1,2,3,4,5,6])#6个数据
                XK = GM_Model(X0,X1,a,tmp)

                #预测值还原
                X_Return = Return(XK)

                #预测值即预测值精度表
                X_Compare1 = np.ones(len(X0))
                X_Compare2 = np.ones(len(X0))
                for i,j in enumerate(X0.index):
                    X_Compare1[i] = X0[j]-X_Return[j]
                    X_Compare2[i] = X_Compare1[i]/X0[j]*100
                Compare = {'GM':XK,'1—AGO':np.cumsum(X0),'Returnvalue':X_Return,'Realityvalue':X0,'Error':X_Compare1,'RelativeError(%)':X_Compare2}
                X_Compare = DataFrame(Compare, index=X0.index)
                # X_Compare = DataFrame(Compare,index=date)
                print ('预测值即预测值精度表')
                print (X_Compare)

                # print('-'*80)
                # XK= GM_ratio_bpnn(X0, X1, a_all, tmp)
                #预测值还原
                # X_Return = Return(XK)

                #预测值即预测值精度表
                # X_Compare1 = np.ones(len(X0))
                # X_Compare2 = np.ones(len(X0))
                # for i,j in enumerate(X0.index):
                #     X_Compare1[i] = X0[j]-X_Return[j]
                #     X_Compare2[i] = X_Compare1[i]/X0[j]*100
                # Compare = {'GM':XK,'1—AGO':np.cumsum(X0),'Returnvalue':X_Return,'Realityvalue':X0,'Error':X_Compare1,'RelativeError(%)':X_Compare2}
                # X_Compare = DataFrame(Compare, index=X0.index)
                # # X_Compare = DataFrame(Compare,index=date)
                # print ('预测值即预测值精度表')
                # print (X_Compare)



                x0_mean=np.mean(X0)
                s1=np.sqrt(sum([(x-x0_mean)**2 for x in X0.values])/(len(X0)-1))
                X_theta = X0 - X_Return[:-1]
                theta=np.sum(X0-X_Return)/len(X0)#series必须用np.sum;list用sum即可
                s2=np.sqrt(sum([(t-theta)**2 for t in X_theta.values])/(len(X0)-1))
                print(s1);print(s2)
                c=s2/s1
                print(c)
                res_list = []
                for index, value in zip(X0.index, X0):
                    res_dict = {}
                    res_dict[u'年份'] = index;
                    res_dict[u'分数'] = value
                    res_list.append(res_dict)
                if c <=0.35:
                    res_dict={}
                    res_dict['2018']=round((X_Return.values)[-1],0)
                    res_dict[u'预测精度等级']='优'
                    res_list.append(res_dict)

                elif c  <= 0.5:
                    res_dict = {}
                    res_dict['2018'] = round((X_Return.values)[-1], 0)
                    res_dict[u'预测精度等级'] = '合格'
                    res_list.append(res_dict)
                elif c  <= 0.65:
                    res_dict = {}
                    res_dict['2018'] = round((X_Return.values)[-1], 0)
                    res_dict[u'预测精度等级'] = '勉强合格'
                    res_list.append(res_dict)
                elif c > 0.65:
                    res_dict = {}
                    res_dict[u'预测精度等级'] = '不合格：灰色模型后验差比>0.65，未通过检验，故不能对{}{}{}进行灰色预测。'.format(province,classy,batch)

                    res_list.append(res_dict)
                # 简单说就是dump需类似于文件指针的参数就是说可以将dict转成str然后存入文件中；而dumps直接给的是str，也就是将字典转成str。
                res_json=json.dumps(res_list, ensure_ascii=False)#编码不对！
                return HttpResponse(res_json,content_type='application/json;charset=utf-8')

# def predict_major_line(request):#专业分数线缺失数据太多，无法做预测
def predict_school_line(request):
    school=request.GET.get('school')
    classy=request.GET.get('classy')
    province=request.GET.get('province')
    batch=request.GET.get('batch')
    pro_univer_nation_line_t=models.pro_univer_nation_line.objects.filter(school=school,classy=classy,province=province,batch=batch)
    pro_univer_nation_line_df = pd.DataFrame.from_records(pro_univer_nation_line_t.values())
    print(school,province,classy,batch)
    print(pro_univer_nation_line_df)
    data = np.array(pro_univer_nation_line_df['d_a'])
    X_average=pro_univer_nation_line_df['average']
    tmp0 = np.array(pro_univer_nation_line_df['year'])
    # date = pd.period_range('2000', '2005', freq='A-DEC')##？？？？？？？？？？？
    X = (pd.Series(data, index=tmp0).sort_index()).dropna(axis=0)
    print(X)
    # 如果某一年有缺失值，则取最近几年的连续数值：
    d = {};
    first = True;
    index = X.index;
    values = X.values
    for i in range(len(X)):  # ,x.index,x.values):
        if i > 0:
            if index[i] == index[i - 1] + 1:
                if first:
                    d = {}
                    d[index[i - 1]] = values[i - 1]
                first = False
                d[index[i]] = values[i]
            else:
                first = True
    # 所剩连续数值必须大于4年才能继续做灰色预测：
    X0 = pd.Series(d).sort_index()
    print(X0)
    if len(X0) < 4  or (X0.index)[-1]!=2016:#对最后一年非2016的数据进行预测是没有意义的！
        return HttpResponse("有效数据不足，不能做灰色预测！")
    else:
        # #归一化:为啥归一化还通不过检验？！！
        # X0_stadard=X.apply(lambda x:(x-min(X))/(max(X)-min(X)))
        # 级比检验
        a_start = math.e ** (-2 / (len(X0) + 1));
        a_end = math.e ** (2 / (len(X0) + 1))
        X0_list = X0.values
        X0_lamda = [X0_list[i - 1] / X0_list[i] for i, j in enumerate(X0_list) if i > 0]
        print(X0_lamda, a_start, a_end)
        for each in X0_lamda:
            if each < a_start or each > a_end:
                print(each, a_start, a_end)
                return HttpResponse('不能通过灰色预测的级比检验，无法对{}{}{}的分数线进行预测。'.format(province, classy, batch))
            else:
                # 累加
                X1 = X0.cumsum()
                # 辨识算法:计算a,b:'numpy.ndarray'
                a = Identification_Algorithm(X0,0.4,0.6)  # 算式中会有X0到累加X1的计算，所以放入X0即可
                print('a矩阵为:\n', a)

                # 检验|a|是否小于2:适用范围是与发展系数。相关的

                if abs(a[0]) > 2:
                    return HttpResponse('发展系数不能通过检验，无法对{}{}{}的分数线进行预测。'.format(province, classy, batch))
                else:
                    # GM(1,1)模型
                    tmp = np.array([i + 1 for i in range(len(X0) + 1)])  # tmp = np.array([1,2,3,4,5,6])#6个数据
                    # XK = GM_gbnn(X0, X1, a, tmp)
                    # # 预测值还原
                    # X_Return = Return(XK)
                    #
                    # # 预测值即预测值精度表
                    # X_Compare1 = np.ones(len(X0))
                    # X_Compare2 = np.ones(len(X0))
                    # for i, j in enumerate(X0.index):
                    #     X_Compare1[i] = X0[j] - X_Return[j]
                    #     X_Compare2[i] = X_Compare1[i] / X0[j] * 100
                    # Compare = {'GM': XK, '1—AGO': np.cumsum(X0), 'Returnvalue': X_Return, 'Realityvalue': X0,
                    #            'Error': X_Compare1, 'RelativeError(%)': X_Compare2}
                    # X_Compare = DataFrame(Compare, index=X0.index)
                    # # X_Compare = DataFrame(Compare,index=date)
                    # print('预测值即预测值精度表')
                    # print(X_Compare)
                    XK = GM_Model(X0, X1, a, tmp)

                    # 预测值还原
                    X_Return = Return(XK)

                    # 预测值即预测值精度表
                    X_Compare1 = np.ones(len(X0))
                    X_Compare2 = np.ones(len(X0))
                    for i, j in enumerate(X0.index):
                        X_Compare1[i] = X0[j] - X_Return[j]
                        X_Compare2[i] = X_Compare1[i] / X0[j] * 100
                    Compare = {'GM': XK, '1—AGO': np.cumsum(X0), 'Returnvalue': X_Return, 'Realityvalue': X0,
                               'Error': X_Compare1, 'RelativeError(%)': X_Compare2}
                    X_Compare = DataFrame(Compare, index=X0.index)
                    # X_Compare = DataFrame(Compare,index=date)
                    print('预测值即预测值精度表')
                    print(X_Compare)

                    x0_mean = np.mean(X0)
                    s1 = np.sqrt(sum([(x - x0_mean) ** 2 for x in X0.values]) / (len(X0) - 1))
                    X_theta = X0 - X_Return[:-1]
                    theta = np.sum(X0 - X_Return) / len(X0)  # series必须用np.sum;list用sum即可
                    s2 = np.sqrt(sum([(t - theta) ** 2 for t in X_theta.values]) / (len(X0) - 1))
                    print(X0)
                    print(X_Return)
                    print(s1);
                    print(s2)
                    c = s2 / s1
                    print(c)
                    res_list = []
                    for index, value in X_average.iteritems():
                        res_dict = {}
                        res_dict[u'年份'] = index;
                        res_dict[u'分数'] = value
                        res_list.append(res_dict)
                    line_t=models.nation_line_lowest.objects.filter(classy=classy,province=province,year=2017)
                    line_17= [str(each.line) for each in line_t][0]
                    print(line_17)
                    if c <= 0.35:
                        res_dict = {}
                        res_dict['2017'] = round((X_Return.values)[-1], 0)+int(line_17)
                        res_dict[u'预测精度等级'] = '优'
                        res_list.append(res_dict)

                    elif c <= 0.5:
                        res_dict = {}
                        res_dict['2017'] = round((X_Return.values)[-1], 0)+int(line_17)
                        res_dict[u'预测精度等级'] = '合格'
                        res_list.append(res_dict)
                    elif c <= 0.65:
                        res_dict = {}
                        res_dict['2017'] = round((X_Return.values)[-1], 0)+int(line_17)
                        res_dict[u'预测精度等级'] = '勉强合格'
                        res_list.append(res_dict)
                    elif c > 0.65:
                        res_dict = {}
                        res_dict[u'预测精度等级'] = '不合格：灰色模型后验差比>0.65，未通过检验，故不能对{}{}{}进行灰色预测。'.format(province,classy, batch)

                        res_list.append(res_dict)
                    # 简单说就是dump需类似于文件指针的参数就是说可以将dict转成str然后存入文件中；而dumps直接给的是str，也就是将字典转成str。
                    res_json = json.dumps(res_list, ensure_ascii=False)  # 编码不对！
                    return HttpResponse(res_json,content_type='application/json; charset=utf-8')

