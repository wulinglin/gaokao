# -*- coding: utf-8 -*-
from django.shortcuts import render,HttpResponse,render_to_response
import sys,json
sys.path.append('G:\MY\gaokao\web\myweb')
from myweb.models import *
from django.db import connection
from django.core import serializers
import pandas as pd
import pymysql
import numpy as np
from  numpy import nan as nan

def index(request,html):
    return render(request,html)

def search(request):
    province_t=province.objects.all()
    province_df = pd.DataFrame.from_records(province_t.values())
    provinces = list(province_df['province'].values)
    school_type_t = school_type.objects.all()
    school_type_df = pd.DataFrame.from_records(school_type_t.values())
    school_types = list(school_type_df['school_type'].values)
    year_t = year.objects.all()
    year_df = pd.DataFrame.from_records(year_t.values())
    years = list(year_df['year'].values)
    return render(request,'search.html',{'provinces':provinces,'school_types':school_types,'years':years})

def nationLine(request):#nationLine
    error=False
    if 'year' in request.GET and 'classy' in request.GET and 'province' in request.GET and 'batch' in request.GET:
        year = request.GET.get('year');
        classy = request.GET.get('classy');
        province = request.GET.get('province');
        batch = request.GET.get('batch');
        if  year=='全部' or classy=='全部' or province=='全部' or batch=='全部':
            # error=True
            nation_line_t = nation_line.objects.all()
            nation_line_df = pd.DataFrame.from_records(nation_line_t.values())#.sort_values(by=['year'],ascending=False)
            nation_line_json = nation_line_df.to_json(orient='index', force_ascii=False)
            return HttpResponse(nation_line_json, content_type='application/json;charset=utf8')
        else:
            if year == '全部':
                nation_line_t = nation_line.objects.filter(classy=classy, province=province, batch=batch)
                nation_line_df=pd.DataFrame.from_records(nation_line_t.values()).sort_values(by=['year'])
                if len(nation_line_df) < 1:
                    return HttpResponse('找不到历史相关数据，请改变您的查询条件！')
                else:
                    categories=list(nation_line_df['year'].values)
                    data=list(nation_line_df['line'].values)
                    print(categories,data)
                    return render_to_response('search_nationLine.html',{'categories': categories, 'data': data},)
            else:
                nation_line_t = nation_line.objects.filter(year=year, classy=classy, province=province, batch=batch)
                nation_line_df = pd.DataFrame.from_records(nation_line_t.values())
                if len(nation_line_df) < 1:
                    return HttpResponse('找不到历史相关数据，请改变您的查询条件！')
                else:
                    nation_line_json=nation_line_df.to_json(orient='index',force_ascii=False)
                    print(nation_line_json)
                    return HttpResponse(nation_line_json,content_type='application/json;charset=utf8')
    return render(request,'search.html', {'error': error})#加上这个是错的 content_type='application/json;charset=utf8'

def majorLine(request):
    error=False
    if 'school'in request.GET and 'major'in request.GET and 'classy'in request.GET  and 'province'in request.GET :
        school=request.GET.get('school');major=request.GET.get('major');classy=request.GET.get('classy')
        province=request.GET.get('province');#batch=request.GET.get('batch')
        print(school,major,classy,province)
        if school=='请输入高校名' or major=='请输入专业名' or classy=='全部' or province=='全部' :
            # error=True
            univer_major_line_t = univer_major_line.objects.all()
            univer_major_line_df = pd.DataFrame.from_records(univer_major_line_t.values())
            univer_major_line_json = univer_major_line_df.to_json(orient='index', force_ascii=False)
            return HttpResponse(univer_major_line_json, content_type='application/json;charset=utf8')
        else:
            univer_major_line_t = univer_major_line.objects.filter(school__contains=school, major__contains=major, classy=classy,province=province)
            univer_major_line_df=pd.DataFrame.from_records(univer_major_line_t.values())
            print(univer_major_line_df)
            if len(univer_major_line_df) <1:
                return HttpResponse('找不到历史相关数据，请改变您的查询条件！')
            else:
                univer_major_line_json=univer_major_line_df.to_json(orient='index',force_ascii=False)
                return HttpResponse(univer_major_line_json,content_type='application/json;charset=utf8')
    return render_to_response('search.html',{'error':error})


def schoolLine(request):#schoolLine的结果和相似学校
    error=False
    if 'school' in request.GET and 'province' in request.GET and 'classy' in request.GET:
        school = request.GET.get('school');classy = request.GET.get('classy')
        province = request.GET.get('province');
        print(school)#value='请输入高校名'
        if school=='请输入高校名' or province=='全部' or classy=='全部':
            # error=True
            pro_univer_point_t = pro_univer_point.objects.all()
            pro_univer_point_df = pd.DataFrame.from_records(pro_univer_point_t.values())
            pro_univer_point_json = pro_univer_point_df.to_json(orient='index', force_ascii=False)
            return HttpResponse(pro_univer_point_json, content_type='application/json;charset=utf8')
        else:
            pro_univer_point_t=pro_univer_point.objects.filter(school__contains=school,province=province,classy=classy)
            pro_univer_point_df = pd.DataFrame.from_records(pro_univer_point_t.values())
            print(pro_univer_point_df)
            print(pd.unique(pro_univer_point_df['school']))
            if len(pro_univer_point_df) < 1:
                return HttpResponse('找不到历史相关数据，请改变您的查询条件！')
            if len(pd.unique(pro_univer_point_df['school'])) == 1:
                pro_univer_point_dict = pro_univer_point_df.to_dict(orient='records')#没有orient=
                print(pro_univer_point_dict)
                school=pro_univer_point_dict[0]['school']
                sim_dict=similarity(pro_univer_point_df, school, province, classy)
                print(sim_dict)
                res_dict={}
                res_dict['search_res']=pro_univer_point_dict
                res_dict['sim_recommend']=sim_dict
                print(res_dict)
                res_json=json.dumps(res_dict,ensure_ascii=False)
                return HttpResponse(res_json,content_type='application/json;charset=utf8')
            else:
                pro_univer_point_json = pro_univer_point_df.to_json(orient='index', force_ascii=False)
                return HttpResponse(pro_univer_point_json, content_type='application/json;charset=utf8')

    return render_to_response('search.html',{'error':error})

def schoolInfo(request):
    school=request.GET.get('school');leixing=request.GET.get('leixing');lishuyu=request.GET.get('lishuyu')
    province = request.GET.get('province');#有无硕士点博士点数据可能不对！is_master=request.GET.get('is_master');is_doctor=request.GET.get('is_doctor');
    school_type=request.GET.get('school_type')
    lishuyu_dict = {'其它':['-----'],'教育部':['教育部'],'其它中央部委':['上海市教育委员会','中华全国妇女联合会','中华全国总工会',\
         '中国共产主义青年团中央', '中国民用航空总局','中国科学院', '交通运输部', '公安部', '北京市教育委员会', '卫生部', '司法部',\
        '国务院侨务办公室', '国家体育总局','国家安全生产监督管理局', '国家民族事务委员会', '外交部', '天津市教育委员会','中国地震局', \
        '工业与信息化部', '新疆生产建设兵团', '重庆市教育委员会'], '地方所属':['云南省教育厅','内蒙古自治区教育厅','吉林省教育厅',\
         '四川省教育厅','宁夏回族自治区教育厅','安徽省教育厅','陕西省教育厅','青海省教育厅','黑龙江省教育厅',\
        '山东省教育厅','山西省教育厅','广东省教育厅','广西壮族自治区教育厅','新疆维吾尔自治区教育厅','江苏省教育厅', \
         '江西省教育厅','河北省教育厅','河南省教育厅','浙江省教育厅','海南省教育厅','湖北省教育厅','湖南省教育厅', \
        '甘肃省教育厅','福建省教育厅','西藏自治区教育厅','贵州省教育厅','辽宁省教育厅']}
    leixing_dict={"高校属性":["211,985","211","-----"],"211,985":["211,985"],"211":["211"],"其它":["-----"]}
    province_dict={'全国':['北京', '天津', '辽宁', '吉林', '黑龙江', '上海', '江苏', '浙江', '安徽', '福建', '山东', '湖北', '湖南',
     '广东', '重庆', '四川', '陕西', '甘肃', '河北', '山西', '内蒙古', '河南', '海南', '广西', '贵州', '云南',
     '西藏', '青海', '宁夏', '新疆', '江西', '香港', '澳门', '台湾']}
    school_type_dict={'高校类型':["综合", "工科", "师范", "农林", "医药", "语言", "财经", "体育", "政法", "艺术", "民族", "海洋"]}
    # 有学校名只需学校名即可查询
    if school !='请输入高校名':
        univer_info_t=univer_info.objects.filter(school__contains=school)
    #无学校名则需别的条件筛选
    else:
        leixing=leixing_dict[leixing]
        if lishuyu=='全部':
            lishuyu=[];[lishuyu.extend(v) for k,v in lishuyu_dict.items()]
            lishuyu=lishuyu
        else:lishuyu=lishuyu_dict[lishuyu]
        if school_type== '全部':school_type= school_type_dict[school_type]
        else:school_type = [school_type]
        if province=='全部':province = province_dict[province]
        else:province=[province]
        univer_info_t=univer_info.objects.filter(leixing__in=leixing,lishuyu__in=lishuyu,school_type__in=school_type,province__in=province)
# .values('school','leixing','lishuyu','school_type','province', 'academician','doctor','master','address','telephone','website','email','post','city'
# 'provincial_capital','development','school_rank')\
        print(leixing,lishuyu,school_type,province)
    univer_info_df=pd.DataFrame.from_records(univer_info_t.values())
    if len(univer_info_df) < 1:
        return HttpResponse('找不到历史相关数据，请改变您的查询条件！')
    univer_info_json=univer_info_df.to_json(orient='index',force_ascii=False,)
    return HttpResponse(univer_info_json, content_type='application/json;charset=utf8')
    # univer_info_json=json.dumps(list(univer_info_t),ensure_ascii=False)#是queryset对象(<QuerySet [{'master': 54,...)，里面是dict，需要先转换成list.
    # univer_info_json=serializers.serialize('json',univer_info_t,ensure_ascii=False)#{"model": "myweb.univer_info", "pk": 1, "fields": {"school":
    #json串行器只能在使用values('school','leixing',...)的时候使用。why??

def similarity(pro_univer_point_df, school, province, classy):

    sql_univer_info = '''
    select b.school,
        b.major	
from myweb_univer_info a	
left join 
(select school,group_concat(major_new,',',classes,',',catalogue Separator ',') as major from myweb_univer_major_line_lb  where year =2015 and school is not null  group by school ) b 
on a.school=b.school; 
'''
    sql_univer_line = '''
        select c.school_a,c.school_b,c.province,c.classy,c.batch,count(*) cnt,abs(round(avg(d_d_a),2))  from 
        (select a.school school_a,b.school school_b,a.province,a.classy,a.year,a.batch,a.d_a d_a_a,b.d_a d_a_b,(a.d_a-b.d_a)d_d_a	
        from myweb_pro_univer_nation_line a
        left join myweb_pro_univer_nation_line b
        on  a.province =b.province  and a.classy=b.classy and a.year=b.year and a.batch=b.batch
        where a.school ='{}' and a.province='{}' and a.classy='{}' and b.school !='{}' )c
        group by c.school_a,c.school_b,c.province,c.classy,c.batch;
    '''.format(school, province, classy, school)  # province是否加入？加入可能会运行速度更快，但是也需要用户提供更多的信息才可,且可能有更多的局限数据！
    sql_major_ratio = 'select * from major_ratio'
    sql_school_major_ratio = '''
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
    sql_univer_info3 = '''
    select a.school,a.leixing,a.province,a.provincial_capital,a.development,a.school_type,
        case 	when lishuyu in ('-----') then '其它'
                when lishuyu in ('教育部') then '教育部' 
                when lishuyu in ('上海市教育委员会','中华全国妇女联合会','中华全国总工会','中国共产主义青年团中央','中国地震局',
                            '中国民用航空总局','中国科学院', '交通运输部', '公安部', '北京市教育委员会', '卫生部', '司法部',
                             '国务院侨务办公室', '国家体育总局','国家安全生产监督管理局', '国家民族事务委员会', '外交部', '天津市教育委员会', 
                             '工业与信息化部', '新疆生产建设兵团', '重庆市教育委员会') then '其它中央部委'
                when lishuyu in ('云南省教育厅','内蒙古自治区教育厅','吉林省教育厅','四川省教育厅','宁夏回族自治区教育厅','安徽省教育厅',
                     '山东省教育厅','山西省教育厅','广东省教育厅','广西壮族自治区教育厅','新疆维吾尔自治区教育厅','江苏省教育厅',
                     '江西省教育厅','河北省教育厅','河南省教育厅','浙江省教育厅','海南省教育厅','湖北省教育厅','湖南省教育厅',
                     '甘肃省教育厅','福建省教育厅','西藏自治区教育厅','贵州省教育厅','辽宁省教育厅','陕西省教育厅','青海省教育厅',
                     '黑龙江省教育厅') then '地方所属'
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
    connect = pymysql.Connect(host='172.19.235.29', user='root', passwd='mingming', port=3306,
                              db='gaokao_py3', charset='utf8')
    cursor = connect.cursor()
    cursor.execute(sql_univer_info)
    univer_info_t = cursor.fetchall()  # 这里只得到元组类型
    cursor.execute(sql_univer_line)
    univer_line_t = cursor.fetchall()
    cursor.execute(sql_major_ratio)
    major_ratio = cursor.fetchall()
    cursor.execute(sql_school_major_ratio)
    school_major_ratio = cursor.fetchall()
    cursor.close()

    univer_line_df = pd.DataFrame.from_records(list(univer_line_t))
    print(sql_univer_line, univer_line_df)
    univer_line_df = univer_line_df.groupby([0, 1, 2, 3, ], as_index=False)[4, 6].agg({5: 'max'})
    univer_line_df = (univer_line_df[5]).sort_values(by=[6],
                                                     ascending=True)  # 多重索引，[5]是一个包含0.1.2.3的dataframe#sort_values vs sort_index
    univer_line_df = univer_line_df[univer_line_df[6] <= 13]
    univer_line_df = univer_line_df.copy()  # A value is trying to be set on a copy of a slice from a DataFrame.
    univer_line_df['score'] = (univer_line_df[6].max() - univer_line_df[6]) / (
        univer_line_df[6].max() - univer_line_df[6].min())
    print(univer_line_df)  # 34875
    res_rank = pd.Series(univer_line_df['score'].values, index=univer_line_df[1].values)
    print('res_rank\n', res_rank)
    res_rank.fillna(0,inplace=True)
    school_candidates = list(univer_line_df[1].values)  # numpy.ndarray:iterrows里面是

    # def major_sim(major_ratio,school_major_ratio,school_candidates):
    major_ratio = pd.DataFrame.from_records(list(major_ratio))
    school_major_ratio = pd.DataFrame.from_records(list(school_major_ratio))
    major_ratio = major_ratio[major_ratio[0].isin(school_candidates)]
    print(major_ratio, '\n--------------\n', school_major_ratio)
    res_dict = {}
    res_dict[school] = 1
    for index, value in major_ratio.iterrows():  # iteritems():行，iterrows():行
        # print(school_major_ratio.ix[0,1:],'\n-------------',value[1:],major_ratio.ix[index,1:],type(value[0]))
        score = np.sum(abs(school_major_ratio.ix[0, 1:] - value[1:]))
        name = value[0]
        res_dict[name] = score
        # sorted(res_dict.items(), key=lambda d: d[1], reverse=True
    print(res_dict)
    res_major = pd.Series(res_dict)  # list
    res_major.fillna(0, inplace=True)
    print('res_rank\n', res_rank)
    print('res_major\n', res_major)
    res_major=res_major.astype('float64')
    res_rank=res_rank.astype('float64')
    res = res_major + res_rank
    res.dropna(inplace=True)
    print('res\n',res)
    print(pd.DataFrame([res,res_rank,res_major]))
    res_df=(pd.DataFrame([res,res_rank,res_major]).T).sort_values(by=[0],ascending=False)
    print(res_df)
    # res_standard = (res.max() - res) / (res.max() - res.min() + 0.1)
    # res_sort = res.sort_values(ascending=False)
    res_sort=sorted(res.to_dict().items(),key=lambda d:d[1],reverse=True)
    return res_sort#to_dict() got an unexpected keyword argument 'orient'；难道是reries没有？？？

