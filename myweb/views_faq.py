from myweb.models import *
from django.http import JsonResponse
import logging
logger = logging.getLogger('sourceDns.webdns.views')  # 刚才在setting.py中配置的logger
from myweb.faq import *
from myweb.fcm_cluster import *
def aq(request):
    request.session.set_expiry(0)  # 用户关闭浏览器session就会失效。秒数\时间戳等
    search_dict = {'1': '历年国家线查询', '2': '历年高校录取线查询', '3': '历年高校专业录取线查询',
                   '4': '心仪高校的专业排名', '5': '心仪专业的全国排名', '6': '心仪高校的全国排名', '7': '心仪高校的基本信息'}
    roads_search = {'历年高校专业录取线查询': {'school': '高校名称', 'province': '生源地（省份）', 'classy': '科类（文/理科）', 'major': '专业名称'},
                    '历年高校录取线查询': {'school': '高校名称', 'province': '生源地（省份）', 'classy': '科类（文/理科）'},
                    '历年国家线查询': {'province': '生源地（省份）', 'classy': '科类（文/理科）'},  # 不要批次
                    '心仪高校的专业排名': {'school': '高校名称', 'major': '专业名称'},  # rank:排名？
                    '心仪专业的全国排名': {'major': '专业名称'},  # 'rank': ''
                    '心仪高校的全国排名': {'school': '高校名称'},
                    '心仪高校的基本信息': {'school': '高校名称'}
                    }
    roads_recommend = {
        '推荐学校': {'province': '生源地（省份）', 'classy': '科类（文/理科）', 'point': '分数', 'want_province': '想去的省份'},
        '推荐专业': {},
        '学校及专业推荐': {'province': '生源地（省份）', 'classy': '科类（文/理科）', 'point': '分数',
                    'isMajor': '是否有专业目标(若有则输入<专业名称>、无则输入<无>）'},  # y/n
        '是否能被录取': {'school': '高校名称', 'province': '生源地（省份）', 'classy': '科类（文/理科）', 'point': '分数'},  # 专业可有可无
        '专业前景': {'major': '专业名称'},
        '复读': {},
    }
    print("绘画id" ,request.session.session_key)
    if 1== 1:  # 你就没明白这个函数的作用  晚上跟你说吧 先这么写  这里先放着  我晚上回去再凶你!!!
        # while True: #??什么鬼  一直循环干什么?那request.session['category_session_number']为什么错了？
        q = request.GET["value"]
        print(q)
        seg_q = getJieba(q.strip(), custom=True)
        q1, q0, info = getPronoun(seg_q)
        for k, v in info.items():
            request.session[k] = v
        print('hh')
        if not request.session.exists('category'):
            category = questionCategory(q)
            request.session["category"] = category
        else:
            category = request.session["category"]

        # TODO 非测试的时候删掉
        request.session['category'] = '历史数据查询'
        # request.session['category'] = '推荐学校'
        print(222, category)
        print(request.session['category'])

        if request.session['category'] == '历史数据查询':  # 检查 用户session的随机字符串 在数据库中是否存在，如果不存在则会报错
            request_session = dict(request.session)
            print(request_session)
            print("是否存在category_session_number", 'category_session_number' in request_session)

            if 'category_session_number' in request_session.keys():  # 我的问题  这里报错了 语法不对
                print('i am here!')
                request.session['category_session_number'] += 1
                print(request.session['category_session_number'])
            else:
                print('i am not here!')
                request.session['category_session_number'] = 0
                print(request.session['category_session_number'])
            category_session_number = request.session['category_session_number']
            print('category_session_number', category_session_number)
            if category_session_number == 0:
                answer = question(search=True)
                return JsonResponse({"answer": answer})
            else:#request.session['category_session_number']
                print("q=", str(q))
                if str(q) in list('1234567'):  # isdigit()
                    print('hhhh')
                    category_search = search_dict[q]
                    request.session['category_search'] = category_search
                    print(category_search)
                    info = dict(request.session)  # 这里debug不到了
                    print(info)
                    # print(request_session)
                    while isLack(roads_search, category_search, info):
                        # 这里只要缺少就会一直提问？提问以后呢，怎么提取关键词？
                        # 先假设用户只一次补充就能成功！
                        question_lack = getLack(roads_search, category_search, info)
                        return JsonResponse({"answer": question_lack})

                info = dict(request.session)  # 这里debug不到了
                print('info', info)
                category_search = request.session['category_search']  # 因为category_search不能在第二次传递
                print('category_search', category_search)
                print(info['classy'])
                print(info['province'])
                print("渲染")
                print(category_search,info)
                data_str = searchInfo(category_search, info)
                print(data_str)
                return JsonResponse({'answer': data_str})

        elif request.session['category'] == '推荐学校':
            request_session=dict(request.session)
            if 'category_session_number' in request_session.keys():
                request.session['category_session_number']+=1
            else:
                request.session['category_session_number']=0

            category_session_number=request.session['category_session_number']
            request_session = dict(request.session)

            if category_session_number == 0:
                answer=question(province=True)

                request.session['want_province']=''
                request_session = dict(request.session)
                print(request_session)
                return JsonResponse({'answer': answer})
            else:
                request_session = dict(request.session)
                print('request_session',request_session)
                if 'want_province'in request_session.keys() and request_session['want_province']=='':
                    print('1111request.session.items')
                    print(request.session.items())#dict_items([('category', '推荐学校'), ('category_session_number', 1),
                    request.session['want_province']=info['province']
                    print(request.session.items())
                    del request.session['province']#删不掉？
                    print(request.session.items())

                    print('2222')
                    info = dict(request.session)
                    print('info',info)
                    category = request.session["category"]#TODO 非测试的时候删掉
                    while isLack(roads_recommend, category, info):
                        print('3333')
                        question_lack = getLack(roads_recommend, category, info)
                        print('4444')
                        return JsonResponse({"answer": question_lack})
                elif 'want_province'in request_session.keys() and request_session['want_province']!=''and 'province'in request_session.keys():

                    print('5555')
                    info = dict(request.session)
                    print(info)
                    answer=recommendSchool(info)
                    print(answer)
                    return JsonResponse({'answer':answer})


    #     elif roads_recommend[category] == '推荐专业':
    #         # info=lack(roads_recommend,category,info)
    #         q = question(ismajor=True)
    #         if q == 'y':
    #             return 'http:......'  # ?????????????????????????
    #         else:
    #             aqSystem(q)
    #
    #     elif roads_recommend[category] == '学校及专业推荐':
    #         info = lack(roads_recommend, category, info)
    #         if info['ismajor'] == '无':
    #             # 您 的分数适合的学校有：
    #             # recommend_school()
    #             # 关于专业可以跳转到专业测试链接
    #             return 'http....'
    #         else:
    #             print('niuzitong')
    #             # recommend_major&school()
    #     elif roads_recommend[category] == '是否能被录取':
    #         info = lack(roads_recommend, category, info)
    #         if 'major' not in info.keys():
    #             print('niuzitong')
    #             # recommend_school()
    #         else:
    #             print('niuzitong')
    #
    #             # recommend_major&school()
    #     elif roads_recommend[category] == '专业前景':
    #         info = lack(roads_recommend, category, info)
    #         return 'sql....'
    #     elif roads_recommend[category] == '复读':
    #         info = lack(roads_recommend, category, info)
    #         return '一些需要考虑的现实问题'
    #     else:
    #         q = question(noUnderstand=True)
    #         aqSystem(q)
    # print(info)

    # request.session["category_comment"] = roads_recommend[category]
    # print(333)
    # for key,value in info:
    #     print(key, value)
    #     request.session["category_comment"][key] = value
    #
    # if  request.session["category"]=="学校及专业推荐":
    # TODO
    # info,anwser=aqSystem(value)

    # 思路
    # 1. 分词 找到关键字
    # 1.1 如果没有关键字 就返回让他补全关键字 回到步骤1
    # 2. 根据关键字 调用不同的view视图（函数） 将关键字词传入输入
    # 3. 应用faq的模型
    # 4. 返回结果

    # TODO 写算法
    return JsonResponse({"answer": 456})

def recommendSchool(info):
    want_province =info['want_province']
    user_point=info['point']
    province =info['province'][0]
    classy = info['classy'];
    print(want_province,user_point,province,classy)
    if province in ['西藏']:
        return JsonResponse('西藏、新疆少数民族分布情况复杂，暂不提供推荐。<br>')
    else:
        batches17 = ['第二批', '第三批','本科']  # '第一批',艺术类本科，体育教育，专科提前批，专科，
        line17_sql = nation_line.objects.filter(province=province, year=2017,classy=classy, batch__in=batches17)
        lines17 = []
        for i in line17_sql: lines17.append(int(i.line))
        line17 = min(lines17)
        sub_point = user_point - line17
        print(line17, user_point)
        if sub_point < 0:  # 如果该用户的分数没有达到二本分数线，暂不推荐。
            response = '本系统暂不提供本科以外的推荐。<br>'
            return JsonResponse(response)
        else:
            if want_province == ['无']:
                want_province = ['北京', '天津', '辽宁', '吉林', '黑龙江', '上海', '江苏', '浙江', '安徽', '福建', '山东', '湖北', '湖南',
                                 '广东', '重庆', '四川', '陕西', '甘肃', '河北', '山西', '内蒙古', '河南', '海南', '广西', '贵州', '云南',
                                 '西藏', '青海', '宁夏', '新疆', '江西', '香港', '澳门', '台湾']
            # 构造候选集：选择生源地、年份（？？选哪些年份呢）、科类、专业符合要求，(学校最低分)=<分数差值<=学校最高分+10
            print('11111111111')
            candidate_school_0 = univer_info.objects.filter(province__in=want_province)
            candidate_school_1 = []  # 学校候选集candidate_school_1：就是选出学生想去的省份的学校
            [candidate_school_1.append(each.school) for each in candidate_school_0]
            print(candidate_school_1)
            # 提取国家线这个表
            pro_univer_nation_line_t = pro_univer_nation_line.objects.filter(
                province=province, school__in=candidate_school_1, classy=classy)
            pro_univer_nation_line_df = pd.DataFrame.from_records(pro_univer_nation_line_t.values())
            # 必须先转换成浮点类型才能median（）；astype(int)不能转换NaN哦
            print(pro_univer_nation_line_df)
            pro_univer_nation_line_df[['d_hl', 'd_h', 'd_l', 'd_a', 'd_ha', 'd_la']] = \
            pro_univer_nation_line_df[
                ['d_hl', 'd_h', 'd_l', 'd_a', 'd_ha', 'd_la']].astype(np.float)
            # 使用中位数填充空缺值
            # d_l居然有小于0的异常值，全部剔除！！#去掉groupby后的 'batch'
            grouped = pro_univer_nation_line_df[
                (pro_univer_nation_line_df['d_l'] >= 0) & (pro_univer_nation_line_df['d_a'] >= 0)].groupby \
                (['school', 'province', 'classy'], as_index=False)[
                ['d_hl', 'd_h', 'd_l', 'd_a', 'd_ha', 'd_la']].median()

            candidate_school_dict = {}
            for index, row in grouped.iterrows():  # values返回numpy类型
                if np.isnan(row['d_l']) == False and np.isnan(row['d_h']) == False:
                    if row['d_l'] <= sub_point < row['d_h']:
                        if row['school'] not in candidate_school_dict:
                            print('最低分：' + str(row['d_l']) + '最高分：' + str(row['d_h']) + str(
                                row['school']) + '：在最高分与最低分之间！')
                            candidate_school_dict[row['school']] = 0
                elif np.isnan(row['d_l']) == False and np.isnan(row['d_a']) == False:
                    if row['d_l'] <= sub_point < row['d_a'] + 15:
                        if row['school'] not in candidate_school_dict:
                            print('最低分：' + str(row['d_l']) + '平均分：' + str(row['d_a']) + str(
                                row['school']) + '：在最低分与平均分+15之间！')
                            candidate_school_dict[row['school']] = 0
                elif np.isnan(row['d_l']) == False and np.isnan(row['d_a']) == True and np.isnan(
                        row['d_h']) == True:
                    if row['d_l'] <= sub_point < row['d_l'] + 30:
                        if row['school'] not in candidate_school_dict:
                            print('最低分：' + str(row['d_l']) + '.' + str(row['school']) + '：在最低分与最低分+30之间！')
                            candidate_school_dict[row['school']] = 0
                elif np.isnan(row['d_l']) == True and np.isnan(row['d_a']) == False and np.isnan(
                        row['d_h']) == True:
                    if row['d_a'] - 5 <= sub_point < row['d_a'] + 5:
                        if row['school'] not in candidate_school_dict:
                            print('平均分：' + str(row['d_a']) + '.' + str(row['school']) + '：在平均分+-5之间！')
                            candidate_school_dict[row['school']] = 0
                else:
                    pass

            sql = '''
                select school,
            sum(case  year when '2010' then d_a else  null end)as  '2010',
            sum(case year when  '2011' then d_a else  null  end)as  '2011',
            sum(case  year when '2012' then d_a else null end  )as  '2012',
            sum(case  year when '2013' then d_a else null end  )as  '2013',
            sum(case year  when '2014' then d_a else null end  )as  '2014',
            sum(case year  when '2015' then d_a else null end ) as  '2015',
            sum(case year  when '2016' then d_a else null end ) as  '2016'
            from myweb_pro_univer_nation_line a where a.classy='理科' and a.province='四川' and a.school in {} group by a.school''' \
                .format(tuple(candidate_school_dict.keys()))
            print(candidate_school_dict)
            connect = pymysql.Connect(host='172.19.235.29', user='root', passwd='mingming', port=3306,
                                      db='gaokao_py3', charset='utf8')
            cursor = connect.cursor()
            cursor.execute(sql)
            data_t = cursor.fetchall()
            cursor.close()

            data_df = pd.DataFrame.from_records(list(data_t))
            print(data_df)
            list_fillna = []
            for k, v in data_df.iterrows():
                v0 = v[:1]
                v1 = v[1:]
                v1.fillna(v1.mean(), inplace=True)
                v = v0.append(v1)
                list_fillna.append(v.values)
            data_fillna = pd.DataFrame(list_fillna)
            print(data_fillna.ix[:, 1:])
            data_fillna.dropna(how='any', inplace=True)
            # print(data_fillna)
            points = np.array(data_fillna.ix[:, 1:], dtype=np.int)
            # # 给定同维向量数据集合points,数目为n,将其聚为C类（在矩阵U里面体现），m为权重值,u为初始匹配度矩阵（n*C，和为1）,采用闵式距离算法,其参数为p,迭代终止条件为终止值e（取值范围(0，1））及终止轮次。
            # points=np.array(data,dtype=np.int)#[1 4 21 ..., 0 Decimal('15') 0]:decimal模块用于十进制数学计算'decimal.Decimal' and 'float'
            p = 2;
            m = 2;
            e = 0.1;
            terminateturn = 1000
            u0 = np.random.rand(len(points), 3)  # np.random.random(3) vs  np.random.rand(3,5)
            u = np.array([x / np.sum(x, axis=0) for x in u0])  # for x in u0呈现的是行，所以用行axis=0即可
            # 其中p是一个变参数。当p=1时，就是曼哈顿距离;当p=2时，就是欧氏距离;当p→∞时，就是切比雪夫距离;闵氏距离可以表示一类的距离。
            print('u\n', u)
            print('points\n', points)
            fcm_res = alg_fcm(points, u, m, p, e, terminateturn)  # u2, k, centroids
            print('res1\n', fcm_res[1], '\nres0\n', fcm_res[0])
            print('res2\n', fcm_res[2])
            centroids = [each.mean() for each in fcm_res[2]]
            centroids_sort = pd.Series(centroids).sort_values()
            centroids_dict = {}
            print(centroids_sort)
            for i, j, k in zip(['保', '稳', '冲'], centroids_sort.values, centroids_sort.index):
                print(i, j, k)
                centroids_dict[k] = i
            print(centroids_dict)

            labels = [each.index(max(each)) for each in fcm_res[0]]  # list.index(max(each))找出最大值对应的索引index
            print('labels\n', labels)
            labels_bwc = [centroids_dict[i] for i in labels]
            print('labels_bwc\n', labels_bwc)
            candidate_school_2 = data_df[0].values
            print(candidate_school_2)

            school_dict = {'development': {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1},
                           'provincial_capital': {0: 0, 1: 4},
                           'is_doctor': {0: 0, 1: 1}, 'is_master': {0: 0, 1: 1},
                           'leixing_int': {0: 0, 1: 3, 2: 5}};
            candidate_school_t = univer_info.objects.filter(school__in=candidate_school_2)
            print('candidate_school_t\n', pd.DataFrame.from_records(candidate_school_t.values()))
            for each in candidate_school_t:
                candidate_school_dict[each.school] += (school_dict['development'][int(each.development)] +
                                                       school_dict['provincial_capital'][
                                                           int(each.provincial_capital)]) / 10 * 0.43 \
                                                      + (school_dict['leixing_int'][int(each.leixing_int)] +
                                                         school_dict['is_doctor'][int(each.is_doctor)] \
                                                         + school_dict['is_master'][
                                                             int(each.is_master)]) / 10 * 0.57
            candidate_school_list = [(k, v) for k, v in candidate_school_dict.items()]
            candidate_school_list = sorted(candidate_school_list, key=lambda d: d[1],
                                           reverse=True)  # 先排序，不要最后才排序
            print(candidate_school_list)
            bwc_dict = {}
            b, w, c = [], [], []
            for k, j, v in zip([each[0] for each in candidate_school_list],
                               [each[1] for each in candidate_school_list], labels_bwc):
                if v == '保': b.append((k, j))
                if v == '稳': w.append((k, j))
                if v == '冲': c.append((k, j))
            bwc_dict['bao'] = b;
            bwc_dict['wen'] = w;
            bwc_dict['chong'] = c;  # {'wen': [('云南农业大学', 0.501), ('重庆邮电大学', 0.444),...
            print(bwc_dict)
            answer='根据您的信息，我们为您推荐的学校有：<br>'
            print(answer)
            for key, value in bwc_dict.items():
                print(key, value)
                # for each in value:
                if key == 'wen':
                    answer_ = '<strong>稳：</strong><br>'
                    c = 0
                    for each in value:
                        if c != len(value) - 1:
                            answer_ += str(each[0]) + '、'
                        else:
                            answer_ += str(each[0]) + '；<br>'
                        c += 1
                elif key == 'bao':
                    answer_ = '<strong>保：</strong><br>'
                    c = 0
                    for each in value:
                        if c != len(value) - 1:
                            answer_ += str(each[0]) + '、'
                        else:
                            answer_ += str(each[0]) + '；<br>'
                        c += 1
                elif key == 'chong':
                    answer_ = '<strong>冲：</strong><br>'
                    c = 0
                    for each in value:#
                        if c != len(value) - 1:
                            answer_ += str(each[0]) + '、'
                        else:
                            answer_ += str(each[0]) + '；<br>'
                        c += 1
                print(answer_)
                print(answer)
                answer += answer_
            print('hhhhhh')
            print(answer)
            return answer
