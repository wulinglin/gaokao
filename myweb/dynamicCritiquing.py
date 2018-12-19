#动态范式评价
import pandas as pd
import numpy as np
from django.shortcuts import HttpResponse,render_to_response,render

def itemRecommend(q,ci):#这里的ci是不是应该传递？？
    #satisfies(ci,q)
    q=list(q)#['?', '1', ' ', '=', '>', ' ', '=', '6'],['>', '8', '，', '?', '2']
    print(q)
    quality={};qualitys={}
    print(ci)
    for k,v in ci.iterrows():#v是series
        # for i,j in zip(['<','>','=','?']:
        count=0#len(s1[1:][s1[1:]==s2[1:]])/len(s1[1:])
        if k==0:
            v1=v.values[eval(q[1])];v2=v.values[eval(q[-1])]
            s1=v.values
            print('s1','\n',s1)
        else:#除去第一行
            if q[1] !='8' and q[-1]!='8':
                    #相似度
                s2=v.values
                similarity=len(s1[1:][s1[1:]==s2[1:]])/len(s1[1:])
                #相容性
                if q[0]=='=':
                    if v.values[eval(q[1])]==v1:count+=1
                elif q[0]=='?':
                    if v.values[eval(q[1])] != v1: count += 1
                elif q[0]=='>':
                    if v.values[eval(q[1])] > v1: count += 1
                elif q[0]=='<':
                    if v.values[eval(q[1])] < v1: count += 1

                if q[4]=='=':
                    if v.values[eval(q[-1])]==v2:count+=1
                elif q[4]=='?':
                    if v.values[eval(q[-1])] != v2: count += 1
                elif q[4]=='>':
                    if v.values[eval(q[-1])] > v2: count += 1
                elif q[4]=='<':
                    if v.values[eval(q[-1])] < v2: count += 1

                compatibility=count/2
                #质量打分
                quality[v.values[0]]=np.float(compatibility) * np.float(similarity)
                qualitys[v.values[0]] = [np.float(compatibility) * np.float(similarity),compatibility,similarity]
            elif q[1] =='8' and q[-1]!='8':
                # 相容性
                if q[0] == '=':
                    if v.values[eval(q[1])] == v1: count += 1
                elif q[0] == '?':
                    if v.values[eval(q[1])] != v1: count += 1
                elif q[0] == '>':
                    if v.values[eval(q[1])] > v1: count += 1
                elif q[0] == '<':
                    if v.values[eval(q[1])] < v1: count += 1
                compatibility = (count+0.1) / 2
                similarity=v.values[eval(q[1])]
                # 质量打分
                quality[v.values[0]] = np.float(compatibility) * np.float(similarity)
                qualitys[v.values[0]] = [np.float(compatibility) * np.float(similarity), compatibility, similarity]
            elif q[1] !='8' and q[-1]=='8':
                if q[4]=='=':
                    if v.values[eval(q[-1])]==v2:count+=1
                elif q[4]=='?':
                    if v.values[eval(q[-1])] != v2: count += 1
                elif q[4]=='>':
                    if v.values[eval(q[-1])] > v2: count += 1
                elif q[4]=='<':
                    if v.values[eval(q[-1])] < v2: count += 1

                compatibility = (count + 0.1) / 2
                print(v.values[eval(q[-1])])
                similarity = float(v.values[eval(q[-1])])
                # 质量打分
                quality[v.values[0]] = np.float(compatibility) * np.float(similarity)
                qualitys[v.values[0]] = [np.float(compatibility) * np.float(similarity), compatibility, similarity]
    quality_sort=sorted(quality.items(),key=lambda d:d[1],reverse=True)
    print('quality_sort',quality_sort)
    print(qualitys)
    r=[each[0] for each in quality_sort][0]
    return r

def compoundCritiques(r, ci, k, sigma):#ci包含了r
    def critiquePatterns(r,ci):
        d = {}
        ci.fillna(0,inplace=True)
        for i in range(len(ci)):
            if i != 0:
                for j in range(9):
                    if j not in [0,1,3,4,6,8]:#2,5,7#?=
                        if ci.ix[i, j] == ci.ix[0, j]:
                            d.setdefault(str(j), []).append('=')
                        else:
                            d.setdefault(str(j), []).append('?')
                    elif j in [1,3,4,6]:#>=<
                        if ci.ix[i, j] < ci.ix[0, j]:
                            d.setdefault(str(j), []).append('<')
                        elif ci.ix[i, j] == ci.ix[0, j]:
                            d.setdefault(str(j), []).append('=')
                        elif ci.ix[i, j] > ci.ix[0, j]:
                            d.setdefault(str(j), []).append('>')
                    elif j in [8]:#>0.5,<0.5
                        print(float(ci.ix[i, j]) )
                        print(ci)
                        print(ci.ix[:,8].median())
                        if float(ci.ix[i, j]) >ci.ix[:,8].median():#测试-1可以，这里就不行？!df_obj.ix[:,-1].median()  0.5   == ci.ix[0, j]:
                            d.setdefault(str(j), []).append('>')
                        else:
                            d.setdefault(str(j), []).append('<')
        print (d)
        df1 = pd.DataFrame(d)
        print(df1)
        return df1#这里不包含第一行？？？

    def appriori(df1):#df1是d过来，索引是字符
        cols=['985/211属性','省份','是否位于省会城市','所在城市经济等级','学校类型','隶属于','地理分区','专业相似度']#不包含school，因为df1已经去掉了第一列
        dic0,dic1,dic2= {},{},{}
        print(df1.columns)
        for i in df1.columns:
            for j in df1.columns:
                if i != j:
                    df2 = df1.groupby([i,j])[j].count()
                    # df2=df1.groupby([i,j]).agg({j: 'count'})#count()##agg不可行！不是value_counts()也不是counts()是count()
                    # print(df2)
                    for m in df1[i].unique():#?  =
                        for x in range(len(df2.index.levels[0])):
                            df3 = df2[m]# df3 = df2[df2.index.levels[0][x] == m]
                            if len(df3) == 1:
                                s0= str(m) + str(i) + ',' + str(df3.index[0]) +str(j)
                                s1 = str(m) + cols[eval(i)-1] + ',' + str(df3.index[0]) + cols[eval(j)-1] # > 1:df3.index[0];df3[0]
                                dic0[s0] = df3[0]/len(df1)
                                dic1[s1]=df3[0]/len(df1)
                                dic2[s0]=s1
                            else:
                                pass
        print(dic0,'\n',dic1,'\n',dic2)
        return dic0,dic1,dic2#dic0,dic1,dic2一个是数字、一个是字段、一个数字对应字段

    def selectCritiques(cc,cc2,k):
        cc_sort=sorted(cc.items(),key=lambda x:x[1],reverse=False)#list嵌套tuple
        cc_k=[each[0] for each in cc_sort][:k]
        print(cc_k)
        sc={}
        for c0 in cc_k:
            sc[c0]=cc2[c0]
        print(sc)
        return sc

    cp=critiquePatterns(r,ci)
    cc0,cc1,cc2=appriori(cp)#一个是数字、一个是字段、一个数字对应字段
    sc=selectCritiques(cc0,cc2,k)

    return sc

# def userReview(r, ci, cc):
#     def critique(r,cc):
#         return render('.html',{'r':r,'cc':cc})
#         cc=request.GET.get('cc')
#         return cc
#
#     q=critique(r,cc)
#
#     ci=ci[q]
#     return ci