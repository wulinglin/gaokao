#参考博客：代码：http://blog.csdn.net/qq_30091945/article/details/54379922
# http://blog.csdn.net/qq547276542/article/details/77865341
import numpy as np
from pandas import Series
from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
from .bpnn import *
def Identification_Algorithm(X0,ratio1,ratio2):    #辨识算法:(BT.B)-1.BT.Y
    B = np.array([[1]*2]*(len(X0)-1))#[[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
    tmp = np.cumsum(X0)#array-->list
    for i,j in enumerate(X0.index):#range(len(x)-1):
        if i < len(tmp) - 1:
            B[i][0] = (ratio1*tmp[j] + ratio2*tmp[j + 1]) * (-1.0)  # -Z
            # B[i][0] = ( tmp[j] + tmp[j+1] ) * (-1.0) / 2#-Z
    Y = np.transpose(X0[1:])
    BT = np.transpose(B)
    a = np.linalg.inv(np.dot(BT,B))#矩阵求逆  ;;np.linalg.det()：矩阵求行列式（标量）
    a = np.dot(a,BT)
    a = np.dot(a,Y)
    a = np.transpose(a)
    return a;

def GM_gbnn(X0,X1,a,tmp):          #GM(1,1)模型的预测值
    A = np.ones(len(X1))#元素全为1的数组，无论多少维都可
    cases ,labels= [],[]
    for i in range(len(X1)):#[[], [], [], []]
        cases.append([])
        labels.append([])
    # for j in range(len(X1)):pattern[j].append([]);pattern[j].append([])#[[[], []], [[], []], [[], []], [[], []]]
    for j, k in enumerate(X1.index):
        print('GM(1,1)模型为:\nX(k) = ', X1[k] - a[1] / a[0], 'exp(', -a[0], tmp[i]-1-j,')+', a[1] / a[0])
        for i in range(len(X1)):#tmp[i] - 1：k-m:
            p1=round(a[1] / a[0] + (X1[k] - a[1] / a[0]) * np.exp(a[0] * (tmp[i]-1-j) * (-1)),2)
            ##这里tmp[i]-1-j是因为tmp是从1开始，若从0开始则无需-1
            if i==0:
                cases[i].append(p1)
            else:
                cases[i].append(round(p1-p0,2))
            p0=p1
        labels[j].append(X0[k])
    print(cases)
    print(labels)
    nn = BPNeuralNetwork()
    # 创建一个神经网络：输入层有两个节点、隐藏层有两个节点、输出层有一个节点
    nh=math.floor((math.sqrt(len(X1)*2)+3))#隐藏层的设置公式：sqrt(m+n)+a，这里吧a设置为3
    ao_list=nn.test(cases,labels,len(X1),nh,1)
    XK_ao=Series(ao_list,index=X1.index)
    print('GM(1,1)_BPNN模型计算值为:')
    print(XK_ao)
    return XK_ao;

    # 看看训练好的权重（当然可以考虑把训练好的权重持久化）
    # n.weights()

    # print ('GM(1,1)模型为:\nX(k) = ',X0[j]-a[1]/a[0],'exp(',-a[0],'(k-1))',a[1]/a[0])
    # XK = Series(A,index=X0.index)#'A-DEC'是年第12月底最后一个日历日
    # print ('GM(1,1)模型计算值为:')
    # print (XK)
    # return XK;

# def GM_Model(X0,a,tmp):          #GM(1,1)模型的预测值
#     A = np.ones(len(X0))#元素全为1的数组，无论多少维都可
#     j=X0.index[len(X0)-1]
#     # j =2017
#     for i in range(len(A)):#本该X1[j]，但是因为这里取第一个为初始值，则此时X1[j]=X0[j]
#         A[i] = a[1]/a[0] + (X0[j]-a[1]/a[0])*np.exp(a[0]*(tmp[i]-1)*(-1))#预测值:tmp = np.array([1,2,3,4,5,6])
#     print ('GM(1,1)模型为:\nX(k) = ',X0[j]-a[1]/a[0],'exp(',-a[0],'(k-1))',a[1]/a[0])
#     XK = Series(A,index=X0.index)#'A-DEC'是年第12月底最后一个日历日
#     # XK=Series(A,Series(tmp,index=pd.period_range('2000','2005',freq = 'A-DEC')))
#     #index: PeriodIndex(['2000', '2001', '2002', '2003', '2004', '2005'], dtype='period[A-DEC]', freq='A-DEC')
#     print ('GM(1,1)模型计算值为:')
#     print (XK)
#     return XK;

def GM_Model(X0,X1,a,tmp):          #GM(1,1)模型的预测值
    A = np.ones(len(X1)+1)#元素全为1的数组，无论多少维都可
    k=X1.index[0]
    for i in range(len(A)):#本该X1[j]，但是因为这里取第一个为初始值，则此时X1[j]=X0[j]
        A[i] = round(a[1]/a[0] + (X1[k]-a[1]/a[0])*np.exp(a[0]*(tmp[i]-1)*(-1)),2)#预测值:tmp = np.array([1,2,3,4,5,6])
        #这里tmp[i]-1是因为tmp是从1开始，若从0开始则无需-1
    print ('GM(1,1)模型为:\nX(k) = ',X1[k]-a[1]/a[0],'exp(',-a[0],'(k-1))',a[1]/a[0])
    index=list(X0.index)
    index_p=X0.index[-1]+1
    index.append(index_p)
    XK = Series(A,index=index)#'A-DEC'是年第12月底最后一个日历日
    # XK=Series(A,Series(tmp,index=pd.period_range('2000','2005',freq = 'A-DEC')))
    #index: PeriodIndex(['2000', '2001', '2002', '2003', '2004', '2005'], dtype='period[A-DEC]', freq='A-DEC')
    print ('GM(1,1)模型计算值为:')
    print (XK)
    return XK;

def GM_ratio_bpnn(X0, X1, a_all, tmp):
    cases= []
    labels=[[i] for i in X0.values]
    k = X1.index[0]
    for i in range(len(X1)):  # [[], [], [], []]
        cases.append([])
    print(a_all)
    for a in a_all:
        # for j in range(len(X1)):pattern[j].append([]);pattern[j].append([])#[[[], []], [[], []], [[], []], [[], []]]
        print('GM(1,1)模型为:\nX(k) = ', X1[k] - a[1] / a[0], 'exp(', -a[0], tmp[i] - 1, ')+', a[1] / a[0])
        for i in range(len(X1)):#len(X1)+1
            p1 = round(a[1]/a[0] + (X1[k]-a[1]/a[0])*np.exp(a[0]*(tmp[i]-1)*(-1)),2)
            ##这里tmp[i]-1-j是因为tmp是从1开始，若从0开始则无需-1
            if i == 0:
                cases[i].append(p1)
            else:
                cases[i].append(round(p1 - p0, 2))
            p0 = p1

    print(cases)
    print(labels)
    nn = BPNeuralNetwork()
    # 创建一个神经网络：输入层有两个节点、隐藏层有两个节点、输出层有一个节点
    # nh = math.floor((math.sqrt(len(X1) * 2) + 3))  # 隐藏层的设置公式：sqrt(m+n)+a，这里吧a设置为3
    ao_list = nn.test(cases[1:], labels[1:], 4, 5, 1)#4种模型
    XK_ao = Series(ao_list, index=X1.index[1:])
    print('GM(1,1)_BPNN模型计算值为:')
    print(XK_ao)
    return XK_ao;



def Return(XK):                 #预测值还原
    tmp = np.ones(len(XK))
    for i,j in enumerate(XK.index):
        if j == min(XK.index):
            tmp[i] = XK[j]
        else:
            tmp[i] = XK[j] - XK[j-1]
    X_Return = Series(tmp,index=XK.index)
    print ('还原值为:\n')
    print (X_Return)
    return X_Return

if __name__ == '__main__':
    #初始化原始数据
    date = pd.period_range('2000','2005',freq = 'A-DEC')
    tmp = np.array([1,2,3,4,5,6])
    data = np.array([132,92,118,130,187,207])
    X0 = Series(data,index = date)
    X0_copy = Series(data,index=tmp)
    print ('原始数据为:\n')
    print(X0)

    #对原始数据惊醒一次累加
    X1 = np.cumsum(X0)
    print ('原始数据累加为:')
    print(X1)

    #辨识算法
    a = Identification_Algorithm(data)
    print ('a矩阵为:')
    print (a)

    #GM(1,1)模型
    XK = GM_Model(X0,a,tmp)

    #预测值还原
    X_Return = Return(XK)

    #预测值即预测值精度表
    X_Compare1 = np.ones(len(X0))
    X_Compare2 = np.ones(len(X0))
    for i in range(len(data)):
        X_Compare1[i] = data[i]-X_Return[i]
        X_Compare2[i] = X_Compare1[i]/data[i]*100
    Compare = {'GM':XK,'1—AGO':np.cumsum(data),'Returnvalue':X_Return,'Realityvalue':data,'Error':X_Compare1,'RelativeError(%)':X_Compare2}
    X_Compare = DataFrame(Compare,index=date)
    print ('预测值即预测值精度表')
    print (X_Compare)

    #模型检验
    error_square = np.dot(X_Compare,np.transpose(X_Compare))    #残差平方和
    error_avg = np.mean(error_square)                           #平均相对误差

    S = 0                                                       #X0的关联度
    for i in range(1,len(X0)-1,1):
        S += X0[i]-X0[0]+(XK[-1]-XK[0])/2
    S = np.abs(S)

    SK = 0                                                      #XK的关联度
    for i in range(1,len(XK)-1,1):
        SK += XK[i]-XK[0]+(XK[-1]-XK[0])/2
    SK = np.abs(SK)

    S_Sub = 0                                                   #|S-SK|b
    for i in range(1,len(XK)-1,1):
        S_Sub += X0[i]-X0[0]-(XK[i]-XK[0])+((X0[-1]-X0[0])-(XK[i]-XK[0]))/2
    S_Sub = np.abs(S_Sub)

    T = (1+S+SK)/(1+S+SK+S_Sub)

    if T >= 0.9:
        print ('精度为一级')
        print ('可以用GM(1,1)模型\nX(k) = ',X0[0]-a[1]/a[0],'exp(',-a[0],'(k-1))',a[1]/a[0])
    elif T >= 0.8:
        print ('精度为二级')
        print ('可以用GM(1,1)模型\nX(k) = ',X0[0]-a[1]/a[0],'exp(',-a[0],'(k-1))',a[1]/a[0])
    elif T >= 0.7:
        print ('精度为三级')
        print ('谨慎用GM(1,1)模型\nX(k) = ',X0[0]-a[1]/a[0],'exp(',-a[0],'(k-1))',a[1]/a[0])
    elif T >= 0.6:
        print ('精度为四级')
        print ('尽可能不用GM(1,1)模型\nX(k) = ',X0[0]-a[1]/a[0],'exp(',-a[0],'(k-1))',a[1]/a[0])



    X2006 = Series(np.array([259.4489]),index=pd.period_range('2006','2006',freq = 'A-DEC'))
    X_Return = X_Return.append(X2006)
    print (X_Return)

    B = pd.DataFrame([X0,X_Return],index=['X0','X_Return'])
    B = np.transpose(B)
    B.plot()