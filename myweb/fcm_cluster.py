# http://blog.csdn.net/chenge_j/article/details/72537568
import sys
import numpy as np
# 给定同维向量数据集合points，数目为n，将其聚为C类，m为权重值，u为初始匹配度矩阵（n*C），采用闵式距离算法，其参数为p，迭代终止条件为终止值e（取值范围(0，1））及终止轮次。
# 计算停止时返回计算的轮次和匹配度矩阵，返回值为一个tuple:(最终计算过后的u矩阵,计算轮次（从0开始）)
#计算皮尔逊相关度：
def pearson(p,q):#只计算两者共同有的
    n =len(p)
    sumx = sum([p[i] for i in range(n)])
    sumy = sum([q[i] for i in range(n)])
    sumxsq = sum([p[i]**2 for i in range(n)])
    sumysq = sum([q[i]**2 for i in range(n)])
    sumxy = sum([p[i]*q[i] for i in range(n)])
    #求出pearson相关系数
    up = sumxy - sumx*sumy/n
    down = ((sumxsq - pow(sumxsq,2)/n)*(sumysq - pow(sumysq,2)/n))**.5
    #若down为零则不能计算，return 0
    if down == 0 :return 0
    r = up/down
    return r

#Minkowski 距离
def dis_minkowski(rate1,rate2,r):#其中p是一个变参数。当p=1时，就是曼哈顿距离;当p=2时，就是欧氏距离;当p→∞时，就是切比雪夫距离;根据变参数的不同，闵氏距离可以表示一类的距离。
    distance = 0
    commonRating = False
    for j in range(len(rate1)):
        distance += pow(abs(rate1[j] - rate2[j]), r)
        commonRating = True
    if commonRating:
        return pow(distance,1/r)
    else:
        return -1

def alg_fcm(points, u, m, p, e, terminateturn):
    assert (len(points) == len(u));
    assert (len(points) > 0);
    assert (len(u[0]) > 0);
    assert (m > 0);
    assert (p > 0);
    assert (e > 0);

    u1 = u;
    k = 0;
    while (True):
        # calculate one more turn
        u2 ,centroids= fcm_oneturn(points, u1, m, p);
        # max difference between u1 and u2
        maxu = fcm_maxu(u1, u2);

        if (maxu < e):
            break;
        u1 = u2;
        k = k + 1;
        if k > terminateturn:
            break;

    return (u2, k, centroids);


# 每一轮计算的函数fcm_oneturn代码如下，参数与主函数相同，返回值只有匹配度矩阵
def fcm_oneturn(points, u, m, p):
    assert (len(points) == len(u));
    assert (len(points) > 0);
    assert (len(u[0]) > 0);
    assert (m > 0);
    assert (p > 0);

    n = len(points);
    c = len(u[0]);

    # calculate centroids of clusters
    centroids = fcm_c(points, u, m);
    assert (len(centroids) == c);

    # calculate new u matrix
    u2 = fcm_u(points, centroids, m, p);
    assert (len(u2) == n);
    assert (len(u2[0]) == c);

    return u2,centroids;


# 计算终止值的函数：

def fcm_maxu(u1, u2):
    assert (len(u1) == len(u2));
    assert (len(u1) > 0);
    assert (len(u1[0]) > 0);

    ret = 0;
    n = len(u1);
    c = len(u1[0]);
    for i in range(n):
        for j in range(c):
            ret = max(np.fabs(u1[i][j] - u2[i][j]), ret);

    return ret;


# 每一个轮次计算匹配度矩阵的函数：参数centroids表示每个类别的质心集合
def fcm_u(points, centroids, m, p):
    assert (len(points) > 0);
    assert (len(centroids) > 0);
    assert (m > 1);
    assert (p > 0);

    n = len(points);
    c = len(centroids);
    ret = [[0 for j in range(c)] for i in range(n)];
    for i in range(n):
        for j in range(c):
            sum1 = 0;
            d1 = dis_minkowski(points[i], centroids[j], p);
            for k in range(c):
                d2 = dis_minkowski(points[i], centroids[k], p);
                if d2 != 0:
                    sum1 += np.power(d1 / d2, float(2) / (float(m) - 1));
            if sum1 != 0:
                ret[i][j] = 1 / sum1;

    return ret;


# 每一个轮次计算类别质心的函数
def fcm_c(points, u, m):
    assert (len(points) == len(u));
    assert (len(points) > 0);
    assert (len(u[0]) > 0);
    assert (m > 0);

    n = len(points);
    c = len(u[0]);
    ret = [];
    for j in range(c):
        sum1 = 0;
        sum2 = 0;
        for i in range(n):
            sum2 += np.power(u[i][j], m);
            sum1 += np.dot(points[i], np.power(u[i][j], m));
        if sum2 != 0:
            cj = sum1 / sum2;
        else:
            cj = [0 for d in range(len(points[i]))];
        ret.append(cj);

    return ret;