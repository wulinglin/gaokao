from __future__ import unicode_literals
from django.db import models
from django.utils.encoding import python_2_unicode_compatible
import numpy as np
# Create your models here.
class mbti_chractor(models.Model):
    # id=models.AutoField(verbose_name='MBTI编号')
    mbti=models.CharField(max_length=20)#做外键时必须表明唯一性（主键）primary_key=True
    #设置外键会多出来_id这个东东，而且默认int，弄的数据全部更变！
    charactor=models.CharField(max_length=60)
    explanation=models.TextField()

class mbti_major(models.Model):
    major_new =models.CharField(max_length=30)#做外键时必须表明唯一性（主键）primary_key=True
    classes=models.CharField(max_length=30)
    mbti=models.CharField(max_length=30)
    # mbti=models.ManyToManyField(mbti_chractor)##?书对应作者的问题

class nation_line(models.Model):
    year=models.IntegerField(null=True)
    province=models.CharField(max_length=10)
    classy=models.CharField(max_length=10)
    batch=models.CharField(max_length=50)
    line = models.IntegerField(null=True)

class nation_line_lowest(models.Model):
    year=models.IntegerField(null=True)
    province=models.CharField(max_length=10)
    classy=models.CharField(max_length=10)
    batch=models.CharField(max_length=50,null=True)
    line = models.IntegerField(null=True)

class univer_info(models.Model):
    school=models.CharField(max_length=30)#做外键时必须表明唯一性（主键）primary_key=True
    leixing=models.CharField(max_length=30)
    lishuyu=models.CharField(max_length=30)
    province = models.CharField(max_length=10)
    academician=models.IntegerField(default=0)
    doctor=models.IntegerField(default=0)
    master=models.IntegerField(default=0)
    address=models.CharField(max_length=50)
    telephone=models.CharField(max_length=40)
    website=models.CharField(max_length=30)
    email=models.CharField(max_length=30)
    post=models.CharField(max_length=20)
    city=models.CharField(max_length=60,default='')
    provincial_capital=models.CharField(max_length=60,default='')
    development=models.CharField(max_length=60,default='')
    leixing_int=models.IntegerField(null=True)
    is_doctor=models.IntegerField(null=True)
    is_master=models.IntegerField(null=True)
    school_rank = models.IntegerField(null=True)
    school_type=models.CharField(max_length=30,default='')

class univer_info_search(models.Model):
    school=models.CharField(max_length=30)#做外键时必须表明唯一性（主键）primary_key=True
    leixing=models.CharField(max_length=30)
    lishuyu=models.CharField(max_length=30)
    province = models.CharField(max_length=10)
    academician=models.IntegerField(default=0)
    doctor=models.IntegerField(default=0)
    master=models.IntegerField(default=0)
    address=models.CharField(max_length=50)
    telephone=models.CharField(max_length=40)
    website=models.CharField(max_length=30)
    email=models.CharField(max_length=30)
    post=models.CharField(max_length=20)
    city=models.CharField(max_length=60,default='')
    development=models.CharField(max_length=60,default='')
    school_rank = models.IntegerField(null=True)
    school_type=models.CharField(max_length=30,default='')


class univer_major_line(models.Model):
    major=models.CharField(max_length=30)
    school=models.CharField(max_length=30)
    # school=models.ForeignKey(univer_info)
    average = models.CharField(max_length=10,null=True)
    hightst = models.CharField(max_length=10,null=True)
    province=models.CharField(max_length=10)
    # province=models.ForeignKey(nation_line)##省份不具唯一性，不能设外键
    classy=models.CharField(max_length=10)
    year = models.IntegerField(null=True)
    batch=models.CharField(max_length=50)
    major_new=models.CharField(max_length=30,default='')
    # major_new = models.ForeignKey(mbti_major)

class pro_univer_point(models.Model):
    # school,province,classy,batch
    school=models.CharField(max_length=30)
    province=models.CharField(max_length=10)
    classy=models.CharField(max_length=10)
    year=models.IntegerField(null=True)
    lowest=models.IntegerField(null=True)
    hightst=models.IntegerField(null=True)
    average=models.IntegerField(null=True)
    amount=models.IntegerField(null=True)
    batch=models.CharField(max_length=50)

class pro_univer_nation_line(models.Model):
    # school,province,classy,batch
    school=models.CharField(max_length=30)
    province=models.CharField(max_length=10)
    classy=models.CharField(max_length=10)
    year=models.IntegerField(null=True)
    lowest=models.IntegerField(null=True)
    hightst=models.IntegerField(null=True)
    average=models.IntegerField(null=True)
    amount=models.IntegerField(null=True)
    batch=models.CharField(max_length=50)
    line = models.IntegerField(null=True)
    d_hl=models.IntegerField(null=True)
    d_h=models.IntegerField(null=True)
    d_l=models.IntegerField(null=True)
    d_a=models.IntegerField(null=True)
    d_ha=models.IntegerField(null=True)
    d_la=models.IntegerField(null=True)

class univer_major_pro_line(models.Model):
    major = models.CharField(max_length=30)
    school = models.CharField(max_length=30)
    average = models.CharField(max_length=10,null=True)
    hightst = models.CharField(max_length=10,null=True)
    province = models.CharField(max_length=10)
    classy = models.CharField(max_length=10)
    year = models.IntegerField(null=True)
    batch = models.CharField(max_length=50)
    major_new = models.CharField(max_length=30, default='')
    lowest_pro=models.IntegerField(null=True)
    hightst_pro=models.IntegerField(null=True)
    average_pro=models.IntegerField(null=True)
    amount_pro=models.IntegerField(null=True)
    batch_pro=models.CharField(max_length=50)

class user_info(models.Model):
    city=models.CharField(max_length=30)
    point=models.IntegerField(null=True)
    yingjie=models.CharField(max_length=30)
    hukou=models.CharField(max_length=30)
    sex=models.CharField(max_length=30)
    kelei=models.CharField(max_length=30)
    nation=models.CharField(max_length=30)
    add_point=models.IntegerField(null=True)
    mbti=models.CharField(max_length=30)
    #recommend_major太大了，数据库资源耗损。所以这里只记录mbti，
    # recommend_major=models.CharField(max_length=3000)
    major_recom_interst=models.CharField(max_length=300)
    recommend=models.CharField(max_length=300)

class major16_lb(models.Model):
    major = models.CharField(max_length=30,null=True)
    classes = models.CharField(max_length=30,null=True)
    mbti=models.CharField(max_length=40,null=True)
    catalogue=models.CharField(max_length=30,null=True)

class univer_major_line_lb(models.Model):
    major= models.CharField(max_length=30,null=True)
    school= models.CharField(max_length=30,null=True)
    average=models.IntegerField(null=True)
    hightst=models.IntegerField(null=True)
    province= models.CharField(max_length=10)
    classy= models.CharField(max_length=10)
    year= models.IntegerField(null=True)
    batch=models.CharField(max_length=50)
    major_new= models.CharField(max_length=30,null=True)
    line= models.IntegerField(null=True)
    classes = models.CharField(max_length=30,null=True)
    catalogue = models.CharField(max_length=30,null=True)
    a_l= models.IntegerField(null=True)
    h_l= models.IntegerField(null=True)
    h_a= models.IntegerField(null=True)
    recruit_count=models.IntegerField(null=True)
    major_rank=models.IntegerField(null=True)
    school_rank=models.IntegerField(null=True)

#学校类型、排名，专业排名加上了
class province(models.Model):
    province = models.CharField(max_length=10)

class school_type(models.Model):
    school_type = models.CharField(max_length=20)

class year(models.Model):
    year = models.IntegerField(null=True)