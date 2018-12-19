"""web URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url,include
from django.contrib import admin
from web.view import *

urlpatterns = [
    url(r'^admin/', admin.site.urls),##登陆使用的！
    # url(r'^home/$', view.home),##这里的/很重要，不能丢；只有一个网页的时候可丢？
    url('^index/',include('myweb.urls')),#这里不能带$

    # url(r'^home/part1_1/$', view.part1_1),
    # url(r'^home/part1_2/$', view.part1_2),
    # url(r'^home/part1_3/$', view.part1_3),
    # url(r'^home/part1_4/$', view.part1_4),
    # url(r'^home/part1_5/$', view.part1_5),
    # url(r'^home/part1_6/$', view.part1_6),
    # url(r'^home/part2_1/$', view.part2_1),
    # url(r'^home/part2_2/$', view.part2_2),
    # url(r'^home/part2_2_1/$', view.part2_2_1),
    # url(r'^home/part2_3/$', view.part2_3),
    # url(r'^home/part2_3_1/$', view.part2_3_1),
    # url(r'^home/part2_4/$', view.part2_4),
    # url(r'^home/part2_4_1/$', view.part2_4_1),
    # url(r'^home/part2_5/$', view.part2_5),
    # url(r'^home/part2_5_1/$', view.part2_5_1),
]
