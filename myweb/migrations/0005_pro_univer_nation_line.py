# -*- coding: utf-8 -*-
# Generated by Django 1.11.7 on 2017-12-06 11:47
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myweb', '0004_auto_20171203_1249'),
    ]

    operations = [
        migrations.CreateModel(
            name='pro_univer_nation_line',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('school', models.CharField(max_length=30)),
                ('province', models.CharField(max_length=10)),
                ('classy', models.CharField(max_length=10)),
                ('year', models.IntegerField(null=True)),
                ('lowest', models.IntegerField(null=True)),
                ('hightst', models.IntegerField(null=True)),
                ('average', models.IntegerField(null=True)),
                ('amount', models.IntegerField(null=True)),
                ('batch', models.CharField(max_length=50)),
                ('line', models.IntegerField(null=True)),
            ],
        ),
    ]
