# -*- coding: utf-8 -*-
# Generated by Django 1.11.7 on 2017-12-03 04:49
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myweb', '0003_auto_20171203_1204'),
    ]

    operations = [
        migrations.AlterField(
            model_name='univer_major_line',
            name='average',
            field=models.CharField(max_length=10, null=True),
        ),
        migrations.AlterField(
            model_name='univer_major_line',
            name='hightst',
            field=models.CharField(max_length=10, null=True),
        ),
        migrations.AlterField(
            model_name='univer_major_pro_line',
            name='average',
            field=models.CharField(max_length=10, null=True),
        ),
        migrations.AlterField(
            model_name='univer_major_pro_line',
            name='hightst',
            field=models.CharField(max_length=10, null=True),
        ),
    ]
