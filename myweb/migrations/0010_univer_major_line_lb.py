# -*- coding: utf-8 -*-
# Generated by Django 1.11.7 on 2017-12-19 07:02
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myweb', '0009_major16_lb'),
    ]

    operations = [
        migrations.CreateModel(
            name='univer_major_line_lb',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('major', models.CharField(max_length=30, null=True)),
                ('school', models.CharField(max_length=30, null=True)),
                ('average', models.IntegerField(null=True)),
                ('hightst', models.IntegerField(null=True)),
                ('province', models.CharField(max_length=10)),
                ('classy', models.CharField(max_length=10)),
                ('year', models.IntegerField(null=True)),
                ('batch', models.CharField(max_length=50)),
                ('major_new', models.CharField(max_length=30, null=True)),
                ('line', models.IntegerField(null=True)),
                ('classes', models.CharField(max_length=30, null=True)),
                ('catalogue', models.CharField(max_length=30, null=True)),
                ('a_l', models.IntegerField(null=True)),
                ('h_l', models.IntegerField(null=True)),
                ('h_a', models.IntegerField(null=True)),
            ],
        ),
    ]
