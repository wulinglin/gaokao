# -*- coding: utf-8 -*-
# Generated by Django 1.11.7 on 2017-12-10 10:57
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myweb', '0008_auto_20171208_2146'),
    ]

    operations = [
        migrations.CreateModel(
            name='major16_lb',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('major', models.CharField(max_length=30, null=True)),
                ('classes', models.CharField(max_length=30, null=True)),
                ('mbti', models.CharField(max_length=40, null=True)),
                ('catalogue', models.CharField(max_length=30, null=True)),
            ],
        ),
    ]
