#!/usr/bin/env python
# encoding=utf-8
"""
解析总则：
枚举类型 -> 全部转为 有限编码，通过配置文件获取全部类型, (loan-specTraDataList 内部 specTraType 也是枚举类型，暂没有为期制定编码）
数值类型 -> 1、计算获得型，均以数值形式存在，如【'infoUpdateDays': -1040】, 原始数值型 保留为 原始数字字符串， 如'totalNum': '24',
日期 -> 保留 纯数字字符串 年份："2020"，年月："202001"，日期 ："20200101" / 并全部增加相对天数字段 originalDate -> originalDays(int)
自然语言文本 -> 保留

--- 其他：
1、缺省值，可能以三种形式存在："None"，""， 或 空容器 []、{}。 个别字段也可能存在 '--'形式的缺省值，此为原始报告填充形式，未能全部标准化。
2、所有 Date类型 处理后返回两个字段 8/6位纯数字字符串、相对天数（相对于reportDate 的天数  -> thisDate-reportDate）
"""

import datetime
import math
import re
# from logger import root_logger as logger
import logging as logger


class ZxMidDataCenter(object):
    pass
