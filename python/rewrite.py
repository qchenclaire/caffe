#!/usr/bin/python

import sys


def replace_line(file_name, line_num, text):
    lines = open(file_name, 'r').readlines()
    lines[line_num] = text
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()

text = 'iter = ' + str(sys.argv[1])+'\n'
replace_line('python/cifar_mine.py',7,text)
