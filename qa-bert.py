# -*- coding: utf-8 -*-
"""
__title__="qa-dcbert"
__author__="ngc7293"
__mtime__="2021/1/19"
"""
a = """When did Beyonce start becoming popular ? [SEP] Beyoncé Giselle Knowles - Carter ( /biːˈjɒnseɪ/ bee - YON - say ) ( born September 4 , 1981 ) is an American singer , songwriter , record producer and actress . Born and raised in Houston , Texas , she performed in various singing and dancing competitions as a child , and rose to fame in the late 1990s as lead singer of R&B girl - group Destiny 's Child . Managed by her father , Mathew Knowles , the group became one of the world 's best - selling girl groups of all time . Their hiatus saw the release of Beyoncé 's debut album , Dangerously in Love ( 2003 ) , which established her as a solo artist worldwide , earned five Grammy Awards and featured the Billboard Hot 100 number - one singles " Crazy in Love " and " Baby Boy " . <pad> <pad> <pad> <pad> <pad>"""
for i in range(len(a.split())):
    if a.split()[i] == 'in':
        print(i)