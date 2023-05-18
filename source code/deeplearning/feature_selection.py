# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 15:35:41 2018

@author: liuze
"""
from __future__ import division
import sys
from functools import reduce
import re
import operator
from math import log
import numpy as np
from math import sqrt
import math
def kmer(seq):
    mer2={}
    mer3={}
    mer4={}
    for n1 in 'ATCG':
        for n2 in 'ATCG':
            mer2[n1+n2]=0
            for n3 in 'ATCG':
                mer3[n1+n2+n3]=0
                for n4 in 'ATCG':
                    mer4[n1+n2+n3+n4]=0
    seq=seq.replace('N','')
    seq_len=len(seq)
    for p in range(0,seq_len-3):
        mer2[seq[p:p+2]]+=1
        mer3[seq[p:p+3]]+=1
        mer4[seq[p:p+4]]+=1
    mer2[seq[p+1:p+3]]+=1
    mer2[seq[p+2:p+4]]+=1
    mer3[seq[p+1:p+4]]+=1
    v2=[]
    v3=[]
    v4=[]
    for n1 in 'ACGT':
        for n2 in 'ACGT':
            v2.append(mer2[n1+n2])
            for n3 in 'ACGT':
                v3.append(mer3[n1+n2+n3])
                for n4 in 'ACGT':
                    v4.append(mer4[n1+n2+n3+n4])
    v=v2+v3+v4
    return v

def ksnpf(seq):
    kn=5
    freq=[]
    v=[]
    for i in range(0,kn):
        freq.append({})
        for n1 in 'ATCGN':
            freq[i][n1]={}
            for n2 in 'ATCGN':
                freq[i][n1][n2]=0
    seq=seq.strip('N')
    seq_len=len(seq)
    for k in range(0,kn):
        for i in range(seq_len-k-1):
            n1=seq[i]
            n2=seq[i+k+1]
            freq[k][n1][n2]+=1
    for i in range(0,kn):
        for n1 in 'ATCG':
            for n2 in 'ATCG':
                v.append(freq[i][n1][n2])
    return v
def binary_code(seq):
    binary_dictionary={'A':[1,1,1],'T':[0,1,0],'G':[1,0,0],'C':[0,0,1],'N':[0,0,0]}
    cnt=[]
    for i in seq:
        cnt.append(binary_dictionary[i])
    return reduce(operator.add,cnt)

def binary_code_2D(seq):
    binary_dictionary={'A':[1,1,1],'T':[0,1,0],'G':[1,0,0],'C':[0,0,1],'N':[0,0,0]}
    cnt=[]
    for i in seq:
        cnt.append(binary_dictionary[i])
    return cnt

def binary_code_3D(seq):
    binary_dictionary={'A':[[1],[1],[1]],'T':[[0],[1],[0]],'G':[[1],[0],[0]],'C':[[0],[0],[1]],'N':[[0],[0],[0]]}
    cnt=[]
    for i in seq:
        cnt.append(binary_dictionary[i])
    return cnt

def nucleic_shift(seq):
    nucleic_dictionary={'A':0,'T':0,'C':0,'G':0,'N':0}
    p=0
    v=[]
    for n in seq:
        p=p+1
        nucleic_dictionary[n]+=1
        v.append(nucleic_dictionary[n]/p)
    return v
def matrix_motif1_func(sequence,motif_matrix):
    #seq=seq[0:50]+seq[52:101]
    binary_dictionary={'A':0, 'C':1, 'G':2, 'T':3}
#    cnt=[]
#    temp_score=[]
#    score=[]
#    for i in seq:
#        cnt.append(binary_dictionary[i])
#    for j in range(len(cnt)-motif_matrix.shape[1]):
#        #k=j
#        for k in range(j,j+motif_matrix.shape[1]):
#            temp_score.append(motif_matrix[cnt[k]][k])
#            score.append(sum(temp_score))
#    return score
    score_vector=[]
    for q in range(0,len(sequence)-9):
        cnt=[]
        score=[]
        seq=sequence[q:q+10]
        for i in seq:
            cnt.append(binary_dictionary[i])
        for j in range(len(cnt)):
            score.append(motif_matrix[cnt[j]][j])
        score_vector.append(sum(score))
    return score_vector
def matrix_motif2_func(sequence,motif_matrix):
    #seq=seq[0:50]+seq[52:101]
    binary_dictionary={'A':0, 'C':1, 'G':2, 'T':3}
    score_vector=[]
    for q in range(0,len(sequence)-8):
        cnt=[]
        score=[]
        seq=sequence[q:q+9]
        for i in seq:
            cnt.append(binary_dictionary[i])
        for j in range(len(cnt)):
            score.append(motif_matrix[cnt[j]][j])
        score_vector.append(sum(score))
    return score_vector
def EIIP(seq):  
    #1.给AAA、AAC等赋值
    E={'A':[0.1260],'T':[0.1340],'G':[0.0806],'C':[0.1335]}
    a={}
    for i in 'ACGT':
        for j in 'ACGT':
            for k in 'ACGT':
                a[i+j+k]=(np.sum(E[i]+E[j]+E[k]))#三个单核苷酸的EIIP值相加
    bl=[]#字典a里的数据转到列表里
    for i in 'ACGT':
        for j in 'ACGT':
            for k in 'ACGT':
                bl.append(a[i+j+k])   
    #2.计算序列中AAA等的频率                  
    c={}#给AAA等赋0值
    for i in 'ACGT':
        for j in 'ACGT':
            for k in 'ACGT':
                c[i+j+k]=0
    seq_len=len(seq)
    for p in range(0,seq_len-2):#计算序列中三核苷酸的个数
        c[seq[p:p+3]]+=1
    d=[]##字典c里的数据转到列表里
    for i in 'ACGT':
        for j in 'ACGT':
            for k in 'ACGT':
                d.append((c[i+j+k])/float(seq_len-2))#序列中三核苷酸AAA等的频率
    e=np.multiply(np.array(bl),np.array(d))
    e=e.tolist()
    #e = map(lambda a,b :a*b,zip(b,d))
    return e

def TF_content_location(seq):
    mer1={}
    mer2={}
    for n1 in 'ATCG':
        mer1[n1]=0#单核苷酸赋0值
        for n2 in 'ATCG':
            mer2[n1+n2]=0#双核苷酸赋0值
    seq_len=len(seq)#序列的长度
    for p in range(0,seq_len-1):#序列中单、双核苷酸分别的个数
        mer1[seq[p:p+1]]+=1
        mer2[seq[p:p+2]]+=1
    mer1[seq[p+1:p+2]]+=1
    
    s=seq[seq_len-1]
    p={}#16维 转移概率
    for i in 'ACGT':
        for j in 'ACGT':
            if (i!=s and mer1[j]!=0):
                p[i+j]=abs(mer2[i+j]/mer1[j])
            elif ((mer1[j]-1)!=0):
                p[i+j]=abs(mer2[i+j]/(mer1[j]-1))
            else:
                p[i+j]=0
    #4维 含量            
    c={}
    for i in 'ACGT':#分别计算A、C、G、T的频率
        c[i]=(mer1[i]/seq_len)
    #4维 位置和   
    sum0={}
    for i in 'ACGT':
        sum0[i]=0
    for i in range(0,seq_len):
        sum0[seq[i:i+1]]+=i+1#将A、C、G、T所有的位置数值求和
    d={}
    for i in 'ACGT':
        d[i]=0
    for i in 'ACGT':
        d[i]=(2*sum0[i])/(seq_len*(seq_len+1))
    
    T1=[]
    T2=[]
    T3=[]
    for i in 'ACGT':
        T2.append(c[i])
        T3.append(d[i])
        for j in 'ACGT':
            T1.append(p[i+j])
    T=T1+T2+T3
    return T

def kskip(seq):
    skip0={}#k=0
    skip1={}#k=1
    skip2={}#k=2
    skip3={}#k=3
    for n1 in 'ACGT':
        for n2 in 'ACGT':
            skip0[n1+n2]=0
            skip1[n1+n2]=0
            skip2[n1+n2]=0
            skip3[n1+n2]=0

    seq_len=len(seq)
    for p in range(0,seq_len-4):
        skip0[seq[p]+seq[p+1]]+=1
        skip1[seq[p]+seq[p+2]]+=1
        skip2[seq[p]+seq[p+3]]+=1
        skip3[seq[p]+seq[p+4]]+=1
    skip0[seq[p+1]+seq[p+2]]+=1
    skip0[seq[p+2]+seq[p+3]]+=1
    skip0[seq[p+3]+seq[p+4]]+=1
    skip1[seq[p+1]+seq[p+3]]+=1
    skip1[seq[p+2]+seq[p+4]]+=1
    skip2[seq[p+1]+seq[p+4]]+=1
    v0=[]
    v1=[]
    v2=[]
    v3=[]
    for i in 'ACGT':
        for j in 'ACGT':
            v0.append(skip0[i+j])
            v1.append(skip1[i+j])
            v2.append(skip2[i+j])
            v3.append(skip3[i+j])
    v=v0+v1+v2+v3
    return v
    
def globalDescription(seq):
    a={}
    c={}
    d={}
    for i in 'ATCG':
        a[i]=0
        c[i]=0
        d[i]=0
    seq_len=len(seq)
    #1.碱基频率
    for i in range(0,seq_len):#A、C、G、T的频率
        a[seq[i]]+=1
    comp=[]
    for i in 'ACGT':
        comp.append(100*a[i]/float(seq_len))
    #2.百分比
    b={'AC':0, 'AG':0, 'AT':0, 'CG':0, 'CT':0, 'GT':0,}#双碱基前后不一致的情况
    for i in range(0,seq_len-1):
        if seq[i:i+2]=='AC' or seq[i:i+2]=='CA':
            b['AC']+=1
        elif seq[i:i+2]=='AG' or seq[i:i+2]=='GA':
            b['AG']+=1
        elif seq[i:i+2]=='AT' or seq[i:i+2]=='TA':
            b['AT']+=1
        elif seq[i:i+2]=='CG' or seq[i:i+2]=='GC':
            b['CG']+=1
        elif seq[i:i+2]=='CT' or seq[i:i+2]=='TC':
            b['CT']+=1
        elif seq[i:i+2]=='GT' or seq[i:i+2]=='TG':          
            b['GT']+=1
    tran=[]
    tran.append(100*b['AC']/float(seq_len-1))#频率
    tran.append(100*b['AG']/float(seq_len-1))
    tran.append(100*b['AT']/float(seq_len-1))
    tran.append(100*b['CG']/float(seq_len-1))
    tran.append(100*b['CT']/float(seq_len-1))
    tran.append(100*b['GT']/float(seq_len-1))
    #3.描述位置
    for i in range(0,seq_len):
        c[seq[i:i+1]]+=1
    za=[]#分别提取'ACGT'的索引  
    zc=[]
    zg=[]
    zt=[]
    za1=[]
    zc1=[]
    zg1=[]
    zt1=[]
    for i in range(0,seq_len):#A、C、G、T的位置数值
        if seq[i]=='A':
            za.append(i+1)
        elif seq[i]=='C':
            zc.append(i+1)
        elif seq[i]=='G':
            zg.append(i+1)
        else:
            zt.append(i+1)
    for i in (0,0.25,0.5,0.75,1):#同一碱基处于这五个位置时计算公式
        if ((c['A']*i)>0 and (c['A']*i)<1)or c['A']==0:        
            za1.append(0)
        elif(c['A']*i)==0:
            za1.append(100*za[0]/(seq_len))
        else:
            za1.append(100*(za[int(c['A']*i)-1])/seq_len)
    for i in (0,0.25,0.5,0.75,1):
        if ((c['C']*i)>0 and (c['C']*i)<1)or c['C']==0:        
            zc1.append(0)
        elif(c['C']*i)==0:
            zc1.append(100*zc[0]/(seq_len))
        else:
            zc1.append(100*(zc[int(c['C']*i)-1])/seq_len)
    for i in (0,0.25,0.5,0.75,1):
        if ((c['G']*i)>0 and (c['G']*i)<1)or c['G']==0:        
            zg1.append(0)
        elif(c['G']*i)==0:
            zg1.append(100*zg[0]/(seq_len))
        else:
            zg1.append(100*(zg[int(c['G']*i)-1])/seq_len)
    for i in (0,0.25,0.5,0.75,1):
        if ((c['T']*i)>0 and (c['T']*i)<1)or c['T']==0:        
            zt1.append(0)
        elif(c['T']*i)==0:
            zt1.append(100*zt[0]/(seq_len))
        else:
            zt1.append(100*(zt[int(c['T']*i)-1])/seq_len)
    distri=[]
    distri=za1+zc1+zg1+zt1    
    #4.非监督预测基因算法
    h=0#熵
    for i in 'ACGT':
        if a[i]!=0:
            h+=(-(a[i]/seq_len*(log(a[i]/seq_len))))#数学公式
        else:
            h+=0
    H=[]
    H.append(h)
    s={}#EDP向量
    for i in 'ACGT':
        if a[i]!=0:
            s[i]=(-1/h)*(a[i]/seq_len)*log(a[i]/seq_len)
        else:
            s[i]=0
    S=list(s.values())#将字典里的数值单独提出来转成列表
    p=0#基因的碱基成分约束
    for i in 'ACGT':
        p+=(a[i]/seq_len)*(a[i]/seq_len)
    P=[]
    P.append(p)
    v=comp+tran+distri+S+H+P
    return v
###二联核苷酸相对丰度特征提取方法，衡量两个相邻碱基之间的相关性
def DRA(seq):
    len_seq=len(seq)
    mer1={}
    mer2={}
    for n1 in 'ATGC':###给单元、二元核苷酸出现次数赋初值0
        mer1[n1]=0
        for n2 in 'ATGC':
            mer2[n1+n2]=0
    seq_len=len(seq)
    for p in range(0,seq_len-1):###计算单元、二元核苷酸出现次数
        mer1[seq[p:p+1]]+=1
        mer2[seq[p:p+2]]+=1
    mer1[seq[p+1:p+2]]+=1    
    v1={}
    v2={}
    T={}
    for n1 in 'ACGT':###计算单元、二元核苷酸出现频率
        v1[n1]=mer1[n1]/float((seq_len))
        for n2 in 'ACGT':
            v2[n1+n2]=mer2[n1+n2]/float((seq_len-1))
    for n1 in 'ATGC':###计算二联核苷酸相对丰度
        for n2 in 'ATGC':
            if v1[n1]*v1[n2]==0: ###不存在的二元核苷酸指定其相对丰度为0
                T[n1+n2]=0
            else:
                T[n1+n2]=(v2[n1+n2]/(v1[n1]*v1[n2]))
    f=[]
    for n1 in 'ATGC':
        for n2 in 'ATGC':
            f.append(T[n1+n2])
    return f

###二元核苷酸数值映射特征提取方法，计算16个二元核苷酸的平均个数、期望值、方差
def FE(seq):
    len_seq=len(seq)
    n={}#平均数
    u={}#期望
    D={}#方差
    f={}#例如，第i个位置时AA的数量
    v=[]
    for n1 in 'ATGC':
        for n2 in 'ATGC':
            n[n1+n2]=0
            f[n1+n2]=0
            u[n1+n2]=0
            D[n1+n2]=0
    for i in range(0,len_seq-1):###采用累加累乘计算二元核苷酸的平均次数、期望及其方差
        n[seq[i:i+2]]+=1
        f[seq[i:i+2]]=1
        u[seq[i:i+2]]+=(i+1)*f[seq[i:i+2]]/float(n[seq[i:i+2]])
        t=(i+1-u[seq[i:i+2]])*(i+1-u[seq[i:i+2]])
        D[seq[i:i+2]]+=t*f[seq[i:i+2]]/float(n[seq[i:i+2]]*(len_seq-1))
    for n1 in 'ATGC':
        for n2 in 'ATGC':
            v.append(n[n1+n2])
            v.append(u[n1+n2])
            v.append(D[n1+n2])
    return v

###正四面体特征提取方法，以正四面体内任意一点到四个面的距离为常数构造一个映射
def RT(seq):
    len_seq=len(seq)
    mer={}
    v={}
    t=[]
    x={}
    y={}
    z={}
    for n1 in 'ATGC':###给二元核苷酸出现次数赋初值0
        for n2 in 'ATGC':
            mer[n1+n2]=0
    seq_len=len(seq)
    for p in range(0,seq_len-1):
        mer[seq[p:p+2]]+=1
    for n1 in 'ACGT': ###计算二元核苷酸出现频率
        for n2 in 'ACGT':
            v[n1+n2]=mer[n1+n2]/float((seq_len-1))
    for i in 'ACGT': ###按照公式构造映射
        x[i]=0.5*sqrt(3)*(v[i+'G']+v[i+'T'])
        y[i]=0.5*sqrt(3)*(v[i+'C']+v[i+'G'])
        z[i]=0.5*sqrt(3)*(v[i+'C']+v[i+'T'])
    for i in 'ACGT':
        t.append(x[i])
        t.append(y[i])
        t.append(z[i])
    return t

###多元互信息特征提取方法，按照合并后的二元核苷酸以及三元核苷酸进行计算
def MMI(seq):
    seq_len=len(seq)
    mer1={}
    mer2={}
    mer3={}
    v1={}
    v2={}
    v3={}
    f1={}
    f2={}
    f3={}
    I2={}
    I3={}
    Z=[]
    for n1 in 'ATGC': ###指定默认的每个二元、三元核苷酸的多元互信息取值为0
        mer1[n1]=0
        f1[n1]=0
        for n2 in 'ATGC':
            mer2[n1+n2]=0
            f2[n1+n2]=0
            for n3 in 'ATGC':
                mer3[n1+n2+n3]=0
                f3[n1+n2+n3]=0
    for p in range(seq_len-2):###计算每个一元、二元、三元核苷酸出现次数
        mer1[seq[p:p+1]]+=1
        mer2[seq[p:p+2]]+=1
        mer3[seq[p:p+3]]+=1
    mer1[seq[p+1:p+2]]+=1
    mer1[seq[p+2:p+3]]+=1    
    mer2[seq[p+1:p+3]]+=1
    for n1 in 'ACGT': ###计算一元、二元、三元核苷酸出现频率
        v1[n1]=mer1[n1]/float(seq_len)
        for n2 in 'ACGT':
            v2[n1+n2]=mer2[n1+n2]/float((seq_len-1))
            for n3 in 'ACGT':
                v3[n1+n2+n3]=mer3[n1+n2+n3]/float((seq_len-2))    
    f1['A']=v1['A']
    f1['C']=v1['C']
    f1['G']=v1['G']
    f1['T']=v1['T']            
    f2['AA']=v2['AA']###合并二元核苷酸为10种
    f2['CC']=v2['CC']
    f2['TT']=v2['TT']
    f2['GG']=v2['GG']
    f2['AC']=v2['AC']+v2['CA']
    f2['AG']=v2['AG']+v2['GA']
    f2['AT']=v2['AT']+v2['TA']
    f2['CG']=v2['CG']+v2['GC']
    f2['CT']=v2['CT']+v2['TC']
    f2['GT']=v2['GT']+v2['TG']
    f3['AAA']=v3['AAA'] ###合并三元核苷酸为20种
    f3['CCC']=v3['CCC']
    f3['GGG']=v3['GGG']
    f3['TTT']=v3['TTT']
    f3['AAC']=v3['AAC']+v3['ACA']+v3['CAA']
    f3['AAG']=v3['AAG']+v3['AGA']+v3['GAA']
    f3['AAT']=v3['AAT']+v3['ATA']+v3['TAA']
    f3['ACC']=v3['ACC']+v3['CAC']+v3['CCA']
    f3['ACG']=v3['ACG']+v3['AGC']+v3['CAG']+v3['CGA']+v3['GAC']+v3['GCA']
    f3['ACT']=v3['ACT']+v3['ATC']+v3['CAT']+v3['CTA']+v3['TAC']+v3['TCA']
    f3['AGG']=v3['AGG']+v3['GAG']+v3['GGA']
    f3['AGT']=v3['AGT']+v3['ATG']+v3['GAT']+v3['GTA']+v3['TAG']+v3['TGA']
    f3['ATT']=v3['ATT']+v3['TTA']+v3['TAT']    
    f3['CCG']=v3['CCG']+v3['CGC']+v3['GCC']
    f3['CCT']=v3['CCT']+v3['CTC']+v3['TCC']
    f3['CGG']=v3['CGG']+v3['GCG']+v3['GGC']
    f3['CGT']=v3['CGT']+v3['CTG']+v3['GCT']+v3['GTC']+v3['TCG']+v3['TGC']
    f3['CTT']=v3['CTT']+v3['TCT']+v3['TTC']
    f3['GGT']=v3['GGT']+v3['GTG']+v3['TGG']
    f3['GTT']=v3['GTT']+v3['TGT']+v3['TTG']
    I2={'AA':0, 'CC':0, 'TT':0, 'GG':0, 'AC':0, 'AG':0, 'AT':0, 'CG':0, 'CT':0, 'GT':0}
    I2={'AAA':0, 'AAC':0, 'AAG':0, 'AAT':0, 'ACC':0, 'ACG':0, 'ACT':0, 'AGG':0, 'AGT':0, 'ATT':0, 'CCC':0, 'CCG':0, 'CCT':0, 'CGG':0, 'CGT':0, 'CTT':0, 'GGG':0, 'GGT':0, 'GTT':0, 'TTT':0}
    if f2['AA']!=0:###按照二元核苷酸是否存在讨论，存在则按照公式计算，不存在默认结果为0
        I2['AA']=f2['AA']*log(f2['AA']/float(f1['A']*f1['A']))
    if f2['AC']!=0:
        I2['AC']=f2['AC']*log(f2['AC']/float(f1['A']*f1['C']))
    if f2['AG']!=0:
        I2['AG']=f2['AG']*log(f2['AG']/float(f1['A']*f1['G']))
    if f2['AT']!=0:
        I2['AT']=f2['AT']*log(f2['AT']/float(f1['A']*f1['T']))
    if f2['CC']!=0:
        I2['CC']=f2['CC']*log(f2['CC']/float(f1['C']*f1['C']))
    if f2['CG']!=0:
        I2['CG']=f2['CG']*log(f2['CG']/float(f1['C']*f1['G']))
    if f2['CT']!=0:
        I2['CT']=f2['CT']*log(f2['CT']/float(f1['C']*f1['T']))
    if f2['GG']!=0:
        I2['GG']=f2['GG']*log(f2['GG']/float(f1['G']*f1['G']))
    if f2['GT']!=0:
        I2['GT']=f2['GT']*log(f2['GT']/float(f1['G']*f1['T']))
    if f2['TT']!=0:
        I2['TT']=f2['TT']*log(f2['TT']/float(f1['T']*f1['T']))      
    if f3['AAA']!=0:###按照三元核苷酸中包含的二元核苷酸是否存在讨论具体情况，如果存在则保留对应加和项，不存在则删除对应加和项
        I3['AAA']=f2['AA']*log(f2['AA']/float(f1['A']*f1['A']))+f2['AA']*log(f2['AA']/float(f1['A']))/float(f1['A'])-f3['AAA']*log(f3['AAA']/float(f2['AA']))/float(f2['AA'])    
    if f3['AAC']!=0 and f2['AC']!=0 and f2['AA']!=0:   
        I3['AAC']=f2['AA']*log(f2['AA']/float(f1['A']*f1['A']))+f2['AC']*log(f2['AC']/float(f1['C']))/float(f1['C'])-f3['AAC']*log(f3['AAC']/float(f2['AC']))/float(f2['AC'])
    if f3['AAC']!=0 and f2['AC']!=0 and f2['AA']==0:   
        I3['AAC']=f2['AC']*log(f2['AC']/float(f1['C']))/float(f1['C'])-f3['AAC']*log(f3['AAC']/float(f2['AC']))/float(f2['AC'])
    if f3['AAC']!=0 and f2['AC']==0 and f2['AA']!=0:   
        I3['AAC']=f2['AA']*log(f2['AA']/float(f1['A']*f1['A']))        
    if f3['AAG']!=0 and f2['AG']!=0 and f2['AA']!=0:    
        I3['AAG']=f2['AA']*log(f2['AA']/float(f1['A']*f1['A']))+f2['AG']*log(f2['AG']/float(f1['G']))/float(f1['G'])-f3['AAG']*log(f3['AAG']/float(f2['AG']))/float(f2['AG'])
    if f3['AAG']!=0 and f2['AG']!=0 and f2['AA']==0:    
        I3['AAG']=f2['AG']*log(f2['AG']/float(f1['G']))/float(f1['G'])-f3['AAG']*log(f3['AAG']/float(f2['AG']))/float(f2['AG'])
    if f3['AAG']!=0 and f2['AG']==0 and f2['AA']!=0:    
        I3['AAG']=f2['AA']*log(f2['AA']/float(f1['A']*f1['A']))        
    if f3['AAT']!=0 and f2['AT']!=0 and f2['AA']!=0:    
        I3['AAT']=f2['AA']*log(f2['AA']/float(f1['A']*f1['A']))+f2['AT']*log(f2['AT']/float(f1['T']))/float(f1['T'])-f3['AAT']*log(f3['AAT']/float(f2['AT']))/float(f2['AT'])
    if f3['AAT']!=0 and f2['AT']!=0 and f2['AA']==0:    
        I3['AAT']=f2['AT']*log(f2['AT']/float(f1['T']))/float(f1['T'])-f3['AAT']*log(f3['AAT']/float(f2['AT']))/float(f2['AT'])
    if f3['AAT']!=0 and f2['AT']==0 and f2['AA']!=0:    
        I3['AAT']=f2['AA']*log(f2['AA']/float(f1['A']*f1['A']))        
    if f3['ACC']!=0 and f2['AC']!=0 and f2['CC']!=0:    
        I3['ACC']=f2['AC']*log(f2['AC']/float(f1['A']*f1['C']))+f2['AC']*log(f2['AC']/float(f1['C']))/float(f1['C'])-f3['ACC']*log(f3['ACC']/float(f2['CC']))/float(f2['CC'])
    if f3['ACC']!=0 and f2['AC']!=0 and f2['CC']==0:    
        I3['ACC']=f2['AC']*log(f2['AC']/float(f1['A']*f1['C']))+f2['AC']*log(f2['AC']/float(f1['C']))/float(f1['C'])
    if f3['ACC']!=0 and f2['AC']==0 and f2['CC']!=0:    
        I3['ACC']=f3['ACC']*log(f3['ACC']/float(f2['CC']))/float(f2['CC'])   
    if f3['ACG']!=0 and f2['AC']!=0 and f2['CG']!=0 and f2['AG']!=0:    
        I3['ACG']=f2['AC']*log(f2['AC']/float(f1['A']*f1['C']))+f2['AG']*log(f2['AG']/float(f1['G']))/float(f1['G'])-f3['ACG']*log(f3['ACG']/float(f2['CG']))/float(f2['CG'])
    if f3['ACG']!=0 and f2['AC']!=0 and f2['CG']!=0 and f2['AG']==0:    
        I3['ACG']=f2['AC']*log(f2['AC']/float(f1['A']*f1['C']))-f3['ACG']*log(f3['ACG']/float(f2['CG']))/float(f2['CG'])
    if f3['ACG']!=0 and f2['AC']!=0 and f2['CG']==0 and f2['AG']!=0:    
        I3['ACG']=f2['AC']*log(f2['AC']/float(f1['A']*f1['C']))+f2['AG']*log(f2['AG']/float(f1['G']))/float(f1['G'])
    if f3['ACG']!=0 and f2['AC']==0 and f2['CG']!=0 and f2['AG']!=0:    
        I3['ACG']=f2['AG']*log(f2['AG']/float(f1['G']))/float(f1['G'])-f3['ACG']*log(f3['ACG']/float(f2['CG']))/float(f2['CG'])  
    if f3['ACT']!=0 and f2['AC']!=0 and f2['CT']!=0 and f2['AT']!=0:    
        I3['ACT']=f2['AC']*log(f2['AC']/float(f1['A']*f1['C']))+f2['AT']*log(f2['AT']/float(f1['T']))/float(f1['T'])-f3['ACT']*log(f3['ACT']/float(f2['CT']))/float(f2['CT'])
    if f3['ACT']!=0 and f2['AC']!=0 and f2['CT']!=0 and f2['AT']==0:    
        I3['ACT']=f2['AC']*log(f2['AC']/float(f1['A']*f1['C']))-f3['ACT']*log(f3['ACT']/float(f2['CT']))/float(f2['CT'])
    if f3['ACT']!=0 and f2['AC']!=0 and f2['CT']==0 and f2['AT']!=0:    
        I3['ACT']=f2['AC']*log(f2['AC']/float(f1['A']*f1['C']))+f2['AT']*log(f2['AT']/float(f1['T']))/float(f1['T'])
    if f3['ACT']!=0 and f2['AC']==0 and f2['CT']!=0 and f2['AT']!=0:    
        I3['ACT']=f2['AT']*log(f2['AT']/float(f1['T']))/float(f1['T'])-f3['ACT']*log(f3['ACT']/float(f2['CT']))/float(f2['CT'])   
    if f3['AGG']!=0 and f2['AG']!=0 and f2['GG']!=0:    
        I3['AGG']=f2['AG']*log(f2['AG']/float(f1['A']*f1['G']))+f2['AG']*log(f2['AG']/float(f1['G']))/float(f1['G'])-f3['AGG']*log(f3['AGG']/float(f2['GG']))/float(f2['GG'])
    if f3['AGG']!=0 and f2['AG']!=0 and f2['GG']==0:    
        I3['AGG']=f2['AG']*log(f2['AG']/float(f1['A']*f1['G']))+f2['AG']*log(f2['AG']/float(f1['G']))/float(f1['G'])
    if f3['AGG']!=0 and f2['AG']==0 and f2['GG']!=0:    
        I3['AGG']=f3['AGG']*log(f3['AGG']/float(f2['GG']))/float(f2['GG'])   
    if f3['AGT']!=0 and f2['AT']!=0 and f2['GT']!=0 and f2['AG']==0: 
        I3['AGT']=f2['AT']*log(f2['AT']/float(f1['T']))/float(f1['T'])-f3['AGT']*log(f3['AGT']/float(f2['GT']))/float(f2['GT'])
    if f3['AGT']!=0 and f2['AT']!=0 and f2['GT']==0 and f2['AG']!=0: 
        I3['AGT']=f2['AG']*log(f2['AG']/float(f1['A']*f1['G']))+f2['AT']*log(f2['AT']/float(f1['T']))/float(f1['T'])
    if f3['AGT']!=0 and f2['AT']==0 and f2['GT']!=0 and f2['AG']!=0: 
        I3['AGT']=f2['AG']*log(f2['AG']/float(f1['A']*f1['G']))-f3['AGT']*log(f3['AGT']/float(f2['GT']))/float(f2['GT'])
    if f3['AGT']!=0 and f2['AT']!=0 and f2['GT']!=0 and f2['AG']!=0: 
        I3['AGT']=f2['AG']*log(f2['AG']/float(f1['A']*f1['G']))+f2['AT']*log(f2['AT']/float(f1['T']))/float(f1['T'])-f3['AGT']*log(f3['AGT']/float(f2['GT']))/float(f2['GT'])    
    if f3['ATT']!=0 and f2['AT']!=0 and f2['TT']!=0:    
        I3['ATT']=f2['AT']*log(f2['AT']/float(f1['A']*f1['T']))+f2['AT']*log(f2['AT']/float(f1['T']))/float(f1['T'])-f3['ATT']*log(f3['ATT']/float(f2['TT']))/float(f2['TT'])
    if f3['ATT']!=0 and f2['AT']!=0 and f2['TT']==0:    
        I3['ATT']=f2['AT']*log(f2['AT']/float(f1['A']*f1['T']))+f2['AT']*log(f2['AT']/float(f1['T']))/float(f1['T'])
    if f3['ATT']!=0 and f2['AT']==0 and f2['TT']!=0:    
        I3['ATT']=f3['ATT']*log(f3['ATT']/float(f2['TT']))/float(f2['TT'])        
    if f3['CCC']!=0:    
        I3['CCC']=f2['CC']*log(f2['CC']/float(f1['C']*f1['C']))+f2['CC']*log(f2['CC']/float(f1['C']))/float(f1['C'])-f3['CCC']*log(f3['CCC']/float(f2['CC']))/float(f2['CC'])   
    if f3['CCG']!=0 and f2['CG']!=0 and f2['CC']!=0:    
        I3['CCG']=f2['CC']*log(f2['CC']/float(f1['C']*f1['C']))+f2['CG']*log(f2['CG']/float(f1['G']))/float(f1['G'])-f3['CCG']*log(f3['CCG']/float(f2['CG']))/float(f2['CG'])
    if f3['CCG']!=0 and f2['CG']!=0 and f2['CC']==0:    
        I3['CCG']=f2['CG']*log(f2['CG']/float(f1['G']))/float(f1['G'])-f3['CCG']*log(f3['CCG']/float(f2['CG']))/float(f2['CG'])
    if f3['CCG']!=0 and f2['CG']==0 and f2['CC']!=0:    
        I3['CCG']=f2['CC']*log(f2['CC']/float(f1['C']*f1['C']))       
    if f3['CCT']!=0 and f2['CT']!=0 and f2['CC']!=0:    
        I3['CCT']=f2['CC']*log(f2['CC']/float(f1['C']*f1['C']))+f2['CT']*log(f2['CT']/float(f1['T']))/float(f1['T'])-f3['CCT']*log(f3['CCT']/float(f2['CT']))/float(f2['CT'])
    if f3['CCT']!=0 and f2['CT']!=0 and f2['CC']==0:    
        I3['CCT']=f2['CT']*log(f2['CT']/float(f1['T']))/float(f1['T'])-f3['CCT']*log(f3['CCT']/float(f2['CT']))/float(f2['CT'])
    if f3['CCT']!=0 and f2['CT']==0 and f2['CC']!=0:    
        I3['CCT']=f2['CC']*log(f2['CC']/float(f1['C']*f1['C']))        
    if f3['CGG']!=0 and f2['CG']!=0 and f2['GG']!=0:    
        I3['CGG']=f2['CG']*log(f2['CG']/float(f1['C']*f1['G']))+f2['CG']*log(f2['CG']/float(f1['G']))/float(f1['G'])-f3['CGG']*log(f3['CGG']/float(f2['GG']))/float(f2['GG'])
    if f3['CGG']!=0 and f2['CG']!=0 and f2['GG']==0:    
        I3['CGG']=f2['CG']*log(f2['CG']/float(f1['C']*f1['G']))+f2['CG']*log(f2['CG']/float(f1['G']))/float(f1['G'])
    if f3['CGG']!=0 and f2['CG']==0 and f2['GG']!=0:    
        I3['CGG']=f3['CGG']*log(f3['CGG']/float(f2['GG']))/float(f2['GG'])    
    if f3['CGT']!=0 and f2['CG']!=0 and f2['GT']!=0 and f2['CT']!=0:    
        I3['CGT']=f2['CG']*log(f2['CG']/float(f1['C']*f1['G']))+f2['CT']*log(f2['CT']/float(f1['T']))/float(f1['T'])-f3['CGT']*log(f3['CGT']/float(f2['GT']))/float(f2['GT'])
    if f3['CGT']!=0 and f2['CG']!=0 and f2['GT']!=0 and f2['CT']==0:    
        I3['CGT']=f2['CG']*log(f2['CG']/float(f1['C']*f1['G']))-f3['CGT']*log(f3['CGT']/float(f2['GT']))/float(f2['GT'])
    if f3['CGT']!=0 and f2['CG']!=0 and f2['GT']==0 and f2['CT']!=0:    
        I3['CGT']=f2['CG']*log(f2['CG']/float(f1['C']*f1['G']))+f2['CT']*log(f2['CT']/float(f1['T']))/float(f1['T'])
    if f3['CGT']!=0 and f2['CG']==0 and f2['GT']!=0 and f2['CT']!=0:    
        I3['CGT']=f2['CT']*log(f2['CT']/float(f1['T']))/float(f1['T'])-f3['CGT']*log(f3['CGT']/float(f2['GT']))/float(f2['GT'])        
    if f3['CTT']!=0 and f2['CT']!=0 and f2['TT']!=0:    
        I3['CTT']=f2['CT']*log(f2['CT']/float(f1['C']*f1['T']))+f2['CT']*log(f2['CT']/float(f1['T']))/float(f1['T'])-f3['CTT']*log(f3['CTT']/float(f2['TT']))/float(f2['TT'])
    if f3['CTT']!=0 and f2['CT']!=0 and f2['TT']==0:    
        I3['CTT']=f2['CT']*log(f2['CT']/float(f1['C']*f1['T']))+f2['CT']*log(f2['CT']/float(f1['T']))/float(f1['T'])
    if f3['CTT']!=0 and f2['CT']==0 and f2['TT']!=0:    
        I3['CTT']=f3['CTT']*log(f3['CTT']/float(f2['TT']))/float(f2['TT'])       
    if f3['GGG']!=0:    
        I3['GGG']=f2['GG']*log(f2['GG']/float(f1['G']*f1['G']))+f2['GG']*log(f2['GG']/float(f1['G']))/float(f1['G'])-f3['GGG']*log(f3['GGG']/float(f2['GG']))/float(f2['GG'])   
    if f3['GGT']!=0 and f2['GT']!=0 and f2['GG']!=0:    
        I3['GGT']=f2['GG']*log(f2['GG']/float(f1['G']*f1['G']))+f2['GT']*log(f2['GT']/float(f1['T']))/float(f1['T'])-f3['GGT']*log(f3['GGT']/float(f2['GT']))/float(f2['GT'])
    if f3['GGT']!=0 and f2['GT']!=0 and f2['GG']==0:    
        I3['GGT']=f2['GT']*log(f2['GT']/float(f1['T']))/float(f1['T'])-f3['GGT']*log(f3['GGT']/float(f2['GT']))/float(f2['GT'])
    if f3['GGT']!=0 and f2['GT']==0 and f2['GG']!=0:    
        I3['GGT']=f2['GG']*log(f2['GG']/float(f1['G']*f1['G']))       
    if f3['GTT']!=0 and f2['GT']!=0 and f2['TT']!=0:   
        I3['GTT']=f2['GT']*log(f2['GT']/float(f1['G']*f1['T']))+f2['GT']*log(f2['GT']/float(f1['T']))/float(f1['T'])-f3['GTT']*log(f3['GTT']/float(f2['TT']))/float(f2['TT'])
    if f3['GTT']!=0 and f2['GT']!=0 and f2['TT']==0:   
        I3['GTT']=f2['GT']*log(f2['GT']/float(f1['G']*f1['T']))+f2['GT']*log(f2['GT']/float(f1['T']))/float(f1['T'])
    if f3['GTT']!=0 and f2['GT']==0 and f2['TT']!=0:   
        I3['GTT']=f3['GTT']*log(f3['GTT']/float(f2['TT']))/float(f2['TT'])
    if f3['TTT']!=0:    
        I3['TTT']=f2['TT']*log(f2['TT']/float(f1['T']*f1['T']))+f2['TT']*log(f2['TT']/float(f1['T']))/float(f1['T'])-f3['TTT']*log(f3['TTT']/float(f2['TT']))/float(f2['TT'])
    z=list(f2.values())+list(f3.values())
    return z
def hash1(seq):
    binary_dictionary={'A':0, 'C':1, 'G':2, 'T':3}
    seq=seq.strip('N')
    seq_len=len(seq)
    cnt=[]
    for i in seq:
        cnt.append(binary_dictionary[i])
    v2=[]
    for p in range(0,seq_len-1):
        v2.append(4*cnt[p]+cnt[p+1])
    return v2
def frequence(seq):
    mer1={}
    mer2={}
    PA=[]
    PT=[]
    PC=[]
    PG=[]
    X=[]
    M=[]
    seq_len=len(seq)
    for n in 'ATGC':
        mer1[n]=0
        
    for p in range(0,seq_len):
        mer1[seq[p]]+=1
    X.append(mer1['A']/float(seq_len))
    X.append(mer1['T']/float(seq_len))
    X.append(mer1['G']/float(seq_len))
    X.append(mer1['C']/float(seq_len))
    PA.append(mer1['A']/float(seq_len))
    PT.append(mer1['T']/float(seq_len))
    PG.append(mer1['G']/float(seq_len))
    PC.append(mer1['C']/float(seq_len))
    mer1={}
    m=int(seq_len/2)
    for n in 'ATGC':
        mer1[n]=0
    for i in range(0,m):
        mer1[seq[i]]+=1
    PA.append(mer1['A']/float(m))
    PT.append(mer1['T']/float(m))
    PG.append(mer1['G']/float(m))
    PC.append(mer1['C']/float(m))
    for n in 'ATGC':
        mer1[n]=0
    for i1 in range(m,seq_len):
        mer1[seq[i1]]+=1
    PA.append(mer1['A']/float(m+seq_len-m*2))
    PT.append(mer1['T']/float(m+seq_len-m*2))
    PG.append(mer1['G']/float(m+seq_len-m*2))
    PC.append(mer1['C']/float(m+seq_len-m*2))

    M.append((PA[1]+PA[2]-PG[1]-PG[2])/float(m))
    M.append((PA[1]+PA[2]-PC[1]-PC[2])/float(m))
    M.append((PA[1]+PA[2]-PT[1]-PT[2])/float(m))
    M.append((PG[1]+PG[2]-PC[1]-PC[2])/float(m))
    M.append((PG[1]+PG[2]-PT[1]-PT[2])/float(m))
    M.append((PC[1]+PC[2]-PT[1]-PT[2])/float(m))
    
    for i2 in range(0,6):
        X.append(math.fabs(M[i2]))
        
    return X
def condon(seq):
#two bases that are adjacent to f5c tend to be A,U,G, but not C    
    cnt=[]
    if seq[49]=='T':
        cnt.append(0.75)
    elif seq[49]=='A':
        cnt.append(0.5)
    elif seq[49]=='G':
        cnt.append(0.25)
    else:
        cnt.append(0)

    if seq[51]=='T':
        cnt.append(0.75)
    elif seq[51]=='A':
        cnt.append(0.5)
    elif seq[51]=='G':
        cnt.append(0.25)
    else:
        cnt.append(0)
    
#C in the third position of condons tends to contain a higher proportion of f5c modification.
#frame 3
    if seq[48:51]=='ATC'or seq[48:51]=='TTC'or seq[48:51]=='TAC'or seq[48:51]=='AAC':
        cnt.append(0.75)
    elif seq[48:51]=='GTC'or seq[48:51]=='GAC'or seq[48:51]=='CAC'or seq[48:51]=='AGC':
        cnt.append(0.5)
    elif seq[48:51]=='GCC'or seq[48:51]=='ACC'or seq[48:51]=='CTC'or seq[48:51]=='TGC'or seq[48:51]=='GGC':
        cnt.append(0.25)
    else:
        cnt.append(0)
#frame 2
    if seq[49:52]=='TCT'or seq[49:52]=='GCT'or seq[49:52]=='ACT'or seq[49:52]=='ACA'or seq[49:52]=='TCA':
        cnt.append(0.75)
    elif seq[49:52]=='TCG'or seq[49:52]=='ACG'or seq[49:52]=='CCT'or seq[49:52]=='GCG':
        cnt.append(0.5)
    elif seq[49:52]=='GCA':
        cnt.append(0.25)
    else:
        cnt.append(0)
#frame 1
    if seq[50:53]=='CAA'or seq[50:53]=='CTA'or seq[50:53]=='CAT'or seq[50:53]=='CTT':
        cnt.append(0.75)
    elif seq[50:53]=='CTG'or seq[50:53]=='CGT'or seq[50:53]=='CAG':
        cnt.append(0.5)
    else:
        cnt.append(0)        
    return cnt    
        
        
#v1=condon('ACAGGCCGCACAAGAGTACTCTACGGAAAAAAATACAAACACCTTGCCATCATTGGTCAAGGGGGCCAACATTGCCAGCTTCGTCATGGTGGCTGACGCAA')
#v=nucleic_shift('ACGTACGT')  
#print (v) 

def CKSNAP(fastas, gap=5, **kw):
    if gap < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0

    if check_sequences.get_min_sequence_length(fastas) < gap + 2:
        print('Error: all the sequence length should be larger than the (gap value) + 2 = ' + str(gap + 2) + '\n\n')
        return 0

    AA = kw['order'] if kw['order'] != None else 'ACGT'
    encodings = []
    aaPairs = []
    for aa1 in AA:
        for aa2 in AA:
            aaPairs.append(aa1 + aa2)

    header = ['#', 'label']
    for g in range(gap + 1):
        for aa in aaPairs:
            header.append(aa + '.gap' + str(g))
    encodings.append(header)

    for i in fastas:
        name, sequence, label = i[0], i[1], i[2]
        code = [name, label]
        for g in range(gap + 1):
            myDict = {}
            for pair in aaPairs:
                myDict[pair] = 0
            sum = 0
            for index1 in range(len(sequence)):
                index2 = index1 + g + 1
                if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[
                    index2] in AA:
                    myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                    sum = sum + 1
            for pair in aaPairs:
                code.append(myDict[pair] / sum)
        encodings.append(code)
    return encodings