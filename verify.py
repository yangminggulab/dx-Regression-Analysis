#!/usr/bin/env python3
"""核验笔记条目、既有原文哈希和核心回归公式；不调用 OCR 或转换程序。"""
from pathlib import Path
from fractions import Fraction as F
import hashlib, json, re
import numpy as np
project = Path(__file__).resolve().parent
items = json.loads((project / '条目清单.json').read_text())
assert len(items) == 82
assert len({x['original'] for x in items if x['original']}) == 60
tex = '\n'.join((project / f).read_text() for f in ['第一章.tex','第二章.tex','第三章.tex','作业.tex'])
labels = re.findall(r'\\label\{([^}]+)\}',tex)
assert len(labels) == len(set(labels)), '标签重复'
for x in items:
    assert tex.count('\\label{'+x['label']+'}') == 1
hw = (project/'作业.tex').read_text()
assert hw.count('\\section{') == 3
assert hw.count('\\begin{homework}') == 11
assert hw.count('\\begin{dxtips}') == 11
assert hw.count('\\begin{proposition}') == 16
assert hashlib.sha256((project/'elegantbook.cls').read_bytes()).hexdigest() == '1c5402a8aa9d71bda86a9d239b2b72a6392d3bd931116cb3cfeb3cc86e6ea1b3'
# 源文件仅在课程工作区可用，公开仓库不收录未整理的教材全文。
manifest=json.loads((project/'原文校验清单.json').read_text())
if all((project.parent/x['path']).is_file() for x in manifest):
    for x in manifest:
        assert hashlib.sha256((project.parent/x['path']).read_bytes()).hexdigest()==x['sha256'], x['path']
    print(f'原文保护：{len(manifest)} 文件 SHA-256 均未改变')
else:
    print('当前目录无原始课程文件：跳过原文哈希比对')
# 保留每一组既有阅读笔记正文；只允许记录在校勘表中的一处条件修正。
for n,name in [(1,'第一次作业'),(2,'第二次作业'),(3,'第三次作业')]:
    source=project.parent/'回归分析'/name/'main.tex'
    if source.is_file():
        blocks=re.findall(r'\\begin\{(?:note|summary)\}\{[^}]*\}(.*?)\\end\{(?:note|summary)\}',source.read_text(),re.S)
        for body in blocks:
            body=body.replace('对任意可积的可测函数 $h(X)$','对任意满足 $\\E|h(X)e|<\\infty$ 的可测函数 $h(X)$')
            assert body.strip() in hw, '遗漏既有阅读笔记'
# 第 11 题：明确以行为 Y、列为 X。
p=np.array([[.1,.2],[.4,.3]])
px=p.sum(axis=0);mu=p[1]/px
assert np.allclose(px,[.5,.5])
assert np.allclose(mu,[.8,.6])
assert np.allclose(mu*(1-mu),[.16,.24])
# 第 18 题：对多项式联合密度精确积分。
def moment(a,b):
    return F(3,2)*(F(1,(a+3)*(b+1))+F(1,(a+1)*(b+3)))
assert moment(0,0)==1
ex,ey=moment(1,0),moment(0,1)
var=moment(2,0)-ex*ex;cov=moment(1,1)-ex*ey
beta=cov/var;alpha=ey-beta*ex
assert (ex,ey,var,cov,beta,alpha)==(F(5,8),F(5,8),F(73,960),F(-1,64),F(-15,73),F(55,73))
# 非正交样本检验 OLS、投影与限制回归，避免只检验单位阵。
X=np.column_stack([np.ones(7),np.arange(7),[1,-1,2,0,3,-2,1]])
b=np.array([1.,2.,-.5]); noise=np.array([.2,-.7,.4,.5,-.3,.8,-.1]); y=X@b+noise
Q=np.linalg.inv(X.T@X); bhat=Q@X.T@y; P=X@Q@X.T; M=np.eye(7)-P; uh=y-X@bhat
assert np.allclose(P,P.T) and np.allclose(P@P,P)
assert np.allclose(M,M.T) and np.allclose(M@M,M)
assert np.allclose(X.T@uh,0) and np.allclose(P@M,0)
assert np.isclose(np.trace(P),3) and np.isclose(np.trace(M),4)
R=np.array([[0.,1.,1.]]);r=R@b; H=np.linalg.inv(R@Q@R.T)
A=Q@R.T@H@R@Q;D=(Q-A)@X.T;C=M+X@A@X.T
bcls=bhat-Q@R.T@H@(R@bhat-r);ucls=y-X@bcls
assert np.allclose(R@bcls,r)
assert np.allclose(D@D.T,Q-A)
assert np.allclose(C@C,C) and np.isclose(np.trace(C),5)
assert np.allclose(ucls,C@noise)
assert np.linalg.eigvalsh(A).min()>-1e-10
rss=uh@uh;rssc=ucls@ucls
quadratic=float((R@bhat-r).T@H@(R@bhat-r))
assert np.isclose(rssc-rss,quadratic)
tss=((y-y.mean())**2).sum();ess=((P@y-y.mean())**2).sum()
assert np.isclose(tss,ess+rss)
print('结构：82 个正文条目 / 60 个原编号；3 次作业 / 16 组原笔记 / 11 道习题及提示')
print('数学校验：离散条件矩、密度积分、OLS 正规方程、P/M 投影、CLS 限制与自由度、两种 F 分子均通过')
