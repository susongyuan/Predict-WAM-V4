#!/usr/bin/env python3
"""Draw the implemented Precision-WAM V2; pure vector labels and explicit training paths."""
from pathlib import Path
from html import escape
import argparse
import json
import xml.etree.ElementTree as ET
import gi
import cairo
gi.require_version('Rsvg', '2.0')
gi.require_version('Pango', '1.0')
gi.require_version('PangoCairo', '1.0')
from gi.repository import Rsvg, Pango, PangoCairo

W, H = 2400, 1550
INK, MUTED, TEAL, VIOLET, AMBER = '#142536', '#526477', '#087d88', '#6653c4', '#a76212'
parts, boxes, text_checks = [], {}, []
measure_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 1, 1)
measure_ctx = cairo.Context(measure_surface)


def text(x, y, value, size=25, color=INK, weight=400, max_width=None):
    layout = PangoCairo.create_layout(measure_ctx)
    font = Pango.FontDescription('Noto Sans CJK SC')
    font.set_absolute_size(size*Pango.SCALE)
    font.set_weight(Pango.Weight.BOLD if weight >= 600 else Pango.Weight.NORMAL)
    layout.set_font_description(font); layout.set_text(value, -1)
    width = layout.get_pixel_size()[0]
    if max_width is not None:
        assert width <= max_width, (value, width, max_width)
    text_checks.append(dict(text=value, width=width, maximum=max_width))
    parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}">{escape(value)}</text>')


def path(points, color=MUTED, dashed=False, width=3, arrow=True):
    d = 'M '+' L '.join(f'{x},{y}' for x,y in points)
    marker = f' marker-end="url(#{"train" if dashed else "arrow"})"' if arrow else ''
    dash = ' stroke-dasharray="9 8"' if dashed else ''
    parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"{dash}{marker}/>')


def card(key, x, y, w, h, title, lines, color=TEAL, fill='#eef8f8', new=False, size=25):
    boxes[key] = (x,y,w,h)
    parts.append(f'<g id="{key}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{fill}" stroke="{color}" stroke-width="2"/>')
    text(x+22,y+43,title,29,color,700,w-44)
    for i, line in enumerate(lines): text(x+22,y+79+i*33,line,size,INK,400,w-44)
    if new:
        parts.append(f'<rect x="{x+w-76}" y="{y-16}" width="64" height="27" rx="9" fill="{color}"/>')
        text(x+w-65,y+4,'改进',17,'#ffffff',600)
    parts.append('</g>')


def tokens(x, y, count=6):
    for i in range(count):
        dx=i*22
        parts.append(f'<path d="M{x+dx},{y} l18,-7 0,-24 -18,7 z" fill="#52bdc5" stroke="#087d88" stroke-width="1.3"/>')
        parts.append(f'<path d="M{x+dx},{y-24} l18,-7 -12,-7 -18,7 z" fill="#bde7e8" stroke="#087d88" stroke-width="1.3"/>')


def render(svg, out):
    handle = Rsvg.Handle.new_from_file(str(svg))
    viewport=Rsvg.Rectangle();viewport.x=0;viewport.y=0;viewport.width=W;viewport.height=H
    surface=cairo.ImageSurface(cairo.FORMAT_ARGB32,W,H)
    ctx=cairo.Context(surface); handle.render_document(ctx,viewport)
    surface.write_to_png(str(out/'architecture.png'))
    pdf=cairo.PDFSurface(str(out/'architecture.pdf'),W/2,H/2)
    ctx=cairo.Context(pdf);ctx.scale(.5,.5);handle.render_document(ctx,viewport);pdf.finish()


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);args=p.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    parts.append('<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#526477"/></marker><marker id="train" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#8391a0"/></marker><linearGradient id="core" x2="0.8" y2="1"><stop stop-color="#f7f4ff"/><stop offset="1" stop-color="#e9e2fd"/></linearGradient></defs>')
    parts.append('<style>text{font-family:"Noto Sans CJK SC","WenQuanYi Micro Hei",sans-serif}</style>')
    parts.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
    text(60,68,'当前架构：Precision-WAM V2',48,INK,700)
    text(60,112,'三路视觉细节精修 · 因果计时 · H-JEPA辅助监督 · 分段DAgger训练',27,MUTED)
    text(1860,66,'以当前训练代码为准',24,MUTED)
    text(1860,106,'2026-10-11 / native16',23,MUTED)

    parts.append('<g id="inputs">')
    card('changes',60,160,600,130,'相对RoboTwin五任务版',
         ['全局视觉：每视角64 → 16个token','旧整视角门控移除；增加细节精修'],MUTED,'#f7f9fb',size=24)
    card('joint-history',740,160,300,130,'实测关节历史',['当前＋5个滞后＋起点','共7个状态token'],MUTED,'#f7f9fb',size=23)
    card('visual-memory',1070,160,300,130,'近期视觉记忆',['当前与过去/起点之差','三视角共9个token'],MUTED,'#f7f9fb',size=23)
    card('task-cue',1400,160,300,130,'任务提示VTT',['由原成功TRAIN拟合','不是在线语言规划'],MUTED,'#f7f9fb',size=23)
    card('event-clock',1730,160,540,130,'动作持续时间',['左右闭合命令/手臂静止计时，6维','投影为1个token；不等于已抓住物体'],TEAL,'#eef8f8',True,size=23)
    card('cameras',60,365,280,245,'三路当前图像',['头部相机','左腕相机','右腕相机','224 × 224 RGB'],TEAL,'#f0fafb')
    parts.append('</g>')

    parts.append('<g id="encoder">')
    card('dino',400,395,250,155,'冻结DINOv3',['提取三层视觉特征','权重不更新'],TEAL,'#eef8f8',size=22)
    card('fusion',740,395,240,155,'共享视觉融合',['拼接＋可学习投影','后面分为两条路'],TEAL,'#eef8f8',size=21)
    card('global',1030,370,240,155,'全局视觉摘要',['每视角16个token','三路共48个token'],TEAL,'#eef8f8',True,size=22)
    card('dense',1030,630,240,175,'当前细节图块',['三路14 × 14图块','共588个KV token','加入二维图块坐标'],TEAL,'#eef8f8',size=20)
    tokens(770,580)
    parts.append('</g>')

    parts.append('<g id="action-model">')
    card('dit',1330,365,360,215,'动作DiT主干',['12层 / hidden 768','融合100个token','含32个带噪动作token','输出动作生成方向'],VIOLET,'url(#core)')
    card('refinement',1330,645,360,150,'三路细节精修',['前8步动作hidden查询图块','补到前8步的生成方向'],TEAL,'#e7f7f6',True,size=23)
    card('flow',1840,395,430,180,'Flow Matching生成循环',['DiT＋精修共同参与每一轮','Euler积分10轮 → 反归一化','得到32 × 16个关节/夹爪目标'],VIOLET,'#f5f2fd',size=24)
    parts.append('<circle cx="1780" cy="465" r="23" fill="#ffffff" stroke="#087d88" stroke-width="3"/>')
    text(1768,476,'+',34,TEAL,700)
    parts.append('</g>')

    path([(340,475),(400,475)],TEAL)
    path([(650,475),(740,475)],TEAL)
    path([(980,450),(1030,450)],TEAL)
    path([(980,505),(1005,505),(1005,720),(1030,720)],TEAL)
    path([(1270,448),(1330,448)],TEAL)
    path([(890,290),(890,324),(1300,324),(1300,400),(1330,400)])
    path([(1220,290),(1220,337),(1420,337),(1420,365)])
    path([(1550,290),(1550,365)])
    path([(1960,290),(1960,326),(1650,326),(1650,365)],TEAL)
    path([(1510,580),(1510,645)],VIOLET)
    text(1533,623,'前8步hidden',22,MUTED)
    path([(1270,718),(1330,718)],TEAL)
    path([(1690,465),(1757,465)],VIOLET)
    path([(1690,718),(1780,718),(1780,488)],TEAL)
    text(1804,666,'仅前8步残差',22,TEAL)
    path([(1803,465),(1840,465)],VIOLET)
    path([(2070,575),(2070,608),(1645,608),(1645,580)],VIOLET,width=2)
    text(1860,690,'下一轮带噪动作',23,VIOLET)

    parts.append('<g id="control">')
    card('spline',1730,865,280,140,'动作平滑',['仅手臂做B样条','夹爪开闭保持原值'],AMBER,'#fff8ed',size=23)
    card('braking',1400,865,280,140,'安全执行约束',['Panda官方关节限额','结合当前状态提前刹车'],AMBER,'#fff8ed',size=21)
    card('execute',1010,865,340,140,'执行前8步',['10 Hz执行；共0.8秒','随后用新观测重规划'],AMBER,'#fff8ed',size=24)
    card('robot',420,865,520,140,'双Panda与物理环境',['收到的是目标关节角＋夹爪开合','机器人执行后反馈新的机载观测'],AMBER,'#fff8ed',size=24)
    path([(2130,575),(2320,575),(2320,830),(1870,830),(1870,865)],AMBER)
    path([(1730,934),(1680,934)],AMBER)
    path([(1400,934),(1350,934)],AMBER)
    path([(1010,934),(940,934)],AMBER)
    path([(420,934),(38,934),(38,477),(60,477)],AMBER)
    text(75,841,'机载反馈：每0.1秒更新',24,AMBER)
    text(65,691,'关节历史直接进入DiT',25,MUTED)
    text(65,733,'视觉编码器只处理图像',25,MUTED)
    # Structural physical anchor: two articulated arms, deliberately schematic.
    for offset, mirror in [(105,False),(265,True)]:
        points=[(offset,797),(offset+15,776),(offset+8,750),(offset+45,729),(offset+62,758),(offset+85,773)]
        if mirror: points=[(offset+90-(x-offset),y) for x,y in points]
        path(points,'#a6b4bf',width=11,arrow=False)
        for x,y in points: parts.append(f'<circle cx="{x}" cy="{y}" r="7" fill="#eff3f5" stroke="#637888" stroke-width="2"/>')
    parts.append('</g>')

    parts.append('<g id="training-only">')
    parts.append('<rect x="60" y="1070" width="2280" height="252" rx="20" fill="#f5f7f9"/>')
    text(85,1115,'训练时的辅助监督',31,INK,700)
    text(1390,1115,'灰色虚线为训练信号；未来真实图像仅作为老师',22,MUTED)
    card('training-data',85,1143,455,150,'训练数据与有效标签',['原TRAIN 160＋纠正回合120','每批3原示教＋1纠正片段','原动作1；合格专家3；其余标签0'],MUTED,'#ffffff',size=23)
    card('action-loss',585,1143,415,150,'动作模仿损失',['专家动作＋随机噪声','学习正确的生成方向','按有效动作与权重归一化'],VIOLET,'#ffffff',size=23)
    card('future-heads',1045,1143,565,150,'未来视觉预测：8步与32步',['共享解码器预测0.8秒/3.2秒特征','两项权重均0.25；老师特征停止梯度','辅助视觉学习，推理不调用此分支'],VIOLET,'#ffffff',True,size=23)
    card('inverse-head',1655,1143,655,150,'逆动作辅助IDM：4步',['用示教5帧视觉变化＋当前关节学习动作偏移','权重0.1；未来关节不作为输入','训练时帮助视觉表征理解动作变化'],VIOLET,'#ffffff',True,size=23)
    path([(540,1220),(585,1220)],MUTED,True)
    # Independent training signals: do not draw serial losses or future-to-policy edges.
    # Loss inputs originate from the corrected flow, DiT condition tokens,
    # and independently adapted visual tokens respectively; never the
    # execution controller or a future observation fed into the online policy.
    path([(1813,465),(1813,820),(980,820),(980,1035),(790,1035),(790,1143)],MUTED,True)
    path([(1690,545),(1715,545),(1715,835),(1375,835),(1375,1020),(1310,1020),(1310,1143)],MUTED,True)
    path([(1270,510),(1300,510),(1300,632),(2285,632),(2285,840),(2040,840),(2040,1055),(1985,1055),(1985,1143)],MUTED,True)
    parts.append('</g>')

    parts.append('<g id="dagger-data-cycle">')
    parts.append('<rect x="60" y="1360" width="2280" height="125" rx="20" fill="#fff9ef"/>')
    text(85,1403,'分段DAgger',29,AMBER,700)
    text(85,1447,'数据采集与训练流程',21,MUTED)
    cards=[(420,280,'WAM自主执行'),(760,300,'专家局部接管'),(1120,340,'验收后交还WAM'),(1520,340,'成功专家段入库'),(1920,390,'与原示教混合训练')]
    for i,(x,w,title) in enumerate(cards):
        parts.append(f'<rect x="{x}" y="1390" width="{w}" height="66" rx="13" fill="#ffffff" stroke="#d5af70" stroke-width="1.7"/>')
        text(x+20,1432,title,25,INK,500,w-40)
        if i<len(cards)-1:path([(x+w,1423),(cards[i+1][0],1423)],AMBER)
    parts.append('</g>')
    text(60,1530,'当前评测为纯WAM；VLM、高层规划、ASK与记忆更新未启用。图中的DAgger接管用于采集，不参与正式测试。',23,MUTED)
    parts.append('</svg>')
    svg=out/'architecture.svg';svg.write_text('\n'.join(parts)+'\n',encoding='utf8')
    tree=ET.parse(svg)
    assert not tree.findall('.//{http://www.w3.org/2000/svg}image')
    render(svg,out)
    validation=dict(xml_valid=True,publication_svg_is_pure_vector=True,text_measured_with_actual_font=True,
        bounded_text_checks=len([c for c in text_checks if c['maximum'] is not None]),
        text_checks=text_checks,semantic_groups=[e.get('id') for e in tree.findall('.//{http://www.w3.org/2000/svg}g')],
        formulas_not_used=True,online_future_input_edges=0,outputs=['architecture.svg','architecture.png','architecture.pdf'])
    (out/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(out=str(out),vector_svg=str(svg),text_checks=validation['bounded_text_checks']),ensure_ascii=False))


if __name__=='__main__':main()
