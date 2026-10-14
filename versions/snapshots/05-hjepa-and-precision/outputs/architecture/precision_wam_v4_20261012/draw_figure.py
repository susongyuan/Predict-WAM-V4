#!/usr/bin/env python3
"""CPU-only vector architecture with three unmodified archived sensor frames."""
from pathlib import Path
from html import escape
import json, base64, xml.etree.ElementTree as ET
import os
EXEC_STEPS=int(os.environ.get('FIGURE_EXEC_STEPS','8'))
PREFIX=os.environ.get('FIGURE_PREFIX','architecture')
import gi, cairo
gi.require_version('Rsvg','2.0'); gi.require_version('Pango','1.0'); gi.require_version('PangoCairo','1.0')
from gi.repository import Rsvg,Pango,PangoCairo
OUT=Path(__file__).resolve().parent
W,H=2800,1610
INK='#13254b'; MUTED='#52667e'; CYAN='#088caa'; PURPLE='#7540d1'; BLUE='#3859c0'; AMBER='#dd8b13'; GREEN='#35a95b'
p=[]; checks=[]
ctx=cairo.Context(cairo.ImageSurface(cairo.FORMAT_ARGB32,1,1))
def text(x,y,s,size=26,color=INK,bold=False,maxw=None,anchor='start'):
    original=size
    while True:
        lay=PangoCairo.create_layout(ctx); font=Pango.FontDescription('Noto Sans CJK SC'); font.set_absolute_size(size*Pango.SCALE);font.set_weight(Pango.Weight.BOLD if bold else Pango.Weight.NORMAL);lay.set_font_description(font);lay.set_text(s,-1);width=lay.get_pixel_size()[0]
        if maxw is None or width<=maxw: break
        size-=1
        assert size>=18,(s,width,maxw)
    checks.append(dict(text=s,width=width,maxw=maxw,size=size))
    p.append(f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" font-weight="{700 if bold else 400}" text-anchor="{anchor}">{escape(s)}</text>')
def box(x,y,w,h,fill='#fff',stroke='#d1dbe8',r=16):
    p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
def line(pts,color=BLUE,width=4,arrow=True,dash=False):
    d='M'+' L'.join(f'{a},{b}' for a,b in pts)
    marker={'#088caa':'cyan','#7540d1':'purple','#3859c0':'blue','#dd8b13':'amber','#52667e':'gray','#35a95b':'green'}.get(color,'blue')
    p.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"'+(f' marker-end="url(#{marker})"' if arrow else '')+(' stroke-dasharray="9 8"' if dash else '')+'/>')
def cube(x,y,w=20,h=52,color=CYAN):
    palette={CYAN:('#40b9d8','#b1e4f0','#1486a6'),BLUE:('#6b8ff0','#bacbff','#3757b3'),PURPLE:('#ad84e8','#dfc9ff','#794ec0'),AMBER:('#f5b658','#ffe0a4','#bd7b1a'),GREEN:('#69c780','#cbefd4','#319551'),'#a6b4c7':('#bec8d5','#e6eaf0','#8a99ad')}
    front,top,side=palette[color]
    p.append(f'<path d="M{x},{y} l{w},0 0,{h} -{w},0z" fill="{front}" stroke="{side}" stroke-width="1"/><path d="M{x},{y} l9,-7 {w},0 -9,7z" fill="{top}" stroke="{side}" stroke-width="1"/><path d="M{x+w},{y} l9,-7 0,{h} -9,7z" fill="{side}"/>')
def tokens(x,y,n=7,color=CYAN,w=20,h=42,gap=29):
    for i in range(n):cube(x+i*gap,y,w,h,color)
def image(name,x,y,w,h):
    b=base64.b64encode((OUT/(name+'.png')).read_bytes()).decode()
    p.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="data:image/png;base64,{b}" preserveAspectRatio="xMidYMid meet"/>')
def panel(key,x,y,w,h,title,color=BLUE,fill='#f3f6fc',fontsize=30):
    p.append(f'<g id="{key}">');box(x,y,w,h,fill,color);text(x+20,y+43,title,fontsize,color,True,w-40)
def end():p.append('</g>')
def clock(x,y):
    p.append(f'<circle cx="{x}" cy="{y}" r="30" fill="#fff" stroke="{BLUE}" stroke-width="3"/>')
    for dx,dy in [(0,-23),(23,0),(0,23),(-23,0)]:p.append(f'<circle cx="{x+dx}" cy="{y+dy}" r="2.7" fill="{BLUE}"/>')
    line([(x,y-17),(x,y),(x+13,y+7)],BLUE,3,False)
def robot():
    # Vector schematic, explicitly not an empirical success image.
    box(2460,980,235,22,'#dbe3ee','#b0bdce',3)
    p.append('<path d="M2468,1002 v84 M2686,1002 v84" stroke="#7d8b9f" stroke-width="12"/>')
    for pts in [[(2469,970),(2470,929),(2448,864),(2480,813),(2520,844),(2509,902),(2534,926)],[(2685,970),(2690,929),(2710,864),(2678,813),(2638,844),(2649,902),(2624,926)]]:
        line(pts,'#a6b4c7',23,False)
        for x,y in pts[:-1]:p.append(f'<circle cx="{x}" cy="{y}" r="14" fill="#edf1f6" stroke="#65758a" stroke-width="3"/>')
        x,y=pts[-1];line([(x-10,y+4),(x-9,y+22)],INK,5,False);line([(x+10,y+4),(x+9,y+22)],INK,5,False)
    p.append('<ellipse cx="2580" cy="982" rx="37" ry="9" fill="#cad0db"/><path d="M2543,956 q37,16 74,0 l-7,25 q-30,15 -60,0z" fill="#70b8c7" stroke="#388694" stroke-width="2"/><ellipse cx="2580" cy="956" rx="37" ry="10" fill="#bce2e8" stroke="#388694" stroke-width="2"/>')

def main():
    p.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    p.append('<defs>')
    for name,c in [('cyan',CYAN),('purple',PURPLE),('blue',BLUE),('amber',AMBER),('gray',MUTED),('green',GREEN)]:p.append(f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10z" fill="{c}"/></marker>')
    p.append('<linearGradient id="violetglass" x2="1" y2="1"><stop stop-color="#faf7ff"/><stop offset="1" stop-color="#ece1fc"/></linearGradient><linearGradient id="cyanglass" x2=".8" y2="1"><stop stop-color="#f3fcff"/><stop offset="1" stop-color="#dcf3fa"/></linearGradient><linearGradient id="orangeglass" x2="1" y2="1"><stop stop-color="#fffaf0"/><stop offset="1" stop-color="#fff0d5"/></linearGradient></defs>')
    p.append('<style>text{font-family:"Noto Sans CJK SC","WenQuanYi Micro Hei",sans-serif}</style><rect width="2800" height="1610" fill="white"/>')
    text(55,90,'Precision-WAM V4',74,INK,True)
    text(910,78,'因果 Chunk-VTT16 条件 · 全 32 步稠密视觉精修',37,INK,True,1800)
    text(910,128,'当前固定双臂任务  |  三相机输入  |  保持原 DiT 与 48 visual tokens',25,MUTED)
    box(55,121,690,44,'#fff4ec','#e9b27f',14);text(78,151,'改造与短跑通过；闭环任务提升仍待正式验证',23,'#a45e23',True,647)
    # NEW causal branch
    panel('chunk-vtt16',700,195,2040,265,'新增：Chunk-VTT-16 Predictor',PURPLE,'url(#violetglass)',34)
    text(1280,238,'预测、投影及 α 乘积：每次重规划仅算 1 次，10 次 Euler 共享',25,PURPLE,False,1400)
    box(720,260,438,175,'#fbf8ff','#cbb8e9')
    text(739,293,'当前 gₜ：三视角 DINO 均值（LN）→ 3072',23,INK,False,400)
    text(739,329,'因果历史 gₜ − gₜ₋₁₆（归一化）→ 3072',23,INK,False,400)
    text(739,365,'当前实测状态（原归一化）→ 16',23,INK,False,400)
    text(739,401,'Task-VTT（LN）→ 1024',23,INK,False,400)
    text(1180,283,'轻量 MLP：3 个 Linear',25,PURPLE,True,390)
    for xx,hh in [(1238,68),(1350,92),(1462,116)]:cube(xx,325-(hh-68)/2,38,hh,PURPLE)
    line([(1290,359),(1343,359)],PURPLE,3);line([(1402,359),(1455,359)],PURPLE,3)
    text(1180,425,'7184 → 512 → 512 → 3072',23,INK,False,396)
    line([(1158,353),(1210,353)],PURPLE)
    text(1630,282,'预测未来 16 步',26,PURPLE,True,190)
    text(1630,316,'视觉变化 Δĝ₁₆',25,INK,False,190)
    tokens(1640,347,6,CYAN,16,25,25);tokens(1640,382,6,CYAN,16,25,25)
    text(1630,437,'三视角 × 1024',22,INK,False,190)
    line([(1548,353),(1616,353)],PURPLE)
    box(1860,296,206,106,'#fff','#a789da');text(1878,330,'新增 Projection',23,PURPLE,True,175);text(1878,372,'3072 → 768',25,INK,False,175)
    line([(1824,353),(1860,353)],PURPLE)
    p.append(f'<circle cx="2150" cy="353" r="30" fill="white" stroke="{PURPLE}" stroke-width="3"/>');text(2150,364,'×α',27,PURPLE,True,anchor='middle')
    text(2098,294,'α 初始 0',24,PURPLE,True,140);text(2075,436,'非概率系数',19,MUTED,False,150)
    line([(2066,353),(2120,353)],PURPLE)
    box(2230,280,488,149,'#fff','#a789da');text(2249,320,'c_fused = c_task + α P(Δĝ₁₆)',28,PURPLE,True,450);tokens(2300,349,9,PURPLE,18,30,32);text(2249,418,'替换已有 Task 条件；不新增 token',23,INK,False,450)
    line([(2180,353),(2230,353)],PURPLE)
    end()
    # sensor physical anchor
    panel('current-sensors',55,500,270,664,'三路机载 RGB',CYAN,'url(#cyanglass)',29)
    for name,yy,label in [('head_camera',570,'头部相机'),('left_camera',765,'左腕相机'),('right_camera',960,'右腕相机')]:
        image(name,90,yy,200,150);text(190,yy+180,label,24,INK,True,anchor='middle')
    end();text(60,1198,'原始机载观测示例',22,MUTED);text(60,1230,'不是 V4 成功结果',22,MUTED)
    panel('frozen-dino',370,500,270,664,'冻结 DINOv3',CYAN,'url(#cyanglass)',30)
    text(390,580,'224 × 224 / 视角',25,INK,False,230)
    text(390,622,'取 −4 层均值 → gₜ',23,INK,False,230)
    for yy in [700,816,932]:
        # Dimension cues encode dense patch grid, not fake sensor imagery.
        for row in range(4):
            for col in range(6):cube(395+col*33+row*4,yy+row*12,26,12,CYAN)
    text(390,1069,'三层特征拼接 / patch',22,INK,False,230)
    text(390,1110,'融合：3072 → 1024',23,INK,True,230)
    text(390,1146,'移除 CLS / register',21,MUTED,False,230)
    end()
    line([(325,801),(370,801)],CYAN)
    line([(640,587),(675,587),(675,350),(720,350)],PURPLE)
    text(396,483,'仅当前与过去观测',22,PURPLE)
    # global pathway
    panel('perceiver',700,535,300,345,'Perceiver',CYAN,'url(#cyanglass)',33)
    text(720,615,'每视角 16 queries',26,INK,False,260)
    text(720,659,'共 48 visual tokens',25,INK,True,260)
    tokens(727,705,7,CYAN,21,33,32);tokens(727,755,7,CYAN,21,33,32)
    text(720,845,'48 × 768',28,INK,True,260)
    end();line([(640,721),(670,721),(670,708),(700,708)],CYAN)
    panel('dit-backbone',1050,535,540,345,'DiT 主干（层数 / 宽度保持不变）',BLUE,'#f1f4fb',29)
    box(1070,605,270,245,'#fff','#bcc8e5')
    text(1088,641,'现有条件',25,BLUE,True,230)
    text(1088,679,'实测关节历史',23,INK,False,230)
    text(1088,715,'因果视觉记忆',23,INK,False,230)
    text(1088,751,'Task token ← c_fused',23,INK,False,230)
    clock(1300,793);text(1088,793,'因果时钟：6 维',22,INK,False,175);text(1088,829,'闭合 / 静止持续时长',20,MUTED,False,230)
    tokens(1370,662,6,PURPLE,20,85,31)
    text(1358,785,'32 action tokens',23,INK,True,210);text(1358,827,'hidden：32 × 768',21,INK,False,210)
    end();line([(1000,708),(1050,708)],CYAN)
    line([(2460,460),(2460,493),(1310,493),(1310,605)],PURPLE)
    text(1450,522,'融合到 DiT 的 Task 条件',23,PURPLE)
    # dense refinement branch
    panel('dense-patches',700,955,355,208,'当前 588 dense patches',AMBER,'url(#orangeglass)',29)
    for row in range(2):tokens(726+row*6,1020+row*29,9,AMBER,18,18,29)
    text(720,1111,'3 × 196；XY(2) + 相机射线(3)',21,INK,False,315)
    text(720,1145,'K,V：1029 → 768',24,INK,True,315)
    end();line([(640,1090),(700,1090)],AMBER)
    panel('all32-refinement',1110,955,480,208,'全 32 步稠密视觉精修',AMBER,'url(#orangeglass)',31)
    text(1130,1021,'Q：DiT action hidden，32 × 768',25,INK,False,440)
    tokens(1140,1047,14,AMBER,18,38,30)
    text(1130,1146,'32 queries 对当前 588 patches 做 cross-attention',21,INK,False,440)
    end();line([(1055,1058),(1110,1058)],AMBER)
    line([(1480,880),(1480,955)],BLUE);text(1510,928,'Q',25,BLUE,True)
    text(1620,990,'Δv：32 × 16',25,AMBER,True,180);tokens(1625,1034,5,AMBER,22,57,32)
    line([(1590,1090),(1620,1090)],AMBER)
    # sum comes BEFORE Euler
    p.append(f'<circle cx="1788" cy="708" r="32" fill="white" stroke="{BLUE}" stroke-width="3"/>');text(1788,723,'+',41,BLUE,True,anchor='middle')
    line([(1590,708),(1756,708)],BLUE);text(1610,678,'Base flow v',24,BLUE,True)
    line([(1715,1070),(1788,1070),(1788,740)],AMBER)
    text(1635,924,'精修所有 32 步',22,AMBER)
    panel('euler-integration',1855,535,255,345,'10 次 Euler 积分',BLUE,'#f1f4fb',27)
    text(1874,613,'每轮：v + Δv',25,BLUE,True,217)
    for i in range(4):p.append(f'<circle cx="{1882+i*51}" cy="691" r="14" fill="{["#d7e0fd","#b4c7fb","#809fee","#5e6fd0"][i]}" stroke="{BLUE}"/>')
    line([(1882,720),(2038,720)],BLUE,3)
    text(1874,775,'更新当前带噪动作',22,INK,False,217)
    text(1874,818,'DiT + 精修每轮重算',20,MUTED,False,217)
    text(1874,852,'Chunk 条件保持相同',20,PURPLE,True,217)
    end();line([(1820,708),(1855,708)],BLUE)
    line([(1985,880),(1985,909),(1555,909),(1555,880)],BLUE,2);text(1830,946,'下一轮 noisy actions + t',19,BLUE)
    # execution strip with actual 32 tokens 8 green and24gray represented compactly
    panel('action-chunk',2160,535,240,345,'32 步动作块',GREEN,'#f2fbf4',30)
    text(2180,616,'32 × 16',27,INK,True,200)
    for i in range(32):cube(2180+i*6,672,4,55,GREEN if i<EXEC_STEPS else '#a6b4c7')
    text(2180,769,f'前 {EXEC_STEPS} 步执行',25,GREEN,True,200)
    text(2180,811,f'后 {32-EXEC_STEPS} 步丢弃',23,MUTED,False,200)
    text(2180,849,f'{EXEC_STEPS/10:.1f} 秒后重规划',23,INK,False,200)
    end();line([(2110,708),(2160,708)],BLUE)
    panel('robot-control',2430,535,310,628,'双臂 Panda 执行',BLUE,'#f7f9fc',28)
    text(2450,616,'原 B 样条：只平滑手臂',23,INK,False,270)
    text(2450,655,'夹爪事件保持原值',23,INK,False,270)
    text(2450,694,'Panda 官方提前制动',23,INK,False,270)
    robot();text(2450,1138,'机器人图为结构示意',21,MUTED,False,270)
    end();line([(2400,751),(2430,751)],GREEN)
    # Feedback loop
    line([(2585,1163),(2585,1212),(190,1212),(190,1164)],BLUE,4)
    box(1080,1182,730,56,'#fff','#abbddd',18);text(1445,1220,f'执行 {EXEC_STEPS} 步 → 新观测 → 重规划 → 更新因果历史',25,BLUE,True,680,anchor='middle')
    # train-only signals, isolated from online forward paths
    panel('training-only-objectives',55,1284,2685,250,'训练专用监督：未来真实图像只生成标签，不进入在线策略',MUTED,'#f7f9fc',30)
    cards=[(75,425,'Flow Matching',BLUE),(535,645,'Future 8 / 32',CYAN),(1215,610,'IDM 4 步',BLUE),(1860,860,'Chunk-VTT16',PURPLE)]
    for x,w,title,c in cards:
        box(x,1353,w,160,'#fff','#cbd6e2');text(x+20,1395,title,29,c,True,w-40)
    text(95,1437,'L_flow：动作流场 MSE',23,INK,False,385);text(95,1477,'专家动作 + 随机噪声',23,MUTED,False,385)
    tokens(1030,1377,4,CYAN,17,26,28);text(555,1437,'未来 8 / 32 步绝对 DINO patch 特征',23,INK,False,605);text(555,1477,'0.25 L_future8 + 0.25 L_future32（Cosine）',22,MUTED,False,605)
    tokens(1678,1377,4,AMBER,17,26,28);text(1235,1437,'从示教视觉变化预测 4 步动作残差',23,INK,False,570);text(1235,1477,'0.1 L_IDM（Smooth L1）',23,MUTED,False,570)
    text(1880,1437,'标签：冻结 DINO 的 g(Iₜ₊₁₆) − g(Iₜ)，停止梯度',23,INK,False,820);text(1880,1477,'0.1 × [Huber + 0.1 × Cosine]；预测的是变化',23,PURPLE,False,820)
    end()
    text(55,1573,'新增可训练参数：7,877,377  |  不增加 DiT 层数、宽度或 token 数量  |  图中未包含未启用的高层规划 / 恢复 / 触觉',24,MUTED)
    p.append('</svg>')
    svg=OUT/f'{PREFIX}.svg';svg.write_text('\n'.join(p),'utf8');tree=ET.parse(svg)
    handle=Rsvg.Handle.new_from_file(str(svg));viewport=Rsvg.Rectangle();viewport.x=0;viewport.y=0;viewport.width=W;viewport.height=H
    surface=cairo.ImageSurface(cairo.FORMAT_ARGB32,W,H);handle.render_document(cairo.Context(surface),viewport);surface.write_to_png(str(OUT/f'{PREFIX}.png'))
    pdf=cairo.PDFSurface(str(OUT/f'{PREFIX}.pdf'),W/2,H/2);c=cairo.Context(pdf);c.scale(.5,.5);handle.render_document(c,viewport);pdf.finish()
    val=dict(xml_valid=True,editable_structure=True,intentional_raster_images=len(tree.findall('.//{http://www.w3.org/2000/svg}image')),image_source='outputs/zhuhai2026_tasks12/wam_precision_v2_task2_dagger_80k_fixed5_20261011_v23/zhuhai_task2/seed_932101/onboard/observation_000000.npz',images_are_original_archived_arrays=True,text_checks=checks,dimensions=[W,H],chunk_input_dim=7184,chunk_output_dim=3072,new_parameter_count=7877377,perceiver_shape=[48,768],refinement_output_shape=[32,16],chunk_calculations_per_replan=1,euler_iterations=10,online_future_input_edges=0,alpha_is_unbounded_learnable_scalar=True,full32_refined=True,execute_steps=EXEC_STEPS,discard_steps=32-EXEC_STEPS,svg_png_pdf_rendered=True,visual_inspection_pending=True,execution16_protocol_pending=EXEC_STEPS==16)
    (OUT/f'{PREFIX}_validation.json').write_text(json.dumps(val,ensure_ascii=False,indent=2)+'\n')
    print(str(svg))
if __name__=='__main__': main()
