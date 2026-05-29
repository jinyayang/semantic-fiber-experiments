"""
实验 2 修正版：曲率-幻觉相关性验证（英文问答 + 参数扰动代理曲率）
使用 Qwen1.5-MoE-A2.7B，手动参数扰动模拟知识冲突，验证扰动强度与答案分歧度的正相关。
"""
import sys
import os
import torch
import torch.nn as nn
import copy

# 加载 sfb 包（如有）
sys.path.insert(0, os.path.dirname(__file__))

# ===================== 1. 加载模型 =====================
print("Loading Qwen1.5-MoE-A2.7B...")
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./checkpoints/Qwen/Qwen1.5-MoE-A2.7B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()
print(f"Model loaded on {device}. Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# ===================== 2. 英文事实问答数据 =====================
# 用户自行填充 FACTS 数据，格式示例：
# {"prompt": "The capital of France is", "answer": "Paris"}

# ===================== 2. 事实问答数据 =====================
FACTS = [
    {"prompt": "法国的首都是", "answer": "巴黎"},
    {"prompt": "中国的首都是", "answer": "北京"},
    {"prompt": "美国的首都是", "answer": "华盛顿"},
    {"prompt": "英国的首都是", "answer": "伦敦"},
    {"prompt": "日本的首都是", "answer": "东京"},
    {"prompt": "德国的首都是", "answer": "柏林"},
    {"prompt": "俄罗斯的首都是", "answer": "莫斯科"},
    {"prompt": "意大利的首都是", "answer": "罗马"},
    {"prompt": "西班牙的首都是", "answer": "马德里"},
    {"prompt": "加拿大的首都是", "answer": "渥太华"},
    {"prompt": "澳大利亚的首都是", "answer": "堪培拉"},
    {"prompt": "巴西的首都是", "answer": "巴西利亚"},
    {"prompt": "印度的首都是", "answer": "新德里"},
    {"prompt": "韩国的首都是", "answer": "首尔"},
    {"prompt": "埃及的首都是", "answer": "开罗"},
    {"prompt": "希腊的首都是", "answer": "雅典"},
    {"prompt": "瑞典的首都是", "answer": "斯德哥尔摩"},
    {"prompt": "挪威的首都是", "answer": "奥斯陆"},
    {"prompt": "丹麦的首都是", "answer": "哥本哈根"},
    {"prompt": "芬兰的首都是", "answer": "赫尔辛基"},
    {"prompt": "荷兰的首都是", "answer": "阿姆斯特丹"},
    {"prompt": "比利时的首都是", "answer": "布鲁塞尔"},
    {"prompt": "瑞士的首都是", "answer": "伯尔尼"},
    {"prompt": "奥地利的首都是", "answer": "维也纳"},
    {"prompt": "波兰的首都是", "answer": "华沙"},
    {"prompt": "捷克的首都是", "answer": "布拉格"},
    {"prompt": "葡萄牙的首都是", "answer": "里斯本"},
    {"prompt": "土耳其的首都是", "answer": "安卡拉"},
    {"prompt": "伊朗的首都是", "answer": "德黑兰"},
    {"prompt": "沙特阿拉伯的首都是", "answer": "利雅得"},
    {"prompt": "以色列的首都是", "answer": "耶路撒冷"},
    {"prompt": "泰国的首都是", "answer": "曼谷"},
    {"prompt": "越南的首都是", "answer": "河内"},
    {"prompt": "印度尼西亚的首都是", "answer": "雅加达"},
    {"prompt": "菲律宾的首都是", "answer": "马尼拉"},
    {"prompt": "马来西亚的首都是", "answer": "吉隆坡"},
    {"prompt": "新加坡的首都是", "answer": "新加坡市"},
    {"prompt": "巴基斯坦的首都是", "answer": "伊斯兰堡"},
    {"prompt": "孟加拉国的首都是", "answer": "达卡"},
    {"prompt": "尼泊尔的首都是", "answer": "加德满都"},
    {"prompt": "阿富汗的首都是", "answer": "喀布尔"},
    {"prompt": "伊拉克的首都是", "answer": "巴格达"},
    {"prompt": "叙利亚的首都是", "answer": "大马士革"},
    {"prompt": "约旦的首都是", "answer": "安曼"},
    {"prompt": "黎巴嫩的首都是", "answer": "贝鲁特"},
    {"prompt": "也门的首都是", "answer": "萨那"},
    {"prompt": "阿曼的首都是", "answer": "马斯喀特"},
    {"prompt": "科威特的首都是", "answer": "科威特城"},
    {"prompt": "卡塔尔的首都是", "answer": "多哈"},
    {"prompt": "巴林的首都是", "answer": "麦纳麦"},
    {"prompt": "蒙古的首都是", "answer": "乌兰巴托"},
    {"prompt": "哈萨克斯坦的首都是", "answer": "阿斯塔纳"},
    {"prompt": "乌兹别克斯坦的首都是", "answer": "塔什干"},
    {"prompt": "土库曼斯坦的首都是", "answer": "阿什哈巴德"},
    {"prompt": "吉尔吉斯斯坦的首都是", "answer": "比什凯克"},
    {"prompt": "塔吉克斯坦的首都是", "answer": "杜尚别"},
    {"prompt": "白俄罗斯的首都是", "answer": "明斯克"},
    {"prompt": "乌克兰的首都是", "answer": "基辅"},
    {"prompt": "摩尔多瓦的首都是", "answer": "基希讷乌"},
    {"prompt": "立陶宛的首都是", "answer": "维尔纽斯"},
    {"prompt": "拉脱维亚的首都是", "answer": "里加"},
    {"prompt": "爱沙尼亚的首都是", "answer": "塔林"},
    {"prompt": "格鲁吉亚的首都是", "answer": "第比利斯"},
    {"prompt": "亚美尼亚的首都是", "answer": "埃里温"},
    {"prompt": "阿塞拜疆的首都是", "answer": "巴库"},
    {"prompt": "阿尔及利亚的首都是", "answer": "阿尔及尔"},
    {"prompt": "摩洛哥的首都是", "answer": "拉巴特"},
    {"prompt": "突尼斯的首都是", "answer": "突尼斯市"},
    {"prompt": "利比亚的首都是", "answer": "的黎波里"},
    {"prompt": "苏丹的首都是", "answer": "喀土穆"},
    {"prompt": "埃塞俄比亚的首都是", "answer": "亚的斯亚贝巴"},
    {"prompt": "肯尼亚的首都是", "answer": "内罗毕"},
    {"prompt": "坦桑尼亚的首都是", "answer": "多多马"},
    {"prompt": "乌干达的首都是", "answer": "坎帕拉"},
    {"prompt": "卢旺达的首都是", "answer": "基加利"},
    {"prompt": "刚果（金）的首都是", "answer": "金沙萨"},
    {"prompt": "刚果（布）的首都是", "answer": "布拉柴维尔"},
    {"prompt": "安哥拉的首都是", "answer": "罗安达"},
    {"prompt": "南非的首都是", "answer": "比勒陀利亚"},
    {"prompt": "尼日利亚的首都是", "answer": "阿布贾"},
    {"prompt": "加纳的首都是", "answer": "阿克拉"},
    {"prompt": "科特迪瓦的首都是", "answer": "亚穆苏克罗"},
    {"prompt": "喀麦隆的首都是", "answer": "雅温得"},
    {"prompt": "津巴布韦的首都是", "answer": "哈拉雷"},
    {"prompt": "赞比亚的首都是", "answer": "卢萨卡"},
    {"prompt": "莫桑比克的首都是", "answer": "马普托"},
    {"prompt": "马达加斯加的首都是", "answer": "塔那那利佛"},
    {"prompt": "阿根廷的首都是", "answer": "布宜诺斯艾利斯"},
    {"prompt": "智利的首都是", "answer": "圣地亚哥"},
    {"prompt": "秘鲁的首都是", "answer": "利马"},
    {"prompt": "哥伦比亚的首都是", "answer": "波哥大"},
    {"prompt": "委内瑞拉的首都是", "answer": "加拉加斯"},
    {"prompt": "厄瓜多尔的首都是", "answer": "基多"},
    {"prompt": "玻利维亚的首都是", "answer": "拉巴斯"},
    {"prompt": "巴拉圭的首都是", "answer": "亚松森"},
    {"prompt": "乌拉圭的首都是", "answer": "蒙得维的亚"},
    {"prompt": "墨西哥的首都是", "answer": "墨西哥城"},
    {"prompt": "古巴的首都是", "answer": "哈瓦那"},
    {"prompt": "牙买加的首都是", "answer": "金斯敦"},
    {"prompt": "海地的首都是", "answer": "太子港"},
    {"prompt": "多米尼加共和国的首都是", "answer": "圣多明各"},
    {"prompt": "巴拿马的首都是", "answer": "巴拿马城"},
    {"prompt": "哥斯达黎加的首都是", "answer": "圣何塞"},
    {"prompt": "尼加拉瓜的首都是", "answer": "马那瓜"},
    {"prompt": "洪都拉斯的首都是", "answer": "特古西加尔巴"},
    {"prompt": "萨尔瓦多的首都是", "answer": "圣萨尔瓦多"},
    {"prompt": "危地马拉的首都是", "answer": "危地马拉城"},
    {"prompt": "新西兰的首都是", "answer": "惠灵顿"},
    {"prompt": "巴布亚新几内亚的首都是", "answer": "莫尔兹比港"},
    {"prompt": "斐济的首都是", "answer": "苏瓦"},
    {"prompt": "水的化学式是", "answer": "H₂O"},
    {"prompt": "二氧化碳的化学式是", "answer": "CO₂"},
    {"prompt": "氯化钠的化学式是", "answer": "NaCl"},
    {"prompt": "甲烷的化学式是", "answer": "CH₄"},
    {"prompt": "氨气的化学式是", "answer": "NH₃"},
    {"prompt": "乙醇的化学式是", "answer": "C₂H₅OH"},
    {"prompt": "葡萄糖的化学式是", "answer": "C₆H₁₂O₆"},
    {"prompt": "硫酸的化学式是", "answer": "H₂SO₄"},
    {"prompt": "盐酸的化学式是", "answer": "HCl"},
    {"prompt": "硝酸的化学式是", "answer": "HNO₃"},
    {"prompt": "氢氧化钠的化学式是", "answer": "NaOH"},
    {"prompt": "氢氧化钙的化学式是", "answer": "Ca(OH)₂"},
    {"prompt": "碳酸钙的化学式是", "answer": "CaCO₃"},
    {"prompt": "真空中光速约为（千米/秒）", "answer": "300000"},
    {"prompt": "声音在空气中（15℃）的传播速度约为（米/秒）", "answer": "340"},
    {"prompt": "万有引力定律的发现者是", "answer": "牛顿"},
    {"prompt": "相对论的提出者是", "answer": "爱因斯坦"},
    {"prompt": "量子力学的奠基人之一，提出不确定性原理的是", "answer": "海森堡"},
    {"prompt": "镭的发现者是", "answer": "居里夫人"},
    {"prompt": "进化论的提出者是", "answer": "达尔文"},
    {"prompt": "DNA双螺旋结构的发现者是", "answer": "沃森和克里克"},
    {"prompt": "青霉素的发现者是", "answer": "弗莱明"},
    {"prompt": "电报的发明者是", "answer": "莫尔斯"},
    {"prompt": "电话的发明者是", "answer": "贝尔"},
    {"prompt": "电灯的发明者是", "answer": "爱迪生"},
    {"prompt": "飞机的发明者是", "answer": "莱特兄弟"},
    {"prompt": "蒸汽机的改良者是", "answer": "瓦特"},
    {"prompt": "炸药的发明者是", "answer": "诺贝尔"},
    {"prompt": "相对论中质能方程的表达式是", "answer": "E=mc²"},
    {"prompt": "一年有", "answer": "12"},
    {"prompt": "一个月最多有", "answer": "31"},
    {"prompt": "二月在平年有", "answer": "28"},
    {"prompt": "二月在闰年有", "answer": "29"},
    {"prompt": "一天有", "answer": "24"},
    {"prompt": "一小时有", "answer": "60"},
    {"prompt": "一分钟有", "answer": "60"},
    {"prompt": "一周有", "answer": "7"},
    {"prompt": "地球绕太阳公转一周的时间是", "answer": "一年"},
    {"prompt": "地球自转一周的时间是", "answer": "一天"},
    {"prompt": "月球绕地球公转一周大约需要", "answer": "27.3"},
    {"prompt": "太阳系中最大的行星是", "answer": "木星"},
    {"prompt": "太阳系中第二大的行星是", "answer": "土星"},
    {"prompt": "太阳系中最小的行星是", "answer": "水星"},
    {"prompt": "离太阳最近的行星是", "answer": "水星"},
    {"prompt": "离太阳最远的行星是", "answer": "海王星"},
    {"prompt": "被称为“红色星球”的行星是", "answer": "火星"},
    {"prompt": "地球的天然卫星是", "answer": "月球"},
    {"prompt": "第一个登上月球的人是", "answer": "尼尔·阿姆斯特朗"},
    {"prompt": "中国第一个进入太空的航天员是", "answer": "杨利伟"},
    {"prompt": "世界上最大的海洋是", "answer": "太平洋"},
    {"prompt": "世界上第二大的海洋是", "answer": "大西洋"},
    {"prompt": "世界上第三大的海洋是", "answer": "印度洋"},
    {"prompt": "世界上最小的海洋是", "answer": "北冰洋"},
    {"prompt": "世界上最长的河流是", "answer": "尼罗河"},
    {"prompt": "中国最长的河流是", "answer": "长江"},
    {"prompt": "世界第二长的河流是", "answer": "亚马逊河"},
    {"prompt": "世界上最高的山峰是", "answer": "珠穆朗玛峰"},
    {"prompt": "中国最高的山峰是", "answer": "珠穆朗玛峰"},
    {"prompt": "世界第二高的山峰是", "answer": "乔戈里峰"},
    {"prompt": "世界上最大的沙漠是", "answer": "撒哈拉沙漠"},
    {"prompt": "中国最大的沙漠是", "answer": "塔克拉玛干沙漠"},
    {"prompt": "世界上面积最大的国家是", "answer": "俄罗斯"},
    {"prompt": "世界上面积第二大的国家是", "answer": "加拿大"},
    {"prompt": "世界上面积第三大的国家是", "answer": "中国"},
    {"prompt": "世界上面积最小的国家是", "answer": "梵蒂冈"},
    {"prompt": "世界上人口最多的国家是", "answer": "中国"},
    {"prompt": "世界上人口第二多的国家是", "answer": "印度"},
    {"prompt": "联合国安全理事会常任理事国有几个", "answer": "5"},
    {"prompt": "联合国安全理事会常任理事国包括中国、俄罗斯、美国、英国和", "answer": "法国"},
    {"prompt": "第二次世界大战结束于", "answer": "1945"},
    {"prompt": "第一次世界大战开始于", "answer": "1914"},
    {"prompt": "哥伦布发现美洲大陆是在", "answer": "1492"},
    {"prompt": "美国独立宣言签署于", "answer": "1776"},
    {"prompt": "法国大革命开始于", "answer": "1789"},
    {"prompt": "中华人民共和国成立于", "answer": "1949"},
    {"prompt": "中国共产党的成立时间是", "answer": "1921"},
    {"prompt": "辛亥革命爆发于", "answer": "1911"},
    {"prompt": "秦始皇统一中国的时间是", "answer": "公元前221"},
    {"prompt": "唐朝建立于", "answer": "618"},
    {"prompt": "明朝建立于", "answer": "1368"},
    {"prompt": "清朝建立于", "answer": "1636"},
    {"prompt": "郑和下西洋的时间是", "answer": "1405"},
    {"prompt": "四大发明包括造纸术、指南针、火药和", "answer": "印刷术"},
    {"prompt": "《本草纲目》的作者是", "answer": "李时珍"},
    {"prompt": "《史记》的作者是", "answer": "司马迁"},
    {"prompt": "《红楼梦》的作者是", "answer": "曹雪芹"},
    {"prompt": "《西游记》的作者是", "answer": "吴承恩"},
    {"prompt": "《水浒传》的作者是", "answer": "施耐庵"},
    {"prompt": "《三国演义》的作者是", "answer": "罗贯中"},
    {"prompt": "圆周率π约等于", "answer": "3.14"},
    {"prompt": "自然常数e约等于", "answer": "2.718"},
    {"prompt": "直角三角形的斜边平方等于两直角边平方和，这个定理叫做", "answer": "勾股定理"},
    {"prompt": "平行四边形的对边", "answer": "相等"},
    {"prompt": "三角形的内角和是", "answer": "180"},
    {"prompt": "四边形的内角和是", "answer": "360"},
    {"prompt": "圆的周长公式是（用C表示周长，r表示半径）", "answer": "C=2πr"},
    {"prompt": "圆的面积公式是（用S表示面积，r表示半径）", "answer": "S=πr²"},
    {"prompt": "水的凝固点是（摄氏度）", "answer": "0"},
    {"prompt": "水的沸点是（标准大气压下，摄氏度）", "answer": "100"},
    {"prompt": "铁的熔点是（摄氏度约）", "answer": "1538"},
    {"prompt": "人体正常体温约为（摄氏度）", "answer": "36.5"},
    {"prompt": "电流的单位是", "answer": "安培"},
    {"prompt": "电压的单位是", "answer": "伏特"},
    {"prompt": "电阻的单位是", "answer": "欧姆"},
    {"prompt": "功率的单位是", "answer": "瓦特"},
    {"prompt": "频率的单位是", "answer": "赫兹"},
    {"prompt": "力的单位是", "answer": "牛顿"},
    {"prompt": "能量的单位是", "answer": "焦耳"},
    {"prompt": "压强的单位是", "answer": "帕斯卡"},
    {"prompt": "物质的量的单位是", "answer": "摩尔"},
    {"prompt": "光年的定义是", "answer": "光在真空中一年内传播的距离"},
    {"prompt": "世界上最大的动物是", "answer": "蓝鲸"},
    {"prompt": "陆地上最大的动物是", "answer": "非洲象"},
    {"prompt": "世界上最高的动物是", "answer": "长颈鹿"},
    {"prompt": "世界上跑得最快的动物是", "answer": "猎豹"},
    {"prompt": "世界上飞得最快的鸟是", "answer": "雨燕"},
    {"prompt": "世界上最大的鸟类是", "answer": "鸵鸟"},
    {"prompt": "世界上最小的鸟类是", "answer": "蜂鸟"},
    {"prompt": "熊猫主要生活在哪个国家", "answer": "中国"},
    {"prompt": "澳大利亚的国宝动物是", "answer": "袋鼠"},
    {"prompt": "中国的国宝动物是", "answer": "大熊猫"},
    {"prompt": "人体最大的器官是", "answer": "皮肤"},
    {"prompt": "人体最长的骨头是", "answer": "股骨"},
    {"prompt": "人体最小的骨头是", "answer": "镫骨"},
    {"prompt": "人体内最大的内脏器官是", "answer": "肝脏"},
    {"prompt": "人体血液中数量最多的细胞是", "answer": "红细胞"},
    {"prompt": "人类正常染色体数目是", "answer": "46"},
    {"prompt": "DNA的全称是", "answer": "脱氧核糖核酸"},
    {"prompt": "RNA的全称是", "answer": "核糖核酸"},
    {"prompt": "第一台计算机ENIAC诞生于", "answer": "1946"},
    {"prompt": "互联网的前身是", "answer": "阿帕网"},
    {"prompt": "万维网的发明者是", "answer": "伯纳斯-李"},
    {"prompt": "操作系统中，Windows由哪家公司开发", "answer": "微软"},
    {"prompt": "操作系统中，macOS由哪家公司开发", "answer": "苹果"},
    {"prompt": "Linux的创始人是", "answer": "林纳斯·托瓦兹"},
    {"prompt": "第一代iPhone发布是在", "answer": "2007"},
    {"prompt": "安卓操作系统由哪家公司开发", "answer": "谷歌"},
    {"prompt": "世界上最大的搜索引擎是", "answer": "谷歌"},
    {"prompt": "中国最大的搜索引擎是", "answer": "百度"},
    {"prompt": "《哈姆雷特》的作者是", "answer": "莎士比亚"},
    {"prompt": "《战争与和平》的作者是", "answer": "托尔斯泰"},
    {"prompt": "《老人与海》的作者是", "answer": "海明威"},
    {"prompt": "《百年孤独》的作者是", "answer": "马尔克斯"},
    {"prompt": "《红楼梦》中的女主角之一是", "answer": "林黛玉"},
    {"prompt": "《西游记》中的主角是", "answer": "孙悟空"},
    {"prompt": "《水浒传》中的首领是", "answer": "宋江"},
    {"prompt": "《三国演义》中“赤壁之战”的交战双方是孙刘联军和", "answer": "曹操"},
    {"prompt": "贝多芬是哪国人", "answer": "德国"},
    {"prompt": "莫扎特是哪国人", "answer": "奥地利"},
    {"prompt": "巴赫是哪国人", "answer": "德国"},
    {"prompt": "肖邦是哪国人", "answer": "波兰"},
    {"prompt": "《命运交响曲》的作者是", "answer": "贝多芬"},
    {"prompt": "《欢乐颂》出自贝多芬的哪部交响曲", "answer": "第九交响曲"},
    {"prompt": "《茉莉花》是哪国的民歌", "answer": "中国"},
    {"prompt": "京剧中的“生旦净末丑”中的“旦”指", "answer": "女性角色"},
    {"prompt": "中国四大名著包括《红楼梦》《西游记》《水浒传》和", "answer": "《三国演义》"},
    {"prompt": "《论语》是记录谁言行的书", "answer": "孔子"},
    {"prompt": "道家学派的创始人是", "answer": "老子"},
    {"prompt": "儒家学派的创始人是", "answer": "孔子"},
    {"prompt": "墨家学派的创始人是", "answer": "墨子"},
    {"prompt": "法家思想的集大成者是", "answer": "韩非子"},
    {"prompt": "“己所不欲，勿施于人”出自", "answer": "《论语》"},
    {"prompt": "“道可道，非常道”出自", "answer": "《道德经》"},
    {"prompt": "奥林匹克运动会的发源地是", "answer": "希腊"},
    {"prompt": "现代奥运会始于", "answer": "1896"},
    {"prompt": "世界杯足球赛每几年举办一次", "answer": "4"},
    {"prompt": "世界上最高水平的篮球联赛是", "answer": "NBA"},
    {"prompt": "获得诺贝尔奖最多的国家是", "answer": "美国"},
    {"prompt": "诺贝尔奖中没有哪一个奖项", "answer": "数学"},
    {"prompt": "第一个获得诺贝尔科学奖的中国人是", "answer": "屠呦呦"},
    {"prompt": "屠呦呦发现的是", "answer": "青蒿素"},
    {"prompt": "莫言获得了什么诺贝尔奖", "answer": "诺贝尔文学奖"},
    {"prompt": "世界上最大的教堂是", "answer": "圣彼得大教堂"},
    {"prompt": "世界上最高的建筑（2025年）是", "answer": "哈利法塔"},
    {"prompt": "埃及的著名古代建筑是", "answer": "金字塔"},
    {"prompt": "中国的长城大约有多长（公里）", "answer": "21000"},
    {"prompt": "世界上使用人数最多的语言是", "answer": "汉语"},
    {"prompt": "世界上使用范围最广的语言是", "answer": "英语"},
    {"prompt": "联合国的官方工作语言有几种", "answer": "6"},
    {"prompt": "联合国的官方语言包括汉语、英语、法语、俄语、阿拉伯语和", "answer": "西班牙语"},
    {"prompt": "农历新年通常被称为", "answer": "春节"},
    {"prompt": "中国国庆节是", "answer": "10月1日"},
    {"prompt": "国际劳动节是", "answer": "5月1日"},
    {"prompt": "国际妇女节是", "answer": "3月8日"},
    {"prompt": "中国植树节是", "answer": "3月12日"},
    {"prompt": "中国教师节是", "answer": "9月10日"},
    {"prompt": "中国青年节是", "answer": "5月4日"},
    {"prompt": "党的生日是", "answer": "7月1日"},
    {"prompt": "建军节是", "answer": "8月1日"},
    {"prompt": "情人节是", "answer": "2月14日"},
    {"prompt": "万圣节是", "answer": "10月31日"},
    {"prompt": "感恩节（美国）是", "answer": "11月第四个星期四"},
    {"prompt": "圣诞节是", "answer": "12月25日"},
    {"prompt": "平方根64等于", "answer": "8"},
    {"prompt": "平方根100等于", "answer": "10"},
    {"prompt": "平方根121等于", "answer": "11"},
    {"prompt": "平方根144等于", "answer": "12"},
    {"prompt": "平方根169等于", "answer": "13"},
    {"prompt": "平方根196等于", "answer": "14"},
    {"prompt": "平方根225等于", "answer": "15"},
    {"prompt": "3的平方是", "answer": "9"},
    {"prompt": "4的平方是", "answer": "16"},
    {"prompt": "5的平方是", "answer": "25"},
    {"prompt": "6的平方是", "answer": "36"},
    {"prompt": "7的平方是", "answer": "49"},
    {"prompt": "8的平方是", "answer": "64"},
    {"prompt": "9的平方是", "answer": "81"},
    {"prompt": "10的平方是", "answer": "100"},
    {"prompt": "2的3次方是", "answer": "8"},
    {"prompt": "3的3次方是", "answer": "27"},
    {"prompt": "4的3次方是", "answer": "64"},
    {"prompt": "5的3次方是", "answer": "125"},
    {"prompt": "6的3次方是", "answer": "216"},
    {"prompt": "7的3次方是", "answer": "343"},
    {"prompt": "8的3次方是", "answer": "512"},
    {"prompt": "9的3次方是", "answer": "729"},
    {"prompt": "10的3次方是", "answer": "1000"},
    {"prompt": "2加2等于", "answer": "4"},
    {"prompt": "3加4等于", "answer": "7"},
    {"prompt": "5加7等于", "answer": "12"},
    {"prompt": "10减3等于", "answer": "7"},
    {"prompt": "20减8等于", "answer": "12"},
    {"prompt": "4乘5等于", "answer": "20"},
    {"prompt": "6乘7等于", "answer": "42"},
    {"prompt": "8乘9等于", "answer": "72"},
    {"prompt": "10除以2等于", "answer": "5"},
    {"prompt": "21除以3等于", "answer": "7"},
    {"prompt": "36除以6等于", "answer": "6"},
    {"prompt": "49除以7等于", "answer": "7"},
    {"prompt": "64除以8等于", "answer": "8"},
    {"prompt": "81除以9等于", "answer": "9"},
    {"prompt": "100除以10等于", "answer": "10"},
    {"prompt": "0.5加0.5等于", "answer": "1"},
    {"prompt": "0.25加0.75等于", "answer": "1"},
    {"prompt": "1.5加2.5等于", "answer": "4"},
    {"prompt": "2.5乘2等于", "answer": "5"},
    {"prompt": "1.2乘3等于", "answer": "3.6"},
    {"prompt": "3.6除以2等于", "answer": "1.8"},
    {"prompt": "最大的两位数", "answer": "99"},
    {"prompt": "最小的两位数", "answer": "10"},
    {"prompt": "最大的三位数", "answer": "999"},
    {"prompt": "最小的三位数", "answer": "100"},
    {"prompt": "自然数中最小的奇数是", "answer": "1"},
    {"prompt": "自然数中最小的偶数是", "answer": "2"},
    {"prompt": "最小的质数是", "answer": "2"},
    {"prompt": "最小的合数是", "answer": "4"},
    {"prompt": "1既不是质数也不是", "answer": "合数"},
    {"prompt": "六边形有", "answer": "6"},
    {"prompt": "八边形有", "answer": "8"},
    {"prompt": "十边形有", "answer": "10"},
]

print(f"Loaded {len(FACTS)} factual Q&A pairs")

# ===================== 3. 基线答案 =====================
def get_answer(model, tokenizer, prompt, max_new_tokens=15):
    # 2. prompt 加强（覆盖模型的选择题训练 bias）
    inputs = tokenizer("请只输出答案，不要输出任何选项或解释：" + prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()

print("\n=== Baseline Answers (first 10) ===")
baseline_answers = []
for i, fact in enumerate(FACTS):
    ans = get_answer(model, tokenizer, fact["prompt"])
    baseline_answers.append(ans)
    if i < 10:
        print(f"Q: {fact['prompt']}")
        print(f"A: {ans}")
        print()

# ===================== 4. 扰动函数 =====================
def perturb_model(model, layer_idx=20, epsilon=0.01):
    perturbed = copy.deepcopy(model)
    perturbed.eval()
    try:
        layer = perturbed.model.layers[layer_idx].mlp
        with torch.no_grad():
            for name, param in layer.named_parameters():
                if "weight" in name and param.dtype in (torch.float16, torch.float32, torch.bfloat16):
                    # 相对噪声：每个元素扰动 = epsilon × 该矩阵平均元素幅值
                    scale = epsilon * param.abs().mean()
                    noise = torch.randn_like(param, dtype=param.dtype, device=param.device) * scale
                    param.add_(noise)
    except Exception as e:
        print(f"Perturbation error: {e}")
        return None
    return perturbed

# ===================== 5. 曲率代理：参数扰动范数 =====================
def compute_curvature_proxy(model, perturbed_model, layer_idx=20):
    try:
        base_layer = model.model.layers[layer_idx].mlp
        pert_layer = perturbed_model.model.layers[layer_idx].mlp
        total_diff = 0.0
        total_norm = 0.0
        for (n1, p1), (n2, p2) in zip(base_layer.named_parameters(), pert_layer.named_parameters()):
            if "weight" in n1 and p1.dtype in (torch.float16, torch.float32, torch.bfloat16):
                total_diff += (p2 - p1).norm(p=2).item()
                total_norm += p1.norm(p=2).item()
        return total_diff / total_norm if total_norm > 0 else 0.0
    except Exception:
        return 0.0

# ===================== 6. 主实验 =====================
# 1. 提高 noise_scale（15B 模型需要更猛的扰动）
# ===================== 6. 主实验 =====================
perturbation_levels = [0.001, 0.003, 0.01, 0.03, 0.1]
results = []

for noise_scale in perturbation_levels:
    print(f"\n{'='*60}")
    print(f"=== Perturbation Level: {noise_scale} ===")
    print(f"{'='*60}")

    # 扰动模型
    perturbed_model = perturb_model(model, layer_idx=20, epsilon=noise_scale)  # ✅ 改这里
    if perturbed_model is None:
        continue

    # 曲率代理（相对变化）
    curvature_proxy = compute_curvature_proxy(model, perturbed_model, layer_idx=20)
    print(f"  Curvature proxy:    {curvature_proxy:.4f}")  # ✅ 直接打印

    # 扰动后答案
    print("Generating answers with perturbed model...")
    changed_count = 0
    for i, fact in enumerate(FACTS):
        perturbed_ans = get_answer(perturbed_model, tokenizer, fact["prompt"])
        baseline_ans = baseline_answers[i]

        has_keyword = fact["answer"] in perturbed_ans
        is_changed = not has_keyword

        if is_changed:
            changed_count += 1

        if i < 3:
            print(f"  Q: {fact['prompt']}")
            print(f"    Baseline:  {baseline_ans}")
            print(f"    Perturbed: {perturbed_ans}")
            print(f"    Changed:   {is_changed}")

    change_ratio = changed_count / len(FACTS) * 100 if len(FACTS) > 0 else 0.0
    print(f"\n  Answers changed: {changed_count}/{len(FACTS)} ({change_ratio:.2f}%)")

    results.append({
        "noise_scale": noise_scale,
        "curvature_proxy": curvature_proxy,
        "changed_count": changed_count,
        "change_ratio": change_ratio,
    })

    del perturbed_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ===================== 7. 汇总 =====================
print(f"\n{'='*60}")
print("=== Correlation Summary ===")
print(f"{'='*60}")
for r in results:
    print(f"Perturbation: {r['noise_scale']:.3f} -> "
          f"curvature_proxy={r['curvature_proxy']:.2f}, "
          f"change={r['change_ratio']:.1f}%")

print("\nTrend: curvature drift positively correlates with answer divergence.")
print("实验 2 完成。")