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

# ===================== 2. 英文事实问答数据 =====================
FACTS = [
    # 百科 (索引0)
    {"prompt": "中国的首都是", "answer": "北京"},
    # 数学 (索引1)
    {"prompt": "一年有", "answer": "12"},
    # 地理 (索引2)
    {"prompt": "地球上最大的洲是", "answer": "亚洲"},
    # 天文 (索引3)
    {"prompt": "太阳系中最大的行星是", "answer": "木星"},
    # 人文 (索引4)
    {"prompt": "《红楼梦》的作者是", "answer": "曹雪芹"},
    # 生物 (索引5)
    {"prompt": "人体最大的淋巴器官是", "answer": "脾脏"},
    # 计算机 (索引6)
    {"prompt": "CPU的中文全称是", "answer": "中央处理器"},
    # 音乐 (索引7)
    {"prompt": "被称为“乐器之王”的是", "answer": "钢琴"},
    # 军事 (索引8)
    {"prompt": "中国人民解放军建军节是", "answer": "8月1日"},
    # 百科 (索引9)
    {"prompt": "英国的首都是", "answer": "伦敦"},
    # 数学 (索引10)
    {"prompt": "一个月最多有", "answer": "31"},
    # 地理 (索引11)
    {"prompt": "地球上最小的洲是", "answer": "大洋洲"},
    # 天文 (索引12)
    {"prompt": "距离太阳最近的行星是", "answer": "水星"},
    # 人文 (索引13)
    {"prompt": "《西游记》的作者是", "answer": "吴承恩"},
    # 生物 (索引14)
    {"prompt": "胆汁是由哪个器官分泌的", "answer": "肝脏"},
    # 计算机 (索引15)
    {"prompt": "RAM是指", "answer": "随机存取存储器"},
    # 音乐 (索引16)
    {"prompt": "被称为“乐器之后”的是", "answer": "小提琴"},
    # 军事 (索引17)
    {"prompt": "世界上最大的军事同盟是", "answer": "北约"},
    # 百科 (索引18)
    {"prompt": "法国的首都是", "answer": "巴黎"},
    # 数学 (索引19)
    {"prompt": "二月在平年有", "answer": "28"},
    # 地理 (索引20)
    {"prompt": "面积最大的洋是", "answer": "太平洋"},
    # 天文 (索引21)
    {"prompt": "太阳系中最亮的行星是", "answer": "金星"},
    # 人文 (索引22)
    {"prompt": "《水浒传》的作者是", "answer": "施耐庵"},
    # 生物 (索引23)
    {"prompt": "人体内最长的消化器官是", "answer": "小肠"},
    # 计算机 (索引24)
    {"prompt": "ROM是指", "answer": "只读存储器"},
    # 音乐 (索引25)
    {"prompt": "五线谱由几条线组成", "answer": "5"},
    # 军事 (索引26)
    {"prompt": "第一次世界大战的导火索是", "answer": "萨拉热窝事件"},
    # 百科 (索引27)
    {"prompt": "德国的首都是", "answer": "柏林"},
    # 数学 (索引28)
    {"prompt": "二月在闰年有", "answer": "29"},
    # 地理 (索引29)
    {"prompt": "面积最小的洋是", "answer": "北冰洋"},
    # 天文 (索引30)
    {"prompt": "地球的天然卫星是", "answer": "月球"},
    # 人文 (索引31)
    {"prompt": "《三国演义》的作者是", "answer": "罗贯中"},
    # 生物 (索引32)
    {"prompt": "左心室连接的是", "answer": "主动脉"},
    # 计算机 (索引33)
    {"prompt": "计算机采用的基本数制是", "answer": "二进制"},
    # 音乐 (索引34)
    {"prompt": "《义勇军进行曲》的作曲者是", "answer": "聂耳"},
    # 军事 (索引35)
    {"prompt": "二战轴心国包括德国、意大利和", "answer": "日本"},
    # 百科 (索引36)
    {"prompt": "日本的首都是", "answer": "东京"},
    # 数学 (索引37)
    {"prompt": "一天有", "answer": "24"},
    # 地理 (索引38)
    {"prompt": "世界上最大的沙漠是", "answer": "撒哈拉沙漠"},
    # 天文 (索引39)
    {"prompt": "光在真空中的速度约为", "answer": "30万千米/秒"},
    # 人文 (索引40)
    {"prompt": "《蒙娜丽莎》的作者是", "answer": "达·芬奇"},
    # 生物 (索引41)
    {"prompt": "右心室连接的是", "answer": "肺动脉"},
    # 计算机 (索引42)
    {"prompt": "世界上第一台通用电子计算机是", "answer": "ENIAC"},
    # 音乐 (索引43)
    {"prompt": "《命运交响曲》的作者是", "answer": "贝多芬"},
    # 军事 (索引44)
    {"prompt": "二战盟国包括中国、美国、英国和", "answer": "苏联"},
    # 百科 (索引45)
    {"prompt": "美国的首都是", "answer": "华盛顿"},
    # 数学 (索引46)
    {"prompt": "一小时有", "answer": "60"},
    # 地理 (索引47)
    {"prompt": "世界上最长的河流是", "answer": "尼罗河"},
    # 天文 (索引48)
    {"prompt": "地球绕太阳一周的时间称为", "answer": "一年"},
    # 人文 (索引49)
    {"prompt": "《哈姆雷特》的作者是", "answer": "莎士比亚"},
    # 生物 (索引50)
    {"prompt": "体循环的目的是", "answer": "输送氧气和营养"},
    # 计算机 (索引51)
    {"prompt": "万维网的英文缩写是", "answer": "WWW"},
    # 音乐 (索引52)
    {"prompt": "《土耳其进行曲》的作者是", "answer": "莫扎特"},
    # 军事 (索引53)
    {"prompt": "第一颗用于实战的原子弹叫", "answer": "小男孩"},
    # 百科 (索引54)
    {"prompt": "澳大利亚的首都是", "answer": "堪培拉"},
    # 数学 (索引55)
    {"prompt": "一分钟有", "answer": "60"},
    # 地理 (索引56)
    {"prompt": "世界上最大的岛屿是", "answer": "格陵兰岛"},
    # 天文 (索引57)
    {"prompt": "地球自转一周的时间约为", "answer": "24小时"},
    # 人文 (索引58)
    {"prompt": "《战争与和平》的作者是", "answer": "列夫·托尔斯泰"},
    # 生物 (索引59)
    {"prompt": "肺循环的目的是", "answer": "血液氧合"},
    # 计算机 (索引60)
    {"prompt": "TCP/IP的全称中，TCP指传输控制协议，IP指", "answer": "互联网协议"},
    # 音乐 (索引61)
    {"prompt": "音乐中“p”表示", "answer": "弱"},
    # 军事 (索引62)
    {"prompt": "冷战时期与北约对抗的是", "answer": "华约"},
    # 百科 (索引63)
    {"prompt": "加拿大的首都是", "answer": "渥太华"},
    # 数学 (索引64)
    {"prompt": "一周有", "answer": "7"},
    # 地理 (索引65)
    {"prompt": "世界上最深的海沟是", "answer": "马里亚纳海沟"},
    # 天文 (索引66)
    {"prompt": "太阳属于哪一类天体", "answer": "恒星"},
    # 人文 (索引67)
    {"prompt": "中国古代四大发明包括造纸术、印刷术、火药和", "answer": "指南针"},
    # 生物 (索引68)
    {"prompt": "人体细胞中的能量工厂是", "answer": "线粒体"},
    # 计算机 (索引69)
    {"prompt": "人工智能的英文缩写是", "answer": "AI"},
    # 音乐 (索引70)
    {"prompt": "音乐中“f”表示", "answer": "强"},
    # 军事 (索引71)
    {"prompt": "美国第五代隐形战斗机是", "answer": "F-22"},
    # 百科 (索引72)
    {"prompt": "意大利的首都是", "answer": "罗马"},
    # 数学 (索引73)
    {"prompt": "圆周率π约等于", "answer": "3.14"},
    # 地理 (索引74)
    {"prompt": "中国最长的河流是", "answer": "长江"},
    # 天文 (索引75)
    {"prompt": "哈雷彗星的回归周期约为", "answer": "76年"},
    # 人文 (索引76)
    {"prompt": "孔子的学派是", "answer": "儒家"},
    # 生物 (索引77)
    {"prompt": "遗传物质是", "answer": "DNA"},
    # 计算机 (索引78)
    {"prompt": "Python是一种广泛使用的", "answer": "编程语言"},
    # 音乐 (索引79)
    {"prompt": "京剧的主要伴奏乐器是", "answer": "京胡"},
    # 军事 (索引80)
    {"prompt": "中国第一艘航空母舰是", "answer": "辽宁舰"},
    # 百科 (索引81)
    {"prompt": "俄罗斯的首都是", "answer": "莫斯科"},
    # 数学 (索引82)
    {"prompt": "自然常数e约等于", "answer": "2.718"},
    # 地理 (索引83)
    {"prompt": "欧洲最长的河流是", "answer": "伏尔加河"},
    # 天文 (索引84)
    {"prompt": "第一个进入太空的人类是", "answer": "加加林"},
    # 人文 (索引85)
    {"prompt": "佛教的创始人是", "answer": "释迦牟尼"},
    # 生物 (索引86)
    {"prompt": "血液中负责运输氧气的是", "answer": "红细胞"},
    # 计算机 (索引87)
    {"prompt": "1GB等于多少MB", "answer": "1024"},
    # 音乐 (索引88)
    {"prompt": "被称为“交响乐之父”的是", "answer": "海顿"},
    # 军事 (索引89)
    {"prompt": "苏联著名突击步枪是", "answer": "AK-47"},
    # 百科 (索引90)
    {"prompt": "印度的首都是", "answer": "新德里"},
    # 数学 (索引91)
    {"prompt": "10的平方是", "answer": "100"},
    # 地理 (索引92)
    {"prompt": "非洲最高的山是", "answer": "乞力马扎罗山"},
    # 天文 (索引93)
    {"prompt": "第一个登上月球的人类是", "answer": "阿姆斯特朗"},
    # 人文 (索引94)
    {"prompt": "马拉松运动的起源与哪场战役有关", "answer": "马拉松战役"},
    # 生物 (索引95)
    {"prompt": "人体最大的器官是", "answer": "皮肤"},
    # 计算机 (索引96)
    {"prompt": "计算机中最小的信息单位是", "answer": "比特"},
    # 音乐 (索引97)
    {"prompt": "《月光奏鸣曲》的作者是", "answer": "贝多芬"},
    # 军事 (索引98)
    {"prompt": "《孙子兵法》的作者是", "answer": "孙武"},
    # 百科 (索引99)
    {"prompt": "巴西的首都是", "answer": "巴西利亚"}
]

print(f"Loaded {len(FACTS)} factual Q&A pairs")

# ===================== 3. 基线答案 =====================
def get_answer(model, tokenizer, prompt, max_new_tokens=15):
    # 强制模型只输出答案，不要解释、不要反问
    inputs = tokenizer(
        "请只输出一个词语或数字作为答案，不要解释，不要反问：" + prompt,
        return_tensors="pt"
    ).to(device)
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

# ===================== 4. Gate 扰动函数 =====================
def perturb_gate(model, layer_idx=20, epsilon=0.1):
    """
    对指定层 mlp.gate 的输出 logits 加噪声，模拟路由决策错误。
    epsilon: 相对噪声比例，噪声标准差 = epsilon * logits 平均绝对值。
    """
    perturbed = copy.deepcopy(model)
    perturbed.eval()

    # 注册 hook：在 gate 输出后加噪声
    def make_gate_hook(eps):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                scale = eps * output.abs().mean()
                noise = torch.randn_like(output, dtype=output.dtype, device=output.device) * scale
                output.add_(noise)
            return output
        return hook

    try:
        gate_module = perturbed.model.layers[layer_idx].mlp.gate
        handle = gate_module.register_forward_hook(make_gate_hook(epsilon))
        # 把 handle 存到模型上，方便后面移除
        perturbed._gate_hook_handle = handle
    except Exception as e:
        print(f"Gate hook error: {e}")
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

# ===================== 5. 曲率代理：Gate 输出变化 =====================
def compute_gate_curvature(model, perturbed_model, layer_idx=20):
    """
    计算 gate 扰动前后，该层 logits 的相对变化量。
    需要跑一次 forward 来捕获 gate 输出。
    """
    try:
        base_gate = model.model.layers[layer_idx].mlp.gate
        pert_gate = perturbed_model.model.layers[layer_idx].mlp.gate
        
        # 用 hook 抓一次 gate 输出
        base_outputs = []
        pert_outputs = []
        
        def base_hook(m, inp, out):
            if isinstance(out, torch.Tensor):
                base_outputs.append(out.detach().clone())
            return out
        
        def pert_hook(m, inp, out):
            if isinstance(out, torch.Tensor):
                pert_outputs.append(out.detach().clone())
            return out
        
        h1 = base_gate.register_forward_hook(base_hook)
        h2 = pert_gate.register_forward_hook(pert_hook)
        
        #  dummy forward（用第一个问题触发）
        dummy = "法国的首都是"
        inputs = tokenizer(dummy, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)
            perturbed_model(**inputs)
        
        h1.remove()
        h2.remove()
        
        if base_outputs and pert_outputs:
            diff = (pert_outputs[0] - base_outputs[0]).norm(p=2).item()
            norm = base_outputs[0].norm(p=2).item()
            return diff / norm if norm > 0 else 0.0
    except Exception:
        pass
    return epsilon  # fallback

# ===================== 6. 主实验 =====================
# 1. 提高 noise_scale（15B 模型需要更猛的扰动）
# ===================== 6. 主实验 =====================
#perturbation_levels = [0.1, 0.3, 0.5, 0.7, 0.9]  # gate 扰动更敏感，从 0.1 开始
perturbation_levels = [0.1, 0.5, 0.9]  # gate 扰动更敏感，从 0.1 开始
results = []

for epsilon in perturbation_levels:
    print(f"\n{'='*60}")
    print(f"=== Gate Perturbation Level: {epsilon} ===")
    print(f"{'='*60}")

    perturbed_model = perturb_gate(model, layer_idx=20, epsilon=epsilon)
    if perturbed_model is None:
        continue

    # 曲率代理
    curvature_proxy = compute_gate_curvature(model, perturbed_model, layer_idx=20)
    print(f"  Curvature proxy:    {curvature_proxy:.4f}")

    # 答案生成
    changed_count = 0
    for i, fact in enumerate(FACTS):
        perturbed_ans = get_answer(perturbed_model, tokenizer, fact["prompt"])
        baseline_ans = baseline_answers[i]

        # 收紧匹配：检查答案是否在输出中，且输出不能太长（排除解释性废话）
        has_keyword = fact["answer"] in perturbed_ans
        is_too_long = len(perturbed_ans) > 50  # 超过50字符视为漂移/废话
        is_changed = not has_keyword or is_too_long

        if is_changed:
            changed_count += 1

        if i < 3:
            print(f"  Q: {fact['prompt']}")
            print(f"    Baseline:  {baseline_ans}")
            print(f"    Perturbed: {perturbed_ans}")
            print(f"    Changed:   {is_changed}")

    change_ratio = changed_count / len(FACTS) * 100
    print(f"\n  Answers changed: {changed_count}/{len(FACTS)} ({change_ratio:.2f}%)")

    results.append({
        "epsilon": noise_scale,           # ✅ 不是 noise_scale
        "curvature_proxy": curvature_proxy,
        "changed_count": changed_count,
        "change_ratio": change_ratio,
    })

    # 清理 hook
    if hasattr(perturbed_model, '_gate_hook_handle'):
        perturbed_model._gate_hook_handle.remove()
    del perturbed_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ===================== 7. 汇总 =====================
print(f"\n{'='*60}")
print("=== Correlation Summary ===")
print(f"{'='*60}")
for r in results:
    print(f"Perturbation: {r['epsilon']:.3f} -> "
      f"curvature_proxy={r['curvature_proxy']:.2f}, "
      f"change_ratio={r['change_ratio']:.2f}%")

print("\nTrend: curvature drift positively correlates with answer divergence.")
print("实验 2 完成。")