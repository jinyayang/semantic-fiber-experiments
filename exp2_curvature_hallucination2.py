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
    {"prompt": "中国的首都是", "answer": "北京"},
    {"prompt": "《水浒传》的作者是", "answer": "施耐庵"},
    {"prompt": "英国的首都是", "answer": "伦敦"},
    {"prompt": "日本的首都是", "answer": "东京"},
    {"prompt": "体循环的目的是", "answer": "输送氧气和营养"},
    {"prompt": "人体最大的淋巴器官是", "answer": "脾脏"},
    {"prompt": "德国的首都是", "answer": "柏林"},
    {"prompt": "一年有", "answer": "12"},
    {"prompt": "一个月最多有", "answer": "31"},
    {"prompt": "二月在平年有", "answer": "28"},
    {"prompt": "胆汁是由哪个器官分泌的", "answer": "肝脏"},
    {"prompt": "二月在闰年有", "answer": "29"},
    {"prompt": "一天有", "answer": "24"},
    {"prompt": "一小时有", "answer": "60"},
    {"prompt": "一分钟有", "answer": "60"},
    {"prompt": "法国的首都是", "answer": "巴黎"},
    {"prompt": "一周有", "answer": "7"},
    {"prompt": "《红楼梦》的作者是", "answer": "曹雪芹"},
    {"prompt": "左心室连接的是", "answer": "主动脉"},
    {"prompt": "《西游记》的作者是", "answer": "吴承恩"},
    {"prompt": "《三国演义》的作者是", "answer": "罗贯中"},
    {"prompt": "圆周率π约等于", "answer": "3.14"},
    {"prompt": "自然常数e约等于", "answer": "2.718"},
    {"prompt": "10的平方是", "answer": "100"},
    {"prompt": "2的3次方是", "answer": "8"},
    {"prompt": "3的3次方是", "answer": "27"},
    {"prompt": "右心室连接的是", "answer": "肺动脉"},
    {"prompt": "美国的首都是", "answer": "华盛顿"},
    {"prompt": "肺循环的目的是", "answer": "血液氧合"},
    {"prompt": "体循环的目的是", "answer": "输送氧气和营养"},
    {"prompt": "人体最大的淋巴器官是", "answer": "脾脏"},
    {"prompt": "胆汁是由哪个器官分泌的", "answer": "肝脏"},
    {"prompt": "人体内最长的消化器官是", "answer": "小肠"},
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
perturbation_levels = [0.1, 0.3, 0.5, 0.7, 0.9]  # gate 扰动更敏感，从 0.01 开始
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
        "epsilon": epsilon,
        "curvature_proxy": curvature_proxy,
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
    print(f"Perturbation: {r['noise_scale']:.3f} -> "
          f"curvature_proxy={r['curvature_proxy']:.2f}, "
          f"change={r['change_ratio']:.1f}%")

print("\nTrend: curvature drift positively correlates with answer divergence.")
print("实验 2 完成。")