import torch
import time
from models.pidnet import get_pred_model

# ================= 导师硬核配置区 =================
# 1. 你的两个 best.pt 路径
WEIGHT_PATH_A = r"output/custom/pidnet_small_local_v4/best.pt"  # 模型 A (比如你的原生S)
# 2. 你的网络规格 (根据实际修改, 如果你已经改成了带 SPM/ODConv 的模型，它会自动加载)
MODEL_NAME = 'pidnet_s' 
NUM_CLASSES = 3
INPUT_SIZE = (1, 3, 512, 512) # 512x512 是你的滑窗标准尺寸

# 测试性能参数
WARMUP_STEPS = 100  # 显卡预热步数
TEST_STEPS = 300    # 正式测试步数
# ===================================================

def run_benchmark(weight_path):
    print(f"\n================ 开始体检: {os.path.basename(weight_path)} ================")
    
    # 1. 实例化模型
    model = get_pred_model(name=MODEL_NAME, num_classes=NUM_CLASSES)
    
    if os.path.exists(weight_path):
        # 兼容性加载
        checkpoint = torch.load(weight_path, map_location='cuda:0')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

        # 多重前缀剥离：DataParallel('module.') + FullModel('model.')
        model_dict = model.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            if name.startswith('model.'):
                name = name[6:]  # 去掉 'model.' 这 6 个字符
            if name in model_dict:
                new_state_dict[name] = v

        match_count = len(new_state_dict)
        print(f"=> 模型结构键数: {len(model_dict)}, 权重文件键数: {len(state_dict)}, 匹配: {match_count}")
        model.load_state_dict(new_state_dict, strict=False)
        print(f"=> 成功加载权重: {weight_path}")
    else:
        print("⚠️ 未找到权重文件，将测试【未训练的纯空表结构】(参数和计算量完全一致)")

    model = model.cuda()
    model.eval()

    # 2. 统计参数量 (Params) 和计算量 (FLOPs)
    try:
        from thop import profile
        dummy_input = torch.randn(*INPUT_SIZE).cuda()
        flops, params = profile(model, inputs=(dummy_input, ), verbose=False)
        print(f"👉 真实参数量 (Params) : {params / 1e6:.4f} M (百万)")
        print(f"👉 运行计算量 (FLOPs)  : {flops / 1e9:.4f} G (十亿次浮点运算)")
    except ImportError:
        # 备用原生统计
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"👉 真实参数量 (Params) : {params / 1e6:.4f} M")
        print("⚠️ 未安装 thop 库，跳过 GFLOPs 计算。")

    # 3. 显卡预热 (Warm-up) —— 拒绝学术造假，还原真实速度
    print(f"=> 正在对 {torch.cuda.get_device_name(0)} 进行 {WARMUP_STEPS} 次热身运动...")
    dummy_input = torch.randn(*INPUT_SIZE).cuda()
    with torch.no_grad():
        for _ in range(WARMUP_STEPS):
            _ = model(dummy_input)
            
    # 4. 严谨的时钟同步测试 FPS
    print(f"=> 正在进行 {TEST_STEPS} 次极限速度测试...")
    torch.cuda.synchronize() # 强行同步 GPU 硬件时钟
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(TEST_STEPS):
            _ = model(dummy_input)
            
    torch.cuda.synchronize() # 强行等显卡把所有任务算完
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_latency = (total_time / TEST_STEPS) * 1000 # 毫秒 (ms)
    fps = TEST_STEPS / total_time
    
    print("\n----------------- 性能体检结果 -----------------")
    print(f"👉 平均单张推理延迟 (Latency) : {avg_latency:.2f} ms")
    print(f"👉 极限每秒处理帧数 (FPS)     : {fps:.2f} 帧/秒")
    print("------------------------------------------------")

if __name__ == "__main__":
    import os
    run_benchmark(WEIGHT_PATH_A)