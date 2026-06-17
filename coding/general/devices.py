import subprocess
import torch
def get_available_cuda_devices(min_memory_gb=6, max_memory_usage=10, max_gpu_util=40):
    try:
        result = subprocess.check_output([
            'nvidia-smi',
            '--query-gpu=memory.total,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'
        ], encoding='utf-8')
        lines = result.strip().split('\n')
        candidates = []
        for i, line in enumerate(lines):
            try:
                total_mem, used_mem, gpu_util = map(int, line.strip().split(','))
            except ValueError:
                continue
            total_gb = total_mem / 1024
            used_gb = used_mem / 1024
            mem_usage_percent = used_gb / total_gb * 100
            if total_gb >= min_memory_gb and mem_usage_percent < max_memory_usage:
                if gpu_util < max_gpu_util:
                    candidates.append((i, gpu_util))
                else:
                    print(f'\033[32mdevice{i}:{gpu_util} > {max_gpu_util}\033[0m')
        candidates = sorted(candidates, key=lambda x: x[1])
        available_devices = [i for i, _ in candidates]
        return available_devices, len(lines)
    except Exception as e:
        print("无法调用 nvidia-smi，请确认已安装 NVIDIA 驱动且 nvidia-smi 可用。")
        print("错误信息：", e)
        return [], 0