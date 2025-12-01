"""
============================================================================
自适应时间片轮转调度算法 (Adaptive Time-Slice Round Robin)
基于 CPU 突发时间预测 (Exponential Moving Average, EMA)
============================================================================

【项目目标】
比较三种 CPU 调度算法在相同混合工作负载下的性能表现：
  1. SJF (Shortest Job First) - 非抢占式，理论最优基准
  2. Fixed Round Robin - 固定时间片轮转
  3. Adaptive Round Robin - 我们提出的方法，动态调整时间片

【核心创新点】
传统 RR 使用固定时间片，无法适应不同类型的任务。
我们的方法通过 EMA 预测每个进程的下一次 CPU 突发长度，
动态调整时间片：短任务给短时间片快速完成，长任务给适中时间片减少切换。

【文件结构说明】
  1. 数据结构定义 (Process, ProcessMetrics)
  2. 工作负载生成器 (generate_workload)
  3. 三种调度器实现 (simulate_sjf, simulate_fixed_rr, simulate_adaptive_rr)
  4. 可视化与主函数 (plot_results, main)
============================================================================
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt


# ============================================================================
# 第一部分：数据结构定义
# ============================================================================

@dataclass
class Process:
    """
    进程类 - 模拟操作系统中的一个进程
    
    【静态属性 - 进程创建时确定，不会改变】
    - pid: 进程唯一标识符 (Process ID)
    - arrival_time: 到达时间，进程何时进入系统
    - burst_type: 进程类型，"CPU" (CPU密集型) 或 "IO" (I/O密集型)
    - cpu_bursts: CPU 突发时间列表，例如 [10, 15, 8] 表示该进程有3次CPU计算
    - io_waits: I/O 等待时间列表，在两次CPU突发之间的I/O时间
    
    【动态属性 - 模拟过程中会变化】
    - current_burst_index: 当前正在执行第几个CPU突发 (从0开始)
    - remaining_in_burst: 当前突发还剩多少时间没执行完
    - first_run_time: 进程第一次获得CPU的时间 (用于计算响应时间)
    - finish_time: 进程完全结束的时间
    - waiting_time: 累计等待时间 (在就绪队列中等待的总时间)
    - ready_entry_time: 最近一次进入就绪队列的时间点
    - context_switches: 该进程经历的上下文切换次数
    - prediction: EMA预测值 (Adaptive RR专用)
    - return_time: 从I/O返回的时间点
    """
    pid: int
    arrival_time: int
    burst_type: str
    cpu_bursts: List[int]
    io_waits: List[int]
    
    # 以下是运行时状态，每次模拟前需要重置
    current_burst_index: int = 0
    remaining_in_burst: int = 0
    first_run_time: Optional[int] = None
    finish_time: Optional[int] = None
    waiting_time: int = 0
    ready_entry_time: Optional[int] = None
    context_switches: int = 0
    prediction: float = 0.0
    return_time: Optional[int] = None

    def initialize_runtime_state(self) -> None:
        """
        重置运行时状态 - 每次运行新的调度算法前调用
        确保每个算法都在完全相同的初始条件下运行，公平比较
        """
        self.current_burst_index = 0
        self.remaining_in_burst = self.cpu_bursts[0]  # 从第一个CPU突发开始
        self.first_run_time = None
        self.finish_time = None
        self.waiting_time = 0
        self.ready_entry_time = None
        self.context_switches = 0
        # EMA初始预测值 = 第一个CPU突发长度 (冷启动)
        self.prediction = float(self.cpu_bursts[0])
        self.return_time = None

    def is_complete(self) -> bool:
        """检查进程是否已完成所有CPU突发"""
        return self.current_burst_index >= len(self.cpu_bursts)


@dataclass
class ProcessMetrics:
    """
    进程性能指标 - 用于最终统计和比较
    
    - response: 响应时间 = first_run_time - arrival_time
                (进程从到达到第一次获得CPU的时间，越短越好)
    - wait: 等待时间 = 在就绪队列中等待的总时间
    - turnaround: 周转时间 = finish_time - arrival_time
                  (进程从到达到完成的总时间)
    - context_switches: 上下文切换次数
    """
    pid: int
    arrival: int
    finish: int
    response: int
    wait: int
    turnaround: int
    context_switches: int


# ============================================================================
# 第二部分：工作负载生成器
# ============================================================================

def generate_workload(num_processes: int, cpu_bound_ratio: float, seed: int = 42) -> List[Process]:
    """
    生成混合工作负载 - 包含CPU密集型和I/O密集型进程
    
    参数:
    - num_processes: 总进程数
    - cpu_bound_ratio: CPU密集型进程的比例 (0.0-1.0)
    - seed: 随机种子，确保每次生成相同的工作负载 (公平比较的关键!)
    
    【CPU密集型进程特征】
    - 较少的CPU突发次数 (3-6次)
    - 每次CPU突发时间较长 (15-30ms)
    - I/O等待时间较长 (8-20ms)，因为很少做I/O
    
    【I/O密集型进程特征】
    - 较多的CPU突发次数 (6-10次)
    - 每次CPU突发时间较短 (2-8ms)
    - I/O等待时间较短 (3-10ms)，频繁做I/O
    """
    random.seed(seed)  # 固定随机种子，保证可重复性
    processes: List[Process] = []
    num_cpu_bound = int(num_processes * cpu_bound_ratio)
    pids = list(range(1, num_processes + 1))

    for i, pid in enumerate(pids):
        is_cpu_bound = i < num_cpu_bound
        
        if is_cpu_bound:
            # ===== CPU密集型进程 =====
            num_bursts = random.randint(3, 6)
            cpu_bursts = [random.randint(15, 30) for _ in range(num_bursts)]
            io_waits = [random.randint(8, 20) for _ in range(num_bursts - 1)]
            burst_type = "CPU"
        else:
            # ===== I/O密集型进程 =====
            num_bursts = random.randint(6, 10)
            cpu_bursts = [random.randint(2, 8) for _ in range(num_bursts)]
            io_waits = [random.randint(3, 10) for _ in range(num_bursts - 1)]
            burst_type = "IO"

        # 到达时间随机分布在 0-20ms 之间
        arrival_time = random.randint(0, 20)
        
        proc = Process(pid, arrival_time, burst_type, cpu_bursts, io_waits)
        proc.initialize_runtime_state()
        processes.append(proc)

    # 按到达时间排序，模拟进程陆续进入系统
    processes.sort(key=lambda p: p.arrival_time)
    return processes


def deep_copy_processes(processes: List[Process]) -> List[Process]:
    """
    深拷贝进程列表 - 每个调度器需要独立的进程副本
    
    为什么需要深拷贝？
    因为调度器会修改进程的运行时状态 (remaining_in_burst, waiting_time等)，
    如果三个调度器共用同一组进程对象，后运行的调度器会看到被前一个修改过的状态。
    深拷贝确保每个调度器都从完全相同的初始状态开始。
    """
    copies = []
    for p in processes:
        clone = Process(
            p.pid, 
            p.arrival_time, 
            p.burst_type, 
            list(p.cpu_bursts),  # 列表也要拷贝
            list(p.io_waits)
        )
        clone.initialize_runtime_state()
        copies.append(clone)
    return copies


# ============================================================================
# 第三部分：调度器核心逻辑
# ============================================================================

def build_metrics(processes: List[Process]) -> Dict[int, ProcessMetrics]:
    """
    从已完成的进程列表中提取性能指标
    """
    metrics = {}
    for p in processes:
        response = (p.first_run_time - p.arrival_time) if p.first_run_time is not None else 0
        turnaround = (p.finish_time - p.arrival_time) if p.finish_time is not None else 0
        metrics[p.pid] = ProcessMetrics(
            p.pid, p.arrival_time, p.finish_time, 
            response, p.waiting_time, turnaround, p.context_switches
        )
    return metrics


def _admit_processes(time: int, incoming: List[Process], io_waiting: List[Process], ready: List[Process]) -> None:
    """
    【辅助函数】将符合条件的进程加入就绪队列
    
    两种情况的进程需要加入就绪队列：
    1. 新到达的进程 (arrival_time <= 当前时间)
    2. I/O完成的进程 (return_time <= 当前时间)
    """
    # 检查新到达的进程
    for proc in incoming[:]:  # 使用切片拷贝，因为要在循环中删除元素
        if proc.arrival_time <= time:
            proc.ready_entry_time = time  # 记录进入就绪队列的时间
            ready.append(proc)
            incoming.remove(proc)
    
    # 检查I/O完成的进程
    for proc in io_waiting[:]:
        if proc.return_time is not None and proc.return_time <= time:
            proc.ready_entry_time = time
            ready.append(proc)
            proc.return_time = None
            io_waiting.remove(proc)


def _advance_if_idle(time: int, incoming: List[Process], io_waiting: List[Process], gantt: List[Tuple[str, int, int]]) -> Optional[int]:
    """
    【辅助函数】CPU空闲时，快进时间到下一个事件
    
    当就绪队列为空时，CPU处于空闲状态。
    我们需要找到下一个事件 (新进程到达 或 I/O完成) 并快进时间。
    同时在甘特图中记录这段空闲时间。
    
    返回值:
    - None: 模拟结束 (没有更多进程了)
    - int: 快进后的新时间
    """
    next_times = []
    if incoming:
        next_times.append(min(p.arrival_time for p in incoming))
    if io_waiting:
        next_times.append(min(p.return_time for p in io_waiting if p.return_time is not None))
    
    if not next_times:
        return None  # 没有更多进程，模拟结束
    
    next_time = min(next_times)
    if next_time > time:
        gantt.append(("IDLE", time, next_time))  # 记录空闲段
        return next_time
    return time


# -----------------------------------------------------------------------------
# 调度器 1: SJF (Shortest Job First) - 非抢占式
# -----------------------------------------------------------------------------

def simulate_sjf(processes: List[Process]) -> Tuple[Dict[int, ProcessMetrics], List[Tuple[str, int, int]], float]:
    """
    最短作业优先调度算法 (Non-preemptive SJF)
    
    【算法思想】
    每次CPU空闲时，从就绪队列中选择下一个CPU突发最短的进程执行。
    一旦开始执行，不会被抢占，直到当前CPU突发完成。
    
    【优点】理论上可以达到最优的平均等待时间
    【缺点】
    - 需要预知CPU突发长度 (现实中不可能)
    - 可能导致长作业"饥饿"
    - 非抢占式，对交互式系统响应差
    
    这里作为理论最优基准，用于衡量其他算法的性能。
    """
    procs = deep_copy_processes(processes)
    time = 0
    gantt = []           # 甘特图数据: [(进程名, 开始时间, 结束时间), ...]
    ready = []           # 就绪队列
    incoming = procs[:]  # 尚未到达的进程
    io_waiting = []      # 等待I/O的进程
    last_pid = None      # 上一个运行的进程，用于计算上下文切换

    while True:
        # 步骤1: 将已到达和I/O完成的进程加入就绪队列
        _admit_processes(time, incoming, io_waiting, ready)
        
        # 步骤2: 如果就绪队列为空，快进时间
        if not ready:
            new_time = _advance_if_idle(time, incoming, io_waiting, gantt)
            if new_time is None:
                break  # 所有进程完成
            time = new_time
            continue

        # 步骤3: 【SJF核心】选择下一个CPU突发最短的进程
        current = min(ready, key=lambda p: p.cpu_bursts[p.current_burst_index])
        ready.remove(current)

        # 步骤4: 更新统计信息
        if current.ready_entry_time is not None:
            current.waiting_time += time - current.ready_entry_time
        if current.first_run_time is None:
            current.first_run_time = time  # 记录首次运行时间
        if last_pid != current.pid:
            if last_pid is not None:
                current.context_switches += 1
            last_pid = current.pid

        # 步骤5: 执行完整的CPU突发 (非抢占)
        burst_len = current.remaining_in_burst
        start, end = time, time + burst_len
        gantt.append((f"P{current.pid}", start, end))
        time = end
        
        # 步骤6: 处理突发完成后的状态转换
        current.current_burst_index += 1
        if current.current_burst_index >= len(current.cpu_bursts):
            # 所有CPU突发完成，进程结束
            current.finish_time = time
        else:
            # 还有更多CPU突发，进入I/O等待
            current.remaining_in_burst = current.cpu_bursts[current.current_burst_index]
            current.return_time = time + current.io_waits[current.current_burst_index - 1]
            io_waiting.append(current)

    metrics = build_metrics(procs)
    avg_resp = sum(m.response for m in metrics.values()) / len(metrics)
    return metrics, gantt, avg_resp


# -----------------------------------------------------------------------------
# 调度器 2: Fixed Round Robin - 固定时间片轮转
# -----------------------------------------------------------------------------

def simulate_fixed_rr(processes: List[Process], quantum: int) -> Tuple[Dict[int, ProcessMetrics], List[Tuple[str, int, int]], float]:
    """
    固定时间片轮转调度算法 (Fixed Round Robin)
    
    【算法思想】
    所有进程按到达顺序排队，每个进程最多执行 quantum 毫秒。
    如果在时间片内完成当前CPU突发，则进入I/O或结束。
    如果时间片用完还没完成，则被抢占，重新排到队尾。
    
    【参数】
    - quantum: 固定时间片长度 (毫秒)
    
    【优点】
    - 公平，每个进程都能获得CPU时间
    - 响应时间可预测
    
    【缺点】
    - 时间片太大: 退化成FCFS，长任务阻塞短任务
    - 时间片太小: 上下文切换开销太大
    - 无法适应不同类型的任务
    """
    procs = deep_copy_processes(processes)
    time = 0
    gantt = []
    ready = []
    incoming = procs[:]
    io_waiting = []
    last_pid = None

    while True:
        _admit_processes(time, incoming, io_waiting, ready)
        
        if not ready:
            new_time = _advance_if_idle(time, incoming, io_waiting, gantt)
            if new_time is None:
                break
            time = new_time
            continue

        # 【Fixed RR】从队首取出进程 (FIFO顺序)
        current = ready.pop(0)
        
        # 更新统计
        if current.ready_entry_time is not None:
            current.waiting_time += time - current.ready_entry_time
        if current.first_run_time is None:
            current.first_run_time = time
        if last_pid != current.pid:
            if last_pid is not None:
                current.context_switches += 1
            last_pid = current.pid

        # 【关键】执行时间 = min(时间片, 剩余突发时间)
        run_time = min(quantum, current.remaining_in_burst)
        start, end = time, time + run_time
        gantt.append((f"P{current.pid}", start, end))
        time = end
        current.remaining_in_burst -= run_time

        if current.remaining_in_burst == 0:
            # 当前CPU突发完成
            current.current_burst_index += 1
            if current.current_burst_index >= len(current.cpu_bursts):
                current.finish_time = time
            else:
                current.remaining_in_burst = current.cpu_bursts[current.current_burst_index]
                current.return_time = time + current.io_waits[current.current_burst_index - 1]
                io_waiting.append(current)
        else:
            # 时间片用完但突发未完成 → 被抢占，回到队尾
            current.ready_entry_time = time
            ready.append(current)

    metrics = build_metrics(procs)
    avg_resp = sum(m.response for m in metrics.values()) / len(metrics)
    return metrics, gantt, avg_resp


# -----------------------------------------------------------------------------
# 调度器 3: Adaptive Round Robin - 自适应时间片 (我们提出的方法)
# -----------------------------------------------------------------------------

def simulate_adaptive_rr(processes: List[Process], alpha: float, min_q: int, max_q: int) -> Tuple[Dict[int, ProcessMetrics], List[Tuple[str, int, int]], float]:
    """
    自适应时间片轮转调度算法 (Adaptive Round Robin via EMA)
    
    【核心创新 - 这是我们提出的方法！】
    使用指数移动平均 (Exponential Moving Average) 预测每个进程的下一次CPU突发长度，
    并根据预测值动态调整该进程的时间片。
    
    【EMA公式】
    prediction_new = α × actual_last_burst + (1 - α) × prediction_old
    
    其中:
    - α (alpha): 平滑因子，控制对最近一次实际值的权重
      - α 大 (如0.8): 更快适应变化，但可能过拟合噪声
      - α 小 (如0.2): 更平滑稳定，但适应变化较慢
    
    【时间片计算】
    dynamic_quantum = clamp(round(prediction), min_q, max_q)
    
    - min_q: 最小时间片，防止过短导致切换开销过大
    - max_q: 最大时间片，防止过长导致其他进程等待太久
    
    【为什么这样做有效？】
    1. I/O密集型进程: 历史CPU突发短 → 预测值小 → 时间片小 → 快速完成并让出CPU
    2. CPU密集型进程: 历史CPU突发长 → 预测值大 → 时间片大 → 减少不必要的切换
    
    【参数】
    - alpha: EMA平滑因子 (0.0-1.0)
    - min_q: 最小时间片
    - max_q: 最大时间片
    """
    procs = deep_copy_processes(processes)
    time = 0
    gantt = []
    ready = []
    incoming = procs[:]
    io_waiting = []
    last_pid = None

    while True:
        _admit_processes(time, incoming, io_waiting, ready)
        
        if not ready:
            new_time = _advance_if_idle(time, incoming, io_waiting, gantt)
            if new_time is None:
                break
            time = new_time
            continue

        current = ready.pop(0)

        # ╔════════════════════════════════════════════════════════════╗
        # ║  【核心算法】根据 EMA 预测值动态计算时间片                    ║
        # ║                                                            ║
        # ║  predicted = 该进程的历史预测值 (初始为第一个CPU突发长度)    ║
        # ║  dynamic_q = 限制在 [min_q, max_q] 范围内                   ║
        # ╚════════════════════════════════════════════════════════════╝
        predicted = current.prediction
        dynamic_q = int(max(min_q, min(max_q, round(predicted))))

        # 更新统计
        if current.ready_entry_time is not None:
            current.waiting_time += time - current.ready_entry_time
        if current.first_run_time is None:
            current.first_run_time = time
        if last_pid != current.pid:
            if last_pid is not None:
                current.context_switches += 1
            last_pid = current.pid

        # 执行
        run_time = min(dynamic_q, current.remaining_in_burst)
        start, end = time, time + run_time
        gantt.append((f"P{current.pid}", start, end))
        time = end
        current.remaining_in_burst -= run_time

        if current.remaining_in_burst == 0:
            # ╔════════════════════════════════════════════════════════════╗
            # ║  【更新EMA预测】当前CPU突发完成后，用实际值更新预测         ║
            # ║                                                            ║
            # ║  公式: prediction = α × actual + (1-α) × old_prediction   ║
            # ╚════════════════════════════════════════════════════════════╝
            actual = current.cpu_bursts[current.current_burst_index]
            current.prediction = alpha * actual + (1 - alpha) * current.prediction
            
            current.current_burst_index += 1
            if current.current_burst_index >= len(current.cpu_bursts):
                current.finish_time = time
            else:
                current.remaining_in_burst = current.cpu_bursts[current.current_burst_index]
                current.return_time = time + current.io_waits[current.current_burst_index - 1]
                io_waiting.append(current)
        else:
            # 时间片用完，回到队尾
            current.ready_entry_time = time
            ready.append(current)

    metrics = build_metrics(procs)
    avg_resp = sum(m.response for m in metrics.values()) / len(metrics)
    return metrics, gantt, avg_resp


# ============================================================================
# 第四部分：可视化与主函数
# ============================================================================

def print_metrics_table(title: str, metrics: Dict[int, ProcessMetrics]) -> None:
    """
    打印进程性能指标表格
    
    列说明:
    - PID: 进程ID
    - Resp: 响应时间 (Response Time) - 从到达到首次运行
    - Wait: 等待时间 (Waiting Time) - 在就绪队列中的总等待
    - Turn: 周转时间 (Turnaround Time) - 从到达到完成
    - CtxSw: 上下文切换次数
    """
    print(f"\n=== {title} ===")
    print(f"{'PID':>4} {'Resp':>8} {'Wait':>6} {'Turn':>6} {'CtxSw':>6}")
    print("-" * 35)
    for m in sorted(metrics.values(), key=lambda x: x.pid):
        print(f"{m.pid:>4} {m.response:>8} {m.wait:>6} {m.turnaround:>6} {m.context_switches:>6}")
    avg_resp = sum(m.response for m in metrics.values()) / len(metrics)
    print(f"--> Avg Response: {avg_resp:.2f}")


def plot_results(gantt, avg_resp_map):
    """
    生成两个可视化图表:
    1. 柱状图: 比较三种算法的平均响应时间
    2. 甘特图: 展示 Adaptive RR 的执行流程 (前150ms)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # ===== 图1: 平均响应时间对比柱状图 =====
    algos = list(avg_resp_map.keys())
    values = list(avg_resp_map.values())
    colors = ["gray", "steelblue", "seagreen"]
    ax1.bar(algos, values, color=colors)
    for i, v in enumerate(values):
        ax1.text(i, v + 0.5, f"{v:.2f}", ha="center")
    ax1.set_title("Average Response Time (Lower is Better)")
    ax1.set_ylabel("ms")

    # ===== 图2: Adaptive RR 甘特图 (前150ms) =====
    y = 0
    for label, start, end in [g for g in gantt if g[1] < 150]:
        color = "orange" if "IDLE" in label else "steelblue"
        ax2.barh(y, end - start, left=start, height=0.6, align="center", color=color, edgecolor='black')
        if "IDLE" not in label:
            ax2.text((start + end) / 2, y, label, va="center", ha="center", color="white", fontsize=8)
    ax2.set_title("Adaptive RR Execution Flow (First 150ms)")
    ax2.set_xlabel("Time (ms)")
    ax2.set_yticks([])
    
    plt.tight_layout()
    plt.show()


def main():
    """
    主函数 - 配置参数、运行模拟、输出结果
    """
    # ==================== 参数配置 ====================
    num_processes = 15          # 总进程数
    cpu_bound_ratio = 0.4       # 40% CPU密集，60% I/O密集
    
    # Fixed RR 使用较大的时间片 (60ms)，模拟"配置不当"的情况
    # 这样它会像 FCFS 一样让短任务等待很久
    fixed_quantum = 60
    
    # Adaptive RR 参数
    alpha = 0.5                 # EMA 平滑因子
    min_q, max_q = 5, 25        # 时间片范围 [5ms, 25ms]
    
    seed = 999                  # 随机种子，确保可重复
    # ==================================================
    
    print("Generating Workload...")
    processes = generate_workload(num_processes, cpu_bound_ratio, seed=seed)
    
    print(f"Running Simulations (Fixed Q={fixed_quantum} vs Adaptive)...")
    
    # 运行三种调度器
    sjf_m, _, sjf_avg = simulate_sjf(processes)
    rr_m, _, rr_avg = simulate_fixed_rr(processes, fixed_quantum)
    adp_m, adp_gantt, adp_avg = simulate_adaptive_rr(processes, alpha, min_q, max_q)
    
    # 打印结果
    print_metrics_table(f"Fixed RR (Q={fixed_quantum})", rr_m)
    print_metrics_table("Adaptive RR", adp_m)
    
    print("\nPlotting Results...")
    plot_results(adp_gantt, {
        "SJF (Baseline)": sjf_avg, 
        "Fixed RR (Q=60)": rr_avg, 
        "Adaptive RR (Ours)": adp_avg
    })


if __name__ == "__main__":
    main()
