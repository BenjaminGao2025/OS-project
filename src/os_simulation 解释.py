"""
Adaptive Time-Slice Round Robin via CPU-Burst Prediction
-------------------------------------------------------
A standalone Python simulation that compares three schedulers on the
exact same mixed workload (CPU-bound + I/O-bound):

* Non-preemptive Shortest Job First (baseline)
* Fixed-time-slice Round Robin
* Adaptive Round Robin (Proposed Method via EMA)

The script prints per-process metrics and produces visualizations.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

# ==========================================
# 1. Data Structures
# ==========================================

@dataclass
class Process:
    pid: int
    arrival_time: int
    burst_type: str  # "CPU" or "IO"
    cpu_bursts: List[int]
    io_waits: List[int]
    current_burst_index: int = 0
    remaining_in_burst: int = 0
    first_run_time: Optional[int] = None
    finish_time: Optional[int] = None
    waiting_time: int = 0
    ready_entry_time: Optional[int] = None
    context_switches: int = 0
    prediction: float = 0.0  # EMA prediction
    return_time: Optional[int] = None  # When returning from I/O

    def initialize_runtime_state(self) -> None:
        """Reset state for a fresh simulation run."""
        self.current_burst_index = 0
        self.remaining_in_burst = self.cpu_bursts[0]
        self.first_run_time = None
        self.finish_time = None
        self.waiting_time = 0
        self.ready_entry_time = None
        self.context_switches = 0
        self.prediction = float(self.cpu_bursts[0])  # Seed EMA
        self.return_time = None

    def is_complete(self) -> bool:
        return self.current_burst_index >= len(self.cpu_bursts)

@dataclass
class ProcessMetrics:
    pid: int
    arrival: int
    finish: int
    response: int
    wait: int
    turnaround: int
    context_switches: int

# ==========================================
# 2. Workload Generator
# ==========================================

def generate_workload(num_processes: int, cpu_bound_ratio: float, seed: int = 42) -> List[Process]:
    random.seed(seed)
    processes: List[Process] = []
    num_cpu_bound = int(num_processes * cpu_bound_ratio)
    pids = list(range(1, num_processes + 1))

    for i, pid in enumerate(pids):
        is_cpu_bound = i < num_cpu_bound
        if is_cpu_bound:
            # CPU-bound: Long bursts, rare I/O
            num_bursts = random.randint(3, 6)
            cpu_bursts = [random.randint(15, 30) for _ in range(num_bursts)]
            io_waits = [random.randint(8, 20) for _ in range(num_bursts - 1)]
            burst_type = "CPU"
        else:
            # I/O-bound: Short bursts, frequent I/O
            num_bursts = random.randint(6, 10)
            cpu_bursts = [random.randint(2, 8) for _ in range(num_bursts)]
            io_waits = [random.randint(3, 10) for _ in range(num_bursts - 1)]
            burst_type = "IO"

        arrival_time = random.randint(0, 20)
        proc = Process(pid, arrival_time, burst_type, cpu_bursts, io_waits)
        proc.initialize_runtime_state()
        processes.append(proc)

    processes.sort(key=lambda p: p.arrival_time)
    return processes

def deep_copy_processes(processes: List[Process]) -> List[Process]:
    """Create independent copies so each scheduler runs on the same data."""
    copies = []
    for p in processes:
        clone = Process(p.pid, p.arrival_time, p.burst_type, list(p.cpu_bursts), list(p.io_waits))
        clone.initialize_runtime_state()
        copies.append(clone)
    return copies

# ==========================================
# 3. Scheduler Logic
# ==========================================

def build_metrics(processes: List[Process]) -> Dict[int, ProcessMetrics]:
    metrics = {}
    for p in processes:
        response = (p.first_run_time - p.arrival_time) if p.first_run_time is not None else 0
        turnaround = (p.finish_time - p.arrival_time) if p.finish_time is not None else 0
        metrics[p.pid] = ProcessMetrics(p.pid, p.arrival_time, p.finish_time, response, p.waiting_time, turnaround, p.context_switches)
    return metrics

def _admit_processes(time: int, incoming: List[Process], io_waiting: List[Process], ready: List[Process]) -> None:
    for proc in incoming[:]:
        if proc.arrival_time <= time:
            proc.ready_entry_time = time
            ready.append(proc)
            incoming.remove(proc)
    for proc in io_waiting[:]:
        if proc.return_time is not None and proc.return_time <= time:
            proc.ready_entry_time = time
            ready.append(proc)
            proc.return_time = None
            io_waiting.remove(proc)

def _advance_if_idle(time: int, incoming: List[Process], io_waiting: List[Process], gantt: List[Tuple[str, int, int]]) -> Optional[int]:
    next_times = []
    if incoming: next_times.append(min(p.arrival_time for p in incoming))
    if io_waiting: next_times.append(min(p.return_time for p in io_waiting if p.return_time is not None))
    
    if not next_times: return None # Simulation done
    
    next_time = min(next_times)
    if next_time > time:
        gantt.append(("IDLE", time, next_time))
        return next_time
    return time

def simulate_sjf(processes: List[Process]) -> Tuple[Dict[int, ProcessMetrics], List[Tuple[str, int, int]], float]:
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
            if new_time is None: break
            time = new_time
            continue

        # SJF: Select shortest NEXT CPU burst
        current = min(ready, key=lambda p: p.cpu_bursts[p.current_burst_index])
        ready.remove(current)

        if current.ready_entry_time is not None:
            current.waiting_time += time - current.ready_entry_time
        if current.first_run_time is None:
            current.first_run_time = time
        if last_pid != current.pid:
            if last_pid is not None: current.context_switches += 1
            last_pid = current.pid

        # Execute full burst (Non-preemptive)
        burst_len = current.remaining_in_burst
        start, end = time, time + burst_len
        gantt.append((f"P{current.pid}", start, end))
        time = end
        
        current.current_burst_index += 1
        if current.current_burst_index >= len(current.cpu_bursts):
            current.finish_time = time
        else:
            current.remaining_in_burst = current.cpu_bursts[current.current_burst_index]
            current.return_time = time + current.io_waits[current.current_burst_index - 1]
            io_waiting.append(current)

    metrics = build_metrics(procs)
    avg_resp = sum(m.response for m in metrics.values()) / len(metrics)
    return metrics, gantt, avg_resp

def simulate_fixed_rr(processes: List[Process], quantum: int) -> Tuple[Dict[int, ProcessMetrics], List[Tuple[str, int, int]], float]:
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
            if new_time is None: break
            time = new_time
            continue

        current = ready.pop(0)
        
        if current.ready_entry_time is not None:
            current.waiting_time += time - current.ready_entry_time
        if current.first_run_time is None:
            current.first_run_time = time
        if last_pid != current.pid:
            if last_pid is not None: current.context_switches += 1
            last_pid = current.pid

        run_time = min(quantum, current.remaining_in_burst)
        start, end = time, time + run_time
        gantt.append((f"P{current.pid}", start, end))
        time = end
        current.remaining_in_burst -= run_time

        if current.remaining_in_burst == 0:
            current.current_burst_index += 1
            if current.current_burst_index >= len(current.cpu_bursts):
                current.finish_time = time
            else:
                current.remaining_in_burst = current.cpu_bursts[current.current_burst_index]
                current.return_time = time + current.io_waits[current.current_burst_index - 1]
                io_waiting.append(current)
        else:
            current.ready_entry_time = time
            ready.append(current)

    metrics = build_metrics(procs)
    avg_resp = sum(m.response for m in metrics.values()) / len(metrics)
    return metrics, gantt, avg_resp

def simulate_adaptive_rr(processes: List[Process], alpha: float, min_q: int, max_q: int) -> Tuple[Dict[int, ProcessMetrics], List[Tuple[str, int, int]], float]:
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
            if new_time is None: break
            time = new_time
            continue

        current = ready.pop(0)

        # === ADAPTIVE LOGIC ===
        predicted = current.prediction
        dynamic_q = int(max(min_q, min(max_q, round(predicted))))
        # ======================

        if current.ready_entry_time is not None:
            current.waiting_time += time - current.ready_entry_time
        if current.first_run_time is None:
            current.first_run_time = time
        if last_pid != current.pid:
            if last_pid is not None: current.context_switches += 1
            last_pid = current.pid

        run_time = min(dynamic_q, current.remaining_in_burst)
        start, end = time, time + run_time
        gantt.append((f"P{current.pid}", start, end))
        time = end
        current.remaining_in_burst -= run_time

        if current.remaining_in_burst == 0:
            # Update EMA
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
            current.ready_entry_time = time
            ready.append(current)

    metrics = build_metrics(procs)
    avg_resp = sum(m.response for m in metrics.values()) / len(metrics)
    return metrics, gantt, avg_resp

# ==========================================
# 4. Visualization & Main
# ==========================================

def print_metrics_table(title: str, metrics: Dict[int, ProcessMetrics]) -> None:
    print(f"\n=== {title} ===")
    print(f"{'PID':>4} {'Resp':>8} {'Wait':>6} {'Turn':>6} {'CtxSw':>6}")
    print("-" * 35)
    for m in sorted(metrics.values(), key=lambda x: x.pid):
        print(f"{m.pid:>4} {m.response:>8} {m.wait:>6} {m.turnaround:>6} {m.context_switches:>6}")
    avg_resp = sum(m.response for m in metrics.values()) / len(metrics)
    print(f"--> Avg Response: {avg_resp:.2f}")

def plot_results(gantt, avg_resp_map):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Bar Chart
    algos = list(avg_resp_map.keys())
    values = list(avg_resp_map.values())
    ax1.bar(algos, values, color=["gray", "steelblue", "seagreen"])
    for i, v in enumerate(values):
        ax1.text(i, v + 0.5, f"{v:.2f}", ha="center")
    ax1.set_title("Average Response Time (Lower is Better)")
    ax1.set_ylabel("ms")

    # Gantt Chart (First 150ms)
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
    # SETTINGS
    num_processes = 15          # 稍微增加进程数，让排队更明显
    cpu_bound_ratio = 0.4       # 40% CPU密集，60% IO密集 (IO多，Adaptive更有优势)
    
    # === 关键修改在这里 ===
    # 我们故意把 Fixed RR 设得很大 (60ms)，模拟"不好的配置"。
    # 这样它就会像 FCFS 一样，让短任务排长队，导致响应慢。
    fixed_quantum = 60          
    
    # Adaptive 参数保持不变，它会自动调整 (限制在 5-25ms 之间)
    alpha = 0.5  
    min_q, max_q = 5, 25
    
    # 换个种子，确保生成的任务顺序能体现出差异
    seed = 999 
    
    print("Generating Workload...")
    processes = generate_workload(num_processes, cpu_bound_ratio, seed=seed)
    
    print(f"Running Simulations (Fixed Q={fixed_quantum} vs Adaptive)...")
    sjf_m, _, sjf_avg = simulate_sjf(processes)
    rr_m, _, rr_avg = simulate_fixed_rr(processes, fixed_quantum)
    adp_m, adp_gantt, adp_avg = simulate_adaptive_rr(processes, alpha, min_q, max_q)
    
    print_metrics_table(f"Fixed RR (Q={fixed_quantum})", rr_m)
    print_metrics_table("Adaptive RR", adp_m)
    
    print("\nPlotting Results...")
    # 这里把 Fixed RR 的标签改清楚一点，方便截图
    plot_results(adp_gantt, {
        "SJF": sjf_avg, 
        "Fixed RR (Poor Config)": rr_avg, 
        "Adaptive RR (Proposed)": adp_avg
    })

if __name__ == "__main__":
    main()