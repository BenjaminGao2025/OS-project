"""
============================================================================
Adaptive Time-Slice Round Robin via CPU-Burst Prediction
Using Exponential Moving Average (EMA)
============================================================================

【PROJECT OBJECTIVE】
Compare three CPU scheduling algorithms on the same mixed workload:
  1. SJF (Shortest Job First) - Non-preemptive, theoretical optimal baseline
  2. Fixed Round Robin - Static time quantum
  3. Adaptive Round Robin - Our proposed method with dynamic time quantum

【KEY INNOVATION】
Traditional RR uses a fixed time quantum that cannot adapt to different task types.
Our method uses EMA to predict each process's next CPU burst length,
then dynamically adjusts the time quantum:
  - Short tasks → small quantum → complete quickly
  - Long tasks → moderate quantum → reduce unnecessary context switches

【FILE STRUCTURE】
  1. Data Structures (Process, ProcessMetrics)
  2. Workload Generator (generate_workload)
  3. Three Scheduler Implementations (simulate_sjf, simulate_fixed_rr, simulate_adaptive_rr)
  4. Visualization & Main Function (plot_results, main)
============================================================================
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt


# ============================================================================
# PART 1: DATA STRUCTURES
# ============================================================================

@dataclass
class Process:
    """
    Process Class - Models a process in the operating system
    
    【STATIC ATTRIBUTES - Set at creation, never change】
    - pid: Process ID (unique identifier)
    - arrival_time: When the process enters the system
    - burst_type: "CPU" (CPU-bound) or "IO" (I/O-bound)
    - cpu_bursts: List of CPU burst durations, e.g., [10, 15, 8] means 3 CPU phases
    - io_waits: List of I/O wait times between CPU bursts
    
    【DYNAMIC ATTRIBUTES - Change during simulation】
    - current_burst_index: Which CPU burst we're currently on (0-indexed)
    - remaining_in_burst: Time left in current CPU burst
    - first_run_time: When process first got CPU (for response time calculation)
    - finish_time: When process completed entirely
    - waiting_time: Total time spent waiting in ready queue
    - ready_entry_time: When process most recently entered ready queue
    - context_switches: Number of context switches for this process
    - prediction: EMA predicted value (used by Adaptive RR)
    - return_time: When process will return from I/O
    """
    pid: int
    arrival_time: int
    burst_type: str
    cpu_bursts: List[int]
    io_waits: List[int]
    
    # Runtime state - must be reset before each simulation
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
        Reset runtime state - Called before each new scheduling algorithm
        Ensures each algorithm starts with identical initial conditions for fair comparison
        """
        self.current_burst_index = 0
        self.remaining_in_burst = self.cpu_bursts[0]  # Start with first CPU burst
        self.first_run_time = None
        self.finish_time = None
        self.waiting_time = 0
        self.ready_entry_time = None
        self.context_switches = 0
        # Initial EMA prediction = first CPU burst length (cold start)
        self.prediction = float(self.cpu_bursts[0])
        self.return_time = None

    def is_complete(self) -> bool:
        """Check if process has finished all CPU bursts"""
        return self.current_burst_index >= len(self.cpu_bursts)


@dataclass
class ProcessMetrics:
    """
    Process Performance Metrics - For final statistics and comparison
    
    - response: Response Time = first_run_time - arrival_time
                (Time from arrival to first CPU access; lower is better)
    - wait: Waiting Time = Total time spent in ready queue
    - turnaround: Turnaround Time = finish_time - arrival_time
                  (Total time from arrival to completion)
    - context_switches: Number of context switches
    """
    pid: int
    arrival: int
    finish: int
    response: int
    wait: int
    turnaround: int
    context_switches: int


# ============================================================================
# PART 2: WORKLOAD GENERATOR
# ============================================================================

def generate_workload(num_processes: int, cpu_bound_ratio: float, seed: int = 42) -> List[Process]:
    """
    Generate Mixed Workload - Contains both CPU-bound and I/O-bound processes
    
    Parameters:
    - num_processes: Total number of processes
    - cpu_bound_ratio: Fraction of CPU-bound processes (0.0-1.0)
    - seed: Random seed for reproducibility (CRITICAL for fair comparison!)
    
    【CPU-BOUND PROCESS CHARACTERISTICS】
    - Fewer CPU bursts (3-6)
    - Longer CPU burst duration (15-30ms)
    - Longer I/O waits (8-20ms) - rare I/O operations
    
    【I/O-BOUND PROCESS CHARACTERISTICS】
    - More CPU bursts (6-10)
    - Shorter CPU burst duration (2-8ms)
    - Shorter I/O waits (3-10ms) - frequent I/O operations
    """
    random.seed(seed)  # Fix random seed for reproducibility
    processes: List[Process] = []
    num_cpu_bound = int(num_processes * cpu_bound_ratio)
    pids = list(range(1, num_processes + 1))

    for i, pid in enumerate(pids):
        is_cpu_bound = i < num_cpu_bound
        
        if is_cpu_bound:
            # ===== CPU-BOUND PROCESS =====
            num_bursts = random.randint(3, 6)
            cpu_bursts = [random.randint(15, 30) for _ in range(num_bursts)]
            io_waits = [random.randint(8, 20) for _ in range(num_bursts - 1)]
            burst_type = "CPU"
        else:
            # ===== I/O-BOUND PROCESS =====
            num_bursts = random.randint(6, 10)
            cpu_bursts = [random.randint(2, 8) for _ in range(num_bursts)]
            io_waits = [random.randint(3, 10) for _ in range(num_bursts - 1)]
            burst_type = "IO"

        # Arrival times distributed randomly between 0-20ms
        arrival_time = random.randint(0, 20)
        
        proc = Process(pid, arrival_time, burst_type, cpu_bursts, io_waits)
        proc.initialize_runtime_state()
        processes.append(proc)

    # Sort by arrival time to simulate processes entering the system over time
    processes.sort(key=lambda p: p.arrival_time)
    return processes


def deep_copy_processes(processes: List[Process]) -> List[Process]:
    """
    Deep Copy Process List - Each scheduler needs its own independent copy
    
    Why deep copy?
    Schedulers modify process runtime state (remaining_in_burst, waiting_time, etc.).
    If three schedulers share the same process objects, later schedulers would see
    states modified by earlier ones. Deep copy ensures each scheduler starts fresh.
    """
    copies = []
    for p in processes:
        clone = Process(
            p.pid, 
            p.arrival_time, 
            p.burst_type, 
            list(p.cpu_bursts),  # Must copy lists too
            list(p.io_waits)
        )
        clone.initialize_runtime_state()
        copies.append(clone)
    return copies


# ============================================================================
# PART 3: SCHEDULER CORE LOGIC
# ============================================================================

def build_metrics(processes: List[Process]) -> Dict[int, ProcessMetrics]:
    """
    Extract performance metrics from completed processes
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
    【HELPER FUNCTION】Add eligible processes to ready queue
    
    Two types of processes need to be added:
    1. Newly arrived processes (arrival_time <= current time)
    2. Processes returning from I/O (return_time <= current time)
    """
    # Check for newly arrived processes
    for proc in incoming[:]:  # Use slice copy since we're removing during iteration
        if proc.arrival_time <= time:
            proc.ready_entry_time = time  # Record when entered ready queue
            ready.append(proc)
            incoming.remove(proc)
    
    # Check for processes returning from I/O
    for proc in io_waiting[:]:
        if proc.return_time is not None and proc.return_time <= time:
            proc.ready_entry_time = time
            ready.append(proc)
            proc.return_time = None
            io_waiting.remove(proc)


def _advance_if_idle(time: int, incoming: List[Process], io_waiting: List[Process], gantt: List[Tuple[str, int, int]]) -> Optional[int]:
    """
    【HELPER FUNCTION】When CPU is idle, fast-forward time to next event
    
    When ready queue is empty, CPU is idle.
    Find the next event (new arrival or I/O completion) and advance time.
    Also record this idle period in the Gantt chart.
    
    Returns:
    - None: Simulation complete (no more processes)
    - int: New time after fast-forward
    """
    next_times = []
    if incoming:
        next_times.append(min(p.arrival_time for p in incoming))
    if io_waiting:
        next_times.append(min(p.return_time for p in io_waiting if p.return_time is not None))
    
    if not next_times:
        return None  # No more processes, simulation ends
    
    next_time = min(next_times)
    if next_time > time:
        gantt.append(("IDLE", time, next_time))  # Record idle segment
        return next_time
    return time


# -----------------------------------------------------------------------------
# SCHEDULER 1: SJF (Shortest Job First) - Non-preemptive
# -----------------------------------------------------------------------------

def simulate_sjf(processes: List[Process]) -> Tuple[Dict[int, ProcessMetrics], List[Tuple[str, int, int]], float]:
    """
    Shortest Job First Scheduling Algorithm (Non-preemptive SJF)
    
    【ALGORITHM CONCEPT】
    When CPU becomes free, select the process with the shortest NEXT CPU burst.
    Once execution begins, it runs to completion (non-preemptive).
    
    【ADVANTAGES】
    - Theoretically achieves optimal average waiting time
    
    【DISADVANTAGES】
    - Requires knowing CPU burst lengths in advance (impossible in reality)
    - May cause starvation for long jobs
    - Non-preemptive, poor for interactive systems
    
    Used here as a theoretical optimal baseline to benchmark other algorithms.
    """
    procs = deep_copy_processes(processes)
    time = 0
    gantt = []           # Gantt chart data: [(process_name, start_time, end_time), ...]
    ready = []           # Ready queue
    incoming = procs[:]  # Processes not yet arrived
    io_waiting = []      # Processes waiting for I/O
    last_pid = None      # Last running process (for context switch counting)

    while True:
        # Step 1: Admit arrived and I/O-completed processes to ready queue
        _admit_processes(time, incoming, io_waiting, ready)
        
        # Step 2: If ready queue empty, fast-forward time
        if not ready:
            new_time = _advance_if_idle(time, incoming, io_waiting, gantt)
            if new_time is None:
                break  # All processes complete
            time = new_time
            continue

        # Step 3: 【SJF CORE】Select process with shortest next CPU burst
        current = min(ready, key=lambda p: p.cpu_bursts[p.current_burst_index])
        ready.remove(current)

        # Step 4: Update statistics
        if current.ready_entry_time is not None:
            current.waiting_time += time - current.ready_entry_time
        if current.first_run_time is None:
            current.first_run_time = time  # Record first run time
        if last_pid != current.pid:
            if last_pid is not None:
                current.context_switches += 1
            last_pid = current.pid

        # Step 5: Execute entire CPU burst (non-preemptive)
        burst_len = current.remaining_in_burst
        start, end = time, time + burst_len
        gantt.append((f"P{current.pid}", start, end))
        time = end
        
        # Step 6: Handle state transition after burst completion
        current.current_burst_index += 1
        if current.current_burst_index >= len(current.cpu_bursts):
            # All CPU bursts complete, process terminates
            current.finish_time = time
        else:
            # More CPU bursts remaining, enter I/O wait
            current.remaining_in_burst = current.cpu_bursts[current.current_burst_index]
            current.return_time = time + current.io_waits[current.current_burst_index - 1]
            io_waiting.append(current)

    metrics = build_metrics(procs)
    avg_resp = sum(m.response for m in metrics.values()) / len(metrics)
    return metrics, gantt, avg_resp


# -----------------------------------------------------------------------------
# SCHEDULER 2: Fixed Round Robin - Static Time Quantum
# -----------------------------------------------------------------------------

def simulate_fixed_rr(processes: List[Process], quantum: int) -> Tuple[Dict[int, ProcessMetrics], List[Tuple[str, int, int]], float]:
    """
    Fixed Round Robin Scheduling Algorithm
    
    【ALGORITHM CONCEPT】
    All processes are queued in arrival order. Each process runs for at most
    'quantum' milliseconds. If the current CPU burst completes within the
    time slice, the process moves to I/O or terminates. If the time slice
    expires before completion, the process is preempted and moved to queue end.
    
    【PARAMETERS】
    - quantum: Fixed time slice length (milliseconds)
    
    【ADVANTAGES】
    - Fair: Every process gets CPU time
    - Predictable response time
    
    【DISADVANTAGES】
    - Quantum too large: Degenerates to FCFS, long tasks block short ones
    - Quantum too small: Context switch overhead becomes excessive
    - Cannot adapt to different task types
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

        # 【Fixed RR】Take process from queue front (FIFO order)
        current = ready.pop(0)
        
        # Update statistics
        if current.ready_entry_time is not None:
            current.waiting_time += time - current.ready_entry_time
        if current.first_run_time is None:
            current.first_run_time = time
        if last_pid != current.pid:
            if last_pid is not None:
                current.context_switches += 1
            last_pid = current.pid

        # 【KEY】Run time = min(quantum, remaining burst time)
        run_time = min(quantum, current.remaining_in_burst)
        start, end = time, time + run_time
        gantt.append((f"P{current.pid}", start, end))
        time = end
        current.remaining_in_burst -= run_time

        if current.remaining_in_burst == 0:
            # Current CPU burst complete
            current.current_burst_index += 1
            if current.current_burst_index >= len(current.cpu_bursts):
                current.finish_time = time
            else:
                current.remaining_in_burst = current.cpu_bursts[current.current_burst_index]
                current.return_time = time + current.io_waits[current.current_burst_index - 1]
                io_waiting.append(current)
        else:
            # Time slice expired but burst incomplete → preempt, move to queue end
            current.ready_entry_time = time
            ready.append(current)

    metrics = build_metrics(procs)
    avg_resp = sum(m.response for m in metrics.values()) / len(metrics)
    return metrics, gantt, avg_resp


# -----------------------------------------------------------------------------
# SCHEDULER 3: Adaptive Round Robin - Dynamic Time Quantum (OUR PROPOSED METHOD)
# -----------------------------------------------------------------------------

def simulate_adaptive_rr(processes: List[Process], alpha: float, min_q: int, max_q: int) -> Tuple[Dict[int, ProcessMetrics], List[Tuple[str, int, int]], float]:
    """
    Adaptive Round Robin Scheduling Algorithm via EMA
    
    【CORE INNOVATION - THIS IS OUR PROPOSED METHOD!】
    Uses Exponential Moving Average (EMA) to predict each process's next CPU burst,
    then dynamically adjusts that process's time quantum based on the prediction.
    
    【EMA FORMULA】
    prediction_new = α × actual_last_burst + (1 - α) × prediction_old
    
    Where:
    - α (alpha): Smoothing factor controlling weight of most recent actual value
      - High α (e.g., 0.8): Adapts faster to changes, but may overfit to noise
      - Low α (e.g., 0.2): Smoother/more stable, but slower to adapt
    
    【TIME QUANTUM CALCULATION】
    dynamic_quantum = clamp(round(prediction), min_q, max_q)
    
    - min_q: Minimum quantum to prevent excessive context switching
    - max_q: Maximum quantum to prevent other processes from waiting too long
    
    【WHY THIS WORKS】
    1. I/O-bound processes: Short historical bursts → low prediction → small quantum
       → Complete quickly and release CPU
    2. CPU-bound processes: Long historical bursts → high prediction → larger quantum
       → Reduce unnecessary preemption overhead
    
    【PARAMETERS】
    - alpha: EMA smoothing factor (0.0-1.0)
    - min_q: Minimum time quantum
    - max_q: Maximum time quantum
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

        # ╔════════════════════════════════════════════════════════════════╗
        # ║  【CORE ALGORITHM】Calculate dynamic quantum based on EMA      ║
        # ║                                                                ║
        # ║  predicted = This process's historical prediction              ║
        # ║              (initialized to first CPU burst length)           ║
        # ║  dynamic_q = Clamped to [min_q, max_q] range                   ║
        # ╚════════════════════════════════════════════════════════════════╝
        predicted = current.prediction
        dynamic_q = int(max(min_q, min(max_q, round(predicted))))

        # Update statistics
        if current.ready_entry_time is not None:
            current.waiting_time += time - current.ready_entry_time
        if current.first_run_time is None:
            current.first_run_time = time
        if last_pid != current.pid:
            if last_pid is not None:
                current.context_switches += 1
            last_pid = current.pid

        # Execute
        run_time = min(dynamic_q, current.remaining_in_burst)
        start, end = time, time + run_time
        gantt.append((f"P{current.pid}", start, end))
        time = end
        current.remaining_in_burst -= run_time

        if current.remaining_in_burst == 0:
            # ╔════════════════════════════════════════════════════════════════╗
            # ║  【UPDATE EMA】After burst completion, update prediction       ║
            # ║                                                                ║
            # ║  Formula: prediction = α × actual + (1-α) × old_prediction    ║
            # ╚════════════════════════════════════════════════════════════════╝
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
            # Time slice expired, move to queue end
            current.ready_entry_time = time
            ready.append(current)

    metrics = build_metrics(procs)
    avg_resp = sum(m.response for m in metrics.values()) / len(metrics)
    return metrics, gantt, avg_resp


# ============================================================================
# PART 4: VISUALIZATION & MAIN FUNCTION
# ============================================================================

def print_metrics_table(title: str, metrics: Dict[int, ProcessMetrics]) -> None:
    """
    Print Process Performance Metrics Table
    
    Column descriptions:
    - PID: Process ID
    - Resp: Response Time - From arrival to first run
    - Wait: Waiting Time - Total time in ready queue
    - Turn: Turnaround Time - From arrival to completion
    - CtxSw: Context Switch count
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
    Generate two visualizations:
    1. Bar Chart: Compare average response time across three algorithms
    2. Gantt Chart: Show Adaptive RR execution flow (first 150ms)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # ===== Chart 1: Average Response Time Comparison Bar Chart =====
    algos = list(avg_resp_map.keys())
    values = list(avg_resp_map.values())
    colors = ["gray", "steelblue", "seagreen"]
    ax1.bar(algos, values, color=colors)
    for i, v in enumerate(values):
        ax1.text(i, v + 0.5, f"{v:.2f}", ha="center")
    ax1.set_title("Average Response Time (Lower is Better)")
    ax1.set_ylabel("ms")

    # ===== Chart 2: Adaptive RR Gantt Chart (first 150ms) =====
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
    Main Function - Configure parameters, run simulations, output results
    """
    # ==================== PARAMETER CONFIGURATION ====================
    num_processes = 15          # Total number of processes
    cpu_bound_ratio = 0.4       # 40% CPU-bound, 60% I/O-bound
    
    # Fixed RR uses large time slice (60ms) to simulate "poor configuration"
    # This makes it behave like FCFS, causing short tasks to wait long
    fixed_quantum = 60
    
    # Adaptive RR parameters
    alpha = 0.5                 # EMA smoothing factor
    min_q, max_q = 5, 25        # Time quantum range [5ms, 25ms]
    
    seed = 999                  # Random seed for reproducibility
    # ==================================================================
    
    print("Generating Workload...")
    processes = generate_workload(num_processes, cpu_bound_ratio, seed=seed)
    
    print(f"Running Simulations (Fixed Q={fixed_quantum} vs Adaptive)...")
    
    # Run all three schedulers
    sjf_m, _, sjf_avg = simulate_sjf(processes)
    rr_m, _, rr_avg = simulate_fixed_rr(processes, fixed_quantum)
    adp_m, adp_gantt, adp_avg = simulate_adaptive_rr(processes, alpha, min_q, max_q)
    
    # Print results
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
