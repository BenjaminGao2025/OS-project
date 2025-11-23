import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import copy
import matplotlib.pyplot as plt


@dataclass
class Process:
	pid: int
	arrival_time: int
	burst_type: str  # 'CPU' or 'IO'
	cpu_bursts: List[int]
	io_waits: List[int]  # len = len(cpu_bursts) - 1 (no IO after last CPU burst)
	current_burst_index: int = 0
	remaining_in_burst: int = 0
	first_run_time: Optional[int] = None
	finish_time: Optional[int] = None
	waiting_time: int = 0
	ready_entry_time: Optional[int] = None
	context_switches: int = 0
	# Adaptive RR prediction of next full CPU burst (EMA)
	prediction: float = 0.0
	return_time: Optional[int] = None  # When returning from IO

	def initialize(self):
		self.remaining_in_burst = self.cpu_bursts[0]
		# Initial prediction can be first CPU burst length
		self.prediction = float(self.cpu_bursts[0])

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


def generate_workload(num_processes: int, cpu_bound_ratio: float, seed: int = 42) -> List[Process]:
	"""Generate a mixed workload of CPU-bound and IO-bound processes.
	CPU-bound: fewer bursts, long CPU bursts, long IO waits (rare IO)
	IO-bound: more bursts, short CPU bursts, short IO waits (frequent IO)
	cpu_bound_ratio: fraction of processes that are CPU-bound.
	"""
	random.seed(seed)
	processes: List[Process] = []
	num_cpu_bound = int(num_processes * cpu_bound_ratio)
	pids = list(range(1, num_processes + 1))
	for i, pid in enumerate(pids):
		is_cpu = i < num_cpu_bound
		if is_cpu:
			# CPU-bound characteristics
			num_bursts = random.randint(3, 6)
			cpu_bursts = [random.randint(15, 30) for _ in range(num_bursts)]
			io_waits = [random.randint(8, 20) for _ in range(num_bursts - 1)]
			burst_type = 'CPU'
		else:
			# IO-bound characteristics
			num_bursts = random.randint(6, 10)
			cpu_bursts = [random.randint(2, 8) for _ in range(num_bursts)]
			io_waits = [random.randint(3, 10) for _ in range(num_bursts - 1)]
			burst_type = 'IO'
		# Stagger arrivals moderately
		arrival_time = random.randint(0, 20)
		proc = Process(pid=pid, arrival_time=arrival_time, burst_type=burst_type,
					   cpu_bursts=cpu_bursts, io_waits=io_waits)
		proc.initialize()
		processes.append(proc)
	# Sort by arrival time for convenience
	processes.sort(key=lambda p: p.arrival_time)
	return processes


def deep_copy_processes(processes: List[Process]) -> List[Process]:
	new_list = []
	for p in processes:
		new_p = Process(pid=p.pid,
						arrival_time=p.arrival_time,
						burst_type=p.burst_type,
						cpu_bursts=list(p.cpu_bursts),
						io_waits=list(p.io_waits))
		new_p.initialize()
		new_list.append(new_p)
	return new_list


def build_metrics(processes: List[Process]) -> Dict[int, ProcessMetrics]:
	metrics: Dict[int, ProcessMetrics] = {}
	for p in processes:
		response = (p.first_run_time - p.arrival_time) if p.first_run_time is not None else 0
		turnaround = (p.finish_time - p.arrival_time) if p.finish_time is not None else 0
		metrics[p.pid] = ProcessMetrics(
			pid=p.pid,
			arrival=p.arrival_time,
			finish=p.finish_time if p.finish_time is not None else -1,
			response=response,
			wait=p.waiting_time,
			turnaround=turnaround,
			context_switches=p.context_switches
		)
	return metrics


def simulate_sjf(processes: List[Process]) -> Tuple[Dict[int, ProcessMetrics], List[Tuple[str, int, int]], float]:
	procs = deep_copy_processes(processes)
	time = 0
	gantt: List[Tuple[str, int, int]] = []
	ready: List[Process] = []
	incoming = procs[:]
	io_waiting: List[Process] = []
	last_pid: Optional[int] = None
	context_switches = 0

	while True:
		# Move newly arrived processes
		for p in incoming[:]:
			if p.arrival_time <= time:
				p.ready_entry_time = time
				ready.append(p)
				incoming.remove(p)
		# Move IO return processes
		for p in io_waiting[:]:
			if p.return_time is not None and p.return_time <= time:
				p.ready_entry_time = time
				ready.append(p)
				p.return_time = None
				io_waiting.remove(p)

		if not ready:
			# Advance time to next arrival or IO return if exists
			next_times = []
			if incoming:
				next_times.append(min(p.arrival_time for p in incoming))
			if io_waiting:
				next_times.append(min(p.return_time for p in io_waiting if p.return_time is not None))
			if not next_times:
				break  # All done
			next_time = min(next_times)
			if next_time > time:
				# Idle segment
				gantt.append(('IDLE', time, next_time))
				time = next_time
			continue

		# Select process with shortest next CPU burst
		selected = min(ready, key=lambda p: p.cpu_bursts[p.current_burst_index])
		ready.remove(selected)
		# Waiting time accumulation
		if selected.ready_entry_time is not None:
			selected.waiting_time += time - selected.ready_entry_time
		if selected.first_run_time is None:
			selected.first_run_time = time
		if last_pid != selected.pid:
			if last_pid is not None:
				context_switches += 1
			selected.context_switches += 1
			last_pid = selected.pid
		burst_len = selected.remaining_in_burst  # full burst
		start = time
		end = time + burst_len
		gantt.append((f'P{selected.pid}', start, end))
		time = end
		# Finish this burst
		selected.current_burst_index += 1
		if selected.current_burst_index >= len(selected.cpu_bursts):
			selected.finish_time = time
		else:
			# Prepare next burst
			selected.remaining_in_burst = selected.cpu_bursts[selected.current_burst_index]
			# Schedule IO
			io_wait = selected.io_waits[selected.current_burst_index - 1]
			selected.return_time = time + io_wait
			io_waiting.append(selected)

	metrics = build_metrics(procs)
	avg_response = sum(m.response for m in metrics.values()) / len(metrics)
	return metrics, gantt, avg_response


def simulate_fixed_rr(processes: List[Process], quantum: int) -> Tuple[Dict[int, ProcessMetrics], List[Tuple[str, int, int]], float]:
	procs = deep_copy_processes(processes)
	time = 0
	gantt: List[Tuple[str, int, int]] = []
	ready: List[Process] = []
	incoming = procs[:]
	io_waiting: List[Process] = []
	last_pid: Optional[int] = None
	context_switches = 0

	while True:
		# Admit new arrivals
		for p in incoming[:]:
			if p.arrival_time <= time:
				p.ready_entry_time = time
				ready.append(p)
				incoming.remove(p)
		for p in io_waiting[:]:
			if p.return_time is not None and p.return_time <= time:
				p.ready_entry_time = time
				ready.append(p)
				p.return_time = None
				io_waiting.remove(p)

		if not ready:
			next_times = []
			if incoming:
				next_times.append(min(p.arrival_time for p in incoming))
			if io_waiting:
				next_times.append(min(p.return_time for p in io_waiting if p.return_time is not None))
			if not next_times:
				break
			next_time = min(next_times)
			if next_time > time:
				gantt.append(('IDLE', time, next_time))
				time = next_time
			continue

		current = ready.pop(0)
		# Update waiting time
		if current.ready_entry_time is not None:
			current.waiting_time += time - current.ready_entry_time
		if current.first_run_time is None:
			current.first_run_time = time
		if last_pid != current.pid:
			if last_pid is not None:
				context_switches += 1
			current.context_switches += 1
			last_pid = current.pid
		run_time = min(quantum, current.remaining_in_burst)
		start = time
		end = time + run_time
		gantt.append((f'P{current.pid}', start, end))
		time = end
		current.remaining_in_burst -= run_time
		if current.remaining_in_burst == 0:
			# Burst finished
			current.current_burst_index += 1
			if current.current_burst_index >= len(current.cpu_bursts):
				current.finish_time = time
			else:
				current.remaining_in_burst = current.cpu_bursts[current.current_burst_index]
				io_wait = current.io_waits[current.current_burst_index - 1]
				current.return_time = time + io_wait
				io_waiting.append(current)
		else:
			# Preempted - requeue
			current.ready_entry_time = time
			ready.append(current)

	metrics = build_metrics(procs)
	avg_response = sum(m.response for m in metrics.values()) / len(metrics)
	return metrics, gantt, avg_response


def simulate_adaptive_rr(processes: List[Process], alpha: float, min_quantum: int, max_quantum: int) -> Tuple[Dict[int, ProcessMetrics], List[Tuple[str, int, int]], float]:
	procs = deep_copy_processes(processes)
	time = 0
	gantt: List[Tuple[str, int, int]] = []
	ready: List[Process] = []
	incoming = procs[:]
	io_waiting: List[Process] = []
	last_pid: Optional[int] = None
	context_switches = 0

	while True:
		# Admit arrivals
		for p in incoming[:]:
			if p.arrival_time <= time:
				p.ready_entry_time = time
				ready.append(p)
				incoming.remove(p)
		for p in io_waiting[:]:
			if p.return_time is not None and p.return_time <= time:
				p.ready_entry_time = time
				ready.append(p)
				p.return_time = None
				io_waiting.remove(p)

		if not ready:
			next_times = []
			if incoming:
				next_times.append(min(p.arrival_time for p in incoming))
			if io_waiting:
				next_times.append(min(p.return_time for p in io_waiting if p.return_time is not None))
			if not next_times:
				break
			next_time = min(next_times)
			if next_time > time:
				gantt.append(('IDLE', time, next_time))
				time = next_time
			continue

		current = ready.pop(0)
		# Dynamic quantum via EMA prediction (clamped)
		predicted = current.prediction
		dynamic_q = int(max(min_quantum, min(max_quantum, round(predicted))))
		if current.ready_entry_time is not None:
			current.waiting_time += time - current.ready_entry_time
		if current.first_run_time is None:
			current.first_run_time = time
		if last_pid != current.pid:
			if last_pid is not None:
				context_switches += 1
			current.context_switches += 1
			last_pid = current.pid
		run_time = min(dynamic_q, current.remaining_in_burst)
		start = time
		end = time + run_time
		gantt.append((f'P{current.pid}', start, end))
		time = end
		current.remaining_in_burst -= run_time
		if current.remaining_in_burst == 0:
			# Completed a full CPU burst -> update EMA prediction for NEXT burst
			actual = current.cpu_bursts[current.current_burst_index]
			# EMA: prediction_new = alpha * actual_last_burst + (1 - alpha) * last_prediction
			current.prediction = alpha * actual + (1 - alpha) * current.prediction
			current.current_burst_index += 1
			if current.current_burst_index >= len(current.cpu_bursts):
				current.finish_time = time
			else:
				current.remaining_in_burst = current.cpu_bursts[current.current_burst_index]
				io_wait = current.io_waits[current.current_burst_index - 1]
				current.return_time = time + io_wait
				io_waiting.append(current)
		else:
			# Preempted - remain in same burst, prediction unchanged until full burst completes
			current.ready_entry_time = time
			ready.append(current)

	metrics = build_metrics(procs)
	avg_response = sum(m.response for m in metrics.values()) / len(metrics)
	return metrics, gantt, avg_response


def print_metrics_table(title: str, metrics: Dict[int, ProcessMetrics]):
	print(f"\n=== {title} ===")
	header = f"{'PID':>4} {'Arrival':>7} {'Finish':>7} {'Response':>8} {'Wait':>6} {'Turnaround':>11} {'CtxSw':>6}"
	print(header)
	print('-' * len(header))
	for m in sorted(metrics.values(), key=lambda x: x.pid):
		print(f"{m.pid:>4} {m.arrival:>7} {m.finish:>7} {m.response:>8} {m.wait:>6} {m.turnaround:>11} {m.context_switches:>6}")
	avg_resp = sum(m.response for m in metrics.values()) / len(metrics)
	avg_wait = sum(m.wait for m in metrics.values()) / len(metrics)
	avg_turn = sum(m.turnaround for m in metrics.values()) / len(metrics)
	print(f"Averages -> Response: {avg_resp:.2f}, Wait: {avg_wait:.2f}, Turnaround: {avg_turn:.2f}")


def plot_gantt(gantt: List[Tuple[str, int, int]], title: str):
	fig, ax = plt.subplots(figsize=(10, 3))
	y = 0
	for segment in gantt:
		label, start, end = segment
		ax.barh(y, end - start, left=start, height=0.6, align='center')
		ax.text((start + end) / 2, y, label, va='center', ha='center', color='white', fontsize=8)
	ax.set_xlabel('Time')
	ax.set_ylabel('CPU')
	ax.set_title(title)
	ax.set_yticks([])
	ax.grid(axis='x', linestyle='--', alpha=0.4)
	plt.tight_layout()
	plt.show()


def plot_response_comparison(avg_resp: Dict[str, float]):
	fig, ax = plt.subplots(figsize=(6, 4))
	algos = list(avg_resp.keys())
	values = [avg_resp[a] for a in algos]
	ax.bar(algos, values, color=['gray', 'steelblue', 'seagreen'])
	for i, v in enumerate(values):
		ax.text(i, v + 0.5, f"{v:.2f}", ha='center')
	ax.set_ylabel('Average Response Time')
	ax.set_title('Scheduler Response Time Comparison')
	plt.tight_layout()
	plt.show()


def main():
	# Configuration
	num_processes = 12
	cpu_bound_ratio = 0.5  # 50% CPU-bound, 50% IO-bound
	seed = 2025
	fixed_quantum = 10
	alpha = 0.6  # EMA smoothing factor
	min_q = 5
	max_q = 20

	processes = generate_workload(num_processes=num_processes, cpu_bound_ratio=cpu_bound_ratio, seed=seed)

	# Run simulations
	sjf_metrics, sjf_gantt, sjf_avg_resp = simulate_sjf(processes)
	rr_metrics, rr_gantt, rr_avg_resp = simulate_fixed_rr(processes, fixed_quantum)
	adaptive_metrics, adaptive_gantt, adaptive_avg_resp = simulate_adaptive_rr(processes, alpha, min_q, max_q)

	# Print tables
	print_metrics_table('SJF (Non-Preemptive Baseline)', sjf_metrics)
	print_metrics_table(f'Fixed Round Robin (Q={fixed_quantum})', rr_metrics)
	print_metrics_table(f'Adaptive Round Robin (EMA alpha={alpha})', adaptive_metrics)

	# Plot Gantt for Adaptive RR (core algorithm)
	plot_gantt(adaptive_gantt, 'Adaptive Round Robin Gantt Chart')

	# Plot comparison (Average Response Time)
	avg_resp_map = {
		'SJF': sjf_avg_resp,
		'Fixed RR': rr_avg_resp,
		'Adaptive RR': adaptive_avg_resp
	}
	plot_response_comparison(avg_resp_map)


if __name__ == '__main__':
	main()

