# Adaptive Time-Slice Round Robin via CPU-Burst Prediction

This project simulates three CPU scheduling algorithms on an identical mixed workload (CPU-bound + I/O-bound processes):

1. Shortest Job First (SJF, non-preemptive baseline)
2. Fixed Round Robin (static quantum)
3. Adaptive Round Robin (dynamic quantum using Exponential Moving Average prediction of CPU bursts)

## Key Idea
Adaptive RR predicts the next CPU burst using EMA:
```
prediction_new = alpha * actual_last_burst + (1 - alpha) * prediction_old
```
The time quantum for a process = `clamp(round(prediction), min_quantum, max_quantum)`.

## Metrics Collected
- Response Time (first run - arrival)
- Waiting Time (time spent ready but not running)
- Turnaround Time (finish - arrival)
- Context Switch count per process

## Visualization
The script outputs:
- Gantt chart of Adaptive RR execution
- Bar chart comparing average response time across SJF, Fixed RR, Adaptive RR

## Project Structure
```
├── src/
│   └── os_simulation.py        # Main simulation script (Adaptive + Fixed + SJF)
├── figures/
│   └── response_time_comparison.png  # Saved plot example
├── requirements.txt
├── README.md
└── (docs, PDFs)
```

## Running
Create / activate a virtual environment (optional) and install dependencies, then run the simulation module:
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
python src/os_simulation.py
```

## Configuration
Inside `src/os_simulation.py` modify:
- `num_processes`
- `cpu_bound_ratio`
- `seed`
- `fixed_quantum`
- `alpha`, `min_q`, `max_q`

## Output
The console prints per-process metrics and averaged statistics; plots open in windows.

## License
Educational use for OS scheduling comparison.
