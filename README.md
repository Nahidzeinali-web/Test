# Detecting Robot Failures by Learning How a Robot Normally Moves

Master's thesis project on the **RoboAI UR5e dataset**.

**Main question:** can a model that has only seen a robot working normally tell us when something goes wrong?

**Idea:** a model learns to predict the robot's next moves. When the robot moves in a way the model did not expect (a bump, a freeze, slow drift, a faulty sensor), it raises an alarm.

## Research questions

| | Question | Status |
|---|---|---|
| RQ1 | How accurately can the robot's future motion be predicted? | answered |
| RQ2 | Can prediction errors detect failures, how early, and which method works best? | answered |
| RQ3 | Can extra information reduce false alarms? | open (the dataset's commands only repeat the recorded movement) |

## Results so far

- Learned models predict the hand position 10 steps ahead with an error of **16–18 mm**. Simple rules are off by **62–68 mm**.
- **Fast failures** (bump, sensor noise) are easy: a simple rule already scores 0.98 (1.0 = perfect).
- **Slow failures** (freeze, drift) need a learned model: **0.80** versus 0.37 for the best simple rule.

## Files

| File | What it is |
|---|---|
| `dataset_review_and_project.ipynb` | **Start here.** Explains the dataset and the project step by step |
| `option1_motion_anomaly_detection.ipynb` | All experiments: training, testing and comparing 7 methods |
| `processed/robot_motion.npz` | Robot numbers extracted from the dataset (4 MB) |
| `results/` | Result tables (CSV) created by the experiments notebook |
| `dataset/` | Dataset description files; the data files themselves are not in Git |

## How to run

1. Install Python 3.10 or newer, then the libraries:
   ```
   pip install -r requirements.txt
   ```
   For a GPU, install PyTorch with CUDA from <https://pytorch.org> first. A GPU is optional; everything runs on a normal laptop in a few minutes.

2. Open the notebooks in VS Code or Jupyter and run them from top to bottom.

The experiments only need `processed/robot_motion.npz`, which is already included. The full dataset (about 24 GB) is only needed to show camera images or to re-create that file. To get it, download it from <https://huggingface.co/datasets/Rouhis/RoboAI_UR5e> and put the files from `robo_ai_u_r5e/1.0.0/` into the `dataset/` folder.

## Dataset

Rouhiainen, J. (2025). *RoboAI UR5e Dataset.* RoboAI Research Center, Satakunta University of Applied Sciences. License: CC BY 4.0.
