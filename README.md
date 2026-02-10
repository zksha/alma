<h1 align="center">
  <!-- Project Logo -->
  <img src="misc/arts_pic.png" width="200" /><br>
  <b>Learning to Continually Learn via Meta-learning Agentic Memory Designs</b><br>
</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge"></a>
  <a href="https://yimingxiong.me/alma"><img src="https://img.shields.io/badge/-Website-%238D6748?style=for-the-badge&logo=Website&logoColor=white"></a>
  <a href="https://arxiv.org/pdf/2602.07755"><img src="https://img.shields.io/badge/arXiv-2602.07755-b31b1b.svg?logo=arxiv&style=for-the-badge"></a>
  <!-- <a href="https://twitter.com/your_handle"><img src="https://img.shields.io/badge/twitter-%230077B5.svg?&style=for-the-badge&logo=twitter&logoColor=white&color=00acee"></a> -->
</p>

## Overview
This project introduces **ALMA** (**A**utomated meta-**L**earning of **M**emory designs for **A**gentic systems), a framework that meta-learns memory designs to replace hand-engineered memory designs, therefore minimizing human effort and enabling agentic systems to be continual learners across diverse domains. ALMA employs a Meta Agent that searches over memory designs expressed as executable code in an open-ended manner, theoretically allowing the discovery of arbitrary memory designs, including database schemas as well as their retrieval and update mechanisms.

<p align="center">
  <img src="misc/main_workflow.png" width="90%" /><br>
  <em>Open-ended Exploration Process of ALMA.
</em>
</p>
The Meta Agent first ideates and proposes a plan by reflecting on the code and evaluation logs of the sampled memory design. It then implements the plan by programming the new design in code. Finally, it verifies the correctness of the new memory design and evaluates it with an agentic system. The evaluated memory design is subsequently added to the memory design archive for future sampling.

<div style="background-color: #f6f8fa; border: 1px solid #d1d5da; border-radius: 6px; padding: 16px;">
  <span style="font-size: 0.95em; color: #586069;">
    The learned memory designs and logs required to give the results presented in our work are available at:
    <a href="https://drive.google.com/drive/folders/1D6l1SwZDbyk4iQQ4ajCVgXQ2PSBhPQMD?usp=sharing" target="_blank" style="color: #0366d6; text-decoration: none; font-weight: bold;">
      Learning Memory Designs & Logs
    </a>
  </span>
</div>

## Key Features

* 🧠 **Automatic Memory Design Discovery** - ALMA learns memory designs rather than hand-engineered designs
* 🎯 **Domain Adaptation** - Automatically specializes memory designs for diverse sequential decision-making tasks
* 🔬 **Comprehensive Evaluation** - Tested across four domains: AlfWorld, TextWorld, BabaisAI, and MiniHack
* 📈 **Superior Performance** - Outperforms state-of-the-art human-designed baselines across all benchmarks
* ⚡ **Cost Efficiency** - Learned designs are more efficient than most human-designed baselines

---

## Setup
```bash
# Cloning project
git clone https://github.com/zksha/alma.git
cd ./alma

# Create environment
conda create -n alma python=3.11
conda activate alma

# Install dependencies
pip install -r requirements.txt

# Then add your API key to the `.env` file:
# .env
OPENAI_API_KEY=your_openai_api_key_here
```

---

## Running Experiments

### Setup Testing Environments

> [!WARNING]
> This repository executes **model-generated code** as part of the memory design search process.
> While the code goes through a verification and debugging stage, dynamically generated code may behave unpredictably.
> **Use at your own risk**, ideally inside a sandboxed or isolated environment.

#### For ALFWorld
```bash
# Build ALFWorld Docker image
cd envs_docker/alfworld
bash image_build.sh
```

#### For TextWorld, BabaisAI, and MiniHack (BALROG)
```bash
# Build BALROG Docker image (used for TextWorld, BabaisAI, and MiniHack)
cd envs_docker/BALROG
bash image_build.sh
```

> [!TIP]
> The BALROG image is shared across TextWorld, BabaisAI, and MiniHack domains, so you only need to build it once.

### Learning of Memory Designs

To run the learning process that discovers new memory designs:
```bash
python run_main.py \
    --rollout_type batched \
    --meta_model gpt-5 \
    --execution_model gpt-5-nano \
    --batch_max_update_concurrent 10 \
    --batch_max_retrieve_concurrent 10 \
    --task_type alfworld \
    --status train \
    --train_size 30
```

**Parameters:**

<div align="center">
<table>
  <thead>
    <tr>
      <th align="left">Parameter</th>
      <th align="left">Description</th>
      <th align="center">Options</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>--rollout_type</code></td>
      <td>Execution strategy for evaluations, <code>sequential</code> allows both update and retrieval in deployment phase</td>
      <td align="center"><code>batched</code>, <code>sequential</code></td>
    </tr>
    <tr>
      <td><code>--meta_model</code></td>
      <td>Model used by the meta agent to propose memory designs</td>
      <td align="center"><code>gpt-5</code>, <code>gpt-4.1</code>, etc.</td>
    </tr>
    <tr>
      <td><code>--execution_model</code></td>
      <td>Model used by agents during task execution</td>
      <td align="center"><code>gpt-5-mini/medium</code>, <code>gpt-5-nano/low</code>, <code>gpt-4o-mini</code>, etc.</td>
    </tr>
    <tr>
      <td><code>--batch_max_update_concurrent</code></td>
      <td>Max concurrent memory update operations</td>
      <td align="center">Integer (e.g., <code>10</code>)</td>
    </tr>
    <tr>
      <td><code>--batch_max_retrieve_concurrent</code></td>
      <td>Max concurrent memory retrieval operations</td>
      <td align="center">Integer (e.g., <code>10</code>)</td>
    </tr>
    <tr>
      <td><code>--task_type</code></td>
      <td>Domain to run experiments on</td>
      <td align="center"><code>alfworld</code>, <code>textworld</code>, <code>babaisai</code>, <code>minihack</code></td>
    </tr>
    <tr>
      <td><code>--status</code></td>
      <td>Execution mode</td>
      <td align="center"><code>train</code>, <code>eval_in_distribution</code>, <code>eval_out_of_distribution</code></td>
    </tr>
    <tr>
      <td><code>--train_size</code></td>
      <td>Number of training tasks</td>
      <td align="center">Integer (e.g., <code>30</code>, <code>50</code>, <code>100</code>)</td>
    </tr>
    <tr>
      <td><code>--memo_SHA</code></td>
      <td>Memory designs' SHA, provided for testing</td>
      <td align="center">String (e.g., <code>g-memory</code>, <code>53cee295</code>)</td>
    </tr>
  </tbody>
</table>
</div>


> [!TIP]
> Example configurations for different domains is in `training.sh` and `testing.sh`.
> Learned memory design should be store in `memo_archive`.
> Learning logs should be store in `logs`.

### Adding New Domains

To extend the benchmark to a new domain:

1. Build up image for new domain.
2. Adding prompts, configs, and `{env_name}_envs.py` in `envs archive`.
3. Adding task descriptions for meta agent in `meta_agent_prompt.py`.
4. Register container and name for the new benchmark in `eval_in_container.py`.
4. Run the meta agent to discover specialized memory designs.
5. Evaluate results against baseline memory designs.
---

## Results
<p align="left">
  <b>
    Our learned memory designs consistently outperform state-of-the-art human-designed memory across all benchmarks.
  </b><br>
  <span style="color:#555">
    Numbers indicate <i>overall success rate</i> (higher is better). Improvements are relative to the no-memory baseline.
  </span>
</p>
<div align="center">
<table>
  <thead>
    <tr>
      <th align="left">FM in Agentic System</th>
      <th align="center">GPT-5-nano / low</th>
      <th align="center">GPT-5-mini / medium</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>No Memory</b></td>
      <td align="center">6.1</td>
      <td align="center">41.1</td>
    </tr>
    <tr>
      <td colspan="3"><i>Manual Memory Designs</i></td>
    </tr>
    <tr>
      <td>Trajectory Retrieval</td>
      <td align="center">8.6 (+2.5)</td>
      <td align="center">48.6 (+7.5)</td>
    </tr>
    <tr>
      <td>Reasoning Bank</td>
      <td align="center">7.5 (+1.4)</td>
      <td align="center">40.1 (−1.0)</td>
    </tr>
    <tr>
      <td>Dynamic Cheatsheet</td>
      <td align="center">7.2 (+1.1)</td>
      <td align="center">46.5 (+5.4)</td>
    </tr>
    <tr>
      <td>G-Memory</td>
      <td align="center">7.7 (+1.6)</td>
      <td align="center">46.0 (+4.9)</td>
    </tr>
    <tr>
      <td colspan="3"><i>Learned Memory Design</i></td>
    </tr>
    <tr>
      <td><b>Our Method</b></td>
      <td align="center"><b>12.3 (+6.2)</b></td>
      <td align="center"><b>53.9 (+12.8)</b></td>
    </tr>
  </tbody>
</table>
</div>

**Key findings:**
- Learned designs adapt to domain-specific requirements automatically
- Better performance scaling with memory size
- Faster learning under task distribution shifts
- Lower computational costs compared to human-designed baselines

---

## Acknowledgements

This research was supported by the Vector Institute, the Canada CIFAR AI Chairs program, a grant from Schmidt Futures, an NSERC Discovery Grant, and a generous donation from Rafael Cosman. Resources used in preparing this research were provided, in part, by the Province of Ontario, the Government of Canada through CIFAR, and companies sponsoring the Vector Institute (https://vectorinstitute.ai/partnerships/current-partners/). Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the sponsors.

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
