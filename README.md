<div align="center">
  <h1 align="center"> OpenTrack </h1>
  <h3 align="center"> GALBOT · Tsinghua </h3>
<!--   <p align="center">
    <a href="README.md"> English </a> | <a href="README_zh.md">中文</a>
  </p>     -->

:page_with_curl:[Paper](https://arxiv.org/abs/2509.13833) | :house:[Website](https://zzk273.github.io/Any2Track/)


This repository is the official implementation of OpenTrack, an open-source humanoid motion tracking codebase that uses MuJoCo JAX for simulation and supports multi-GPU parallel training.
</div>

# News 🚩
[May 28, 2026] Real-world deployment code released.

[May 23, 2026] AnyAdapter training code released. **Now you can finetune your base tracker to endow it with dynamics adaptivity.**

[May 18, 2026] LAFAN1 generalist v2 released. **A single policy can accurately track all 40/40 motions in LaFAN1, including highly challenging ones.**

[April 24, 2026] DAgger training code released. **Now you can distill your teacher trackers into a generalist student tracker.**

[November 30, 2025] LAFAN1 generalist v1 released. **Now you can track cartwheel, kungfu, fall and getup, and many other motions within a single policy.**

[September 19, 2025] Simple Domain Randomization released.

[September 19, 2025] Tracking codebase released.

# TODOs

- [x] Release motion tracking codebase
- [x] Release domain randomization
- [x] Release pretrained LAFAN1 generalist checkpoints
- [x] Release DAgger code
- [x] Release AnyAdapter
- [x] Release real-world deployment code

# Prepare

1. Clone the repository:
    ```shell
    git clone git@github.com:GalaxyGeneralRobotics/OpenTrack.git
    ```

2. Create a virtual environment and install dependencies:
    ```shell
    uv sync -i https://pypi.org/simple 
    ```

3. Create a `.env` file in the project directory with the following content:
    ```bash
    export GLI_PATH=<your_project_path>
    export WANDB_PROJECT=<your_project_name>
    export WANDB_ENTITY=<your_wandb_entity>
    export WANDB_API_KEY=<your_wandb_api_key>
    ```

4. Download the [mocap data](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets/tree/main/Lafan1/mocap/UnitreeG1) and put them under `storage/data/mocap/`. Thanks for the retargeting motions of LAFAN1 dataset from [LocoMuJoCo](https://github.com/robfiras/loco-mujoco/)! 

    The file structure should be like:

    ```
    storage                   
    ├── assets
    │   ├── mujoco_menagerie
    │   └── unitree_g1
    ├── data
    │   ├── hfield            
    │   │   └── terrain.npz
    │   └── mocap
    │       └── lafan1   
    │           └── UnitreeG1
    ├── logs
    │   ├── dagger      # dagger student checkpoints
    │   └── track       # tracking checkpoints  
    ├── training_configs
    │   └── dagger
    │       └── demo.json    
    └── ...
    ```

5. Initialize assets
    ```shell
    # First, initialize the environment
    source .venv/bin/activate; source .env;
    # Then, initialize the assets
    python track_mj/app/mj_playground_init.py
    ```

# Usage

## Initialize environment

1. Initialize the MuJoCo environment:
  ```shell
   source .venv/bin/activate; source .env;
  ```

## Optional MJX-Warp acceleration

Training uses the JAX MJX backend by default. To try the NVIDIA Warp backend for MJX simulation:

```shell
uv sync
export OPENTRACK_MJX_IMPL=warp

# Defaults match the MuJoCo Warp Unitree G1 benchmark and can be raised if contacts overflow.
export OPENTRACK_MJX_WARP_NCONMAX_PER_ENV=48
export OPENTRACK_MJX_WARP_NJMAX=192
```

`OPENTRACK_MJX_WARP_NCONMAX_PER_ENV` is multiplied by the training `num_envs` to size MJX-Warp's
global contact buffer. You can override the final buffer directly with `OPENTRACK_MJX_WARP_NACONMAX`.

## Play pretrained checkpoints

1. Download pretrained checkpoints and configs from [checkpoints and configs](https://drive.google.com/drive/folders/1wDL4Chr6sGQiCx1tbvhf9DowN73cP_PF?usp=drive_link), and put them under `storage/logs/dagger/`. Visualization results of LAFAN1 generalist v2: [videos](https://drive.google.com/drive/folders/1wDL4Chr6sGQiCx1tbvhf9DowN73cP_PF?usp=drive_link).

2. Run the evaluation script:

    ```shell
    python -m track_mj.eval.dagger.mj_onnx_video --task G1TrackingGeneral --exp_name <your_exp_name> [--use_viewer] [--use_renderer] [--play_ref_motion]
    ```

As of **May 18, 2026**, we have open-sourced **LAFAN1 generalist v1 and v2**. Generalist v1 is daggered from four teachers and can track most motions. However, it fails on some extremely challenging motions (mainly because some teachers fail to track these motions). Generalist v2 is daggered from eight more carefully designed teachers. Each teacher policy successfully tracks all motions in its motion cluster. The student policy perfectly inherits the capabilities of different teachers and successfully tracked all 40/40 motions.

The checkpoints are trained with simple domain randomization (DR). You may try to deploy it on a Unitree G1 robot with the deployment code.

## Train from scratch

### Prepare the motion data

  Prepare motions separately before training or evaluation. Runtime loading is read-only: it requires complete state fields at the task's control frequency (50 Hz by default) and never extends, resamples, recalculates or saves the source files. Configurations with `recalculate_vel_in_reference_motion=True`, including the released generalist configurations, require reference velocities saved by this preparation step even when the input is already complete at 50 Hz.

  Put the source `.npz` files under `<source-root>/<dataset-name>/UnitreeG1/` and select the motions in the task's `reference_traj_config.name`. Preparation uses the existing trajectory functions to align the sampling frequency, complete state features and recalculate reference velocities when configured. It writes new artifacts under `<output-root>/<dataset-name>/UnitreeG1/`, preserves source metadata and records the source path, optional external publication reference and processing operations. Source and output roots must differ; existing output artifacts are refused.

  ```shell
  python scripts/process_motion/preprocess_motion.py \
    --task G1TrackingGeneral \
    --source-root /path/to/raw-mocap \
    --output-root /path/to/prepared-mocap \
    --num-batches 1 --batch-idx 0
  ```

  Set `--source-reference` to the published dataset version or source ID when available. Publication pins source versions externally; preparation records that reference without computing file identities.

  Use `--flat-source` if the source root contains the `.npz` files directly. To partition processing, set the same `--num-batches N` for each process and supply each `--batch-idx` from `0` through `N-1` once. All batches write distinct artifacts into the same new output tree.

  Point the training or evaluation configuration at the prepared tree using the existing `reference_traj_config` field:

  ```python
  env_config.reference_traj_config.root = "/path/to/prepared-mocap"
  # For example: /path/to/prepared-mocap/lafan1/UnitreeG1/walk1_subject1.npz
  ```

  For a saved configuration, set `env_config.reference_traj_config.root` to that same directory before constructing the environment. If the field is omitted, runtime loading uses `storage/data/mocap/`.

  Smoothing is disabled by default. Add `--smooth-start-end` to explicitly prepend a transition from the default pose; use a separate output root for this variant. The prepared artifact records the smoothing parameters and the number of added prefix frames.

  An inverse-kinematics solver generates a smooth stepping motion from the default pose to the first motion frame. The solver uses QP optimization (OSQP) with foot position/orientation constraints and CoM balance constraints. One or two foot steps are automatically chosen based on the foot-placement gap between default pose and the pose of the first frame.

  Here is a comparison of the original motion and the smoothed motion:

  <table>
    <tr>
      <th>Original Motion</th>
      <th>Smoothed Motion</th>
    </tr>
    <tr>
      <td><img src="./storage/assets/demo/original.gif" width="100%"></td>
      <td><img src="./storage/assets/demo/smoothed.gif" width="100%"></td>
    </tr>
  </table>


### Train the specialist teachers:
  ```shell
   python -m track_mj.learning.train.train_ppo_track --task G1TrackingGeneralDR --exp_name <your_exp_name>
  ```
  If you want to modify the training configuration, you should edit the config file associated with the corresponding environment. For example, for the command above, you should modify the `g1_tracking_general_dr_task_config` in the `OpenTrack/track_mj/envs/g1_tracking/train/g1_env_tracking_general_dr.py`.

### Train the generalist student:
  For training on one GPU:
  ```shell
   python -m track_mj.learning.train.train_dagger --task G1TrackingGeneralDR --exp_name <your_exp_name> --dagger-config-path <your_config_path>
  ```
  For training on multiple GPUs (8 for example):
  ```shell
   torchrun --nproc_per_node=8 -m track_mj.learning.train.train_dagger \
    --task G1TrackingGeneralDR \
    --exp-name <your_exp_name> \
    --dagger-config-path <your_config_path> \
    --use_ddp
  ```

  For an example DAgger config, please refer to `storage/training_configs/dagger/demo.json`.

  DAgger reports batch preparation, the pre-update rank barrier (`2.rank_skew_wait`), and
  optimizer updates as separate timing sections. Their execution order is unchanged. These are
  host wall-clock intervals with no added device synchronization; the barrier interval includes
  its existing synchronization cost and should not be read as a pure measure of rank arrival skew.
  Earlier logs included all three sections in `2.optimizer_step`.

  `--training.num-envs` is the global requested count, divided across ranks with integer division.
  The startup log reports the effective global count, world size, and learning rate. The learning
  rate is used as configured, with no runtime scaling by world size.

  In our experiments, we found that our specialist-to-generalist training framework is a flexible, controllable, and scalable approach for building a general tracker. Different teachers can be flexibly assigned to different categories of motions to maximize tracking quality. As long as you have a specialist teacher, you can seamlessly distill its capabilities into a generalist student. The capabilities of different teachers can be perfectly inherited by the student. The student’s performance improves significantly with more motion data, more teachers, longer training time, and larger model capacity.

### Train the adapter:
  After training a base policy in an environment with no (or only limited) dynamics randomization, you can further train an adapter in an environment with dynamics randomization enabled (or with a larger degree of randomization).
  ```shell
   python -m track_mj.learning.train.train_adapter \
     --task G1TrackingGeneralDR \
     --load-exp-name <your_pretrained_tracker_exp_name> \
     --exp-name <your_adapter_exp_name>
  ```

## Evaluate the model

Programmatic evaluation of the native tracking and DAgger training environments
can use `step_no_autoreset(state, action, trajectory_data)`. Their matching
`VmapWrapper` and domain-randomization vector wrapper expose this method and
`physical_data(state)`. Use the vector wrapper directly so action repetition
cannot hide a terminal boundary. Evaluation retains the true terminal state and
copies the input info and metrics containers before advancing. Training `step`
keeps its native automatic-reset policy and RNG behavior.

`physical_data` runs native forward on owned endpoint data. Its refreshed
kinematics do not replace policy observations or continuation data. The interface
declares measured substep contact forces unavailable. Native subsequent-step
numerical reproducibility needs separate qualification; CPU tests use fake physics
and do not establish native engine equivalence. Reset isolation corrections are
outside this interface.

### Evaluate the specialist teachers

  ```shell
  # First, convert the tracker checkpoint to ONNX
  python -m track_mj.eval.tracking.export_onnx --task G1TrackingGeneral --exp_name <your_exp_name>
  
  # Next, run the evaluation script
  python -m track_mj.eval.tracking.mj_onnx_video --task G1TrackingGeneral --exp_name <your_exp_name> [--use_viewer] [--use_renderer] [--play_ref_motion]
  ```

### Evaluate the generalist student:
  ```shell
   # First, convert the Torch model checkpoint to the ONNX format
   python -m track_mj.eval.dagger.torch2onnx --ckpt_dir "storage/logs/dagger/<your_exp_name>" 
   # This writes ONNX to storage/logs/dagger/<your_exp_name>/checkpoints/model.onnx
   # Next, run the evaluation script
   python -m track_mj.eval.dagger.mj_onnx_video --task G1TrackingGeneral --exp_name <your_exp_name> [--use_viewer] [--use_renderer] [--play_ref_motion]
  ```

### Evaluate the adapter
  ```shell
  # First, convert the checkpoint to ONNX
   python -m track_mj.eval.adapter.export_onnx --task G1TrackingGeneral --exp_name <your_adapter_exp_name>

   # Next, run the evaluation script
   python -m track_mj.eval.adapter.mj_onnx_video --task G1TrackingGeneralDR --exp_name <your_adapter_exp_name> [--use_viewer] [--use_renderer] [--play_ref_motion]
  ```

## Real-world deployment

Please refer to [`deploy/README_deployment.md`](deploy/README_deployment.md).

# Acknowledgement

This repository is build upon `jax`, `brax`, `loco-mujoco`, and `mujoco_playground`.

If you find this repository helpful, please cite our work:

```bibtex
@misc{zhang2025trackmotionsdisturbances,
      title={Track Any Motions under Any Disturbances}, 
      author={Zhikai Zhang and Jun Guo and Chao Chen and Jilong Wang and Chenghuai Lin and Yunrui Lian and Han Xue and Zhenrong Wang and Maoqi Liu and Jiangran Lyu and Huaping Liu and He Wang and Li Yi},
      year={2025},
      eprint={2509.13833},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2509.13833}, 
}
```
