# SUGAR Repo Architecture Overview

Tài liệu này mô tả kiến trúc tổng quan của repo, vai trò của từng thư mục/file chính, và mối quan hệ giữa các thành phần trong pipeline train/inference.

## 1. Repo này giải quyết bài toán gì?

Repo triển khai SUGAR: một pipeline humanoid loco-manipulation gồm 3 tầng:

1. `refiner`: policy RL đầu tiên học bám motion/object interaction cơ bản.
2. `tracker`: policy RL thứ hai học tốt hơn từ rollout của `refiner`.
3. `generator`: policy imitation/diffusion sinh action sequence từ quan sát trạng thái mức thấp.

Về mặt code, repo chia thành 2 package lớn:

- `source/sugar_rl`: phần môi trường IsaacLab, task registration, RL training, rollout.
- `source/sugar_il`: phần imitation learning cho `generator`, dataset zarr, model diffusion, workspace train.

Hai package này được nối với nhau bằng dữ liệu rollout:

- `sugar_rl` rollout ra `raw_npz`
- script xử lý convert sang dataset
- `sugar_il` đọc dataset đó để train `generator`


## 2. Điểm vào chính của repo

### 2.1. Entry scripts ở root

- `train.sh`
  - Entry point cho full training pipeline.
  - Chạy tuần tự:
    - train `refiner`
    - rollout `refiner`
    - process rollout thành RL dataset cho `tracker`
    - train `tracker`
    - rollout `tracker`
    - process rollout thành IL dataset cho `generator`
    - train `generator`

- `inference.sh`
  - Entry point cho suy luận/demo.
  - Gọi `scripts/sugar_rl/play.py` với checkpoint `tracker` và `generator`.

- `README.md`
  - Tài liệu public của dự án.
  - Mô tả cách cài, cách chạy train/inference, và trạng thái release.

### 2.2. Scripts Python

- `scripts/sugar_rl/train.py`
  - Entry point train RL bằng IsaacLab + RSL-RL.
  - Khởi tạo simulator/app, parse task config, tạo env từ gym registry, tạo runner và train.

- `scripts/sugar_rl/play.py`
  - Entry point chạy env với checkpoint đã train.
  - Dùng cho:
    - inference/demo
    - rollout dữ liệu để phục vụ stage sau

- `scripts/sugar_rl/process_refiner_rollout.py`
  - Convert rollout của `refiner` thành RL dataset folder cho `tracker`.

- `scripts/sugar_rl/process_tracker_rollout.py`
  - Convert rollout của `tracker` thành zarr dataset cho `generator`.
  - Đây là điểm nối quan trọng giữa `sugar_rl` và `sugar_il`.

- `scripts/sugar_il/train.py`
  - Entry point train `generator` bằng Hydra config.
  - Nạp workspace class từ config và chạy training loop.

- `scripts/list_envs.py`
  - Import và liệt kê các env `Sugar-*` đã được register.


## 3. Cấu trúc thư mục cấp cao

### 3.1. `assets/`

Chứa asset tĩnh cho tài liệu, hiện tại chủ yếu là hình minh họa phương pháp:

- `assets/method_overview.png`

Không tham gia trực tiếp vào runtime train/inference.

### 3.2. `scripts/`

Chứa entrypoint thực thi.

- `scripts/sugar_rl/`
  - train/play/process rollout cho nhánh RL
- `scripts/sugar_il/`
  - train cho nhánh imitation learning

### 3.3. `source/`

Chứa mã nguồn chính dưới dạng editable package:

- `source/sugar_rl`
- `source/sugar_il`


## 4. Kiến trúc phần `sugar_rl`

`sugar_rl` là phần mô phỏng, task definition, train RL, rollout dữ liệu.

### 4.1. `source/sugar_rl/sugar_rl/tasks`

Đây là trung tâm của phần RL.

#### `tasks/locomanip/mdp/`

Nhóm file định nghĩa logic MDP:

- `commands.py`
  - Phần quan trọng nhất của repo RL.
  - Chứa:
    - `MotionLoader`: load motion/object/contact từ dataset folder
    - `MotionCommand`: điều phối motion hiện tại, teacher motion, rollout, eval mode, generator integration
  - Đây là nơi env đọc motion data và áp đặt command/reference cho humanoid.

- `observations.py`
  - Định nghĩa observation mà agent nhìn thấy.

- `rewards.py`
  - Định nghĩa reward terms.

- `terminations.py`
  - Điều kiện kết thúc episode.

- `events.py`
  - Event/randomization/reset logic phụ trợ.

Nói ngắn gọn:

- `commands.py` quyết định “env đang theo motion nào, object target nào, có generator hay teacher hay không”
- `observations.py`, `rewards.py`, `terminations.py` là 3 mặt còn lại của MDP

#### `tasks/locomanip/agents/`

Chứa config cho thuật toán RL:

- `rsl_rl_ppo_cfg.py`
  - PPO config chuẩn
- `rsl_rl_bcppo_cfg.py`
  - Biến thể PPO có teacher / behavior cloning component

Các config này được `scripts/sugar_rl/train.py` nạp vào runner.

#### `tasks/locomanip/robots/g129dof/`

Chứa env config cụ thể cho robot/task.

Phân thành 3 nhóm:

- `train_refiner/`
  - Env config cho giai đoạn train `refiner`
- `train_tracker/`
  - Env config cho giai đoạn train `tracker`
- `inference/`
  - Env config cho giai đoạn inference/play

Mỗi task như `CarryBox`, `KickBox`, `PushBox`, `SitChair`, `StandBottle`, `PickBottle` có file riêng cho từng phase.

Ví dụ:

- `carry_box_refiner_env_cfg.py`
- `carry_box_tracker_env_cfg.py`
- `carry_box_inference_env_cfg.py`

Quan hệ:

- cùng một task
- nhưng khác phase
- sẽ dùng env config khác nhau
- vì command source, reward emphasis, checkpoint integration và rollout behavior khác nhau

### 4.2. `source/sugar_rl/sugar_rl/assets`

Chứa định nghĩa asset cho robot và object:

- `assets/robots/unitree.py`
- `assets/robots/unitree_actuators.py`
- `assets/objects/objects.py`

Các file này cung cấp mô tả robot/object để env config sử dụng.

### 4.3. `source/sugar_rl/sugar_rl/utils`

Helper cho phần RL:

- `parser_cfg.py`
  - Parse env config từ gym task name
- `rsl_rl_bcppo.py`
  - Thuật toán BCPPO custom, được inject vào `rsl_rl.algorithms`


## 5. Kiến trúc phần `sugar_il`

`sugar_il` là phần train `generator` từ dataset rollout đã xử lý.

### 5.1. `source/sugar_il/sugar_il/config/`

Hydra config cho training.

- `train_generator_workspace.yaml`
  - Config tổng cho train generator:
    - optimizer
    - dataloader
    - training loop
    - policy
    - device/log/checkpoint

- `config/task/*.yaml`
  - Config theo task
  - Chỉ ra:
    - dataset path
    - `shape_meta`
    - dataset class
    - env runner class

### 5.2. `source/sugar_il/sugar_il/workspace/`

Chứa training workspace abstraction.

- `base_workspace.py`
  - Base class quản lý checkpoint/output dir/state chung

- `train_generator_workspace.py`
  - Training loop thực tế cho `generator`
  - Các bước chính:
    - instantiate model từ Hydra
    - instantiate dataset
    - build dataloader
    - fit/load normalizer
    - train theo epoch
    - checkpoint/logging

### 5.3. `source/sugar_il/sugar_il/dataset/`

Định nghĩa dataset cho generator.

- `generator_dataset.py`
  - Dataset chính.
  - Đọc zarr dataset với các key như:
    - `obj_pos_b`
    - `obj_ori_b`
    - `joint_pos`
    - `project_gravity`
    - `target_obj_pos_b`
    - `target_obj_ori_b`
    - `action`
    - `last_action`
  - Tách train/val mask, sample sequence, trả về batch `obs` + `action`.

- `base_dataset.py`
  - Base abstraction cho low-dimensional dataset.

### 5.4. `source/sugar_il/sugar_il/policy/`

- `generator.py`
  - Model wrapper cấp cao cho policy generator.
  - Gồm:
    - obs encoder
    - diffusion transformer
    - normalizer
    - loss function
    - inference sampling

### 5.5. `source/sugar_il/sugar_il/model/`

Các thành phần neural network chi tiết:

- `model/encoder/generator_state_encoder.py`
  - Mã hóa low-dimensional observation thành token

- `model/diffusion/transformer_for_action_diffusion.py`
  - Backbone transformer cho action diffusion

- `model/diffusion/ema_model.py`
  - EMA helper

- `model/common/*`
  - utility layers, scheduler, normalizer, tensor helpers

### 5.6. `source/sugar_il/sugar_il/common/`

Hạ tầng phụ trợ cho training/data:

- `streaming_replay_buffer.py`
  - Reader cho zarr dataset

- `replay_buffer.py`
  - buffer abstraction

- `sampler.py`
  - sample sequence cho dataset

- `checkpoint_util.py`
  - top-k checkpoint management

- `json_logger.py`
  - logging ra file json text

- `pytorch_util.py`
  - utility xử lý nested tensor dict

### 5.7. `source/sugar_il/sugar_il/wrapper/`

- `sugar_il_wrapper.py`
  - Cầu nối để phần RL/inference gọi được `generator`
  - Đây là điểm mà `sugar_rl` dùng model `sugar_il` trong runtime

### 5.8. `source/sugar_il/sugar_il/env_runner/`

- `generator_runner.py`
  - Runner interface cho rollout/eval ở nhánh IL
  - Hiện implementation rất mỏng, gần như placeholder


## 6. Quan hệ giữa `sugar_rl` và `sugar_il`

Đây là mối quan hệ quan trọng nhất trong repo.

### 6.1. Hướng dữ liệu

1. `refiner` train trên motion data gốc trong `data/<TASK>`
2. `refiner` rollout ra `raw_npz`
3. `process_refiner_rollout.py` convert rollout thành folder `rl_dataset`
4. `tracker` train trên `rl_dataset`
5. `tracker` rollout ra `raw_npz`
6. `process_tracker_rollout.py` convert rollout thành `il_dataset` dạng zarr
7. `generator` train trên `il_dataset`
8. inference env có thể nạp cả checkpoint `tracker` và `generator`

### 6.2. Hướng phụ thuộc code

- `sugar_rl` không chỉ là producer dữ liệu, nó còn là consumer của `sugar_il`
- Trong `commands.py`, phần RL import:
  - `GeneratorWrapper`
  - `GeneratorObs`
  - `DOWNSAMPLE_RATE`
  từ `sugar_il.wrapper.sugar_il_wrapper`

Điều này có nghĩa:

- lúc train rollout/inference nâng cao, env RL có thể gọi `generator`
- nên `sugar_rl` và `sugar_il` không hoàn toàn tách rời
- chúng nối với nhau cả ở mức dữ liệu lẫn runtime


## 7. Pipeline train end-to-end trong repo

### Stage A: Refiner

File chính:

- `train.sh`
- `scripts/sugar_rl/train.py`
- `source/sugar_rl/sugar_rl/tasks/locomanip/robots/g129dof/train_refiner/*`

Input:

- `data/<TASK>`

Output:

- `outputs/<TASK>_<EXP>/ckpts/refiner.pt`

### Stage B: Refiner rollout -> Tracker dataset

File chính:

- `scripts/sugar_rl/play.py`
- `scripts/sugar_rl/process_refiner_rollout.py`

Input:

- checkpoint `refiner.pt`
- motion data gốc

Output:

- `outputs/.../rollout_datasets/refiner/raw_npz`
- `outputs/.../rollout_datasets/refiner/rl_dataset`

### Stage C: Tracker

File chính:

- `scripts/sugar_rl/train.py`
- `source/sugar_rl/sugar_rl/tasks/locomanip/robots/g129dof/train_tracker/*`

Input:

- `refiner.pt`
- `refiner/rl_dataset`

Output:

- `outputs/<TASK>_<EXP>/ckpts/tracker.pt`

### Stage D: Tracker rollout -> Generator dataset

File chính:

- `scripts/sugar_rl/play.py`
- `scripts/sugar_rl/process_tracker_rollout.py`

Input:

- `tracker.pt`
- `refiner.pt`
- `refiner/rl_dataset`

Output:

- `outputs/.../rollout_datasets/tracker/raw_npz`
- `outputs/.../rollout_datasets/tracker/il_dataset`

### Stage E: Generator

File chính:

- `scripts/sugar_il/train.py`
- `source/sugar_il/sugar_il/workspace/train_generator_workspace.py`
- `source/sugar_il/sugar_il/dataset/generator_dataset.py`
- `source/sugar_il/sugar_il/policy/generator.py`

Input:

- `tracker/il_dataset`

Output:

- `outputs/<TASK>_<EXP>/ckpts/generator.ckpt`


## 8. Pipeline inference

Entry:

- `inference.sh`
- `scripts/sugar_rl/play.py`

Runtime dependency:

- env inference config trong `source/sugar_rl/.../inference/*`
- tracker checkpoint
- generator checkpoint
- motion data trong `data/<TASK>`

Luồng tổng quát:

1. `play.py` parse task và env config
2. env được tạo từ task registry của `sugar_rl`
3. checkpoint `tracker` được nạp vào RL policy
4. nếu env config yêu cầu, `generator` cũng được nạp qua wrapper
5. simulator chạy step-by-step trong IsaacLab


## 9. Các file/folder nên đọc đầu tiên nếu muốn hiểu repo

Nếu mới vào repo, thứ tự đọc nên là:

1. `README.md`
2. `train.sh`
3. `scripts/sugar_rl/train.py`
4. `scripts/sugar_rl/play.py`
5. `source/sugar_rl/sugar_rl/tasks/locomanip/mdp/commands.py`
6. `scripts/sugar_rl/process_tracker_rollout.py`
7. `source/sugar_il/sugar_il/dataset/generator_dataset.py`
8. `source/sugar_il/sugar_il/workspace/train_generator_workspace.py`
9. `source/sugar_il/sugar_il/policy/generator.py`

Lý do:

- `train.sh` cho bức tranh toàn cục
- `commands.py` cho logic runtime quan trọng nhất của env
- `process_tracker_rollout.py` cho format dữ liệu nối RL sang IL
- `generator_dataset.py` và `generator.py` cho phần model/data của stage cuối


## 10. Những gì repo có và chưa có

Repo hiện có:

- complete training pipeline của `refiner`, `tracker`, `generator`
- processed data interface
- inference path với checkpoint

Repo chưa có đầy đủ trong code public:

- pipeline xử lý từ RGB-D human videos thô sang training data đầu vào
- sim-to-sim pipeline hoàn chỉnh theo TODO trong `README.md`

Điểm này quan trọng vì:

- phần đầu vào `data/<TASK>` được giả định là đã tồn tại
- repo train được từ mức “processed motion/object dataset” trở đi
- không phải từ raw human video


## 11. Tóm tắt một câu

Repo này là một hệ thống hai tầng:

- `sugar_rl` chịu trách nhiệm mô phỏng, task, RL training và rollout dữ liệu
- `sugar_il` chịu trách nhiệm train diffusion generator từ rollout đã xử lý

và `train.sh` là orchestration script nối toàn bộ pipeline đó thành một flow end-to-end.
