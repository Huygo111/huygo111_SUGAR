# Training Pipeline

Tài liệu này mô tả pipeline training end-to-end của repo SUGAR.

## 1. Tổng quan

Pipeline training của repo này gồm 3 stage chính:

1. `refiner`
2. `tracker`
3. `generator`

Toàn bộ orchestration nằm trong file [train.sh](/home/huytn14/Documents/VinMotion/huygo111_SUGAR/train.sh:1).

Luồng tổng quát:

```text
data/<TASK>
-> train refiner
-> rollout refiner
-> process rollout thành rl_dataset
-> train tracker
-> rollout tracker
-> process rollout thành il_dataset
-> train generator
```

## 2. Input ban đầu

Pipeline bắt đầu từ:

```text
data/<TASK>
```

Ví dụ:

- `data/CarryBox`
- `data/KickBox`
- `data/PushBox`

Mỗi task chứa nhiều trajectory sample, mỗi sample có:

- `robot_50hz.npz`
- `obj_motion_global_50hz.pkl`
- `contact_labels_50hz.npy`

Đây là processed trajectory dataset đầu vào, không phải raw RGB-D video.

## 3. Stage 1: Train `refiner`

Script được gọi trong [train.sh](/home/huytn14/Documents/VinMotion/huygo111_SUGAR/train.sh:10):

```bash
python scripts/sugar_rl/train.py \
    --task "Sugar-G129dof-${TASK_NAME}-Refiner" \
    --motion_folder "data/${TASK_NAME}"
```

Ý nghĩa:

- `refiner` là policy RL đầu tiên
- nó học trực tiếp từ motion/object trajectory gốc trong `data/<TASK>`

Code chính tham gia:

- `scripts/sugar_rl/train.py`
- `source/sugar_rl/sugar_rl/tasks/locomanip/robots/g129dof/train_refiner/*`
- `source/sugar_rl/sugar_rl/tasks/locomanip/mdp/*`

Input:

```text
data/<TASK>
```

Output chính:

```text
outputs/<TASK>_<EXP>/ckpts/refiner.pt
```

## 4. Rollout là gì?

**Rollout** là quá trình chạy một policy (model đã train) trong Isaac Sim để thu thập các trajectory mới.

Cụ thể:

- Simulator chạy ở **50 Hz**
- Policy nhận observation từ simulator, output action, simulator bước tiếp
- Dữ liệu từng bước được thu thập và lưu vào file `.npz`
- Mỗi rollout là một **cửa sổ cố định** gồm `rollout_window_length` bước (~100 steps = 2 giây)
- Nhiều môi trường song song có thể rollout cùng lúc

**Mỗi file `.npz` raw rollout chứa ~50 trường dữ liệu ở 50 Hz:**

Trạng thái robot:
- `joint_pos`, `joint_vel` — vị trí và vận tốc 29 khớp
- `root_pos_w`, `root_quat_w` — vị trí/hướng torso trong world frame
- `root_lin_vel_b`, `root_ang_vel_b` — vận tốc trong body frame
- `project_gravity` — vector trọng lực trong frame của robot
- `body_pos_w`, `body_quat_w` — vị trí/hướng từng body trong world

Object:
- `obj_pos_w`, `obj_quat_w` — vị trí/hướng object trong world
- `obj_pos_b`, `obj_quat_b` — vị trí/hướng object tương đối theo torso
- `target_obj_pos_b`, `target_obj_quat_b` — goal pose của object

Reference command (từ motion file gốc, dùng làm nhãn):
- `ref_joint_pos` — joint target
- `ref_root_lin_vel_b`, `ref_root_ang_vel_b` — velocity target
- `ref_contact_label` — trạng thái contact mong muốn
- `action` — action thực sự được thực thi

Contact:
- `hands_contact_label` / `foot_contact_label` / `sit_contact_label` — tùy task

Tên file rollout có dạng:
```text
motion_{motion_id}_env_{env_id}_t{start}-{end}_idx_{idx}.npz
```

---

## 5. Stage 2: Rollout `refiner`

Sau khi train xong `refiner`, repo không dùng data gốc trực tiếp để train `tracker`.

Thay vào đó, nó chạy policy `refiner` trong simulator để sinh rollout mới.

Script được gọi trong [train.sh](/home/huytn14/Documents/VinMotion/huygo111_SUGAR/train.sh:21):

```bash
python scripts/sugar_rl/play.py \
    --task "Sugar-G129dof-${TASK_NAME}-Refiner-Rollout" \
    --checkpoint "outputs/${TASK_NAME}_${EXP_NAME}/ckpts/refiner.pt" \
    --rollout_dir "outputs/${TASK_NAME}_${EXP_NAME}/rollout_datasets/refiner/raw_npz" \
    --motion_folder "data/${TASK_NAME}"
```

Ý nghĩa:

- dùng checkpoint `refiner.pt`
- chạy policy trong env rollout
- sinh dữ liệu trajectory mới

Input:

```text
outputs/<TASK>_<EXP>/ckpts/refiner.pt
data/<TASK>
```

Output:

```text
outputs/<TASK>_<EXP>/rollout_datasets/refiner/raw_npz
```

Format:

- nhiều file `.npz`
- mỗi file là một cửa sổ rollout được save từ simulator

## 6. Stage 3: Process rollout của `refiner` thành `rl_dataset`

Script được gọi trong [train.sh](/home/huytn14/Documents/VinMotion/huygo111_SUGAR/train.sh:30):

```bash
python scripts/sugar_rl/process_refiner_rollout.py \
    --data_dir "outputs/${TASK_NAME}_${EXP_NAME}/rollout_datasets/refiner" \
    --task_name "${TASK_NAME}"
```

Ý nghĩa:

- convert rollout thô `raw_npz`
- về format motion dataset mà env RL có thể load lại

Input:

```text
outputs/.../rollout_datasets/refiner/raw_npz
```

Output:

```text
outputs/.../rollout_datasets/refiner/rl_dataset
```

### `rl_dataset` là gì?

`rl_dataset` là kết quả chuyển đổi raw rollout (nhiều file `.npz` lộn xộn) về đúng format mà RL environment có thể load — cùng cấu trúc với `data/<TASK>` gốc.

**Tại sao cần bước này?**

Raw rollout lưu ~50 trường dữ liệu mỗi step (body states, velocities, contacts...), không đúng format mà motion loader của RL env cần. Bước process tái cấu trúc lại thành 3 file chuẩn.

**Xử lý gì trong bước này?**

1. **Stabilize initial frames (tùy chọn):** Tìm frame đầu tiên mà robot đứng yên (lin_vel < 0.05 m/s, ang_vel < 0.1 rad/s trong 25 frame liên tiếp) và thay thế toàn bộ frame trước đó bằng trạng thái đứng yên đó — giúp policy học từ điểm khởi đầu sạch.
2. **Giữ nguyên 50 Hz:** Không downsample, vì RL env cần control ở tần số gốc.
3. **Tái cấu trúc thành 3 file:**

Mỗi sample trong `rl_dataset` có 3 file quen thuộc:

- `robot_50hz.npz` — chứa `joint_pos (T,29)`, `joint_vel (T,29)`, `body_pos_w (T,nb,3)`, `body_quat_w (T,nb,4)`, `body_lin_vel_w`, `body_ang_vel_w`
- `obj_motion_global_50hz.pkl` — chứa `obj_trans (T,3)`, `obj_rot (T,3,3)` (ma trận rotation), `obj_lin_vel (T,3)`, `obj_ang_vel (T,3)`
- `contact_labels_50hz.npy` — shape `(T,)`, nhãn contact theo từng task (tay / chân / hông)

Cấu trúc thư mục:
```text
rl_dataset/
  data_005_003_t0/
    robot_50hz.npz
    obj_motion_global_50hz.pkl
    contact_labels_50hz.npy
  data_005_003_t100/
    ...
  stable_frame_records.txt
```

Tức là rollout mới của `refiner` được biến về cùng kiểu format với `data/<TASK>` gốc — nên `tracker` có thể train với cùng code, chỉ trỏ `motion_folder` khác đi.

## 7. Stage 4: Train `tracker`

Script được gọi trong [train.sh](/home/huytn14/Documents/VinMotion/huygo111_SUGAR/train.sh:35):

```bash
python scripts/sugar_rl/train.py \
    --task "Sugar-G129dof-${TASK_NAME}-Tracker" \
    --teacher_ckpt "outputs/${TASK_NAME}_${EXP_NAME}/ckpts/refiner.pt" \
    --motion_folder "outputs/${TASK_NAME}_${EXP_NAME}/rollout_datasets/refiner/rl_dataset" \
    --teacher_motion_folder "data/${TASK_NAME}"
```

Ý nghĩa:

- `tracker` là policy RL thứ hai
- nó học từ rollout mới sinh bởi `refiner`
- đồng thời vẫn tham chiếu:
  - teacher checkpoint: `refiner.pt`
  - teacher motion data gốc: `data/<TASK>`

Nói ngắn gọn:

- `motion_folder` = dữ liệu student mới
- `teacher_motion_folder` = dữ liệu gốc
- `teacher_ckpt` = policy teacher

Input:

```text
outputs/<TASK>_<EXP>/rollout_datasets/refiner/rl_dataset
data/<TASK>
outputs/<TASK>_<EXP>/ckpts/refiner.pt
```

Output chính:

```text
outputs/<TASK>_<EXP>/ckpts/tracker.pt
```

## 8. Stage 5: Rollout `tracker`

Sau khi train xong `tracker`, repo tiếp tục rollout policy này để tạo dữ liệu cho `generator`.

Script được gọi trong [train.sh](/home/huytn14/Documents/VinMotion/huygo111_SUGAR/train.sh:49):

```bash
python scripts/sugar_rl/play.py \
    --task "Sugar-G129dof-${TASK_NAME}-Tracker-Rollout" \
    --checkpoint "outputs/${TASK_NAME}_${EXP_NAME}/ckpts/tracker.pt" \
    --rollout_dir "outputs/${TASK_NAME}_${EXP_NAME}/rollout_datasets/tracker/raw_npz" \
    --motion_folder "outputs/${TASK_NAME}_${EXP_NAME}/rollout_datasets/refiner/rl_dataset" \
    --teacher_motion_folder "data/${TASK_NAME}" \
    --teacher_ckpt "outputs/${TASK_NAME}_${EXP_NAME}/ckpts/refiner.pt"
```

Ý nghĩa:

- rollout từ policy `tracker`
- tạo trajectory mới phục vụ imitation learning

Input:

```text
outputs/<TASK>_<EXP>/ckpts/tracker.pt
outputs/<TASK>_<EXP>/rollout_datasets/refiner/rl_dataset
data/<TASK>
outputs/<TASK>_<EXP>/ckpts/refiner.pt
```

Output:

```text
outputs/<TASK>_<EXP>/rollout_datasets/tracker/raw_npz
```

## 9. Stage 6: Process rollout của `tracker` thành `il_dataset`

Script được gọi trong [train.sh](/home/huytn14/Documents/VinMotion/huygo111_SUGAR/train.sh:60):

```bash
python scripts/sugar_rl/process_tracker_rollout.py \
    --data_dir "outputs/${TASK_NAME}_${EXP_NAME}/rollout_datasets/tracker"
```

Ý nghĩa:

- convert rollout thô của `tracker`
- thành dataset cho nhánh imitation learning `sugar_il`

Input:

```text
outputs/.../rollout_datasets/tracker/raw_npz
```

Output:

```text
outputs/.../rollout_datasets/tracker/il_dataset
```

### `il_dataset` là gì?

`il_dataset` là dataset dành cho **imitation learning** — khác hoàn toàn với `rl_dataset`.

**Tại sao khác format?**

`generator` là diffusion/IL model, không phải RL policy. Nó không cần chạy trong simulator mà chỉ cần đọc batch dữ liệu. Vì vậy:
- Dùng **zarr** (chunked array format) thay vì nhiều file `.npz` rời
- **Downsample 50 Hz → 10 Hz** (stride=5): lấy mỗi 5 frame 1 lần, giảm kích thước dữ liệu 5× và giảm tần số control về mức phù hợp hơn cho generator
- Tất cả episode được **nối liền thành một mảng lớn**, dùng `episode_ends` để đánh dấu ranh giới

**Xử lý gì trong bước này?**

1. Downsample 50 Hz → 10 Hz (stride = 5)
2. Chuyển orientation sang **6D representation** (2 cột đầu của ma trận rotation 3×3) — ổn định hơn quaternion khi học
3. Tạo `action = [ref_joint_pos(29), ref_root_lin_vel_b(3), ref_root_ang_vel_b(3), ref_contact_label(1)]` — tổng 36 chiều
4. Tạo `last_action = action[t-1]` (frame đầu pad bằng frame đầu tiên)
5. Padding tại ranh giới: frame đầu nhân đôi nếu rollout bắt đầu từ 0; frame cuối pad 7 lần nếu là cuối trajectory

Format `il_dataset` là zarr, chứa các key:

- `obj_pos_b`
- `obj_ori_b`
- `joint_pos`
- `project_gravity`
- `target_obj_pos_b`
- `target_obj_ori_b`
- `action`
- `last_action`

Đây là format mà `generator_dataset.py` đọc để train `generator`.

### `il_dataset` gồm gì?

`il_dataset` không phải kiểu nhiều thư mục `data_000/`, `data_001/` nữa.
Nó là một zarr dataset có cấu trúc:

```text
il_dataset/
  data/
    obj_pos_b
    obj_ori_b
    joint_pos
    project_gravity
    target_obj_pos_b
    target_obj_ori_b
    action
    last_action
  meta/
    episode_ends
```

- `data/` chứa các mảng đã ghép nối tất cả episode theo trục thời gian
- `meta/episode_ends` đánh dấu điểm kết thúc từng episode

Repo tạo format này trong `scripts/sugar_rl/process_tracker_rollout.py`.
Rollout thô từ `tracker` được downsample từ `50Hz` xuống `10Hz` trước khi lưu vào `il_dataset`.

### Mỗi key trong `il_dataset/data` chứa gì?

- `obj_pos_b`
  - shape `(T, 3)`
  - vị trí object tương đối theo robot

- `obj_ori_b`
  - shape `(T, 6)`
  - orientation object tương đối theo robot
  - được mã hóa thành 6D rotation representation

- `joint_pos`
  - shape `(T, 29)`
  - trạng thái 29 khớp của robot

- `project_gravity`
  - shape `(T, 3)`
  - vector gravity trong frame của robot

- `target_obj_pos_b`
  - shape `(T, 3)`
  - vị trí mục tiêu của object tương đối theo robot

- `target_obj_ori_b`
  - shape `(T, 6)`
  - orientation mục tiêu của object
  - cũng dùng biểu diễn 6D

- `action`
  - shape `(T, 36)`
  - action target mà `generator` phải học dự đoán
  - gồm:
    - `29` chiều `ref_joint_pos`
    - `3` chiều `ref_root_lin_vel_b`
    - `3` chiều `ref_root_ang_vel_b`
    - `1` chiều `ref_contact_label`

- `last_action`
  - shape `(T, 36)`
  - action của bước trước
  - gần như là `action[t-1]`

### `meta/episode_ends`

- shape `(N,)` với N là số episode
- lưu **cumulative index** kết thúc của từng episode trong mảng đã ghép nối

Ví dụ: 3 episode mỗi 20 step → `episode_ends = [20, 40, 60]`

Ý nghĩa:

- tất cả episode được nối thành các mảng lớn theo trục thời gian
- `episode_ends` giúp `SequenceSampler` biết ranh giới từng episode, tránh sample sequence vượt qua ranh giới episode (vì hai episode liền kề không liên quan nhau về thời gian)

**Tại sao dùng zarr?**

- Chunked storage (chunk 1000 frame): đọc nhanh bất kỳ đoạn nào mà không load toàn bộ file vào RAM
- Memory-mapped: dataset lớn hàng GB vẫn train được mà không cần load hết
- `generator_dataset.py` đọc zarr qua `StreamingReplayBuffer` và dùng `SequenceSampler` để lấy sequence có độ dài cố định trong ranh giới episode

## 10. Stage 7: Train `generator`

Script được gọi trong [train.sh](/home/huytn14/Documents/VinMotion/huygo111_SUGAR/train.sh:74):

```bash
python scripts/sugar_il/train.py \
    --config-name train_generator_workspace.yaml \
    task="${TASK_NAME}" \
    use_target=${USE_TARGET} \
    num_epochs=1001 \
    log_path="outputs/${TASK_NAME}_${EXP_NAME}/logs/generator" \
    dataset_path="outputs/${TASK_NAME}_${EXP_NAME}/rollout_datasets/tracker/il_dataset"
```

Ý nghĩa:

- `generator` là model imitation learning / diffusion
- nó học từ rollout của `tracker`, không học trực tiếp từ data gốc

Code chính tham gia:

- `scripts/sugar_il/train.py`
- `source/sugar_il/sugar_il/workspace/train_generator_workspace.py`
- `source/sugar_il/sugar_il/dataset/generator_dataset.py`
- `source/sugar_il/sugar_il/policy/generator.py`

Input:

```text
outputs/<TASK>_<EXP>/rollout_datasets/tracker/il_dataset
```

Output chính:

```text
outputs/<TASK>_<EXP>/ckpts/generator.ckpt
```

## 11. Vai trò của từng stage

### `refiner`

- policy RL đầu tiên
- học từ processed motion data gốc

### `tracker`

- policy RL thứ hai
- học từ rollout mới của `refiner`
- vẫn bám teacher data và teacher policy

### `generator`

- model imitation/diffusion
- học từ rollout của `tracker`

## 12. Mối quan hệ giữa `sugar_rl` và `sugar_il`

Pipeline training đi qua 2 package chính:

### `sugar_rl`

Phụ trách:

- train `refiner`
- rollout `refiner`
- process rollout cho `tracker`
- train `tracker`
- rollout `tracker`
- process rollout cho `generator`

### `sugar_il`

Phụ trách:

- train `generator`

Nói ngắn gọn:

- `sugar_rl` tạo và tinh chỉnh motion/control policy
- `sugar_il` học action generator từ rollout cuối

## 13. Tóm tắt một câu

Pipeline training của repo này là:

- học `refiner` từ data gốc
- dùng `refiner` sinh rollout mới để học `tracker`
- dùng `tracker` sinh rollout mới để học `generator`

tức là càng về sau, model càng học từ dữ liệu được chính pipeline RL trước đó tạo ra.
