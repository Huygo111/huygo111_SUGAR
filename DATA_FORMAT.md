# Data Format

Tài liệu này mô tả format dữ liệu trong thư mục `data/` của repo SUGAR.

## 1. Cấu trúc thư mục

Thư mục `data/` hiện có 6 task:

- `CarryBox`
- `KickBox`
- `PickBottle`
- `PushBox`
- `SitChair`
- `StandBottle`

Mỗi task có nhiều sample:

```text
data/<TASK>/data_000/
data/<TASK>/data_001/
data/<TASK>/data_002/
...
```

Mỗi thư mục `data_XXX` là một trajectory/motion sample.

## 2. File bên trong mỗi sample

Mỗi `data_XXX/` thường có 3 file:

- `robot_50hz.npz`
- `obj_motion_global_50hz.pkl`
- `contact_labels_50hz.npy`

Ý nghĩa:

- `robot_50hz.npz`: chuyển động của robot
- `obj_motion_global_50hz.pkl`: chuyển động của object
- `contact_labels_50hz.npy`: nhãn contact theo thời gian

`50hz` nghĩa là dữ liệu được lấy mẫu ở 50 frame mỗi giây.

## 3. File `.npy` nói chung

File `.npy` là định dạng của NumPy để lưu một `ndarray`.

Một file `.npy` thường thể hiện:

- `shape`: kích thước mảng
- `dtype`: kiểu dữ liệu
- `data`: giá trị mảng

Trong repo này, `contact_labels_50hz.npy` là một mảng boolean theo thời gian.

Ví dụ thực tế:

- shape: `(500,)`
- dtype: `bool`

Ý nghĩa:

- có 500 timestep
- mỗi phần tử là `True/False`
- cho biết tại timestep đó có contact đúng loại hay không

## 4. File `.npz` nói chung

File `.npz` là một gói chứa nhiều `ndarray` theo `key`.

Khác nhau:

- `.npy`: 1 mảng
- `.npz`: nhiều mảng

Trong `robot_50hz.npz`, mỗi key là một `ndarray`.

## 5. `robot_50hz.npz`

File này mô tả trạng thái chuyển động của robot theo thời gian.

### `fps`

- shape: `(1,)`
- dtype: `int32`
- ý nghĩa: tần số lấy mẫu

Giá trị hiện có là `50`.

### `joint_pos`

- shape: `(T, 29)`
- dtype: `float32`
- ý nghĩa: góc/vị trí của 29 khớp robot theo thời gian

Hiểu theo trục:

- `T`: số timestep
- `29`: số khớp

`joint_pos[t]` là trạng thái toàn bộ khớp ở thời điểm `t`.

### `joint_vel`

- shape: `(T, 29)`
- dtype: `float32`
- ý nghĩa: vận tốc của 29 khớp theo thời gian

`joint_vel[t]` cho biết mỗi khớp đang chuyển động nhanh/chậm thế nào tại thời điểm `t`.

### `body_pos_w`

- shape: `(T, 35, 3)`
- dtype: `float32`
- ý nghĩa: vị trí của 35 rigid body của robot trong world frame

Hiểu theo trục:

- `T`: số timestep
- `35`: số body
- `3`: tọa độ `x, y, z`

`body_pos_w[t, i]` là vị trí body thứ `i` ở thời điểm `t`.

### `body_quat_w`

- shape: `(T, 35, 4)`
- dtype: `float32`
- ý nghĩa: orientation của 35 body trong world frame

Hiểu theo trục:

- `T`: số timestep
- `35`: số body
- `4`: quaternion

`body_quat_w[t, i]` là orientation của body thứ `i` ở thời điểm `t`.

### `body_lin_vel_w`

- shape: `(T, 35, 3)`
- dtype: `float32`
- ý nghĩa: vận tốc tuyến tính của từng body trong world frame

`3` là vector vận tốc theo `x, y, z`.

### `body_ang_vel_w`

- shape: `(T, 35, 3)`
- dtype: `float32`
- ý nghĩa: vận tốc góc của từng body

`3` là vector vận tốc góc quanh các trục.

## 6. File `.pkl` nói chung

File `.pkl` là file `pickle` của Python, dùng để lưu object Python bất kỳ.

Nó có thể chứa:

- `dict`
- `list`
- `numpy.ndarray`
- object Python khác

Trong repo này, `obj_motion_global_50hz.pkl` là một `dict`.

## 7. `obj_motion_global_50hz.pkl`

File này mô tả trạng thái của object theo thời gian trong global/world frame.

### `obj_trans`

- shape: `(T, 3)`
- dtype: `float32`
- ý nghĩa: vị trí của object theo thời gian

Mỗi dòng là:

- `[x, y, z]`

`obj_trans[t]` là vị trí của object tại thời điểm `t`.

### `obj_rot`

- shape: `(T, 3, 3)`
- dtype: `float32`
- ý nghĩa: orientation của object theo thời gian

Mỗi timestep có một rotation matrix `3x3`.

`obj_rot[t]` cho biết object đang xoay theo hướng nào trong world frame.

### `obj_scale`

- type: `float`
- ý nghĩa: scale của object

### `obj_lin_vel`

- shape: `(T, 3)`
- dtype: `float32`
- ý nghĩa: vận tốc tuyến tính của object theo thời gian

Mỗi dòng là:

- `[vx, vy, vz]`

### `obj_ang_vel`

- shape: `(T, 3)`
- dtype: `float32`
- ý nghĩa: vận tốc góc của object theo thời gian

Mỗi dòng là:

- `[wx, wy, wz]`

## 8. `contact_labels_50hz.npy`

File này là chuỗi nhãn contact theo thời gian.

Ví dụ thực tế:

- shape: `(500,)`
- dtype: `bool`

Ý nghĩa:

- mỗi phần tử ứng với một timestep
- `True`: có contact đúng loại
- `False`: không có contact

Loại contact phụ thuộc task:

- `CarryBox`, `PushBox`, `PickBottle`, `StandBottle`: hand contact
- `KickBox`: foot contact
- `SitChair`: sit contact

## 9. Ý nghĩa tổng thể của một sample

Một thư mục `data/<TASK>/data_XXX/` là một trajectory hoàn chỉnh gồm:

- motion của robot: `robot_50hz.npz`
- motion của object: `obj_motion_global_50hz.pkl`
- nhãn contact: `contact_labels_50hz.npy`

Nói ngắn gọn:

- robot đang làm gì
- object đang ở đâu và chuyển động thế nào
- thời điểm nào có contact quan trọng

## 10. Cách repo dùng dữ liệu này

Thư mục `data/<TASK>` là đầu vào cho stage train đầu tiên của pipeline.

Luồng tổng quát:

1. `train.sh` dùng `data/<TASK>` để train `refiner`
2. env load các sample `data_XXX` làm motion reference
3. rollout mới được sinh ra để train `tracker`
4. rollout tiếp theo được xử lý để train `generator`

Vì vậy, `data/` trong repo này là processed trajectory dataset đầu vào, không phải raw RGB-D video.
