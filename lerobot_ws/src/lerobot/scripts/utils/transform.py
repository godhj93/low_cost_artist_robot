import numpy as np

X_MIN_BOUND = -0.06
X_MAX_BOUND = 0.06

# Y_MIN_BOUND = 0.23
# Y_MAX_BOUND = 0.35

Y_MIN_BOUND = 0.23
Y_MAX_BOUND = 0.35

# Z_MIN_BOUND = 0.054
# Z_MAX_BOUND = 0.20

# HJ Sim 
# Z_MIN_BOUND = 0.1
# Z_MAX_BOUND = 0.15

# Real robot
Z_MIN_BOUND = 0.054
Z_MAX_BOUND = 0.10

Z_COR = 0.002

def transform_to_configuration_space(positions):
    """
    데이터를 configuration space로 변환:
    - x는 [-0.2, 0.2] 범위로 정규화
    - y는 [0.1, 0.2] 범위로 정규화
    - z는 [0.1, 0.2] 범위로 정규화
    - x:y 비율은 유지
    """
    transformed_positions = []

    # x와 y의 최소값과 최대값 계산
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()

    print(x_min, x_max)
    print(y_min, y_max)

    width = x_max - x_min
    height = y_max - y_min

    width_is_longer = None
    if width > height:
        width_is_longer = True
    else:
        width_is_longer = False

    # z의 최소값과 최대값 계산
    z_min, z_max = positions[:, 2].min(), positions[:, 2].max()

    for pos in positions:
        x, y, z = pos

        # x 정규화, y 정규화
        if width_is_longer == True:
            x_normalized = X_MIN_BOUND + (x - x_min) / (x_max - x_min) * (X_MAX_BOUND - X_MIN_BOUND)
            y_normalized = Y_MIN_BOUND + (y - y_min) / (x_max - x_min) * (Y_MAX_BOUND - Y_MIN_BOUND)
        else:
            x_normalized = X_MIN_BOUND + (x - x_min) / (y_max - y_min) * (X_MAX_BOUND - X_MIN_BOUND)
            y_normalized = Y_MIN_BOUND + (y - y_min) / (y_max - y_min) * (Y_MAX_BOUND - Y_MIN_BOUND)

        # x_normalized += 0.005

        # scale_factor = y_normalized / y if y != 0 else 1.0
        # x_normalized *= scale_factor

        # z 정규화
        z_normalized = Z_MIN_BOUND + (z - z_min) / (z_max - z_min) * (Z_MAX_BOUND - Z_MIN_BOUND)
        # z_correction = ((-2) * Z_COR) / (Y_MAX_BOUND - Y_MIN_BOUND) * (np.sqrt(x_normalized**2 + y_normalized**2) - (Y_MAX_BOUND + Y_MIN_BOUND) / 2)
        # if z_correction > Z_COR or z_correction < (-1)*Z_COR:
            # z_correction = 0
            # print("NONONO")
        # z_normalized += z_correction

        # 결과 추가
        transformed_positions.append([x_normalized, y_normalized, z_normalized])

    return np.array(transformed_positions)

# def transform_to_configuration_space(positions):
#     """
#     데이터를 configuration space로 변환:
#     - x는 [-0.2, 0.2] 범위로 정규화
#     - y는 [0.1, 0.2] 범위로 정규화
#     - z는 [0.1, 0.2] 범위로 정규화
#     - x:y 비율은 유지
#     """
#     transformed_positions = []

#     # x와 y의 최소값과 최대값 계산
#     x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
#     y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
#     # z의 최소값과 최대값 계산
#     z_min, z_max = positions[:, 2].min(), positions[:, 2].max()
    
#     for pos in positions:
#         x, y, z = pos

#         # x 정규화
#         x_normalized = X_MIN_BOUND + (x - x_min) / (x_max - x_min) * (X_MAX_BOUND - (X_MIN_BOUND))

#         # y 정규화
#         y_normalized = Y_MIN_BOUND + (y - y_min) / (y_max - y_min) * (Y_MAX_BOUND - Y_MIN_BOUND)
#         # scale_factor = y_normalized / y if y != 0 else 1.0
#         # x_normalized *= scale_factor

#         # z 정규화
#         z_normalized = Z_MIN_BOUND + (z - z_min) / (z_max - z_min) * (Z_MAX_BOUND - Z_MIN_BOUND)

#         # 결과 추가
#         transformed_positions.append([x_normalized, y_normalized, z_normalized])

#     return np.array(transformed_positions)