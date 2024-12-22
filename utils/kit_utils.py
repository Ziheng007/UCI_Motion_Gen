import numpy as np
def get_hml_joint_mask(joints,place):
    HML_binary = np.array([i in joints for i in range(NUM_KIT_JOINTS)])
    #TODO devide left foot contact and right foot contact
    # l_foot r_foot 2+2
    if place == 'lower':
        HML_mask = np.concatenate(([True]*(1+2+1),
                                        HML_binary[1:].repeat(3),
                                        HML_binary[1:].repeat(6),
                                        HML_binary.repeat(3),
                                        [True]*4))
    elif place == 'upper':
        HML_mask = np.concatenate(([False]*(1+2+1),
                                        HML_binary[1:].repeat(3),
                                        HML_binary[1:].repeat(6),
                                        HML_binary.repeat(3),
                                        [False]*4))
    return HML_mask

KIT_JOINT_NAMES = [
    "root",
    "BP",
    "BT",
    "BLN",
    "BUN",
    "LS",
    "LE",
    "LW",
    "RS",
    "RE",
    "RW",
    "LH",
    "LK",
    "LA",
    "LMrot",
    "LF",
    "RH",
    "RK",
    "RA",
    "RMrot",
    "RF",
]
NUM_KIT_JOINTS = len(KIT_JOINT_NAMES)  # 22 SMPLH body joints
#TODO check spine
KIT_LEFT_ARM_JOINTS = [KIT_JOINT_NAMES.index(name) for name in ["LS","LE","LW","BP","BT","BLN","BUN"]]
KIT_RIGHT_ARM_JOINTS = [KIT_JOINT_NAMES.index(name) for name in ["RS","RE","RW","BP","BT","BLN","BUN"]]
KIT_LEFT_LEG_JOINTS = [KIT_JOINT_NAMES.index(name) for name in ["LH","LK","LA","LMrot","LF","root"]]
KIT_RIGHT_LEG_JOINTS = [KIT_JOINT_NAMES.index(name) for name in ["RH","RK","RA","RMrot","RF","root"]]
KIT_SPINE_JOINTS = [KIT_JOINT_NAMES.index(name) for name in ["BP","BT","BLN","BUN"]]

KIT_LOWER_BODY_JOINTS = [KIT_JOINT_NAMES.index(name) for name in ["RH","RK","RA","RMrot","RF","LH","LK","LA","LMrot","LF","root"]]
KIT_UPPER_BODY_JOINTS = [i for i in range(len(KIT_JOINT_NAMES)) if i not in KIT_LOWER_BODY_JOINTS]

# 251
# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3) #表示除根节点外的所有关节的相对位置,每个关节3维
# rot_data (B, seq_len, (joint_num - 1)*6) #表示除根节点外的所有关节的旋转,每个关节6维
# local_velocity (B, seq_len, joint_num*3) #所有关节的局部速度,包括根节点,每个关节3维
# foot contact (B, seq_len, 4)
KIT_ROOT_BINARY = np.array([True] + [False] * (NUM_KIT_JOINTS-1))
KIT_ROOT_MASK = np.concatenate(([True]*(1+2+1),
                                KIT_ROOT_BINARY[1:].repeat(3),
                                KIT_ROOT_BINARY[1:].repeat(6),
                                KIT_ROOT_BINARY.repeat(3),
                                [False] * 4))

KIT_LOWER_BODY_JOINTS_BINARY = np.array([i in KIT_LOWER_BODY_JOINTS for i in range(NUM_KIT_JOINTS)])
KIT_LOWER_BODY_MASK = np.concatenate(([True]*(1+2+1),
                                     KIT_LOWER_BODY_JOINTS_BINARY[1:].repeat(3),
                                     KIT_LOWER_BODY_JOINTS_BINARY[1:].repeat(6),
                                     KIT_LOWER_BODY_JOINTS_BINARY.repeat(3),
                                     [True]*4))

KIT_UPPER_BODY_MASK = ~KIT_LOWER_BODY_MASK
KIT_LEFT_ARM_MASK=get_hml_joint_mask(KIT_LEFT_ARM_JOINTS,'upper')
KIT_RIGHT_ARM_MASK=get_hml_joint_mask(KIT_RIGHT_ARM_JOINTS,'upper')
KIT_LEFT_LEG_MASK=get_hml_joint_mask(KIT_LEFT_LEG_JOINTS,'lower')
KIT_RIGHT_LEG_MASK=get_hml_joint_mask(KIT_RIGHT_LEG_JOINTS,'lower')
KIT_SPINE_MASK=get_hml_joint_mask(KIT_SPINE_JOINTS,'upper')

ALL_JOINT_FALSE = np.full(*KIT_ROOT_BINARY.shape, False)
HML_UPPER_BODY_JOINTS_BINARY = np.array([i in KIT_UPPER_BODY_JOINTS for i in range(NUM_KIT_JOINTS)])

UPPER_JOINT_Y_TRUE = np.array([ALL_JOINT_FALSE[1:], HML_UPPER_BODY_JOINTS_BINARY[1:], ALL_JOINT_FALSE[1:]])
UPPER_JOINT_Y_TRUE = UPPER_JOINT_Y_TRUE.T
UPPER_JOINT_Y_TRUE = UPPER_JOINT_Y_TRUE.reshape(ALL_JOINT_FALSE[1:].shape[0]*3)

KIT_UPPER_JOINT_Y_MASK = np.concatenate(([False]*(1+2+1),
                                UPPER_JOINT_Y_TRUE,
                                ALL_JOINT_FALSE[1:].repeat(6),
                                ALL_JOINT_FALSE.repeat(3),
                                [False] * 4))
  
KIT_OVER_LAP_UPPER_MASK=KIT_RIGHT_ARM_MASK&KIT_LEFT_ARM_MASK
KIT_OVER_LAP_LOWER_MASK=KIT_RIGHT_LEG_MASK&KIT_LEFT_LEG_MASK
