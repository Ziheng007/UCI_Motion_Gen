import numpy as np
def get_hml_joint_mask(joints,place):
    HML_binary = np.array([i in joints for i in range(NUM_HML_JOINTS)])
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

HML_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
]

NUM_HML_JOINTS = len(HML_JOINT_NAMES)  # 22 SMPLH body joints
#TODO check spine
HML_LEFT_ARM_JOINTS = [HML_JOINT_NAMES.index(name) for name in ['left_collar', 'left_shoulder', 'left_elbow', 'left_wrist','spine3','spine2','spine1','head','neck']]
HML_RIGHT_ARM_JOINTS = [HML_JOINT_NAMES.index(name) for name in ['right_collar', 'right_shoulder', 'right_elbow', 'right_wrist','spine3','spine2','spine1','head','neck']]
HML_LEFT_LEG_JOINTS = [HML_JOINT_NAMES.index(name) for name in ['left_ankle', 'left_foot','left_hip','left_knee','pelvis']]
HML_RIGHT_LEG_JOINTS = [HML_JOINT_NAMES.index(name) for name in ['right_ankle', 'right_foot','right_hip','right_knee','pelvis']]
HML_SPINE_JOINTS = [HML_JOINT_NAMES.index(name) for name in ['spine1','spine2','spine3','head','neck']]

HML_LOWER_BODY_JOINTS = [HML_JOINT_NAMES.index(name) for name in ['pelvis', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_foot', 'right_foot',]]
SMPL_UPPER_BODY_JOINTS = [i for i in range(len(HML_JOINT_NAMES)) if i not in HML_LOWER_BODY_JOINTS]
# general vq -> quantizer -> 4 toekn_emb cacat -> emb -> decoder 

# generation: text -> trasformer -> token_index -> quantizer -> token_emb -> decoder
# residual: right arm: base quantizer + 5 residual quantizers
# traning vq: moiton(263) -> 4 encoder -> 4 encoder_emb -> 4 quantizer -> 4 token_emb -> 1 or 4 decoder -> motion
# training transformer: motion -> 4 encoder -> 4 encoder_emb -> 4 quantizer -> 4 token_index

# codebook key: index (1,2-128) value: emb
# multi vq ->  
# 263
# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3) #表示除根节点外的所有关节的相对位置,每个关节3维
# rot_data (B, seq_len, (joint_num - 1)*6) #表示除根节点外的所有关节的旋转,每个关节6维
# local_velocity (B, seq_len, joint_num*3) #所有关节的局部速度,包括根节点,每个关节3维
# foot contact (B, seq_len, 4)
HML_ROOT_BINARY = np.array([True] + [False] * (NUM_HML_JOINTS-1))
HML_ROOT_MASK = np.concatenate(([True]*(1+2+1),
                                HML_ROOT_BINARY[1:].repeat(3),
                                HML_ROOT_BINARY[1:].repeat(6),
                                HML_ROOT_BINARY.repeat(3),
                                [False] * 4))

HML_LOWER_BODY_JOINTS_BINARY = np.array([i in HML_LOWER_BODY_JOINTS for i in range(NUM_HML_JOINTS)])
HML_LOWER_BODY_MASK = np.concatenate(([True]*(1+2+1),
                                     HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(3),
                                     HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(6),
                                     HML_LOWER_BODY_JOINTS_BINARY.repeat(3),
                                     [True]*4))
HML_UPPER_BODY_MASK = ~HML_LOWER_BODY_MASK
HML_LEFT_ARM_MASK=get_hml_joint_mask(HML_LEFT_ARM_JOINTS,'upper')
HML_RIGHT_ARM_MASK=get_hml_joint_mask(HML_RIGHT_ARM_JOINTS,'upper')
HML_LEFT_LEG_MASK=get_hml_joint_mask(HML_LEFT_LEG_JOINTS,'lower')
HML_RIGHT_LEG_MASK=get_hml_joint_mask(HML_RIGHT_LEG_JOINTS,'lower')
HML_SPINE_MASK=get_hml_joint_mask(HML_SPINE_JOINTS,'upper')

ALL_JOINT_FALSE = np.full(*HML_ROOT_BINARY.shape, False)
HML_UPPER_BODY_JOINTS_BINARY = np.array([i in SMPL_UPPER_BODY_JOINTS for i in range(NUM_HML_JOINTS)])

UPPER_JOINT_Y_TRUE = np.array([ALL_JOINT_FALSE[1:], HML_UPPER_BODY_JOINTS_BINARY[1:], ALL_JOINT_FALSE[1:]])
UPPER_JOINT_Y_TRUE = UPPER_JOINT_Y_TRUE.T
UPPER_JOINT_Y_TRUE = UPPER_JOINT_Y_TRUE.reshape(ALL_JOINT_FALSE[1:].shape[0]*3)

UPPER_JOINT_Y_MASK = np.concatenate(([False]*(1+2+1),
                                UPPER_JOINT_Y_TRUE,
                                ALL_JOINT_FALSE[1:].repeat(6),
                                ALL_JOINT_FALSE.repeat(3),
                                [False] * 4))
  
OVER_LAP_UPPER_MASK=HML_RIGHT_ARM_MASK&HML_LEFT_ARM_MASK
OVER_LAP_LOWER_MASK=HML_RIGHT_LEG_MASK&HML_LEFT_LEG_MASK
