import numpy as np

# def get_frameidx(*, mode, nframes, exact_frame, frames_to_keep):
#     # return: frameidx is a list (including the frames to be rendered)
#     if mode == "sequence":
#         frameidx = np.linspace(0, nframes - 1, frames_to_keep)
#         frameidx = np.round(frameidx).astype(int)
#         frameidx = list(frameidx)
#     elif mode == "frame":
#         index_frame = int(exact_frame*nframes)
#         frameidx = [index_frame]
#     elif mode == "video":
#         frameidx = range(0, nframes)
#     else:
#         raise ValueError(f"Not support {mode} render mode")
#     return frameidx

def get_frameidx(*, mode, nframes, exact_frame=None, frames_to_keep=None):
    # return: frameidx is a list (including the frames to be rendered)
    if mode == "sequence":
        # 使用余弦函数分布，以在中间增加密度
        x = np.linspace(0, np.pi, frames_to_keep)
        weights = (1 - np.cos(x)) / 2  # 余弦权重，从两侧到中间逐渐增加
        frameidx = np.cumsum(weights)  # 积分得到累加位置
        frameidx = (frameidx / frameidx[-1]) * (nframes - 1)  # 归一化到 [0, nframes - 1]
        frameidx = np.round(frameidx).astype(int)
        frameidx = sorted(list(set(frameidx)))  # 去重并排序
        # kicks
        #frameidx = [11, 114, 120, 126,131, 138]
        # dive 
        #frameidx =  [40,45,52,55,60,187]
        # knee
        #frameidx = [15,27,32,42,49,55,59,68,78,84,96,109,143]
        # mbappe coma、
        
        # old lady Coma
        #frameidx = [86, 81, 11,30]
        
        # jump-spin coma
        #frameidx= [30,37,41,47,54,63,70,144,189,203,241,310,322,329]
        
        # crawl
        #frameidx= [29,68,120,132,174,275,353,364]
        
        print('Frame index:',frameidx)
    elif mode == "frame":
        index_frame = int(exact_frame * nframes)
        frameidx = [index_frame]
    elif mode == "video":
        frameidx = list(range(0, nframes))
    else:
        raise ValueError(f"Not support {mode} render mode")
    return frameidx