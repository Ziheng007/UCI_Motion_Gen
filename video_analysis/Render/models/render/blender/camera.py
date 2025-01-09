import bpy
import math

class Camera:
    def __init__(self, *, first_root, mode, is_mesh):
        camera = bpy.data.objects['Camera']

        ## initial position
        #camera.location.x = 7.36# 如果在中心点，感觉直接x设置为0
        #camera.location.x = 0# 如果在中心点，感觉直接x设置为0
        camera.location.x = 7.36

        #camera.location.y = -30.93# 那么y就要设置为一个正值
        #camera.location.y = -6.93# 那么y就要设置为一个正值
        #camera.location.x = 8
        #camera.location.y = 0
        camera.location.y = -6.93
        
        #camera.rotation_euler[0] = math.radians(65)  # 只需要正对着看就行，就全部000
        #camera.rotation_euler[1] = math.radians(0)   # y
        #camera.rotation_euler[2] = math.radians(30) # z
        #camera.rotation_euler[0] = math.radians(90)  # 只需要正对着看就行，就全部000
        #camera.rotation_euler[1] = math.radians(0)   # y
        #camera.rotation_euler[2] = math.radians(0) # z
        if is_mesh:
            # camera.location.z = 5.45
            camera.location.z = 5.6
        else:
            camera.location.z = 5.2#这个调整比较单一，就往下一些就行
            
        # camera.location.x = 10.36
        # camera.location.y = -9.93
        # camera.location.z = 8.0
        #camera.location.z = 2.0
        # camera.location.z = 5.2
        # wider point of view
        if mode == "sequence":
            if is_mesh:
                camera.data.lens = 65
            else:
                camera.data.lens = 85
        elif mode == "frame":
            if is_mesh:
                camera.data.lens = 130
            else:
                camera.data.lens = 85
        elif mode == "video":
            if is_mesh:
                camera.data.lens = 110
            else:
                # avoid cutting person
                camera.data.lens = 85
                # camera.data.lens = 140

        # camera.location.x += 0.75

        self.mode = mode
        self.camera = camera

        self.camera.location.x += first_root[0]
        self.camera.location.y += first_root[1]

        print("注意",camera.rotation_euler[0],camera.rotation_euler[1],camera.rotation_euler[2]) #1.1,0.0,0.8
        self._root = first_root

    def update(self, newroot):
        delta_root = newroot - self._root

        self.camera.location.x += delta_root[0]
        self.camera.location.y += delta_root[1]

        self._root = newroot
