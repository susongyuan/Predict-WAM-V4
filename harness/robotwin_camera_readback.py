"""Optional serialized native Color readback; no camera poses/physics changed.

Used to diagnose a get_picture(Color) stall with three asynchronously submitted
cameras. Rendering parameters and RGB conversion match native get_rgba exactly.
No stale-frame fallback, retries, new views, or object/segmentation reads.
"""
import numpy as np
class SerialColorReadback:
    def __init__(self,cameras):
        self.cameras=cameras;self.cache=None;self.frames=0
        self.original_update=cameras.update_picture;self.original_rgba=cameras.get_rgba
        cameras.update_picture=self.update_picture;cameras.get_rgba=self.get_rgba
    def update_picture(self):
        self.cache=None
        pairs=[]
        if self.cameras.collect_wrist_camera:
            pairs.extend([('left_camera',self.cameras.left_camera),('right_camera',self.cameras.right_camera)])
        pairs.extend(zip(self.cameras.static_camera_name,self.cameras.static_camera_list))
        result={}
        for name,camera in pairs:
            camera.take_picture()
            color=camera.get_picture('Color')
            rgba=(np.asarray(color)*255).clip(0,255).astype('uint8')
            if rgba.ndim!=3 or rgba.shape[-1]!=4:raise ValueError('invalid native Color texture')
            if name!='head_camera' or self.cameras.collect_head_camera:result[name]={'rgba':rgba}
        self.cache=result;self.frames+=1
    def get_rgba(self):
        if self.cache is None:raise RuntimeError('no completed current serial camera readback')
        return {name:{'rgba':value['rgba'].copy()} for name,value in self.cache.items()}
    def close(self):
        self.cameras.update_picture=self.original_update;self.cameras.get_rgba=self.original_rgba;self.cache=None
