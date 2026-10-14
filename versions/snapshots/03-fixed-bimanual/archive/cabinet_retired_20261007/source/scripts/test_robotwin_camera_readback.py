import unittest
from types import SimpleNamespace
import numpy as np
from harness.robotwin_camera_readback import SerialColorReadback
class Cam:
    def __init__(self,name,events):self.name=name;self.events=events;self.t=0
    def take_picture(self):self.t+=1;self.events.append((self.name,'take'))
    def get_picture(self,channel):self.events.append((self.name,'read'));return np.full((2,3,4),self.t/10,dtype=np.float32)
class Tests(unittest.TestCase):
    def test_serial_order_current_rgb_and_restore(self):
        events=[];c=SimpleNamespace(collect_wrist_camera=True,collect_head_camera=True,left_camera=Cam('left',events),right_camera=Cam('right',events),static_camera_name=['head_camera'],static_camera_list=[Cam('head',events)],update_picture=lambda:None,get_rgba=lambda:None)
        update,rgba=c.update_picture,c.get_rgba;r=SerialColorReadback(c)
        with self.assertRaises(RuntimeError):c.get_rgba()
        c.update_picture();self.assertEqual(events,[('left','take'),('left','read'),('right','take'),('right','read'),('head','take'),('head','read')])
        a=c.get_rgba();self.assertTrue((a['left_camera']['rgba']==25).all());a['left_camera']['rgba'][:]=0;self.assertTrue((c.get_rgba()['left_camera']['rgba']==25).all())
        c.update_picture();self.assertTrue((c.get_rgba()['left_camera']['rgba']==51).all());r.close();self.assertIs(c.update_picture,update);self.assertIs(c.get_rgba,rgba)
if __name__=='__main__':unittest.main()
