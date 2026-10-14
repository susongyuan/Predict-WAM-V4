"""Explicitly requested first-case observer camera; never a policy input."""
from pathlib import Path
import hashlib
import json
import math
import os
import subprocess
import numpy as np

SEED=932126
RESOLUTION=(960,720)
POSITION=np.array([1.9,-2.5,2.1])
TARGET=np.array([0.,-.1,1.05])


class ObserverRecorder:
    def __init__(self,env):
        from pxr import Gf,UsdGeom
        import omni.replicator.core as rep
        import imageio_ffmpeg
        self.env=env;self.directory=env.output_dir
        self.path=self.directory/'third_person_full.mp4'
        if self.path.exists():raise ValueError('do not overwrite observer video')
        self.camera_path='/World/RequestedObserverCamera'
        camera=UsdGeom.Camera.Define(env.stage,self.camera_path)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.025,20))
        camera.CreateProjectionAttr(UsdGeom.Tokens.perspective)
        camera.CreateVerticalApertureAttr(24.)
        camera.CreateHorizontalApertureAttr(24.*RESOLUTION[0]/RESOLUTION[1])
        camera.CreateFocalLengthAttr(24./(2*math.tan(math.radians(55)/2)))
        forward=TARGET-POSITION;forward/=np.linalg.norm(forward)
        right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
        up=np.cross(right,forward)
        matrix=np.eye(4);matrix[:3,:3]=np.column_stack([right,up,-forward]);matrix[:3,3]=POSITION
        UsdGeom.Xformable(camera).AddTransformOp().Set(Gf.Matrix4d(*matrix.T.reshape(-1).tolist()))
        self.product=rep.create.render_product(self.camera_path,RESOLUTION)
        self.annotator=rep.AnnotatorRegistry.get_annotator('rgb');self.annotator.attach(self.product)
        command=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y',
            '-f','rawvideo','-pix_fmt','rgb24','-s','960x720','-r','10','-i','-',
            '-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(self.path)]
        self.encoder_log=(self.directory/'third_person_encoder.log').open('xb')
        self.encoder=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=self.encoder_log)
        self.index=(self.directory/'third_person_frames.jsonl').open('x')
        self.frames=0;self.last_time=None;self.command=command

    def append(self):
        import cv2
        now=self.env.physics_time
        if self.last_time is not None and abs(now-self.last_time-.1)>1e-6:
            raise ValueError('observer frames must follow original10Hz physical clock')
        data=np.asarray(self.annotator.get_data())
        if data.shape[:2]!=(720,960) or data.ndim!=3 or data.shape[2]<3:
            raise ValueError('requested observer actual RGB unavailable')
        rgb=np.ascontiguousarray(data[:,:,:3],dtype=np.uint8)
        if np.std(rgb)<1:raise ValueError('observer rendered constant image')
        score=self.env.judge.report()['score']
        frame=rgb.copy()
        cv2.rectangle(frame,(0,0),(960,58),(12,12,12),-1)
        cv2.putText(frame,f'V3 fork80k +30k v42 | seed{SEED} | {now:.1f}s | step{self.frames}',(12,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1,cv2.LINE_AA)
        cv2.putText(frame,f'Actual third-person view | score{score["base_score"]} | full task{score["full_task_success"]}',(12,47),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1,cv2.LINE_AA)
        self.encoder.stdin.write(frame.tobytes())
        self.index.write(json.dumps(dict(index=self.frames,physics_time_seconds=now,
            original_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            encoded_rgb_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),score=score['base_score']))+'\n')
        self.index.flush();self.frames+=1;self.last_time=now

    def close(self):
        self.encoder.stdin.close();code=self.encoder.wait(timeout=60)
        self.encoder_log.close();self.index.close()
        self.annotator.detach();self.product.destroy()
        if code!=0 or not self.path.is_file() or self.path.stat().st_size==0:
            raise ValueError('requested observer encoding failed; preserve artifacts')
        receipt=dict(schema='requested-third-person-video-v42',seed=SEED,status='encoded',
            path=str(self.path.resolve()),frames=self.frames,fps=10,duration_seconds=self.frames/10,
            video_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            camera_path=self.camera_path,position_world_m=POSITION.tolist(),target_world_m=TARGET.tolist(),
            resolution_wh=list(RESOLUTION),policy_inputs_unchanged=['head_camera','left_camera','right_camera'],
            policy_reads_observer=False,physics_advanced_by_recording=False,encoder_command=self.command,
            frame_index_sha256=hashlib.sha256((self.directory/'third_person_frames.jsonl').read_bytes()).hexdigest(),
            success_selection=False,selection='first preregistered case before any new outcomes')
        (self.directory/'third_person_video_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


def install():
    from harness import isaac_zhuhai_runtime as runtime
    Original=runtime.IsaacZhuhaiEnv
    class RecordingEnv(Original):
        observer=None
        def setup(self,*args,**kwargs):
            result=super().setup(*args,**kwargs)
            requested=os.environ.get('P2_V3_THIRD_VIEW_SEED')
            if requested not in (None,str(SEED)):raise ValueError('unknown requested observer case')
            seed=kwargs.get('seed',args[1] if len(args)>1 else None)
            if requested is not None and seed==SEED:self.observer=ObserverRecorder(self)
            return result
        def capture(self):
            result=super().capture()
            if self.observer is not None:self.observer.append()
            return result
        def rule_state(self):
            result=super().rule_state()
            result['video_recording']=self.observer is not None
            result['observer_video_only']=self.observer is not None
            return result
        def close(self):
            try:
                if self.observer is not None:
                    self.observer.close();self.observer=None
                    # Attach completed encoder evidence BEFORE SimulationApp.close exits.
                    path=self.output_dir/'result.json'
                    if path.exists():
                        raw=json.loads(path.read_text())
                        raw['observer_video_receipt']=json.loads((self.output_dir/'third_person_video_receipt.json').read_text())
                        tmp=path.with_suffix('.json.close.tmp')
                        tmp.write_text(json.dumps(raw,sort_keys=True,indent=2)+'\n');tmp.replace(path)
            finally:super().close()
    runtime.IsaacZhuhaiEnv=RecordingEnv
