"""CPU-only checks that disabled recording does not touch rendering or emit video."""
from pathlib import Path
import tempfile
import unittest
from scripts.eval_robotwin_novideo import disabled_factory


class RecordingTests(unittest.TestCase):
    def test_no_render_camera_or_video(self):
        class ForbiddenEnvironment:
            def __getattr__(self,name):raise AssertionError('disabled recording touched '+name)
        with tempfile.TemporaryDirectory() as d:
            recorder,camera=disabled_factory(ForbiddenEnvironment(),d,'ep',1,30)
            self.assertIsNone(camera)
            recorder.capture(ForbiddenEnvironment(),camera,0)
            recorder.capture(ForbiddenEnvironment(),camera,1)
            recorder.close(complete=True);recorder.close(complete=False)
            self.assertEqual([p.name for p in Path(d).iterdir()],['recording.json'])
            import json
            receipt=json.loads((Path(d)/'recording.json').read_text())
            self.assertFalse(receipt['enabled']);self.assertFalse(receipt['third_view_camera_created'])
            self.assertEqual(receipt['video_files_created'],0)


if __name__=='__main__':unittest.main()
