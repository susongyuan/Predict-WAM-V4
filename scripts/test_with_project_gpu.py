import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import with_project_gpu as gpu


class DriverEnvironmentTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        library = root / 'libcuda.so'
        library.write_bytes(b'verified driver')
        self.bundle = root / 'bundle.json'
        self.metadata = dict(kernel_version='595.91.07', userspace_version='595.91.07',
                             environment={'LD_LIBRARY_PATH': str(root), 'VK_DRIVER_FILES': 'local.icd'},
                             file_hashes={str(library): hashlib.sha256(library.read_bytes()).hexdigest()})
        self.bundle.write_text(json.dumps(self.metadata))
        self.library = library

    def choose(self, health, environment=None):
        with patch.object(gpu, 'BUNDLE', self.bundle), \
             patch.object(gpu, 'kernel_version', return_value='595.91.07'), \
             patch.object(gpu, 'healthy', side_effect=health):
            return gpu.choose_environment(environment or {'PATH': '/bin'})

    def test_healthy_environment_needs_no_bundle(self):
        self.bundle.unlink()
        self.assertEqual(self.choose([True]), {'PATH': '/bin'})

    def test_matched_fallback_preserves_command_environment(self):
        original = {'PATH': '/bin', 'LD_LIBRARY_PATH': '/custom', 'CUDA_VISIBLE_DEVICES': '0'}
        result = self.choose([False, True], original)
        self.assertEqual(result['LD_LIBRARY_PATH'], str(self.library.parent) + ':/custom')
        self.assertEqual(result['CUDA_VISIBLE_DEVICES'], '0')
        self.assertEqual(original['LD_LIBRARY_PATH'], '/custom')

    def test_different_kernel_rejected_before_library_load(self):
        self.metadata['kernel_version'] = '595.99.02'
        self.bundle.write_text(json.dumps(self.metadata))
        with self.assertRaisesRegex(RuntimeError, 'No verified local'):
            self.choose([False])

    def test_changed_library_rejected(self):
        self.library.write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'library changed'):
            self.choose([False])

    def test_missing_library_rejected(self):
        self.library.unlink()
        with self.assertRaises(FileNotFoundError):
            self.choose([False])

    def test_unhealthy_gpu_does_not_launch(self):
        with self.assertRaisesRegex(RuntimeError, 'still fail'):
            self.choose([False, False])


if __name__ == '__main__':
    unittest.main()
