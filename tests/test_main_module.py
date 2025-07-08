import subprocess
import sys


def test_module_invocation():
    result = subprocess.run([sys.executable, '-m', 'msk_io', '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert 'Usage' in result.stdout
