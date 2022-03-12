import os, pathlib
import pytest

def test_submission():
    try:
        s = os.system(r"jupyter nbconvert --to notebook --execute Assignment_2.ipynb")
        assert s == 0
    except:
        pytest.fail("Error while running notebook.")

def test_verification():
    curr_path = os.getcwd()
    try:
        os.chdir(os.path.join(curr_path, 'submit'))
        os.system('python verify.py')
    except:
        pytest.fail("Verification script failed.")
