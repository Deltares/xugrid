import os
import shutil


def remove_dir_content(path: str) -> None:
    for root, dirs, files in os.walk(path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))


remove_dir_content("api")
remove_dir_content("examples")
remove_dir_content("examples-dev")
remove_dir_content("sample_data")
remove_dir_content("_build")
