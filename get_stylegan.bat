@echo off

:: Clone the repository
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git

:: Navigate into the repository folder
cd stylegan2-ada-pytorch

:: Apply the patch (assuming you have a way to apply patches on Windows, e.g., Git Bash)
git apply ..\scripts\myModification.patch

:: Navigate back to the previous directory
cd ..


:: Download the model (using curl instead of wget, which is available on Windows 10+)
python .\scripts\get_dataset_and_model.py

pause
