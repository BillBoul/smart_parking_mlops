@echo off
echo 🔁 Forcing update_dataset to re-run...

REM Update the dummy dependency with current timestamp
echo %date% %time% > .always_run_trigger

REM Run DVC pipeline
dvc repro