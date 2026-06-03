@echo off
setlocal

set "MESSAGE=%~1"
if "%MESSAGE%"=="" set "MESSAGE=Auto-commit: %date% %time%"

echo Staging all changes...
git add -A

for /f "delims=" %%i in ('git status --porcelain') do (
    set "CHANGES=1"
    goto :commit
)
echo No changes to commit.
goto :eof

:commit
echo Changes detected. Committing...
git commit -m "%MESSAGE%"

if "%~2"=="nopush" goto :skip_push
echo Pushing to remote...
git push
echo Successfully pushed to GitHub!

:skip_push
git status --short