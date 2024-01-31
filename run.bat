@echo off
setlocal enabledelayedexpansion

:: Define the current directory and the parent directory
set "CurrentDir=%CD%"
set "ParentDir=%CurrentDir%\.."

:: Define the destination "code" directory path in the parent directory
set "CodeDirPath=%ParentDir%\code"

:: Copy all contents of the current directory to the "code" directory
xcopy "%CurrentDir%\*" "%CodeDirPath%" /E /I /Y

:: Empty the "model" folder within the "code" directory, if it exists
if exist "%CodeDirPath%\model\*" (
    del /Q "%CodeDirPath%\model\*"
)
for /d %%x in ("%CodeDirPath%\model\*") do @rd /s /q "%%x"

:: Remove .vscode, .git, and .tmp.driveupload directories from the "code" directory, if they exist
if exist "%CodeDirPath%\.vscode\" rd /s /q "%CodeDirPath%\.vscode"
if exist "%CodeDirPath%\.git\" rd /s /q "%CodeDirPath%\.git"
if exist "%CodeDirPath%\.tmp.driveupload\" rd /s /q "%CodeDirPath%\.tmp.driveupload"

:: Use PowerShell to zip the "code" directory
powershell.exe -command "Compress-Archive -Path '%CodeDirPath%\*' -DestinationPath '%ParentDir%\code.zip' -Force"

echo Process completed.
pause
