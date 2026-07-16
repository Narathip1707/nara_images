@echo off
REM Copy the course test images into images\ after cloning.
REM They are the lecturer's teaching material, so they are not committed to git.
REM If your course folder lives elsewhere, edit COURSE below.
REM
REM NOTE: keep this file pure ASCII. cmd.exe cannot reliably parse a .bat
REM containing Thai text, and `chcp 65001` does not fix it.

set "COURSE=D:\ru_com_sci\COS\ALLCOS 1.69\COS4301"

if not exist "%COURSE%\images" (
    echo.
    echo   Course folder not found:
    echo     %COURSE%
    echo.
    echo   This script only helps if you have the COS3401 course folder on
    echo   this machine. You do NOT need it to use the app:
    echo.
    echo     - Just run run.bat and drag any image onto the web page.
    echo     - Or copy any images you like into the images\ folder.
    echo     - Or, if you do have the course folder somewhere else,
    echo       edit the COURSE variable at the top of this file.
    echo.
    pause
    exit /b 1
)

if not exist "%~dp0images" mkdir "%~dp0images"

echo Copying test images...
copy /y "%COURSE%\images\*"                 "%~dp0images\" >nul 2>&1
copy /y "%COURSE%\Boss\lena_noise_images\*" "%~dp0images\" >nul 2>&1

for %%f in (dark.jpg dark2.png dark_image.png bright_image.png gray2.png gray3.png ^
            cameraman.tif shade.png document1.png document2.jpg bank4.jpg A1.tif A3.png ^
            RGB_test.png Xray_share.jpg breast.bmp cortex.png jetplane.tif lake.tif) do (
    copy /y "%COURSE%\Boss\image\%%f" "%~dp0images\" >nul 2>&1
)

if exist "%~dp0images\.DS_Store" del /q "%~dp0images\.DS_Store"

set N=0
for %%f in ("%~dp0images\*.*") do set /a N+=1

echo.
echo   Done - %N% images in images\
echo   Now run run.bat
echo.
pause
