@echo off
title Football Match Predictor
echo.
echo  FOOTBALL MATCH PREDICTOR
echo  -----------------------
echo.
where node >nul 2>nul
if errorlevel 1 (
  echo Node.js is not installed.
  echo Install the LTS version from https://nodejs.org and run this file again.
  pause
  exit /b 1
)
if not exist node_modules (
  echo Installing required Node packages. This is needed only the first time...
  call npm install
  if errorlevel 1 (
    echo Package installation failed. Check your internet connection and try again.
    pause
    exit /b 1
  )
)
echo.
echo Starting the website...
echo Open http://localhost:3000 in your browser.
echo Keep this window open while using the website.
echo.
call npm start
pause
