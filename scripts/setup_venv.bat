@echo off
setlocal ENABLEDELAYEDEXPANSION

REM ------------------------------
REM Config
REM ------------------------------
set "CONFIG_FILE=config\venv_config.yaml"

if not exist "%CONFIG_FILE%" (
    echo [ERROR] config 파일을 찾을 수 없습니다: %CONFIG_FILE%
    exit /b 1
)

REM ------------------------------
REM Profile 결정 (인자 > YAML > 기본 dev)
REM ------------------------------
set "PROFILE=%~1"

if "%PROFILE%"=="" (
    for /f "tokens=2 delims=:" %%A in ('findstr /b /r "venv_profile:" "%CONFIG_FILE%"') do (
        set "PROFILE=%%A"
    )
    set "PROFILE=%PROFILE: =%"
)

if "%PROFILE%"=="" (
    set "PROFILE=dev"
)

if /I not "%PROFILE%"=="dev" if /I not "%PROFILE%"=="prod" (
    echo [WARN] 알 수 없는 프로필: %PROFILE% ^> dev로 변경
    set "PROFILE=dev"
)

REM ------------------------------
REM YAML에서 값 읽기 (dev/prod 프로필별)
REM ------------------------------
set "VENV_NAME="
set "PY_CMD="
set "REQ_FILE="

for /f "tokens=2 delims=:" %%A in ('findstr /b /r "venv_name_%PROFILE%:" "%CONFIG_FILE%"') do (
    set "VENV_NAME=%%A"
)
for /f "tokens=2 delims=:" %%A in ('findstr /b /r "venv_python_%PROFILE%:" "%CONFIG_FILE%"') do (
    set "PY_CMD=%%A"
)
for /f "tokens=2 delims=:" %%A in ('findstr /b /r "venv_requirements_%PROFILE%:" "%CONFIG_FILE%"') do (
    set "REQ_FILE=%%A"
)

REM 공백 제거
set "VENV_NAME=%VENV_NAME: =%"
set "PY_CMD=%PY_CMD: =%"
set "REQ_FILE=%REQ_FILE: =%"

echo [INFO] Profile       : %PROFILE%
echo [INFO] Using venv    : %VENV_NAME%
echo [INFO] Python command: %PY_CMD%
echo [INFO] Requirements  : %REQ_FILE%
echo.

if "%VENV_NAME%"=="" (
    echo [ERROR] venv_name_%PROFILE% 값을 읽지 못했습니다.
    exit /b 1
)
if "%PY_CMD%"=="" (
    echo [ERROR] venv_python_%PROFILE% 값을 읽지 못했습니다.
    exit /b 1
)
if "%REQ_FILE%"=="" (
    echo [ERROR] venv_requirements_%PROFILE% 값을 읽지 못했습니다.
    exit /b 1
)

REM ------------------------------
REM Python 존재 여부 체크
REM ------------------------------
where %PY_CMD% >nul 2>&1
if errorlevel 1 (
    echo [ERROR] %PY_CMD% 이(가) 설치되어 있지 않습니다.
    exit /b 1
)

echo [INFO] Python OK:
%PY_CMD% --version
echo.

REM ------------------------------
REM 기존 VENV 확인
REM ------------------------------
if exist "%VENV_NAME%" (
    echo [WARN] 이미 존재하는 가상환경입니다: %VENV_NAME%
    echo [INFO] 그대로 사용합니다.
) else (
    echo [INFO] Creating virtual environment...
    %PY_CMD% -m venv "%VENV_NAME%"
)

REM ------------------------------
REM VENV 활성화
REM ------------------------------
echo [INFO] Activating virtual environment...
if not exist "%VENV_NAME%\Scripts\activate.bat" (
    echo [ERROR] activate 스크립트를 찾을 수 없습니다: %VENV_NAME%\Scripts\activate.bat
    exit /b 1
)

call "%VENV_NAME%\Scripts\activate.bat"

REM ------------------------------
REM 필요한 패키지 설치
REM ------------------------------
echo [INFO] Installing requirements...
pip install --upgrade pip
pip install -r "%REQ_FILE%"

echo.
echo [SUCCESS] Virtual environment setup complete!
echo [INFO] Now active: %VENV_NAME%
echo.

endlocal