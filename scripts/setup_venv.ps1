Param(
    [string]$Profile = ""
)

# ------------------------------
# Config File Check
# ------------------------------
$ConfigPath = "config/venv_config.yaml"

if (!(Test-Path $ConfigPath)) {
    Write-Host "[ERROR] config 파일을 찾을 수 없습니다: $ConfigPath"
    exit 1
}

# ------------------------------
# Profile 결정 (인자 > YAML > 기본 dev)
# ------------------------------
if ([string]::IsNullOrWhiteSpace($Profile)) {
    $profileLine = Select-String -Path $ConfigPath -Pattern "^venv_profile:"
    if ($profileLine) {
        $Profile = ($profileLine.Line.Split(":")[1]).Trim()
    }
}

if ([string]::IsNullOrWhiteSpace($Profile)) {
    $Profile = "dev"
}

if ($Profile -notin @("dev", "prod")) {
    Write-Host "[WARN] 알 수 없는 프로필: $Profile → dev로 변경"
    $Profile = "dev"
}

# ------------------------------
# YAML에서 값 읽기
# ------------------------------
function Read-YamlValue($key) {
    $line = Select-String -Path $ConfigPath -Pattern $key
    if ($line) {
        return ($line.Line.Split(":")[1]).Trim()
    }
    return ""
}

$VenvName  = Read-YamlValue "venv_name_$Profile"
$PythonCmd = Read-YamlValue "venv_python_$Profile"
$ReqFile   = Read-YamlValue "venv_requirements_$Profile"

Write-Host "[INFO] Profile       : $Profile"
Write-Host "[INFO] Using venv    : $VenvName"
Write-Host "[INFO] Python command: $PythonCmd"
Write-Host "[INFO] Requirements  : $ReqFile"
Write-Host ""

if (-not $VenvName -or -not $PythonCmd -or -not $ReqFile) {
    Write-Host "[ERROR] venv_config.yaml 에서 값을 제대로 읽지 못했습니다."
    exit 1
}

# ------------------------------
# Python 존재 여부 체크
# ------------------------------
$pythonExists = (Get-Command $PythonCmd -ErrorAction SilentlyContinue)

if (-not $pythonExists) {
    Write-Host "[ERROR] $PythonCmd 이(가) 설치되어 있지 않습니다."
    exit 1
}

Write-Host "[INFO] Python OK: " -NoNewline
& $PythonCmd --version
Write-Host ""

# ------------------------------
# 기존 VENV 확인
# ------------------------------
if (Test-Path $VenvName) {
    Write-Host "[WARN] 이미 존재하는 가상환경입니다: $VenvName"
    Write-Host "[INFO] 그대로 사용합니다."
} else {
    Write-Host "[INFO] Creating virtual environment..."
    & $PythonCmd -m venv $VenvName
}

# ------------------------------
# VENV 활성화
# ------------------------------
Write-Host "[INFO] Activating virtual environment..."
$activatePath = ".\$VenvName\Scripts\Activate.ps1"

if (!(Test-Path $activatePath)) {
    Write-Host "[ERROR] activate 스크립트를 찾을 수 없습니다: $activatePath"
    exit 1
}

. $activatePath

# ------------------------------
# 필요한 패키지 설치
# ------------------------------
Write-Host "[INFO] Installing requirements..."
pip install --upgrade pip
pip install -r $ReqFile

Write-Host ""
Write-Host "[SUCCESS] Virtual environment setup complete!"
Write-Host "[INFO] Now active: $VenvName"
Write-Host ""