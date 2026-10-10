$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$venvPath = Join-Path $projectRoot '.venv-timesfm'
python -m venv $venvPath
if ($LASTEXITCODE -ne 0) { throw '가상환경 생성 실패' }
$pythonPath = Join-Path $venvPath 'Scripts\python.exe'
& $pythonPath -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip 설치 실패' }
& $pythonPath -m pip install -r (Join-Path $PSScriptRoot 'requirements_timesfm.txt')
if ($LASTEXITCODE -ne 0) { throw 'TimesFM 의존성 설치 실패' }
Write-Host "설치 완료: $pythonPath"
