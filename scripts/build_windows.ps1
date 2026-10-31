# Build the standalone Windows game and zip it for a GitHub release.
#
#   pwsh scripts/build_windows.ps1
#
# The build uses a separate Python 3.12 environment so PyInstaller does not
# depend on the developer's global Python (the project itself supports 3.11+).

param(
    [string]$Python = "3.12",
    [string]$Name = "tbs"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    if (-not (Test-Path ".build-venv")) {
        uv venv --python $Python .build-venv
    }
    $py = Join-Path $root ".build-venv\Scripts\python.exe"
    uv pip install --python $py -e ".[ui]" pyinstaller

    & $py -m PyInstaller `
        --noconfirm --clean --windowed `
        --name $Name `
        --add-data "data;data" `
        --paths . `
        --specpath build `
        scripts\run_game.py

    $zip = Join-Path $root "dist\$Name-windows-x64.zip"
    if (Test-Path $zip) { Remove-Item $zip -Force }
    Compress-Archive -Path "dist\$Name\*" -DestinationPath $zip
    Write-Host "built $zip"
}
finally {
    Pop-Location
}
