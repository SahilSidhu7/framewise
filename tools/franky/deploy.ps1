# Copy the Framewise engine + this setup to the home server and run setup.sh there.
#   powershell -ExecutionPolicy Bypass -File tools\franky\deploy.ps1 [-Host home]
param([string]$Server = "home", [string]$Dir = "framewise-clips")
$ErrorActionPreference = "Stop"
$repo = Resolve-Path "$PSScriptRoot\..\.."
$tmp = Join-Path $env:TEMP "framewise-clips.tgz"

# engine source only (no venv, caches or tests' media)
tar -czf $tmp -C "$repo" --exclude=".venv" --exclude="__pycache__" --exclude="*.pyc" --exclude=".pytest_cache" `
    engine/pyproject.toml engine/src tools/franky
if ($LASTEXITCODE) { throw "tar failed" }

ssh $Server "mkdir -p ~/$Dir/app ~/$Dir/setup"
scp $tmp "${Server}:~/$Dir/upload.tgz"
ssh $Server @"
set -e
cd ~/$Dir && tar -xzf upload.tgz && rm upload.tgz
rm -rf app/src && mv engine/src app/src && mv engine/pyproject.toml app/ && rm -rf engine
cp -r tools/franky/. setup/ && rm -rf tools
bash setup/setup.sh
"@
Remove-Item $tmp
