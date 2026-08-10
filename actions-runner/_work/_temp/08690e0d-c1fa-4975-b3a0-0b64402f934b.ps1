$ErrorActionPreference = 'stop'
docker pull sofdev1/spam-detection:latest
if ((Test-Path -LiteralPath variable:\LASTEXITCODE)) { exit $LASTEXITCODE }