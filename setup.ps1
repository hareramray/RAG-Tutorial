$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    py -3.13 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.13 is required for this Windows setup. See README for other Python versions.' }
}

$workshopPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (Get-Command uv -ErrorAction SilentlyContinue) {
    uv pip sync requirements.lock.txt --python $workshopPython
} else {
    & $workshopPython -m pip install -r requirements.lock.txt
}
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Review the error above.' }

& $workshopPython -m ipykernel install --sys-prefix --name rag-workshop --display-name 'Python (RAG Workshop)'
if ($LASTEXITCODE -ne 0) { throw 'Kernel installation failed.' }
& $workshopPython scripts/smoke_check.py
if ($LASTEXITCODE -ne 0) { throw 'Local smoke check failed.' }
Write-Host 'Ready. Launch with: .\.venv\Scripts\jupyter-lab.exe notebooks'
