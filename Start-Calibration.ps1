$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location $PSScriptRoot

$venv = Join-Path $PSScriptRoot ".venv-calibration"
$python = Join-Path $venv "Scripts\python.exe"
$versionProbe = "import sys; raise SystemExit(0 if sys.version_info[:2] in ((3,11),(3,12)) else 1)"

if (-not (Test-Path $python)) {
    $created = $false
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($selector in @("-3.12", "-3.11")) {
            & py $selector -c $versionProbe 2>$null
            if ($LASTEXITCODE -eq 0) {
                & py $selector -m venv $venv
                if ($LASTEXITCODE -eq 0) { $created = $true; break }
            }
        }
    }
    if (-not $created -and (Get-Command python -ErrorAction SilentlyContinue)) {
        & python -c $versionProbe 2>$null
        if ($LASTEXITCODE -eq 0) {
            & python -m venv $venv
            if ($LASTEXITCODE -eq 0) { $created = $true }
        }
    }
    if (-not $created) {
        throw "Python 3.11 or 3.12 is required. Install it, or create .venv-calibration with py -3.12 -m venv .venv-calibration."
    }
}

& $python -c $versionProbe
if ($LASTEXITCODE -ne 0) {
    throw "Existing .venv-calibration is not Python 3.11/3.12. Recreate it or use a separate environment."
}

& $python -m pip install --disable-pip-version-check -r (Join-Path $PSScriptRoot "requirements-calibration.txt")
if ($LASTEXITCODE -ne 0) { throw "Calibration dependency installation failed." }

if ($args.Count -gt 0) {
    & $python -X utf8 -B -m task_skills.run_calibration_ui @args
} else {
    & $python -X utf8 -B -m task_skills.run_calibration_ui --open-browser
}
exit $LASTEXITCODE
