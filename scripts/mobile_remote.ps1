param([switch]$Sim, [switch]$Stop, [switch]$NoShow, [string]$BindAddress)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$mode = if ($Sim) { 'sim' } else { 'live' }
$port = if ($Sim) { 9444 } else { 9443 }
$cardDir = Join-Path $env:LOCALAPPDATA "Nova2Remote\$mode"
$card = Join-Path $cardDir 'pairing.html'
$owners = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*-m task_skills.run_mobile_remote *' -and
                    $_.CommandLine -like "*--card-dir `"$cardDir`"*" })
# Always stop a running service first (it stops and disables the robot cleanly),
# so every launch runs the current code and Wi-Fi address.
if ($owners.Count) {
    New-Item -Path $cardDir -ItemType Directory -Force | Out-Null
    New-Item -Path (Join-Path $cardDir 'stop.request') -ItemType File -Force | Out-Null
    foreach ($owner in $owners) { Wait-Process -Id $owner.ProcessId -Timeout 40 -ErrorAction SilentlyContinue }
    if ($owners | Where-Object { Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue }) {
        throw 'Shutdown is still waiting for the robot. Check the pendant before restarting.'
    }
    Remove-Item -LiteralPath (Join-Path $cardDir 'stop.request') -ErrorAction SilentlyContinue
    $owners = @()
}
if ($Stop) { return }
if (-not $BindAddress) {
    $wireless = @(Get-NetAdapter -Physical | Where-Object { $_.Status -eq 'Up' -and $_.NdisPhysicalMedium -eq 9 })
    $ips = @(foreach ($adapter in $wireless) {
        Get-NetIPAddress -InterfaceIndex $adapter.ifIndex -AddressFamily IPv4 |
            Where-Object { $_.AddressState -eq 'Preferred' -and $_.IPAddress -notlike '169.254.*' }
    })
    if ($ips.Count -ne 1) { throw 'Select the Wi-Fi IPv4 address with -BindAddress.' }
    $BindAddress = $ips[0].IPAddress
}
if ($true) {
    $python = if ($env:NOVA2_PYTHON) { $env:NOVA2_PYTHON } else { 'C:\software\anaconda3\envs\dobot-xtrainer\python.exe' }
    if (-not (Test-Path -LiteralPath $python)) { throw 'Set NOVA2_PYTHON to the robot Python executable.' }
    New-Item -Path $cardDir -ItemType Directory -Force | Out-Null
    $opts = "-X utf8 -m task_skills.run_mobile_remote --bind $BindAddress --port $port --card-dir `"$cardDir`""
    if (-not $Sim) { $opts += ' --live' }
    $timestamp = [DateTime]::UtcNow
    $process = Start-Process -FilePath $python -WorkingDirectory $repo -ArgumentList $opts -PassThru -WindowStyle Hidden -RedirectStandardError (Join-Path $cardDir 'errors.log') -RedirectStandardOutput (Join-Path $cardDir 'service.log')
    $deadline = $timestamp.AddSeconds(20)
    do {
        Start-Sleep -Milliseconds 150
        if ($process.HasExited) { throw "Startup failed. See $cardDir\errors.log" }
        $fresh = (Test-Path $card) -and (Get-Item $card).LastWriteTimeUtc -ge $timestamp.AddSeconds(-0.05)
    } while (-not $fresh -and [DateTime]::UtcNow -lt $deadline)
    if (-not $fresh) { throw "Pairing card was not created. See $cardDir\errors.log" }
}
if (-not $NoShow) { Start-Process -FilePath $card }
Write-Output "https://${BindAddress}:$port/"
Write-Output $card
