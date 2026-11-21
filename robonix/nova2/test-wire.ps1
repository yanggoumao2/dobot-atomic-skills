# Run the simulated providers and the official Atlas natively on Windows.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$RobonixSource,
    [string]$Python = 'python',
    [string]$Cargo = 'cargo'
)
$ErrorActionPreference = 'Stop'
$sourceRoot = (Resolve-Path -LiteralPath $RobonixSource).Path
$revision = 'fb76dc9c2766f71936c7f20b39f09994965e1176'
$head = git -C $sourceRoot rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $head -ne $revision) {
    throw "Check out Robonix $revision and initialize its IDL submodules."
}

function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program exited with code $LASTEXITCODE" }
}

$target = Join-Path $sourceRoot 'target'
Invoke-Checked $Cargo @('build', '--locked', '--manifest-path', "$sourceRoot/Cargo.toml",
    '--target-dir', $target, '-p', 'robonix-atlas', '-p', 'robonix-codegen')
$codegen = Join-Path $target 'debug/robonix-codegen.exe'
$caps = Join-Path $sourceRoot 'capabilities'
$atlasProto = Join-Path $sourceRoot 'system/atlas/proto'
$stage = Join-Path $PSScriptRoot 'rbnx-build/windows/proto-staging'
$null = New-Item -ItemType Directory -Force -Path $stage
Invoke-Checked $codegen @('--lang', 'proto', '-I', "$caps/lib", '--contracts', $caps, '-o', $stage)
$inputs = @(Get-ChildItem -Path "$atlasProto/*.proto","$stage/*.proto" -File | Sort-Object FullName |
    Select-Object -ExpandProperty FullName)
foreach ($package in 'nova2_arm', 'nova2_gripper') {
    $generated = Join-Path $PSScriptRoot "primitives/$package/rbnx-build/codegen"
    $bindings = Join-Path $generated 'proto_gen'
    $null = New-Item -ItemType Directory -Force -Path $bindings
    $protocArgs = @('-m', 'grpc_tools.protoc', '-I', $atlasProto, '-I', $stage,
        "--python_out=$bindings", "--grpc_python_out=$bindings") + $inputs
    Invoke-Checked $Python $protocArgs
    if ($package -eq 'nova2_gripper') {
        Invoke-Checked $codegen @('--lang', 'mcp', '-I', "$caps/lib", '-o', "$generated/robonix_mcp_types")
    }
}
Invoke-Checked $Python @("$PSScriptRoot/tests/wire_smoke.py", '--atlas-bin',
    "$target/debug/robonix-atlas.exe", '--capabilities', $caps)
