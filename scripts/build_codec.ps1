param([string]$VcVars = "")
$ErrorActionPreference = "Stop"
$recipeRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $VcVars) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path -LiteralPath $vswhere)) { throw "Install Visual Studio Build Tools with C++ and MSVC 14.44." }
    $vsInstall = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vsInstall) { throw "Visual Studio C++ Build Tools were not found." }
    $VcVars = Join-Path $vsInstall "VC\Auxiliary\Build\vcvars64.bat"
}
if (-not (Test-Path -LiteralPath $VcVars)) { throw "vcvars64.bat was not found." }
$codecOut = Join-Path $recipeRoot ".work\native"
New-Item -ItemType Directory -Path $codecOut -Force | Out-Null
$codecSource = Join-Path $recipeRoot "tools\lossless_q6_codec.cpp"
foreach ($pathValue in @($recipeRoot, $VcVars)) {
    if ($pathValue.IndexOfAny([char[]]"`r`n`"%!") -ge 0) { throw "Choose a path without quotes, percent, exclamation marks, or newlines." }
}
$codecBatch = Join-Path $codecOut "build-codec.cmd"
$codecCommands = @"
@echo off
setlocal EnableExtensions DisableDelayedExpansion
call "$VcVars" -vcvars_ver=14.44 >nul
if errorlevel 1 exit /b 1
cd /d "$codecOut"
cl /nologo /O2 /LD /EHsc /Z7 "$codecSource" /Fe:"$codecOut\lossless_q6_codec.dll" /link /INCREMENTAL:NO
if errorlevel 1 exit /b 1
exit /b 0
"@
[System.IO.File]::WriteAllText($codecBatch, $codecCommands.Replace("`n", "`r`n"))
& cmd.exe /d /c $codecBatch
if ($LASTEXITCODE -ne 0) { throw "The lossless codec build failed." }
$codecReceipt = [ordered]@{
    schema = "lossless-compatibility-codec-build-v1"
    compiled = $true
    actual_exit = $LASTEXITCODE
    source_sha256 = (Get-FileHash -LiteralPath $codecSource -Algorithm SHA256).Hash.ToLower()
    dll_sha256 = (Get-FileHash -LiteralPath (Join-Path $codecOut "lossless_q6_codec.dll") -Algorithm SHA256).Hash.ToLower()
    model_or_gpu_started = $false
}
$codecReceipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $codecOut "codec-build.json") -Encoding utf8
$codecReceipt | ConvertTo-Json
