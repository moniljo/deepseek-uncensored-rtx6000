# Run from a VS2022 x64 Developer PowerShell / Native Tools environment.
# Default dry; -Run performs a fresh complete source build, never inference.
param(
    [switch]$Run,
    [string]$CudaRoot = $env:CUDA_PATH,
    [string]$PythonExe = 'python',
    [string]$CMakeExe = 'cmake',
    [string]$NinjaExe = 'ninja',
    [string]$WorkDir = '',
    [string]$BuildDir = '',
    [string]$NativeSourceManifest = '',
    [ValidateRange(1,16)][int]$Parallel = 3
)
$ErrorActionPreference = 'Stop'
$RecipeRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (-not $WorkDir) { $WorkDir = Join-Path $RecipeRoot '.work/source' }
if (-not $BuildDir) { $BuildDir = Join-Path $RecipeRoot '.work/build-native' }
$WorkDir = [IO.Path]::GetFullPath($WorkDir)
$BuildDir = [IO.Path]::GetFullPath($BuildDir)
$Locks = Get-Content -LiteralPath (Join-Path $RecipeRoot 'pins/source-locks.json') -Raw | ConvertFrom-Json
if (-not $NativeSourceManifest) { $NativeSourceManifest = Join-Path $RecipeRoot 'pins/source-overlays.json' }
$NativeSourceManifest = [IO.Path]::GetFullPath($NativeSourceManifest)
$StrataSource = Join-Path $WorkDir $Locks.strata.destination
$LlamaSource = Join-Path $WorkDir $Locks.llama_cpp.destination
$Configure = @('-S', $StrataSource, '-B', $BuildDir, '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CUDA_ARCHITECTURES=120a',
    "-DSTRATA_GGML_DIR=$LlamaSource", "-DSTRATA_BUILD_ID=$($Locks.build_id)")
foreach ($Property in $Locks.required_CMake_flags.PSObject.Properties) {
    $Configure += "-D$($Property.Name)=$($Property.Value)"
}
if ($CudaRoot) {
    $CudaRoot = [IO.Path]::GetFullPath($CudaRoot)
    $Configure += "-DCMAKE_CUDA_COMPILER=$(Join-Path $CudaRoot 'bin/nvcc.exe')"
    $Configure += "-DCUDAToolkit_ROOT=$CudaRoot"
}
$Configure += "-DCMAKE_MAKE_PROGRAM=$NinjaExe"
$Build = @('--build', $BuildDir, '--target', 'strata-dsv41', '--parallel', [string]$Parallel)
if (-not $Run) {
    @{ dry = $true; source = $StrataSource; build = $BuildDir; target = 'strata-dsv41';
       configure_arguments = $Configure; build_arguments = $Build; old_objects_reused = $false;
       native_source_manifest = $NativeSourceManifest;
       model_GPU_or_network_actions = 0 } | ConvertTo-Json -Depth 5
    return
}
if (Test-Path -LiteralPath $BuildDir) { throw 'Use a NEW build directory; existing outputs are never reused.' }
$RecipePins = @{}
foreach ($Relative in @('scripts/build_native.ps1','scripts/checkout_source.py','pins/source-locks.json','pins/source-overlays.json')) {
    $RecipePins[$Relative] = (Get-FileHash -LiteralPath (Join-Path $RecipeRoot $Relative) -Algorithm SHA256).Hash.ToLowerInvariant()
}
$NativeManifestSHA = (Get-FileHash -LiteralPath $NativeSourceManifest -Algorithm SHA256).Hash.ToLowerInvariant()
if (-not $CudaRoot -or -not (Test-Path -LiteralPath (Join-Path $CudaRoot 'bin/nvcc.exe'))) {
    throw 'Pass -CudaRoot for the CUDA13.2 toolkit, or set CUDA_PATH.'
}
$Compiler = Get-Command cl.exe -ErrorAction Stop
$NinjaCommand = Get-Command $NinjaExe -ErrorAction Stop
$CMakeCommand = Get-Command $CMakeExe -ErrorAction Stop
$Configure[$Configure.Length - 1] = "-DCMAKE_MAKE_PROGRAM=$($NinjaCommand.Source)"
& $PythonExe -B (Join-Path $PSScriptRoot 'checkout_source.py') --verify --work-dir $WorkDir
if ($LASTEXITCODE -ne 0) { throw 'Pinned source verification failed.' }
$SourceCheckoutSHA = (Get-FileHash -LiteralPath (Join-Path $WorkDir 'PREPARED-SOURCE.json') -Algorithm SHA256).Hash.ToLowerInvariant()
$VerifiedBefore = $true
$NvccVersion = (& (Join-Path $CudaRoot 'bin/nvcc.exe') --version 2>&1 | Out-String)
if ($LASTEXITCODE -ne 0 -or $NvccVersion -notmatch 'release 13\.2') { throw 'The measured recipe requires CUDA13.2.' }
$ClVersion = (& $Compiler.Source /? 2>&1 | Out-String)
if ($LASTEXITCODE -ne 0 -or $ClVersion -notmatch '19\.44\.') { throw 'Select the VS2022 MSVC14.44 toolset for the measured recipe.' }
$PreviousCL = $env:_CL_
try {
    $env:_CL_ = '/Z7'
    & $CMakeCommand.Source @Configure
    if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed; outputs are preserved.' }
    & $CMakeCommand.Source @Build
    if ($LASTEXITCODE -ne 0) { throw 'Native build failed; outputs are preserved.' }
    & $PythonExe -B (Join-Path $PSScriptRoot 'checkout_source.py') --verify --work-dir $WorkDir
    if ($LASTEXITCODE -ne 0) { throw 'Source changed during build.' }
    $VerifiedAfter = $true
    if ((Get-FileHash -LiteralPath (Join-Path $WorkDir 'PREPARED-SOURCE.json') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $SourceCheckoutSHA) {
        throw 'Prepared-source receipt changed during the build.'
    }
    if ((Get-FileHash -LiteralPath $NativeSourceManifest -Algorithm SHA256).Hash.ToLowerInvariant() -ne $NativeManifestSHA) {
        throw 'Declared native-source manifest changed during the build.'
    }
    foreach ($Relative in $RecipePins.Keys) {
        $Actual = (Get-FileHash -LiteralPath (Join-Path $RecipeRoot $Relative) -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($Actual -ne $RecipePins[$Relative]) { throw 'A source-build recipe file changed during the build.' }
    }
    $Exe = Join-Path $BuildDir 'strata-dsv41.exe'
    if (-not (Test-Path -LiteralPath $Exe)) { throw 'Native executable missing.' }
    $SourcePins = @{}
    foreach ($Name in @('ds4_model.cu','ds4_main.cpp','ds4_model.hpp')) {
        $SourcePins[$Name] = (Get-FileHash -LiteralPath (Join-Path $StrataSource "src/dsv41/$Name") -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    $DeclaredManifest = Get-Content -LiteralPath $NativeSourceManifest -Raw | ConvertFrom-Json
    if ($DeclaredManifest.frozen_files_sha256) {
        foreach ($Name in $SourcePins.Keys) {
            if ($DeclaredManifest.frozen_files_sha256.$Name -ne $SourcePins[$Name]) { throw 'Compiled LF source differs from declared manifest.' }
        }
    }
    $Receipt = @{ schema = 'fresh-portable-native-build-v1'; compiled = $true; compile_and_link_rc = 0;
        exe = $Exe; exe_sha256 = (Get-FileHash -LiteralPath $Exe -Algorithm SHA256).Hash.ToLowerInvariant();
        compiled_source_pin_map_LF = $SourcePins; source_pin_map = $SourcePins;
        source_overlay_manifest_sha256 = $Locks.overlay_manifest_sha256;
        native_source_manifest = $NativeSourceManifest; source_manifest_sha256 = $NativeManifestSHA;
        source_checkout_verified_before_build = $VerifiedBefore; source_checkout_verified_after_build = $VerifiedAfter;
        source_overlay_verified_before_after = ($VerifiedBefore -and $VerifiedAfter);
        prepared_source_receipt_sha256 = $SourceCheckoutSHA;
        source_locks_sha256 = (Get-FileHash -LiteralPath (Join-Path $RecipeRoot 'pins/source-locks.json') -Algorithm SHA256).Hash.ToLowerInvariant();
        upstream_Strata_commit = $Locks.strata.commit; upstream_llama_commit = $Locks.llama_cpp.commit;
        build_id = $Locks.build_id; all_native_GGML_objects_and_four_custom_static_libraries_rebuilt_from_source = $true;
        external_prebuilt_CUDA_toolkit_libraries = $true; recipe_pins = $RecipePins;
        old_objects_or_libraries_copied = $false; CUDA_architecture = '120a'; CUDA_version_output = $NvccVersion.Trim();
        host_compiler = $Compiler.Source; host_compiler_version_output = $ClVersion.Trim(); configure_arguments = $Configure; build_arguments = $Build;
        API_model_or_service_execution = $false; qualification_not_inferred_from_compilation = $true }
    $Receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $BuildDir 'BUILD.json') -Encoding UTF8
    $Receipt | ConvertTo-Json -Depth 8
} finally {
    if ($null -eq $PreviousCL) { Remove-Item Env:_CL_ -ErrorAction SilentlyContinue }
    else { $env:_CL_ = $PreviousCL }
}
