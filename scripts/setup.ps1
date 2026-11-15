# One-time setup for the GTA 6 Stream Auto Clipper.
# Everything is downloaded and installed inside the project folder; nothing goes to C:.
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
param(
    [switch]$Force,           # re-download tools even if present
    [switch]$SkipTools,
    [switch]$SkipVenv
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# --- keep every temp / cache inside the project -------------------------------
$Tmp   = Join-Path $Root '.tmp'
$Cache = Join-Path $Root '.cache'
$Tools = Join-Path $Root 'tools'
$Dl    = Join-Path $Tmp 'downloads'
foreach ($d in @($Tmp, $Cache, $Tools, $Dl, "$Cache\pip", "$Cache\deno", "$Cache\yt-dlp")) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}
$env:TEMP = $Tmp
$env:TMP = $Tmp
$env:PIP_CACHE_DIR = "$Cache\pip"
$env:DENO_DIR = "$Cache\deno"
$env:XDG_CACHE_HOME = $Cache
$env:PYTHONPYCACHEPREFIX = $null

function Get-Zip([string[]]$Urls, [string]$Name) {
    $zip = Join-Path $Dl $Name
    foreach ($u in $Urls) {
        try {
            Write-Host "  downloading $u"
            Invoke-WebRequest -Uri $u -OutFile $zip -UseBasicParsing
            return $zip
        } catch {
            Write-Warning "  failed: $($_.Exception.Message)"
        }
    }
    throw "Could not download $Name"
}

if (-not $SkipTools) {
    # --- ffmpeg (BtbN GPL build: nvenc, cuvid, scale_cuda, overlay_cuda) -------
    $ffDir = Join-Path $Tools 'ffmpeg'
    if ($Force -or -not (Test-Path "$ffDir\bin\ffmpeg.exe")) {
        Write-Host 'Installing ffmpeg (portable, CUDA/NVENC build)'
        $zip = Get-Zip @(
            'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.0-latest-win64-gpl-8.0.zip',
            'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip'
        ) 'ffmpeg.zip'
        $ex = Join-Path $Dl 'ffmpeg_x'
        if (Test-Path $ex) { Remove-Item -Recurse -Force $ex }
        Expand-Archive -Path $zip -DestinationPath $ex -Force
        $inner = Get-ChildItem $ex -Directory | Select-Object -First 1
        if (Test-Path $ffDir) { Remove-Item -Recurse -Force $ffDir }
        Move-Item $inner.FullName $ffDir
        Remove-Item -Recurse -Force $ex
        Remove-Item -Force $zip
    }
    & "$ffDir\bin\ffmpeg.exe" -hide_banner -version | Select-Object -First 1

    # --- deno (JS runtime yt-dlp needs for YouTube) ------------------------------
    $denoDir = Join-Path $Tools 'deno'
    if ($Force -or -not (Test-Path "$denoDir\deno.exe")) {
        Write-Host 'Installing deno (portable)'
        $zip = Get-Zip @(
            'https://github.com/denoland/deno/releases/latest/download/deno-x86_64-pc-windows-msvc.zip'
        ) 'deno.zip'
        New-Item -ItemType Directory -Force -Path $denoDir | Out-Null
        Expand-Archive -Path $zip -DestinationPath $denoDir -Force
        Remove-Item -Force $zip
    }
    & "$denoDir\deno.exe" --version | Select-Object -First 1
}

if (-not $SkipVenv) {
    $venvPy = Join-Path $Root '.venv\Scripts\python.exe'
    if (-not (Test-Path $venvPy)) {
        Write-Host 'Creating .venv'
        python -m venv (Join-Path $Root '.venv')
    }
    & $venvPy -m pip install --upgrade pip --disable-pip-version-check
    & $venvPy -m pip install -r (Join-Path $Root 'requirements.txt') --disable-pip-version-check
    & $venvPy -m pip install -U 'yt-dlp[default]' --disable-pip-version-check
}

foreach ($d in @('data', 'music', 'hud_refs', 'secrets', 'site')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Root $d) | Out-Null
}
Write-Host 'Setup complete.'
