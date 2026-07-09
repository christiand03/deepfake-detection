<#
.SYNOPSIS
  End-to-end smoke test for the Phase 3/4 sweeps + UAP on a few videos.

.DESCRIPTION
  Runs every sweep and UAP variant the weekend queue will run, but on a tiny
  seeded subset with minimal grids and --epochs 1, so you can confirm the whole
  pipeline executes on THIS machine (real data + checkpoints) before committing
  the full ~60 h run. Each step is timed; a summary at the end says whether it is
  safe to queue the full sweeps.

  Requires all three checkpoints via env vars and a materialised test split
  (test.h5 populated + data/normalized/*.mp4). W&B runs go to OFFLINE mode by
  default so the smoke does not pollute the project -- set $env:WANDB_MODE =
  "online" beforehand to override.

.EXAMPLE
  $env:VIDEOMAE_CKPT_PATH   = "checkpoints/videomae.ckpt"
  $env:WAV2VEC2_CKPT_PATH   = "checkpoints/wav2vec2.ckpt"
  $env:MULTIMODAL_CKPT_PATH = "checkpoints/multimodal.ckpt"
  ./scripts/smoke_phase34.ps1
#>

[CmdletBinding()]
param(
    [int]$N = 6,
    [string]$Metadata = "data/processed/test_metadata.csv",       # sweeps: subset source
    [string]$FitMetadata = "data/processed/train_metadata.csv",   # UAP fit split
    [string]$EvalMetadata = "data/processed/test_metadata.csv",   # UAP eval split
    [string]$NormalizedDir = "data/normalized",
    [string]$Scratch = ".smoke"
)

$ErrorActionPreference = "Continue"
$RepoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $RepoRoot

if (-not $env:WANDB_MODE) { $env:WANDB_MODE = "offline" }
# The sweeps / UAP log non-ASCII progress (delta, Delta, x). Force UTF-8 stdio so
# they do not crash with UnicodeEncodeError on a cp1252 console. Recommended for
# the full weekend run too.
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Phase 3/4 + UAP smoke -- repo: $RepoRoot" -ForegroundColor Cyan
Write-Host "WANDB_MODE=$($env:WANDB_MODE)  N=$N  scratch=$Scratch" -ForegroundColor DarkGray

# --- Prerequisite checks -------------------------------------------------------
$fail = $false

function Test-CkptEnv([string]$Name) {
    $val = [Environment]::GetEnvironmentVariable($Name)
    if (-not $val) { Write-Host "  MISSING  $Name" -ForegroundColor Red; return $false }
    if (-not (Test-Path $val)) { Write-Host "  BAD PATH $Name -> $val" -ForegroundColor Red; return $false }
    Write-Host "  OK       $Name" -ForegroundColor Green
    return $true
}

Write-Host "`nChecking checkpoints:"
if (-not (Test-CkptEnv "VIDEOMAE_CKPT_PATH")) { $fail = $true }
if (-not (Test-CkptEnv "WAV2VEC2_CKPT_PATH")) { $fail = $true }
if (-not (Test-CkptEnv "MULTIMODAL_CKPT_PATH")) { $fail = $true }

Write-Host "`nChecking data:"
if (-not (Test-Path $Metadata)) {
    Write-Host "  MISSING  $Metadata" -ForegroundColor Red
    $fail = $true
}
else {
    $rows = (Get-Content $Metadata | Measure-Object -Line).Lines
    if ($rows -le 1) { Write-Host "  EMPTY    $Metadata (header only)" -ForegroundColor Red; $fail = $true }
    else { Write-Host "  OK       $Metadata ($rows lines)" -ForegroundColor Green }
}
if (-not (Test-Path $NormalizedDir)) {
    Write-Host "  MISSING  $NormalizedDir" -ForegroundColor Red
    $fail = $true
}
else {
    $mp4 = (Get-ChildItem -Path $NormalizedDir -Filter *.mp4 -ErrorAction SilentlyContinue | Measure-Object).Count
    if ($mp4 -lt 1) {
        Write-Host "  EMPTY    $NormalizedDir (no .mp4 -- run scripts/backfill_normalized.py)" -ForegroundColor Red
        $fail = $true
    }
    else { Write-Host "  OK       $NormalizedDir ($mp4 mp4s)" -ForegroundColor Green }
}

if ($fail) {
    Write-Host "`nPrerequisites missing -- fix the above, then re-run." -ForegroundColor Red
    exit 1
}

New-Item -ItemType Directory -Force -Path $Scratch | Out-Null
$UapOut = Join-Path $Scratch "uap"
New-Item -ItemType Directory -Force -Path $UapOut | Out-Null
$Subset = Join-Path $Scratch "smoke_subset.csv"

# --- Step runner ---------------------------------------------------------------
$results = New-Object System.Collections.ArrayList

function Invoke-Step {
    param([string]$Name, [string[]]$PyArgs)
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & python @PyArgs
    $code = $LASTEXITCODE
    $sw.Stop()
    $ok = ($code -eq 0)
    [void]$results.Add([pscustomobject]@{
            Step    = $Name
            Ok      = $ok
            Code    = $code
            Seconds = [math]::Round($sw.Elapsed.TotalSeconds, 1)
        })
    if ($ok) { Write-Host ("PASS  {0}  ({1:n1}s)" -f $Name, $sw.Elapsed.TotalSeconds) -ForegroundColor Green }
    else { Write-Host ("FAIL  {0}  (exit {1})" -f $Name, $code) -ForegroundColor Red }
}

# 0. Seeded subset shared by the sweeps
Invoke-Step "0 - subset ($N videos)" @(
    "scripts/sample_sweep_subset.py", "--metadata", $Metadata, "--n", "$N", "--out", $Subset
)

# 1. Phase 3 robustness: video + audio + upscale + multimodal, 1 grid point
Invoke-Step "1 - Phase 3 robustness" @(
    "scripts/eval_robustness_sweep.py", "--metadata", $Subset,
    "--crf-grid", "30", "--fps-grid", "15", "--audio-bitrate-grid", "64",
    "--multimodal", "--wandb-run-name", "smoke-robustness"
)

# 2. Phase 4 adversarial: unimodal video (exercises P0 argmax + re-max-pool)
Invoke-Step "2 - Phase 4 adversarial (video)" @(
    "scripts/eval_adversarial_sweep.py", "--metadata", $Subset,
    "--epsilon-grid", "0.03", "--methods", "FGSM", "--wandb-run-name", "smoke-adv-video"
)

# 3. Phase 4 adversarial: multimodal (P0-multimodal)
Invoke-Step "3 - Phase 4 adversarial (multimodal)" @(
    "scripts/eval_adversarial_sweep.py", "--metadata", $Subset, "--multimodal",
    "--attack-modalities", "both", "--epsilon-grid", "0.03", "--methods", "FGSM",
    "--wandb-run-name", "smoke-adv-mm"
)

# 4-7. UAP: video/multimodal x REAL/FAKE (H5-backed, tiny fit + balanced eval)
foreach ($m in @("video", "multimodal")) {
    foreach ($t in @("REAL", "FAKE")) {
        $pyArgs = @(
            "scripts/compute_uap.py", "--modality", $m, "--target-class", $t,
            "--fit-metadata", $FitMetadata, "--eval-metadata", $EvalMetadata,
            "--max-fit-chunks", "3", "--eval-balanced", "2", "--epochs", "1",
            "--output-dir", $UapOut, "--wandb-run-name", "smoke-uap-$m-$($t.ToLower())"
        )
        if ($m -eq "multimodal") { $pyArgs += @("--attack-modalities", "both") }
        Invoke-Step "4 - UAP $m -> $t" $pyArgs
    }
}

# --- Summary -------------------------------------------------------------------
Write-Host "`n================ SMOKE SUMMARY ================" -ForegroundColor Cyan
$results | Format-Table -AutoSize Step, Ok, Code, Seconds | Out-Host

$failed = @($results | Where-Object { -not $_.Ok })
if ($failed.Count -gt 0) {
    Write-Host ("{0} step(s) FAILED -- do NOT queue the full run yet." -f $failed.Count) -ForegroundColor Red
    exit 1
}
Write-Host "All steps passed -- pipeline runs end-to-end. Safe to queue the full sweeps." -ForegroundColor Green
Write-Host "Artefacts under $Scratch ; offline W&B runs under ./wandb (both deletable)." -ForegroundColor DarkGray
exit 0
