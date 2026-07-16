<#
.SYNOPSIS
  Full Phase 3/4 + UAP evaluation run over the WHOLE test split (no smoke shrinking).

.DESCRIPTION
  Runs the real Phase 3 robustness sweep, the Phase 4 adversarial sweeps, and the
  UAP transfer evals one after another, on ALL videos, using each script's own
  default grids (full CRF/FPS/bitrate, full epsilon grid with FGSM+PGD, UAP
  epochs=5 / eval-balanced=200 / all fit chunks). This is the production
  counterpart to scripts/smoke_phase34.ps1 — expect a multi-hour run.

  Unlike the smoke, W&B runs are NOT forced offline: results log to the project as
  usual. Each step is independent and logs its own run, so a single failing arm
  does not abort the rest; a summary at the end lists PASS/FAIL + wall time. The
  whole console is captured to a transcript under -LogDir.

  Requires the three (frozen Phase-2) checkpoints via env vars, a materialised test
  split (test.h5 + data/normalized/*.mp4), and a metadata CSV whose h5_path column
  points at an existing .h5 (the script verifies this — it is the exact failure
  that a stale ablation h5_path produces).

.EXAMPLE
  $env:VIDEOMAE_CKPT_PATH   = "checkpoints/videomae_phase2.ckpt"
  $env:WAV2VEC2_CKPT_PATH   = "checkpoints/wav2vec2_phase2.ckpt"
  $env:MULTIMODAL_CKPT_PATH = "checkpoints/multimodal_phase2.ckpt"
  ./scripts/run_phase34.ps1

.EXAMPLE
  # Run against the class-balanced subset instead of the full split, and resume
  # from just the UAP steps:
  ./scripts/run_phase34.ps1 -Metadata data/processed/sweep_subset_balanced.csv `
      -SkipRobustness -SkipAdversarial
#>

[CmdletBinding()]
param(
    # Robustness + adversarial sweeps read this (all test videos by default).
    [string]$Metadata = "data/processed/test_metadata.csv",
    # UAP fits delta* on the train split and transfer-evals on the test split.
    [string]$FitMetadata = "data/processed/train_metadata.csv",
    [string]$EvalMetadata = "data/processed/test_metadata.csv",
    [string]$NormalizedDir = "data/normalized",
    [string]$OutputDir = "artifacts/uap",
    [string]$LogDir = "logs/phase34",
    # Optional cap for the robustness/adversarial sweeps (0 = all videos). Useful
    # for a scaled-down but still "real-grid" run; UAP is unaffected (it uses
    # -UapMaxFitChunks / -UapEvalBalanced instead).
    [int]$MaxVideos = 0,
    [int]$UapEpochs = 5,
    [int]$UapEvalBalanced = 200,
    [int]$UapMaxFitChunks = 0,   # 0 = all label-matched fit chunks (script default)
    # Adversarial grid overrides (empty = eval_adversarial_sweep.py defaults:
    # epsilon 0.01 0.02 0.03 0.05 0.1, methods FGSM PGD). Pass epsilons as strings
    # (e.g. "0.02","0.05","0.1") so no locale decimal conversion mangles them.
    [string[]]$EpsilonGrid = @(),
    [string[]]$Methods = @(),
    # When set, adversarial configs checkpoint each grid point to
    # <ResumeDir>/adv_<config>.csv and skip completed points on restart (resumable
    # across sessions). Empty = original behaviour (table written only at end).
    [string]$ResumeDir = "",
    [switch]$SkipRobustness,
    [switch]$SkipAdversarial,
    [switch]$SkipUap
)

$ErrorActionPreference = "Continue"
$RepoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $RepoRoot

# The sweeps / UAP log non-ASCII progress (delta, Delta, x). Force UTF-8 stdio so
# they do not crash with UnicodeEncodeError on a cp1252 console.
$env:PYTHONIOENCODING = "utf-8"
if (-not $env:WANDB_MODE) { $env:WANDB_MODE = "online" }

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$Transcript = Join-Path $LogDir "run_phase34_$Stamp.log"
Start-Transcript -Path $Transcript -Append | Out-Null

Write-Host "Phase 3/4 + UAP FULL run -- repo: $RepoRoot" -ForegroundColor Cyan
Write-Host "WANDB_MODE=$($env:WANDB_MODE)  Metadata=$Metadata  MaxVideos=$MaxVideos" -ForegroundColor DarkGray
if ($env:WANDB_MODE -eq "offline") {
    Write-Host "  NOTE: WANDB_MODE=offline -- results will NOT sync to the project until 'wandb sync'." -ForegroundColor Yellow
}

# --- Prerequisite checks -------------------------------------------------------
$fail = $false

function Test-CkptEnv([string]$Name) {
    $val = [Environment]::GetEnvironmentVariable($Name)
    if (-not $val) { Write-Host "  MISSING  $Name" -ForegroundColor Red; return $false }
    if (-not (Test-Path $val)) { Write-Host "  BAD PATH $Name -> $val" -ForegroundColor Red; return $false }
    Write-Host "  OK       $Name -> $val" -ForegroundColor Green
    return $true
}

Write-Host "`nChecking checkpoints (expect the frozen Phase-2 models):"
if (-not (Test-CkptEnv "VIDEOMAE_CKPT_PATH")) { $fail = $true }
if (-not (Test-CkptEnv "WAV2VEC2_CKPT_PATH")) { $fail = $true }
if (-not (Test-CkptEnv "MULTIMODAL_CKPT_PATH")) { $fail = $true }

Write-Host "`nChecking data:"
foreach ($csv in @($Metadata, $FitMetadata, $EvalMetadata) | Select-Object -Unique) {
    if (-not (Test-Path $csv)) {
        Write-Host "  MISSING  $csv" -ForegroundColor Red; $fail = $true; continue
    }
    $rows = (Get-Content $csv | Measure-Object -Line).Lines
    if ($rows -le 1) { Write-Host "  EMPTY    $csv (header only)" -ForegroundColor Red; $fail = $true }
    else { Write-Host "  OK       $csv ($rows lines)" -ForegroundColor Green }
}

# Guard the exact failure a stale ablation h5_path produces: the metadata's
# h5_path column must point at an .h5 that exists (UAP opens it directly).
function Test-MetadataH5([string]$csv) {
    if (-not (Test-Path $csv)) { return }
    $header = (Get-Content $csv -TotalCount 1)
    $cols = $header -split ','
    $idx = [Array]::IndexOf($cols, 'h5_path')
    if ($idx -lt 0) { Write-Host "  WARN     $csv has no h5_path column." -ForegroundColor Yellow; return }
    $firstData = Get-Content $csv -TotalCount 2 | Select-Object -Last 1
    $h5 = ($firstData -split ',')[$idx]
    $resolved = if ([System.IO.Path]::IsPathRooted($h5)) { $h5 } else { Join-Path $RepoRoot $h5 }
    if (-not (Test-Path $resolved)) {
        Write-Host "  BAD H5   $csv -> h5_path='$h5' does not exist ($resolved)." -ForegroundColor Red
        Write-Host "           Fix the h5_path column to point at data/processed/*.h5 before running UAP." -ForegroundColor Red
        $script:fail = $true
    }
    else { Write-Host "  OK       $csv h5_path -> $h5" -ForegroundColor Green }
}
if (-not $SkipUap) {
    Test-MetadataH5 $FitMetadata
    Test-MetadataH5 $EvalMetadata
}

if (-not (Test-Path $NormalizedDir)) {
    Write-Host "  MISSING  $NormalizedDir" -ForegroundColor Red; $fail = $true
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
    Stop-Transcript | Out-Null
    exit 1
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$CapArgs = @()
if ($MaxVideos -gt 0) { $CapArgs = @("--max-videos", "$MaxVideos") }

# --- Step runner ---------------------------------------------------------------
$results = New-Object System.Collections.ArrayList

function Invoke-Step {
    param([string]$Name, [string[]]$PyArgs)
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    Write-Host "python $($PyArgs -join ' ')" -ForegroundColor DarkGray
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & python @PyArgs
    $code = $LASTEXITCODE
    $sw.Stop()
    $ok = ($code -eq 0)
    [void]$results.Add([pscustomobject]@{
            Step    = $Name
            Ok      = $ok
            Code    = $code
            Minutes = [math]::Round($sw.Elapsed.TotalMinutes, 1)
        })
    if ($ok) { Write-Host ("PASS  {0}  ({1:n1} min)" -f $Name, $sw.Elapsed.TotalMinutes) -ForegroundColor Green }
    else { Write-Host ("FAIL  {0}  (exit {1}) -- continuing with remaining steps" -f $Name, $code) -ForegroundColor Red }
}

# 1. Phase 3 robustness: video + audio + upscale + multimodal, full default grids.
if (-not $SkipRobustness) {
    Invoke-Step "Phase 3 robustness (full grids)" (@(
            "scripts/eval_robustness_sweep.py", "--metadata", $Metadata,
            "--normalized-dir", $NormalizedDir, "--multimodal",
            "--wandb-run-name", "phase3-robustness"
        ) + $CapArgs)
}

# 2. Phase 4 adversarial: epsilon grid + methods from -EpsilonGrid/-Methods, else
#    eval_adversarial_sweep.py defaults (5 epsilons x FGSM+PGD).
if (-not $SkipAdversarial) {
    $AdvArgs = @()
    if ($EpsilonGrid.Count -gt 0) { $AdvArgs += @("--epsilon-grid") + $EpsilonGrid }
    if ($Methods.Count -gt 0) { $AdvArgs += @("--methods") + $Methods }
    # Per-config resume checkpoint (separate file per config: unimodal-video and
    # mm-video share the (method,"video",eps) key, so they must NOT share a file).
    if ($ResumeDir) { New-Item -ItemType Directory -Force -Path $ResumeDir | Out-Null }
    function Resume-Arg([string]$tag) {
        if ($ResumeDir) { return @("--resume-csv", (Join-Path $ResumeDir "adv_$tag.csv")) } else { return @() }
    }

    Invoke-Step "Phase 4 adversarial (video)" (@(
            "scripts/eval_adversarial_sweep.py", "--metadata", $Metadata,
            "--normalized-dir", $NormalizedDir, "--wandb-run-name", "phase4-adv-video"
        ) + $CapArgs + $AdvArgs + (Resume-Arg "video"))

    foreach ($mod in @("audio", "video", "both")) {
        Invoke-Step "Phase 4 adversarial (multimodal / $mod)" (@(
                "scripts/eval_adversarial_sweep.py", "--metadata", $Metadata,
                "--normalized-dir", $NormalizedDir, "--multimodal",
                "--attack-modalities", $mod, "--wandb-run-name", "phase4-adv-mm-$mod"
            ) + $CapArgs + $AdvArgs + (Resume-Arg "mm-$mod"))
    }
}

# 3. UAP: video/multimodal x REAL/FAKE, real fit (all chunks) + fake-enriched eval.
if (-not $SkipUap) {
    foreach ($m in @("video", "multimodal")) {
        foreach ($t in @("REAL", "FAKE")) {
            $pyArgs = @(
                "scripts/compute_uap.py", "--modality", $m, "--target-class", $t,
                "--fit-metadata", $FitMetadata, "--eval-metadata", $EvalMetadata,
                "--epochs", "$UapEpochs", "--eval-balanced", "$UapEvalBalanced",
                "--output-dir", $OutputDir, "--wandb-run-name", "uap-$m-$($t.ToLower())"
            )
            if ($UapMaxFitChunks -gt 0) { $pyArgs += @("--max-fit-chunks", "$UapMaxFitChunks") }
            if ($m -eq "multimodal") { $pyArgs += @("--attack-modalities", "both") }
            Invoke-Step "UAP $m -> $t" $pyArgs
        }
    }
}

# --- Summary -------------------------------------------------------------------
Write-Host "`n================ RUN SUMMARY ================" -ForegroundColor Cyan
$results | Format-Table -AutoSize Step, Ok, Code, Minutes | Out-Host
$totalMin = ($results | Measure-Object -Property Minutes -Sum).Sum
Write-Host ("Total wall time: {0:n1} min across {1} step(s)." -f $totalMin, $results.Count) -ForegroundColor DarkGray
Write-Host "Transcript: $Transcript" -ForegroundColor DarkGray

$failed = @($results | Where-Object { -not $_.Ok })
Stop-Transcript | Out-Null
if ($failed.Count -gt 0) {
    Write-Host ("{0} step(s) FAILED -- inspect the transcript; re-run just those with the -Skip* switches." -f $failed.Count) -ForegroundColor Red
    exit 1
}
Write-Host "All steps passed." -ForegroundColor Green
exit 0
