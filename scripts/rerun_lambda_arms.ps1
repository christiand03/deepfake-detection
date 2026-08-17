<#
.SYNOPSIS
    Re-run the lambda>0 sweep arms so every arm has a batch-6000 checkpoint.

.DESCRIPTION
    The first sweep trained all arms for 6000 batches, but save_top_k=2 (mode=min)
    stopped writing checkpoints once val/loss began to rise -- which it does for every
    arm with a localization penalty, because that rise IS the measured trade-off. The
    lambda>0 arms were therefore evaluated at batch 3000 while the control was evaluated
    at batch 6000, so the trade-off curve mixed points from different training amounts.

    save_top_k is now -1, so every validation writes a checkpoint. Re-training is the
    only way to recover the batch-6000 weights: they were never written to disk.

    The control (lambda=0) is NOT re-trained. Its val/loss falls, so its checkpoints
    saved through to batch 6000 and its existing result is already correct. It is only
    re-evaluated, so all three numbers come from one pass of the same eval code.

    Sequential by necessity: Gate G2 measured 7.57 GB peak on an 8 GB card.

.EXAMPLE
    powershell -File scripts/rerun_lambda_arms.ps1
#>
param(
    [string]$LogDir = "temp",
    [switch]$EvalOnly
)

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-Stamp($Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Output $line
    Add-Content -Path "$LogDir/rerun_lambda.log" -Value $line -Encoding utf8
}

# Returns the checkpoint with the highest step, parsed from the filename rather than by
# mtime or by trusting last.ckpt. last.ckpt is a copy of the most recent SAVE EVENT, not
# an end-of-training write, which is exactly how the previous run silently evaluated the
# wrong weights.
function Get-FinalCheckpoint($RunDir) {
    $ckpts = Get-ChildItem "$RunDir/checkpoints" -Filter "*.ckpt" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match 'step_(\d+)' }
    if (-not $ckpts) { return $null }
    return $ckpts | Sort-Object { [int]([regex]::Match($_.Name, 'step_(\d+)').Groups[1].Value) } |
        Select-Object -Last 1
}

$arms = @(
    @{ Name = "lambda002"; Exp = "sweep_relevance_lambda002"; Train = $true },
    @{ Name = "lambda01";  Exp = "sweep_relevance_lambda01";  Train = $true },
    @{ Name = "lambda0";   Exp = "sweep_relevance_lambda0";   Train = $false;
       RunDir = "logs/train/runs/2026-08-16_12-37-38" }
)

Write-Stamp "re-run started; training 2 arms, re-evaluating 3"

# The VS Code DVC extension re-hashes the 220 GB dataset on a timer and competes for the
# same disk the dataloader reads from. Clearing it once buys a quieter start; it will
# respawn, which is a known and accepted cost for this run.
$dvc = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*dvc data status*' }
if ($dvc) {
    $dvc | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Write-Stamp "stopped $($dvc.Count) dvc status process(es) competing for disk"
}

if (-not $EvalOnly) {
    foreach ($arm in $arms | Where-Object { $_.Train }) {
        Write-Stamp "=== training $($arm.Name) ==="
        Write-Stamp "GPU before: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>$null)"
        & python -u src/train.py experiment=$($arm.Exp) logger=csv extras.enforce_tags=false test=false 2>&1 |
            Tee-Object -FilePath "$LogDir/rerun_$($arm.Name).log"
        Write-Stamp "$($arm.Name) finished with exit code $LASTEXITCODE"

        $dir = Get-ChildItem "logs/train/runs" -Directory | Sort-Object CreationTime -Descending | Select-Object -First 1
        $arm.RunDir = $dir.FullName
        Write-Stamp "$($arm.Name) run dir: $($dir.Name)"

        $final = Get-FinalCheckpoint $dir.FullName
        if ($final) { Write-Stamp "$($arm.Name) final checkpoint: $($final.Name)" }
        else { Write-Stamp "WARNING: $($arm.Name) produced no step-tagged checkpoint" }
    }
}

Write-Stamp "=== evaluating (all arms at their FINAL checkpoint) ==="
foreach ($arm in $arms) {
    if (-not $arm.RunDir) { Write-Stamp "no run dir for $($arm.Name) - skipping"; continue }
    $ckpt = Get-FinalCheckpoint $arm.RunDir
    if (-not $ckpt) { Write-Stamp "no step-tagged checkpoint for $($arm.Name) - skipping"; continue }

    Write-Stamp "evaluating $($arm.Name): $($ckpt.Name)"
    Remove-Item "$LogDir/loc_final_$($arm.Name).csv" -Force -ErrorAction SilentlyContinue
    & python -u -m scripts.eval_localization --ckpt $ckpt.FullName --split test `
        --resume-csv "$LogDir/loc_final_$($arm.Name).csv" `
        --summary-json "$LogDir/loc_final_$($arm.Name).json" 2>&1 |
        Tee-Object -FilePath "$LogDir/eval_final_$($arm.Name).log"
    Write-Stamp "$($arm.Name) eval finished with exit code $LASTEXITCODE"
}

Write-Stamp "re-run complete - compare temp/loc_final_*.json"
