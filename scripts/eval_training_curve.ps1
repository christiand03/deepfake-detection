<#
.SYNOPSIS
    Evaluate every intermediate checkpoint, giving localization as a function of
    training duration.

.DESCRIPTION
    The lambda sweep answers "how much localization per lambda, at a fixed 6000-batch
    budget". It cannot answer whether that budget is enough -- and the re-run showed it
    probably is not: localization more than doubled between batch 3000 and 6000
    (lambda=0.02: 3.41 -> 8.21) with no sign of a plateau.

    Because save_top_k is now -1, every validation checkpoint survives, so the whole
    curve can be measured WITHOUT retraining anything. Each arm contributes 2-4 points
    at batches 1500/3000/4500/6000.

    Checkpoints already evaluated at batch 6000 are skipped; their results are reused
    from temp/loc_final_*.json.

.EXAMPLE
    powershell -File scripts/eval_training_curve.ps1
#>
param([string]$LogDir = "temp")

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-Stamp($Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Output $line
    Add-Content -Path "$LogDir/training_curve.log" -Value $line -Encoding utf8
}

# accum: optimizer steps -> batches. The lambda arms use manual optimization with
# loc_accumulate_grad_batches=3; the aux head uses automatic optimization with none.
$arms = @(
    @{ Name = "lambda0";   Dir = "logs/train/runs/2026-08-16_12-37-38"; Accum = 3 },
    @{ Name = "lambda002"; Dir = "logs/train/runs/2026-08-17_00-40-19"; Accum = 3 },
    @{ Name = "lambda01";  Dir = "logs/train/runs/2026-08-17_04-24-10"; Accum = 3 },
    @{ Name = "auxhead";   Dir = "logs/train/runs/2026-08-16_23-13-27"; Accum = 1 }
)

$total = 0
foreach ($arm in $arms) {
    $ckpts = Get-ChildItem "$($arm.Dir)/checkpoints" -Filter "*.ckpt" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match 'step_(\d+)' }
    $total += ($ckpts | Measure-Object).Count
}
Write-Stamp "training-curve evaluation: $total checkpoints across $($arms.Count) arms"

foreach ($arm in $arms) {
    $ckpts = Get-ChildItem "$($arm.Dir)/checkpoints" -Filter "*.ckpt" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match 'step_(\d+)' } |
        Sort-Object { [int]([regex]::Match($_.Name, 'step_(\d+)').Groups[1].Value) }

    foreach ($c in $ckpts) {
        $optStep = [int]([regex]::Match($c.Name, 'step_(\d+)').Groups[1].Value)
        $batch = $optStep * $arm.Accum
        $out = "$LogDir/loc_curve_$($arm.Name)_b$batch.json"

        if (Test-Path $out) { Write-Stamp "skip $($arm.Name) batch $batch (already evaluated)"; continue }

        Write-Stamp "evaluating $($arm.Name) batch $batch : $($c.Name)"
        & python -u -m scripts.eval_localization --ckpt $c.FullName --split test `
            --resume-csv "$LogDir/loc_curve_$($arm.Name)_b$batch.csv" `
            --summary-json $out 2>&1 |
            Tee-Object -FilePath "$LogDir/eval_curve_$($arm.Name)_b$batch.log" | Out-Null
        Write-Stamp "  -> exit $LASTEXITCODE"
    }
}

Write-Stamp "training-curve evaluation complete"
