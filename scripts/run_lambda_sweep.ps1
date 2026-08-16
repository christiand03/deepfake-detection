<#
.SYNOPSIS
    Run the three-point lambda sweep for the relevance regularization, in sequence.

.DESCRIPTION
    Runs sweep_relevance_lambda0 (control), lambda002 and lambda01 back to back, then
    evaluates every resulting checkpoint with scripts/eval_localization.py against the
    same 624 test clips as the baseline.

    Sequential by necessity, not preference: Gate G2 measured 7.57 GB peak on an 8 GB
    card, so two runs at once would spill to shared memory and slow both by roughly an
    order of magnitude.

    Each stage continues even if the previous one exits non-zero. The arms are
    independent measurements of the same trade-off curve; a queue that abandons the
    remaining points because one hiccupped would waste the night. Exit codes are logged.

.EXAMPLE
    powershell -File scripts/run_lambda_sweep.ps1
    powershell -File scripts/run_lambda_sweep.ps1 -SkipTraining   # evaluate only
#>
param(
    [string]$LogDir = "temp",
    [switch]$SkipTraining
)

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-Stamp($Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Output $line
    Add-Content -Path "$LogDir/lambda_sweep.log" -Value $line -Encoding utf8
}

$arms = @(
    @{ Name = "lambda0";   Exp = "sweep_relevance_lambda0" },
    @{ Name = "lambda002"; Exp = "sweep_relevance_lambda002" },
    @{ Name = "lambda01";  Exp = "sweep_relevance_lambda01" }
)

Write-Stamp "lambda sweep started ($($arms.Count) arms)"

if (-not $SkipTraining) {
    foreach ($arm in $arms) {
        Write-Stamp "=== training $($arm.Name) ==="
        Write-Stamp "GPU before: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>$null)"
        & python -u src/train.py experiment=$($arm.Exp) logger=csv extras.enforce_tags=false test=false 2>&1 |
            Tee-Object -FilePath "$LogDir/sweep_$($arm.Name).log"
        Write-Stamp "$($arm.Name) finished with exit code $LASTEXITCODE"

        # Record the run directory NOW, while it is unambiguously the newest. Resolving
        # it later by searching config files is unsafe: the experiment names share
        # prefixes ("sweep_relevance_lambda0" is a prefix of "...lambda01" and
        # "...lambda002"), so a substring search matched the wrong run and evaluated
        # lambda0 against lambda01's checkpoint.
        $dir = Get-ChildItem "logs/train/runs" -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        $arm.RunDir = $dir.FullName
        Write-Stamp "$($arm.Name) run dir: $($dir.Name)"
    }
}

# ── Evaluate every checkpoint the sweep produced ──────────────────────────────
# The training metrics are single-sample-per-step and far too noisy to compare arms.
# scripts/eval_localization.py on the shared 624-clip test set is the pre-registered
# metric, and running all arms through it is what makes Gate G4 a like-for-like test.
Write-Stamp "=== evaluating checkpoints ==="
foreach ($arm in $arms) {
    if (-not $arm.RunDir) { Write-Stamp "no run dir recorded for $($arm.Name) - skipping eval"; continue }

    # last.ckpt, deliberately: it is the only file guaranteed to hold the END of
    # training, which is what makes the arms step-matched. The monitored checkpoints
    # are NOT comparable across arms -- save_top_k keeps the lowest val/loss, and
    # val/loss falls for the control but rises for the penalised arms, so the retained
    # files are late for one and early for the others.
    $ckpt = Get-Item "$($arm.RunDir)/checkpoints/last.ckpt" -ErrorAction SilentlyContinue
    if (-not $ckpt) { Write-Stamp "no last.ckpt for $($arm.Name) - skipping eval"; continue }

    Write-Stamp "evaluating $($arm.Name): $($ckpt.FullName)"
    & python -u -m scripts.eval_localization --ckpt $ckpt.FullName --split test `
        --resume-csv "$LogDir/loc_sweep_$($arm.Name).csv" `
        --summary-json "$LogDir/loc_sweep_$($arm.Name).json" 2>&1 |
        Tee-Object -FilePath "$LogDir/eval_sweep_$($arm.Name).log"
    Write-Stamp "$($arm.Name) eval finished with exit code $LASTEXITCODE"
}

Write-Stamp "sweep complete - compare temp/loc_sweep_*.json against temp/loc_baseline.json"
