<#
Remove unnecessary or superseded checkpoints. Dry run by default: lists what each
tier would remove and its size. Nothing is deleted without -Execute.

    .\scratch\cleanup_checkpoints.ps1                         # report all tiers
    .\scratch\cleanup_checkpoints.ps1 -Tiers ResumeSteps       # report one tier
    .\scratch\cleanup_checkpoints.ps1 -Tiers ResumeSteps,Blends -Execute

Tiers
  ResumeSteps   Intermediate state-step-* resume states (optimizer + weights) of finished
                runs. Finals and exported milestone models are kept.
  ResumeFinal   Final training-state resume dirs of finished runs, except the latest long
                round (long-r5) and the latest candidate (completion-v2/completion).
                Only needed to continue a run with its exact optimizer state.
  Blends        Blend variants nobody builds on (u30/u70/ramp*/qk...). Every blend is
                rebuilt from its two parents with merge_checkpoints; the lineage of the
                reference long1-u50 and the evaluated finalists stay.
  Experiments   Weights of early or superseded 2B experiments (conversion trials, LR
                sweeps, smoke/probe/bench runs, ablations, non-lineage on-policy rounds,
                superseded agentic pilots). configs, milestone.json and logs are kept.
  Legacy        Weights of the pre-hybrid sidecar/PLE era (..\runs, code_training,
                gr_retrofit). configs and logs are kept.

Never touched: teacher-hf, StudentSourceModel, student-*, gguf*, teacher caches,
merges-long1\u50 and its lineage, long-r1..r5 finals, the 10-05..10-08 experiment
finals (control arms, masked replay, execution-grounded, completion-v2).
#>
param(
    [string[]]$Tiers = @('ResumeSteps', 'ResumeFinal', 'Blends', 'Experiments', 'Legacy'),
    [switch]$Execute
)
$ErrorActionPreference = 'Stop'
# `powershell -File` passes "A,B" as one string; accept commas either way.
$valid = 'ResumeSteps', 'ResumeFinal', 'Blends', 'Experiments', 'Legacy'
$Tiers = @($Tiers | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
foreach ($t in $Tiers) { if ($valid -notcontains $t) { throw "unknown tier '$t'; choose from $($valid -join ', ')" } }
$root = 'D:\DeepThought\Projects\HybridModel'
$dg = "$root\DistillKit\scratch\dense_gr"
$weights = @('.safetensors', '.bin', '.pt', '.pth')

if ($Execute -and (Get-Process python -ErrorAction SilentlyContinue)) {
    throw 'python processes are running; stop training/evaluation before deleting checkpoints'
}

# Blends that are parents of long1-u50 (the reference) or were evaluated as finalists.
$keepBlends = @(
    'merges-r2\ramp0-70-tt', 'merges-r5b\ramp0-70', 'merges-r67\r7ramp0-70',
    'merges-r6\ramp0-70', 'merges-r8b\ramp0-70', 'merges-r6r8b\u50',
    'merges-long1\u50', 'merges-long4\u50', 'merges-long5\u50', 'merges-long5\ramp0-70'
) | ForEach-Object { "$dg\$_" }

$keepResume = @(
    "$dg\checkpoints-2b-long-r5\smoke-r1-1-gr-s25-csa2\training-state",
    "$dg\completion-v2\completion\checkpoints\smoke-r1-1-gr-s25-csa2\training-state"
)

# Whole directories (pure resume state) or weight files inside directories.
$plan = @{}
$plan.ResumeSteps = @{ Whole = $true; Dirs = @(Get-ChildItem $dg -Recurse -Directory -Filter 'state-step-*' |
    Select-Object -ExpandProperty FullName) }
$plan.ResumeFinal = @{ Whole = $true; Dirs = @(Get-ChildItem $dg -Recurse -Directory -Filter 'training-state' |
    Select-Object -ExpandProperty FullName | Where-Object { $keepResume -notcontains $_ }) }
$plan.Blends = @{ Whole = $false; Dirs = @(Get-ChildItem $dg -Directory -Filter 'merges-*' |
    ForEach-Object { Get-ChildItem $_.FullName -Directory } | Select-Object -ExpandProperty FullName |
    Where-Object { $keepBlends -notcontains $_ }) }

$experiments = @(
    'checkpoints', 'checkpoints2', 'checkpoints-arm', 'checkpoints-cont', 'checkpoints-conv',
    'checkpoints-decomp', 'checkpoints-smoke', 'checkpoints-smoke-ce', 'checkpoints-smoke-kd',
    'checkpoints-smoke-tp', 'checkpoints-sparse', 'checkpoints-sparse3', 'checkpoints-2b',
    'checkpoints-2b-borrow', 'checkpoints-2b-borrow-control', 'checkpoints-2b-borrow-ffuffu',
    'checkpoints-2b-borrow-fufufu', 'checkpoints-2b-borrow-fufufu-ownrot', 'checkpoints-2b-ceonly',
    'checkpoints-2b-chat', 'checkpoints-2b-control-r3', 'checkpoints-2b-finish',
    'checkpoints-2b-general-pilot-resumed', 'checkpoints-2b-hedge-both', 'checkpoints-2b-hedge-fix',
    'checkpoints-2b-hedge-nofix', 'checkpoints-2b-kd', 'checkpoints-2b-repair', 'checkpoints-2b-scale',
    'checkpoints-2b-sparse', 'checkpoints-ablate-capgen', 'checkpoints-ablate-plain',
    'checkpoints-ablate-strip', 'checkpoints-bench-head-old', 'checkpoints-bench-head-shared',
    'checkpoints-kahan-probe', 'checkpoints-memory-probe', 'checkpoints-memory-probe-16384',
    'checkpoints-memory-probe-32768', 'checkpoints-preflight-20260922',
    'repair-continuous-20260922', 'repair-continuous-repeat-20260922', 'repair-resume-20260922',
    'repair-resumed-20260922', 'checkpoints-2b-onpolicy-r7-smoke',
    # on-policy rounds outside the long1-u50 lineage (r2, r5b, r6, r7, r8b and thinking-pass are in it)
    'checkpoints-2b-onpolicy-r1', 'checkpoints-2b-onpolicy-r3', 'checkpoints-2b-onpolicy-r4',
    'checkpoints-2b-onpolicy-r5', 'checkpoints-2b-onpolicy-r8', 'checkpoints-2b-onpolicy-r8c',
    'checkpoints-2b-onpolicy-r9', 'checkpoints-2b-onpolicy-r9b',
    'agentic-pilot-v1\agentic\checkpoints', 'agentic-pilot-v1\replay\checkpoints',
    'agentic-pilot-v2-balanced\agentic\checkpoints', 'agentic-pilot-v2-balanced\replay\checkpoints'
) | ForEach-Object { "$dg\$_" }
$experiments += @(Get-ChildItem $dg -Directory -Filter 'sweep-lr-*' | Select-Object -ExpandProperty FullName)
$plan.Experiments = @{ Whole = $false; Dirs = $experiments }

$legacy = @("$root\DistillKit\scratch\code_training\checkpoints", "$root\DistillKit\scratch\gr_retrofit")
$legacy += @(Get-ChildItem "$root\runs" -Directory -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName)
$plan.Legacy = @{ Whole = $false; Dirs = $legacy }

$grand = 0
foreach ($tier in $Tiers) {
    $entry = $plan[$tier]
    $tierBytes = 0
    $targets = @()
    foreach ($dir in $entry.Dirs | Where-Object { Test-Path -LiteralPath $_ }) {
        if ($entry.Whole) {
            $files = @(Get-ChildItem -LiteralPath $dir -Recurse -File)
            $targets += , @($dir, $null)
        } else {
            # Weight files only; configs, milestone.json, merge.json and logs remain as the record.
            # State-step/training-state dirs below them belong to the Resume tiers.
            # Filter by extension explicitly: -Include is silently ignored with -LiteralPath,
            # which once made this tier delete every file (configs, logs, sources).
            $files = @(Get-ChildItem -LiteralPath $dir -Recurse -File |
                Where-Object { $weights -contains $_.Extension.ToLower() -and $_.FullName -notmatch '\\(state-step-\d+|training-state)\\' })
            if ($files.Count) { $targets += , @($dir, $files) }
        }
        $bytes = ($files | Measure-Object Length -Sum).Sum
        if ($bytes) {
            $tierBytes += $bytes
            '{0,8:N2} GB  {1}' -f ($bytes / 1GB), $dir.Replace("$root\", '')
        }
    }
    '== {0}: {1:N1} GB in {2} locations' -f $tier, ($tierBytes / 1GB), $targets.Count
    ''
    $grand += $tierBytes
    if ($Execute) {
        foreach ($t in $targets) {
            if ($null -eq $t[1]) { Remove-Item -LiteralPath $t[0] -Recurse -Force }
            else {
                $bad = @($t[1] | Where-Object { $weights -notcontains $_.Extension.ToLower() })
                if ($bad.Count) { throw "refusing to delete non-weight file $($bad[0].FullName)" }
                $t[1] | Remove-Item -Force
            }
        }
        "   deleted $tier"
    }
}
'TOTAL {0:N1} GB{1}' -f ($grand / 1GB), $(if ($Execute) { ' deleted' } else { ' (dry run; add -Execute to delete)' })
