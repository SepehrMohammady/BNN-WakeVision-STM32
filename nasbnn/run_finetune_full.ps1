# run_finetune_full.ps1
# Fine-tune NAS-BNN architectures (keys 3,4,5,6) on the FULL WakeVision dataset.
# Architecture definitions reused from existing search results.
# Run from: c:\Projects\PhD\NAS-BNN\WakeVision
# Usage: .\run_finetune_full.ps1

Set-Location "c:\Projects\PhD\NAS-BNN\WakeVision"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& "c:\Projects\PhD\NAS-BNN\WakeVision\nasbnn_env\Scripts\Activate.ps1"

# ── Paths ─────────────────────────────────────────────────────────────────────
$DATA_PATH      = "./data/WakeVision_Full"
$SEARCH_INFO    = "./work_dirs/wakevision_nasbnn_LARGEXP_run/search/info.pth.tar"
$SUPERNET_CKPT  = "./work_dirs/wakevision_nasbnn_LARGEXP_run/checkpoint.pth.tar"
$OUT_BASE       = "./work_dirs/wakevision_nasbnn_FULLEXP_run"

# ── Hyperparameters ────────────────────────────────────────────────────────────
$ARCH           = "superbnn_wakevision_large"
$DATASET        = "WakeVision"
$IMG_SIZE       = 128
$BATCH          = 256      # 512 caused VRAM overflow → shared RAM spill → slowdown
$LR             = "1e-4"   # Linear scale: 2e-4 * (256/512) = 1e-4
$WD             = 0
$EPOCHS         = 30       # Fewer epochs needed with larger batch + better LR
$WORKERS        = 12       # More prefetch threads for 5.76M images
$GPU            = 0
$PRINT_FREQ     = 50

# ── Keys to fine-tune ──────────────────────────────────────────────────────────
$KEYS = @(3, 4, 5, 6)

foreach ($KEY in $KEYS) {
    $OUT_DIR = "$OUT_BASE/finetuned_ops_key$KEY"
    New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  Fine-tuning Key $KEY  →  $OUT_DIR" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan

    $CMD_PARTS = @(
        "python", "train_single.py",
        "--dataset", $DATASET,
        "-a", $ARCH,
        "--img-size", $IMG_SIZE,
        "-b", $BATCH,
        "--lr", $LR,
        "--wd", $WD,
        "--epochs", $EPOCHS,
        "--ops", $KEY,
        "--workers", $WORKERS,
        "--pretrained", $SUPERNET_CKPT,
        "--gpu", $GPU,
        "--print-freq", $PRINT_FREQ,
        "--save-freq", "1",
        $DATA_PATH,
        $SEARCH_INFO,
        $OUT_DIR
    )

    # Resume if checkpoint exists
    $RESUME_CKPT = "$OUT_DIR/checkpoint.pth.tar"
    if (Test-Path $RESUME_CKPT) {
        Write-Host "  Resuming from $RESUME_CKPT" -ForegroundColor Yellow
        $CMD_PARTS += @("--resume", $RESUME_CKPT)
    } else {
        Write-Host "  Starting fresh (pretrained from supernet)" -ForegroundColor Green
    }

    & $CMD_PARTS[0] $CMD_PARTS[1..($CMD_PARTS.Length-1)]

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Fine-tuning Key $KEY failed (exit code $LASTEXITCODE)" -ForegroundColor Red
        Write-Host "Stopping. Fix the error and re-run; completed keys will be skipped (resume)." -ForegroundColor Red
        exit $LASTEXITCODE
    }

    Write-Host "Key $KEY done." -ForegroundColor Green
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  All 4 keys fine-tuned on full WakeVision dataset." -ForegroundColor Green
Write-Host "  Results in: $OUT_BASE" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
