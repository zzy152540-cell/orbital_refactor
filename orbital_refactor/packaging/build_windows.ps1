param(
    [string]$PythonExe = "python",
    [switch]$SkipSmoke,
    [switch]$SkipArchive
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SpecPath = Join-Path $PSScriptRoot "SatelliteStateEstimation.spec"
$ApplicationDir = Join-Path $ProjectRoot "dist\SatelliteStateEstimation"
$Executable = Join-Path $ApplicationDir "SatelliteStateEstimation.exe"

Push-Location $ProjectRoot
try {
    & $PythonExe -m PyInstaller --noconfirm --clean $SpecPath
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }

    Copy-Item -Recurse -Force "configs" (Join-Path $ApplicationDir "configs")
    New-Item -ItemType Directory -Force (Join-Path $ApplicationDir "results") | Out-Null
    Copy-Item -Force "packaging\README_zh.md" (Join-Path $ApplicationDir "README_zh.md")

    $Commit = (& git rev-parse HEAD).Trim()
    $BuildTime = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    @(
        "product=Satellite State Estimation"
        "version=0.4.0-rc2"
        "git_commit=$Commit"
        "platform=Windows-x64"
        "built_at_utc=$BuildTime"
    ) | Set-Content -Encoding UTF8 (Join-Path $ApplicationDir "VERSION.txt")

    if (-not $SkipSmoke) {
        $SmokeResult = Join-Path $ApplicationDir "results\package_smoke\interface_conformance.json"
        $WorkerArguments = @(
            "--worker-module",
            "experiments.run_interface_conformance_acceptance",
            "--output",
            "`"$SmokeResult`""
        )
        $Worker = Start-Process -FilePath $Executable -ArgumentList $WorkerArguments `
            -Wait -PassThru -WindowStyle Hidden
        if ($Worker.ExitCode -ne 0 -or -not (Test-Path $SmokeResult)) {
            throw "Packaged interface smoke test failed."
        }
        $Payload = Get-Content -Raw $SmokeResult | ConvertFrom-Json
        if (-not $Payload.overallPassed) { throw "Packaged interface verdict is FAIL." }

        $PreviousQpaPlatform = $env:QT_QPA_PLATFORM
        try {
            $env:QT_QPA_PLATFORM = "offscreen"
            $Gui = Start-Process -FilePath $Executable -ArgumentList "--gui-smoke" `
                -Wait -PassThru -WindowStyle Hidden
            if ($Gui.ExitCode -ne 0) { throw "Packaged GUI smoke test failed." }
        }
        finally {
            $env:QT_QPA_PLATFORM = $PreviousQpaPlatform
        }
    }

    if (-not $SkipArchive) {
        $ReleaseDir = Join-Path $ProjectRoot "release"
        $Archive = Join-Path $ReleaseDir "SatelliteStateEstimation-Windows-x64-0.4.0-rc2.zip"
        New-Item -ItemType Directory -Force $ReleaseDir | Out-Null
        if (Test-Path $Archive) { Remove-Item -LiteralPath $Archive -Force }
        Compress-Archive -Path (Join-Path $ApplicationDir "*") `
            -DestinationPath $Archive -CompressionLevel Optimal
        Write-Host "Release archive ready: $Archive"
    }

    Write-Host "Release package ready: $ApplicationDir"
}
finally {
    Pop-Location
}
