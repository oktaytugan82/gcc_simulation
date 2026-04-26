param(
  [string]$DatasetRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) "data\ds004504-main"),
  [string]$DatasetId = "ds004504",
  [string]$Version = "1.0.8"
)

$ErrorActionPreference = "Stop"

function Invoke-GraphQL($Query) {
  $body = @{ query = $Query } | ConvertTo-Json -Compress
  return Invoke-RestMethod -Uri "https://openneuro.org/crn/graphql" -Method Post -ContentType "application/json" -Body $body
}

function Get-LocalSize($Path) {
  if (Test-Path -LiteralPath $Path) {
    return (Get-Item -LiteralPath $Path).Length
  }
  return 0
}

$query = "query { snapshot(datasetId: `"$DatasetId`", tag: `"$Version`") { files(recursive: true) { filename size directory annexed urls } } }"
$response = Invoke-GraphQL $query
$files = @($response.data.snapshot.files | Where-Object {
  -not $_.directory -and (
    $_.filename -eq "README" -or
    $_.filename -eq "dataset_description.json" -or
    $_.filename -eq "participants.json" -or
    $_.filename -eq "participants.tsv" -or
    $_.filename -like "derivatives/*"
  )
})

$totalBytes = ($files | Measure-Object -Property size -Sum).Sum
$doneBytes = 0L
$index = 0

Write-Host ("Downloading {0} files ({1:N2} GiB) from OpenNeuro {2} {3}" -f $files.Count, ($totalBytes / 1GB), $DatasetId, $Version)
Write-Host ("Target: {0}" -f $DatasetRoot)

foreach ($file in $files) {
  $index += 1
  $relative = [string]$file.filename
  $expected = [int64]$file.size
  $url = [string]$file.urls[0]
  $dest = Join-Path $DatasetRoot ($relative -replace '/', [IO.Path]::DirectorySeparatorChar)
  $dir = Split-Path -Parent $dest

  if (-not (Test-Path -LiteralPath $dir)) {
    New-Item -ItemType Directory -Path $dir | Out-Null
  }

  $current = Get-LocalSize $dest
  if ($current -eq $expected -and $expected -gt 0) {
    Write-Host ("[{0}/{1}] OK existing {2} ({3:N2} MiB)" -f $index, $files.Count, $relative, ($expected / 1MB))
    $doneBytes += $expected
    continue
  }

  Write-Host ("[{0}/{1}] Download {2} ({3:N2} MiB)" -f $index, $files.Count, $relative, ($expected / 1MB))

  $curlArgs = @(
    "-L",
    "--fail",
    "--retry", "5",
    "--retry-delay", "5",
    "--connect-timeout", "30",
    "--output", $dest
  )

  if ($current -gt 1048576 -and $current -lt $expected) {
    $curlArgs = @(
      "-L",
      "--fail",
      "--retry", "5",
      "--retry-delay", "5",
      "--connect-timeout", "30",
      "--continue-at", "-",
      "--output", $dest
    )
  }

  & curl.exe @curlArgs $url
  if ($LASTEXITCODE -ne 0) {
    throw "curl failed for $relative with exit code $LASTEXITCODE"
  }

  $after = Get-LocalSize $dest
  if ($after -ne $expected) {
    throw "Size mismatch for $relative`: expected $expected bytes, got $after bytes"
  }

  $doneBytes += $expected
  Write-Host ("      verified; cumulative {0:N2}/{1:N2} GiB" -f ($doneBytes / 1GB), ($totalBytes / 1GB))
}

Write-Host "Download complete."
