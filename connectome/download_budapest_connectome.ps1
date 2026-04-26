param(
    [string]$OutDir = (Join-Path $PSScriptRoot "data\budapest_connectome")
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$downloads = @(
    @{
        File = "brc_v3_default_20k_fibercount_median.csv"
        Url = "https://pitgroup.org/apps/connectome/getgraph.php?format=csv&version=2&population=0&minOccurrences=209&minStrength=0&combineMode=median&weightFunction=1&totalFiberNumber=0"
    },
    @{
        File = "brc_v3_default_20k_fibercount_median.graphml"
        Url = "https://pitgroup.org/apps/connectome/getgraph.php?format=graphml&version=2&population=0&minOccurrences=209&minStrength=0&combineMode=median&weightFunction=1&totalFiberNumber=0"
    },
    @{
        File = "brc_v3_20k_fibercount_median_conf25.csv"
        Url = "https://pitgroup.org/apps/connectome/getgraph.php?format=csv&version=2&population=0&minOccurrences=105&minStrength=0&combineMode=median&weightFunction=1&totalFiberNumber=0"
    },
    @{
        File = "brc_v3_20k_fibercount_median_conf10.csv"
        Url = "https://pitgroup.org/apps/connectome/getgraph.php?format=csv&version=2&population=0&minOccurrences=42&minStrength=0&combineMode=median&weightFunction=1&totalFiberNumber=0"
    },
    @{
        File = "brc_v3_200k_fibercount_median_conf50.csv"
        Url = "https://pitgroup.org/apps/connectome/getgraph.php?format=csv&version=2&population=0&minOccurrences=239&minStrength=0&combineMode=median&weightFunction=1&totalFiberNumber=1"
    },
    @{
        File = "brc_v3_1m_fibercount_median_conf50.csv"
        Url = "https://pitgroup.org/apps/connectome/getgraph.php?format=csv&version=2&population=0&minOccurrences=238&minStrength=0&combineMode=median&weightFunction=1&totalFiberNumber=2"
    },
    @{
        File = "brc_v3_20k_electrical_median_conf50.csv"
        Url = "https://pitgroup.org/apps/connectome/getgraph.php?format=csv&version=2&population=0&minOccurrences=209&minStrength=0&combineMode=median&weightFunction=0&totalFiberNumber=0"
    }
)

foreach ($download in $downloads) {
    $target = Join-Path $OutDir $download.File
    Write-Host "Downloading $($download.File)"
    curl.exe -L --fail --silent --show-error -o $target $download.Url
}

Write-Host "Done. Run prepare_budapest_connectome.py to build analysis-ready matrices."
