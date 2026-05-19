param(
    [string]$FunctionName = "stat-arb-daily",
    [string]$EcrRepository = "stat-arb-lambda",
    [string]$ArtifactsBucket = "",
    [string]$ArtifactsPrefix = "stat-arb",
    [string]$ScheduleRuleName = "stat-arb-daily-rule",
    [string]$ScheduleExpression = "cron(30 21 ? * MON-FRI *)",
    [int]$MemorySize = 3008,
    [int]$TimeoutSeconds = 900,
    [switch]$ExecuteTrades,
    [switch]$AllowStale,
    [switch]$SkipResearch,
    [switch]$SkipReady,
    [switch]$SkipAlpaca
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

function Get-RequiredCommand {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $Name"
    }
}

function Assert-LastExitCode {
    param([string]$Context)
    if ($LASTEXITCODE -ne 0) {
        throw $Context
    }
}

function Get-AwsConfigValue {
    param([string]$Name)
    $value = aws configure get $Name 2>$null
    if ($LASTEXITCODE -ne 0) {
        return ""
    }
    return $value.Trim()
}

function Load-DotEnv {
    param([string]$Path)
    $values = @{}
    if (-not (Test-Path $Path)) {
        return $values
    }

    foreach ($line in Get-Content $Path) {
        if ($line -match '^\s*#' -or $line -notmatch '=') {
            continue
        }
        $parts = $line.Split('=', 2)
        $key = $parts[0].Trim()
        $value = $parts[1].Trim().Trim('"').Trim("'")
        if ($key) {
            $values[$key] = $value
        }
    }
    return $values
}

function New-TempJsonFile {
    param([object]$Value)
    $path = [System.IO.Path]::GetTempFileName()
    $Value | ConvertTo-Json -Depth 10 -Compress | Set-Content -Path $path -Encoding ascii
    return $path
}

function Ensure-Bucket {
    param(
        [string]$Bucket,
        [string]$Region
    )
    if ([string]::IsNullOrWhiteSpace($Bucket)) {
        throw "Artifacts bucket name is required."
    }

    try {
        aws s3api head-bucket --bucket $Bucket 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            return
        }
    } catch {
    }

    if ($Region -eq "us-east-1") {
        aws s3api create-bucket --bucket $Bucket | Out-Null
    } else {
        aws s3api create-bucket --bucket $Bucket --create-bucket-configuration LocationConstraint=$Region | Out-Null
    }
}

function Ensure-EcrRepository {
    param([string]$RepositoryName)
    $exists = $true
    try {
        aws ecr describe-repositories --repository-names $RepositoryName 2>$null | Out-Null
        $exists = ($LASTEXITCODE -eq 0)
    } catch {
        $exists = $false
    }
    if (-not $exists) {
        aws ecr create-repository --repository-name $RepositoryName | Out-Null
    }
}

function Ensure-LambdaRole {
    param(
        [string]$RoleName,
        [string]$BucketArn
    )

    $role = $null
    $roleExists = $true
    try {
        $role = aws iam get-role --role-name $RoleName --output json 2>$null | ConvertFrom-Json
        $roleExists = ($LASTEXITCODE -eq 0)
    } catch {
        $roleExists = $false
    }
    if (-not $roleExists) {
        $trustPolicyPath = New-TempJsonFile -Value @{
            Version = "2012-10-17"
            Statement = @(
                @{
                    Effect = "Allow"
                    Principal = @{ Service = "lambda.amazonaws.com" }
                    Action = "sts:AssumeRole"
                }
            )
        }

        $role = aws iam create-role --role-name $RoleName --assume-role-policy-document "file://$trustPolicyPath" --output json | ConvertFrom-Json
        Assert-LastExitCode "Failed to create IAM role $RoleName."
        aws iam attach-role-policy --role-name $RoleName --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole | Out-Null
        Assert-LastExitCode "Failed to attach CloudWatch Logs policy to IAM role $RoleName."
        Start-Sleep -Seconds 10
    }

    $policyPath = New-TempJsonFile -Value @{
        Version = "2012-10-17"
        Statement = @(
            @{
                Effect = "Allow"
                Action = @("s3:PutObject", "s3:AbortMultipartUpload")
                Resource = @("$BucketArn/*")
            },
            @{
                Effect = "Allow"
                Action = @("s3:ListBucket")
                Resource = @($BucketArn)
            }
        )
    }

    aws iam put-role-policy --role-name $RoleName --policy-name "$RoleName-s3-artifacts" --policy-document "file://$policyPath" | Out-Null
    Assert-LastExitCode "Failed to attach inline S3 policy to IAM role $RoleName."
    return $role.Role.Arn
}

Get-RequiredCommand "aws"
Get-RequiredCommand "docker"
docker info | Out-Null
Assert-LastExitCode "Docker Desktop is installed but not running. Start Docker Desktop, then rerun the deployment."

$repoRoot = Split-Path -Parent $PSScriptRoot
$envValues = Load-DotEnv -Path (Join-Path $repoRoot ".env")

$region = $env:AWS_REGION
if ([string]::IsNullOrWhiteSpace($region)) {
    $region = Get-AwsConfigValue "region"
}
if ([string]::IsNullOrWhiteSpace($region)) {
    throw "AWS region is not configured. Set AWS_REGION or run 'aws configure'."
}

$caller = aws sts get-caller-identity --output json | ConvertFrom-Json
$accountId = $caller.Account
$ecrUri = "$accountId.dkr.ecr.$region.amazonaws.com/$EcrRepository"
$roleName = "$FunctionName-role"
$bucketName = $ArtifactsBucket
if ([string]::IsNullOrWhiteSpace($bucketName)) {
    $bucketName = "$FunctionName-artifacts-$accountId-$($region.ToLower())"
}
$bucketArn = "arn:aws:s3:::$bucketName"

Write-Host "Ensuring S3 bucket $bucketName"
Ensure-Bucket -Bucket $bucketName -Region $region

Write-Host "Ensuring ECR repository $EcrRepository"
Ensure-EcrRepository -RepositoryName $EcrRepository

Write-Host "Logging Docker into ECR"
aws ecr get-login-password --region $region | docker login --username AWS --password-stdin "$accountId.dkr.ecr.$region.amazonaws.com"
Assert-LastExitCode "Failed to authenticate Docker to ECR."

Write-Host "Building Lambda image"
docker build --platform linux/amd64 --provenance=false -f Dockerfile.lambda -t "${FunctionName}:latest" $repoRoot
Assert-LastExitCode "Docker build failed."
docker tag "${FunctionName}:latest" "${ecrUri}:latest"
Assert-LastExitCode "Docker tag failed."

Write-Host "Pushing Lambda image"
docker push "${ecrUri}:latest"
Assert-LastExitCode "Docker push failed."

Write-Host "Ensuring Lambda role $roleName"
$roleArn = Ensure-LambdaRole -RoleName $roleName -BucketArn $bucketArn

$environmentMap = @{
    ALPACA_API_KEY = $envValues["ALPACA_API_KEY"]
    ALPACA_SECRET_KEY = $envValues["ALPACA_SECRET_KEY"]
    ALPACA_BASE_URL = $envValues["ALPACA_BASE_URL"]
    ALPACA_DRY_RUN = $(if ($ExecuteTrades) { "false" } else { "true" })
    HOME = "/tmp"
    MPLCONFIGDIR = "/tmp/matplotlib"
    STAT_ARB_ARTIFACTS_BUCKET = $bucketName
    STAT_ARB_ARTIFACTS_PREFIX = $ArtifactsPrefix
    STAT_ARB_SKIP_RESEARCH = $(if ($SkipResearch) { "true" } else { "false" })
    STAT_ARB_SKIP_READY = $(if ($SkipReady) { "true" } else { "false" })
    STAT_ARB_SKIP_ALPACA = $(if ($SkipAlpaca) { "true" } else { "false" })
    STAT_ARB_EXECUTE_TRADES = $(if ($ExecuteTrades) { "true" } else { "false" })
    STAT_ARB_ALLOW_STALE = $(if ($AllowStale) { "true" } else { "false" })
}

if ([string]::IsNullOrWhiteSpace($environmentMap["ALPACA_API_KEY"]) -or [string]::IsNullOrWhiteSpace($environmentMap["ALPACA_SECRET_KEY"])) {
    throw "Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in .env."
}
if ([string]::IsNullOrWhiteSpace($environmentMap["ALPACA_BASE_URL"])) {
    $environmentMap["ALPACA_BASE_URL"] = "https://paper-api.alpaca.markets"
}

$environmentPath = New-TempJsonFile -Value @{ Variables = $environmentMap }

$function = $null
$functionExists = $true
try {
    $function = aws lambda get-function --function-name $FunctionName --output json 2>$null | ConvertFrom-Json
    $functionExists = ($LASTEXITCODE -eq 0)
} catch {
    $functionExists = $false
}
if (-not $functionExists) {
    Write-Host "Creating Lambda function $FunctionName"
    aws lambda create-function `
        --function-name $FunctionName `
        --package-type Image `
        --code ImageUri="${ecrUri}:latest" `
        --role $roleArn `
        --memory-size $MemorySize `
        --timeout $TimeoutSeconds `
        --environment "file://$environmentPath" | Out-Null
    Assert-LastExitCode "Failed to create Lambda function $FunctionName."
} else {
    Write-Host "Updating Lambda code"
    aws lambda update-function-code --function-name $FunctionName --image-uri "${ecrUri}:latest" | Out-Null
    Assert-LastExitCode "Failed to update Lambda image for $FunctionName."
    aws lambda wait function-updated-v2 --function-name $FunctionName
    Assert-LastExitCode "Lambda code update did not finish for $FunctionName."
    Write-Host "Updating Lambda configuration"
    aws lambda update-function-configuration `
        --function-name $FunctionName `
        --role $roleArn `
        --memory-size $MemorySize `
        --timeout $TimeoutSeconds `
        --environment "file://$environmentPath" | Out-Null
    Assert-LastExitCode "Failed to update Lambda configuration for $FunctionName."
}

Write-Host "Waiting for Lambda to become active"
aws lambda wait function-active-v2 --function-name $FunctionName
Assert-LastExitCode "Lambda function $FunctionName did not become active."

Write-Host "Ensuring EventBridge rule $ScheduleRuleName"
$ruleArn = aws events put-rule --name $ScheduleRuleName --schedule-expression $ScheduleExpression --state ENABLED --output text --query RuleArn
Assert-LastExitCode "Failed to create or update EventBridge rule $ScheduleRuleName."

$functionArn = aws lambda get-function --function-name $FunctionName --query 'Configuration.FunctionArn' --output text
$targetsRequestPath = New-TempJsonFile -Value @{
    Rule = $ScheduleRuleName
    Targets = @(
        @{
            Id = "1"
            Arn = $functionArn
        }
    )
}

aws events put-targets --cli-input-json "file://$targetsRequestPath" | Out-Null
Assert-LastExitCode "Failed to add Lambda target to EventBridge rule $ScheduleRuleName."
try {
    aws lambda add-permission `
        --function-name $FunctionName `
        --statement-id "$ScheduleRuleName-invoke" `
        --action lambda:InvokeFunction `
        --principal events.amazonaws.com `
        --source-arn $ruleArn 2>$null | Out-Null
} catch {
}

Write-Host ""
Write-Host "Deployment complete."
Write-Host "Function: $FunctionName"
Write-Host "Region: $region"
Write-Host "Artifacts bucket: s3://$bucketName/$ArtifactsPrefix/"
Write-Host "Schedule: $ScheduleExpression"
Write-Host "Invoke manually with:"
Write-Host "aws lambda invoke --function-name $FunctionName response.json"
