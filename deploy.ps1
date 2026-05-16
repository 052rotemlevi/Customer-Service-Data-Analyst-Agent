<#
.SYNOPSIS
    Deploy the CS Agent to AWS (full stack).

.DESCRIPTION
    Step 1: Request ACM certificate (if needed)
    Step 2: SAM build + deploy (Lambda, API GW, DynamoDB, S3, CloudFront, Route53)
    Step 3: Upload frontend to S3
    Step 4: Invalidate CloudFront cache

.PARAMETER StackName
    CloudFormation stack name (default: cs-agent)

.PARAMETER Region
    AWS region (default: us-east-1, required for ACM + CloudFront)

.PARAMETER DomainName
    Custom domain (default: CS-Agent.demo.Rotem.click)

.PARAMETER HostedZoneId
    Route 53 Hosted Zone ID for the domain

.EXAMPLE
    .\deploy.ps1 -HostedZoneId Z0123456789ABCDEF
#>

param(
    [string]$StackName = "cs-agent",
    [string]$Region = "us-east-1",
    [string]$DomainName = "cs-agent.demo.rotem.click",
    [Parameter(Mandatory=$true)]
    [string]$HostedZoneId
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  CS Agent — AWS Full-Stack Deploy" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Stack:    $StackName"
Write-Host "  Region:   $Region"
Write-Host "  Domain:   $DomainName"
Write-Host "  Zone ID:  $HostedZoneId"
Write-Host "========================================`n" -ForegroundColor Cyan

# ── Prerequisites ──
Write-Host "[1/6] Checking prerequisites..." -ForegroundColor Yellow
foreach ($cmd in @("aws", "sam")) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Write-Error "$cmd CLI not found. Please install it first."
        exit 1
    }
}
Write-Host "  OK" -ForegroundColor Green

# ── ACM Certificate ──
Write-Host "`n[2/6] Checking ACM certificate for $DomainName..." -ForegroundColor Yellow
$CertArn = aws acm list-certificates --region $Region --query "CertificateSummaryList[?DomainName=='$DomainName'].CertificateArn" --output text 2>$null

if (-not $CertArn -or $CertArn -eq "None") {
    Write-Host "  Requesting new ACM certificate..." -ForegroundColor Yellow
    $CertArn = aws acm request-certificate `
        --domain-name $DomainName `
        --validation-method DNS `
        --region $Region `
        --query "CertificateArn" `
        --output text

    Write-Host "  Certificate ARN: $CertArn" -ForegroundColor Cyan
    Write-Host "  Waiting for DNS validation records..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10

    # Get the DNS validation record
    $ValidationJson = aws acm describe-certificate `
        --certificate-arn $CertArn `
        --region $Region `
        --query "Certificate.DomainValidationOptions[0].ResourceRecord" `
        --output json
    $Validation = $ValidationJson | ConvertFrom-Json

    Write-Host "`n  Creating DNS validation record in Route 53..." -ForegroundColor Yellow
    $ChangeBatch = @{
        Changes = @(@{
            Action = "UPSERT"
            ResourceRecordSet = @{
                Name = $Validation.Name
                Type = $Validation.Type
                TTL = 300
                ResourceRecords = @(@{ Value = $Validation.Value })
            }
        })
    } | ConvertTo-Json -Depth 5 -Compress

    $ChangeBatch | Out-File -FilePath "$env:TEMP\dns-change.json" -Encoding utf8
    aws route53 change-resource-record-sets `
        --hosted-zone-id $HostedZoneId `
        --change-batch "file://$env:TEMP\dns-change.json"

    Write-Host "  Waiting for certificate validation (this may take a few minutes)..." -ForegroundColor Yellow
    aws acm wait certificate-validated --certificate-arn $CertArn --region $Region
    Write-Host "  Certificate validated!" -ForegroundColor Green
} else {
    Write-Host "  Using existing certificate: $CertArn" -ForegroundColor Green
}

# ── Nebius API Key ──
Write-Host "`n[3/6] Reading Nebius API key..." -ForegroundColor Yellow
$ApiKey = ""
$EnvFile = Join-Path $ProjectRoot ".env"
if (Test-Path $EnvFile) {
    $line = Get-Content $EnvFile | Where-Object { $_ -match "^NEBIUS_API_KEY=" }
    if ($line) { $ApiKey = $line -replace "^NEBIUS_API_KEY=", "" }
}
if (-not $ApiKey) {
    $SecureKey = Read-Host "Enter your Nebius API key" -AsSecureString
    $BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureKey)
    $ApiKey = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)
}
Write-Host "  OK" -ForegroundColor Green

# ── SAM Build ──
Write-Host "`n[4/6] Building SAM application..." -ForegroundColor Yellow
Push-Location $ProjectRoot
sam build --template-file template.yaml --region $Region
Pop-Location

# ── SAM Deploy ──
Write-Host "`n[5/6] Deploying to AWS..." -ForegroundColor Yellow
Push-Location $ProjectRoot
sam deploy `
    --stack-name $StackName `
    --region $Region `
    --capabilities CAPABILITY_IAM `
    --parameter-overrides `
        "NebiusApiKey=$ApiKey" `
        "DomainName=$DomainName" `
        "HostedZoneId=$HostedZoneId" `
        "CertificateArn=$CertArn" `
    --resolve-s3 `
    --no-confirm-changeset `
    --no-fail-on-empty-changeset
Pop-Location

# ── Upload Frontend ──
Write-Host "`n[6/6] Uploading frontend to S3..." -ForegroundColor Yellow
$BucketName = aws cloudformation describe-stacks `
    --stack-name $StackName `
    --region $Region `
    --query "Stacks[0].Outputs[?OutputKey=='FrontendBucketName'].OutputValue" `
    --output text

aws s3 sync "$ProjectRoot\frontend" "s3://$BucketName/" --delete --cache-control "max-age=3600"

# Invalidate CloudFront
$CfId = aws cloudfront list-distributions `
    --query "DistributionList.Items[?Comment=='CS Agent - $DomainName'].Id" `
    --output text

if ($CfId -and $CfId -ne "None") {
    Write-Host "  Invalidating CloudFront cache..." -ForegroundColor Yellow
    aws cloudfront create-invalidation --distribution-id $CfId --paths "/*" | Out-Null
}

# ── Done ──
Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  Deployment Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "  URL:        https://$DomainName" -ForegroundColor Cyan
Write-Host "  Dashboard:  https://$Region.console.aws.amazon.com/cloudwatch/home?region=$Region#dashboards:name=$StackName-dashboard" -ForegroundColor Cyan
Write-Host "`nTest:" -ForegroundColor Yellow
Write-Host "  curl https://$DomainName/api/health" -ForegroundColor White
Write-Host "  curl -X POST https://$DomainName/api/chat -H 'Content-Type: application/json' -d '{""query"": ""What categories exist?""}'" -ForegroundColor White
