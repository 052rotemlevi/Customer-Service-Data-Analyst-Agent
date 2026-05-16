# Deploy the Agent to AWS Lambda
# ================================
# Prerequisites: AWS CLI + SAM CLI installed and configured
#
# Install SAM CLI: https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html
#
# Usage:
#   deploy.ps1                          # Deploy with default stack name
#   deploy.ps1 -StackName "my-agent"    # Custom stack name

param(
    [string]$StackName = "customer-service-agent",
    [string]$Region = "us-east-1"
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Deploying Agent to AWS Lambda" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check prerequisites
Write-Host "`nChecking prerequisites..." -ForegroundColor Yellow
try { aws --version | Out-Null } catch { Write-Error "AWS CLI not found. Install: https://aws.amazon.com/cli/"; exit 1 }
try { sam --version | Out-Null } catch { Write-Error "SAM CLI not found. Install: https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html"; exit 1 }

# Prompt for API key
$ApiKey = Read-Host "Enter your Nebius API key" -AsSecureString
$BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($ApiKey)
$PlainApiKey = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)

# Build
Write-Host "`nBuilding SAM application..." -ForegroundColor Yellow
sam build --template-file template.yaml --region $Region

# Deploy
Write-Host "`nDeploying to AWS..." -ForegroundColor Yellow
sam deploy `
    --stack-name $StackName `
    --region $Region `
    --capabilities CAPABILITY_IAM `
    --parameter-overrides "NebiusApiKey=$PlainApiKey" `
    --resolve-s3 `
    --no-confirm-changeset

# Get output
Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  Deployment Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

$ApiUrl = aws cloudformation describe-stacks `
    --stack-name $StackName `
    --region $Region `
    --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" `
    --output text

Write-Host "`nAPI URL: $ApiUrl" -ForegroundColor Cyan
Write-Host "`nTest it:" -ForegroundColor Yellow
Write-Host "  curl -X POST `"$($ApiUrl)chat`" -H 'Content-Type: application/json' -d '{`"query`": `"What categories exist?`"}'" -ForegroundColor White
