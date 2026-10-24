#!/usr/bin/env bash
set -euo pipefail

# INSTALL CDK (in AWS CloudShell use sudo for the global npm install)
# sudo npm install -g aws-cdk@latest
# python3 -m venv .venv
# source .venv/bin/activate
# pip3 install -r requirements.txt

# AWS CloudShell only: jsii refuses a group/world-writable home directory
# chmod 755 ~

# DEPLOY STACK
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=$(aws configure get region)
cdk bootstrap aws://${ACCOUNT_ID}/${AWS_REGION}
cdk deploy --parameters ProjectName=mlflow --require-approval never
