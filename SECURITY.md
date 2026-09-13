# Security Policy

## Scope

This policy covers security issues in the NexMind source tree and project infrastructure.

## Do not disclose secrets

Do not commit passwords, API keys, access tokens, private keys, cloud credentials, or other authentication material.

If a secret is accidentally committed, revoke or rotate it immediately. Removing it from a later commit does not make the exposed credential safe.

## Reporting

For a suspected vulnerability that could affect users or infrastructure, please report it privately to the project maintainer rather than publishing working exploit details in a public issue.

Until a dedicated security contact is configured, use the repository owner's GitHub account to request a private security discussion.

## Responsible disclosure

Please provide enough information to reproduce and validate the issue, while avoiding unnecessary disclosure of sensitive data.
