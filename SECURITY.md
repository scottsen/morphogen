# Security Policy

## Supported Versions

Morphogen is currently in pre-1.0 development. Security updates will be provided for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.11.x  | :white_check_mark: |
| 0.10.x  | :white_check_mark: |
| < 0.10  | :x:                |

Once Morphogen reaches v1.0, we will maintain security updates for the current major version and the previous major version.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

If you discover a security vulnerability in Morphogen, please report it privately to help us address it before public disclosure.

### How to Report

1. **Email** (preferred): Create a private security advisory via GitHub:
   - Go to https://github.com/scottsen/morphogen/security/advisories
   - Click "Report a vulnerability"
   - Provide detailed information about the vulnerability

2. **Alternative**: If you cannot use GitHub security advisories, please open a regular issue with the title "SECURITY: [Brief Description]" and mark it as sensitive. We will respond promptly and move the discussion to a private channel.

### What to Include

Please include the following information in your report:

- **Type of vulnerability** (e.g., code execution, injection, authentication bypass)
- **Affected component** (e.g., compiler, runtime, specific domain)
- **Affected versions** (if known)
- **Steps to reproduce** the vulnerability
- **Potential impact** of the vulnerability
- **Suggested fix** (if you have one)
- **Any proof-of-concept code** (if applicable)

### Response Timeline

- **Initial response**: Within 48 hours of report
- **Status update**: Within 7 days with assessment and next steps
- **Fix timeline**: Depends on severity
  - Critical: Within 7 days
  - High: Within 30 days
  - Medium: Within 90 days
  - Low: Next regular release

### Disclosure Policy

- We follow **coordinated disclosure** practices
- We will work with you to understand and resolve the issue
- We ask that you do not publicly disclose the vulnerability until we have released a fix
- We will credit you in the security advisory (unless you prefer to remain anonymous)

### Security Updates

Security updates will be released as:
- Patch releases (e.g., 0.11.1) for supported versions
- Announced in release notes and CHANGELOG.md
- Documented in GitHub security advisories

### Scope

This security policy applies to:

- **Core Morphogen compiler and runtime**
- **Standard library implementations** (domains, operators)
- **Official examples** that could lead to vulnerabilities if used as templates
- **Build and distribution infrastructure**

Out of scope:
- Third-party dependencies (report to respective projects)
- User-written Morphogen programs (users are responsible for their own code)
- Theoretical attacks without practical impact

### Known Security Considerations

As Morphogen is a compiler and execution platform, users should be aware of:

1. **Code execution**: Morphogen programs execute arbitrary code. Only run programs from trusted sources.
2. **Deterministic RNG**: While deterministic execution is a feature, be aware that fixed seeds may reduce security in cryptographic contexts.
3. **External I/O**: Audio, visual, and file I/O operations interact with the host system. Review code that performs I/O operations.
4. **GPU execution**: GPU-accelerated code has access to GPU memory and resources.

### Security Best Practices for Users

- Only run Morphogen programs from trusted sources
- Review code before execution, especially code from unknown sources
- Use appropriate permissions when running programs that access files or devices
- Keep Morphogen updated to the latest version
- Report any suspicious behavior

## Public Security Advisories

Published security advisories are available at:
https://github.com/scottsen/morphogen/security/advisories

## Contact

For general security questions (not vulnerability reports), you can:
- Open a GitHub discussion: https://github.com/scottsen/morphogen/discussions
- Tag issues with the "security" label

Thank you for helping keep Morphogen and its users safe!
