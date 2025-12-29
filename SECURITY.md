# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please report it responsibly.

### How to Report

1. **Do NOT open a public issue** for security vulnerabilities
2. Email: security@Chrissis-Tech.com
3. Or use GitHub Security Advisories (preferred):
   - Go to Security → Advisories → New draft advisory

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response Timeline

| Stage | Timeline |
|-------|----------|
| Acknowledgment | 48 hours |
| Initial assessment | 7 days |
| Fix development | 30 days (depending on severity) |
| Public disclosure | After fix is released |

### Severity Classification

| Severity | Criteria | Response |
|----------|----------|----------|
| Critical | Remote code execution, credential theft | Immediate patch |
| High | Data exposure, authentication bypass | Patch within 7 days |
| Medium | Limited data exposure, DoS | Patch within 30 days |
| Low | Minor issues, hardening | Next regular release |

## Security Best Practices

When using Meridian:

1. **API Keys**
   - Store in environment variables, never in code
   - Use separate keys for development/production
   - Rotate keys regularly

2. **Data Handling**
   - Do not evaluate prompts containing PII
   - Enable output redaction for sensitive contexts
   - Purge old evaluation data regularly

3. **Deployment**
   - Use authentication for web interface
   - Run behind a reverse proxy in production
   - Restrict database file permissions

4. **Updates**
   - Subscribe to releases for security updates
   - Review CHANGELOG.md before updating
   - Test updates in staging first

## Known Security Considerations

- Model outputs are stored in plaintext (use redaction for sensitive data)
- SQLite database has no access control (rely on filesystem permissions)
- Test suites are executed as-is (review external suites before use)

## Acknowledgments

We thank all security researchers who responsibly disclose vulnerabilities.

---

For general questions, use GitHub Discussions.
For security issues, follow the reporting process above.
