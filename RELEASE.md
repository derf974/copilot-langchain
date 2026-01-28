# Release Process

This document describes how to release a new version of `langchain-copilot` to PyPI.

## Prerequisites

Before creating a release, ensure you have:

1. ✅ **PyPI Trusted Publishing configured**
   - Go to [PyPI Publishing Settings](https://pypi.org/manage/account/publishing/)
   - Add Trusted Publisher:
     - PyPI project name: `langchain-copilot`
     - Owner: `derf974`
     - Repository name: `copilot-langchain`
     - Workflow name: `release.yml`
     - Environment name: `pypi`

2. ✅ **TestPyPI Trusted Publishing configured** (optional but recommended)
   - Go to [TestPyPI Publishing Settings](https://test.pypi.org/manage/account/publishing/)
   - Add Trusted Publisher with same settings as above but:
     - Environment name: `testpypi`

3. ✅ **GitHub Environments configured**
   - Go to repository Settings → Environments
   - Create `pypi` environment:
     - ✅ Required reviewers: Add yourself or team members
     - ✅ Deployment branches: Only `main`
   - Create `testpypi` environment (optional):
     - Reviewers optional for testing

4. ✅ **uv installed** for local version management
   ```bash
   pip install uv
   ```

## Release Workflow

### 1. Test on TestPyPI (Recommended)

Before releasing to production PyPI, test the package on TestPyPI:

#### Option A: Manual Trigger (workflow_dispatch)

1. Go to GitHub Actions → "Test Publish to TestPyPI"
2. Click "Run workflow"
3. Enter version (e.g., `0.2.1-rc1`)
4. Optional: Enable "Dry run" to test build without publishing
5. Click "Run workflow"

#### Option B: Local Testing

```bash
# Install from TestPyPI after publishing
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple \
            langchain-copilot==0.2.1-rc1

# Test the package
python -c "from langchain_copilot import CopilotChatModel; print('✅ Import successful')"
```

### 2. Prepare the Release

#### Update Version

Use `hatch version` to bump the version (follows Semantic Versioning):

```bash
# Patch release (0.2.0 → 0.2.1) - bug fixes
hatch version patch

# Minor release (0.2.0 → 0.3.0) - new features, backward compatible
hatch version minor

# Major release (0.2.0 → 1.0.0) - breaking changes
hatch version major

# Specific version
hatch version "1.0.0"
```

This automatically updates both `pyproject.toml` and `src/langchain_copilot/__init__.py`.

#### Update CHANGELOG.md

Edit [`CHANGELOG.md`](./CHANGELOG.md) to document changes for the new version:

```markdown
## [Unreleased]

## [0.2.1] - 2026-02-15

### Added
- New feature description

### Changed
- Change description

### Fixed
- Bug fix description

## [0.2.0] - 2026-01-28
...
```

Follow [Keep a Changelog](https://keepachangelog.com/) format with sections:
- `Added` - new features
- `Changed` - changes in existing functionality
- `Deprecated` - soon-to-be removed features
- `Removed` - removed features
- `Fixed` - bug fixes
- `Security` - security fixes

#### Commit Changes

```bash
# Commit version bump and changelog
git add pyproject.toml src/langchain_copilot/__init__.py CHANGELOG.md
git commit -m "chore: bump version to 0.2.1"
git push origin main
```

### 3. Create Release Tag

Create and push a Git tag to trigger the release workflow:

```bash
# Create tag (format: vX.Y.Z)
git tag v0.2.1

# Push tag to GitHub
git push origin v0.2.1
```

**⚠️ Important:** The tag version MUST match the version in `pyproject.toml` (without the `v` prefix).

### 4. Monitor Release Workflow

1. Go to GitHub Actions → "Release to PyPI"
2. Watch the workflow progress:
   - ✅ **GitHub Release** - Creates release with CHANGELOG notes
   - ✅ **Build** - Builds sdist and wheel, verifies version match
   - ✅ **PyPI Publish** - Publishes to PyPI (requires environment approval)

3. If using protected environments, approve the deployment when prompted

### 5. Verify Publication

After successful workflow completion:

1. **Check GitHub Release**
   - Go to [Releases](https://github.com/derf974/copilot-langchain/releases)
   - Verify release notes match CHANGELOG

2. **Check PyPI**
   - Visit https://pypi.org/project/langchain-copilot/
   - Verify new version appears
   - Check package metadata (description, links, etc.)

3. **Test Installation**
   ```bash
   # Create fresh virtual environment
   python -m venv test-env
   source test-env/bin/activate
   
   # Install from PyPI
   pip install langchain-copilot==0.2.1
   
   # Verify import
   python -c "from langchain_copilot import CopilotChatModel; print('✅ Success')"
   ```

## Troubleshooting

### Version Mismatch Error

If the workflow fails with "Version mismatch detected":

```bash
# Fix version in pyproject.toml or __init__.py
hatch version "0.2.1"  # Sets both files

# Delete and recreate tag
git tag -d v0.2.1
git push origin :refs/tags/v0.2.1
git tag v0.2.1
git push origin v0.2.1
```

### CHANGELOG Section Not Found

If GitHub Release has no body or generic message:

1. Ensure CHANGELOG.md has section `## [0.2.1]` matching the version
2. Check format matches [Keep a Changelog](https://keepachangelog.com/)
3. Verify no extra spaces in version header

### PyPI Publishing Fails

**If Trusted Publishing not configured:**
```
ERROR: Trusted publishing exchange failure
```

→ Configure Trusted Publisher on PyPI (see Prerequisites)

**If environment protection blocks:**
```
Waiting for environment protection to pass
```

→ Go to GitHub deployment and approve manually

### Build Fails

```bash
# Test build locally
uv build

# Check dist/ folder
ls -lh dist/
```

## Version Management

### Semantic Versioning

We follow [Semantic Versioning 2.0.0](https://semver.org/):

- **MAJOR** (X.0.0) - Incompatible API changes
- **MINOR** (0.X.0) - New features, backward compatible
- **PATCH** (0.0.X) - Bug fixes, backward compatible

### Pre-release Versions

For testing, use pre-release version formats:

```bash
# Release candidates
hatch version "0.3.0-rc1"

# Beta releases  
hatch version "0.3.0-beta.1"

# Alpha releases
hatch version "0.3.0-alpha.1"
```

Publish pre-releases to TestPyPI only, not production PyPI.

## Quick Reference

```bash
# 1. Bump version
hatch version minor

# 2. Update CHANGELOG.md
vim CHANGELOG.md

# 3. Commit
git add -A
git commit -m "chore: bump version to 0.3.0"
git push

# 4. Tag and release
git tag v0.3.0
git push origin v0.3.0

# 5. Watch workflow
# GitHub Actions → Release to PyPI

# 6. Verify
pip install langchain-copilot==0.3.0
```

## Support

For issues with the release process:
- Check [GitHub Actions logs](https://github.com/derf974/copilot-langchain/actions)
- Review [PyPI project page](https://pypi.org/project/langchain-copilot/)
- Open an [issue](https://github.com/derf974/copilot-langchain/issues)
