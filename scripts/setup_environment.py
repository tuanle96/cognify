#!/usr/bin/env python3
"""
Environment Setup Script for Cognify
Helps users set up their environment variables securely.
"""

import os
import secrets
import sys
from pathlib import Path


def generate_secret_key(length: int = 32) -> str:
    """Generate a secure random secret key."""
    return secrets.token_urlsafe(length)


def check_existing_env_file(env_path: Path) -> bool:
    """Check if environment file exists."""
    return env_path.exists()


def create_env_file(env_path: Path, template_path: Path) -> None:
    """Create environment file from template."""
    if not template_path.exists():
        print(f"❌ Template file not found: {template_path}")
        return
    
    # Read template
    with open(template_path, 'r') as f:
        content = f.read()
    
    # Generate secure keys
    secret_key = generate_secret_key()
    jwt_secret_key = generate_secret_key()
    
    # Replace placeholders
    content = content.replace(
        'your-secret-key-here-change-in-production-must-be-32-chars-min',
        secret_key
    )
    content = content.replace(
        'your-jwt-secret-key-change-in-production-must-be-32-chars-min',
        jwt_secret_key
    )
    
    # Write to new file
    with open(env_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Created {env_path}")
    print(f"   Generated SECRET_KEY: {secret_key[:20]}...")
    print(f"   Generated JWT_SECRET_KEY: {jwt_secret_key[:20]}...")


def validate_environment() -> dict:
    """Validate current environment variables."""
    issues = []
    warnings = []
    
    # Critical variables
    critical_vars = [
        'OPENAI_API_KEY',
        'SECRET_KEY',
        'JWT_SECRET_KEY'
    ]
    
    for var in critical_vars:
        value = os.getenv(var)
        if not value:
            issues.append(f"Missing required variable: {var}")
        elif var.endswith('_KEY') and value.startswith('your-'):
            warnings.append(f"Default placeholder value for: {var}")
        elif var.endswith('_KEY') and len(value) < 20:
            warnings.append(f"Short key length for: {var} (should be 32+ chars)")
    
    # Optional but recommended
    recommended_vars = [
        'DATABASE_URL',
        'REDIS_URL',
        'QDRANT_URL'
    ]
    
    for var in recommended_vars:
        if not os.getenv(var):
            warnings.append(f"Recommended variable not set: {var}")
    
    return {
        'issues': issues,
        'warnings': warnings
    }


def main():
    """Main setup function."""
    print("🔧 Cognify Environment Setup")
    print("=" * 50)
    
    project_root = Path(__file__).parent.parent
    template_path = project_root / ".env.example"
    
    # Environment options
    environments = {
        '1': ('development', project_root / ".env.development"),
        '2': ('production', project_root / ".env.production"),
        '3': ('testing', project_root / ".env.testing"),
        '4': ('custom', project_root / ".env")
    }
    
    print("\nSelect environment to set up:")
    for key, (name, _) in environments.items():
        print(f"  {key}. {name}")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice not in environments:
        print("❌ Invalid choice")
        sys.exit(1)
    
    env_name, env_path = environments[choice]
    
    # Check if file exists
    if check_existing_env_file(env_path):
        overwrite = input(f"\n⚠️  {env_path} already exists. Overwrite? (y/N): ").strip().lower()
        if overwrite != 'y':
            print("Cancelled.")
            sys.exit(0)
    
    # Create environment file
    print(f"\n🔨 Creating {env_name} environment...")
    create_env_file(env_path, template_path)
    
    # Provide next steps
    print(f"\n📝 Next steps:")
    print(f"1. Edit {env_path}")
    print(f"2. Set your OPENAI_API_KEY (get from: https://platform.openai.com/api-keys)")
    print(f"3. Update database and other service URLs as needed")
    print(f"4. Load environment: export $(cat {env_path} | xargs)")
    
    # Validate if loading current environment
    if choice == '4':  # custom .env
        print(f"\n🔍 Validating current environment...")
        validation = validate_environment()
        
        if validation['issues']:
            print("\n❌ Critical Issues:")
            for issue in validation['issues']:
                print(f"   - {issue}")
        
        if validation['warnings']:
            print("\n⚠️  Warnings:")
            for warning in validation['warnings']:
                print(f"   - {warning}")
        
        if not validation['issues'] and not validation['warnings']:
            print("✅ Environment looks good!")
    
    print(f"\n🎉 Environment setup complete!")
    print(f"Remember to never commit {env_path} to version control!")


if __name__ == "__main__":
    main()
