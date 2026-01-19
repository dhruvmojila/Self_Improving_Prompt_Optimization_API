"""
Database initialization script for Pixeltable.
Sets up tables, registers UDFs, and optionally seeds test data.
"""

import sys
import os
import click
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.infrastructure.pixeltable import (
    init_pixeltable,
    init_tables,
    register_all_udfs,
    get_pixeltable_client,
    get_schemas,
)
from app.core.config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Pixeltable database management CLI."""
    pass


@cli.command()
@click.option('--reset', is_flag=True, help='Drop all existing tables before creating')
def init(reset: bool):
    """
    Initialize Pixeltable database.
    Creates all tables and registers UDFs.
    """
    logger.info("=" * 60)
    logger.info("Pixeltable Database Initialization")
    logger.info("=" * 60)
    
    # Initialize client
    logger.info("\n1. Initializing Pixeltable client...")
    client = init_pixeltable()
    logger.info(f"   ✓ Pixeltable home: {client._initialized}")
    
    # Reset if requested
    if reset:
        logger.warning("\n2. Resetting database (dropping all tables)...")
        schemas = get_schemas()
        schemas.drop_all_tables()
        logger.info("   ✓ All tables dropped")
    
    # Create tables
    logger.info("\n3. Creating tables...")
    tables = init_tables()
    for name, table in tables.items():
        logger.info(f"   ✓ {name}")
    
    # Register UDFs
    logger.info("\n4. Registering metric UDFs...")
    udfs = register_all_udfs()
    logger.info(f"   ✓ {len(udfs)} UDFs registered")
    
    # Health check
    logger.info("\n5. Running health check...")
    health = client.health_check()
    if health['status'] == 'healthy':
        logger.info(f"   ✓ Status: {health['status']}")
        logger.info(f"   ✓ Tables: {len(health['tables'])}")
    else:
        logger.error(f"   ✗ Status: {health['status']}")
        logger.error(f"   ✗ Error: {health.get('error')}")
        sys.exit(1)
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Database initialization complete!")
    logger.info("=" * 60)


@cli.command()
def status():
    """Check Pixeltable database status."""
    logger.info("Checking Pixeltable status...")
    
    client = get_pixeltable_client()
    health = client.health_check()
    
    print("\n" + "=" * 60)
    print("Pixeltable Status")
    print("=" * 60)
    print(f"Status: {health['status']}")
    print(f"Home: {health.get('pixeltable_home', 'N/A')}")
    print(f"Tables: {health.get('num_tables', 0)}")
    
    if health.get('tables'):
        print("\nTables:")
        for table in health['tables']:
            print(f"  - {table}")
    
    if health.get('error'):
        print(f"\nError: {health['error']}")
    
    print("=" * 60)


@cli.command()
@click.confirmation_option(prompt='Are you sure you want to reset the database?')
def reset():
    """Reset the database (drop all tables)."""
    logger.warning("Resetting database...")
    
    schemas = get_schemas()
    schemas.drop_all_tables()
    
    logger.info("✓ Database reset complete")


@cli.command()
def seed():
    """Seed database with example data."""
    logger.info("Seeding database with example data...")
    
    # TODO: Add seed data functionality
    # This would create:
    # - Example project
    # - Sample sentiment dataset
    # - Baseline prompt
    
    logger.info("✓ Seeding complete (not yet implemented)")


if __name__ == '__main__':
    cli()
