"""
Pixeltable client connection manager.
Handles database initialization and table management using namespaces.
"""

from typing import Optional
import pixeltable as pxt
from pathlib import Path
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class PixeltableClient:
    """
    Singleton Pixeltable client for managing database connections.
    
    Uses namespace prefixes for organizing tables (e.g., "prompts.registry").
    Ensures top-level directories exist before creating tables.
    """
    
    _instance: Optional['PixeltableClient'] = None
    _initialized: bool = False
    
    def __new__(cls):
        """Singleton pattern - only one instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize Pixeltable client (called once)."""
        if not self._initialized:
            self._setup_pixeltable()
            self._initialized = True
    
    def _setup_pixeltable(self):
        """Setup Pixeltable environment."""
        # Set Pixeltable home directory if configured
        home_path = getattr(settings, "PIXELTABLE_HOME", None)
        if home_path:
            print(f"Setting Pixeltable home to: {home_path}")
            pxt.HOME = home_path
        
        # Create database directory
        db_path = Path(settings.PIXELTABLE_DB_PATH)
        db_path.mkdir(parents=True, exist_ok=True)

        # Ensure top-level directories exist
        for dir_name in ["prompts", "datasets", "evaluations", "optimization"]:
            try:
                # Try to get the directory - if it exists, this succeeds
                pxt.get_dir_contents(dir_name)
                logger.debug(f"Directory {dir_name} already exists")
            except Exception as err:
                logger.debug(f"Directory {dir_name} does not exist: {err}")
                try:
                    # Attempt to create the directory; if exists, ignore
                    pxt.create_dir(dir_name, if_exists='ignore')
                    logger.info(f"Ensured Pixeltable directory exists: {dir_name}")
                except Exception as e:
                    logger.warning(f"Could not create Pixeltable directory {dir_name}: {e}")
        
        logger.info(
            f"Pixeltable initialized"
            + (f" with home: {settings.PIXELTABLE_HOME}" if settings.PIXELTABLE_HOME else "")
        )

    
    def get_or_create_table(
        self,
        table_path: str,
        schema: Optional[dict] = None
    ) -> pxt.Table:
        """
        Get or create a Pixeltable table.
        
        Args:
            table_path: Full table path with namespace (e.g., "prompts.registry")
            schema: Table schema (required for creation)
            
        Returns:
            Pixeltable table object
        """
        try:
            # Try to get existing table
            return pxt.get_table(table_path)
        except Exception:
            # Table doesn't exist, create it
            if schema is None:
                raise ValueError(f"Schema required to create table {table_path}")
            
            logger.info(f"Creating Pixeltable table: {table_path}")
            return pxt.create_table(table_path, schema)
    
    def drop_table(self, table_path: str, force: bool = False):
        """
        Drop a Pixeltable table.
        
        Args:
            table_path: Full table path
            force: If True, force deletion even if referenced
        """
        try:
            pxt.drop_table(table_path, force=force)
            logger.info(f"Dropped table: {table_path}")
        except Exception as e:
            logger.warning(f"Failed to drop table {table_path}: {e}")
    
    def list_tables(self) -> list:
        """
        List all tables.
        
        Returns:
            List of table paths
        """
        try:
            return pxt.list_tables()
        except Exception as e:
            logger.warning(f"Failed to list tables: {e}")
            return []
    
    def table_exists(self, table_path: str) -> bool:
        """Check if a table exists."""
        try:
            pxt.get_table(table_path)
            return True
        except Exception:
            return False
    
    def reset_database(self):
        """
        Reset the entire database (WARNING: destructive).
        Useful for testing and development.
        """
        logger.warning("Resetting Pixeltable database - all data will be lost!")
        
        # Drop all known tables
        known_tables = [
            'prompts.registry',
            'datasets.data',
            'evaluations.results',
            'optimization.runs',
        ]
        
        for table_path in known_tables:
            try:
                self.drop_table(table_path, force=True)
            except Exception as e:
                logger.debug(f"Could not drop {table_path}: {e}")
    
    def health_check(self) -> dict:
        """
        Check Pixeltable health.
        
        Returns:
            Health status dictionary
        """
        try:
            # Try to list tables as a basic health check
            tables = pxt.list_tables()
            
            return {
                "status": "healthy",
                "pixeltable_home": settings.PIXELTABLE_HOME,
                "num_tables": len(tables),
                "tables": tables
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Global client instance
_client: Optional[PixeltableClient] = None


def get_pixeltable_client() -> PixeltableClient:
    """
    Get the global Pixeltable client instance.
    
    Returns:
        PixeltableClient singleton
    """
    global _client
    if _client is None:
        _client = PixeltableClient()
    return _client


def init_pixeltable() -> PixeltableClient:
    """
    Initialize Pixeltable client.
    Convenience function for startup.
    
    Returns:
        Initialized PixeltableClient
    """
    client = get_pixeltable_client()
    logger.info("Pixeltable client initialized successfully")
    return client
