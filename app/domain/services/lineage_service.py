"""
Lineage Service - Prompt version tracking and semantic diffs.
Manages prompt version DAG and calculates semantic differences.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import difflib
import hashlib

from app.infrastructure.pixeltable import get_schemas
from app.domain.models import PromptVersion, PromptDiff, PromptLineage, PromptSignature
from app.infrastructure.llm import LLMFactory, LLMConfig

logger = logging.getLogger(__name__)


class LineageService:
    """
    Prompt lineage and versioning service.
    
    Tracks prompt version history as a DAG (Directed Acyclic Graph)
    and provides semantic diff computation.
    """
    
    def __init__(self):
        self.schemas = get_schemas()
        self.llm = None  # Lazy load for changelog generation
    
    def create_version(
        self,
        prompt_data: Dict[str, Any],
        parent_id: Optional[str] = None
    ) -> PromptVersion:
        """
        Create a new prompt version.
        
        Args:
            prompt_data: Prompt data (template, signature, etc.)
            parent_id: Optional parent prompt ID for lineage
            
        Returns:
            Created PromptVersion
        """
        import uuid
        
        # Generate version hash
        content = f"{prompt_data['template']}:{str(prompt_data['signature'])}"
        version_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        # Get parent hash if parent exists
        parent_hash = None
        if parent_id:
            parent = self.get_version(parent_id)
            parent_hash = parent.version_hash
        
        # Insert into Pixeltable
        prompts_table = self.schemas.get_table('prompts')
        
        prompt_id = str(uuid.uuid4())
        
        prompts_table.insert([{
            'prompt_id': prompt_id,
            'version_hash': version_hash,
            'template_str': prompt_data['template'],
            'dspy_signature': prompt_data['signature'],
            'parent_hash': parent_hash or '',
            'name': prompt_data.get('name', 'Untitled'),
            'description': prompt_data.get('description', ''),
            'tags': prompt_data.get('tags', []),
            'created_at': datetime.utcnow(),
            'author_id': prompt_data.get('author_id', ''),
            'artifact_path': prompt_data.get('artifact_path', ''),
            'optimizer_type': prompt_data.get('optimizer_type', ''),
            'baseline_metric': prompt_data.get('baseline_metric'),
            'optimized_metric': prompt_data.get('optimized_metric'),
        }])
        
        logger.info(f"Created prompt version {prompt_id} with hash {version_hash}")
        
        return self.get_version(prompt_id)
    
    def get_version(self, prompt_id: str) -> PromptVersion:
        """Get a specific prompt version."""
        prompts_table = self.schemas.get_table('prompts')
        
        results = prompts_table.where(prompts_table.prompt_id == prompt_id).collect()
        
        if not results:
            raise ValueError(f"Prompt {prompt_id} not found")
        
        row = results[0]
        
        return PromptVersion(
            id=row['prompt_id'],
            version_hash=row['version_hash'],
            name=row['name'],
            template=row['template_str'],
            signature=PromptSignature(**row['dspy_signature']),
            description=row['description'],
            tags=row['tags'],
            parent_hash=row['parent_hash'],
            author_id=row['author_id'],
            created_at=row['created_at'],
            dspy_artifact_path=row['artifact_path'],
            optimizer_type=row['optimizer_type'],
            baseline_metric=row['baseline_metric'],
            optimized_metric=row['optimized_metric'],
        )
    
    def get_lineage(self, prompt_id: str) -> PromptLineage:
        """
        Get complete version lineage for a prompt.
        
        Traverses the DAG to find all ancestors and descendants.
        
        Args:
            prompt_id: Starting prompt ID
            
        Returns:
            PromptLineage with full version history
        """
        # Get all prompts to build DAG
        prompts_table = self.schemas.get_table('prompts')
        all_prompts = prompts_table.collect()
        
        # Build version map
        versions = {}
        for row in all_prompts:
            version = PromptVersion(
                id=row['prompt_id'],
                version_hash=row['version_hash'],
                name=row['name'],
                template=row['template_str'],
                signature=PromptSignature(**row['dspy_signature']),
                parent_hash=row['parent_hash'],
                created_at=row['created_at'],
            )
            versions[row['version_hash']] = version
        
        # Build parent-child map
        parent_child_map = {}
        for version in versions.values():
            if version.parent_hash:
                if version.parent_hash not in parent_child_map:
                    parent_child_map[version.parent_hash] = []
                parent_child_map[version.parent_hash].append(version.version_hash)
        
        # Find all related versions (ancestors + descendants)
        current_version = self.get_version(prompt_id)
        related_versions = self._collect_related_versions(
            current_version.version_hash,
            versions,
            parent_child_map
        )
        
        # Sort by creation time
        version_history = sorted(related_versions, key=lambda v: v.created_at)
        
        return PromptLineage(
            prompt_id=prompt_id,
            version_history=version_history,
            parent_child_map=parent_child_map,
            total_versions=len(version_history),
            optimization_runs=0  # TODO: Query optimization runs
        )
    
    def _collect_related_versions(
        self,
        version_hash: str,
        all_versions: Dict[str, PromptVersion],
        parent_child_map: Dict[str, List[str]]
    ) -> List[PromptVersion]:
        """Recursively collect all ancestors and descendants."""
        related = []
        
        if version_hash not in all_versions:
            return related
        
        current = all_versions[version_hash]
        related.append(current)
        
        # Collect ancestors
        if current.parent_hash and current.parent_hash in all_versions:
            ancestors = self._collect_related_versions(
                current.parent_hash, all_versions, parent_child_map
            )
            related.extend(ancestors)
        
        # Collect descendants
        if version_hash in parent_child_map:
            for child_hash in parent_child_map[version_hash]:
                descendants = self._collect_related_versions(
                    child_hash, all_versions, parent_child_map
                )
                related.extend(descendants)
        
        # Remove duplicates
        seen = set()
        unique = []
        for v in related:
            if v.version_hash not in seen:
                seen.add(v.version_hash)
                unique.append(v)
        
        return unique
    
    def compute_diff(
        self,
        parent_id: str,
        child_id: str,
        generate_changelog: bool = True
    ) -> PromptDiff:
        """
        Compute semantic diff between two prompt versions.
        
        Args:
            parent_id: Parent prompt ID
            child_id: Child prompt ID
            generate_changelog: Whether to generate natural language changelog
            
        Returns:
            PromptDiff with detailed changes
        """
        parent = self.get_version(parent_id)
        child = self.get_version(child_id)
        
        # Template diff
        template_diff = self._unified_diff(parent.template, child.template)
        
        # Instruction diff (if separate system prompts exist)
        instruction_diff = None
        if parent.system_prompt and child.system_prompt:
            instruction_diff = self._unified_diff(
                parent.system_prompt or "", child.system_prompt or ""
            )
        
        # Few-shot examples diff
        # TODO: Extract examples from DSPy artifact
        examples_added = []
        examples_removed = []
        
        # Performance diff
        metric_delta = None
        metric_delta_pct = None
        if parent.optimized_metric and child.optimized_metric:
            metric_delta = child.optimized_metric - parent.optimized_metric
            metric_delta_pct = (metric_delta / parent.optimized_metric * 100) if parent.optimized_metric > 0 else 0
        
        # Generate changelog
        changelog = None
        if generate_changelog:
            changelog = self._generate_changelog(parent, child, template_diff)
        
        return PromptDiff(
            parent_hash=parent.version_hash,
            child_hash=child.version_hash,
            instruction_diff=instruction_diff,
            template_diff=template_diff,
            examples_added=examples_added,
            examples_removed=examples_removed,
            metric_delta=metric_delta,
            metric_delta_pct=metric_delta_pct,
            changelog=changelog
        )
    
    def _unified_diff(self, text_a: str, text_b: str) -> str:
        """Create unified diff between two texts."""
        lines_a = text_a.splitlines(keepends=True)
        lines_b = text_b.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            lines_a, lines_b,
            fromfile='parent', tofile='child',
            lineterm=''
        )
        
        return ''.join(diff)
    
    def _generate_changelog(
        self,
        parent: PromptVersion,
        child: PromptVersion,
        template_diff: str
    ) -> str:
        """
        Generate natural language changelog using LLM.
        
        Args:
            parent: Parent version
            child: Child version
            template_diff: Unified diff of templates
            
        Returns:
            Natural language changelog
        """
        if self.llm is None:
            self.llm = LLMFactory.create_teacher()
        
        # Create changelog prompt
        prompt = f"""You are a technical writer summarizing changes between two prompt versions.

Parent Prompt (v{parent.version_hash[:8]}):
{parent.template[:500]}

Child Prompt (v{child.version_hash[:8]}):
{child.template[:500]}

Diff:
{template_diff[:500]}

Performance:
- Parent metric: {parent.optimized_metric or 'N/A'}
- Child metric: {child.optimized_metric or 'N/A'}

Write a concise changelog (2-3 sentences) explaining:
1. What changed in the prompt
2. Why it might improve performance
3. Any notable additions or removals

Changelog:"""
        
        try:
            config = LLMConfig(temperature=0.3, max_tokens=200)
            response = self.llm.generate(prompt=prompt, config=config)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Changelog generation failed: {e}")
            return "Failed to generate changelog"
    
    def tag_version(self, prompt_id: str, tags: List[str]):
        """Add tags to a prompt version (e.g., 'production', 'deprecated')."""
        prompts_table = self.schemas.get_table('prompts')
        
        # Get current version
        version = self.get_version(prompt_id)
        
        # Merge tags
        new_tags = list(set(version.tags + tags))
        
        # Update (requires delete + insert in Pixeltable)
        # TODO: Implement update operation when Pixeltable supports it
        logger.info(f"Tagged {prompt_id} with {tags}")
    
    def find_production_version(self) -> Optional[PromptVersion]:
        """Find the current production prompt version."""
        prompts_table = self.schemas.get_table('prompts')
        
        # Query for prompts with 'production' tag
        # TODO: Implement JSON querying when available
        
        all_prompts = prompts_table.collect()
        
        for row in all_prompts:
            if 'production' in row.get('tags', []):
                return self.get_version(row['prompt_id'])
        
        return None
