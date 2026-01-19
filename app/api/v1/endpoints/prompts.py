"""
Prompts API endpoints.
Prompt version management, lineage, and execution.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
import pixeltable as pxt
from datetime import datetime
import time

from app.domain.models import (
    PromptVersion, PromptCreate, PromptDiff, PromptLineage,
    PromptRunRequest, PromptRunResponse
)
from app.api.v1.dependencies import get_db, get_current_user
from app.infrastructure.pixeltable import PixeltableClient, get_schemas
from app.infrastructure.llm import LLMFactory, PromptFormatter

router = APIRouter()


@router.post("/", response_model=PromptVersion, status_code=status.HTTP_201_CREATED)
async def create_prompt(
    prompt: PromptCreate,
    current_user: str = Depends(get_current_user),
    db: PixeltableClient = Depends(get_db)
):
    """
    Create a new prompt version.
    
    This creates the initial version or a manual edit (parent_id provided).
    """
    import uuid
    import hashlib
    
    # Generate IDs
    prompt_id = str(uuid.uuid4())
    
    # Generate version hash
    content = f"{prompt.template}:{prompt.signature.json()}"
    version_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    
    # Insert into Pixeltable
    schemas = get_schemas()
    table = schemas.get_table('prompts')
    
    table.insert([{
        'prompt_id': prompt_id,
        'version_hash': version_hash,
        'template_str': prompt.template,
        'dspy_signature': prompt.signature.dict(),
        'parent_hash': prompt.parent_id or '',
        'name': prompt.name,
        'description': prompt.description or '',
        'tags': prompt.tags,
        'created_at': datetime.utcnow(),
        'author_id': current_user,
        'artifact_path': '',
        'optimizer_type': '',
        'baseline_metric': None,
        'optimized_metric': None,
    }])
    
    # Return created prompt
    result = PromptVersion(
        id=prompt_id,
        version_hash=version_hash,
        name=prompt.name,
        template=prompt.template,
        signature=prompt.signature,
        system_prompt=prompt.system_prompt,
        description=prompt.description,
        tags=prompt.tags,
        parent_hash=prompt.parent_id,
        author_id=current_user,
    )
    
    return result


@router.get("/{prompt_id}", response_model=PromptVersion)
async def get_prompt(
    prompt_id: str,
    current_user: str = Depends(get_current_user),
    db: PixeltableClient = Depends(get_db)
):
    """Get prompt version by ID."""
    schemas = get_schemas()
    table = schemas.get_table('prompts')
    
    results = table.where(table.prompt_id == prompt_id).collect()
    
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt {prompt_id} not found"
        )
    
    row = results[0]
    
    from app.domain.models import PromptSignature
    
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
        baseline_metric=row['baseline_metric'],
        optimized_metric=row['optimized_metric'],
    )


@router.get("/{prompt_id}/lineage", response_model=PromptLineage)
async def get_prompt_lineage(
    prompt_id: str,
    current_user: str = Depends(get_current_user),
    db: PixeltableClient = Depends(get_db)
):
    """
    Get prompt version lineage (DAG).
    
    Returns all versions in the lineage with parent-child relationships.
    """
    # For MVP, just return the single prompt
    # TODO: Implement full DAG traversal
    
    prompt = await get_prompt(prompt_id, current_user, db)
    
    return PromptLineage(
        prompt_id=prompt_id,
        version_history=[prompt],
        parent_child_map={},
        total_versions=1,
        optimization_runs=0
    )


@router.post("/{prompt_id}/run", response_model=PromptRunResponse)
async def run_prompt(
    prompt_id: str,
    request: PromptRunRequest,
    current_user: str = Depends(get_current_user),
    db: PixeltableClient = Depends(get_db)
):
    """
    Run a prompt on single input (playground mode).
    
    Executes the prompt with provided inputs and returns the output.
    """
    # Get prompt
    prompt = await get_prompt(prompt_id, current_user, db)
    
    # Format prompt with inputs
    try:
        user_message = PromptFormatter.format_prompt(prompt.template, **request.inputs)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    # Get LLM provider (use student model for inference)
    from app.core.config import settings
    provider = LLMFactory.create_student()
    
    # Build config
    from app.infrastructure.llm import LLMConfig
    config = LLMConfig(
        temperature=request.temperature or 0.7,
        max_tokens=request.max_tokens or 500
    )
    
    # Execute
    start_time = time.time()
    
    response = provider.generate(
        prompt=user_message,
        config=config,
        system_prompt=prompt.system_prompt
    )
    
    latency_ms = (time.time() - start_time) * 1000
    
    # Parse output fields from response
    from app.infrastructure.llm import PromptFormatter
    outputs = PromptFormatter.extract_output_fields(
        response.content,
        prompt.signature.outputs
    )
    
    return PromptRunResponse(
        outputs=outputs,
        model=response.model,
        provider=response.provider,
        tokens_used=response.tokens_used,
        latency_ms=latency_ms
    )
