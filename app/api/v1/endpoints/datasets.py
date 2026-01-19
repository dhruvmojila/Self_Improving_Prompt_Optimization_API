"""
Datasets API endpoints.
Dataset upload, ingestion, and retrieval.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
import pixeltable as pxt
import pandas as pd
import json
import io
from datetime import datetime

from app.domain.models import (
    Dataset, DatasetCreate, DatasetUpload, DatasetRow,
    DatasetSplit, DatasetStats, DatasetRowsResponse
)
from app.api.v1.dependencies import get_db, get_current_user
from app.infrastructure.pixeltable import PixeltableClient, get_schemas

router = APIRouter()


@router.post("/upload", response_model=Dataset, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    input_fields: str = Form(...),  # JSON array as string
    output_fields: str = Form(...),  # JSON array as string
    train_split: float = Form(0.6),
    dev_split: float = Form(0.2),
    test_split: float = Form(0.2),
    current_user: str = Depends(get_current_user),
    db: PixeltableClient = Depends(get_db)
):
    """
    Upload a dataset file (CSV or JSONL).
    
    Automatically splits data into train/dev/test sets and stores in Pixeltable.
    """
    # Parse field lists
    try:
        input_field_list = json.loads(input_fields)
        output_field_list = json.loads(output_fields)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="input_fields and output_fields must be valid JSON arrays"
        )
    
    # Read file
    contents = await file.read()
    
    # Parse based on file type
    if file.filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(contents))
    elif file.filename.endswith('.jsonl') or file.filename.endswith('.json'):
        df = pd.read_json(io.BytesIO(contents), lines=file.filename.endswith('.jsonl'))
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV and JSONL files are supported"
        )
    
    # Validate columns
    all_fields = input_field_list + output_field_list
    missing_fields = set(all_fields) - set(df.columns)
    if missing_fields:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required columns: {missing_fields}"
        )
    
    # Create dataset object
    import uuid
    dataset_id = str(uuid.uuid4())
    
    # Split data
    total_rows = len(df)
    train_end = int(total_rows * train_split)
    dev_end = train_end + int(total_rows * dev_split)
    
    df_train = df.iloc[:train_end]
    df_dev = df.iloc[train_end:dev_end]
    df_test = df.iloc[dev_end:]
    
    # Insert into Pixeltable
    schemas = get_schemas()
    table = schemas.get_table('datasets')
    
    rows_inserted = 0
    for split_name, split_df in [('train', df_train), ('dev', df_dev), ('test', df_test)]:
        for idx, row in split_df.iterrows():
            # Prepare input and output data
            input_data = {field: row[field] for field in input_field_list}
            ground_truth = {field: row[field] for field in output_field_list}
            
            # Insert row
            table.insert([{
                'dataset_id': dataset_id,
                'row_id': f"{dataset_id}_{rows_inserted}",
                'input_data': input_data,
                'ground_truth': ground_truth,
                'split': split_name,
                'modality': 'text',
                'dataset_name': name,
                'project_id': '',
                'created_at': datetime.utcnow(),
            }])
            
            rows_inserted += 1
    
    # Create dataset metadata
    dataset = Dataset(
        id=dataset_id,
        name=name,
        description=description,
        modality='text',
        input_fields=input_field_list,
        output_fields=output_field_list,
        total_rows=total_rows,
        train_rows=len(df_train),
        dev_rows=len(df_dev),
        test_rows=len(df_test),
        file_format=file.filename.split('.')[-1],
    )
    
    return dataset


@router.get("/{dataset_id}", response_model=Dataset)
async def get_dataset(
    dataset_id: str,
    current_user: str = Depends(get_current_user),
    db: PixeltableClient = Depends(get_db)
):
    """Get dataset metadata by ID."""
    schemas = get_schemas()
    table = schemas.get_table('datasets')
    
    # Query for dataset
    results = table.where(table.dataset_id == dataset_id).collect()
    
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )
    
    # Get first row for metadata
    first_row = results[0]
    
    # Count rows by split
    total = len(results)
    train_count = sum(1 for r in results if r['split'] == 'train')
    dev_count = sum(1 for r in results if r['split'] == 'dev')
    test_count = sum(1 for r in results if r['split'] == 'test')
    
    # Infer fields from first row
    input_fields = list(first_row['input_data'].keys())
    output_fields = list(first_row['ground_truth'].keys())
    
    dataset = Dataset(
        id=dataset_id,
        name=first_row['dataset_name'],
        description='',
        modality=first_row['modality'],
        input_fields=input_fields,
        output_fields=output_fields,
        total_rows=total,
        train_rows=train_count,
        dev_rows=dev_count,
        test_rows=test_count,
    )
    
    return dataset


@router.get("/{dataset_id}/rows", response_model=DatasetRowsResponse)
async def get_dataset_rows(
    dataset_id: str,
    split: Optional[str] = None,
    page: int = 1,
    page_size: int = 100,
    current_user: str = Depends(get_current_user),
    db: PixeltableClient = Depends(get_db)
):
    """Get dataset rows with optional filtering by split."""
    schemas = get_schemas()
    table = schemas.get_table('datasets')
    
    # Build query
    query = table.where(table.dataset_id == dataset_id)
    if split:
        query = query.where(table.split == split)
    
    # Get results
    results = query.collect()
    
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )
    
    # Pagination
    total = len(results)
    start = (page - 1) * page_size
    end = start + page_size
    paginated = results[start:end]
    
    # Convert to DatasetRow objects
    rows = []
    for r in paginated:
        row = DatasetRow(
            input=r['input_data'],
            output=r['ground_truth'],
            split=DatasetSplit(r['split']),
            metadata={'row_id': r['row_id']}
        )
        rows.append(row)
    
    return DatasetRowsResponse(
        rows=rows,
        total=total,
        page=page,
        page_size=page_size,
        split=DatasetSplit(split) if split else None
    )


@router.get("/{dataset_id}/stats", response_model=DatasetStats)
async def get_dataset_stats(
    dataset_id: str,
    current_user: str = Depends(get_current_user),
    db: PixeltableClient = Depends(get_db)
):
    """Get dataset statistics."""
    schemas = get_schemas()
    table = schemas.get_table('datasets')
    
    results = table.where(table.dataset_id == dataset_id).collect()
    
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )
    
    # Calculate stats
    total = len(results)
    train_count = sum(1 for r in results if r['split'] == 'train')
    dev_count = sum(1 for r in results if r['split'] == 'dev')
    test_count = sum(1 for r in results if r['split'] == 'test')
    
    first_row = results[0]
    input_fields = list(first_row['input_data'].keys())
    output_fields = list(first_row['ground_truth'].keys())
    
    return DatasetStats(
        total_rows=total,
        train_rows=train_count,
        dev_rows=dev_count,
        test_rows=test_count,
        input_fields=input_fields,
        output_fields=output_fields
    )
