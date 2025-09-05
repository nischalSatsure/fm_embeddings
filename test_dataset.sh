uv run -m pipeline.create_dataset \
dataset=negatives \
dataset.name=tes \
dataset.filepath=data/negatives.geojson \
dataset.output_path=data/negatives.parquet \
max_workers=2