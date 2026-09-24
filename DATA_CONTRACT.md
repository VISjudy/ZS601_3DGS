# Data contract

The notebook expects a user-configured Drive experiment root with:

```
01-datasets/
  synthetic-base/
    synthetic/images/            # training + 10 validation + trajectory test RGB
    cameras/{train,validation,test,virtual_near}/{images,cameras}.txt
    pointclouds/points_3cm.ply
  supervision/arrays/000001.npz  # lp_radius2 metric camera-Z / PCA normals / masks
  virtual-frozen-200/{images,masks}/000001.png
03-experiment-results/
```

All paths are independent of GitHub; no private dataset access is granted by these notebooks.
The loader checks train/validation/test disjointness and requires exactly ten fixed validation views.
Use the same published split files, source point cloud, images and numeric supervision arrays.
The current trainer records their content hashes in its `run_manifest.json`.
The staging helper writes a new descriptive supervision manifest; its hash therefore differs
from the historical operational manifest, while the actual RGB / cloud / target bytes are reused.

Masks are **255 ignore, 0 valid**. For synthetic E, acquired-path synthetic images get zero masks;
the 200 virtual images retain their alpha-derived ignore masks. The helper merges only the
training and virtual pose lists. It never merges test views into training. Existing source inputs
are read-only, and the temporary local staging cache checks hashes before reuse.

Real-data prerequisites (F-real is planned, not yet trained):

```
01-datasets/real-prepared/
  real/{images,masks}/
  cameras/{train,validation,test,virtual_near}/{images,cameras}.txt
  pointclouds/points_3cm.las
01-datasets/real-supervision/{arrays,protocol.json}
01-datasets/real-virtual-frozen-200/{images,masks}/
```

`protocol.json` must identify `dataset_kind: real` and `source_cloud_sha256` for the actual
real LiDAR used to generate targets. The real virtual RGB must also come from real LiDAR.
At publication, the original real RGB/masks and clouds were prepared, but these real supervision
and virtual-render derivatives were **not yet available**. F-real fails preflight until they exist;
it never substitutes synthetic targets or synthetic virtual RGB. The notebook is a parameterized
entrypoint for that future experiment, not evidence that it has run.

The two synthetic OOD test sets require their separate evaluation jobs; the training notebook's
final test is the explicitly configured trajectory test split. Do not label it a complete OOD result.
There is no real unseen-trajectory image GT; corresponding real novel-view visuals are qualitative.
