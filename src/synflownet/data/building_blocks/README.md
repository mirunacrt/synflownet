The building blocks are available upon request from [enamine.net/building-blocks/building-blocks-catalog](https://enamine.net/building-blocks/building-blocks-catalog), Global stock. To save training time, we pre-compute masks that specify compatibility between reaction templates and building blocks. To pre-process the building blocks file and pre-compute masks, run (specify arguments):

```
python select_short_building_blocks.py 
python subsample_building_blocks.py --random True
python sanitize_building_blocks.py
python remove_duplicates.py
python precompute_bb_masks.py
```

Make sure that the filenames match those in `synflownet/tasks/config.py`.