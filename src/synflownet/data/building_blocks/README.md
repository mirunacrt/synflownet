The building blocks are available upon request from [enamine.net/building-blocks/building-blocks-catalog](https://enamine.net/building-blocks/building-blocks-catalog), Global stock. To save training time, we pre-compute masks that specify compatibility between reaction templates and building blocks. To pre-process the building blocks file and pre-compute masks, run (specify arguments):

```
python select_short_building_blocks.py 
python subsample_building_blocks.py --random True
python sanitize_building_blocks.py
python remove_duplicates.py
python precompute_bb_masks.py
```

An out-of-the-box set of arguments could be
```
python select_short_building_blocks.py
python subsample_building_blocks.py --random True --filename short_building_blocks.txt
python sanitize_building_blocks.py --building_blocks_filename short_building_blocks_subsampled_5000.txt --output_filename sanitized_bbs.txt
python remove_duplicates.py --building_blocks_filename sanitized_bbs.txt --output_filename enamine_bbs.txt
python precompute_bb_masks.py
```

Make sure that the filenames match those in `synflownet/tasks/config.py`.